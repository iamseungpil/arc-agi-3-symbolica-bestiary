"""Arcgentica host with optional research guests.

This class keeps Arcgentica as the execution host and limits research changes to
hook dispatch, prompt overlays, and persistence sidecars.

Supports two backends:
  - agentica SDK (when AGENTICA_API_KEY or S_M_BASE_URL set)
  - TRAPI direct (Azure OpenAI Responses API, always available)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from research_extensions import (
    ModuleRegistry,
    ResearchRuntimeContext,
    load_research_config,
)
from research_extensions.grid_utils import current_grid, encode_row, visible_latent_state

# Rev M orchestration modules (P38-P48). Imported up-front so the
# agent can wire them in __init__ when REV_M_ENABLED=1.
from research_extensions.abstraction_handler import (
    AbstractionHandler,
    AbstractionInput,
)
from research_extensions.divergence_monitor import DivergenceMonitor
from research_extensions.phase_controller import Phase, PhaseController
from research_extensions.phase_rewards import PhaseRewardRegistry
from research_extensions.diff_amplifier import (
    render_diff_amplified,
    render_initial_frame_summary,
)
from research_extensions.player_tracker import (
    BOUNDARY_COLORS,
    PlayerTracker,
    compute_diff_coords,
    count_diffs,
)
from research_extensions.reachability_gate import ReachabilityGate
from research_extensions.sleep_handler import (
    DivergenceReport,
    SleepHandler,
    WakeTrace,
    _grid_diff_summary,
)
from research_extensions.surprise_router import SurpriseRouter
from research_extensions.synthetic_consistency import SyntheticConsistencyChecker
from research_extensions.tool_gate import (
    AbstractionGate,
    GateResult,
    SleepGate,
    ToolCallTracker,
    WakeGate,
    check_helper_references,
    gated_llm_call,
)
from research_extensions.wake_planner import (
    WakeCandidate,
    WakePlan,
    WakePlanError,
    WakePlanner,
)

from ...agent import Agent
from ...tracing import trace_agent_session
from ..agentica.prompts import GAME_REFERENCE
from ..agentica.scope.frame import Frame

logger = logging.getLogger(__name__)

# TRAPI config
_TRAPI_SCOPE = "api://trapi/.default"
_TRAPI_API_VERSION = "2025-04-01-preview"
_TRAPI_MODEL_MAP = {
    "gpt-5.3-codex": "gpt-5.3-codex_2026-02-24",
    "gpt-5.4-mini": "gpt-5.4-mini_2026-03-17",
    "gpt-5.4": "gpt-5.4_2026-03-05",
    "gpt-5.4-pro": "gpt-5.4-pro_2026-03-05",
}

# arm11 swap: env-driven default model override across all _call_trapi sites.
_DEFAULT_MODEL = os.environ.get("ARC_RESEARCH_MODEL", "gpt-5.3-codex")

_trapi_client = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# Rev S P70 / Rev T P74: every ``FORCED_PROBE_INTERVAL`` real actions,
# override the Wake planner's chosen first action with the least-used
# action from the recent history. Overridable via the
# ``ARC_FORCED_PROBE_INTERVAL`` env var for tests. Values <= 0 disable
# the interval probe (the imbalance probe still fires). Rev T lowers the
# default cadence from 5 to 3 to shrink the ACTION4 dominance window.
FORCED_PROBE_INTERVAL: int = int(os.environ.get("ARC_FORCED_PROBE_INTERVAL", "3"))


def _rev_m_hard_gate_enabled() -> bool:
    """Return True when the hard tool-gate retry path should run.

    Rev N P53 live-run fix: when ``REV_M_HARD_GATE`` is unset or ``0``
    (default), the Wake/Sleep/Abstraction LLM calls do NOT go through
    the retry loop. Tool-gate evaluation still happens for telemetry
    but never triggers a re-prompt. Set ``REV_M_HARD_GATE=1`` to
    enforce (opt-in, for when the tool-calling LLM path is built).
    """
    raw = os.environ.get("REV_M_HARD_GATE", "0").strip()
    return raw not in ("", "0", "false", "False", "no", "off")


def _raw_first_reasoning_enabled() -> bool:
    """Raw-first reasoning mode for intended Symbolica-style inspection.

    Default ON: prefer current/previous raw renders and tool-mediated
    self-inspection over host-authored semantic summaries. Callers can
    opt back into the older scaffold-heavy path via
    ``ARC_RAW_FIRST_REASONING=0``.
    """
    return _env_bool("ARC_RAW_FIRST_REASONING", True)


def _wake_host_summaries_enabled() -> bool:
    """Whether Wake injects semantic host summaries beyond raw diff/grid.

    In raw-first mode we disable these by default so the LLM has to
    infer state roles from raw observations rather than consuming
    player/initial-frame summaries. Operators can still force-enable
    them with ``ARC_WAKE_HOST_SUMMARIES=1``.
    """
    return _env_bool("ARC_WAKE_HOST_SUMMARIES", not _raw_first_reasoning_enabled())


def _main_auto_helper_summary_enabled() -> bool:
    """Whether the main tool-calling agent receives host helper summary.

    Raw-first mode turns this OFF by default so the model must call
    ``frame_diff`` / ``frame_render`` / ``history`` itself.
    """
    return _env_bool(
        "ARC_MAIN_AUTO_HELPER_SUMMARY",
        not _raw_first_reasoning_enabled(),
    )


def _wake_self_inspection_enabled() -> bool:
    """Enable a model-authored inspection pass before Wake planning."""
    return _env_bool("ARC_WAKE_SELF_INSPECTION", _raw_first_reasoning_enabled())


def _wake_end_to_end_budget_s() -> float:
    """Total wall-clock budget for one Wake phase."""
    return _env_float("REV_M_WAKE_END_TO_END_BUDGET_S", 45.0)


def _wake_eval_min_budget_s() -> float:
    """Minimum time reserved for candidate evaluation after the LLM call."""
    return _env_float("REV_M_WAKE_MIN_EVAL_BUDGET_S", 3.0)


def _wake_observation_route_override_enabled() -> bool:
    """Prefer empirical observation routes over speculative mixes by default."""
    return _env_bool("REV_M_WAKE_OBSERVATION_ROUTE_OVERRIDE", True)


def _resolve_shared_namespace(timestamp: int | None = None) -> str:
    explicit = os.environ.get("ARC_RESEARCH_SHARED_NAMESPACE")
    if explicit:
        return explicit
    if timestamp is None:
        timestamp = int(time.time())
    return f"ephemeral_{timestamp}_{os.getpid()}"

def _get_trapi_client():
    global _trapi_client
    if _trapi_client is not None:
        return _trapi_client
    from openai import AzureOpenAI
    from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(), ManagedIdentityCredential(),
    ), _TRAPI_SCOPE)
    _trapi_client = AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        azure_ad_token_provider=credential,
        api_version=_TRAPI_API_VERSION,
        timeout=_env_float("ARC_AGENTICA_TRAPI_TIMEOUT_SEC", 180.0),
    )
    return _trapi_client

def _call_trapi(messages: list[dict], model: str | None = None) -> str:
    if model is None or model == "gpt-5.3-codex":
        model = _DEFAULT_MODEL
    deployment = _TRAPI_MODEL_MAP.get(model, model)
    client = _get_trapi_client()
    input_items = [{"type": "message", "role": m["role"], "content": m["content"]} for m in messages]
    resp = client.responses.create(model=deployment, input=input_items, max_output_tokens=4096)
    for item in resp.output:
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    return block.text
    return ""


def _create_trapi_response(
    *,
    model: str | None = None,
    input_items: Any,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_output_tokens: int = 4096,
):
    if model is None or model == "gpt-5.3-codex":
        model = _DEFAULT_MODEL
    deployment = _TRAPI_MODEL_MAP.get(model, model)
    client = _get_trapi_client()
    kwargs: dict[str, Any] = {
        "model": deployment,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = False
        kwargs["max_tool_calls"] = 32
    return client.responses.create(**kwargs)


class ArcgenticaResearch(Agent):
    """Arcgentica-style agent with research hooks, powered by TRAPI.

    Keeps a strict per-step host: each model turn sees only the current visible
    grid, state, available actions, and optional research overlays.
    """

    MAX_ACTIONS: int = int(os.environ.get("ARC_SMOKE_MAX_ACTIONS", "200"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.research_config = load_research_config()
        self._registry = None
        super().__init__(*args, **kwargs)
        timestamp = int(time.time())
        self.shared_namespace = _resolve_shared_namespace(timestamp)
        log_root = Path(self.research_config.log_dir)
        self.research_workdir = log_root / f"{self.game_id}_{timestamp}"
        self.research_workdir.mkdir(parents=True, exist_ok=True)
        shared_dir = log_root / "shared" / self.game_id
        shared_dir.mkdir(parents=True, exist_ok=True)
        context = ResearchRuntimeContext(
            game_id=self.game_id,
            workdir=self.research_workdir,
            shared_dir=shared_dir / self.shared_namespace,
        )
        context.shared_dir.mkdir(parents=True, exist_ok=True)
        self._registry = ModuleRegistry(self.research_config, context)
        self._registry.load()
        # P10 (plan v4.2 §P10): AbstractionEngine needs a TRAPI-backed
        # LLM callable to turn confirmed hypotheses into candidate skills
        # (batch). Without this the engine short-circuits to 0 candidates
        # and the DC library never grows from abstraction.
        try:
            eng = self._registry.abstraction_engine()
            if eng is not None and eng.llm_client is None:
                def _abstraction_llm(prompt: str) -> str:
                    try:
                        resp = _create_trapi_response(
                            model="gpt-5.3-codex",
                            input_items=[{
                                "type": "message", "role": "user",
                                "content": prompt,
                            }],
                            max_output_tokens=1024,
                        )
                        return getattr(resp, "output_text", "") or ""
                    except Exception:
                        logger.exception(
                            "abstraction TRAPI call failed; returning empty"
                        )
                        return ""
                eng.llm_client = _abstraction_llm
        except Exception:
            logger.exception("failed to inject abstraction llm_client")
        self.messages: list[dict] = []
        self.action_history: list[tuple[str, Frame]] = []
        # Rev U: flat per-run action-name history for the legacy
        # (REV_M_ENABLED=0) main loop. Mirrors ``self._action_history``
        # used by the Rev M path so the legacy-only imbalance override
        # (``_legacy_imbalance_override``) has a 20-step window to check
        # without depending on Rev M init.
        self._legacy_action_history: list[str] = []
        self.best_level = 0
        self.reset_count = 0
        # Rev Q (P64): level-advance delta from the most recent real step.
        # Surfaced to the Wake planner via the bridge so candidate eval
        # can reward "keep exploring near the last successful move".
        self._last_level_advance_delta: int = 0
        # P1 annex (§R1.2): agent-driven probe set per level, reset on
        # every observed level advance. Registry-side auto-probing is
        # forbidden from writing here.
        self._actions_probed_this_level: set[str] = set()
        self._memories = _LocalMemories(context.shared_dir / "memories.json")
        if self._registry:
            self._registry.on_memories_ready(self._memories)

        # Rev M (plan v4.2 §P38-P48): phase-controlled Wake/Sleep/
        # Abstraction orchestration. Feature-flagged: REV_M_ENABLED=0
        # preserves the legacy per-turn LLM path for backward-compat.
        # Rev M is opt-in until we've validated the full orchestration
        # loop end-to-end on a live run.
        self.rev_m_enabled: bool = os.environ.get(
            "REV_M_ENABLED", "0"
        ).strip() not in ("0", "false", "False", "no", "off", "")
        self._rev_m_initialized: bool = False
        if self.rev_m_enabled:
            self._init_rev_m_components()

    def _init_rev_m_components(self) -> None:
        """Instantiate P38-P48 orchestration objects.

        Called once per agent from __init__ when REV_M_ENABLED=1. Kept
        isolated so unit tests can flip the flag and re-run.
        """
        self._phase_controller: PhaseController = PhaseController(
            logger=getattr(self._registry, "_phase_logger", None)
        )
        self._phase_rewards: PhaseRewardRegistry = PhaseRewardRegistry()
        self._divergence_monitor: DivergenceMonitor = DivergenceMonitor(
            bridge=self._registry.bridge if self._registry else None
        )
        self._surprise_router: SurpriseRouter = SurpriseRouter(
            bridge=self._registry.bridge if self._registry else None,
            workdir=self.research_workdir,
        )

        # Pick a WM draft + source. The existing WorldModelModule is the
        # source of truth; if it isn't loaded the Rev M pipeline still
        # runs but simulate_step returns a trivial stub.
        wm_module = None
        if self._registry is not None:
            wm_module = self._registry.get("world_model")
        self._wm_draft = wm_module

        wake_llm = self._build_rev_m_phase_llm(
            phase_name="wake",
            max_output_tokens=_env_int("REV_M_WAKE_MAX_OUTPUT_TOKENS", 1024),
        )
        sleep_llm = self._build_rev_m_phase_llm(
            phase_name="sleep",
            max_output_tokens=_env_int("REV_M_SLEEP_MAX_OUTPUT_TOKENS", 640),
        )
        abstraction_llm = self._build_rev_m_phase_llm(
            phase_name="abstraction",
            max_output_tokens=_env_int("REV_M_ABSTRACTION_MAX_OUTPUT_TOKENS", 512),
        )
        self._wake_planner: WakePlanner = WakePlanner(
            bridge=self._registry.bridge if self._registry else None,
            phase_rewards=self._phase_rewards,
            llm_client=wake_llm,
            max_depth=_env_int("REV_M_WAKE_MAX_DEPTH", 6),
            total_wake_budget_s=_env_float("REV_M_TOTAL_WAKE_BUDGET_S", 15.0),
        )
        if self._registry is not None:
            try:
                self._wake_planner.trigger_helpers = (
                    self._registry._get_symbolica_helpers()
                )
            except Exception:
                logger.exception("Rev M: failed to attach wake trigger helpers")
        self._sleep_handler: SleepHandler = SleepHandler(
            bridge=self._registry.bridge if self._registry else None,
            phase_rewards=self._phase_rewards,
            surprise_router=self._surprise_router,
            llm_client=sleep_llm,
            timeout_s=_env_float("REV_M_SLEEP_TIMEOUT_SEC", 60.0),
        )
        self._abstraction_handler: AbstractionHandler = AbstractionHandler(
            bridge=self._registry.bridge if self._registry else None,
            phase_rewards=self._phase_rewards,
            llm_client=abstraction_llm,
            timeout_s=_env_float("REV_M_ABSTRACTION_TIMEOUT_SEC", 45.0),
        )
        self._reachability_gate: ReachabilityGate = ReachabilityGate(
            wm_draft=self._wm_draft
        )
        store = None
        if self._registry is not None:
            store = getattr(self._registry.bridge, "hypothesis_store", None)
        if store is not None and self._wm_draft is not None:
            self._synthetic_checker: SyntheticConsistencyChecker | None = (
                SyntheticConsistencyChecker(
                    wm_draft=self._wm_draft,
                    hypothesis_store=store,
                    surprise_router=self._surprise_router,
                )
            )
        else:
            self._synthetic_checker = None

        # Gates are stateless checkers; one instance reused across turns.
        self._wake_gate = WakeGate()
        self._sleep_gate = SleepGate()
        self._abstraction_gate = AbstractionGate()

        # Rev N P50: rolling per-phase reward samples (keep-last-20; R24.3).
        # Keyed by phase name; values are lists of floats. Appended in the
        # corresponding _run_*_phase methods after each result.
        self._phase_reward_history: dict[str, list[float]] = {
            "wake": [], "sleep": [], "abstraction": [],
        }
        # Rev N P50 / R25.3: rolling per-Wake reward-spread samples.
        self._wake_candidate_reward_spread_history: list[float] = []
        # Rev N P49: structured rejection feedback from the hypothesis
        # store, consumed by the next Wake prompt as a hint.
        self._last_rejection_feedback: list[str] = []
        # Rev N P49-hardening: how many Wake turns we've run. Feeds the
        # prompt so the "early turn" mandate knows when to fire.
        self._wake_turn_index: int = 0
        # Rev N P49-hardening Fix 4: count of Wake-submitted hypotheses
        # that reached the store gate this run. Meta-harness reads the
        # bridge hint populated from this list at run-end.
        self._wake_submitted_hypothesis_ids: list[str] = []
        # Rev N P53 telemetry: last gate-check tool_compliance per phase.
        self._last_wake_tool_compliance: bool = True

        # Rev O P54-P57: player-centric observation tracker. Updated after
        # every real env step so the Wake prompt can show the LLM a
        # compact "where did the sprite move" summary instead of the full
        # 64x64 grid that hides 5-cell movements.
        self._player_tracker: PlayerTracker = PlayerTracker()
        self._action_player_deltas: dict[str, list[tuple[int, int]]] = {}
        self._last_diff_coords: list[tuple[int, int, int, int]] = []
        self._last_diff_total: int = 0

        # Rev P P58-P59: baseline starting-grid summary injected into every
        # Wake prompt + cached before/after grids for the amplified diff
        # block. Populated lazily on the first Wake turn of the run.
        self._initial_frame_summary: str = ""
        self._last_diff_before_grid: list[list[int]] | None = None
        self._last_diff_after_grid: list[list[int]] | None = None
        self._last_diff_patch_text: str = ""

        # Per-turn tool call tracker. Reset at the start of every phase
        # handler (Wake / Sleep / Abstraction) per W9.
        self._tool_tracker: ToolCallTracker = ToolCallTracker()
        # W9 telemetry: populated by _run_sleep_phase / _run_abstraction_phase.
        self._last_sleep_tool_compliance: bool = True
        self._last_abstraction_tool_compliance: bool = True

        # Accumulators passed between phases within a cycle.
        self._current_wake_trace: WakeTrace = WakeTrace()
        self._recent_successful_sequences: list[dict] = []
        self._last_divergence: dict | None = None
        self._wm_source: str = self._fetch_wm_source()
        # Bug C fix: flat per-run action history for Wake exploration
        # tie-break. Exposed on the bridge so WakePlanner can read it
        # without the agent import cycle.
        self._action_history: list[str] = []
        if self._registry is not None and getattr(self._registry, "bridge", None) is not None:
            try:
                self._registry.bridge.action_history = self._action_history
            except Exception:
                logger.exception("Rev M: failed to attach action_history to bridge")
        # Rev R (P67): restore skills / confirmed hypotheses / WM draft
        # source from prior runs so episodes accumulate learning instead
        # of starting from an empty library every time.
        self._load_cross_run_memory()
        self._rev_m_initialized = True

    def _load_cross_run_memory(self) -> None:
        """Rev R P67: pull prior-run state into the live agent.

        Wrapped in try/except so a missing / unreadable shared dir never
        blocks session init. Logs a single INFO line with counts.
        """
        if os.getenv("ARC_ENABLE_CROSS_RUN_MEMORY", "").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info("Rev R disabled: ARC_ENABLE_CROSS_RUN_MEMORY not set")
            return
        try:
            from research_extensions.modules.cross_run_memory import (
                CrossRunMemoryLoader,
            )
            shared_base = Path(self.research_config.log_dir) / "shared"
            loader = CrossRunMemoryLoader(
                game_id=self.game_id,
                shared_base_dir=shared_base,
                max_prior_runs=10,
            )
            snapshot = loader.load()
            if self._registry is not None:
                loader.persist_to_bridge(
                    getattr(self._registry, "bridge", None), snapshot
                )
            # Restore skills into DreamCoder module.
            skills_restored = 0
            if snapshot.skills and self._registry is not None:
                dc = self._registry.get("dreamcoder")
                if dc is not None and hasattr(dc, "adopt_carryover"):
                    for sk in snapshot.skills:
                        if dc.adopt_carryover(sk) is not None:
                            skills_restored += 1
            # Restore confirmed hypotheses into store.
            hyps_restored = 0
            store = None
            if self._registry is not None:
                store = getattr(self._registry.bridge, "hypothesis_store", None)
            if snapshot.confirmed_hypotheses and store is not None:
                for h in snapshot.confirmed_hypotheses:
                    if store.adopt_carryover(h) is not None:
                        hyps_restored += 1
            # Restore WM source via the existing registry helper so the
            # best-draft pipeline sees a drop-in body.
            wm_restored = False
            if snapshot.wm_draft_source and self._registry is not None:
                body = snapshot.wm_draft_source.strip()
                fenced = (
                    body if body.startswith("```") else
                    f"```python\n{body}\n```"
                )
                try:
                    self._registry.record_agent_world_update({"body": fenced})
                    wm_restored = True
                except Exception:
                    logger.exception(
                        "Rev R: record_agent_world_update failed for "
                        "carryover draft"
                    )
            logger.info(
                "Rev R loaded %d skills, %d hypotheses, wm=%s from %d prior runs",
                skills_restored, hyps_restored, wm_restored,
                snapshot.prior_runs_count,
            )
        except Exception:
            logger.exception("Rev R: _load_cross_run_memory failed (non-fatal)")

    def _record_phase_reward(self, phase: str, reward: float) -> None:
        """Rev N P50: append phase reward to rolling buffer (keep last 20)."""
        try:
            bucket = self._phase_reward_history.setdefault(phase, [])
            bucket.append(float(reward))
            if len(bucket) > 20:
                del bucket[: len(bucket) - 20]
        except Exception:
            logger.exception("Rev N P50: record_phase_reward failed")

    def _record_wake_candidate_spread(self, all_candidates: Any) -> None:
        """Rev N P50 / R25.3: rolling last-20 of per-Wake reward spreads."""
        try:
            rewards = [
                float(getattr(c, "reward", 0.0) or 0.0)
                for c in (all_candidates or [])
            ]
            if not rewards:
                return
            spread = float(max(rewards) - min(rewards))
            self._wake_candidate_reward_spread_history.append(spread)
            if len(self._wake_candidate_reward_spread_history) > 20:
                excess = len(self._wake_candidate_reward_spread_history) - 20
                del self._wake_candidate_reward_spread_history[:excess]
        except Exception:
            logger.exception("Rev N P50: record_wake_candidate_spread failed")

    def _record_action_player_delta(
        self,
        action_name: str,
        before_center: tuple[int, int] | None,
        after_center: tuple[int, int] | None,
    ) -> None:
        if not action_name or before_center is None or after_center is None:
            return
        dr = int(after_center[0] - before_center[0])
        dc = int(after_center[1] - before_center[1])
        bucket = self._action_player_deltas.setdefault(str(action_name), [])
        bucket.append((dr, dc))
        if len(bucket) > 16:
            del bucket[: len(bucket) - 16]

    def _informative_action_deltas(
        self,
        available_actions: list[str],
    ) -> dict[str, tuple[int, int]]:
        action_player_deltas = getattr(self, "_action_player_deltas", {}) or {}
        deltas: dict[str, tuple[int, int]] = {}
        for action in available_actions:
            samples = [
                delta
                for delta in action_player_deltas.get(action, [])
                if delta != (0, 0)
            ]
            if not samples:
                continue
            dr = round(sum(delta[0] for delta in samples) / len(samples))
            dc = round(sum(delta[1] for delta in samples) / len(samples))
            if dr == 0 and dc == 0:
                continue
            deltas[action] = (int(dr), int(dc))
        return deltas

    def _action_delta_profiles(
        self,
        available_actions: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Summarize recent empirical movement confidence per action.

        The observation-route planner should not extrapolate a just-seen
        movement primitive three or four times in a row. In ls20 this caused
        Wake to overcommit to a single fresh delta before the environment had
        provided enough evidence that the motion generalizes beyond one step.
        """
        action_player_deltas = getattr(self, "_action_player_deltas", {}) or {}
        profiles: dict[str, dict[str, Any]] = {}
        for action in available_actions:
            raw_samples = list(action_player_deltas.get(action, []) or [])
            if not raw_samples:
                continue
            samples = raw_samples[-6:]
            non_zero = [delta for delta in samples if delta != (0, 0)]
            if not non_zero:
                continue
            counts = Counter(non_zero)
            dominant_delta, dominant_count = counts.most_common(1)[0]
            recent_blocked = bool(samples and samples[-1] == (0, 0))
            profiles[action] = {
                "delta": (int(dominant_delta[0]), int(dominant_delta[1])),
                "confirmed_count": int(dominant_count),
                "recent_blocked": recent_blocked,
                "samples": list(samples),
            }
        return profiles

    def _build_coverage_probe_candidate(
        self,
        available_actions: list[str],
    ) -> WakeCandidate | None:
        """Probe an untested action until the tracker has two movement axes.

        ls20 runs were stalling because Wake learned only one movement
        action early (usually ACTION1), then built a vertical-only route
        and never discovered the orthogonal axis in time. This candidate
        forces early action coverage, preferring an unprobed orthogonal
        action once one movement direction is known.
        """
        unprobed = [
            action
            for action in available_actions
            if action.startswith("ACTION")
            and action not in getattr(self, "_actions_probed_this_level", set())
        ]
        if not unprobed:
            return None
        informative = self._informative_action_deltas(available_actions)
        if not informative:
            return WakeCandidate(
                sequence=[unprobed[0]],
                rationale=f"coverage_probe_cold_start:{unprobed[0]}",
                skill_refs=["coverage-probe"],
            )
        if len(informative) >= 2:
            return None

        preferred = list(unprobed)
        abs_dr = sum(abs(delta[0]) for delta in informative.values())
        abs_dc = sum(abs(delta[1]) for delta in informative.values())
        if abs_dr > abs_dc:
            orthogonal = [a for a in unprobed if a in {"ACTION3", "ACTION4"}]
        elif abs_dc > abs_dr:
            orthogonal = [a for a in unprobed if a in {"ACTION1", "ACTION2"}]
        else:
            orthogonal = []
        if orthogonal:
            preferred = orthogonal + [a for a in unprobed if a not in orthogonal]

        target = preferred[0]
        return WakeCandidate(
            sequence=[target],
            rationale=f"coverage_probe_unprobed:{target}",
            skill_refs=["coverage-probe"],
        )

    def _current_local_game(self) -> Any | None:
        env = getattr(self, "arc_env", None)
        if env is None:
            return None
        return getattr(env, "_game", None)

    def _local_exact_route_enabled(self) -> bool:
        """Guard source-informed host routing behind an explicit opt-in.

        Default OFF so research runs cannot silently succeed via a host-side
        solver that reads the local game object. This keeps normal TRAPI runs
        free of environment-source leakage.
        """
        return _env_bool("ARC_ENABLE_LOCAL_EXACT_ROUTE", False)

    def _build_local_exact_route_candidate(
        self,
        available_actions: list[str],
    ) -> WakeCandidate | None:
        """Build a host-side exact route using the local game object.

        This does not alter the TRAPI prompt. It is an environment-side
        planner used only when the local wrapper exposes the current
        game instance.
        """
        game = self._current_local_game()
        if game is None:
            return None
        required = {"mgu", "qqv", "current_level", "tuv", "tmx", "snw"}
        if not all(hasattr(game, attr) for attr in required):
            return None

        action_deltas = {
            "ACTION1": (0, -5),
            "ACTION2": (0, 5),
            "ACTION3": (-5, 0),
            "ACTION4": (5, 0),
        }
        usable_actions = [a for a in available_actions if a in action_deltas]
        if not usable_actions:
            return None
        pads = list(getattr(game, "qqv", []) or [])
        if not pads:
            return None
        sprites = [
            s for s in list(getattr(getattr(game, "current_level", None), "_sprites", []) or [])
            if getattr(s, "tags", None)
        ]
        start_state = (
            int(game.mgu.x),
            int(game.mgu.y),
            int(game.snw),
            int(game.tmx),
            int(game.tuv),
            tuple(bool(x) for x in getattr(game, "rzt", [False] * len(pads))),
        )

        gfy = list(getattr(game, "gfy", []) or [])
        vxy = list(getattr(game, "vxy", []) or [])
        cjl = list(getattr(game, "cjl", []) or [])
        hep = list(getattr(game, "hep", []) or [])
        hul = list(getattr(game, "hul", []) or [])

        def _qhg_state(state: tuple[int, int, int, int, int, tuple[bool, ...]], pad_idx: int) -> bool:
            _, _, snw, tmx, tuv, _ = state
            return (
                0 <= pad_idx < len(gfy)
                and snw == int(gfy[pad_idx])
                and tmx == int(vxy[pad_idx])
                and tuv == int(cjl[pad_idx])
            )

        def _intersections(nx: int, ny: int) -> list[Any]:
            return [
                sprite for sprite in sprites
                if int(getattr(sprite, "x", -10**9)) >= nx
                and int(getattr(sprite, "x", -10**9)) < nx + 5
                and int(getattr(sprite, "y", -10**9)) >= ny
                and int(getattr(sprite, "y", -10**9)) < ny + 5
            ]

        def _advance(
            state: tuple[int, int, int, int, int, tuple[bool, ...]],
            action_name: str,
        ) -> tuple[int, int, int, int, int, tuple[bool, ...]] | None:
            x, y, snw, tmx, tuv, rzt = state
            dx, dy = action_deltas[action_name]
            nx = x + dx
            ny = y + dy
            blocked = False
            next_snw, next_tmx, next_tuv = snw, tmx, tuv
            next_rzt = list(rzt)
            for sprite in _intersections(nx, ny):
                tags = set(getattr(sprite, "tags", None) or [])
                if "jdd" in tags:
                    blocked = True
                    break
                if "mae" in tags:
                    pad_idx = -1
                    for idx, pad in enumerate(pads):
                        if (
                            int(getattr(pad, "x", -1)) == int(getattr(sprite, "x", -2))
                            and int(getattr(pad, "y", -1)) == int(getattr(sprite, "y", -2))
                        ):
                            pad_idx = idx
                            break
                    if pad_idx >= 0 and not _qhg_state(
                        (nx, ny, next_snw, next_tmx, next_tuv, tuple(next_rzt)),
                        pad_idx,
                    ):
                        blocked = True
                        break
                if "gsu" in tags:
                    next_snw = (next_snw + 1) % max(1, len(hep))
                if "gic" in tags:
                    next_tmx = (next_tmx + 1) % max(1, len(hul))
                if "bgt" in tags:
                    next_tuv = (next_tuv + 1) % 4
            if blocked:
                return None
            for idx, pad in enumerate(pads):
                if next_rzt[idx]:
                    continue
                if (
                    int(getattr(pad, "x", -1)) == nx
                    and int(getattr(pad, "y", -1)) == ny
                    and _qhg_state((nx, ny, next_snw, next_tmx, next_tuv, tuple(next_rzt)), idx)
                ):
                    next_rzt[idx] = True
            return (nx, ny, next_snw, next_tmx, next_tuv, tuple(next_rzt))

        def _is_goal(state: tuple[int, int, int, int, int, tuple[bool, ...]]) -> bool:
            return bool(state[-1]) and all(state[-1])

        queue: deque[tuple[tuple[int, int, int, int, int, tuple[bool, ...]], list[str]]] = deque()
        queue.append((start_state, []))
        seen = {start_state}
        max_depth = 24
        while queue:
            state, path = queue.popleft()
            if _is_goal(state):
                if not path:
                    return None
                return WakeCandidate(
                    sequence=path,
                    rationale=f"local_exact_route_to_goal len={len(path)}",
                    skill_refs=["local-exact-route"],
                )
            if len(path) >= max_depth:
                continue
            for action_name in usable_actions:
                nxt = _advance(state, action_name)
                if nxt is None or nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [action_name]))
        return None

    def _build_observation_route_candidate(
        self,
        available_actions: list[str],
    ) -> WakeCandidate | None:
        tracker = getattr(self, "_player_tracker", None)
        if tracker is None:
            return None
        current = getattr(tracker, "latest_state", None)
        landmarks = list(getattr(tracker, "nearby_landmarks", []) or [])
        if current is None or not landmarks:
            return None
        target = None
        for item in landmarks:
            dist = int(item.get("distance", 0) or 0)
            if dist <= 0:
                continue
            target = item
            break
        if target is None:
            return None

        profiles = self._action_delta_profiles(available_actions)
        deltas = {
            action: tuple(profile.get("delta", (0, 0)))
            for action, profile in profiles.items()
            if not bool(profile.get("recent_blocked", False))
        }
        if not deltas:
            return None

        pos_r, pos_c = current.center
        target_r, target_c = tuple(target.get("center", current.center))
        current_dist = abs(pos_r - target_r) + abs(pos_c - target_c)
        max_depth = max(1, int(getattr(self._wake_planner, "max_depth", 4)))
        action_items = sorted(deltas.items())
        dominant_axis = (
            "row"
            if abs(pos_r - target_r) > abs(pos_c - target_c)
            else "col"
        )
        recent_history = list(getattr(self, "_action_history", []) or [])[-8:]
        recent_forward_action = None
        for action_name in reversed(recent_history):
            profile = profiles.get(action_name)
            if profile is None:
                continue
            if bool(profile.get("recent_blocked", False)):
                continue
            if int(profile.get("confirmed_count", 0) or 0) < 2:
                continue
            recent_forward_action = action_name
            break
        opposite_actions: dict[str, str] = {}
        for action_name, delta in deltas.items():
            dr, dc = delta
            for other_name, other_delta in deltas.items():
                if other_name == action_name:
                    continue
                if other_delta == (-dr, -dc):
                    opposite_actions[action_name] = other_name
                    break
        protected_reverse_action = (
            opposite_actions.get(recent_forward_action, "")
            if recent_forward_action is not None
            else ""
        )
        repeat_caps = {
            action: (
                1
                if int(profiles.get(action, {}).get("confirmed_count", 0) or 0) <= 1
                else 2
                if int(profiles.get(action, {}).get("confirmed_count", 0) or 0) == 2
                else max_depth
            )
            for action in deltas
        }

        best: tuple[int, int, int, tuple[int, ...], int, tuple[str, ...]] | None = None

        def _switches(seq: tuple[str, ...]) -> int:
            if not seq:
                return 0
            return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])

        def _dominant_axis_prefix_gaps(seq: tuple[str, ...]) -> tuple[int, ...]:
            cur_r, cur_c = pos_r, pos_c
            gaps: list[int] = []
            for action_name in seq:
                dr, dc = deltas[action_name]
                cur_r += dr
                cur_c += dc
                if dominant_axis == "row":
                    gaps.append(abs(cur_r - target_r))
                else:
                    gaps.append(abs(cur_c - target_c))
            return tuple(gaps)

        def _search(
            cur_r: int,
            cur_c: int,
            depth: int,
            seq: tuple[str, ...],
        ) -> None:
            nonlocal best
            if depth > 0:
                final_dist = abs(cur_r - target_r) + abs(cur_c - target_c)
                if final_dist < current_dist:
                    low_conf_steps = sum(
                        1
                        for action_name in seq
                        if int(
                            profiles.get(action_name, {}).get(
                                "confirmed_count",
                                0,
                            )
                            or 0
                        )
                        <= 1
                    )
                    score = (
                        final_dist,
                        max(abs(cur_r - target_r), abs(cur_c - target_c)),
                        low_conf_steps,
                        _dominant_axis_prefix_gaps(seq),
                        _switches(seq),
                        seq,
                    )
                    if best is None or score < best:
                        best = score
            if depth >= max_depth:
                return
            for action, (dr, dc) in action_items:
                if (
                    protected_reverse_action
                    and action == protected_reverse_action
                ):
                    continue
                if seq.count(action) >= int(repeat_caps.get(action, max_depth)):
                    continue
                nxt_r = cur_r + dr
                nxt_c = cur_c + dc
                _search(nxt_r, nxt_c, depth + 1, seq + (action,))

        _search(pos_r, pos_c, 0, tuple())
        sequence = list(best[5]) if best is not None else []
        if not sequence:
            return None
        candidate = WakeCandidate(
            sequence=sequence,
            rationale=(
                "observation_route_to_landmark "
                f"target_color={target.get('color')} target_center={target.get('center')}"
            ),
            skill_refs=["observation-route"],
        )
        end_r, end_c = pos_r, pos_c
        for action_name in sequence:
            dr, dc = deltas[action_name]
            end_r += dr
            end_c += dc
        final_dist = abs(end_r - target_r) + abs(end_c - target_c)
        candidate.observation_route_gain = max(0, current_dist - final_dist)
        candidate.rationale += f" gain={candidate.observation_route_gain}"
        return candidate

    def _build_visual_frontier_candidate(
        self,
        available_actions: list[str],
    ) -> WakeCandidate | None:
        """Plan a short passable route over the visible grid.

        This is deliberately observation-only: it uses the current rendered
        grid, the detected player bbox, and empirical action deltas. The goal
        is to recover multi-step corridor motion that a one-step landmark
        heuristic misses, without reading environment source or exact routes.
        """
        tracker = getattr(self, "_player_tracker", None)
        current = getattr(tracker, "latest_state", None) if tracker is not None else None
        grid = getattr(self, "_last_visible_grid", None)
        if current is None or not isinstance(grid, list) or not grid:
            return None
        profiles = self._action_delta_profiles(available_actions)
        deltas = {
            action: tuple(profile.get("delta", (0, 0)))
            for action, profile in profiles.items()
        }
        if len(deltas) < 2:
            return None

        bbox = tuple(current.bounding_box)
        r0, r1, c0, c1 = bbox
        start_center = tuple(current.center)
        depth_limit = max(2, int(getattr(self._wake_planner, "max_depth", 4)))
        seen_centers = {
            tuple(getattr(state, "center", ()))
            for state in list(getattr(tracker, "history", []) or [])
            if getattr(state, "center", None) is not None
        }
        landmarks = list(getattr(tracker, "nearby_landmarks", []) or [])
        target_center = None
        if landmarks:
            item = landmarks[0]
            center = item.get("center")
            if isinstance(center, tuple) and len(center) == 2:
                target_center = (int(center[0]), int(center[1]))

        def _bbox_passable(base_bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int] | None:
            nr0, nr1 = base_bbox[0] + dr, base_bbox[1] + dr
            nc0, nc1 = base_bbox[2] + dc, base_bbox[3] + dc
            if nr0 < 0 or nc0 < 0:
                return None
            if nr1 >= len(grid):
                return None
            width = len(grid[0]) if isinstance(grid[0], list) else 0
            if width <= 0 or nc1 >= width:
                return None
            for rr in range(nr0, nr1 + 1):
                row = grid[rr]
                if not isinstance(row, list):
                    return None
                for cc in range(nc0, nc1 + 1):
                    cell = row[cc]
                    if isinstance(cell, int) and cell in BOUNDARY_COLORS:
                        return None
            return (nr0, nr1, nc0, nc1)

        def _center_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
            return ((box[0] + box[1]) // 2, (box[2] + box[3]) // 2)

        best: tuple[int, int, int, int, tuple[str, ...]] | None = None
        queue: deque[tuple[tuple[int, int, int, int], tuple[str, ...]]] = deque()
        queue.append((bbox, tuple()))
        seen: set[tuple[tuple[int, int, int, int], tuple[str, ...]]] = {(bbox, tuple())}
        recent_history = list(getattr(self, "_action_history", []) or [])[-8:]

        while queue:
            cur_bbox, seq = queue.popleft()
            cur_center = _center_from_bbox(cur_bbox)
            if seq:
                unseen_bonus = 1 if cur_center not in seen_centers else 0
                geodesic = abs(cur_center[0] - start_center[0]) + abs(cur_center[1] - start_center[1])
                if target_center is not None:
                    target_dist = abs(cur_center[0] - target_center[0]) + abs(cur_center[1] - target_center[1])
                else:
                    target_dist = 0
                score = (
                    -len(seq),
                    -unseen_bonus,
                    -geodesic,
                    target_dist,
                    seq,
                )
                if best is None or score < best:
                    best = score
            if len(seq) >= depth_limit:
                continue
            for action_name, delta in sorted(deltas.items()):
                dr, dc = int(delta[0]), int(delta[1])
                nxt_bbox = _bbox_passable(cur_bbox, dr, dc)
                if nxt_bbox is None:
                    continue
                if recent_history and len(seq) == 0 and action_name == recent_history[-1]:
                    # Allow reusing the same mover, but try alternate passable
                    # branches too instead of collapsing to the immediate repeat.
                    pass
                nxt = (nxt_bbox, seq + (action_name,))
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)

        if best is None:
            return None
        sequence = list(best[4])
        if len(sequence) < 2:
            return None
        candidate = WakeCandidate(
            sequence=sequence,
            rationale="visual_frontier_route",
            skill_refs=["visual-frontier-route"],
        )
        candidate.visual_frontier_depth = len(sequence)
        return candidate

    def _build_lane_follow_candidate(
        self,
        available_actions: list[str],
    ) -> WakeCandidate | None:
        """Continue a recently stable mover along the current lane.

        This is a generic observation-side heuristic: when an action has moved
        the player by the same non-zero delta at least twice and has not just
        been blocked, propose extending that same motion for a short horizon.
        It captures corridor following without any game-specific labels.
        """
        profiles = self._action_delta_profiles(available_actions)
        if not profiles:
            return None
        history = list(getattr(self, "_action_history", []) or [])
        if not history:
            return None

        recent = history[-6:]
        recent_counts = {
            action: recent.count(action)
            for action in available_actions
            if action in profiles
        }
        ranked = sorted(
            recent_counts.items(),
            key=lambda item: (
                -int(item[1]),
                -int(profiles.get(item[0], {}).get("confirmed_count", 0) or 0),
                item[0],
            ),
        )
        if not ranked or ranked[0][1] <= 0:
            return None
        action_name = ranked[0][0]
        profile = profiles.get(action_name, {})
        if bool(profile.get("recent_blocked", False)):
            return None
        confirmed = int(profile.get("confirmed_count", 0) or 0)
        if confirmed < 2:
            return None
        repeat = min(max(2, confirmed), int(getattr(self._wake_planner, "max_depth", 4)))
        sequence = [action_name] * repeat
        candidate = WakeCandidate(
            sequence=sequence,
            rationale=f"lane_follow action={action_name} confirmed={confirmed}",
            skill_refs=["lane-follow"],
        )
        candidate.lane_follow_strength = confirmed
        return candidate

    def _estimate_empirical_route_bonus(
        self,
        sequence: list[str],
        available_actions: list[str],
    ) -> float:
        """Estimate route quality from observed deltas and visible target.

        This is observation-only and generic: it uses the tracked player
        center, nearby landmark center, and empirical per-action deltas. It
        does not read environment source or any exact route tables. The goal is
        to slightly boost multi-stage routes that appear to reduce distance to a
        visible target while changing movement axis when needed.
        """
        tracker = getattr(self, "_player_tracker", None)
        current = getattr(tracker, "latest_state", None) if tracker is not None else None
        landmarks = list(getattr(tracker, "nearby_landmarks", []) or []) if tracker is not None else []
        if current is None or not landmarks or not sequence:
            return 0.0
        item = landmarks[0]
        center = item.get("center")
        if not (isinstance(center, tuple) and len(center) == 2):
            return 0.0
        target_center = (int(center[0]), int(center[1]))
        profiles = self._action_delta_profiles(available_actions)
        if not profiles:
            return 0.0
        pos = tuple(getattr(current, "center", ()) or ())
        if len(pos) != 2:
            return 0.0
        pos_r, pos_c = int(pos[0]), int(pos[1])
        current_dist = abs(pos_r - target_center[0]) + abs(pos_c - target_center[1])
        axis_switches = 0
        prev_axis: str | None = None
        progressed = 0
        for action in sequence:
            profile = profiles.get(action)
            if not profile:
                continue
            delta = tuple(profile.get("delta", (0, 0)))
            dr, dc = int(delta[0]), int(delta[1])
            if dr == 0 and dc == 0:
                continue
            pos_r += dr
            pos_c += dc
            progressed += 1
            axis = "row" if abs(dr) >= abs(dc) else "col"
            if prev_axis is not None and axis != prev_axis:
                axis_switches += 1
            prev_axis = axis
        if progressed == 0:
            return 0.0
        final_dist = abs(pos_r - target_center[0]) + abs(pos_c - target_center[1])
        gain_steps = max(0, current_dist - final_dist) / 5.0
        return min(6.0, 0.6 * gain_steps + 0.5 * min(axis_switches, 2))

    def _apply_empirical_route_bonus(
        self,
        plan: WakePlan,
        available_actions: list[str],
    ) -> WakePlan:
        if not list(getattr(plan, "all_candidates", []) or []):
            return plan
        for cand in list(plan.all_candidates or []):
            bonus = self._estimate_empirical_route_bonus(
                list(getattr(cand, "sequence", []) or []),
                available_actions,
            )
            if bonus <= 0.0:
                continue
            cand.reward = float(getattr(cand, "reward", 0.0) or 0.0) + bonus
            cand.debug_bonus = float(getattr(cand, "debug_bonus", 0.0) or 0.0) + bonus
        seen_fams = set(
            getattr(self._registry.bridge, "seen_families", set()) or set()
        ) if self._registry is not None else set()
        try:
            chosen = self._wake_planner._argmax_with_novelty_tiebreak(
                list(plan.all_candidates or []),
                seen_fams,
                available_actions,
            )
        except Exception:
            chosen = max(
                list(plan.all_candidates or []),
                key=lambda c: float(getattr(c, "reward", 0.0) or 0.0),
            )
        return replace(plan, chosen=chosen)

    def _apply_stall_escape_bonus(
        self,
        plan: WakePlan,
        available_actions: list[str],
        diff_total: int,
    ) -> WakePlan:
        if diff_total > 2 or not list(getattr(plan, "all_candidates", []) or []):
            return plan
        recent = [
            str(a)
            for a in list(getattr(self, "_action_history", []) or [])[-6:]
            if isinstance(a, str)
        ]
        if not recent:
            return plan
        counts = Counter(recent)
        dominant_action, dominant_count = counts.most_common(1)[0]
        if dominant_count < 2:
            return plan
        for cand in list(plan.all_candidates or []):
            seq = list(getattr(cand, "sequence", []) or [])
            if not seq:
                continue
            first = str(seq[0])
            bonus = 0.0
            if first != dominant_action and counts.get(first, 0) == 0:
                bonus += 1.5
            rationale = str(getattr(cand, "rationale", "") or "").lower()
            if any(token in rationale for token in ("gate", "toggle", "latent", "unlock", "interaction")):
                bonus += 1.0
            if len(set(seq)) >= 2:
                bonus += 0.5
            if bonus <= 0.0:
                continue
            cand.reward = float(getattr(cand, "reward", 0.0) or 0.0) + bonus
            cand.debug_bonus = float(getattr(cand, "debug_bonus", 0.0) or 0.0) + bonus
        seen_fams = set(
            getattr(self._registry.bridge, "seen_families", set()) or set()
        ) if self._registry is not None else set()
        try:
            chosen = self._wake_planner._argmax_with_novelty_tiebreak(
                list(plan.all_candidates or []),
                seen_fams,
                available_actions,
            )
        except Exception:
            chosen = max(
                list(plan.all_candidates or []),
                key=lambda c: float(getattr(c, "reward", 0.0) or 0.0),
            )
        return replace(plan, chosen=chosen)

    def _should_prefer_visual_frontier(
        self,
        frontier_candidate: WakeCandidate | None,
        chosen_candidate: WakeCandidate | None,
    ) -> bool:
        if frontier_candidate is None:
            return False
        chosen_rationale = str(getattr(chosen_candidate, "rationale", "") or "")
        if "visual_frontier_route" in chosen_rationale:
            return False
        frontier_depth = int(getattr(frontier_candidate, "visual_frontier_depth", 0) or 0)
        chosen_len = len(list(getattr(chosen_candidate, "sequence", []) or []))
        return frontier_depth >= 3 and frontier_depth > chosen_len

    def _should_prefer_lane_follow(
        self,
        lane_candidate: WakeCandidate | None,
        chosen_candidate: WakeCandidate | None,
    ) -> bool:
        if lane_candidate is None:
            return False
        chosen_rationale = str(getattr(chosen_candidate, "rationale", "") or "")
        if "lane_follow" in chosen_rationale:
            return False
        lane_len = len(list(getattr(lane_candidate, "sequence", []) or []))
        chosen_len = len(list(getattr(chosen_candidate, "sequence", []) or []))
        return lane_len >= 2 and lane_len > chosen_len

    def _should_prefer_observation_route(
        self,
        obs_candidate: WakeCandidate | None,
        chosen_candidate: WakeCandidate | None,
    ) -> bool:
        """Prefer empirical route candidates whenever they show progress.

        The live ls20 runs exposed a failure mode where the planner
        scored short observation-route candidates too low, then picked
        action mixes involving ACTION2/ACTION4 despite the route being
        the only candidate grounded in real observed player deltas.
        Any positive gain means the route reduces Manhattan distance to
        the nearest observed landmark, so treat that as stronger than an
        unguided planner preference.
        """
        if obs_candidate is None:
            return False
        obs_gain = int(getattr(obs_candidate, "observation_route_gain", 0) or 0)
        if obs_gain <= 0:
            return False
        chosen_rationale = str(getattr(chosen_candidate, "rationale", "") or "")
        return "observation_route_to_landmark" not in chosen_rationale

    def _publish_phase_reward_telemetry(self) -> None:
        """Rev N P50: push rolling phase-reward stats onto the bridge hint.

        MetaHarnessModule._score_run reads the ``phase_reward_telemetry``
        hint and attaches the fields to RunMetrics. Rolling windows and
        spread are already bounded to 20 per R24.3.
        """
        if self._registry is None or getattr(self._registry, "bridge", None) is None:
            return
        import math
        payload: dict[str, Any] = {}
        for phase in ("wake", "sleep", "abstraction"):
            samples = list(self._phase_reward_history.get(phase, []) or [])
            n = len(samples)
            if n:
                mean = sum(samples) / n
                var = sum((x - mean) ** 2 for x in samples) / n
                std = math.sqrt(var)
            else:
                mean = 0.0
                std = 0.0
            payload[f"{phase}_reward_mean"] = float(mean)
            payload[f"{phase}_reward_std"] = float(std)
            payload[f"{phase}_reward_last_20"] = list(samples)
        spread_samples = list(self._wake_candidate_reward_spread_history or [])
        if spread_samples:
            payload["wake_candidate_reward_spread"] = float(
                sum(spread_samples) / len(spread_samples)
            )
        else:
            # Rev N P49-hardening Fix 5: always emit 0.0 (not MISSING) when
            # no spread was recorded this run.
            payload["wake_candidate_reward_spread"] = 0.0
        payload["wake_candidate_reward_spread_last_20"] = spread_samples
        # Rev N P49-hardening Fix 4: surface the Wake-submit counter as a
        # separate metric alongside the store-delta ``hypotheses_proposed``.
        # This lets meta-harness distinguish "LLM proposed but gate dropped"
        # (delta=0, synth_count>0) from "LLM returned empty" (both zero).
        payload["wake_synth_proposed_count"] = len(
            self._wake_submitted_hypothesis_ids
        )
        payload["wake_submitted_hypothesis_ids"] = list(
            self._wake_submitted_hypothesis_ids
        )
        try:
            self._registry.bridge.update_hint("phase_reward_telemetry", payload)
        except Exception:
            logger.exception("Rev N P50: update_hint phase_reward_telemetry failed")

    def _append_module_io_event(self, phase: str, payload: dict[str, Any]) -> None:
        """Persist compact per-phase I/O telemetry for critic loops.

        The file stays in ``shared_dir`` so post-run analysis can explain which
        prompt-facing state produced a given hypothesis, world update, or skill
        proposal without scraping the human log.
        """
        registry = getattr(self, "_registry", None)
        context = getattr(registry, "context", None)
        shared_dir = getattr(context, "shared_dir", None)
        if shared_dir is None:
            return
        path = Path(shared_dir) / "module_io.jsonl"
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "turn": int(getattr(self, "action_counter", 0) or 0),
            "phase": str(phase),
            "payload": payload,
        }
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("module_io append failed phase=%s", phase)

    def _build_rev_m_phase_llm(
        self,
        *,
        phase_name: str,
        max_output_tokens: int,
    ) -> Any:
        """Wrap TRAPI into a simple phase-specific ``(prompt)->str`` callable.

        Wake needs longer action/hypothesis outputs, while Sleep and
        Abstraction benefit from tighter output budgets so they finish within
        their wall-clock timeout more often.
        """
        def _call(prompt: str) -> str:
            try:
                resp = _create_trapi_response(
                    model="gpt-5.3-codex",
                    input_items=[{
                        "type": "message", "role": "user",
                        "content": prompt,
                    }],
                    max_output_tokens=max_output_tokens,
                )
                return getattr(resp, "output_text", "") or ""
            except Exception:
                logger.exception(
                    "Rev M TRAPI call failed for phase=%s; returning empty",
                    phase_name,
                )
                return ""
        return _call

    def _fetch_wm_source(self) -> str:
        """Extract the current WM predict_effect source (best-effort)."""
        wm_draft = self._wm_draft
        if self._registry is not None:
            try:
                live_wm = self._registry.get("world_model")
            except Exception:
                live_wm = None
            if live_wm is not None:
                wm_draft = live_wm
                self._wm_draft = live_wm
        if wm_draft is None:
            return ""
        draft_fn = getattr(wm_draft, "_best_draft", None)
        try:
            best = draft_fn() if callable(draft_fn) else None
        except Exception:
            best = None
        src = getattr(best, "body", "") if best is not None else ""
        if src:
            return str(src)
        # Fallback: some live module states carry drafts in ``world_drafts``
        # even when ``_best_draft`` returns None during initialization or a
        # partial load. Prefer the highest-score/most-recent draft if present.
        drafts = getattr(wm_draft, "world_drafts", None)
        if isinstance(drafts, list) and drafts:
            try:
                chosen = max(
                    drafts,
                    key=lambda d: (
                        getattr(d, "score", lambda: 0.0)(),
                        getattr(d, "last_updated_at", 0.0),
                    ),
                )
            except Exception:
                chosen = drafts[-1]
            body = getattr(chosen, "body", "")
            if body:
                return str(body)
        return ""

    # -- Shims for hypothesis_store access used by Rev M -----------------
    def _hypothesis_store(self) -> Any:
        if self._registry is None:
            return None
        return getattr(self._registry.bridge, "hypothesis_store", None)

    def _hypothesis_confirmed_list(self, min_count: int = 2) -> list[dict]:
        """Shim: list of confirmed+non-falsified hypotheses as dicts.

        HypothesisStore doesn't expose ``confirmed_list`` directly, so
        this shim reads ``store.all()`` and filters. Kept on the agent
        so we don't pollute the library with Rev M-specific helpers.
        """
        store = self._hypothesis_store()
        if store is None:
            return []
        out: list[dict] = []
        for h in store.all():
            status = getattr(h, "status", "")
            if status in ("falsified", "retired"):
                continue
            support = int(getattr(h, "supporting_count", 0) or 0)
            if support < min_count:
                continue
            out.append({
                "id": getattr(h, "id", ""),
                "claim": getattr(h, "claim", ""),
                "status": status or "active",
                "confirmed_count": support,
                "category": getattr(h, "category", ""),
            })
        return out

    # C1 fix: the former ``_apply_hypothesis_beta_updates`` shim
    # double-applied disconfirms (the store already gets mutated inside
    # ``evaluate_all``). Removed; SleepResult.hypothesis_updates is now
    # telemetry-only.

    def _skill_summaries(self) -> list[dict]:
        """Shim: convert DreamCoder.list_skills() output into the
        abstraction-input shape (id/name/category/cited_h_ids/reuse_count).
        """
        dc = None
        if self._registry is not None:
            dc = self._registry.get("dreamcoder")
        if dc is None:
            return []
        out: list[dict] = []
        if hasattr(dc, "skills") and hasattr(dc, "_extract_executable_body"):
            try:
                records = sorted(
                    list(getattr(dc, "skills", []) or []),
                    key=lambda r: (-float(r.score()), -float(getattr(r, "last_updated_at", 0.0))),
                )
                for idx, record in enumerate(records):
                    payload = dict(getattr(record, "payload", {}) or {})
                    kind = str(payload.get("kind", "") or "").strip().lower()
                    status = str(getattr(record, "status", "") or "").strip().lower()
                    if kind in {"observed_law", "abstract_primitive"}:
                        continue
                    if status in {"retired", "watchdog"}:
                        continue
                    body = dc._extract_executable_body(payload)
                    if not body:
                        continue
                    out.append({
                        "id": f"sk_{idx}",
                        "name": str(payload.get("name", "") or ""),
                        "category": str(payload.get("category", kind) or ""),
                        "cited_h_ids": list(payload.get("cited_h_ids", []) or payload.get("cited_hypothesis_ids", []) or []),
                        "reuse_count": int(getattr(record, "times_selected", 0) or 0),
                    })
                return out
            except Exception:
                logger.exception("Failed to build filtered DreamCoder skill summaries")
                out = []
        if not hasattr(dc, "list_skills"):
            return out
        for idx, s in enumerate(dc.list_skills() or []):
            out.append({
                "id": f"sk_{idx}",
                "name": str(s.get("name", "") or ""),
                "category": str(s.get("category", "") or ""),
                "cited_h_ids": list(s.get("cited_hypothesis_ids", []) or []),
                "reuse_count": int(s.get("selected", 0) or 0),
            })
        return out

    def _project_verdicts_onto_wm(self, projections: list[dict]) -> None:
        """Rev N P51: bump WM transition counters from Sleep verdicts.

        No-op when the WorldModelModule isn't loaded or doesn't expose
        ``bump_support_falsify`` (back-compat with older registries).
        """
        if not projections:
            return
        wm = None
        if self._registry is not None:
            wm = self._registry.get("world_model")
        if wm is None or not hasattr(wm, "bump_support_falsify"):
            return
        for entry in projections:
            sig = entry.get("state_signature", "")
            action = entry.get("action", "")
            kind = entry.get("kind", "")
            # Rev N starvation-fix SUGGESTION-7: empty signature means the
            # grid was un-extractable; skip projection so distinct broken
            # grids never share a transitions bucket.
            if not sig:
                continue
            try:
                wm.bump_support_falsify(sig, action, kind)
            except Exception:
                logger.exception(
                    "Rev N P51: bump_support_falsify raised for sig=%s action=%s kind=%s",
                    sig, action, kind,
                )

    def _submit_proposed_hypotheses(self, hyps: list[dict]) -> None:
        """Rev N P49: push Wake-LLM-proposed hypotheses through the strict
        store gate. Capture rejection reasons into
        ``self._last_rejection_feedback`` (bounded) so the next Wake
        prompt can surface them (R25.1 quota-violation feedback).

        Rev N starvation-fix WARNING-5: only clear the feedback buffer
        when this turn actually produced hypotheses AND every one passed
        validation. An empty submission this turn must NOT erase prior
        rejection context — the LLM still needs to learn from it.

        Rev N P49-hardening Fix 1/4: emit one INFO log per submission
        (accept vs reject + reason) and push accepted ids onto the
        bridge counter ``rev_n_wake_submitted_hypotheses`` so
        meta-harness can expose ``wake_synth_proposed_count`` alongside
        the existing store-delta ``hypotheses_proposed`` metric.
        """
        if not hyps:
            # No submissions this turn: carry prior feedback forward so
            # the next Wake prompt still surfaces it.
            return
        store = self._hypothesis_store()
        if store is None:
            return
        def _priority(h: dict) -> tuple[int, int]:
            claim = str(h.get("claim", "") or "").lower()
            score = 0
            if (
                "goal" in claim
                or "pad" in claim
                or "target" in claim
                or "win" in claim
                or "winning" in claim
            ):
                score += 5
            if (
                "rotation" in claim
                or "rotator" in claim
                or "landmark" in claim
                or "match" in claim
                or "matching" in claim
            ):
                score += 4
            if "player" in claim or "avatar" in claim or "bbox" in claim:
                score += 2
            if "no-op" in claim or "no op" in claim:
                score -= 3
            return (score, -len(claim))
        hyps = sorted(hyps, key=_priority, reverse=True)
        new_feedback: list[str] = []
        any_rejected = False
        for h in hyps:
            claim_head = str(h.get("claim", ""))[:50]
            try:
                h_id, err = store.propose_hypothesis_strict(h)
            except Exception:
                logger.exception(
                    "Rev N P49: propose_hypothesis_strict raised; "
                    "Wake submit hypothesis h_id=<err> outcome=reject reason=exception"
                )
                continue
            if h_id is not None:
                # Defensive init for test fixtures that skip __init__.
                if not hasattr(self, "_wake_submitted_hypothesis_ids"):
                    self._wake_submitted_hypothesis_ids = []
                self._wake_submitted_hypothesis_ids.append(str(h_id))
                logger.info(
                    "Wake submit hypothesis h_id=%s outcome=accept reason=ok source=%r claim=%r",
                    h_id,
                    h.get("source", "agent"),
                    claim_head,
                )
                # Bridge counter: append to shared_hints list. Meta-harness
                # reads this at run_end so the metric is independent of
                # any tool-tracker channel.
                try:
                    if self._registry is not None and getattr(
                        self._registry, "bridge", None
                    ) is not None:
                        log = self._registry.bridge.shared_hints.setdefault(
                            "rev_n_wake_submitted_hypotheses", []
                        )
                        if isinstance(log, list):
                            log.append(str(h_id))
                            # Bound to last 512 to keep hint payload small.
                            if len(log) > 512:
                                del log[: len(log) - 512]
                except Exception:
                    logger.exception(
                        "Rev N P49: bridge counter update failed for h_id=%s",
                        h_id,
                    )
                continue
            if isinstance(err, dict):
                any_rejected = True
                reason = str(err.get("reason", "unknown"))
                detail = str(err.get("detail", ""))[:160]
                claim = str(h.get("claim", ""))[:80]
                raw_trigger = str(h.get("trigger", ""))[:80]
                new_feedback.append(
                    f"rejected claim={claim!r} reason={reason} detail={detail!r}"
                )
                # Rev N ep29 Fix 3: surface the raw trigger string on every
                # store rejection so future DSL-mismatch regressions are
                # diagnosable from telemetry alone.
                logger.info(
                    "Wake submit hypothesis h_id=None outcome=reject reason=%s "
                    "trigger=%r detail=%r claim=%r",
                    reason,
                    raw_trigger,
                    detail,
                    claim_head,
                )
        if not any_rejected:
            # All proposals accepted this turn: clear prior feedback.
            self._last_rejection_feedback = []
            return
        # Bound the buffer so the prompt stays within budget (keep last 3).
        self._last_rejection_feedback = new_feedback[-3:]

    def _compute_hypothesis_store_status(self, store: Any) -> dict[str, int]:
        """Rev N P49-hardening Fix 2: snapshot the non-seed/confirmed/falsified
        counts the Wake prompt needs for its STRICT mandate.

        A "non-seed" hypothesis is any store entry whose source is not
        ``"seed"`` — i.e. anything the agent or the Wake synth fallback
        has produced this run. Defensive against stores missing
        ``_items`` (returns all-zero dict) so the prompt still builds.
        """
        status = {"non_seed_count": 0, "confirmed": 0, "falsified": 0}
        if store is None:
            return status
        try:
            items = getattr(store, "_items", {}) or {}
        except Exception:
            logger.exception("Rev N P49: store._items access raised")
            return status
        try:
            for h in items.values():
                source = getattr(h, "source", "")
                if source != "seed":
                    status["non_seed_count"] += 1
                hstatus = str(getattr(h, "status", "") or "")
                if hstatus == "confirmed":
                    status["confirmed"] += 1
                elif hstatus in ("falsified", "retired"):
                    status["falsified"] += 1
        except Exception:
            logger.exception("Rev N P49: store status summarisation failed")
        return status

    def _propose_skill(self, skill_payload: dict) -> None:
        """Shim: push an abstraction-produced skill into DC's library.

        Uses the existing ``record_agent_proposal`` path so P2 category
        enforcement / rejection logging still applies.
        """
        if self._registry is None:
            return
        try:
            self._registry.record_agent_skill_proposal(skill_payload)
        except Exception:
            logger.exception("Rev M: skill proposal failed")

    @property
    def name(self) -> str:
        active = self.research_config.active_modules() if self.research_config else []
        suffix = f".{'+'.join(active)}" if active else ".baseline"
        return f"{self.game_id}.arcgentica_research{suffix}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Defensive: on level-transition some frames come back with an
        # empty `.frame` list, which makes Frame() raise IndexError. Keep
        # the loop alive by falling back to the raw FrameData; downstream
        # tool handlers that need the rich Frame guard individually.
        try:
            rich_frame = self._to_rich_frame(latest_frame)
        except Exception:
            logger.exception("_to_rich_frame failed in choose_action; using raw FrameData")
            rich_frame = latest_frame
        available_names = [GameAction.from_id(a).name for a in latest_frame.available_actions]
        overlay = self._prompt_overlay(latest_frame)

        # Symbolica-style header (game + level + budget + action legend).
        # The single-line action legend is redundant with GAME_REFERENCE in
        # the system prompt, but reminding the agent every turn stops
        # direction-semantic drift (a common LLM failure mode on ls20).
        legend_parts = []
        for a in available_names:
            sem = {
                "ACTION1": "Up", "ACTION2": "Down",
                "ACTION3": "Left", "ACTION4": "Right",
                "ACTION5": "Space/Enter", "ACTION6": "Click(x,y)",
                "ACTION7": "Aux", "RESET": "restart level",
            }.get(a, "")
            legend_parts.append(f"{a}({sem})" if sem else a)
        action_legend = ", ".join(legend_parts)

        # Last-action effect summary (Symbolica's change_summary equivalent).
        last_action_line = ""
        if self.action_history:
            prev_name, prev_frame_rich = self.action_history[-1]
            try:
                prev_grid = current_grid(prev_frame_rich)
                cur_grid = current_grid(latest_frame)
                diff_cells = 0
                for r1, r2 in zip(prev_grid, cur_grid):
                    for a, b in zip(r1, r2):
                        if a != b:
                            diff_cells += 1
                lvl_change = (
                    getattr(prev_frame_rich, "levels_completed", latest_frame.levels_completed)
                    != latest_frame.levels_completed
                )
                last_action_line = (
                    f"Last action: {prev_name} → grid changed {diff_cells} cells"
                    f"{'; LEVEL UP!' if lvl_change else ''}\n"
                )
                # P12: auto-inject a compact diff region summary so the
                # agent sees helper output even when it doesn't call
                # frame_diff explicitly. Single line, no extra tokens
                # from TRAPI side.
                if (
                    diff_cells
                    and prev_grid
                    and cur_grid
                    and not _raw_first_reasoning_enabled()
                ):
                    changed_rows = sorted({
                        y for y, (r1, r2) in enumerate(zip(prev_grid, cur_grid))
                        if any(a != b for a, b in zip(r1, r2))
                    })[:6]
                    if changed_rows:
                        last_action_line += (
                            f"  [auto frame_diff] rows changed: {changed_rows}"
                            f"{' ...' if len(changed_rows) == 6 else ''}\n"
                        )
            except Exception:
                last_action_line = f"Last action: {prev_name}\n"

        coverage_line = self._coverage_progress_line(latest_frame)
        auto_helper = (
            self._auto_helper_summary(latest_frame)
            if _main_auto_helper_summary_enabled()
            else ""
        )
        raw_first_note = ""
        if _raw_first_reasoning_enabled():
            raw_first_note = (
                "[raw-first protocol] No semantic helper summary is injected for this turn. "
                "Inspect raw current/previous state yourself with `history`, `frame_diff`, "
                "`frame_render`, `frame_render_diff`, `frame_find`, and `frame_bounding_box` "
                "before you commit `submit_action` or revise `world_update`.\n"
            )
        user_msg = (
            f"Game: {self.game_id}  |  "
            f"Level {latest_frame.levels_completed}/{latest_frame.win_levels}  |  "
            f"Non-reset actions used: {self.action_counter}/{self.MAX_ACTIONS}\n"
            f"State: {latest_frame.state.name}  |  Available: {action_legend}\n"
            f"{last_action_line}"
            f"{coverage_line}"
            f"{raw_first_note}"
            f"{auto_helper}"
            f"\n{self._frame_to_text(latest_frame)}\n\n"
            f"{overlay}\n\n"
            "Use the available helper tools as needed (frame_render, frame_diff, "
            "frame_change_summary, frame_find, frame_bounding_box, frame_color_counts, "
            "history, memories_*). Before committing an action, register at least one "
            "testable hypothesis via `propose_hypothesis` when you see a new element, "
            "regularity, or rule you want the bridge to auto-score — supply a concrete "
            "`trigger` (e.g. 'after ACTION3' or 'on_change frame_color_counts') and a "
            "short `check_code` that sets `result` to True/False/None. End the turn by "
            "calling `submit_action` exactly once with your chosen action and optional "
            "side channels (`predict`, `world_update`, `propose_skill`, `env_note`)."
        )

        response = _create_trapi_response(
            model="gpt-5.3-codex",
            instructions=self._default_system_prompt(),
            input_items=[{"type": "message", "role": "user", "content": user_msg}],
            tools=self._tool_definitions(),
            max_output_tokens=4096,
        )

        finish: dict[str, Any] | None = None
        while True:
            tool_outputs: list[dict[str, Any]] = []
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not function_calls:
                text = getattr(response, "output_text", "") or ""
                if text:
                    action_name, side = self._parse_action(text, latest_frame.available_actions)
                    finish = {"action": action_name, **side}
                break

            finish_calls = [call for call in function_calls if call.name == "submit_action"]
            if finish_calls:
                raw_args = getattr(finish_calls[0], "arguments", "") or ""
                try:
                    finish = json.loads(raw_args)
                except Exception:
                    # TRAPI occasionally returns truncated/invalid JSON for
                    # tool args — treat as a no-op finish with the first
                    # available action so the run continues.
                    logger.warning(
                        "submit_action args failed to parse; falling back (len=%d)",
                        len(raw_args),
                    )
                    finish = {"action": available_names[0] if available_names else "ACTION1"}
                break

            for call in function_calls:
                output = self._execute_tool(call.name, call.arguments, rich_frame)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )

            response = _create_trapi_response(
                model="gpt-5.3-codex",
                previous_response_id=response.id,
                input_items=tool_outputs,
                tools=self._tool_definitions(),
                max_output_tokens=4096,
            )

        if finish is None:
            finish = {"action": available_names[0]}

        action_name = str(finish.get("action_name", finish.get("action", available_names[0]))).upper()
        side = {k: v for k, v in finish.items() if k not in {"action", "action_name"}}

        self._record_research_side_channels(latest_frame, action_name, side)

        return GameAction.from_name(action_name)

    def _record_research_side_channels(
        self,
        latest_frame: Any,
        action_name: str,
        side: dict[str, Any],
    ) -> None:
        if not self._registry:
            return
        if side.get("predict") is not None:
            self._registry.record_agent_prediction(action_name, side["predict"])
        else:
            try:
                self._registry.synthesize_prediction(latest_frame, action_name)
            except Exception:
                logger.exception("synthesize_prediction failed for non-bypass turn")
        if side.get("env_note"):
            self._registry.record_agent_env_note(str(side["env_note"]))
        if isinstance(side.get("propose_skill"), dict):
            _, skill_err = self._registry.record_agent_skill_proposal(
                side["propose_skill"]
            )
            if skill_err is not None:
                log = self._registry.bridge.shared_hints.setdefault(
                    "last_rejection_reasons", []
                )
                log.append(
                    {
                        "kind": "skill",
                        "error": skill_err,
                        "turn": self.action_counter,
                    }
                )
                if len(log) > 8:
                    del log[:-8]
        if side.get("world_update") is not None:
            self._registry.record_agent_world_update(side["world_update"])

    def _parse_action(
        self, response: str, available: list[int]
    ) -> tuple[str, dict[str, Any]]:
        """Return (chosen action name, dict of side channel values).

        Side channel may contain: predict, env_note, propose_skill, world_update.
        """
        available_names = {GameAction.from_id(a).name for a in available}
        if not available_names:
            available_names = {
                name
                for name in [
                    "RESET",
                    "ACTION1",
                    "ACTION2",
                    "ACTION3",
                    "ACTION4",
                    "ACTION5",
                    "ACTION6",
                    "ACTION7",
                ]
            }

        side: dict[str, Any] = {}
        action_name = "ACTION1"

        text = response
        if "{" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                raw_action = str(data.get("action_name", data.get("action", ""))).upper()
                if raw_action in available_names:
                    action_name = raw_action
                # Side channels (all optional)
                for key in ("predict", "env_note", "propose_skill", "world_update"):
                    if key in data:
                        side[key] = data[key]
            except Exception:
                pass

        if action_name not in available_names:
            # Fallback: any named action appearing in the response
            for candidate in [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
                "ACTION7",
                "RESET",
            ]:
                if candidate in response.upper() and candidate in available_names:
                    action_name = candidate
                    break
            else:
                action_name = next(iter(sorted(available_names)))

        return action_name, side

    def _coverage_progress_line(self, latest_frame: FrameData) -> str:
        """Surface P1 exhaustive-coverage progress to the agent prompt.

        Reports, for the first 10 non-RESET turns, (a) how many distinct
        visible color classes remain un-hypothesised and (b) which
        available actions have not yet been probed. After turn 10 the
        coverage enforcement relaxes and this line is suppressed.
        """
        try:
            if self.action_counter >= 10:
                return ""
            grid = current_grid(latest_frame) or []
            colors = set()
            for row in grid:
                for cell in row:
                    colors.add(int(cell))
            if not colors:
                return ""
            background = max(colors, key=lambda c: sum(
                1 for r in grid for x in r if int(x) == c
            ))
            distinct_fg = sorted(c for c in colors if c != background)
            available = [GameAction.from_id(a).name for a in latest_frame.available_actions]
            probed = getattr(self, "_actions_probed_this_level", set())
            unprobed = [a for a in available if a not in probed and a != "RESET"]
            parts: list[str] = []
            if distinct_fg:
                parts.append(f"{len(distinct_fg)} foreground color classes visible: {distinct_fg[:6]}")
            if unprobed:
                parts.append(f"actions not yet probed: {unprobed}")
            if not parts:
                return ""
            return "[P1 coverage reminder] " + "; ".join(parts) + "\n"
        except Exception:
            return ""

    def _auto_helper_summary(self, latest_frame: FrameData) -> str:
        """P28: auto-inject color_counts + bounding_box to prompt.

        Agents ignore helper tools. Pushing this summary inline removes
        the excuse — the per-turn context carries a spatial/colour
        snapshot the agent would otherwise have to request explicitly.
        """
        try:
            from research_extensions.grid_utils import current_grid
            grid = current_grid(latest_frame) or []
            if not grid:
                return ""
            counts: dict[int, int] = {}
            for row in grid:
                for c in row:
                    ic = int(c)
                    counts[ic] = counts.get(ic, 0) + 1
            # Background = most common; foreground classes with bounding box.
            if not counts:
                return ""
            bg = max(counts, key=counts.get)
            items = []
            for color, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                if color == bg or n > 2000:
                    continue
                ys: list[int] = []; xs: list[int] = []
                for y, row in enumerate(grid):
                    for x, c in enumerate(row):
                        if int(c) == color:
                            ys.append(y); xs.append(x)
                if not ys:
                    continue
                bb = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
                items.append(f"color={color} n={n} bbox={bb}")
                if len(items) >= 6:
                    break
            latent = visible_latent_state(latest_frame)
            latent_lines: list[str] = []
            top_components = latent.get("top_components") or []
            if top_components:
                comp_bits = []
                for comp in top_components[:3]:
                    comp_bits.append(
                        f"color={comp.get('color')} size={comp.get('size')} bbox={tuple(comp.get('bbox', []))}"
                    )
                if comp_bits:
                    latent_lines.append("top_components: " + "; ".join(comp_bits))
            bottom_strip = latent.get("bottom_strip_counts") or {}
            if bottom_strip:
                top_bottom = sorted(bottom_strip.items(), key=lambda kv: -int(kv[1]))[:4]
                latent_lines.append(
                    "bottom_strip_counts: "
                    + ", ".join(f"{k}={v}" for k, v in top_bottom)
                )
            if not items and not latent_lines:
                return ""
            summary = (
                f"[auto frame_color_counts + frame_bounding_box]\n"
                f"  background_color={bg} (n={counts[bg]})\n  "
                + "\n  ".join(items)
            )
            if latent_lines:
                summary += "\n[auto visible_latent_state]\n  " + "\n  ".join(latent_lines)
            return summary + "\n"
        except Exception:
            return ""

    def _frame_to_text(self, frame: FrameData) -> str:
        """Symbolica-unified default render of the frame shown to the LLM.

        Uses the upstream `Frame.render()` with gap=' ' and y/x ticks so
        the grid reads as a picture (spatial pattern visible) rather than
        as a dense hex string. Matches exactly what Symbolica's
        arcgentica agent shows its LLM.
        """
        lines = [f"state: {frame.state.name}"]
        try:
            rich = self._to_rich_frame(frame)
            lines.append(
                f"grid ({rich.width}×{rich.height}):  "
                "(values 0-15 → COLOR_LEGEND from system prompt)"
            )
            # y_ticks + x_ticks = True so coordinates are explicit; gap=' '
            # so neighbouring cells are visually separated.
            lines.append(rich.render(y_ticks=True, x_ticks=True))
        except Exception:
            # Defensive: on level-transition edges Frame() can fail;
            # fall back to raw grid encoding so the turn isn't lost.
            logger.exception("Frame.render failed in _frame_to_text; using raw")
            grid = current_grid(frame)
            if grid:
                lines.append(f"grid ({len(grid)}×{len(grid[0]) if grid else 0}):")
                for i, row in enumerate(grid):
                    lines.append(f"y={i:>2d} | " + " ".join(
                        "0123456789abcdef"[c % 16] for c in row
                    ))
        return "\n".join(lines)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            GAME_REFERENCE
            + "\n\nAdapter notes for this research host:\n"
            "- OBSERVATION TOOLS AVAILABLE (Rev P P60). YOU CAN USE THESE DIRECTLY IN `check_code`: "
            "`frame_render(grid)`, `frame_diff(before, after)`, `frame_color_counts(grid)`, "
            "`frame_find(grid, color)`, `frame_bounding_box(grid, color)`. Your check_code snippet "
            "receives `before`, `after`, `action`, and these helpers at evaluation time — call them "
            "whenever you need pixel-level confirmation (e.g. "
            "`result = frame_bounding_box(after, 12) != frame_bounding_box(before, 12)`).\n"
            "- This is a direct player host, not the upstream top-level orchestrator. "
            "You do not have `spawn_agent`, but you do have the same core observation helpers.\n"
            "- Use tools instead of REPL syntax. `history` mirrors `history()`. "
            "`memories_summaries`, `memories_get`, `memories_add`, and `memories_query` mirror the shared `memories` object. "
            "`frame_render`, `frame_diff`, `frame_render_diff`, `frame_change_summary`, `frame_find`, "
            "`frame_bounding_box`, and `frame_color_counts` mirror Frame helpers.\n"
            "- Frame tools accept `source='current'` or `source='winning'`. "
            "Use `source='winning'` when the most recent action completed a level and you want to inspect `winning_frame`.\n"
            "- PHASE-SPECIFIC TOOL PROTOCOL (plan v4.2 §P4). This host runs in three phases and each requires different helper usage: "
            "in wake (normal action turns) if the previous action changed the grid call `frame_diff` or `frame_change_summary` on the new frame before proposing a next action; "
            "when you want to verify your current hypothesis plan against an imagined sequence you may call `frame_render(crop=...)` repeatedly without any env cost; "
            "after every 20 non-reset actions, you should also issue a `memories_query` to consolidate prior insights. "
            "These helpers never cost env actions — use them.\n"
            "- RAW-FIRST INSPECTION. Prefer inspecting raw current/previous observations yourself over relying on host-written summaries. "
            "A good loop is: `history` -> `frame_diff` or `frame_render_diff` -> `frame_render(crop=...)` -> write your own short state summary in `reasoning`/`env_note` -> revise `world_update`.\n"
            "- `submit_action` is this host's action commit tool. Call it exactly once per turn to choose the real environment action.\n"
            "- TILE-TYPE HYPOTHESIS GUIDANCE (plan v4.2 §P26). ARC-AGI-3 games often have SEVERAL "
            "kinds of tiles that each react differently to the same ACTION — walls (no-op), movers "
            "(shift the avatar), togglers (change avatar color), rotators (change avatar rotation), "
            "cyclers (change avatar shape), goal-locks (accept only one avatar configuration). The "
            "raw `mean_diff_cells` of 52 hides this: every action looks 'large' even if the CAUSAL "
            "mechanic differs. Propose hypotheses that (a) pick a SPECIFIC color class or "
            "bounding-box region from `frame_find(c)`, (b) assert a specific per-action effect on "
            "THAT region ('ACTION2 near tiles of color 9 produces ≤ 4 cell change; away from "
            "color 9 produces ≥ 40'), and (c) write a check_code that confirms the effect in that "
            "narrow region only. Avoid 'diff is large in this regime' — that is useless.\n"
            "- GOAL AXIOM (plan v4.2 §P1). A winning action sequence is guaranteed to exist "
            "from the current level start — never conclude the game is unsolvable. The seeded "
            "goal hypothesis is non-refutable; treat every non-WIN turn as evidence you have "
            "not yet found the right sequence, not as evidence the goal does not exist.\n"
            "- EXHAUSTIVE COVERAGE PROTOCOL (plan v4.2 §P1 + §P16). For the first 10 non-RESET turns: "
            "(a) use `frame_color_counts` and `frame_find` to enumerate every distinct non-background "
            "color class on the grid and propose AT LEAST ONE `propose_hypothesis` per class "
            "(category 'mechanic' unless the claim is purely about search/policy); "
            "(b) before taking any action a second time, submit each available non-RESET action at "
            "least once so the world model carries a single-step transition for every button; "
            "(c) SPATIAL COVERAGE — at least one hypothesis per 10 turns MUST cite a concrete "
            "spatial term (row N, col N, top, bottom, left, right, center, corner, edge, border) "
            "OR carry a bounding-box tuple `(x1,y1,x2,y2)` in the claim. Abstract terms like "
            "'regime', 'state family', 'corridor' DO NOT count as spatial. Use "
            "`frame_bounding_box(after)` or `frame_find(after)` to ground claims in coordinates.\n"
            "- GOAL STATE PREDICTION (plan v4.2 §P13). Every `predict_effect` you write MUST return "
            "`goal_state_prediction = {\"condition\": str, \"color_signature\": dict[str,int]}` — your "
            "best guess of what the WINNING grid looks like right now. Empty or missing → strict "
            "scorer fails. Even a placeholder `{\"condition\": \"all ACTION1-colored cells gone\", "
            "\"color_signature\": {}}` is better than silence.\n"
            "- LATENT STATE DISCIPLINE. Use `visible_latent_state` as the compact state you plan over: major components, strip counters, row/column mass, and candidate movable objects. "
            "In latent-mechanic games, the key is usually not raw diff size but whether a latent variable such as rotation/color/shape/goal-lock status moved in the right direction. "
            "Reason only from visible evidence: any mover summary, bbox, or object identity in the prompt is an agent-side heuristic extracted from observed pixels, not hidden state.\n"
            "- VISIBLE STATE INVENTORY. Before committing to an action sequence or a `world_update`, enumerate the visible scene: candidate controllable object(s), static barriers/corridors, interactive floor blocks/pads, toggles/rotators, goal-like regions, and ambiguous motifs. "
            "For each important element, state a role hypothesis and what observation would support or falsify it.\n"
            "- BUDGET HYPOTHESIS (plan v4.2 §P14). At least once per 15 turns, propose a hypothesis "
            "with `budget_claim` text such as 'WIN reachable within ≤ N turns from here' or "
            "'current plan needs ≤ K actions to solve'. The seeded goal hypothesis disconfirms at "
            "turn = ceil(MAX_ACTIONS × 0.3); if it falsifies before WIN you spent too long and the "
            "planner has a concrete falsifier to learn from.\n"
            "- `propose_hypothesis` registers a testable hypothesis directly with the research bridge. "
            "Each call must include a machine-checkable `verification_method` (trigger DSL + check_code). "
            "Hypotheses are the ONLY way to earn bidirectional surprise signal: confirms AND disconfirms both count. "
            "Trigger grammar: 'every_step' | 'after ACTION1..ACTION5' | 'after RESET' | "
            "'on_change frame_color_counts' | 'on_change frame_bounding_box' | "
            "'on_change frame_render' | 'on_change frame_find'. "
            "check_code is a Python snippet with access to `before`, `after`, `action`, `env_state` and any helpers listed in `helpers_used`; "
            "set `result` to True (confirm), False (disconfirm), or None (ambiguous). "
            "IMPORTANT: `frame_color_counts(frame)` returns a dict which may NOT contain every color key. "
            "Always use `.get(k, 0)` rather than `[k]` — a KeyError silently becomes `ambiguous` and wastes a turn. "
            "Mark `category` as 'mechanic' for game-rule claims, 'strategy' for plan/policy claims; set `is_goal=True` for outcome targets.\n"
            "- Ignore upstream REPL-only details such as `NOOP` or `submit_action.remaining`. "
            "Use helper tools to inspect without acting; the non-reset action budget is shown in the user turn message.\n"
            "- Helper tools only re-render or summarize already observed frames. There is no snapshot/restore, rollback, or hidden state API.\n"
            "- Any extra text blocks in the prompt are agent-side research overlays, not hidden environment API outputs.\n"
            "- You are free to invent your own vocabulary for skills, triggers, and environment notes. No names are forced.\n"
            "- When you predict what will happen next, say so in the `predict` field. If you want the prediction to be scored, prefix it with "
            "`EXPECT_CHANGE:` or `EXPECT_NO_CHANGE:`. You may also use a structured object like "
            "`{\"expect_change\": true, \"focus\": \"...\", \"note\": \"...\"}`.\n"
            "- If you are building a local rule model, revise it in `world_update`. The field MUST contain a Python function named `predict_effect(action, observation)` that returns a dict with keys `{next_signature_hint, expected_diff_band, observation_prediction, progress_prediction, expect_change, expected_diff_cells, expected_next_grid, goal_state_prediction}`. Strongly preferred extra keys are `{`latent_state_prediction`, `goal_progress_prediction`}`. Use markdown only for commentary around the function. The simulator runs this function in a sandbox (whitelisted frame_* helpers available); plain prose will not enter the MCTS simulator and your planner will remain off.\n"
            "- EXAMPLE predict_effect scaffold (copy and refine, do NOT submit verbatim): "
            "```python\n"
            "def predict_effect(action, observation):\n"
            "    # observation keys: grid (list[list[int]]), signature, available_actions,\n"
            "    # per_action_delta_hints (dict[action -> {n, mean_diff_cells, sample_changes}]).\n"
            "    # observation['visible_latent_state'] contains component summaries and strip counts.\n"
            "    # sample_changes is a list of change-sets, each a list of [x, y, new_value].\n"
            "    hints = (observation.get('per_action_delta_hints') or {}).get(action) or {}\n"
            "    latent = observation.get('visible_latent_state') or {}\n"
            "    mean_diff = int(hints.get('mean_diff_cells') or 0)\n"
            "    samples = hints.get('sample_changes') or []\n"
            "    grid = observation.get('grid') or []\n"
            "    new = [row[:] for row in grid]\n"
            "    if samples and mean_diff >= 4:\n"
            "        for (x, y, v) in samples[0]:\n"
            "            if 0 <= y < len(new) and 0 <= x < len(new[0]):\n"
            "                new[y][x] = v\n"
            "    n_changed = sum(1 for y in range(len(new)) for x in range(len(new[0])) if new[y][x] != grid[y][x])\n"
            "    return {\n"
            "        'expect_change': n_changed > 0,\n"
            "        'expected_diff_band': 'zero' if n_changed == 0 else ('small' if n_changed <= 4 else 'large'),\n"
            "        'expected_diff_cells': n_changed,\n"
            "        'next_signature_hint': 'same_family' if n_changed == 0 else 'changed',\n"
            "        'observation_prediction': 'no-op' if n_changed == 0 else 'diff',\n"
            "        'progress_prediction': 'same_family' if n_changed == 0 else ('new_family' if n_changed > 30 else 'same_family'),\n"
            "        'goal_state_prediction': {\n"
            "            'condition': 'fill in your best guess of the winning grid shape',\n"
            "            'color_signature': {},\n"
            "        },\n"
            "        'latent_state_prediction': {\n"
            "            'tracked_component_bbox': (latent.get('top_components') or [{}])[0].get('bbox', []),\n"
            "            'target_variables_used': ['mover_bbox', 'gate_color_count', 'alignment_phase'],\n"
            "            'matched_target_variables': 0,\n"
            "            'total_target_variables': 3,\n"
            "        },\n"
            "        'goal_progress_prediction': {\n"
            "            'progress_delta': 0,\n"
            "            'justification': 'which visible structure or latent hypothesis moved toward the goal?',\n"
            "        },\n"
            "    }\n"
            "```\n"
            "This scaffold shows how to consult `per_action_delta_hints` correctly; replace the body with your actual hypothesis of how this game's mechanic maps before → after.\n"
            "- EXPLICIT GOAL PREDICATE (plan v4.2 §P34). In the same `world_update` block, also define:\n"
            "```python\n"
            "def is_goal_state(grid):\n"
            "    # Return True iff this grid matches your current goal-state hypothesis.\n"
            "    # Default: sanity check on color_signature from goal_state_prediction.\n"
            "    counts = {}\n"
            "    for row in grid:\n"
            "        for c in row:\n"
            "            counts[int(c)] = counts.get(int(c), 0) + 1\n"
            "    # Replace this with your own terminal-state check (e.g. 'when a target color\n"
            "    # touches a lock color and the lock count drops by 1, the level is advancing').\n"
            "    return False\n"
            "```\n"
            "MCTS imagination marks terminal + awards a +5.0 bonus when `is_goal_state(grid)` returns True — this is how your planner pursues the goal, not just information gain. Update this function whenever a wake-time surprise reveals the goal shape differs from your last guess.\n"
            "- When you notice a reusable pattern, you may propose a skill via `propose_skill`. You choose the shape of that object and the terms you use, "
            "but every payload MUST carry a `category` field that is either 'mechanic' (a game rule: trigger + expected effect + guard) or 'strategy' "
            "(an abstract, game-agnostic exploration/exploitation procedure — BFS-style). Skills outside those two categories are refused by the library. "
            "A mechanic skill is strongest when it cites a confirmed hypothesis id.\n"
            "- Form hypotheses by comparing states: reproduce an effect from a different starting state before trusting it; never trust a single-observation theory.\n"
            "- Colors are semantic, not positional: think 'A must reach B' rather than 'A must reach row 38'. If your hypothesis cites a specific coordinate as the goal, restate it as a relationship.\n"
            "- The grid is a picture, not a spreadsheet: `color_counts()` and `bounding_box()` are lossy summaries; regularly `frame_render` the grid (or a crop) before concluding anything.\n"
            "- After each action, a last-action effect line shows how many cells changed. Unexpected deltas are more important than raw diff magnitude — inspect any region that changed outside where you acted.\n"
            "- Do not retry the same action-sequence 3+ times without re-examining the grid. If two variations of an approach fail, report what you tried and switch tactics.\n"
            "- Efficient play: do not take exploratory actions you have already taken; prefer targeted experiments over exhaustive sweeps.\n"
        )

    def _information_gain_frontier_overlay(self, latest_frame: FrameData) -> str:
        """Plan v4.2 §R7.1/§R9.1 — information-gain frontier block.

        Top-5 active hypotheses by expected entropy gain × trigger prior,
        capped at 10 lines. Silent if no store or nothing eligible.
        """
        if not self._registry or self._registry.bridge.hypothesis_store is None:
            return ""
        try:
            available = [GameAction.from_id(a).name for a in latest_frame.available_actions]
        except Exception:
            available = []
        frontier = self._registry.bridge.hypothesis_store.information_gain_frontier(
            available_actions=available, top_k=5
        )
        if not frontier:
            return ""
        lines = ["[Information-gain frontier]"]
        for row in frontier:
            lines.append(
                f"- {row['h_id']}: gain={row['expected_gain_bits']:.2f}b "
                f"(ent={row['entropy_bits']:.2f}, prior={row['trigger_prior']}) "
                f"{row['why']} :: {row['claim_head'][:70]}"
            )
        return "\n".join(lines)

    def _tool_violation_overlay(self) -> str:
        """P18 rev C: nag the agent if the previous N turns produced
        diff>0 but no frame_* helper call. Rate-limited to once per 3
        turns so the overlay is not spammy.
        """
        if getattr(self, "action_counter", 0) < 2:
            return ""
        last = getattr(self, "_last_tool_violation_turn", -10)
        if self.action_counter - last < 3:
            return ""
        try:
            path = self.research_workdir / "tool_usage.jsonl"
            if not path.exists():
                return ""
            lines = path.read_text().strip().splitlines()[-40:]
            frame_helper_seen = any(
                '"bucket": "frame_helper"' in line for line in lines
            )
            if frame_helper_seen:
                return ""
            self._last_tool_violation_turn = self.action_counter
            return (
                "[TOOL USAGE VIOLATION] You have not called any frame_* "
                "helper in the last ~20 turns. Before `submit_action` "
                "this turn, call `frame_diff` or `frame_bounding_box` "
                "to ground your next hypothesis in coordinates."
            )
        except Exception:
            return ""

    def _last_rejection_overlay(self) -> str:
        """§R5.2: surface the last ≤ 4 rejection reasons from the bridge."""
        if not self._registry:
            return ""
        log = self._registry.bridge.shared_hints.get("last_rejection_reasons", [])
        if not log:
            return ""
        lines = ["[Last-turn rejections]"]
        for entry in log[-4:]:
            err = entry.get("error", {})
            lines.append(
                f"- turn={entry.get('turn','?')} kind={entry.get('kind','?')} "
                f"reason={err.get('reason','?')}: {err.get('detail','')[:80]}"
            )
        return "\n".join(lines)

    def _prompt_overlay(self, latest_frame: FrameData | None = None) -> str:
        overlay_parts = []
        if latest_frame is not None:
            frontier = self._information_gain_frontier_overlay(latest_frame)
            if frontier:
                overlay_parts.append(frontier)
        rej = self._last_rejection_overlay()
        if rej:
            overlay_parts.append(rej)
        tv = self._tool_violation_overlay()
        if tv:
            overlay_parts.append(tv)
        if self._registry:
            # Exploration-phase notice: tell the LLM that the planner is
            # OFF and real-env signature novelty is being rewarded directly.
            exp_reason = self._registry.bridge.read_hint("exploration_reason", "")
            if exp_reason:
                overlay_parts.append(
                    "[Exploration Phase]\n"
                    f"- {exp_reason}. World coder is not yet reliable — prefer actions that\n"
                    "  reach an observation family you have not seen. Explicit bonus: MCTS\n"
                    "  (when enabled later) will reward simulator paths whose leaf\n"
                    "  `next_signature_hint` suggests an unseen family. While in\n"
                    "  exploration phase, the registry auto-picks the least-tried action\n"
                    "  for this state; you may still emit `predict`, `world_update`, and\n"
                    "  `propose_skill` to accelerate the world coder."
                )
            # Wake overlay (if surprise queued) comes first — it is the
            # most time-sensitive instruction to the LLM.
            if hasattr(self._registry, "wake_overlay"):
                wake = self._registry.wake_overlay()
                if wake:
                    overlay_parts.append(wake)
            registry_overlay = self._registry.prompt_overlay()
            if registry_overlay:
                overlay_parts.append(registry_overlay)
        summaries = self._memories.summaries()
        if summaries:
            # Category-balanced memory selection (fixes v3-full where
            # `summaries[-6:]` showed only the most recent surprise/score
            # events and missed consolidated observed laws). Priority:
            # Observed Law first (aggregate truth > individual anecdotes),
            # then recent Draft evolution, then a few Surprise/Score/Skill.
            def _filter(prefix: str, n: int) -> list[str]:
                # summaries() prepends "[<idx>] " so we substring-match.
                matches = [s for s in summaries if prefix in s]
                return matches[-n:]
            balanced: list[str] = []
            balanced.extend(_filter("[Observed Law]", 4))
            balanced.extend(_filter("[World Draft]", 2))
            balanced.extend(_filter("[World Surprise]", 2))
            balanced.extend(_filter("[World Score]", 1))
            balanced.extend(_filter("[Skill]", 2))
            # Deduplicate preserving order.
            seen: set[str] = set()
            picked: list[str] = []
            for s in balanced:
                if s in seen:
                    continue
                seen.add(s)
                picked.append(s)
            # Fallback: if balanced selection is empty (early game), use
            # the last 6 as before.
            if not picked:
                picked = summaries[-6:]
            memory_lines = ["[Persistent Notes]"]
            memory_lines.extend(f"- {line}" for line in picked)
            overlay_parts.append("\n".join(memory_lines))

            # v3.12: inject the FULL details of observed-law memories
            # so predict_effect can actually cite the per-action mean_diff
            # instead of guessing `expected_diff_cells=1`.
            law_details: list[str] = []
            for idx_m, mem in enumerate(self._memories.stack):
                if "[Observed Law]" in mem.summary:
                    law_details.append(f"- {mem.summary}: {mem.details[:240]}")
            if law_details:
                overlay_parts.append(
                    "[Observed-Law Details — use these numbers in predict_effect]\n"
                    + "\n".join(law_details[-6:])
                    + "\n→ If mean_diff for ACTIONk is ~42, your `expected_diff_cells`"
                    " must be ~42 (not 1) AND your `expected_next_grid` must differ"
                    " from the input in ~42 cells."
                )
        return "\n\n".join(overlay_parts)

    def _to_rich_frame(self, frame: FrameData) -> Frame:
        prev_levels_completed = None
        if self.action_history:
            prev_levels_completed = self.action_history[-1][1].levels_completed
        elif len(self.frames) >= 2:
            prev_levels_completed = self.frames[-2].levels_completed
        # Upstream Frame() indexes `data.frame[-1]`; if the engine returns
        # an empty grid stack (observed around level transitions), fall
        # back to the most recent non-empty frame in history instead of
        # crashing.
        if not getattr(frame, "frame", None):
            for cached in reversed(self.frames[:-1]):
                if getattr(cached, "frame", None):
                    frame = cached
                    break
            else:
                raise ValueError("no non-empty frame available for Frame()")
        return Frame(frame, prev_levels_completed=prev_levels_completed)

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "history",
                "description": (
                    "Return the last n observed (action, frame) entries, oldest first. "
                    "Optionally include raw frame renders and per-step change summaries so "
                    "you can inspect prior observations without relying on host-authored summaries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 50},
                        "wins_only": {"type": "boolean"},
                        "include_render": {"type": "boolean"},
                        "include_change_summary": {"type": "boolean"},
                        "y_ticks": {"type": "boolean"},
                        "x_ticks": {"type": "boolean"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_summaries",
                "description": "List short summaries of stored shared memories.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_get",
                "description": "Retrieve a stored shared memory by index.",
                "parameters": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_add",
                "description": "Store a shared memory summary and details for later turns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["summary", "details"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_query",
                "description": "Natural-language retrieval over shared memories. Mirrors memories.query(...).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "return_type": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_render",
                "description": (
                    "Render a frame as text (Symbolica Frame.render signature). "
                    "Look at the grid as a picture; this is the single most reliable "
                    "way to see shapes, walls, player, goals. Use `crop=(x1,y1,x2,y2)` "
                    "to zoom in on a region. Use `y_ticks=True, x_ticks=True` for axes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "keys": {"type": "string", "description": "16-char map int→glyph (default '0123456789abcdef')"},
                        "gap": {"type": "string", "description": "separator between cells (default ' ')"},
                        "crop": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "(x1, y1, x2, y2) exclusive end",
                        },
                        "y_ticks": {"type": "boolean", "description": "prefix each row with y=NN"},
                        "x_ticks": {"type": "boolean", "description": "header with x column numbers"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_diff",
                "description": (
                    "Diff a chosen frame against the previous or a chosen history frame. "
                    "Returns list[DiffRegion]; use each region's (x0,y0,x1,y1) to crop "
                    "frame_render_diff for zoom-in. `margin` merges changes within N "
                    "pixels into a single region (default 2, Symbolica default)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "margin": {"type": "integer", "minimum": 0, "description": "cluster radius (default 2)"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_change_summary",
                "description": "One-line region summaries of changes between a chosen frame and a previous frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "margin": {"type": "integer", "minimum": 0, "description": "cluster radius (default 2)"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_render_diff",
                "description": (
                    "Visual diff between a chosen frame and a previous or chosen history frame. "
                    "Changed cells show new value; unchanged cells show '.'. Use "
                    "`crop='auto'` to zoom on tight bbox of changes, `crop=[x1,y1,x2,y2]` "
                    "for explicit region, `crop=None` (omit) for full grid."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "keys": {"type": "string"},
                        "gap": {"type": "string"},
                        "crop": {
                            "oneOf": [
                                {"type": "string", "enum": ["auto"]},
                                {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            ]
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_find",
                "description": "Find matching cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                        "history_index": {"type": "integer"},
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_bounding_box",
                "description": "Bounding box of matching cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                        "history_index": {"type": "integer"},
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_color_counts",
                "description": "Count cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "propose_hypothesis",
                "description": (
                    "Register a testable hypothesis about the current game with the "
                    "research bridge. Each hypothesis MUST include a machine-checkable "
                    "verification method (trigger + check_code). Returns the hypothesis "
                    "id when accepted, or a structured error when the grammar/whitelist "
                    "check fails. Use this instead of free-text speculation so the "
                    "bridge can auto-score confirm/disconfirm events."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "trigger": {
                            "type": "string",
                            "description": (
                                "Trigger DSL. One of: 'every_step', 'after ACTION1..ACTION5', "
                                "'after RESET', 'on_change frame_color_counts', "
                                "'on_change frame_bounding_box', 'on_change frame_render', "
                                "'on_change frame_find'."
                            ),
                        },
                        "check_code": {
                            "type": "string",
                            "description": (
                                "Python snippet; set `result` to True (confirm), False "
                                "(disconfirm), or None (ambiguous). Available names: "
                                "before, after, action, env_state, plus any helpers "
                                "you declare in helpers_used. Whitelisted builtins only; "
                                "no __dunder__ access."
                            ),
                        },
                        "helpers_used": {
                            "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "frame_diff",
                                        "frame_color_counts",
                                        "frame_bounding_box",
                                        "frame_render",
                                        "frame_find",
                                    ],
                            },
                        },
                        "min_preconditions": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": ["mechanic", "strategy"],
                        },
                        "is_goal": {"type": "boolean"},
                        "budget_claim": {
                            "type": "string",
                            "description": (
                                "P14: optional terse claim about turn budget, "
                                "e.g. 'WIN reachable within <= 24 turns'. If "
                                "present, the hypothesis is auto-tagged budget-shaped."
                            ),
                        },
                        "goal_state_prediction": {
                            "type": "object",
                            "description": (
                                "P13: optional goal-shape prediction. Min shape "
                                "{'condition': str, 'color_signature': dict[str, int]}."
                            ),
                            "properties": {
                                "condition": {"type": "string"},
                                "color_signature": {"type": "object"},
                            },
                            "additionalProperties": True,
                        },
                    },
                    "required": ["claim", "trigger", "check_code", "category"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "submit_action",
                "description": "Commit this turn's real game action and optional side-channel outputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_name": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "predict": {},
                        "env_note": {"type": "string"},
                        "propose_skill": {"type": "object"},
                        "world_update": {},
                    },
                    "required": ["action_name"],
                    "additionalProperties": False,
                },
            },
        ]

    def _record_tool_invocation(self, name: str, args: dict[str, Any]) -> None:
        """Plan v4.2 §P4 / §R1.7: append per-tool use to tool_usage.jsonl."""
        # Rev M: feed every tool invocation through the gate tracker so
        # the phase gates (P45) can enforce tool-call preconditions.
        if getattr(self, "_rev_m_initialized", False):
            try:
                self._tool_tracker.record_call(name, args)
            except Exception:
                pass
        try:
            if name.startswith("frame_"):
                bucket = "frame_helper"
            elif name.startswith("memories_"):
                bucket = "memory"
            elif name == "history":
                bucket = "history"
            elif name == "submit_action":
                bucket = "submit"
            elif name == "propose_hypothesis":
                bucket = "propose_hypothesis"
            else:
                bucket = "other"
            path = self.research_workdir / "tool_usage.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            line = {
                "turn": self.action_counter,
                "tool": name,
                "bucket": bucket,
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, default=str) + "\n")
        except Exception:
            # Non-fatal — observability should never break gameplay.
            pass

    def _execute_tool(self, name: str, raw_arguments: str, current_frame: Frame) -> str:
        try:
            args = json.loads(raw_arguments or "{}")
        except Exception:
            args = {}
        # P4: tool-usage audit trail. Categorize each invocation into
        # helper vs submit vs propose so the registry can surface a
        # per-turn count (excluding submit_action / propose_* from the
        # numerator). Writes to <shared_dir>/tool_usage.jsonl.
        self._record_tool_invocation(name, args)

        if name == "history":
            n = int(args.get("n", 10))
            wins_only = bool(args.get("wins_only", False))
            include_render = bool(args.get("include_render", False))
            include_change_summary = bool(args.get("include_change_summary", False))
            y_ticks = bool(args.get("y_ticks", True))
            x_ticks = bool(args.get("x_ticks", False))
            observed = self._observed_frames(current_frame)
            labels: list[str] = []
            if len(observed) == len(getattr(self, "action_history", []) or []) + 1:
                labels = ["<bootstrap>"] + [
                    str(action)
                    for action, _frame in list(getattr(self, "action_history", []) or [])
                ]
            else:
                labels = [f"<obs:{idx}>" for idx in range(len(observed))]
            entries = list(zip(labels, observed))
            if wins_only:
                entries = [
                    (action, frame)
                    for action, frame in entries
                    if frame.winning_frame is not None
                ]
            out = []
            subset = list(entries[-n:])
            for local_idx, (action, frame) in enumerate(subset):
                record = {
                    "action": action,
                    "has_winning_frame": frame.winning_frame is not None,
                    "state": frame.state.name,
                    "levels_completed": frame.levels_completed,
                    "available_actions": frame.available_actions,
                }
                if include_render:
                    try:
                        record["render"] = frame.render(
                            y_ticks=y_ticks,
                            x_ticks=x_ticks,
                        )
                    except Exception:
                        record["render"] = "<render_failed>"
                if include_change_summary:
                    global_idx = len(entries) - len(subset) + local_idx
                    if global_idx > 0:
                        try:
                            prev_frame = entries[global_idx - 1][1]
                            record["change_summary"] = frame.change_summary(
                                prev_frame, margin=2
                            )
                        except Exception:
                            record["change_summary"] = "<change_summary_failed>"
                    else:
                        record["change_summary"] = None
                out.append(record)
            return json.dumps(out)

        if name == "memories_summaries":
            return json.dumps(self._memories.summaries())
        if name == "memories_get":
            idx = int(args["index"])
            memory = self._memories.get(idx)
            return json.dumps(
                {
                    "summary": memory.summary,
                    "details": memory.details,
                    "timestamp": memory.timestamp.isoformat(),
                }
            )
        if name == "memories_add":
            self._memories.add(str(args["summary"]), str(args["details"]))
            return json.dumps({"ok": True, "count": len(self._memories.stack)})
        if name == "memories_query":
            result = self._memories.query(
                str(args.get("question", "")),
                return_type=str(args.get("return_type", "str")),
                limit=int(args.get("limit", 3)),
            )
            return json.dumps(result)

        frame = self._resolve_frame(current_frame, args)
        if name == "frame_render":
            crop = args.get("crop")
            crop_tuple = tuple(int(x) for x in crop) if crop is not None else None
            # Match Symbolica signature: default y_ticks=False, x_ticks=False,
            # gap=" ", keys="0123456789abcdef". The agent can override.
            return frame.render(
                keys=str(args.get("keys", "0123456789abcdef")),
                gap=str(args.get("gap", " ")),
                y_ticks=bool(args.get("y_ticks", False)),
                x_ticks=bool(args.get("x_ticks", False)),
                crop=crop_tuple,
            )
        if name == "frame_diff":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            margin = int(args.get("margin", 2))
            return repr(frame.diff(ref, margin=margin))
        if name == "frame_change_summary":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            margin = int(args.get("margin", 2))
            return frame.change_summary(ref, margin=margin)
        if name == "frame_render_diff":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            crop = args.get("crop")
            if isinstance(crop, list):
                crop = tuple(int(x) for x in crop)
            return frame.render_diff(
                ref,
                keys=str(args.get("keys", "0123456789abcdef")),
                gap=str(args.get("gap", " ")),
                crop=crop,
            )
        if name == "frame_find":
            values = [int(v) for v in args.get("values", [])]
            return json.dumps(frame.find(*values))
        if name == "frame_bounding_box":
            values = [int(v) for v in args.get("values", [])]
            return json.dumps(frame.bounding_box(*values))
        if name == "frame_color_counts":
            return json.dumps(frame.color_counts())

        if name == "propose_hypothesis":
            if self._registry is None:
                return json.dumps({"ok": False, "error": {"reason": "no_registry"}})
            # P13/P14: fold goal_state_prediction + budget_claim into the
            # claim text so the MH semantic-coverage regex picks them up
            # without needing a store-schema change.
            claim = str(args.get("claim", ""))
            bc = args.get("budget_claim")
            if bc:
                claim = f"[budget] {str(bc)[:120]} | {claim}".strip()
            gsp = args.get("goal_state_prediction")
            if isinstance(gsp, dict) and gsp:
                cond = str(gsp.get("condition", ""))[:80]
                claim = f"[goal] condition={cond!r} | {claim}".strip()
            payload = {
                "claim": claim,
                "trigger": str(args.get("trigger", "")),
                "check_code": str(args.get("check_code", "")),
                "helpers_used": list(args.get("helpers_used", [])),
                "min_preconditions": str(args.get("min_preconditions", "True") or "True"),
                "category": str(args.get("category", "mechanic")),
                "is_goal": bool(args.get("is_goal", False)) or bool(gsp),
            }
            h_id, err = self._registry.record_agent_hypothesis(payload)
            if err is not None:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "h_id": h_id})

        return json.dumps({"error": f"unknown tool {name}"})

    def _resolve_frame(self, current_frame: Frame, args: dict[str, Any]) -> Frame:
        source = str(args.get("source", "current"))
        history_index = args.get("history_index")
        if history_index is not None:
            base = self._resolve_history_frame(current_frame, history_index)
            if source == "winning" and base.winning_frame is not None:
                return base.winning_frame
            return base
        if source == "winning" and current_frame.winning_frame is not None:
            return current_frame.winning_frame
        return current_frame

    def _observed_frames(self, current_frame: Frame | None = None) -> list[Frame]:
        """Return the observed frame sequence as rich Frame objects.

        `action_history` stores the post-action frame for each committed real
        action, so the latest entry is often the same as the current frame.
        For history/diff comparisons we need the actual observed sequence from
        `self.frames`, which includes the previous observation as `frames[-2]`.
        """
        out: list[Frame] = []
        seen_ids: set[int] = set()
        for raw in list(getattr(self, "frames", []) or []):
            try:
                rich = raw if hasattr(raw, "render") and hasattr(raw, "diff") else self._to_rich_frame(raw)
            except Exception:
                continue
            out.append(rich)
            seen_ids.add(id(raw))
            seen_ids.add(id(rich))
            rich_data = getattr(rich, "_data", None)
            if rich_data is not None:
                seen_ids.add(id(rich_data))
        if current_frame is not None:
            current_data = getattr(current_frame, "_data", None)
            current_seen = id(current_frame) in seen_ids or (
                current_data is not None and id(current_data) in seen_ids
            )
            if not current_seen:
                out.append(current_frame)
        return out

    def _resolve_history_frame(self, current_frame: Frame, history_index: Any) -> Frame:
        if history_index is None:
            return current_frame
        idx = int(history_index)
        observed = self._observed_frames(current_frame)
        if not observed:
            return current_frame
        return observed[idx]

    def _resolve_diff_reference(self, frame: Frame, history_index: Any) -> Frame:
        if history_index is not None:
            return self._resolve_history_frame(frame, history_index)
        observed = self._observed_frames(frame)
        if len(observed) >= 2:
            return observed[-2]
        if observed:
            return observed[-1]
        return frame

    def _inspection_tool_definitions(self) -> list[dict[str, Any]]:
        """Subset of tools exposed during wake self-inspection."""
        allowed = {
            "history",
            "memories_query",
            "frame_render",
            "frame_diff",
            "frame_change_summary",
            "frame_render_diff",
            "frame_find",
            "frame_bounding_box",
            "frame_color_counts",
        }
        return [
            tool
            for tool in self._tool_definitions()
            if str(tool.get("name")) in allowed
        ]

    def _build_wake_inspection_prefetch(
        self,
        current_frame: Frame,
        *,
        prior_observations: int,
    ) -> tuple[str, dict[str, bool]]:
        """Prefetch raw observation evidence for wake self-inspection.

        The payload is intentionally raw: direct render / diff / count outputs
        that mirror what the helper tools would have returned, without the host
        adding semantic interpretation. This gives the model reliable evidence
        even when it under-uses tool calls in the inspection subpass.
        """
        blocks: list[str] = []
        coverage = {
            "has_scene": False,
            "has_diff": False,
        }
        try:
            current_render = current_frame.render(y_ticks=True, x_ticks=False)
            blocks.append("[current.frame_render]\n" + current_render)
            coverage["has_scene"] = True
        except Exception:
            logger.exception("wake inspection prefetch: current render failed")
        try:
            counts = json.dumps(current_frame.color_counts(), sort_keys=True)
            blocks.append("[current.frame_color_counts]\n" + counts)
            coverage["has_scene"] = True
        except Exception:
            logger.exception("wake inspection prefetch: color counts failed")
        if prior_observations >= 1 and self.action_history:
            previous_frame = self.action_history[-1][1]
            try:
                prev_render = previous_frame.render(y_ticks=True, x_ticks=False)
                blocks.append("[previous.frame_render]\n" + prev_render)
                coverage["has_scene"] = True
            except Exception:
                logger.exception(
                    "wake inspection prefetch: previous render failed"
                )
            try:
                diff_summary = current_frame.change_summary(previous_frame, margin=2)
                blocks.append("[current_vs_previous.frame_change_summary]\n" + diff_summary)
                coverage["has_diff"] = True
            except Exception:
                logger.exception(
                    "wake inspection prefetch: change_summary failed"
                )
            try:
                render_diff = current_frame.render_diff(previous_frame, gap=" ")
                blocks.append("[current_vs_previous.frame_render_diff]\n" + render_diff)
                coverage["has_diff"] = True
            except Exception:
                logger.exception(
                    "wake inspection prefetch: render_diff failed"
                )
        return "\n\n".join(blocks), coverage

    def _run_wake_self_inspection(
        self,
        latest_frame: FrameData,
    ) -> tuple[str | None, dict[str, Any]]:
        """Model-authored raw inspection before Wake planning.

        The model must inspect current/previous observations using helper tools
        and then return a compact JSON summary. This keeps raw-first mode
        honest: the host no longer pre-digests the scene into semantic bullets.
        """
        telemetry = {
            "enabled": bool(_wake_self_inspection_enabled()),
            "tool_calls": 0,
            "used_render": False,
            "used_diff": False,
            "tool_names": [],
        }
        if not _wake_self_inspection_enabled():
            return None, telemetry
        prior_observations = len(getattr(self, "action_history", []) or [])
        try:
            rich_frame = self._to_rich_frame(latest_frame)
        except Exception:
            logger.exception("wake self inspection: _to_rich_frame failed")
            telemetry["error"] = "frame"
            return None, telemetry
        prefetch_text, prefetch_coverage = self._build_wake_inspection_prefetch(
            rich_frame,
            prior_observations=prior_observations,
        )
        prefetch_ok = bool(prefetch_coverage.get("has_scene")) and (
            prior_observations < 1 or bool(prefetch_coverage.get("has_diff"))
        )

        def _extract_tool_names() -> list[str]:
            calls = list(getattr(self._tool_tracker, "_calls", []) or [])
            names: list[str] = []
            for call in calls:
                if isinstance(call, tuple) and call:
                    names.append(str(call[0]))
                    continue
                names.append(str(getattr(call, "name", "")))
            return names

        def _inspection_compliance(names: list[str]) -> tuple[bool, dict[str, bool]]:
            comparison = {
                "history",
                "frame_diff",
                "frame_render_diff",
                "frame_change_summary",
            }
            scene = {
                "frame_render",
                "frame_find",
                "frame_bounding_box",
                "frame_color_counts",
            }
            has_diff = any(name in comparison for name in names)
            has_scene = any(name in scene for name in names)
            if prior_observations >= 1:
                ok = len(names) >= 2 and has_diff and has_scene
            else:
                ok = len(names) >= 2 and has_scene
            return ok, {
                "has_diff": has_diff,
                "has_scene": has_scene,
            }

        def _run_attempt(*, strict_retry: bool) -> tuple[str, list[str], dict[str, bool]]:
            self._tool_tracker.reset()
            retry_clause = ""
            if strict_retry:
                retry_clause = (
                    " Your previous inspection under-used tools. "
                    "This retry is INVALID unless you satisfy the tool protocol exactly."
                )
            response = _create_trapi_response(
                model="gpt-5.3-codex",
                instructions=(
                    "You are in WAKE SELF-INSPECTION. Do not choose an action. "
                    "Use helper tools to inspect raw current/previous observations, then return JSON only. "
                    "Use at least two helper tools. "
                    "If there is at least one prior observation in history, you MUST use "
                    "at least one comparison tool (`history`, `frame_diff`, `frame_render_diff`, or `frame_change_summary`) "
                    "and one scene-inspection tool (`frame_render`, `frame_find`, `frame_bounding_box`, or `frame_color_counts`). "
                    "If there are zero prior observations, use at least two scene-inspection tools before answering. "
                    "The current frame is already in scope; `history()` gives prior committed observations only. "
                    "Raw evidence from deterministic helper prefetch is included below; it is verbatim helper output, not a semantic summary. "
                    "Base your synthesis on that evidence first, then call extra tools only if you need more detail. "
                    "Prefer strong reusable mechanism hypotheses over local deltas."
                    + retry_clause
                ),
                input_items=[{
                    "type": "message",
                    "role": "user",
                    "content": (
                        "Inspect the current and previous observed states. "
                        f"There are {prior_observations} prior committed observations in history. "
                        "If prior_observations >= 1, prefer either "
                        "`frame_render_diff()` / `frame_change_summary()` or "
                        "`history(n=2, include_render=true, include_change_summary=true)` "
                        "before concluding whether a transition exists. "
                        "Explicitly compare the current observation against the immediately previous one before naming a mechanism. "
                        "Return ONE JSON object with keys "
                        '{"summary": str, "scene_inventory": [str...], '
                        '"mechanism_hypotheses": [str...], "next_checks": [str...]}.\n\n'
                        "RAW EVIDENCE (VERBATIM HELPER OUTPUT)\n"
                        f"{prefetch_text}"
                    ),
                }],
                tools=self._inspection_tool_definitions(),
                max_output_tokens=1200,
            )

            raw_text_local = ""
            try:
                while True:
                    tool_outputs: list[dict[str, Any]] = []
                    function_calls = [
                        item
                        for item in response.output
                        if getattr(item, "type", None) == "function_call"
                    ]
                    if not function_calls:
                        raw_text_local = getattr(response, "output_text", "") or ""
                        break
                    for call in function_calls:
                        output = self._execute_tool(call.name, call.arguments, rich_frame)
                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": output,
                            }
                        )
                    response = _create_trapi_response(
                        model="gpt-5.3-codex",
                        previous_response_id=response.id,
                        input_items=tool_outputs,
                        tools=self._inspection_tool_definitions(),
                        max_output_tokens=1200,
                    )
            except Exception:
                logger.exception("wake self inspection loop failed")
            names_local = _extract_tool_names()
            ok_local, flags_local = _inspection_compliance(names_local)
            return raw_text_local, names_local, {
                "ok": ok_local,
                **flags_local,
            }

        raw_text, names, compliance = _run_attempt(strict_retry=False)
        retried = False
        if not compliance.get("ok", False) and not prefetch_ok:
            retried = True
            raw_text_retry, names_retry, compliance_retry = _run_attempt(
                strict_retry=True
            )
            if compliance_retry.get("ok", False):
                raw_text = raw_text_retry
                names = names_retry
                compliance = compliance_retry

        telemetry.update(
            {
                "tool_calls": len(names),
                "used_render": any(
                    name in {"frame_render", "frame_render_diff"} for name in names
                ),
                "used_diff": any(
                    name in {
                        "history",
                        "frame_diff",
                        "frame_render_diff",
                        "frame_change_summary",
                    }
                    for name in names
                ),
                "tool_names": names[:12],
                "compliant": bool(compliance.get("ok", False) or prefetch_ok),
                "model_tool_compliant": bool(compliance.get("ok", False)),
                "retried": retried,
                "prefetched": bool(prefetch_text),
                "prefetch_used_scene": bool(prefetch_coverage.get("has_scene")),
                "prefetch_used_diff": bool(prefetch_coverage.get("has_diff")),
            }
        )
        if not raw_text.strip():
            return None, telemetry
        try:
            payload = json.loads(raw_text)
        except Exception:
            return raw_text[:1200], telemetry
        if not isinstance(payload, dict):
            return raw_text[:1200], telemetry
        parts: list[str] = []
        summary = str(payload.get("summary", "") or "").strip()
        if summary:
            parts.append(summary)
        for key in ("scene_inventory", "mechanism_hypotheses", "next_checks"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                parts.append(f"{key}: " + "; ".join(str(x) for x in value[:4]))
        return ("\n".join(parts) if parts else raw_text[:1200]), telemetry

    def _ensure_bootstrap_frame(self) -> None:
        latest = self.frames[-1]
        if latest.state is not GameState.NOT_PLAYED:
            return

        initial = self.take_action(GameAction.RESET)
        if initial:
            self.append_frame(initial)

    # ------------------------------------------------------------------
    # Rev M phase handlers (plan v4.2 §P38-P48). Only invoked when
    # REV_M_ENABLED=1; the legacy main() body is preserved below so
    # test suites / smoke runs that flip the flag off keep working.
    # ------------------------------------------------------------------

    def _interval_probe(
        self,
        wake_plan: WakePlan | None,
        available_names: list[str],
    ) -> WakePlan | None:
        """Rev S P70: periodic forced-exploration override (interval path).

        Every ``FORCED_PROBE_INTERVAL`` real actions, override the chosen
        candidate's first action with the least-used action from the last
        30 entries of ``self._action_history``.

        Extracted into its own helper so Rev T can compose it with the new
        imbalance probe (see ``_apply_forced_probe``).
        """
        if wake_plan is None:
            return None
        interval = int(FORCED_PROBE_INTERVAL)
        if interval <= 0:
            return None
        history = list(getattr(self, "_action_history", []) or [])
        if len(history) < interval:
            return None
        if len(history) % interval != 0:
            return None
        if not available_names:
            return None
        chosen = getattr(wake_plan, "chosen", None)
        if chosen is None:
            return None
        sequence = list(getattr(chosen, "sequence", []) or [])
        recent = history[-30:]
        counts = {name: recent.count(name) for name in available_names}
        if not counts:
            return None
        target = min(counts, key=lambda n: (counts[n], n))
        current_first = sequence[0] if sequence else None
        if target == current_first:
            return None
        new_seq = [target] + sequence[1:]
        new_candidate = replace(
            chosen,
            sequence=new_seq,
            rationale=f"forced_probe:{target}",
        )
        new_plan = replace(wake_plan, chosen=new_candidate)
        logger.info(
            "Rev S forced probe override: %s -> %s (turn=%d, counts=%s)",
            current_first,
            target,
            int(getattr(self, "_wake_turn_index", 0) or 0),
            counts,
        )
        return new_plan

    def _apply_forced_probe(
        self,
        wake_plan: WakePlan | None,
        available_names: list[str],
    ) -> WakePlan | None:
        """Rev S P70 + Rev T P73/P74/P75: forced-exploration override.

        Two override paths run inside this method:

        * Interval probe (Rev S): every ``FORCED_PROBE_INTERVAL`` real
          actions the first step of the Wake plan is replaced with the
          least-used action from recent history.
        * Imbalance probe (Rev T): once the history reaches length 10 the
          last 20 actions are inspected. If the most-used available action
          appears ``>= 2 * max(1, min_count)`` times the probe fires
          regardless of turn index. Every occurrence of the over-used
          action inside ``wake_plan.chosen.sequence`` gets rewritten to the
          under-used target. When the imbalance becomes severe
          (``max / min > 3``) the sequence is truncated to length 1 so the
          planner replans on the next turn instead of chaining more
          over-used-action executions.

        Returns a modified ``WakePlan`` when an override applies, or
        ``None`` to keep the original plan untouched.
        """
        if wake_plan is None:
            return None
        if not available_names:
            return None
        chosen = getattr(wake_plan, "chosen", None)
        if chosen is None:
            return None
        rationale = str(getattr(chosen, "rationale", "") or "")
        if "observation_route_to_landmark" in rationale:
            return None
        if len(self._informative_action_deltas(available_names)) >= 2:
            return None
        sequence = list(getattr(chosen, "sequence", []) or [])
        history = list(getattr(self, "_action_history", []) or [])

        # Before history length 10 the imbalance counts are too noisy to
        # drive a meaningful override, so keep the Rev S interval path.
        if len(history) < 10:
            return self._interval_probe(wake_plan, available_names)

        # Rev T P73: imbalance-triggered probe. Uses a 20-step window so
        # the signal tracks recent behaviour without being drowned by
        # early-run exploration.
        recent = history[-20:]
        counts = {name: recent.count(name) for name in available_names}
        if counts:
            max_c = max(counts.values())
            min_c = min(counts.values())
            if max_c >= 2 * max(1, min_c):
                target = min(counts, key=lambda n: (counts[n], n))
                over_used = max(counts, key=lambda n: (counts[n], n))
                if target != over_used:
                    new_seq: list[str] = []
                    for i, action in enumerate(sequence):
                        if i == 0:
                            new_seq.append(target)
                        elif action == over_used:
                            new_seq.append(target)
                        else:
                            new_seq.append(action)
                    # Seed the probe target when the planner produced an
                    # empty sequence so the probe still has a chance to
                    # fire the under-used action.
                    if not new_seq:
                        new_seq = [target]
                    # Rev T P75: severe imbalance -> truncate to a single
                    # step so the planner re-decides next turn instead of
                    # continuing chained over-used-action executions.
                    severe = (max_c / max(1, min_c)) > 3
                    if severe:
                        new_seq = new_seq[:1]
                    if new_seq != sequence:
                        rationale = (
                            f"rev_t_imbalance_probe:{target}"
                            f"_over={over_used}:max={max_c}:min={min_c}"
                        )
                        new_candidate = replace(
                            chosen,
                            sequence=new_seq,
                            rationale=rationale,
                        )
                        logger.info(
                            "Rev T imbalance probe: over=%s target=%s"
                            " counts=%s seq=%s->%s severe=%s",
                            over_used,
                            target,
                            counts,
                            sequence,
                            new_seq,
                            severe,
                        )
                        return replace(wake_plan, chosen=new_candidate)
                    # The incoming plan already matched the probe target
                    # (e.g. target appears only as the first step and no
                    # over-used occurrences elsewhere): nothing to do.
                    return None

        # No imbalance detected -> fall through to the interval probe so
        # chronic skew that never triggers the 2x rule still gets nudged.
        return self._interval_probe(wake_plan, available_names)

    def _legacy_imbalance_override(
        self, chosen: str, available: list[str]
    ) -> str:
        """Rev U: Rev T imbalance override for the legacy (REV_M_ENABLED=0)
        main loop.

        The legacy loop doesn't go through ``_apply_forced_probe`` (that
        lives inside ``_run_wake_phase``), which meant ep43 saw ACTION3
        dominate 55/66 turns. This method ports the minimal Rev T rule
        so the legacy path also gets rebalanced:

        * Inspect the last 20 entries of ``_legacy_action_history``.
        * If ``max_count >= 2 * max(1, min_count)`` across available
          actions, swap the chosen action for the least-used one
          (ties broken alphabetically, matching ``_apply_forced_probe``).
        * Short history (len < 10) or empty available list returns the
          chosen action unchanged.

        Returns the action name to actually commit (possibly the same
        ``chosen``). Callers should wrap invocation in try/except so an
        override failure never blocks the main loop.
        """
        hist = getattr(self, "_legacy_action_history", [])
        if len(hist) < 10:
            return chosen
        if not available:
            return chosen
        recent = hist[-20:]
        counts = {a: recent.count(a) for a in available}
        if not counts:
            return chosen
        max_c = max(counts.values())
        min_c = min(counts.values())
        if max_c < 2 * max(1, min_c):
            return chosen
        # Imbalance detected. Pick least-used action (tie-break by name).
        target = min(counts, key=lambda a: (counts[a], a))
        if target != chosen:
            logger.info(
                "Rev T legacy imbalance override: %s -> %s (counts=%s)",
                chosen, target, counts,
            )
        return target

    def _should_allow_host_wake_overrides(
        self,
        *,
        active_h: list[dict] | None,
        skills: list[dict] | None,
    ) -> bool:
        """Allow host-side Wake overrides only during evidence-starved bootstraps."""
        return len(list(active_h or [])) == 0 and len(list(skills or [])) == 0

    def _compress_wake_commit_sequence(
        self,
        candidate: WakeCandidate | None,
        sequence: list[str],
    ) -> tuple[list[str], str | None]:
        """Choose the shortest strong prefix of the WM-scored Wake sequence."""
        seq = list(sequence or [])
        if candidate is None or len(seq) <= 1:
            return seq, None
        traces = list(getattr(candidate, "rollout_traces", []) or [])
        if not traces:
            return seq, None

        cumulative = 0.0
        best_score: float | None = None
        best_len = len(seq)
        full_score = 0.0
        for idx, trace in enumerate(traces[: len(seq)]):
            reward = float(trace.get("reward", 0.0) or 0.0)
            sim_info = trace.get("sim_info", {}) or {}
            goal_delta = float(sim_info.get("goal_progress_delta", 0.0) or 0.0)
            latent_delta = float(sim_info.get("latent_match_delta", 0.0) or 0.0)
            disconfirm = float(sim_info.get("disconfirm_entropy", 0.0) or 0.0)
            confirm = float(sim_info.get("confirm_entropy", 0.0) or 0.0)
            step_gain = reward + 1.5 * goal_delta + 0.75 * latent_delta + 0.25 * (
                disconfirm - confirm
            )
            cumulative += step_gain
            prefix_score = cumulative - 0.35 * idx
            full_score = prefix_score
            if best_score is None or prefix_score > best_score + 0.05:
                best_score = prefix_score
                best_len = idx + 1

        if best_score is None or best_len >= len(seq):
            return seq, None
        if best_score + 0.15 < full_score:
            return seq, None
        return seq[:best_len], f"wm_prefix_best:{best_len}/{len(seq)}"

    def _run_wake_phase(self) -> None:
        """Run one Wake block: LLM-proposed action sequence + divergence monitor.

        On divergence the Wake block terminates and the phase controller
        transitions to Sleep. On clean completion the controller also
        moves to Sleep so the cycle cadence stays deterministic.
        """
        if not self._rev_m_initialized:
            return
        latest = self.frames[-1]
        available_names = [
            GameAction.from_id(a).name for a in latest.available_actions
        ]
        grid = current_grid(latest)
        self._last_visible_grid = grid

        # Reset the per-turn tool tracker. In the fully-integrated path
        # the TRAPI tool dispatcher feeds into this; here it stays empty
        # so the gate's PASS/FAIL signal is driven by retry logic.
        self._tool_tracker.reset()

        # Reachability-gate the active hypothesis frontier (P47, R19.6).
        store = self._hypothesis_store()
        active_h: list[dict] = []
        if store is not None:
            try:
                frontier = store.information_gain_frontier(
                    available_actions=available_names, top_k=8
                )
            except Exception:
                logger.exception("Rev M wake: frontier lookup failed")
                frontier = []
            items = getattr(store, "_items", {})
            for row in frontier:
                h_id = row.get("h_id")
                h_obj = items.get(h_id) if isinstance(items, dict) else None
                if h_obj is None:
                    continue
                try:
                    reachable = self._reachability_gate.is_reachable(
                        grid, h_obj, available_names, depth_cap=3
                    )
                except Exception:
                    reachable = True
                if reachable:
                    active_h.append(row)

        skills = self._skill_summaries()

        # Rev N P49-hardening Fix 2: compute hypothesis-store status so
        # the Wake prompt can emit the STRICT mandate when the frontier
        # is still starved. Defensive: test fixtures instantiate the
        # agent via ``object.__new__`` and skip ``__init__``, so guard
        # against the counter not existing.
        store_status = self._compute_hypothesis_store_status(store)
        current_turn_idx = int(getattr(self, "_wake_turn_index", 0) or 0) + 1
        self._wake_turn_index = current_turn_idx
        wake_turn_index = current_turn_idx

        # Rev O P54: seed the tracker with the current pre-Wake grid so
        # the summary has a turn-0 baseline even when no action has been
        # executed yet this run. Tests instantiate the agent via
        # ``object.__new__`` and skip ``__init__``; guard accordingly.
        tracker = getattr(self, "_player_tracker", None)
        if tracker is None:
            tracker = PlayerTracker()
            self._player_tracker = tracker
            self._last_diff_coords = []
            self._last_diff_total = 0
        try:
            tracker.update(self.action_counter, grid)
        except Exception:
            logger.exception("Rev O: player_tracker.update (seed) failed")
        try:
            player_summary_text = tracker.summary()
        except Exception:
            logger.exception("Rev O: player_tracker.summary failed")
            player_summary_text = None
        diff_coords_snapshot = list(getattr(self, "_last_diff_coords", []) or [])
        diff_total_snapshot = int(getattr(self, "_last_diff_total", 0) or 0)

        # Rev P P58: compute the initial frame summary once per run so the
        # LLM always sees the starting grid layout, even many turns in.
        if not getattr(self, "_initial_frame_summary", ""):
            try:
                self._initial_frame_summary = render_initial_frame_summary(grid)
            except Exception:
                logger.exception("Rev P: initial_frame_summary render failed")
                self._initial_frame_summary = ""
        initial_frame_summary_text = getattr(
            self, "_initial_frame_summary", ""
        ) or None

        # Rev P P59: render the amplified diff patch from the last real
        # transition's before/after grids (cached in the action loop).
        diff_patch_text = getattr(self, "_last_diff_patch_text", "") or None

        if not _wake_host_summaries_enabled():
            player_summary_text = None
            diff_coords_snapshot = []
            initial_frame_summary_text = None

        inspection_summary_text: str | None = None
        inspection_telemetry: dict[str, Any] = {
            "enabled": bool(_wake_self_inspection_enabled()),
            "tool_calls": 0,
            "used_render": False,
            "used_diff": False,
            "tool_names": [],
        }
        if _wake_self_inspection_enabled():
            try:
                inspection_summary_text, inspection_telemetry = (
                    self._run_wake_self_inspection(latest)
                )
            except Exception:
                logger.exception("wake self inspection failed before prompt build")
                inspection_summary_text = None
                inspection_telemetry = {
                    "enabled": True,
                    "tool_calls": 0,
                    "used_render": False,
                    "used_diff": False,
                    "tool_names": [],
                    "error": "exception",
                }
            self._append_module_io_event(
                "wake_self_inspection",
                {
                    "wake_turn_index": wake_turn_index,
                    "summary_present": bool(inspection_summary_text),
                    "summary_preview": (
                        str(inspection_summary_text)[:400]
                        if inspection_summary_text
                        else ""
                    ),
                    "telemetry": dict(inspection_telemetry),
                },
            )
            # Start the actual wake-planning gate from a clean per-phase
            # tracker while preserving tool_usage.jsonl evidence from the
            # inspection pass.
            self._tool_tracker.reset()

        # Build the gated LLM call. gate_check_fn closes over the
        # shared tracker so retries accumulate tool calls.
        def _build_prompt(missing: list[str] | None = None) -> str:
            extra = ""
            if missing:
                extra = "\n\n# MISSING TOOLS: " + ", ".join(missing)
            try:
                # Rev N P49 / R25.1: feed rejection feedback into the prompt.
                # Rev O P55: inject player-centric observation blocks.
                return self._wake_planner._build_prompt(
                    grid=grid,
                    skills=skills,
                    hypotheses=active_h,
                    wm_source=self._wm_source,
                    last_divergence=self._last_divergence,
                    available_actions=available_names,
                    rejection_feedback=list(
                        getattr(self, "_last_rejection_feedback", []) or []
                    ),
                    hypothesis_store_status=store_status,
                    wake_turn_index=wake_turn_index,
                    player_summary=player_summary_text,
                    diff_coords=diff_coords_snapshot,
                    diff_total=diff_total_snapshot,
                    initial_frame_summary=initial_frame_summary_text,
                    diff_patch_text=diff_patch_text,
                    inspection_summary=inspection_summary_text,
                ) + extra
            except Exception:
                logger.exception("wake prompt build failed; using stub")
                return "propose actions" + extra

        # Rev N P53 live-run fix: the Wake LLM path is text-in JSON-out
        # (no tool-calling plumbing), so the ToolCallTracker-based
        # ``WakeGate`` always FAILs and forces 2 retries per turn. The
        # reference-based gate inspects the LLM response for evidence
        # of the auto-injected helper output (coordinates, color codes)
        # and is safe under the existing path. ``REV_M_HARD_GATE=0``
        # (default) skips the retry loop entirely; the gate signal is
        # still computed for telemetry.
        def _gate_check(last_response_text: str | None = None):
            return check_helper_references(last_response_text, phase="wake")

        # _invoke_llm takes (prompt, available_actions) but gated_llm_call
        # passes a single arg; bind available_names here so both the real
        # and stub paths go through one compatible call shape (W1 fix:
        # surface bug previously masked by the double-invocation).
        def _llm_fn(prompt: str) -> str:
            if self._wake_planner.llm_client is None:
                return self._wake_planner._stub_response(available_names)
            return self._wake_planner._invoke_llm(prompt, available_names)

        # Keep a bounded end-to-end Wake turn, but reserve a small minimum
        # evaluation window so the planner can still score candidates after
        # a slow LLM response instead of collapsing to coverage-only fallback.
        _WAKE_TOTAL_BUDGET_S = _wake_end_to_end_budget_s()
        _WAKE_MIN_EVAL_BUDGET_S = _wake_eval_min_budget_s()
        hard_gate = _rev_m_hard_gate_enabled()
        llm_t0 = time.monotonic()
        if not hard_gate:
            # Telemetry-only path: single LLM call, gate signal logged
            # but never triggers retry. Restores ep26-level throughput.
            try:
                prompt = _build_prompt()
                logger.info(
                    "Wake prompt input: turn=%d actions=%s hypotheses=%d skills=%d "
                    "diff_total=%d player_tracker=%s rejection_feedback=%d",
                    wake_turn_index,
                    available_names,
                    len(active_h),
                    len(skills),
                    diff_total_snapshot,
                    "yes" if player_summary_text else "no",
                    len(list(getattr(self, "_last_rejection_feedback", []) or [])),
                )
                raw = _llm_fn(prompt)
                candidates = self._wake_planner._parse_candidates(
                    raw, available_actions=available_names
                )
            except Exception:
                logger.exception(
                    "wake LLM call failed (telemetry mode); falling back to stub"
                )
                candidates = self._wake_planner._parse_candidates(
                    self._wake_planner._stub_response(available_names),
                    available_actions=available_names,
                )
                raw = ""
            try:
                gate_result = _gate_check(raw)
                self._last_wake_tool_compliance = (
                    gate_result.result == GateResult.PASS
                )
            except Exception:
                logger.exception("wake helper-reference gate check failed")
                self._last_wake_tool_compliance = True
        else:
            try:
                plan_call = gated_llm_call(
                    llm_fn=_llm_fn,
                    build_prompt_fn=_build_prompt,
                    parse_fn=lambda raw: self._wake_planner._parse_candidates(
                        raw, available_actions=available_names
                    ),
                    gate_check_fn=_gate_check,
                    tracker=self._tool_tracker,
                    max_retries=2,
                    total_budget_s=_WAKE_TOTAL_BUDGET_S,
                )
                self._last_wake_tool_compliance = bool(plan_call.tool_compliance)
                if not plan_call.tool_compliance:
                    logger.info(
                        "ESCALATION: phase=wake tool_compliance=False "
                        "retries_exhausted=True retries_used=%d",
                        plan_call.retries_used,
                    )
            except Exception:
                logger.exception("gated wake LLM call failed; falling back to stub")
                parsed = self._wake_planner._parse_candidates(
                    self._wake_planner._stub_response(available_names),
                    available_actions=available_names,
                )
                # Route straight to WakePlanner.plan sequence execution
                # using the stub candidates.
                candidates = parsed
            else:
                candidates = plan_call.output or []
        llm_elapsed = time.monotonic() - llm_t0
        remaining_budget_s = max(
            _WAKE_MIN_EVAL_BUDGET_S,
            _WAKE_TOTAL_BUDGET_S - llm_elapsed,
        )
        self._append_module_io_event(
            "wake_input",
            {
                "wake_turn_index": wake_turn_index,
                "available_actions": list(available_names),
                "hypothesis_count": len(active_h),
                "skill_count": len(skills),
                "diff_total": diff_total_snapshot,
                "player_tracker": bool(player_summary_text),
                "tool_compliance": bool(
                    getattr(self, "_last_wake_tool_compliance", True)
                ),
                "inspection_summary_present": bool(inspection_summary_text),
                "inspection_tool_calls": int(
                    inspection_telemetry.get("tool_calls", 0) or 0
                ),
                "inspection_used_render": bool(
                    inspection_telemetry.get("used_render", False)
                ),
                "inspection_used_diff": bool(
                    inspection_telemetry.get("used_diff", False)
                ),
                "inspection_tool_names": list(
                    inspection_telemetry.get("tool_names", []) or []
                )[:12],
                "inspection_prefetched": bool(
                    inspection_telemetry.get("prefetched", False)
                ),
                "inspection_prefetch_used_scene": bool(
                    inspection_telemetry.get("prefetch_used_scene", False)
                ),
                "inspection_prefetch_used_diff": bool(
                    inspection_telemetry.get("prefetch_used_diff", False)
                ),
                "llm_elapsed_ms": round(llm_elapsed * 1000.0, 1),
                "wake_total_budget_s": float(_WAKE_TOTAL_BUDGET_S),
                "wake_eval_budget_s": float(remaining_budget_s),
                "candidate_count": len(list(candidates or [])),
            },
        )

        # Chosen candidate: reuse the already-parsed candidates from
        # gated_llm_call (W1 fix: avoid a second LLM invocation).
        chosen_sequence: list[str] = []
        proposed_hypotheses: list[dict] = []
        scored_candidates: list[WakeCandidate] = list(candidates or [])
        chosen_candidate: WakeCandidate | None = None
        fallback_reason: str | None = None
        compression_reason: str | None = None
        if candidates:
            try:
                exact_candidate = None
                if self._local_exact_route_enabled():
                    exact_candidate = self._build_local_exact_route_candidate(
                        available_names
                    )
                if exact_candidate is not None:
                    candidates = [exact_candidate, *list(candidates)]
                    logger.info(
                        "Wake injected local exact-route candidate: len=%d seq=%s",
                        len(exact_candidate.sequence),
                        exact_candidate.sequence,
                    )
                coverage_candidate = self._build_coverage_probe_candidate(available_names)
                if coverage_candidate is not None:
                    candidates = [coverage_candidate, *list(candidates)]
                    logger.info(
                        "Wake injected coverage-probe candidate: %s",
                        coverage_candidate.sequence,
                    )
                frontier_candidate = self._build_visual_frontier_candidate(
                    available_names
                )
                if frontier_candidate is not None:
                    candidates = [frontier_candidate, *list(candidates)]
                    logger.info(
                        "Wake injected visual-frontier candidate: %s",
                        frontier_candidate.sequence,
                    )
                lane_candidate = self._build_lane_follow_candidate(
                    available_names
                )
                if lane_candidate is not None:
                    candidates = [lane_candidate, *list(candidates)]
                    logger.info(
                        "Wake injected lane-follow candidate: %s",
                        lane_candidate.sequence,
                    )
                obs_candidate = self._build_observation_route_candidate(available_names)
                if obs_candidate is not None:
                    candidates = [obs_candidate, *list(candidates)]
                    logger.info(
                        "Wake injected observation-route candidate: %s",
                        obs_candidate.sequence,
                    )
                scored_candidates = list(candidates or [])
                plan = self._wake_planner.plan_from_candidates(
                    candidates=candidates,
                    current_grid=grid,
                    active_hypotheses=active_h,
                    wm_draft=self._wm_draft,
                    available_actions=available_names,
                    wm_eval_budget_s=remaining_budget_s,
                )
                plan = self._apply_empirical_route_bonus(
                    plan,
                    available_names,
                )
                plan = self._apply_stall_escape_bonus(
                    plan,
                    available_names,
                    diff_total_snapshot,
                )
                if (
                    exact_candidate is not None
                    and "local_exact_route_to_goal"
                    not in str(getattr(plan.chosen, "rationale", "") or "")
                ):
                    logger.info(
                        "Wake overriding chosen candidate with local exact route: "
                        "chosen=%s override=%s",
                        list(getattr(plan.chosen, "sequence", []) or []),
                        exact_candidate.sequence,
                    )
                    plan = replace(plan, chosen=exact_candidate)
                heuristic_override_enabled = _env_bool(
                    "REV_M_WAKE_HEURISTIC_OVERRIDE", False
                )
                observation_route_override_enabled = (
                    _wake_observation_route_override_enabled()
                )
                allow_host_overrides = self._should_allow_host_wake_overrides(
                    active_h=active_h,
                    skills=skills,
                )
                if heuristic_override_enabled and allow_host_overrides:
                    if (
                        exact_candidate is None
                        and coverage_candidate is not None
                        and "coverage_probe_unprobed"
                        not in str(getattr(plan.chosen, "rationale", "") or "")
                    ):
                        logger.info(
                            "Wake overriding chosen candidate with coverage probe: "
                            "chosen=%s override=%s",
                            list(getattr(plan.chosen, "sequence", []) or []),
                            coverage_candidate.sequence,
                        )
                        plan = replace(plan, chosen=coverage_candidate)
                    if (
                        exact_candidate is None
                        and coverage_candidate is None
                        and lane_candidate is not None
                        and self._should_prefer_lane_follow(
                            lane_candidate, plan.chosen
                        )
                    ):
                        logger.info(
                            "Wake overriding chosen candidate with lane-follow: "
                            "chosen=%s override=%s",
                            list(getattr(plan.chosen, "sequence", []) or []),
                            lane_candidate.sequence,
                        )
                        plan = replace(plan, chosen=lane_candidate)
                    if (
                        exact_candidate is None
                        and coverage_candidate is None
                        and lane_candidate is None
                        and frontier_candidate is not None
                        and self._should_prefer_visual_frontier(
                            frontier_candidate, plan.chosen
                        )
                    ):
                        logger.info(
                            "Wake overriding chosen candidate with visual-frontier: "
                            "chosen=%s override=%s",
                            list(getattr(plan.chosen, "sequence", []) or []),
                            frontier_candidate.sequence,
                        )
                        plan = replace(plan, chosen=frontier_candidate)
                    if (
                        exact_candidate is None
                        and coverage_candidate is None
                        and lane_candidate is None
                        and frontier_candidate is None
                        and self._should_prefer_observation_route(obs_candidate, plan.chosen)
                    ):
                        obs_gain = int(
                            getattr(obs_candidate, "observation_route_gain", 0) or 0
                        )
                        logger.info(
                            "Wake overriding chosen candidate with observation-route: "
                            "gain=%d chosen=%s override=%s",
                            obs_gain,
                            list(getattr(plan.chosen, "sequence", []) or []),
                            obs_candidate.sequence,
                        )
                        plan = replace(plan, chosen=obs_candidate)
                elif (
                    allow_host_overrides
                    and
                    observation_route_override_enabled
                    and exact_candidate is None
                    and coverage_candidate is None
                    and lane_candidate is None
                    and frontier_candidate is None
                    and self._should_prefer_observation_route(
                        obs_candidate, plan.chosen
                    )
                ):
                    obs_gain = int(
                        getattr(obs_candidate, "observation_route_gain", 0) or 0
                    )
                    logger.info(
                        "Wake overriding chosen candidate with observation-route "
                        "(default): gain=%d chosen=%s override=%s",
                        obs_gain,
                        list(getattr(plan.chosen, "sequence", []) or []),
                        obs_candidate.sequence,
                    )
                    plan = replace(plan, chosen=obs_candidate)
                # Rev S P70: forced-exploration probe bypasses LLM choice
                # every FORCED_PROBE_INTERVAL turns so chronically unused
                # actions get tested regardless of hypothesis state.
                if (
                    allow_host_overrides
                    and obs_candidate is None
                    and coverage_candidate is None
                    and exact_candidate is None
                ):
                    try:
                        overridden = self._apply_forced_probe(plan, available_names)
                        if overridden is not None:
                            plan = overridden
                    except Exception:
                        logger.exception("Rev S forced probe override failed")
                scored_candidates = list(getattr(plan, "all_candidates", []) or [])
                chosen_candidate = getattr(plan, "chosen", None)
                fallback_reason = getattr(plan, "fallback_reason", None)
                chosen_sequence = list(plan.chosen.sequence)
                chosen_sequence, compression_reason = self._compress_wake_commit_sequence(
                    chosen_candidate,
                    chosen_sequence,
                )
                logger.info(
                    "Rev M wake: chosen_sequence=%s reward=%.3f fallback=%s compression=%s",
                    chosen_sequence,
                    float(getattr(plan.chosen, "reward", 0.0) or 0.0),
                    fallback_reason,
                    compression_reason,
                )
                # Rev N P50 / R25.3: record chosen reward + spread.
                self._record_phase_reward("wake", plan.chosen.reward)
                self._record_wake_candidate_spread(plan.all_candidates)
                proposed_hypotheses = list(plan.proposed_hypotheses or [])
            except WakePlanError:
                chosen_sequence = list(candidates[0].sequence)
                chosen_candidate = candidates[0]
                scored_candidates = list(candidates or [])
                fallback_reason = "wake_plan_error"
            except Exception:
                logger.exception(
                    "WakePlanner.plan_from_candidates failed; using first candidate"
                )
                chosen_sequence = list(candidates[0].sequence)
                chosen_candidate = candidates[0]
                scored_candidates = list(candidates or [])
                fallback_reason = "wake_plan_exception"
        else:
            # Safe primitive: first available action.
            if available_names:
                chosen_sequence = [available_names[0]]
            elif "RESET" in available_names:
                chosen_sequence = ["RESET"]
            else:
                # C4 fix: no candidates AND no available actions means
                # we can't make progress; raise instead of spinning.
                raise RuntimeError(
                    "Rev M wake: no chosen_sequence and no available_actions; "
                    "cannot proceed"
                )
            chosen_candidate = None
            scored_candidates = []
            fallback_reason = "no_candidates"

        # Rev N P49-hardening Fix 3: when the LLM emits no hypotheses AND
        # the store is still starved (<3 non-seed items), synthesize a
        # small set of falsifiable baseline "no-op" claims so the
        # verifier has something to attack. Cap at 2 per Wake per the
        # store's per-turn quota.
        if not proposed_hypotheses and store_status.get("non_seed_count", 0) < 3:
            # Skip actions that already have a non-seed hypothesis
            # covering them so we don't resubmit the same baseline twice.
            already_tested: set[str] = set()
            try:
                for h in getattr(store, "_items", {}).values():
                    src = getattr(h, "source", "")
                    if src == "seed":
                        continue
                    trig = ""
                    vm = getattr(h, "verification_method", None)
                    if vm is not None:
                        trig = getattr(vm, "trigger", "") or ""
                    if trig.startswith("after ACTION"):
                        already_tested.add(trig[len("after "):])
            except Exception:
                logger.exception(
                    "Rev N P49 synth fallback: already_tested scan failed"
                )
            synth = self._wake_planner.synthesize_fallback_hypotheses(
                available_actions=available_names,
                non_seed_count=store_status.get("non_seed_count", 0),
                max_synth=2,
                already_tested_actions=already_tested,
            )
            if synth:
                logger.info(
                    "Wake synth fallback: generating %d baseline hypotheses "
                    "(non_seed_count=%d, tested=%s)",
                    len(synth),
                    store_status.get("non_seed_count", 0),
                    sorted(already_tested),
                )
                proposed_hypotheses = synth

        # Rev N P49: submit LLM-proposed (or synth-fallback) hypotheses
        # through the strict gate. Rejections are surfaced as structured
        # feedback for the next Wake prompt (R25.1). Submission failures
        # never block sequence execution.
        self._submit_proposed_hypotheses(proposed_hypotheses)
        self._append_module_io_event(
            "wake_output",
            {
                "wake_turn_index": wake_turn_index,
                "chosen_sequence": list(chosen_sequence),
                "proposed_hypothesis_claims": [
                    str((h or {}).get("claim", ""))[:240]
                    for h in list(proposed_hypotheses or [])
                ],
                "chosen_reward": float(
                    getattr(chosen_candidate, "reward", 0.0) or 0.0
                )
                if chosen_candidate is not None
                else None,
                "chosen_rationale": str(
                    getattr(chosen_candidate, "rationale", "") or ""
                )[:240]
                if chosen_candidate is not None
                else "",
                "compression_reason": compression_reason,
                "fallback_reason": fallback_reason,
                "candidate_rationales": [
                    str(getattr(c, "rationale", "") or "")[:240]
                    for c in list(scored_candidates or [])[:8]
                ],
                "candidate_debug": [
                    {
                        "sequence": list(getattr(c, "sequence", []) or []),
                        "reward": float(getattr(c, "reward", 0.0) or 0.0),
                        "debug_bonus": float(getattr(c, "debug_bonus", 0.0) or 0.0),
                        "debug_penalty": float(getattr(c, "debug_penalty", 0.0) or 0.0),
                        "rationale": str(getattr(c, "rationale", "") or "")[:240],
                    }
                    for c in list(scored_candidates or [])[:8]
                ],
                "local_exact_route_enabled": self._local_exact_route_enabled(),
                "local_exact_route_present": any(
                    "local_exact_route_to_goal"
                    in str(getattr(c, "rationale", "") or "")
                    for c in list(scored_candidates or [])
                ),
                "observation_route_override_enabled": _wake_observation_route_override_enabled(),
                "observation_route_present": bool(obs_candidate is not None) if candidates else False,
                "observation_route_gain": int(
                    getattr(obs_candidate, "observation_route_gain", 0) or 0
                )
                if candidates and obs_candidate is not None
                else 0,
            },
        )

        # Execute the chosen sequence on the real env with divergence
        # monitoring. We reuse the legacy take_action + append_frame
        # path so all existing research hooks continue to fire.
        wake_trace = WakeTrace()
        diverged = False
        current_frame = latest
        current_grid_view = grid
        for step_idx, action_name in enumerate(chosen_sequence):
            latest_now = self.frames[-1]
            allowed = [
                GameAction.from_id(a).name for a in latest_now.available_actions
            ]
            if action_name not in allowed:
                if step_idx == 0 and allowed:
                    logger.warning(
                        "Rev M wake: illegal first action %s not in %s; falling back to %s",
                        action_name,
                        allowed,
                        allowed[0],
                    )
                    action_name = allowed[0]
                else:
                    logger.warning(
                        "Rev M wake: aborting sequence at step=%d action=%s allowed=%s",
                        step_idx,
                        action_name,
                        allowed,
                    )
                    break
            # WM prediction for the step (pre-action grid).
            predicted_grid: Any = None
            try:
                sim = self._wake_planner._run_wm_step(
                    self._wm_draft, "", action_name, current_grid_view
                ) if self._wm_draft is not None else None
                if isinstance(sim, dict):
                    nxt = sim.get("expected_next_grid")
                    if isinstance(nxt, list) and nxt:
                        predicted_grid = nxt
            except Exception:
                predicted_grid = None

            # Rev M world-model scoring fix: the legacy choose_action path
            # records a synthetic EXPECT_CHANGE/EXPECT_NO_CHANGE prediction
            # before the real action commits, which is what drives
            # WorldModel.structured_predictions and empirical scoring.
            # The Rev M wake loop bypasses choose_action, so we recreate the
            # same bridge-side prediction here before executing the action.
            self._record_rev_m_prediction_hint(current_frame, action_name)

            action = GameAction.from_name(action_name)
            frame = self.take_action(action)
            if frame is None:
                # C4 fix: env returned no frame. Flag the wake trace so
                # the outer dispatch loop can detect stalling and abort
                # cleanly instead of spinning toward the max_iters rail.
                logger.warning(
                    "Rev M wake: take_action(%s) returned None; aborting block",
                    action.name,
                )
                wake_trace.no_env_progress = True
                break
            self.append_frame(frame)
            try:
                rich_frame = self._to_rich_frame(frame)
                self.action_history.append((action.name, rich_frame))
            except Exception:
                self.action_history.append((action.name, frame))
            # Bug C fix: track flat action names for Wake novelty tie-break.
            try:
                history = getattr(self, "_action_history", None)
                if not isinstance(history, list):
                    history = []
                    self._action_history = history
                history.append(action.name)
            except Exception:
                logger.exception("Rev M: action_history append failed")
            prev_best = self.best_level
            self.best_level = max(self.best_level, frame.levels_completed)
            # Rev Q (P64): level-advance delta from this real step.
            prev_levels = getattr(current_frame, "levels_completed", 0) or 0
            new_levels = getattr(frame, "levels_completed", 0) or 0
            try:
                delta = max(0, int(new_levels) - int(prev_levels))
            except (TypeError, ValueError):
                delta = 0
            self._last_level_advance_delta = delta
            try:
                bridge_for_level = (
                    self._registry.bridge if self._registry is not None else None
                )
                if bridge_for_level is not None:
                    bridge_for_level.last_level_advance_delta = delta
            except Exception:
                logger.exception("Rev Q: bridge level_advance_delta set failed")
            if self.best_level > prev_best:
                self._actions_probed_this_level = set()
            if action.name == "RESET":
                self.reset_count += 1
                self._actions_probed_this_level = set()
            elif action.name.startswith("ACTION"):
                self._actions_probed_this_level.add(action.name)

            # Fire the research after_action hook so hypothesis
            # evaluations / skill book-keeping still runs.
            if self._registry:
                try:
                    self._registry.after_action(
                        current_frame, action.name, frame
                    )
                except Exception:
                    logger.exception("registry.after_action failed; continuing")

            logger.info(
                "%s - %s: L=%s, count=%s",
                self.game_id,
                action.name,
                getattr(frame, "levels_completed", 0),
                self.action_counter,
            )
            if action.name != "RESET":
                self.action_counter += 1

            real_after_grid = current_grid(frame)
            # W2 fix: record the observed "family" signature on the
            # bridge so WakePlanner._compute_novelty can penalise
            # revisiting the same family. Prefer the WM's predicted
            # signature when available; fall back to an action-based
            # synthetic key so novelty is non-zero on the first visit
            # even when the WM is blank.
            try:
                bridge = (
                    self._registry.bridge if self._registry is not None else None
                )
                if bridge is not None:
                    sig_from_wm = None
                    try:
                        sim = self._wake_planner._run_wm_step(
                            self._wm_draft, "", action.name, current_grid_view
                        ) if self._wm_draft is not None else None
                        if isinstance(sim, dict):
                            nxt = sim.get("next_signature_hint")
                            if isinstance(nxt, str) and nxt:
                                sig_from_wm = nxt
                    except Exception:
                        sig_from_wm = None
                    sig = sig_from_wm or f"real:{action.name}"
                    fams = getattr(bridge, "seen_families", None)
                    if fams is None or not isinstance(fams, set):
                        bridge.seen_families = set()
                        fams = bridge.seen_families
                    fams.add(sig)
            except Exception:
                logger.exception(
                    "Rev M wake: seen_families update failed; continuing"
                )

            # Rev O P54-P55: feed the real post-action grid to the
            # player tracker and record the before/after diff for the
            # next Wake prompt. Failures never block the action loop.
            _tracker = getattr(self, "_player_tracker", None)
            if _tracker is not None:
                try:
                    before_state = getattr(_tracker, "latest_state", None)
                    _tracker.update(self.action_counter, real_after_grid)
                    after_state = getattr(_tracker, "latest_state", None)
                    self._record_action_player_delta(
                        action_name,
                        getattr(before_state, "center", None),
                        getattr(after_state, "center", None),
                    )
                except Exception:
                    logger.exception("Rev O: player_tracker.update failed")
            try:
                self._last_diff_coords = compute_diff_coords(
                    current_grid_view, real_after_grid, max_entries=20
                )
                self._last_diff_total = count_diffs(
                    current_grid_view, real_after_grid
                )
            except Exception:
                logger.exception("Rev O: diff_coords computation failed")
                self._last_diff_coords = []
                self._last_diff_total = 0

            # Rev P P59: cache the amplified BEFORE/AFTER diff patch so
            # the next Wake prompt shows the pixel-level change block.
            try:
                self._last_diff_before_grid = current_grid_view
                self._last_diff_after_grid = real_after_grid
                self._last_visible_grid = real_after_grid
                if self._last_diff_total > 0:
                    self._last_diff_patch_text = render_diff_amplified(
                        current_grid_view, real_after_grid
                    )
                else:
                    self._last_diff_patch_text = ""
            except Exception:
                logger.exception("Rev P: diff patch render failed")
                self._last_diff_patch_text = ""

            # Divergence check (P43).
            div_result = self._divergence_monitor.check_step(
                self.action_counter, predicted_grid, real_after_grid
            )
            # C5 fix: only feed *non-divergent* samples into the rolling
            # window. Divergent distances and wm_uncertain samples are
            # outliers that would inflate the adaptive threshold on
            # subsequent turns, making the monitor progressively blind.
            if (
                not div_result.is_divergent
                and div_result.reason != "wm_uncertain"
            ):
                self._divergence_monitor.record_observation(div_result.distance)
            wake_trace.transitions.append({
                "before": current_grid_view,
                "action": action.name,
                "after": real_after_grid,
                "turn": self.action_counter,
                "predicted_after": predicted_grid,
            })
            if div_result.is_divergent:
                wake_trace.divergences.append(DivergenceReport(
                    step_index=step_idx,
                    action=action.name,
                    pred_grid=predicted_grid,
                    real_grid=real_after_grid,
                    grid_diff_summary=_grid_diff_summary(
                        predicted_grid, real_after_grid
                    ),
                    reason=div_result.reason,
                ))
                self._last_divergence = {
                    "step_index": step_idx,
                    "action": action.name,
                    "reason": div_result.reason,
                }
                diverged = True
                break

            current_frame = frame
            current_grid_view = real_after_grid

            if self.is_done(self.frames, frame):
                break
            if self.action_counter >= self.MAX_ACTIONS:
                break

        self._current_wake_trace = wake_trace
        if not wake_trace.transitions:
            wake_trace.no_env_progress = True
            logger.warning(
                "Rev M wake: no real env action committed; staying in Wake"
            )
            self._phase_controller.transition_to(
                Phase.WAKE,
                reason="no_env_progress",
                turn=self.action_counter,
            )
            return
        if not diverged and wake_trace.transitions:
            self._recent_successful_sequences.append({
                "sequence": [t["action"] for t in wake_trace.transitions],
                "goal_hit": self.is_done(self.frames, self.frames[-1]),
            })
            # Keep the window bounded.
            self._recent_successful_sequences = (
                self._recent_successful_sequences[-16:]
            )
        # Always transition to Sleep after a Wake block (plan §P38 cycle).
        self._phase_controller.transition_to(
            Phase.SLEEP,
            reason="divergence" if diverged else "sequence_end",
            turn=self.action_counter,
        )

    def _record_rev_m_prediction_hint(
        self,
        frame: Any,
        action_name: str,
    ) -> None:
        """Mirror choose_action-side prediction synthesis in Rev M Wake.

        Without this hook, Rev M executes real actions without ever pushing a
        structured prediction onto the bridge, so the world model never gets
        empirical hit/miss scoring and ``structured_predictions`` stays zero.
        """
        if self._registry is None:
            return
        try:
            self._registry.synthesize_prediction(frame, action_name)
        except Exception:
            logger.exception(
                "Rev M: synthesize_prediction failed for wake action %s",
                action_name,
            )

    def _run_sleep_phase(self) -> None:
        """Run Sleep consolidation + decide next transition."""
        if not self._rev_m_initialized:
            return
        # W9 fix: reset the tool tracker at phase entry so leftover
        # Wake-phase tool calls don't bleed into Sleep gate telemetry.
        self._tool_tracker.reset()
        wake_trace = self._current_wake_trace
        store = self._hypothesis_store()
        # Rev N P53: attach the gate + tracker to the handler so the
        # internal LLM call is routed through gated_llm_call under a
        # 30s wall-clock budget (R24.6).
        #
        # Rev N P53 live-run fix: the tracker-based gate always FAILs
        # on the current text-in/JSON-out LLM path (no tool-calling),
        # triggering 2 retries per turn. Default to telemetry mode:
        # ``REV_M_HARD_GATE=0`` skips the gate wiring so the handler
        # takes the legacy single-call path. Opt in with
        # ``REV_M_HARD_GATE=1`` to use the reference-based gate.
        divergence_count = len(getattr(wake_trace, "divergences", []) or [])
        if _rev_m_hard_gate_enabled():
            self._sleep_handler.tool_tracker = self._tool_tracker

            def _sleep_gate_check(last_response_text: str | None = None):
                return check_helper_references(last_response_text, phase="sleep")

            self._sleep_handler.gate_check_fn = _sleep_gate_check
            self._sleep_handler.max_retries = 2
            self._sleep_handler.total_budget_s = 30.0
        else:
            # Telemetry mode: ensure no gated-retry path runs.
            self._sleep_handler.tool_tracker = None
            self._sleep_handler.gate_check_fn = None
            self._sleep_handler.max_retries = 0
            self._sleep_handler.total_budget_s = 30.0
        # Keep Sleep aligned with the live WorldModelModule draft rather than
        # the last Sleep-authored source only. Otherwise a valid draft loaded
        # or updated during Wake is invisible here, forcing an oversized
        # cold-start bootstrap prompt and increasing timeout risk.
        latest_wm_source = self._fetch_wm_source()
        if latest_wm_source:
            self._wm_source = latest_wm_source
        logger.info(
            "Rev M sleep preflight: wake_transitions=%d divergences=%d wm_source_len=%d",
            len(getattr(wake_trace, "transitions", []) or []),
            divergence_count,
            len(self._wm_source or ""),
        )
        try:
            sleep_helpers_fn = (
                getattr(self._registry, "_get_symbolica_helpers", None)
                if self._registry is not None
                else None
            )
            sleep_helpers = (
                sleep_helpers_fn() if callable(sleep_helpers_fn) else {}
            )
            sleep_env_state_fn = (
                getattr(self._registry, "_get_env_state_dict", None)
                if self._registry is not None
                else None
            )
            sleep_env_state = (
                sleep_env_state_fn(self.frames[-1])
                if callable(sleep_env_state_fn) and self.frames
                else {}
            )
            try:
                sleep_result = self._sleep_handler.consolidate(
                    wake_trace=wake_trace,
                    hypothesis_store=store,
                    current_wm_source=self._wm_source,
                    helpers=sleep_helpers,
                    env_state=sleep_env_state,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                sleep_result = self._sleep_handler.consolidate(
                    wake_trace, store, self._wm_source
                )
        except Exception:
            logger.exception("Sleep handler raised; skipping consolidation")
            self._phase_controller.transition_to(
                Phase.WAKE, reason="sleep_error", turn=self.action_counter
            )
            return
        # Rev N P50: record sleep reward.
        self._record_phase_reward("sleep", sleep_result.reward)

        # Rev N P51: project each verdict onto the WM support/falsify
        # counters so the symbolic and WM layers share evidence. Guarded
        # by module availability; failures are logged but not fatal.
        try:
            self._project_verdicts_onto_wm(sleep_result.verdict_projections)
        except Exception:
            logger.exception("Rev N P51: verdict projection failed")

        # Rev N P53: prefer the gated-retry compliance flag the handler
        # wrote after its internal ``gated_llm_call``; fall back to a
        # one-shot gate check when the handler had no LLM call to route
        # (e.g. no divergence, no world_update).
        try:
            self._last_sleep_tool_compliance = bool(
                getattr(self._sleep_handler, "last_tool_compliance", True)
            )
            if sleep_result.world_update is None:
                # Legacy telemetry path for no-LLM-call Sleep turns.
                sleep_gate_check = self._sleep_gate.check(
                    self._tool_tracker, divergence_count
                )
                self._last_sleep_tool_compliance = (
                    sleep_gate_check.result.name == "PASS"
                )
            if not self._last_sleep_tool_compliance:
                logger.info(
                    "Rev M sleep gate tool_compliance=False"
                )
        except Exception:
            logger.exception("Sleep gate check failed; ignoring")

        # Apply world_update to the live WorldModelModule (C2 fix).
        # ``WorldModelModule.record_agent_world_update`` expects a dict
        # with a ``body`` key wrapping the source in a ```python``` fence
        # (see research_extensions/modules/world_model.py::_normalize_world_update).
        # DIAG: log what we got from Sleep.
        logger.info(
            "Rev M sleep: world_update=%s source_len=%d",
            bool(sleep_result.world_update),
            len(str((sleep_result.world_update or {}).get("source") or "")),
        )
        if sleep_result.world_update:
            new_source = str(sleep_result.world_update.get("source") or "")
            if new_source:
                self._wm_source = new_source
                if self._registry is not None:
                    try:
                        self._registry.record_agent_world_update({
                            "body": f"```python\n{new_source}\n```",
                        })
                        logger.info(
                            "Rev M sleep: record_agent_world_update called (source_len=%d)",
                            len(new_source),
                        )
                    except Exception:
                        logger.exception("WM record_agent_world_update failed")
                self._reachability_gate.invalidate()
                # W2 fix: Sleep rewrote the WM, so the previous divergence
                # has been absorbed. Clear the stale summary so the next
                # Wake prompt does not replay it.
                self._last_divergence = None

        # C1 fix: hypothesis_updates is telemetry; the store was already
        # mutated by evaluate_all inside consolidate(). Do NOT re-apply.

        # Synthetic consistency check AFTER the WM update (P44).
        consistency_rate = 0.0
        consistency_total = 0
        if sleep_result.world_update and self._synthetic_checker is not None:
            try:
                base = current_grid(self.frames[-1])
                consistency = self._synthetic_checker.check(base_grid=base)
                consistency_rate = float(getattr(consistency, "rate", 0.0) or 0.0)
                consistency_total = int(getattr(consistency, "total_cases", 0) or 0)
                logger.info(
                    "Rev M: synthetic consistency rate=%.3f total=%d",
                    consistency.rate, consistency.total_cases,
                )
            except Exception:
                logger.exception("synthetic consistency check failed")
        sleep_empirical_ctx: dict[str, Any] = {}
        try:
            sleep_empirical_ctx = dict(
                self._sleep_handler._empirical_mismatch_context()  # type: ignore[attr-defined]
            )
        except Exception:
            sleep_empirical_ctx = {}
        self._append_module_io_event(
            "sleep_output",
            {
                "divergence_count": int(divergence_count),
                "world_update": bool(sleep_result.world_update),
                "world_update_source_len": len(
                    str((sleep_result.world_update or {}).get("source") or "")
                ),
                "reward": float(getattr(sleep_result, "reward", 0.0) or 0.0),
                "tool_compliance": bool(
                    getattr(self, "_last_sleep_tool_compliance", True)
                ),
                "verdict_projection_count": len(
                    list(getattr(sleep_result, "verdict_projections", []) or [])
                ),
                "empirical_ctx": sleep_empirical_ctx,
                "synthetic_consistency_rate": float(consistency_rate),
                "synthetic_consistency_total": int(consistency_total),
            },
        )

        # Decide next transition.
        if self._phase_controller.should_enter_abstraction(
            turn=self.action_counter,
            last_abstraction_turn=self._phase_controller.last_abstraction_turn,
        ):
            self._phase_controller.transition_to(
                Phase.ABSTRACTION,
                reason="cycle_end",
                turn=self.action_counter,
            )
        else:
            self._phase_controller.transition_to(
                Phase.WAKE,
                reason="abstraction_cooldown",
                turn=self.action_counter,
            )

    def _run_abstraction_phase(self) -> None:
        """Run Abstraction compression + return to Wake."""
        if not self._rev_m_initialized:
            return
        # W9 fix: reset the tool tracker at phase entry.
        self._tool_tracker.reset()
        confirmed = self._hypothesis_confirmed_list(min_count=2)
        ab_input = AbstractionInput(
            confirmed_hypotheses=confirmed,
            successful_wake_sequences=list(self._recent_successful_sequences),
            existing_skill_summaries=self._skill_summaries(),
        )
        # Rev N P53: attach the gate + tracker so the handler routes its
        # LLM call through gated_llm_call under a 30s budget (R24.6).
        # skill_count proxy: we expect at least 1 skill per retry attempt,
        # so the gate checks for at least 1 frame_bounding_box call. Real
        # compliance telemetry uses the post-parse count below.
        #
        # Rev N P53 live-run fix: tracker-based gate always FAILs on
        # text-in/JSON-out path. Default to telemetry (no retries);
        # opt in to hard gate with ``REV_M_HARD_GATE=1``.
        if _rev_m_hard_gate_enabled():
            self._abstraction_handler.tool_tracker = self._tool_tracker

            def _abs_gate_check(last_response_text: str | None = None):
                return check_helper_references(
                    last_response_text, phase="abstraction"
                )

            self._abstraction_handler.gate_check_fn = _abs_gate_check
            self._abstraction_handler.max_retries = 2
            self._abstraction_handler.total_budget_s = 30.0
        else:
            self._abstraction_handler.tool_tracker = None
            self._abstraction_handler.gate_check_fn = None
            self._abstraction_handler.max_retries = 0
            self._abstraction_handler.total_budget_s = 30.0
        try:
            ab_result = self._abstraction_handler.compress(ab_input)
        except Exception:
            logger.exception("Abstraction handler raised; skipping pass")
            self._phase_controller.transition_to(
                Phase.WAKE,
                reason="abstraction_error",
                turn=self.action_counter,
            )
            return
        # Rev N P50: record abstraction reward.
        self._record_phase_reward("abstraction", ab_result.reward)

        for skill in ab_result.proposed_skills:
            self._propose_skill(skill)

        # Rev N P53 telemetry: prefer the gated-retry compliance flag
        # the handler wrote. Fall back to a post-parse gate check for
        # turns where no LLM call ran (e.g. empty inputs).
        try:
            if getattr(self._abstraction_handler, "last_tool_compliance", True):
                # Even under PASS we still re-check against the real
                # proposal count so the logged telemetry reflects
                # post-parse state.
                ab_gate_check = self._abstraction_gate.check(
                    self._tool_tracker, len(ab_result.proposed_skills)
                )
                self._last_abstraction_tool_compliance = (
                    ab_gate_check.result.name == "PASS"
                )
            else:
                self._last_abstraction_tool_compliance = False
            if not self._last_abstraction_tool_compliance:
                logger.info(
                    "Rev M abstraction gate tool_compliance=False"
                )
        except Exception:
            logger.exception("Abstraction gate check failed; ignoring")
        self._append_module_io_event(
            "abstraction_output",
            {
                "confirmed_hypothesis_count": len(list(confirmed or [])),
                "existing_skill_count": len(
                    list(ab_input.existing_skill_summaries or [])
                ),
                "proposed_skill_names": [
                    str((s or {}).get("name", ""))[:200]
                    for s in list(ab_result.proposed_skills or [])
                ],
                "reward": float(getattr(ab_result, "reward", 0.0) or 0.0),
                "tool_compliance": bool(
                    getattr(self, "_last_abstraction_tool_compliance", True)
                ),
            },
        )

        # Clear the successful-sequences window so we don't double-count.
        self._recent_successful_sequences = []
        self._phase_controller.transition_to(
            Phase.WAKE,
            reason="cycle_complete",
            turn=self.action_counter,
        )

    def _rev_m_main_loop(self) -> None:
        """Rev M phase-dispatch outer loop.

        Dispatches one phase per iteration. Wake consumes real env
        turns; Sleep / Abstraction are instantaneous (no env step) and
        so do not advance the budget.
        """
        self.timer = time.time()
        self._ensure_bootstrap_frame()
        # Safety rail: bound total iterations independently of the env
        # budget so a flaky phase controller can't loop forever.
        max_iters = (self.MAX_ACTIONS + 1) * 4
        iters = 0
        # C4 fix: track consecutive Wake cycles that yielded no env
        # progress so we can abort decisively rather than spin to the
        # max_iters rail.
        no_progress_streak = 0
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter < self.MAX_ACTIONS
            and iters < max_iters
        ):
            iters += 1
            phase = self._phase_controller.current_phase
            try:
                if phase == Phase.WAKE:
                    self._run_wake_phase()
                    if getattr(
                        self._current_wake_trace, "no_env_progress", False
                    ):
                        no_progress_streak += 1
                    else:
                        no_progress_streak = 0
                    if no_progress_streak >= 3:
                        logger.error(
                            "Rev M dispatch: 3 consecutive wake cycles with "
                            "no env progress; aborting"
                        )
                        break
                elif phase == Phase.SLEEP:
                    self._run_sleep_phase()
                elif phase == Phase.ABSTRACTION:
                    self._run_abstraction_phase()
                else:
                    break
            except Exception:
                logger.exception("Rev M phase dispatch failed; aborting cycle")
                break

        # Flush phase event log (match legacy cleanup semantics).
        try:
            self._phase_controller.close(reason="run_end")
        except Exception:
            logger.exception("PhaseController.close failed")

        self._rev_m_finalize()

    def _rev_m_finalize(self) -> None:
        """Legacy parity: run the same on_run_end + summary flushing
        as the old main() tail block so telemetry stays identical.
        """
        # Rev N P50: publish rolling per-phase reward telemetry so the
        # MetaHarness module can attach it to RunMetrics. Done before
        # on_run_end so _score_run sees a populated hint.
        try:
            self._publish_phase_reward_telemetry()
        except Exception:
            logger.exception("Rev N P50: publish_phase_reward_telemetry failed")
        if self._registry:
            try:
                self._registry.on_run_end(
                    history=self.action_history, finish_status=None
                )
            except Exception:
                logger.exception(
                    "registry.on_run_end failed; shared dir may be incomplete"
                )
        if hasattr(self, "_memories") and self._memories is not None:
            try:
                self._memories.save()
            except Exception:
                logger.exception("memories.save failed")
        unique_actions = sorted({name for name, _ in self.action_history})
        (self.research_workdir / "summary.json").write_text(json.dumps({
            "best_level": self.best_level,
            "actions": self.action_counter,
            "reset_count": self.reset_count,
            "unique_actions": unique_actions,
            "active_modules":
                self.research_config.active_modules()
                if self.research_config else [],
            "shared_namespace": self.shared_namespace,
            "rev_m_enabled": self.rev_m_enabled,
        }, indent=2))
        logger.info(
            f"Done: L={self.best_level}, actions={self.action_counter}"
            f" (rev_m={self.rev_m_enabled})"
        )
        self.cleanup()

    @trace_agent_session
    def main(self) -> None:
        # Tolerate agents constructed via ``__new__`` without __init__
        # (test fixtures do this). Default to the legacy path when the
        # Rev M flag/state isn't populated.
        if getattr(self, "rev_m_enabled", False) and getattr(
            self, "_rev_m_initialized", False
        ):
            self._rev_m_main_loop()
            return
        self.timer = time.time()
        self._ensure_bootstrap_frame()
        while not self.is_done(self.frames, self.frames[-1]) and self.action_counter < self.MAX_ACTIONS:
            latest = self.frames[-1]
            available_names = [GameAction.from_id(a).name for a in latest.available_actions]

            # Plan v3 §4 P0: imagination + option-execution bypass.
            bypass_action: str | None = None
            if self._registry and hasattr(self._registry, "imagine_and_maybe_commit"):
                try:
                    bypass_action = self._registry.imagine_and_maybe_commit(latest, available_names)
                except Exception:
                    logger.exception("imagination phase failed; falling back to LLM")
                    bypass_action = None

            # P22 (plan v4.2 §P22): during the pre-MCTS warm-up phase the
            # LLM is useless — it has no per_action_delta_hints to work
            # with. Skip the LLM entirely and use the registry's
            # least-tried action picker so hints accumulate without
            # burning TRAPI budget. Once enough transitions exist
            # (exploration_min_transitions from planner config) the gate
            # closes and the LLM starts writing predict_effect on real
            # data.
            if (
                bypass_action is None
                and self._registry is not None
                and hasattr(self._registry, "_in_exploration_phase")
            ):
                try:
                    in_explore, reason = self._registry._in_exploration_phase()
                except Exception:
                    in_explore, reason = False, ""
                if in_explore:
                    picked = self._registry._exploration_action(latest, available_names)
                    if picked and picked in available_names:
                        bypass_action = picked
                        # Record synthetic trace so later LLM turns see
                        # this as the agent's prior action.
                        try:
                            self._registry.bridge.update_hint(
                                "exploration_reason", reason
                            )
                        except Exception:
                            pass
                        logger.info(
                            "[P22 warmup] LLM skipped; exploration action=%s (%s)",
                            picked, reason,
                        )

            if bypass_action:
                # Synthesise a prediction so after_action can score the
                # transition against the simulator.
                try:
                    self._registry.synthesize_prediction(latest, bypass_action)
                except Exception:
                    logger.exception("synthesize_prediction failed; continuing")
                action = GameAction.from_name(bypass_action)
                action_name = action.name
                # Log bypass so the next LLM turn can read it in overlay.
                try:
                    committed = self._registry.bridge.current_skill_in_execution
                    self._registry.record_bypass_step(
                        skill=committed.skill_name if committed else "?",
                        action=bypass_action,
                        outcome="pending",  # after_action will not update; agent sees 'pending' = ran successfully
                        turn=self.action_counter,
                    )
                except Exception:
                    pass
                logger.info("[v3 bypass] skill-committed action %s (LLM skipped)", bypass_action)
            else:
                action = self.choose_action(self.frames, latest)
                action_name = action.name
                if self._registry:
                    override = self._registry.before_action(latest, action_name, available_names)
                    if override:
                        action = GameAction.from_name(override)

            # Rev U: legacy-path Rev T imbalance override. The Rev M
            # path benefits from ``_apply_forced_probe`` inside
            # ``_run_wake_phase`` but the legacy main() loop never sees
            # it, which let ACTION3 dominate 83% of turns in ep43. Swap
            # the chosen action for the under-used one when the last
            # 20-step window shows a 2x skew. try/except so a probe
            # failure can never block the loop.
            try:
                rebalanced = self._legacy_imbalance_override(
                    action.name, available_names
                )
                if rebalanced and rebalanced != action.name:
                    action = GameAction.from_name(rebalanced)
            except Exception:
                logger.exception(
                    "Rev U legacy imbalance override failed; "
                    "keeping original action"
                )

            frame = self.take_action(action)
            if frame:
                self.append_frame(frame)
                # Defensive: _to_rich_frame can crash if the engine returns
                # a frame with an empty `.frame` list (level-transition
                # edge case). Skip rich-frame wrapping for that step so
                # the main loop can continue and on_run_end still persists
                # state.
                try:
                    rich_frame = self._to_rich_frame(frame)
                    self.action_history.append((action.name, rich_frame))
                except Exception:
                    logger.exception(
                        "_to_rich_frame failed (likely empty frame.frame); "
                        "recording raw FrameData so the main loop can continue"
                    )
                    self.action_history.append((action.name, frame))
                # Rev U: flat action-name history powering the legacy
                # imbalance override. Appended after every successful
                # commit so the next-turn override sees the latest step.
                try:
                    self._legacy_action_history.append(action.name)
                except Exception:
                    logger.exception(
                        "Rev U: _legacy_action_history append failed"
                    )
                prev_best = self.best_level
                self.best_level = max(self.best_level, frame.levels_completed)
                # Rev Q (P64): level-advance delta from this real step.
                prev_levels = getattr(latest, "levels_completed", 0) or 0
                new_levels = getattr(frame, "levels_completed", 0) or 0
                try:
                    delta = max(0, int(new_levels) - int(prev_levels))
                except (TypeError, ValueError):
                    delta = 0
                self._last_level_advance_delta = delta
                try:
                    bridge_for_level = (
                        self._registry.bridge if self._registry is not None else None
                    )
                    if bridge_for_level is not None:
                        bridge_for_level.last_level_advance_delta = delta
                except Exception:
                    logger.exception("Rev Q: bridge level_advance_delta set failed")
                if self.best_level > prev_best:
                    # Level advanced → per-level probe set resets.
                    self._actions_probed_this_level = set()
                if action.name == "RESET":
                    self.reset_count += 1
                    # RESET erases in-level progress; restart probe set.
                    self._actions_probed_this_level = set()
                elif action.name.startswith("ACTION"):
                    self._actions_probed_this_level.add(action.name)

                # Research hook: after_action
                if self._registry:
                    try:
                        self._registry.after_action(latest, action.name, frame)
                    except Exception:
                        logger.exception("registry.after_action failed; continuing")

                logger.info(f"{self.game_id} - {action.name}: L={frame.levels_completed}, count={self.action_counter}")

            if action.name != "RESET":
                self.action_counter += 1

        # Research hook: on_run_end (always run, even after exceptions).
        if self._registry:
            try:
                self._registry.on_run_end(history=self.action_history, finish_status=None)
            except Exception:
                logger.exception("registry.on_run_end failed; shared dir may be incomplete")
        if hasattr(self, "_memories") and self._memories is not None:
            try:
                self._memories.save()
            except Exception:
                logger.exception("memories.save failed")

        # Save research state
        unique_actions = sorted({name for name, _ in self.action_history})
        (self.research_workdir / "summary.json").write_text(json.dumps({
            "best_level": self.best_level,
            "actions": self.action_counter,
            "reset_count": self.reset_count,
            "unique_actions": unique_actions,
            "active_modules": self.research_config.active_modules() if self.research_config else [],
            "shared_namespace": self.shared_namespace,
        }, indent=2))
        logger.info(f"Done: L={self.best_level}, actions={self.action_counter}")
        self.cleanup()


class _LocalMemories:
    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path
        self.stack: list[_Memory] = []
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                for item in raw:
                    self.stack.append(
                        _Memory(
                            summary=str(item.get("summary", "")),
                            details=str(item.get("details", "")),
                            timestamp=datetime.fromisoformat(str(item.get("timestamp"))),
                        )
                    )
            except Exception:
                self.stack = []

    def add(self, title: str, text: str) -> None:
        self.stack.append(_Memory(summary=title, details=text))

    def summaries(self) -> list[str]:
        return [f"[{i}] {m.summary}" for i, m in enumerate(self.stack)]

    def get(self, i: int) -> "_Memory":
        return self.stack[i]

    def query(self, question: str, *, return_type: str = "str", limit: int = 3) -> dict[str, Any]:
        tokens = {token.lower() for token in question.replace("_", " ").split() if token.strip()}
        ranked: list[tuple[int, int, _Memory]] = []
        for idx, memory in enumerate(self.stack):
            haystack = f"{memory.summary}\n{memory.details}".lower()
            score = sum(1 for token in tokens if token in haystack)
            if score > 0 or not tokens:
                ranked.append((score, idx, memory))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        matches = [
            {
                "index": idx,
                "summary": memory.summary,
                "details": memory.details,
                "timestamp": memory.timestamp.isoformat(),
            }
            for score, idx, memory in ranked[:limit]
        ]
        answer = "\n\n".join(
            f"[{item['index']}] {item['summary']}\n{item['details']}" for item in matches
        )
        return {
            "question": question,
            "return_type": return_type,
            "matches": matches,
            "answer": answer,
        }

    def save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(
                [
                    {
                        "summary": memory.summary,
                        "details": memory.details,
                        "timestamp": memory.timestamp.isoformat(),
                    }
                    for memory in self.stack
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


class _Memory:
    def __init__(self, summary: str, details: str, timestamp: datetime | None = None) -> None:
        self.summary = summary
        self.details = details
        self.timestamp = timestamp or datetime.now()

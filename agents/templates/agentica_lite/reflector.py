"""v601 conditional Reflector sub-LLM + v607 stuck-trigger emit path.

v601 Plan rev C §3.15: ReflectorOutput with discriminator features, brief
reflexion text, UCB1 exploration boost (values in [0, 0.3], lifetime 5 turns).

v607 Plan rev E §3 Option B: stuck-triggered chid_template emission.
Reflector fires when (turns_since_advance > 3) AND (cooldown == 0) AND
(fires_this_episode < ADAPTIVE_CAP) AND (best_advance_age >= 5) OR new
dominant_transition since last fire. On fire, LLM emits a new chid_template
(verbal, no code) that gets appended to skill_state.json with Beta(0.5, 0.5)
Jeffreys prior. Anti-leak validation enforced before mutation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .predicate_posterior import ArmKey

logger = logging.getLogger(__name__)

_TRAPI_ENDPOINT = "https://trapi.research.microsoft.com/gcr/shared"
_API_VERSION = "2025-04-01-preview"
_MODEL_PREFERENCES = ["gpt-5.4-mini_2026-03-17", "gpt-5.4_2026-03-05"]

REFLECTOR_SYSTEM_PROMPT = (
    "You are a contrastive analyzer for a structured-observation puzzle agent.\n"
    "Two outcomes occurred at the same coordinate but with different signatures.\n"
    "Your task: identify which discriminating feature(s) most likely caused the\n"
    "difference, and suggest a small UCB1 exploration boost (in [0, 0.3]) for\n"
    "arms that should be tried next.\n\n"
    "Output strict JSON:\n"
    "  discriminator_features: list[str]\n"
    "  reflexion_text: str (<= 256 chars)\n"
    "  suggested_exploration_boost: dict[str, float]  # arm key as string -> boost\n"
)


@dataclass
class ReflectorOutput:
    discriminator_features: list[str]
    reflexion_text: str
    suggested_exploration_boost: dict[ArmKey, float] = field(default_factory=dict)


def _parse_arm_key(s: str) -> ArmKey | None:
    """Parse 'pid|rid|status' or 'pid|rid' to ArmKey."""
    parts = s.split("|")
    if len(parts) == 2:
        return ArmKey(parts[0], parts[1])
    if len(parts) == 3:
        return ArmKey(parts[0], parts[1], parts[2])
    return None


def _parse_response(obj: Any) -> ReflectorOutput:
    if not isinstance(obj, dict):
        return ReflectorOutput([], "")
    feats = obj.get("discriminator_features") or []
    if not isinstance(feats, list):
        feats = []
    text = str(obj.get("reflexion_text") or "")[:256]
    raw_boost = obj.get("suggested_exploration_boost") or {}
    boost: dict[ArmKey, float] = {}
    if isinstance(raw_boost, dict):
        for k, v in raw_boost.items():
            arm = _parse_arm_key(str(k))
            if arm is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            f = max(0.0, min(0.3, f))
            boost[arm] = f
    return ReflectorOutput(
        discriminator_features=[str(f) for f in feats],
        reflexion_text=text,
        suggested_exploration_boost=boost,
    )


class Reflector:
    def __init__(self, llm_timeout_s: float = 30.0) -> None:
        self.llm_timeout_s = llm_timeout_s

    async def reflect(self, contrast_payload: dict[str, Any]) -> ReflectorOutput | None:
        """Return ReflectorOutput, or None on failure (fallback)."""
        try:
            return await asyncio.wait_for(
                self._inner(contrast_payload), timeout=self.llm_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.info("Reflector timeout")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("Reflector error: %s", e)
            return None

    async def _inner(self, contrast_payload: dict[str, Any]) -> ReflectorOutput | None:
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
            )
            from openai import AsyncAzureOpenAI
            from .proposer import _breaker_open as _proposer_breaker_open
        except Exception as e:  # noqa: BLE001
            logger.info("Reflector deps missing: %s", e)
            return None
        # v604.1 share the proposer's circuit breaker (same TRAPI endpoint).
        if _proposer_breaker_open():
            logger.info("Reflector skipped: TRAPI circuit breaker active")
            return None

        cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

        def _token_provider() -> str:
            return cred.get_token("api://trapi/.default").token

        client = AsyncAzureOpenAI(
            azure_endpoint=_TRAPI_ENDPOINT,
            api_version=_API_VERSION,
            azure_ad_token_provider=_token_provider,
        )
        user_payload = json.dumps(contrast_payload, default=str)[:6000]
        for model in _MODEL_PREFERENCES:
            try:
                # response_format json_object unsupported on some TRAPI
                # deployments (see proposer.py): apply only to known-good
                # models, otherwise rely on system prompt + JSON extractor.
                kwargs = {"model": model, "messages": [
                    {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ], "temperature": 0.0}
                if "json" in model.lower() or model.endswith("-pro"):
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await client.chat.completions.create(**kwargs)
                from .proposer import _extract_json_block
                raw = resp.choices[0].message.content or "{}"
                obj = json.loads(_extract_json_block(raw))
                return _parse_response(obj)
            except Exception as e:  # noqa: BLE001
                logger.info("Reflector model %s failed: %s", model, e)
        return None


# =============================================================================
# v607 Phase 3 — stuck-trigger + verbal chid_template emit (Plan v607 rev E §3 B)
# =============================================================================

EMIT_SYSTEM_PROMPT = (
    "You are a skill-template synthesizer for an ARC-AGI-3 puzzle agent.\n"
    "The agent is STUCK: many turns without level advance. Existing skills\n"
    "in the library have failed. Propose a NEW skill template that targets\n"
    "a different mechanism class.\n\n"
    "OUTPUT strict JSON ONLY:\n"
    '  {"chid_template": str, "rationale": str}\n\n'
    "RULES:\n"
    "1. chid_template MUST follow form: P_<verb>_<noun>_<region_or_marker_placeholder>\n"
    "   Examples: P_color_pattern_C{m}, P_pixel_match_R{r}, P_edge_align_C{m}\n"
    "2. chid_template MUST NOT contain any of these tokens (anti-leak HARD set):\n"
    "   R28, R31, R12, gqb, bsT, Hkx, NTi, kCv, cwU, elp, Ycb\n"
    "3. chid_template MUST NOT contain cycle237 distinguishing tokens:\n"
    "   crop, compass, sweep, sector, alignment, lower, upper, toggle,\n"
    "   shared, blank, revisit, recently, unclicked, region, plus, nonmarker\n"
    "4. Prefer generic puzzle-mechanism verbs (color/pixel/edge/match/chain/\n"
    "   invert/repeat/scan/probe/test) over ft09-specific ones.\n"
    "5. rationale ≤ 200 chars explaining why this template is novel."
)


@dataclass
class StuckTriggerConfig:
    """v607 §3 Option B trigger + r3 anti-thrashing config."""
    best_advance_age_min: int = 5  # k in plan §3
    cooldown_after_fire: int = 5
    # ADAPTIVE_CAP formula bounds (codex r5 explicit numerics).
    severity_min: int = 0
    severity_max: int = 10


@dataclass
class StuckTriggerState:
    """Mutable per-episode trigger state."""
    last_fire_turn: int = -1
    fires_this_episode: int = 0
    cooldown_remaining: int = 0
    last_dominant_transition: tuple | None = None


def compute_stagnation_severity(turns_since_advance: int, cfg: StuckTriggerConfig | None = None) -> int:
    """Bounded int [severity_min, severity_max].

    severity_min default 0, severity_max default 10.
    """
    cfg = cfg or StuckTriggerConfig()
    raw = turns_since_advance - 3
    return max(cfg.severity_min, min(cfg.severity_max, raw))


def compute_adaptive_cap(episode_length: int, stagnation_severity: int) -> int:
    """Plan v607 rev E §3 Option B: ADAPTIVE_CAP = floor(min(0.4*ep, max(3, sev*2)))."""
    return int(min(0.4 * episode_length, max(3, stagnation_severity * 2)))


def stuck_fires(
    turns_since_advance: int,
    best_advance_age: int,
    episode_length: int,
    current_dt: tuple | None,
    state: StuckTriggerState,
    cfg: StuckTriggerConfig | None = None,
) -> bool:
    """Plan v607 rev E §3 Option B: full AND-OR rule.

    Fires when ALL:
      1. turns_since_advance > 3 (stagnation)
      2. cooldown_remaining == 0
      3. fires_this_episode < ADAPTIVE_CAP (adaptive cap, not fixed)
      4. best_advance_age >= cfg.best_advance_age_min (genuinely stuck, not transient)
    OR
      Episode delivered ≥1 new dominant_transition since last fire (positive
      evidence trigger — even with low turns_since_advance, novel observation
      warrants a Reflector consultation).
    """
    cfg = cfg or StuckTriggerConfig()
    severity = compute_stagnation_severity(turns_since_advance, cfg)
    cap = compute_adaptive_cap(episode_length, severity)
    if state.fires_this_episode >= cap:
        return False
    if state.cooldown_remaining > 0:
        return False
    # AND branch: stagnation + best_advance_age qualified
    stuck_ok = (
        turns_since_advance > 3
        and best_advance_age >= cfg.best_advance_age_min
    )
    # OR branch: new dominant_transition since last fire
    new_dt = (
        current_dt is not None
        and current_dt != state.last_dominant_transition
    )
    return stuck_ok or new_dt


def step_cooldown(state: StuckTriggerState) -> None:
    """Decrement cooldown each turn. Call before checking stuck_fires."""
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1


def record_fire(state: StuckTriggerState, turn: int, cfg: StuckTriggerConfig | None = None) -> None:
    """Update state after a successful Reflector emission."""
    cfg = cfg or StuckTriggerConfig()
    state.last_fire_turn = turn
    state.fires_this_episode += 1
    state.cooldown_remaining = cfg.cooldown_after_fire


def reset_episode(state: StuckTriggerState) -> None:
    state.last_fire_turn = -1
    state.fires_this_episode = 0
    state.cooldown_remaining = 0
    state.last_dominant_transition = None


@dataclass
class EmitResult:
    chid_template: str
    rationale: str
    tokens_consumed: int
    reject_reason: str = ""  # empty if accepted


async def emit_new_chid(
    contrast_payload: dict[str, Any],
    existing_templates: list[str],
    *,
    emit_callable: Callable[[dict, list[str]], dict | None] | None = None,
    llm_timeout_s: float = 30.0,
    max_retries: int = 2,
) -> EmitResult | None:
    """v607 Phase 3: stuck-triggered Reflector LLM call to emit chid_template.

    Args:
        contrast_payload: state snapshot (recent_turns, active_hypotheses,
            visible_regions) for the LLM to reason about.
        existing_templates: chid_template strings already in skill_state, to
            avoid duplicate emission.
        emit_callable: dependency-injection point for tests. If None, uses
            live TRAPI LLM. Tests pass a mock dict-returning callable.
        llm_timeout_s: per-call timeout.
        max_retries: anti-leak validator rejection retries (codex r3 R1).

    Returns EmitResult on accepted emission, None on hard failure.
    """
    from .anti_leak import validate_chid_template

    existing_set = set(existing_templates)
    last_reject = ""
    for attempt in range(max_retries + 1):
        if emit_callable is not None:
            try:
                obj = emit_callable(contrast_payload, list(existing_set))
            except Exception as e:  # noqa: BLE001
                logger.warning("emit_callable error: %s", e)
                return None
            tokens = int(obj.get("_tokens_consumed", 800)) if isinstance(obj, dict) else 0
        else:
            # Live TRAPI path
            obj, tokens = await _live_emit_call(contrast_payload, llm_timeout_s)
            if obj is None:
                return None
        chid = (obj or {}).get("chid_template", "")
        rationale = (obj or {}).get("rationale", "")
        if not isinstance(chid, str) or not chid:
            last_reject = "empty_template"
            continue
        # Substitute placeholder format check (codex r3 R1): must contain
        # at least one of {m}, {r}, or visible-id-like token.
        if chid in existing_set:
            last_reject = "duplicate"
            continue
        ok, reason = validate_chid_template(chid)
        if not ok:
            last_reject = reason
            logger.info(
                "Reflector emit rejected by anti_leak: %s (chid=%r, attempt %d/%d)",
                reason, chid, attempt + 1, max_retries + 1,
            )
            continue
        return EmitResult(
            chid_template=chid,
            rationale=rationale[:256],
            tokens_consumed=tokens,
        )
    return EmitResult(
        chid_template="", rationale="", tokens_consumed=0,
        reject_reason=last_reject,
    )


async def _live_emit_call(
    payload: dict, timeout_s: float,
) -> tuple[dict | None, int]:
    """Live TRAPI call to emit chid_template. Returns (obj, token_count)."""
    try:
        from azure.identity import (
            AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential,
        )
        from openai import AsyncAzureOpenAI
        from .proposer import _breaker_open as _proposer_breaker_open, _extract_json_block
    except Exception as e:  # noqa: BLE001
        logger.info("emit_new_chid deps missing: %s", e)
        return None, 0
    if _proposer_breaker_open():
        logger.info("emit_new_chid skipped: TRAPI breaker active")
        return None, 0
    cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())
    def _tp() -> str:
        return cred.get_token("api://trapi/.default").token
    client = AsyncAzureOpenAI(
        azure_endpoint=_TRAPI_ENDPOINT,
        api_version=_API_VERSION,
        azure_ad_token_provider=_tp,
    )
    user_payload = json.dumps(payload, default=str)[:4000]
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-5.4-mini_2026-03-17",
                messages=[
                    {"role": "system", "content": EMIT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.7,  # diversity for skill exploration
            ),
            timeout=timeout_s,
        )
        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(_extract_json_block(raw))
        tokens = getattr(resp.usage, "total_tokens", 800) if hasattr(resp, "usage") else 800
        return obj, int(tokens)
    except Exception as e:  # noqa: BLE001
        logger.info("emit_new_chid LLM call failed: %s", e)
        return None, 0

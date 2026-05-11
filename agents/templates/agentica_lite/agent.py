"""ArcgenticaLite — plan v601 rev C orchestrator.

Per-turn loop (≤ 60 s wall-clock per plan §1.14):
  Role 1 Proposer (LLM, conditional)  — warm-up / stalemate / new paired-cf
  Role 2 Policy   (deterministic)     — UCB1 + saturation arm-key + verdict
  Role 3 Memory writer (deterministic)— paired-cf detector + Reflector spawn

Backwards-compatible with the v600 smoke test: ArcgenticaLite(game_id, seed)
+ run_turn(state) + episode_end().
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_extender import ExtenderInput, LLMExtender
from .memory_journal import EpisodeRecord, MemoryJournal
from .memory_writer import MemoryWriter, Outcome
from .policy import (
    EpisodeState,
    compute_saturation_status,
    compute_verdict,
    record_outcome,
    select_arm,
)
from .predicate_library import PredicateLibrary
from .predicate_posterior import (
    ArmKey,
    PredicatePosterior,
    increment_emit_count,
    update_chid_posterior,
)
from .proposer import Proposer, ProposerOutput, ProposerResult
from .reflector import (
    Reflector,
    ReflectorOutput,
    StuckTriggerConfig,
    StuckTriggerState,
    emit_new_chid,
    record_fire,
    reset_episode,
    step_cooldown,
    stuck_fires,
)
from .skill_state import (
    EmitChidTemplatePatch,
    SkillState,
    SkillStateMetadata,
    apply_patch,
    load_state,
    save_state,
)
from .stalemate_trigger import StalemateConfig, StalemateTrigger

logger = logging.getLogger(__name__)


@dataclass
class Action:
    predicate_id: str
    region_id: str
    coord_xy: tuple[int, int]


def _coord_xy(c: Any) -> tuple[int, int]:
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return (int(c[0]), int(c[1]))
    return (-1, -1)


class ArcgenticaLite:
    FRAMEWORK_VERSION = "v601"

    def __init__(
        self,
        game_id: str,
        seed: int,
        rasi_arm: str = "null",
        stalemate_cfg: StalemateConfig | None = None,
        global_turn_budget_s: float = 60.0,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.rasi_arm = rasi_arm
        self.global_turn_budget_s = global_turn_budget_s
        self.posterior = PredicatePosterior()
        self.library = PredicateLibrary()
        self.stalemate = StalemateTrigger(stalemate_cfg)
        self.extender = LLMExtender()  # carry v600 for legacy paths
        self.proposer = Proposer()
        self.reflector = Reflector()
        self.memory_writer = MemoryWriter()
        self.journal = MemoryJournal(game_id)
        self.episode_state = EpisodeState()
        self.turns_since_L_plus = 0
        self.last_max_level = 0
        self._turn_latencies: list[float] = []
        self._turn_count = 0
        self._t_episode_start = time.time()
        self._stalemate_events = 0
        self._proposer_calls = 0
        self._reflector_calls = 0
        self._predicates_installed: list[dict] = []
        self._recent_failures: list[dict] = []
        self._last_visible_regions: list[Any] = []
        # Reflector boost lifecycle (plan §3.15: 5-turn lifetime).
        self._exploration_boost: dict[ArmKey, float] = {}
        self._boost_expiry_turn: int = -1
        # Last proposer output (used by next turn's Policy / verdict)
        self._last_proposer_output: ProposerOutput | None = None
        self._last_action_arm_key: ArmKey | None = None
        self._last_action_expected_signature: dict | None = None
        # v607 Phase 6: skill discovery + stuck trigger lifecycle.
        # skill_state.json is per-game; can be overridden via env var.
        self._skill_state_path = Path(os.environ.get(
            "ARC_LITE_SKILL_STATE_PATH",
            f"./skill_state_{game_id}.json",
        ))
        loaded = load_state(self._skill_state_path)
        self._skill_state = (
            loaded if loaded is not None
            else SkillState(metadata=SkillStateMetadata(game_id=game_id))
        )
        self._stuck_cfg = StuckTriggerConfig()
        self._stuck_state = StuckTriggerState()
        self._best_advance_turn = -1
        self._last_dominant_transition: str | None = None
        self._last_emitted_chid_template: str | None = None
        self._v607_emit_calls = 0
        self._v607_emit_accepted = 0
        self._v607_emit_rejected = 0

    @staticmethod
    def _visible_regions(state: Any) -> list[Any]:
        if state is None:
            return []
        if isinstance(state, dict):
            return state.get("visible_regions") or []
        return getattr(state, "visible_regions", []) or []

    @staticmethod
    def _last_observation(state: Any) -> dict[str, Any]:
        if state is None:
            return {}
        if isinstance(state, dict):
            return state.get("last_observation") or state.get("observation") or {}
        return (
            getattr(state, "last_observation", None)
            or getattr(state, "observation", None)
            or {}
        )

    @staticmethod
    def _markers(state: Any) -> list[dict]:
        if isinstance(state, dict):
            return state.get("marker_neighbor_states") or []
        return getattr(state, "marker_neighbor_states", None) or []

    def _max_posterior(self, visible_regions: list[Any]) -> float:
        if not visible_regions or not self.posterior.arms:
            return 0.0
        rids = {self.posterior._region_id(r) for r in visible_regions}
        rids.discard(None)
        if not rids:
            return 0.0
        best = 0.0
        for key, arm in self.posterior.arms.items():
            if key.region_id in rids:
                p = arm.alpha / (arm.alpha + arm.beta)
                if p > best:
                    best = p
        return best

    async def run_turn(self, state: Any) -> Action | None:
        t0 = time.monotonic()
        self._turn_count += 1
        deadline = t0 + self.global_turn_budget_s
        visible_regions = self._visible_regions(state)
        self._last_visible_regions = visible_regions
        obs = self._last_observation(state)

        # Update posterior + memory_writer for previous action's outcome.
        try:
            if self._last_action_arm_key is not None:
                verdict = compute_verdict(obs, self._last_action_expected_signature)
                record_outcome(self.posterior, self._last_action_arm_key, verdict)
                # Memory writer paired-cf detection for the previous coord.
                prev_coord = obs.get("coord")
                # Use the coord we issued previously (stored on Action), not the obs.
            self.posterior.update(obs)
            level_advanced = int(obs.get("level_delta", 0) or 0) > 0
            if level_advanced:
                self.turns_since_L_plus = 0
                self.last_max_level += 1
                self._best_advance_turn = self._turn_count
            else:
                self.turns_since_L_plus += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("posterior.update failed: %s", e)
            level_advanced = False

        # v607 Phase 6: update chid_template posterior with prev turn's verdict.
        if self._last_emitted_chid_template:
            try:
                delta = 1 if level_advanced else 0
                update_chid_posterior(
                    self._skill_state, self._last_emitted_chid_template, delta,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("update_chid_posterior failed: %s", e)
            self._last_emitted_chid_template = None

        # Expire reflector boost.
        if self._boost_expiry_turn >= 0 and self._turn_count > self._boost_expiry_turn:
            self._exploration_boost = {}
            self._boost_expiry_turn = -1

        # v607 Phase 6: stuck trigger + Reflector chid_template emit.
        # Fires only on genuine stagnation (per plan rev E §3 Option B adaptive cap).
        try:
            step_cooldown(self._stuck_state)
            dt_obs = obs.get("dominant_transition") or {}
            curr_dt = (
                f"{dt_obs.get('from')}->{dt_obs.get('to')}"
                if dt_obs else None
            )
            best_advance_age = self._turn_count - max(0, self._best_advance_turn)
            if stuck_fires(
                turns_since_advance=self.turns_since_L_plus,
                best_advance_age=best_advance_age,
                episode_length=30,
                current_dt=curr_dt,
                state=self._stuck_state,
                cfg=self._stuck_cfg,
            ):
                existing_templates = [
                    getattr(s, "chid_template", "")
                    for s in self._skill_state.skill_lifecycle
                    if getattr(s, "chid_template", "")
                ]
                vr_ids = [
                    (r.get("region_id") if isinstance(r, dict) else str(r))
                    for r in visible_regions[:10]
                ]
                contrast_payload = {
                    "turn": self._turn_count,
                    "turns_since_advance": self.turns_since_L_plus,
                    "dominant_transition": curr_dt,
                    "recent_diffs": self._recent_failures[-5:],
                    "visible_regions": [r for r in vr_ids if r],
                }
                self._v607_emit_calls += 1
                emit_res = await emit_new_chid(contrast_payload, existing_templates)
                if emit_res is not None and not emit_res.reject_reason and emit_res.chid_template:
                    patch = EmitChidTemplatePatch(
                        chid_template=emit_res.chid_template,
                        family="reflector",
                        description=emit_res.rationale[:200] if emit_res.rationale else "",
                        emit_run=str(self.seed),
                        emit_turn=self._turn_count,
                        emit_tokens=emit_res.tokens_consumed,
                        cooldown=self._stuck_cfg.cooldown_after_fire,
                    )
                    apply_patch(self._skill_state, patch)
                    record_fire(self._stuck_state, self._turn_count, self._stuck_cfg)
                    self._stuck_state.last_dominant_transition = curr_dt
                    save_state(self._skill_state, self._skill_state_path)
                    self._v607_emit_accepted += 1
                else:
                    self._v607_emit_rejected += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("v607 stuck-trigger emit failed: %s", e)

        # Proposer trigger evaluation (warm-up / stalemate). Paired-cf-driven proposer
        # calls happen via Memory writer below (after action verdict).
        proposer_output: ProposerOutput | None = None
        try:
            warm_up = self.stalemate.warm_up_fires(self._turn_count - 1)
            max_post = self._max_posterior(visible_regions)
            # v605 codex round 3: pass turn_count for periodic-trigger logic
            stalemate_fires = self.stalemate.fires(
                self.turns_since_L_plus, max_post, turn_count=self._turn_count - 1
            )
            if warm_up or stalemate_fires:
                # Build state dict for proposer
                state_for_prop = state if isinstance(state, dict) else {}
                visible_rids = [
                    (r.get("region_id") or r.get("id")) if isinstance(r, dict)
                    else (getattr(r, "region_id", None) or getattr(r, "id", None))
                    for r in visible_regions
                ]
                visible_rids = [r for r in visible_rids if r]
                time_left = max(1.0, deadline - time.monotonic() - 5.0)
                self.proposer.llm_timeout_s = min(self.proposer.llm_timeout_s, time_left)
                if (deadline - time.monotonic()) >= 50.0:
                    pres = await self.proposer.propose(state_for_prop, visible_rids)
                    self._proposer_calls += 1
                    if pres.failure_reason is None:
                        proposer_output = pres.output
                    else:
                        logger.info("proposer_failure reason=%s", pres.failure_reason)
                if warm_up:
                    self.stalemate.mark_warm_up_done()
                if stalemate_fires:
                    self._stalemate_events += 1
                    self.stalemate.mark_fired(turn_count=self._turn_count - 1)
        except Exception as e:  # noqa: BLE001
            logger.warning("proposer trigger path failed: %s", e)

        self._last_proposer_output = proposer_output

        # Policy decision.
        try:
            decision = select_arm(
                state if isinstance(state, dict) else {},
                self.posterior,
                self.library,
                proposer_output,
                self.episode_state,
                exploration_boost=self._exploration_boost or None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("policy.select_arm failed: %s", e)
            decision = None

        if decision is None:
            self._turn_latencies.append(time.monotonic() - t0)
            return None

        self.posterior.record_emission([decision.arm_key])
        self._last_action_arm_key = decision.arm_key
        # v607 Phase 6: identify chid_template family match for posterior tracking.
        if proposer_output is not None and proposer_output.candidate_predicate_id:
            pid = proposer_output.candidate_predicate_id
            for sk in self._skill_state.skill_lifecycle:
                tmpl = getattr(sk, "chid_template", "")
                if not tmpl:
                    continue
                tmpl_prefix = tmpl.split("{")[0] if "{" in tmpl else tmpl
                if tmpl_prefix and pid.startswith(tmpl_prefix):
                    try:
                        increment_emit_count(
                            self._skill_state, tmpl, self._turn_count,
                        )
                        self._last_emitted_chid_template = tmpl
                    except Exception:  # noqa: BLE001
                        pass
                    break
        self._last_action_expected_signature = (
            getattr(proposer_output, "expected_signature", None)
            if proposer_output is not None else None
        )

        # Memory writer outcome record (uses obs from THIS turn after action).
        # In a real env loop, the next turn's obs is the verdict; here we record the
        # current obs's level_delta + dt count as the outcome of the LAST chosen coord.
        try:
            dt = obs.get("dominant_transition") or {}
            outcome = Outcome(
                coord=_coord_xy(decision.coord_xy),
                primary_region_id=obs.get("primary_region_id"),
                level_delta=int(obs.get("level_delta") or 0),
                dt_count=int(dt.get("count", 0)),
                pre_state_features=self._snapshot_features(state),
                turn=self._turn_count,
            )
            mw_decision = self.memory_writer.record(outcome, current_turn=self._turn_count)
            if mw_decision.spawn_reflector and (deadline - time.monotonic()) >= 20.0:
                # Spawn Reflector (best-effort; failure is logged, not crashed)
                contrast = {
                    "coord": list(decision.coord_xy),
                    "outcomes": [
                        {
                            "level_delta": o.level_delta,
                            "dt_count": o.dt_count,
                            "pre_state_features": o.pre_state_features,
                        }
                        for o in (mw_decision.paired_cf_entry.outcomes
                                  if mw_decision.paired_cf_entry else [])
                    ],
                }
                self._reflector_calls += 1
                refl: ReflectorOutput | None = await self.reflector.reflect(contrast)
                if refl is not None and refl.suggested_exploration_boost:
                    # Apply boost for 5 turns (plan §3.15).
                    for arm, b in refl.suggested_exploration_boost.items():
                        self._exploration_boost[arm] = max(self._exploration_boost.get(arm, 0.0), b)
                    self._boost_expiry_turn = self._turn_count + 5
        except Exception as e:  # noqa: BLE001
            logger.warning("memory_writer / reflector path failed: %s", e)

        latency = time.monotonic() - t0
        if latency > self.global_turn_budget_s:
            logger.warning("wall_clock_breach: %.2fs", latency)
        self._turn_latencies.append(latency)
        return Action(decision.arm_key.predicate_id, decision.arm_key.region_id, decision.coord_xy)

    def _snapshot_features(self, state: Any) -> dict[str, Any]:
        feats: dict[str, Any] = {}
        for m in self._markers(state):
            mid = m.get("marker_id")
            if mid is None:
                continue
            clicked, _denom, _status = compute_saturation_status(m)
            feats[f"marker_{mid}_compass_saturation_numerator"] = clicked
        obs = self._last_observation(state)
        dt = obs.get("dominant_transition") or {}
        if dt:
            feats["recent_dominant_transition_direction"] = f"{dt.get('from')}->{dt.get('to')}"
        feats["level_delta"] = int(obs.get("level_delta") or 0)
        return feats

    async def _fire_extender(self, visible_regions: list[Any], deadline: float) -> None:
        """v600 legacy LLM extender path (kept for backward compat with existing rasi pipeline)."""
        time_left = max(1.0, deadline - time.monotonic() - 5.0)
        self.extender.llm_timeout_s = min(self.extender.llm_timeout_s, time_left)
        top10 = self.posterior.rank_top(visible_regions, self.library, k=10)
        try:
            out = await self.extender.propose(ExtenderInput(
                visible_regions=visible_regions,
                posterior_top10=top10,
                recent_failures=self._recent_failures,
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("extender.propose raised: %s", e)
            return
        if not out.accepted or not out.lambda_body:
            return
        try:
            res = self.library.install(
                out.lambda_body,
                predicate_id=out.name,
                family=out.family or "extended",
                coord_policy=out.coord_policy or "centroid",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("library.install raised: %s", e)
            return
        if not res.accepted:
            logger.info("install rejected: %s", res.reason)
            return
        installed_id = out.name or f"L_ext_{res.fingerprint}"
        rids = [r for r in (self.posterior._region_id(r) for r in visible_regions) if r is not None]
        self.posterior.bootstrap(installed_id, rids, anchor_alpha=1.0)
        self._predicates_installed.append(
            {"id": installed_id, "fingerprint": res.fingerprint, "reason": res.reason}
        )

    def episode_end(self) -> EpisodeRecord:
        lat = sorted(self._turn_latencies)
        n = len(lat)
        p50 = lat[n // 2] if n else 0.0
        p95 = lat[max(0, int(0.95 * n) - 1)] if n else 0.0
        worst = lat[-1] if n else 0.0
        top10 = (self.posterior.rank_top(self._last_visible_regions, self.library, k=10)
                 if self._last_visible_regions else [])
        rec = EpisodeRecord(
            episode_id=f"{self.game_id}_seed{self.seed}_t{int(self._t_episode_start)}",
            seed=self.seed,
            framework_version=self.FRAMEWORK_VERSION,
            git_sha="unknown",
            ts_start=self._t_episode_start,
            ts_end=time.time(),
            turns=self._turn_count,
            max_level=self.last_max_level,
            rasi_arm=self.rasi_arm,
            predicates_installed=list(self._predicates_installed),
            stalemate_events=self._stalemate_events,
            posterior_top10_at_end=top10,
            latency_p50=p50,
            latency_p95=p95,
            latency_worst=worst,
            extra={
                "proposer_calls": self._proposer_calls,
                "reflector_calls": self._reflector_calls,
                "confidence_overrides": self.episode_state.confidence_override_count,
                # v607 Phase 6 telemetry
                "v607_emit_calls": self._v607_emit_calls,
                "v607_emit_accepted": self._v607_emit_accepted,
                "v607_emit_rejected": self._v607_emit_rejected,
                "v607_skill_count": len(self._skill_state.skill_lifecycle),
            },
        )
        # v607 Phase 6: persist final skill_state + reset stuck trigger.
        try:
            save_state(self._skill_state, self._skill_state_path)
            reset_episode(self._stuck_state)
        except Exception as e:  # noqa: BLE001
            logger.warning("v607 episode_end persist failed: %s", e)
        self.journal.append(rec)
        return rec

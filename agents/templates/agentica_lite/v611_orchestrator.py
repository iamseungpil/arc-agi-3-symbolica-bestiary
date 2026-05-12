"""v611 orchestrator (Plan rev D §Step 2b, codex round 11-13 ACCEPT).

Manages anchor-streak + spawn-fresh state across turns. Independent
of LLM client — the role runners are injected as callables.

Mechanical separation enforcement:
- 3 distinct calls per turn (M1, M2v, M2e)
- Each call receives only its scoped input dict
- No shared transcript object across roles
- reject_replan: minimal rejection_reason added to M1 retry
  (machine-formatted, max 100 chars; codex round 12 Q2)
- reject_anchor: AnchorCounter records summary; next turn's M1 sees
  ONLY that summary (no transcript carry; codex round 12 Q4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .v611_schemas import (
    M2V_VERDICTS,
    m2e_is_substitute,
    validate_m1_proposer_output,
    validate_m2e_executor_output,
    validate_m2v_verifier_output,
)
from .v611_telemetry import log_turn_event


# ─────────────────────────────────────────────────────────────────
# Structured state summary for M2v Verifier (round 15 fix)
# ─────────────────────────────────────────────────────────────────


# Length contracts (round 18 enforcement):
MAX_STATE_TEXT_CHARS = 4000
MAX_STATE_NOW_CHARS = 200
MAX_SKILL_MD_SUMMARY_CHARS = 2000


def _clamp(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class StateSummary:
    """Structured summary passed to M2v Verifier.

    Plan rev D round 15-18: M2v's anchor-detection thresholds require
    explicit failure-history fields. state_now clamped to 200 chars.
    """

    state_now: str = ""
    last_strategies: list[dict] = field(default_factory=list)
    repeat_axis_count: int = 0
    null_effect_streak: int = 0

    def render(self) -> str:
        """Render to the multi-line text format the M2v prompt expects."""
        clamped_state = _clamp(self.state_now, MAX_STATE_NOW_CHARS)
        lines = [f"state_now: {clamped_state}"]
        lines.append("last_5_strategies:")
        for i, s in enumerate(self.last_strategies[:5], 1):
            text = _clamp(str(s.get("text", "")), 80)
            verdict = s.get("verdict", "?")
            changed = s.get("frame_changed", "?")
            lines.append(f"  {i}. \"{text}\" verdict={verdict} "
                          f"frame_changed={changed}")
        lines.append(f"repeat_axis_count: {self.repeat_axis_count}")
        lines.append(f"null_effect_streak: {self.null_effect_streak}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# AnchorCounter — fresh-spawn signal for next-turn M1
# ─────────────────────────────────────────────────────────────────


@dataclass
class AnchorCounter:
    """Tracks reject_anchor verdicts; on consume, returns fresh summary."""

    streak: int = 0
    pending_summary: str | None = None

    def on_reject_anchor(self, m1_summary: str) -> None:
        """Verifier returned reject_anchor; queue summary for next turn."""
        self.streak += 1
        self.pending_summary = m1_summary

    def consume_for_fresh_spawn(self) -> str | None:
        """Called at next turn start. Returns summary (or None) and clears."""
        s = self.pending_summary
        self.pending_summary = None
        return s

    def on_non_anchor_verdict(self) -> None:
        """Any verdict other than reject_anchor → reset streak."""
        self.streak = 0


# ─────────────────────────────────────────────────────────────────
# Role-runner protocol (injected by agent.py; tests can mock)
# ─────────────────────────────────────────────────────────────────


class M1ProposerCallable(Protocol):
    def __call__(self, state_text: str, skill_md_summary: str,
                 anchor_summary: str | None = None,
                 rejection_reason: str | None = None) -> dict[str, Any]: ...


class M2vVerifierCallable(Protocol):
    def __call__(self, proposer_out: dict[str, Any],
                 state_text_summary: str | StateSummary) -> dict[str, Any]: ...


class M2eExecutorCallable(Protocol):
    def __call__(self, approved_out: dict[str, Any],
                 png_bytes: bytes) -> dict[str, Any]: ...


# ─────────────────────────────────────────────────────────────────
# TurnResult
# ─────────────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    turn_id: int
    success: bool                       # an executor click happened
    skip_reason: str | None = None      # 'm1_invalid', 'm2v_invalid',
                                          # 'reject_anchor', 'second_reject',
                                          # 'm2e_invalid'
    proposer_out: dict[str, Any] | None = None
    verifier_out: dict[str, Any] | None = None
    executor_out: dict[str, Any] | None = None
    click_xy: tuple[int, int] | None = None


def _normalize_rejection_reason(reason_nl: str) -> str:
    """Codex round 12 Q2: rejection_reason must be a minimal machine-
    formatted constraint, max 100 chars. Treats verifier reason as
    a single hint, not a transcript backdoor.
    """
    s = (reason_nl or "").strip().replace("\n", " ")
    if len(s) > 100:
        s = s[:97] + "..."
    return f"avoid:{s}"


def run_v611_turn(
    *,
    turn_id: int,
    state_text: str,
    state_text_summary: str | StateSummary,
    png_bytes: bytes,
    skill_md_summary: str,
    anchor: AnchorCounter,
    m1_proposer: M1ProposerCallable,
    m2v_verifier: M2vVerifierCallable,
    m2e_executor: M2eExecutorCallable,
    seed: int | None = None,
    episode_id: str | None = None,
) -> TurnResult:
    """Execute one v611+Δ7 turn with mechanical role separation.

    Returns TurnResult describing the outcome. Telemetry written to
    `logs/v611_turn_telemetry.jsonl` per Plan rev D §Q6.

    Round 16 fix: if `state_text_summary` is a StateSummary instance,
    it is rendered via .render() before passing to M2v (per M2v prompt
    contract). Plain strings still accepted for backwards compatibility.
    """
    # Render StateSummary to structured multi-line text if provided.
    if isinstance(state_text_summary, StateSummary):
        state_text_summary_str = state_text_summary.render()
    else:
        state_text_summary_str = state_text_summary
    # Round 18 enforcement: clamp lengths before passing to LLM roles.
    state_text = _clamp(state_text, MAX_STATE_TEXT_CHARS)
    skill_md_summary = _clamp(skill_md_summary, MAX_SKILL_MD_SUMMARY_CHARS)

    def _log(role: str, event: str, payload: dict[str, Any]) -> None:
        log_turn_event(turn_id=turn_id, role=role, event=event,
                       payload=payload, seed=seed, episode_id=episode_id)

    # ───── Role 1: M1 Proposer (with anchor-fresh-spawn) ─────
    anchor_summary = anchor.consume_for_fresh_spawn()
    m1_out = m1_proposer(state_text=state_text,
                          skill_md_summary=skill_md_summary,
                          anchor_summary=anchor_summary)
    _log("m1", "role_returned",
         {"anchor_summary_used": anchor_summary is not None,
          "output_keys": list(m1_out.keys()) if isinstance(m1_out, dict)
                          else None})
    v1 = validate_m1_proposer_output(m1_out)
    _log("m1", "validator_ok" if v1.ok else "validator_fail",
         {"violations": v1.violations})
    if not v1.ok:
        # one retry
        m1_out = m1_proposer(state_text=state_text,
                              skill_md_summary=skill_md_summary,
                              anchor_summary=anchor_summary)
        _log("m1", "role_returned", {"retry": True})
        v1 = validate_m1_proposer_output(m1_out)
        _log("m1", "validator_ok" if v1.ok else "validator_fail",
             {"violations": v1.violations, "retry": True})
        if not v1.ok:
            return TurnResult(turn_id=turn_id, success=False,
                              skip_reason="m1_invalid")

    # ───── Role 2: M2v Verifier (separate context, state_text_summary
    # only — NO frame, NO skill_md) ─────
    m2v_out = m2v_verifier(proposer_out=m1_out,
                            state_text_summary=state_text_summary_str)
    _log("m2v", "role_returned",
         {"verdict": (m2v_out or {}).get("verdict")
                       if isinstance(m2v_out, dict) else None})
    v2 = validate_m2v_verifier_output(m2v_out)
    _log("m2v", "validator_ok" if v2.ok else "validator_fail",
         {"violations": v2.violations})
    if not v2.ok:
        m2v_out = m2v_verifier(proposer_out=m1_out,
                                state_text_summary=state_text_summary_str)
        v2 = validate_m2v_verifier_output(m2v_out)
        _log("m2v", "validator_ok" if v2.ok else "validator_fail",
             {"violations": v2.violations, "retry": True})
        if not v2.ok:
            return TurnResult(turn_id=turn_id, success=False,
                              skip_reason="m2v_invalid",
                              proposer_out=m1_out)

    verdict = m2v_out["verdict"]
    assert verdict in M2V_VERDICTS  # validator guarantees

    if verdict == "reject_replan":
        # retry M1 with normalized rejection_reason
        reason = _normalize_rejection_reason(m2v_out.get("reason_nl", ""))
        m1_out = m1_proposer(state_text=state_text,
                              skill_md_summary=skill_md_summary,
                              anchor_summary=anchor_summary,
                              rejection_reason=reason)
        _log("m1", "role_returned", {"replan": True, "reason": reason})
        v1b = validate_m1_proposer_output(m1_out)
        if not v1b.ok:
            return TurnResult(turn_id=turn_id, success=False,
                              skip_reason="m1_replan_invalid",
                              proposer_out=m1_out, verifier_out=m2v_out)
        m2v_out = m2v_verifier(proposer_out=m1_out,
                                state_text_summary=state_text_summary_str)
        v2b = validate_m2v_verifier_output(m2v_out)
        _log("m2v", "role_returned",
             {"replan_followup": True,
              "verdict": (m2v_out or {}).get("verdict")
                           if isinstance(m2v_out, dict) else None})
        if not v2b.ok or m2v_out["verdict"] != "approve":
            return TurnResult(turn_id=turn_id, success=False,
                              skip_reason="second_reject",
                              proposer_out=m1_out, verifier_out=m2v_out)
        verdict = "approve"

    if verdict == "reject_anchor":
        # Queue summary for NEXT turn's fresh M1 spawn.
        summary = (m1_out.get("nl_strategy", "") or "")[:200]
        anchor.on_reject_anchor(summary)
        _log("anchor", "reject_anchor",
             {"streak": anchor.streak,
              "summary_len": len(summary)})
        return TurnResult(turn_id=turn_id, success=False,
                          skip_reason="reject_anchor",
                          proposer_out=m1_out, verifier_out=m2v_out)

    assert verdict == "approve"
    anchor.on_non_anchor_verdict()

    # ───── Role 3: M2e Executor (PNG visual + approved NL only) ─────
    m2e_out = m2e_executor(approved_out=m1_out, png_bytes=png_bytes)
    _log("m2e", "role_returned",
         {"output_keys": list(m2e_out.keys()) if isinstance(m2e_out, dict)
                          else None})
    v3 = validate_m2e_executor_output(m2e_out)
    _log("m2e", "validator_ok" if v3.ok else "validator_fail",
         {"violations": v3.violations})
    if not v3.ok:
        m2e_out = m2e_executor(approved_out=m1_out, png_bytes=png_bytes)
        v3 = validate_m2e_executor_output(m2e_out)
        _log("m2e", "validator_ok" if v3.ok else "validator_fail",
             {"violations": v3.violations, "retry": True})
        if not v3.ok:
            return TurnResult(turn_id=turn_id, success=False,
                              skip_reason="m2e_invalid",
                              proposer_out=m1_out, verifier_out=m2v_out,
                              executor_out=m2e_out)

    # Round 16: emit SUBSTITUTE drift telemetry if M2e flagged it.
    if m2e_is_substitute(m2e_out.get("grounding_text", "")):
        _log("m2e", "substitute_drift",
             {"grounding_text_prefix":
                  m2e_out["grounding_text"][:80]})

    coord = tuple(int(c) for c in m2e_out["click_xy_hint"])
    return TurnResult(turn_id=turn_id, success=True,
                      proposer_out=m1_out, verifier_out=m2v_out,
                      executor_out=m2e_out, click_xy=coord)

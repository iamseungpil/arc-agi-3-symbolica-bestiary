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
    validate_m1_proposer_output,
    validate_m2e_executor_output,
    validate_m2v_verifier_output,
)
from .v611_telemetry import log_turn_event


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
                 state_text_summary: str) -> dict[str, Any]: ...


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
    state_text_summary: str,
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
    """

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
                            state_text_summary=state_text_summary)
    _log("m2v", "role_returned",
         {"verdict": (m2v_out or {}).get("verdict")
                       if isinstance(m2v_out, dict) else None})
    v2 = validate_m2v_verifier_output(m2v_out)
    _log("m2v", "validator_ok" if v2.ok else "validator_fail",
         {"violations": v2.violations})
    if not v2.ok:
        m2v_out = m2v_verifier(proposer_out=m1_out,
                                state_text_summary=state_text_summary)
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
                                state_text_summary=state_text_summary)
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

    coord = tuple(int(c) for c in m2e_out["click_xy_hint"])
    return TurnResult(turn_id=turn_id, success=True,
                      proposer_out=m1_out, verifier_out=m2v_out,
                      executor_out=m2e_out, click_xy=coord)

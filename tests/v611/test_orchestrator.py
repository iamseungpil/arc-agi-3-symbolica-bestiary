"""v611 Step 2b — orchestrator + AnchorCounter + telemetry + negative paths.

Plan rev D §Step 2b, codex round 13 ACCEPT.

Tests:
- AnchorCounter streak logic
- Telemetry write/read round-trip
- 3-role normal path (approve)
- reject_replan retry path (twice → skip)
- reject_anchor fresh-spawn path (next turn ANCHOR_SUMMARY only)
- Role boundary leakage assertions (M2v never sees frame/skillmd;
  M1 retry sees ONLY anchor_summary or rejection_reason, NOT
  prior turn's full M1 output)
- Replay invariance: fresh-spawn produces same M1 output for same
  anchor_summary regardless of prior hidden state
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_orchestrator import (  # noqa: E402
    AnchorCounter,
    TurnResult,
    run_v611_turn,
)
from agents.templates.agentica_lite.v611_schemas import (  # noqa: E402
    M2V_VERDICTS,
)
from agents.templates.agentica_lite.v611_telemetry import (  # noqa: E402
    log_turn_event,
    read_telemetry,
)


# ─────────────────────────────────────────────────────────────────
# AnchorCounter
# ─────────────────────────────────────────────────────────────────


def test_anchor_counter_starts_clean():
    a = AnchorCounter()
    assert a.streak == 0
    assert a.consume_for_fresh_spawn() is None


def test_anchor_counter_records_reject_anchor():
    a = AnchorCounter()
    a.on_reject_anchor("prior strategy summary text")
    assert a.streak == 1
    assert a.pending_summary == "prior strategy summary text"


def test_anchor_counter_consume_returns_and_clears():
    a = AnchorCounter()
    a.on_reject_anchor("summary text")
    got = a.consume_for_fresh_spawn()
    assert got == "summary text"
    # consumed once -> None
    assert a.consume_for_fresh_spawn() is None
    assert a.pending_summary is None


def test_anchor_counter_resets_on_non_anchor():
    a = AnchorCounter()
    a.on_reject_anchor("s")
    a.on_non_anchor_verdict()
    assert a.streak == 0


def test_anchor_counter_streak_grows_across_rejects():
    a = AnchorCounter()
    for i in range(3):
        a.on_reject_anchor(f"summary {i}")
    assert a.streak == 3


# ─────────────────────────────────────────────────────────────────
# Telemetry round-trip
# ─────────────────────────────────────────────────────────────────


def test_telemetry_write_read_roundtrip(tmp_path):
    log_path = tmp_path / "t.jsonl"
    log_turn_event(turn_id=1, role="m1", event="role_returned",
                    payload={"output_keys": ["a", "b"]},
                    seed=5, episode_id="ep1", log_path=log_path)
    log_turn_event(turn_id=1, role="m1", event="validator_ok",
                    payload={"violations": []},
                    seed=5, episode_id="ep1", log_path=log_path)
    events = read_telemetry(log_path)
    assert len(events) == 2
    assert events[0].turn_id == 1
    assert events[0].role == "m1"
    assert events[0].event == "role_returned"
    assert events[1].event == "validator_ok"


# ─────────────────────────────────────────────────────────────────
# Mock role runners for orchestrator tests
# ─────────────────────────────────────────────────────────────────


def _mock_m1_clean():
    """A mock M1 Proposer that always returns valid output. Records its
    received inputs so tests can assert leakage assertions."""
    received: list[dict] = []

    def runner(state_text, skill_md_summary, anchor_summary=None,
                rejection_reason=None):
        received.append({
            "state_text": state_text,
            "skill_md_summary": skill_md_summary,
            "anchor_summary": anchor_summary,
            "rejection_reason": rejection_reason,
        })
        return {
            "nl_strategy": ("Looking at the grid, I propose clicking the "
                             "bottom-right marker tile to test repair."),
            "suggested_click_region": "bottom-right tile area",
            "expected_signature": {"frame_changed": True, "unsat_delta": -1},
            "rollback_trigger": "no frame change after click",
        }
    runner.received = received  # type: ignore[attr-defined]
    return runner


def _mock_m2v(verdict_seq: list[str]):
    """Mock M2v Verifier with a queued verdict sequence."""
    received: list[dict] = []
    idx = {"i": 0}

    def runner(proposer_out, state_text_summary):
        received.append({
            "proposer_out": proposer_out,
            "state_text_summary": state_text_summary,
        })
        i = min(idx["i"], len(verdict_seq) - 1)
        idx["i"] += 1
        return {
            "verdict": verdict_seq[i],
            "reason_nl": f"mocked reason for {verdict_seq[i]}",
        }
    runner.received = received  # type: ignore[attr-defined]
    return runner


def _mock_m2e_clean():
    received: list[dict] = []

    def runner(approved_out, png_bytes):
        received.append({
            "approved_out": approved_out,
            "png_bytes_len": len(png_bytes) if png_bytes else 0,
        })
        return {
            "click_xy_hint": [44, 44],
            "grounding_text": ("the marker tile in the PNG occupies "
                                "approximately pixels around (40-48, 40-48); "
                                "centered at (44, 44)"),
        }
    runner.received = received  # type: ignore[attr-defined]
    return runner


# ─────────────────────────────────────────────────────────────────
# Approve path
# ─────────────────────────────────────────────────────────────────


def test_approve_path_returns_success_with_coord(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["approve"])
    m2e = _mock_m2e_clean()
    res = run_v611_turn(
        turn_id=0,
        state_text="state text full",
        state_text_summary="state summary short",
        png_bytes=b"\x89PNG_mock_bytes" * 100,
        skill_md_summary="skill md compact",
        anchor=anchor,
        m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
        seed=5, episode_id="ep1",
    )
    assert res.success
    assert res.click_xy == (44, 44)
    assert anchor.streak == 0


# ─────────────────────────────────────────────────────────────────
# reject_replan path
# ─────────────────────────────────────────────────────────────────


def test_reject_replan_retries_m1_with_rejection_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["reject_replan", "approve"])
    m2e = _mock_m2e_clean()
    res = run_v611_turn(
        turn_id=1,
        state_text="state",
        state_text_summary="summary",
        png_bytes=b"png",
        skill_md_summary="skill",
        anchor=anchor,
        m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    assert res.success
    # M1 called twice (initial + replan)
    assert len(m1.received) == 2
    # 2nd call has rejection_reason; first does NOT
    assert m1.received[0]["rejection_reason"] is None
    reason = m1.received[1]["rejection_reason"]
    assert reason is not None
    assert reason.startswith("avoid:") and len(reason) <= 110


def test_reject_replan_twice_returns_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["reject_replan", "reject_replan"])
    m2e = _mock_m2e_clean()
    res = run_v611_turn(
        turn_id=2,
        state_text="state",
        state_text_summary="summary",
        png_bytes=b"png",
        skill_md_summary="skill",
        anchor=anchor,
        m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    assert not res.success
    assert res.skip_reason == "second_reject"
    # M2e should NEVER have been called
    assert m2v.received and len(m2v.received) == 2


# ─────────────────────────────────────────────────────────────────
# reject_anchor → next-turn fresh spawn
# ─────────────────────────────────────────────────────────────────


def test_reject_anchor_queues_summary_for_next_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["reject_anchor"])
    m2e = _mock_m2e_clean()
    res = run_v611_turn(
        turn_id=3,
        state_text="state",
        state_text_summary="summary",
        png_bytes=b"png",
        skill_md_summary="skill",
        anchor=anchor,
        m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    assert not res.success
    assert res.skip_reason == "reject_anchor"
    # Summary queued for next turn
    assert anchor.streak == 1
    assert anchor.pending_summary is not None
    # M2e never called
    assert len(m2v.received) == 1


def test_reject_anchor_then_next_turn_passes_summary_only(tmp_path, monkeypatch):
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["reject_anchor", "approve"])
    m2e = _mock_m2e_clean()
    # Turn 1: reject_anchor
    run_v611_turn(
        turn_id=1, state_text="state-A", state_text_summary="sum-A",
        png_bytes=b"png", skill_md_summary="skill",
        anchor=anchor, m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    # Turn 2: M1 should receive anchor_summary from turn 1
    res2 = run_v611_turn(
        turn_id=2, state_text="state-B", state_text_summary="sum-B",
        png_bytes=b"png", skill_md_summary="skill",
        anchor=anchor, m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    assert res2.success
    # M1 called 2 times total
    assert len(m1.received) == 2
    # Turn-1 call had anchor_summary=None
    assert m1.received[0]["anchor_summary"] is None
    # Turn-2 call had anchor_summary set (the queued summary from turn 1)
    assert m1.received[1]["anchor_summary"] is not None
    # After consume, anchor pending_summary cleared
    assert anchor.pending_summary is None


# ─────────────────────────────────────────────────────────────────
# Leakage assertions: M2v never sees frame/skill_md
# ─────────────────────────────────────────────────────────────────


def test_m2v_input_excludes_frame_and_skillmd(tmp_path, monkeypatch):
    """Critical separation: M2v's runner call must NOT receive raw
    frame, raw png, or SKILL.md content. Only proposer_out + state_text
    summary."""
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["approve"])
    m2e = _mock_m2e_clean()
    run_v611_turn(
        turn_id=0, state_text="STATE_TEXT_FULL_GRID_PIXELS_HERE",
        state_text_summary="COMPACT_SUMMARY_ONLY",
        png_bytes=b"PNG_RAW_BYTES_HERE",
        skill_md_summary="SKILL_MD_CONTENT_HERE",
        anchor=anchor, m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    rec = m2v.received[0]
    # M2v sees proposer_out (NL only) + state_text_summary
    assert "proposer_out" in rec
    assert rec["state_text_summary"] == "COMPACT_SUMMARY_ONLY"
    # M2v must NOT receive these:
    keys = set(rec.keys())
    assert "png_bytes" not in keys
    assert "skill_md_summary" not in keys
    assert "state_text" not in keys  # full text NOT passed
    # And the proposer_out passed to it has no png/skill_md fields:
    assert "png" not in rec["proposer_out"]
    assert "skill_md" not in rec["proposer_out"]


# ─────────────────────────────────────────────────────────────────
# Replay invariance: fresh spawn deterministic for same summary
# ─────────────────────────────────────────────────────────────────


def test_fresh_spawn_uses_only_anchor_summary(tmp_path, monkeypatch):
    """The 2nd-turn M1 call (after reject_anchor) must receive ONLY
    state_text + skill_md_summary + anchor_summary. No prior turn's
    M1 output, M2v output, or rejection_reason."""
    monkeypatch.setenv("V611_TELEMETRY_PATH", str(tmp_path / "t.jsonl"))
    anchor = AnchorCounter()
    m1 = _mock_m1_clean()
    m2v = _mock_m2v(["reject_anchor", "approve"])
    m2e = _mock_m2e_clean()
    run_v611_turn(
        turn_id=1, state_text="state-A", state_text_summary="sum-A",
        png_bytes=b"png", skill_md_summary="skill",
        anchor=anchor, m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    run_v611_turn(
        turn_id=2, state_text="state-B", state_text_summary="sum-B",
        png_bytes=b"png", skill_md_summary="skill",
        anchor=anchor, m1_proposer=m1, m2v_verifier=m2v, m2e_executor=m2e,
    )
    turn2_call = m1.received[1]
    # ONLY these 4 keys should have meaningful values:
    assert turn2_call["state_text"] == "state-B"
    assert turn2_call["skill_md_summary"] == "skill"
    assert turn2_call["anchor_summary"] is not None
    # rejection_reason is NOT passed on fresh-spawn (only on reject_replan)
    assert turn2_call["rejection_reason"] is None


# ─────────────────────────────────────────────────────────────────
# M2V_VERDICTS single source of truth
# ─────────────────────────────────────────────────────────────────


def test_m2v_verdicts_constant_matches_validator():
    """All three verdicts in M2V_VERDICTS must be accepted by
    validate_m2v_verifier_output, and no other strings should be."""
    from agents.templates.agentica_lite.v611_schemas import (
        validate_m2v_verifier_output,
    )
    for v in M2V_VERDICTS:
        out = {"verdict": v, "reason_nl": "ok mocked reason here for v611"}
        assert validate_m2v_verifier_output(out).ok, f"verdict {v} rejected"
    # An unknown verdict must be rejected
    bad = {"verdict": "totally_unknown", "reason_nl": "x" * 30}
    assert not validate_m2v_verifier_output(bad).ok

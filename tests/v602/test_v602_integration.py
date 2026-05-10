"""v602 integration tests (plan §5).

INT07 — render integrity round-trip
INT08 — cross-run merge + 1A promotion via Reflector PromoteA1Patch
INT09 — prompt-injection cap (oversized state -> capped prompt context <= 4 KB)

100% deterministic. Each test loads its fixture, exercises the v602 substrate,
and asserts the fixture's `expected` block.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.skill_md_renderer import (  # noqa: E402
    AUTO_GENERATED_HEADER, PROMPT_CONTEXT_BUDGET_BYTES,
    render, render_prompt_context,
)
from agents.templates.agentica_lite.skill_state import (  # noqa: E402
    ConfirmedMechanic, Falsification, PromoteA1Patch, SkillMetrics,
    SkillRecord, SkillState, SkillStateMetadata, apply_patch,
    atomic_write_text, load_state, save_state,
)
from tools.migrate_cross_run_memory import migrate  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _load_fix(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# =============================================================================
# INT07 — Render integrity round-trip
# =============================================================================


def test_int07_render_integrity_roundtrip(tmp_path):
    """Build skill_state from fixture, render, write, reload, render again -> identical."""
    fx = _load_fix("INT07_render_integrity.json")
    state = SkillState.from_dict(fx["input"])

    # Stage 1: render once
    md1 = render(state)
    assert md1.startswith(AUTO_GENERATED_HEADER)

    # Stage 2: write atomically + reload + re-render
    state_path = tmp_path / "skill_state.json"
    md_path = tmp_path / "SKILL.md"
    save_state(state, state_path)  # writes JSON
    atomic_write_text(md_path, md1)

    reloaded = load_state(state_path)
    assert reloaded is not None
    md2 = render(reloaded)

    # Idempotency invariant: render-write-reload-render is byte-identical.
    # NOTE: save_state calls touch() which updates last_updated. To compare
    # apples-to-apples, we render the same `state` (whose last_updated was
    # mutated by save) and compare against md2 (rendered from the reloaded
    # state, which has the same updated timestamp serialized to disk).
    md_post_save = render(state)
    assert md2 == md_post_save, "reload-render must equal post-save render byte-for-byte"

    # Independent invariant: the SKILL.md file we wrote pre-save matches its
    # original render output (the file content hasn't drifted on disk).
    written_md = md_path.read_text(encoding="utf-8")
    assert written_md == md1


def test_int07_render_deterministic_ordering():
    """Ordering invariants for 1A / 1B / F / S sections are stable."""
    fx = _load_fix("INT07_render_integrity.json")
    state = SkillState.from_dict(fx["input"])
    md = render(state)
    # 1A section: A001 should come before A002 (confirmation_count 3 > 2).
    i_a001 = md.index("[A001]")
    i_a002 = md.index("[A002]")
    assert i_a001 < i_a002


def test_int07_render_contains_all_sections():
    """All five sections (1A, 1B, F, S.static, S.dynamic) present in render."""
    fx = _load_fix("INT07_render_integrity.json")
    state = SkillState.from_dict(fx["input"])
    md = render(state)
    for marker in ("## 1A", "## 1B", "## F", "### S.static", "### S.dynamic"):
        assert marker in md, f"missing section: {marker}"
    # Auto-generated header banner present
    assert AUTO_GENERATED_HEADER in md


# =============================================================================
# INT08 — Cross-run merge + 1A promotion
# =============================================================================


def test_int08_cross_run_merge_and_promotion(tmp_path):
    """2 paired_cf entries -> Reflector promote_1A patch -> applier accepts -> 1A entry."""
    fx = _load_fix("INT08_cross_run_merge.json")
    state = SkillState.from_dict(fx["input"]["starting_state"])
    patch_data = fx["input"]["promote_patch"]
    expected = fx["expected"]

    patch = PromoteA1Patch(
        natural_language=patch_data["natural_language"],
        pcf_offsets=patch_data["pcf_offsets"],
        evidence_runs=patch_data["evidence_runs"],
        first_seen_run=patch_data["first_seen_run"],
        first_seen_turn=patch_data["first_seen_turn"],
    )
    res = apply_patch(state, patch)
    assert res.accepted is True
    assert res.reason == "ok"

    assert len(state.confirmed_mechanics) == expected["confirmed_mechanics_count"]
    cm = state.confirmed_mechanics[0]
    assert cm.confirmation_count == expected["first_mechanic_confirmation_count"]
    assert cm.paired_cf_offsets == expected["first_mechanic_pcf_offsets"]
    assert cm.id == "A001"

    # Render produces a SKILL.md citing A001 + both pcf offsets
    md = render(state)
    assert "[A001]" in md
    if expected.get("skill_md_contains_a001"):
        assert "pcf_offsets: [42, 91]" in md


def test_int08_promote_rejected_with_single_offset(tmp_path):
    """PromoteA1Patch with len(pcf_offsets) < 2 is rejected."""
    state = SkillState(metadata=SkillStateMetadata(game_id="reject-test"))
    bad = PromoteA1Patch(
        natural_language="Single-evidence promotion attempt",
        pcf_offsets=[42],  # only 1 offset
        evidence_runs=["cycle237"],
        first_seen_run="cycle237",
        first_seen_turn=27,
    )
    res = apply_patch(state, bad)
    assert res.accepted is False
    assert res.reason == "promote_insufficient_evidence"
    assert len(state.confirmed_mechanics) == 0


def test_int08_promote_rejected_with_duplicate_offsets(tmp_path):
    """codex final-gate Q5: duplicate paired_cf offsets count as weak evidence
    even when len(pcf_offsets) >= 2."""
    state = SkillState(metadata=SkillStateMetadata(game_id="dup-evidence"))
    bad = PromoteA1Patch(
        natural_language="Two offsets but they cite the same paired_cf entry",
        pcf_offsets=[42, 42],  # duplicate citation, len==2 but distinct==1
        evidence_runs=["cycle237"],
        first_seen_run="cycle237",
        first_seen_turn=27,
    )
    res = apply_patch(state, bad)
    assert res.accepted is False
    assert res.reason == "promote_insufficient_distinct_evidence"
    assert len(state.confirmed_mechanics) == 0


def test_int08_migrate_cross_run_memory_idempotent(tmp_path):
    """tools.migrate_cross_run_memory: run twice -> second run is a no-op."""
    src = tmp_path / "cross_run_memory.json"
    dest = tmp_path / "skill_state.json"
    md_path = tmp_path / "SKILL.md"

    src.write_text(json.dumps({
        "confirmed_mechanics": [
            {"id": "A001", "natural_language": "test mechanic",
             "evidence_runs": ["c1"], "first_seen_run": "c1",
             "first_seen_turn": 5, "confirmation_count": 2,
             "paired_cf_offsets": [10, 20]},
        ],
    }), encoding="utf-8")

    s1 = migrate(src, dest, game_id="test", render_skill_md_to=md_path)
    assert s1["status"] == "imported"
    assert s1["imported"] == 1

    # Second run = no-op
    s2 = migrate(src, dest, game_id="test", render_skill_md_to=md_path)
    assert s2["status"] == "already_imported"
    assert s2.get("imported", 0) == 0  # nothing newly imported


# =============================================================================
# INT09 — Prompt injection cap
# =============================================================================


def _build_oversize_state() -> SkillState:
    """Build state per INT09_prompt_injection_cap.json synthetic recipe."""
    state = SkillState(metadata=SkillStateMetadata(game_id="oversize-test"))
    # 50 1A entries with descending confirmation_count
    state.confirmed_mechanics = [
        ConfirmedMechanic(
            id=f"A{i+1:03d}",
            natural_language=f"Confirmed mechanic #{i+1}: " + ("X" * 80),
            evidence_runs=[f"cycle{i+200}", f"cycle{i+300}"],
            first_seen_run=f"cycle{i+200}",
            first_seen_turn=i + 5,
            confirmation_count=50 - i,  # descending
            paired_cf_offsets=[i * 3, i * 3 + 1, i * 3 + 2],
        ) for i in range(50)
    ]
    # 200 F entries
    state.falsifications = [
        Falsification(
            id=f"F_cycle{i}_{i+5}",
            natural_language=f"Falsification {i}: " + ("Y" * 60),
            failure_mode="wrong_state",
            evidence={"coord_x": i, "coord_y": i + 1},
            run_id=f"cycle{i}",
            turn=i + 5,
        ) for i in range(200)
    ]
    # 30 S.dynamic active
    state.skill_lifecycle = [
        SkillRecord(
            id=f"Sext-{i:04d}",
            family="extended",
            description=f"Dynamic skill {i}: " + ("Z" * 40),
            is_static=False,
            install_run=f"cycle{i+10}",
            confirmed_count=30 - i,
            status="active",
        ) for i in range(30)
    ]
    # plus 13 static
    state.skill_lifecycle.extend([
        SkillRecord(
            id=f"S{i:02d}", family="static_family",
            description=f"static {i}", is_static=True,
            confirmed_count=i, falsified_count=2 * i,
        ) for i in range(13)
    ])
    return state


def test_int09_prompt_context_under_4kb():
    """Oversized state -> render_prompt_context output <= 4 KB."""
    state = _build_oversize_state()
    ctx = render_prompt_context(state)
    size = len(ctx.encode("utf-8"))
    assert size <= PROMPT_CONTEXT_BUDGET_BYTES, (
        f"prompt context exceeded budget: {size} > {PROMPT_CONTEXT_BUDGET_BYTES}"
    )


def test_int09_prompt_context_obeys_top_k_caps():
    """Capped prompt limits 1A to 12 (8 top + 4 recent), F to 12, S.dynamic to 8."""
    state = _build_oversize_state()
    ctx = render_prompt_context(state)

    # Count bracketed-id markers per section
    import re
    a_count = len(re.findall(r"\[A\d{3}\]", ctx))
    f_count = len(re.findall(r"\[F_cycle\d+", ctx))
    s_dynamic_count = len(re.findall(r"\[Sext-\d+\]", ctx))

    # Plan v602 §3 caps: 1A top 8 + recent 4 = 12 max (with possible overlap).
    assert a_count <= 12
    # F: top 8 + recent 4 = 12 max (with possible overlap).
    assert f_count <= 12
    # S.dynamic: top 8.
    assert s_dynamic_count <= 8

    # S.static is summary-only (no per-id lines).
    s_static_count = len(re.findall(r"\[S\d{2}\]", ctx))
    assert s_static_count == 0
    assert "static skills" in ctx  # roll-up summary present

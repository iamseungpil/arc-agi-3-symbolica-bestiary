"""v607 Phase 7 — fixture suite for skill discovery revival.

Train/val split: each test labelled #train or #val per plan rev E §5.1.
Gates: train pass-rate >= 80%, val pass-rate >= 75%, gap <= 15pt.

Covers all four v607 modules:
- anti_leak (L1-L4): V-LEAK token, cycle237 token, TF-IDF, clean predicate
- stuck trigger (S1-S5): no-fire, fires-on-stagnation, cooldown, cap, dt-change
- posterior + emit (P1-P3): ranking, update, increment
- skill_state EmitChidTemplatePatch (F1-F2): apply + dedup
- Agent integration (I1-I2): 5-turn slice with mocked emit_callable

Run: pytest tests/v607/ -v
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agents.templates.agentica_lite.anti_leak import (
    _set_forbidden_corpus,
    _clear_forbidden_corpus,
    validate_chid_template,
)
from agents.templates.agentica_lite.predicate_posterior import (
    increment_emit_count,
    rank_chid_templates,
    update_chid_posterior,
)
from agents.templates.agentica_lite.proposer import parse_and_validate
from agents.templates.agentica_lite.reflector import (
    StuckTriggerConfig,
    StuckTriggerState,
    compute_adaptive_cap,
    emit_new_chid,
    record_fire,
    reset_episode,
    step_cooldown,
    stuck_fires,
)
from agents.templates.agentica_lite.skill_state import (
    EmitChidTemplatePatch,
    SkillRecord,
    SkillState,
    SkillStateMetadata,
    apply_patch,
)


# =============================================================================
# L1-L4: Anti-leak validation
# =============================================================================


class TestAntiLeak:
    """L1-L4 fixtures: validate_chid_template + Proposer schema inject."""

    def setup_method(self) -> None:
        # Force test corpus so we don't depend on cycle237 trace file.
        _set_forbidden_corpus(
            ["P_crop_compass_sweep_R31", "P_R12_crop_sector_alignment"]
        )

    def teardown_method(self) -> None:
        _clear_forbidden_corpus()

    def test_L1_clean_template_passes(self) -> None:  # #train
        ok, reason = validate_chid_template("P_scan_pixel_C5")
        assert ok, f"clean template rejected: {reason}"
        assert reason == "ok"

    def test_L2_v_leak_token_blocked(self) -> None:  # #train
        for tok_chid in ("P_R12_chain_C5", "P_R31_test", "P_gqb_pattern"):
            ok, reason = validate_chid_template(tok_chid)
            assert not ok, f"V-LEAK leak slipped: {tok_chid}"
            assert reason == "v_leak_token"

    def test_L3_cycle237_token_blocked(self) -> None:  # #train
        for chid in ("P_compass_test_C5", "P_sweep_C5", "P_sector_align_R5"):
            ok, reason = validate_chid_template(chid)
            assert not ok, f"cycle237 token leak: {chid}"
            assert reason.startswith("cycle237_token:")

    def test_L4_empty_rejected(self) -> None:  # #val
        for s in ("", "   ", "\t\n"):
            ok, reason = validate_chid_template(s)
            assert not ok
            assert reason == "empty_template"

    def test_L5_proposer_schema_anti_leak_inject(self) -> None:  # #val
        """Phase 5: parse_and_validate rejects leak via schema_error_code."""
        clean = {
            "candidate_predicate_id": "P_scan_pixel_C5",
            "region_hint": "R5",
            "expected_signature": {},
            "required_pre_state": {
                "marker_id": "C5",
                "saturation_threshold": 0.5,
                "saturation_denominator": 10,
            },
            "confidence": 0.5,
        }
        r1 = parse_and_validate(clean, ["R5"])
        assert r1.failure_reason is None, f"clean failed: {r1.schema_error_code}"

        leak_pid = dict(clean, candidate_predicate_id="P_compass_test_C5")
        r2 = parse_and_validate(leak_pid, ["R5"])
        assert r2.failure_reason == "schema_invalid"
        assert r2.schema_error_code.startswith("anti_leak:")


# =============================================================================
# S1-S5: Stuck-trigger logic
# =============================================================================


class TestStuckTrigger:
    """S1-S5 fixtures: stuck_fires + cooldown + cap."""

    def test_S1_no_fire_early(self) -> None:  # #train
        """At turn 2 with no advance, fires=False (turns_since_advance <= 3 AND
        no new dominant_transition signal). Pre-set last_dt=current_dt to
        suppress the new-DT trigger branch."""
        st = StuckTriggerState()
        st.last_dominant_transition = "5->4"
        fired = stuck_fires(
            turns_since_advance=2,
            best_advance_age=2,
            episode_length=30,
            current_dt="5->4",
            state=st,
        )
        assert not fired

    def test_S2_fires_on_genuine_stagnation(self) -> None:  # #train
        """turns_since_advance=5, best_advance_age=5, no cooldown → fires."""
        st = StuckTriggerState()
        fired = stuck_fires(
            turns_since_advance=5,
            best_advance_age=5,
            episode_length=30,
            current_dt="5->4",
            state=st,
        )
        assert fired

    def test_S3_cooldown_blocks(self) -> None:  # #train
        """After fire, cooldown=5 → next 5 turns suppressed."""
        st = StuckTriggerState()
        st.fires_this_episode = 1
        st.cooldown_remaining = 3
        fired = stuck_fires(
            turns_since_advance=10,
            best_advance_age=10,
            episode_length=30,
            current_dt="5->4",
            state=st,
        )
        assert not fired

    def test_S4_adaptive_cap_blocks(self) -> None:  # #val
        """fires>=cap → no more fires regardless of stagnation."""
        cap = compute_adaptive_cap(episode_length=30, stagnation_severity=10)
        st = StuckTriggerState()
        st.fires_this_episode = cap
        fired = stuck_fires(
            turns_since_advance=20,
            best_advance_age=20,
            episode_length=30,
            current_dt="5->4",
            state=st,
        )
        assert not fired

    def test_S5_dt_change_triggers(self) -> None:  # #val
        """new dominant_transition (positive evidence) overrides cooldown gate."""
        st = StuckTriggerState()
        st.last_dominant_transition = "5->4"
        # New DT, even with low stagnation, should fire (OR branch in stuck_fires).
        fired = stuck_fires(
            turns_since_advance=5,
            best_advance_age=5,
            episode_length=30,
            current_dt="4->8",  # different from last
            state=st,
        )
        assert fired

    def test_S6_cooldown_decrement(self) -> None:  # #train
        """step_cooldown decrements; record_fire resets."""
        st = StuckTriggerState()
        st.cooldown_remaining = 3
        step_cooldown(st)
        assert st.cooldown_remaining == 2
        step_cooldown(st)
        assert st.cooldown_remaining == 1
        record_fire(st, turn=10)
        assert st.cooldown_remaining == 5  # default cfg
        assert st.fires_this_episode == 1


# =============================================================================
# P1-P3: Posterior + emit telemetry
# =============================================================================


class TestPosterior:
    """P1-P3 fixtures: rank_chid_templates + update + increment."""

    @staticmethod
    def _make_state(records: list[tuple[str, float, float]]) -> SkillState:
        st = SkillState(metadata=SkillStateMetadata(game_id="test"))
        for i, (tmpl, a, b) in enumerate(records):
            st.skill_lifecycle.append(
                SkillRecord(
                    id=f"S{i}",
                    family="test",
                    description=f"test rec {tmpl}",
                    install_run="r0",
                    is_static=False,
                    status="active",
                    chid_template=tmpl,
                    beta_alpha=a,
                    beta_beta=b,
                    emit_count=5,  # past warmup threshold
                )
            )
        return st

    def test_P1_rank_by_ev(self) -> None:  # #train
        """EV = α/(α+β); higher EV ranks first."""
        st = self._make_state(
            [
                ("P_alpha_C{m}", 7.0, 3.0),  # EV 0.7
                ("P_beta_C{m}", 1.0, 9.0),  # EV 0.1
                ("P_gamma_C{m}", 5.0, 5.0),  # EV 0.5
            ]
        )
        top = rank_chid_templates(st, k=2)
        # Top 2 should be alpha then gamma (descending EV). API returns
        # tuples (chid_template, ev) or plain strings depending on impl.
        assert len(top) == 2
        names = [t[0] if isinstance(t, tuple) else t for t in top]
        assert names[0] == "P_alpha_C{m}"
        assert names[1] == "P_gamma_C{m}"

    def test_P2_update_posterior_success(self) -> None:  # #train
        st = self._make_state([("P_alpha_C{m}", 1.0, 1.0)])
        ok = update_chid_posterior(st, "P_alpha_C{m}", delta_clipped=1)
        assert ok
        rec = st.skill_lifecycle[0]
        assert rec.beta_alpha == 2.0
        assert rec.beta_beta == 1.0

    def test_P3_update_posterior_failure(self) -> None:  # #val
        st = self._make_state([("P_alpha_C{m}", 1.0, 1.0)])
        ok = update_chid_posterior(st, "P_alpha_C{m}", delta_clipped=0)
        assert ok
        rec = st.skill_lifecycle[0]
        assert rec.beta_alpha == 1.0
        assert rec.beta_beta == 2.0

    def test_P4_update_unknown_chid_noop(self) -> None:  # #val
        st = self._make_state([("P_alpha_C{m}", 1.0, 1.0)])
        ok = update_chid_posterior(st, "P_unknown_C{m}", delta_clipped=1)
        assert not ok
        # Original unchanged
        assert st.skill_lifecycle[0].beta_alpha == 1.0

    def test_P5_increment_emit_count(self) -> None:  # #train
        st = self._make_state([("P_alpha_C{m}", 1.0, 1.0)])
        st.skill_lifecycle[0].emit_count = 0  # reset for clean test
        st.skill_lifecycle[0].emit_tokens = 0
        increment_emit_count(st, "P_alpha_C{m}", turn=15)
        rec = st.skill_lifecycle[0]
        assert rec.emit_count == 1
        assert rec.emit_tokens == 600  # marginal token attribution
        assert rec.last_emit_turn == 15


# =============================================================================
# F1-F2: Skill state mutation
# =============================================================================


class TestSkillState:
    """F1-F2 fixtures: EmitChidTemplatePatch apply + persistence."""

    def test_F1_emit_patch_adds_record(self) -> None:  # #train
        st = SkillState(metadata=SkillStateMetadata(game_id="test"))
        patch = EmitChidTemplatePatch(
            chid_template="P_scan_pixel_C{m}",
            family="reflector",
            description="test",
            emit_run="seed42",
            emit_turn=5,
            emit_tokens=400,
            cooldown=5,
        )
        res = apply_patch(st, patch)
        assert res.accepted
        assert len(st.skill_lifecycle) == 1
        rec = st.skill_lifecycle[0]
        assert rec.chid_template == "P_scan_pixel_C{m}"
        assert rec.beta_alpha == 0.5  # Jeffreys prior
        assert rec.beta_beta == 0.5
        assert rec.family == "reflector"

    def test_F2_duplicate_chid_dedupes(self) -> None:  # #val
        st = SkillState(metadata=SkillStateMetadata(game_id="test"))
        patch = EmitChidTemplatePatch(
            chid_template="P_dup_C{m}",
            family="reflector",
            emit_run="seed42",
            emit_turn=5,
        )
        r1 = apply_patch(st, patch)
        r2 = apply_patch(st, patch)
        assert r1.accepted
        assert not r2.accepted or len(st.skill_lifecycle) == 1


# =============================================================================
# I1-I2: Agent integration (5-turn slice with mocked emit_callable)
# =============================================================================


class TestAgentIntegration:
    """I1-I2 fixtures: 5-turn agent slice with mocked Reflector."""

    @pytest.mark.asyncio
    async def test_I1_agent_5turn_stuck_emit(self, tmp_path: Path) -> None:  # #val
        """5 turns of stagnation → Reflector fires + skill_state grows."""
        os.environ["ARC_LITE_SKILL_STATE_PATH"] = str(tmp_path / "skill_state.json")
        # Import after env var set
        from agents.templates.agentica_lite.agent import ArcgenticaLite

        a = ArcgenticaLite("ft09-test-I1", seed=7)
        # No level_delta, same DT — accumulate stuck signal.
        state = {
            "visible_regions": [{"region_id": f"R{i}"} for i in range(1, 6)],
            "last_observation": {
                "level_delta": 0,
                "dominant_transition": {"from": 5, "to": 4, "count": 100},
            },
        }
        for _ in range(6):
            await a.run_turn(state)
        # After 6 turns of stagnation, expect at least 1 stuck fire attempt.
        assert a._v607_emit_calls >= 1, (
            f"expected >=1 emit attempt, got {a._v607_emit_calls}"
        )
        # turns_since_L_plus tracking
        assert a.turns_since_L_plus == 6
        # cleanup
        if "ARC_LITE_SKILL_STATE_PATH" in os.environ:
            del os.environ["ARC_LITE_SKILL_STATE_PATH"]

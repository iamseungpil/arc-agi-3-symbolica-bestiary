"""V-gate test suite for the SkillLibrary stack (plan v651 revB).

Coverage map:

    V1a  CRUD + persistence round-trip ........ TestV1aCRUD
    V1b  sandbox red-team (DSL + code helper) .. TestV1bSandbox
    V1c  persistence atomicity / .bak fallback . TestV1cPersistenceAtomicity
    V1d  predicate invariance (deferred) ....... TestV1dInvariance (skipped)
    V2a  query canary at 0 / 1 / 1000 skills .... TestV2aCanary
    --   consolidator gate + dedup behaviour ... TestConsolidator

The tests are intentionally self-contained — they construct ``SkillLibrary``
with ``model="stub"`` and never spawn a real Agentica agent, so they run
without TRAPI credentials and without the agentica-server.  The natural-
language query path is exercised only at construction-time (empty library)
and via the synchronous ``summaries()`` hot path; the actual agent task is
NOT created in this suite (see ``TestV2aCanary.test_thousand_skill_query``).

Fixtures live under ``tests/fixtures/skill_library/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from research_extensions.tools.skill_consolidator import (
    consolidate_from_hypothesis,
    consolidate_from_level_clear,
    dedup_near_duplicate,
)
from research_extensions.tools.skill_library import (
    Skill,
    SkillLibrary,
    SkillLoopError,
    SkillQueryError,
)
from research_extensions.tools.skill_persistence import (
    load_with_fallback,
    save,
)
from research_extensions.tools.skill_sandbox import (
    CodeHelperExecutor,
    DSLEvaluator,
    SkillSandboxError,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skill_library"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_persistence_path(tmp_path: Path) -> str:
    """Return a per-test persistence file path under pytest tmp dir."""
    return str(tmp_path / "skills.json")


@pytest.fixture
def stub_skill_lib(tmp_persistence_path: str) -> SkillLibrary:
    """Build an empty SkillLibrary bound to a tmp persistence path."""
    return SkillLibrary(model="stub", persistence_path=tmp_persistence_path)


def _load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _skill_kwargs_from_fixture(d: dict[str, Any]) -> dict[str, Any]:
    """Extract the kwargs accepted by ``SkillLibrary.add`` from a fixture dict."""
    return {
        "summary": d["summary"],
        "recipe": d["recipe"],
        "evidence": list(d.get("evidence", [])),
        "applicability_conditions": list(d.get("applicability_conditions", [])),
        "parent_hypothesis_ids": list(d.get("parent_hypothesis_ids", [])),
        "category": d["category"],
        "posterior": tuple(d.get("posterior", [0, 0])),
        "predicate": d.get("predicate"),
        "code": d.get("code"),
    }


def _read_events(path: str) -> list[dict[str, Any]]:
    events_path = f"{path}.events.jsonl"
    if not os.path.exists(events_path):
        return []
    lines = Path(events_path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ===========================================================================
# V1a — CRUD + persistence round-trip
# ===========================================================================


class TestV1aCRUD:
    def test_load_synthetic_skills_round_trip(self, tmp_persistence_path: str) -> None:
        lib = SkillLibrary(model="stub", persistence_path=tmp_persistence_path)
        originals: list[dict[str, Any]] = []
        ids: list[str] = []
        for i in range(1, 6):
            fix = _load_fixture(f"synthetic_skill_{i}.json")
            originals.append(fix)
            sid = lib.add(**_skill_kwargs_from_fixture(fix))
            ids.append(sid)
            # If the fixture is quarantined, mimic the orchestrator and flip
            # the flag on the live record so the round-trip preserves it.
            if fix.get("quarantined"):
                from dataclasses import replace

                idx, old = lib._find(sid)
                lib.stack[idx] = replace(
                    old,
                    quarantined=True,
                    quarantine_reason=fix.get("quarantine_reason"),
                )
                lib._maybe_snapshot()

        # snapshot already happened on every add; reload from disk
        lib2 = SkillLibrary.load(tmp_persistence_path, model="stub")
        assert len(lib2.stack) == 5
        assert lib2._load_meta["loaded_from"] == "primary"

        summaries = lib2.summaries()
        assert len(summaries) == 5
        for i, fix in enumerate(originals):
            assert fix["summary"] in summaries[i]

        # spot-check the quarantined fixture survives
        skill_4 = lib2.get(ids[3])
        assert skill_4.quarantined is True
        assert "contaminated" in (skill_4.quarantine_reason or "").lower() \
            or "exceeded" in (skill_4.quarantine_reason or "").lower()

        # spot-check predicate + code survived for fixture 5
        skill_5 = lib2.get(ids[4])
        assert skill_5.predicate == 'f["red_region_count"] >= 1'
        assert skill_5.code is not None
        assert "RESULT" in skill_5.code

    @pytest.mark.asyncio
    async def test_confirm_increments_posterior(
        self, stub_skill_lib: SkillLibrary
    ) -> None:
        sid = stub_skill_lib.add(
            summary="dummy",
            recipe="r",
            evidence=[],
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
            posterior=(0, 0),
        )
        for i in range(3):
            await stub_skill_lib.confirm(sid, f"mem:{i}")
        s = stub_skill_lib.get(sid)
        assert s.posterior == (3, 0)
        assert s.quarantined is False
        assert len(s.evidence) == 3

    @pytest.mark.asyncio
    async def test_falsify_increments_posterior(
        self, stub_skill_lib: SkillLibrary
    ) -> None:
        sid = stub_skill_lib.add(
            summary="dummy",
            recipe="r",
            evidence=[],
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
            posterior=(2, 0),
        )
        await stub_skill_lib.falsify(sid, "did not match", "mem:bad")
        s = stub_skill_lib.get(sid)
        assert s.posterior == (2, 1)
        # 1 ≤ 2 ⇒ NOT auto-quarantined per contamination guardrail
        assert s.quarantined is False

    @pytest.mark.asyncio
    async def test_event_log_appended(
        self,
        stub_skill_lib: SkillLibrary,
        tmp_persistence_path: str,
    ) -> None:
        sid = stub_skill_lib.add(
            summary="dummy",
            recipe="r",
            evidence=[],
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
        )
        await stub_skill_lib.confirm(sid, "mem:a")
        await stub_skill_lib.falsify(sid, "wrong", "mem:b")
        events = _read_events(tmp_persistence_path)
        assert len(events) == 2
        assert events[0]["event"] == "confirm"
        assert events[0]["skill_id"] == sid
        assert events[1]["event"] == "falsify"
        assert events[1]["skill_id"] == sid


# ===========================================================================
# V1b — sandbox red-team
# ===========================================================================


_DSL_MALICIOUS = [
    ("import-via-call", "__import__('os').system('echo')"),
    ("attribute-read", "f['x'].read()"),
    ("lambda-call", "(lambda: True)()"),
    ("listcomp", "[i for i in range(10)]"),
    ("ifexp", "True if True else False"),
]


_CODE_MALICIOUS = [
    ("import-blocked", "import socket\nRESULT = socket.gethostname()"),
    ("open-blocked", "open('/tmp/skill_attack.txt', 'w').write('x')\nRESULT = {}"),
    ("wall-clock-timeout", "while True:\n    pass"),
    ("rlimit-as-oom", "RESULT = [0] * (10 ** 9)"),
    ("non-dict-return", "RESULT = 42"),
]


class TestV1bSandbox:
    @pytest.mark.parametrize("label,payload", _DSL_MALICIOUS)
    def test_dsl_rejects_malicious(self, label: str, payload: str) -> None:
        evaluator = DSLEvaluator()
        with pytest.raises(SkillSandboxError):
            evaluator.evaluate(payload, {"x": 1})

    @pytest.mark.parametrize("label,payload", _CODE_MALICIOUS)
    def test_code_helper_returns_none_on_malicious(
        self, label: str, payload: str
    ) -> None:
        executor = CodeHelperExecutor()
        # Tight bounds: 1.0s wall-clock, 50MB RLIMIT_AS
        result = executor.run(payload, frame=[[0]], wall_clock_s=1.0, mem_limit_mb=50)
        assert result is None, (
            f"sandbox payload {label!r} returned {result!r}; expected None "
            f"(malicious payload should be rejected/killed/non-dict)"
        )

    def test_dsl_allows_legitimate_predicate(self) -> None:
        evaluator = DSLEvaluator()
        assert evaluator.evaluate("f[\"region_count\"] >= 3", {"region_count": 5}) is True
        assert evaluator.evaluate("f[\"region_count\"] >= 3", {"region_count": 1}) is False

    def test_code_helper_allows_legitimate_code(self) -> None:
        executor = CodeHelperExecutor()
        result = executor.run(
            "n = 0\nfor row in frame:\n    n += len(row)\nRESULT = {\"cells\": n}\n",
            frame=[[0, 1], [2, 3]],
            wall_clock_s=2.0,
        )
        assert result == {"cells": 4}

    @pytest.mark.asyncio
    async def test_quarantine_flag_set(self, stub_skill_lib: SkillLibrary) -> None:
        """falsify_n > confirm_n must auto-quarantine."""
        sid = stub_skill_lib.add(
            summary="contaminated candidate",
            recipe="r",
            evidence=[],
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
            posterior=(0, 0),
        )
        await stub_skill_lib.falsify(sid, "bad observation", "mem:x")
        s = stub_skill_lib.get(sid)
        # 1 > 0 ⇒ quarantined
        assert s.quarantined is True
        assert s.quarantine_reason is not None
        assert "bad observation" in s.quarantine_reason


# ===========================================================================
# V1c — persistence atomicity
# ===========================================================================


class TestV1cPersistenceAtomicity:
    def _make_skill(self, suffix: str = "a") -> Skill:
        return Skill(
            skill_id=f"00000000-0000-4000-8000-00000000000{suffix}",
            summary=f"sk-{suffix}",
            recipe="r",
            evidence=[],
            posterior=(1, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
            timestamp=datetime(2026, 5, 15, 0, 0, 0),
        )

    def test_corrupt_primary_fallback_to_bak(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.json")
        save(path, [self._make_skill("a")])  # creates primary
        save(path, [self._make_skill("a"), self._make_skill("b")])  # rotates .bak

        # corrupt primary by truncating last 16 bytes
        with open(path, "rb+") as f:
            f.seek(-16, os.SEEK_END)
            f.truncate()

        loaded, meta = load_with_fallback(path)
        assert meta["loaded_from"] == "bak"
        assert len(loaded) == 1
        assert loaded[0].summary == "sk-a"

    def test_corrupt_both_empty_fallback(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.json")
        save(path, [self._make_skill("a")])
        save(path, [self._make_skill("b")])  # now both files exist

        # corrupt primary
        with open(path, "rb+") as f:
            f.seek(-16, os.SEEK_END)
            f.truncate()
        # corrupt .bak
        bak_path = f"{path}.bak"
        with open(bak_path, "rb+") as f:
            f.seek(-16, os.SEEK_END)
            f.truncate()

        loaded, meta = load_with_fallback(path)
        assert loaded == []
        assert meta["loaded_from"] == "empty"

    def test_orphan_tmp_ignored(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.json")
        save(path, [self._make_skill("a")])

        # simulate crash mid-write: leave a .tmp.<pid> stub with garbage
        orphan = f"{path}.tmp.99999"
        Path(orphan).write_bytes(b"GARBAGE")

        loaded, meta = load_with_fallback(path)
        # load_with_fallback only consults primary + .bak; orphan tmp is ignored
        assert meta["loaded_from"] == "primary"
        assert len(loaded) == 1
        assert loaded[0].summary == "sk-a"
        assert os.path.exists(orphan)  # orphan still on disk — but not consulted

    def test_checksum_mismatch_detected(self, tmp_path: Path) -> None:
        path = str(tmp_path / "s.json")
        save(path, [self._make_skill("a")])
        save(path, [self._make_skill("b")])  # rotate .bak

        # tamper the wrapped JSON: flip a byte inside "summary"
        raw = Path(path).read_bytes()
        # the canonical JSON is sorted, so locate "summary":"sk-b" deterministically
        idx = raw.find(b'"summary":"sk-b"')
        assert idx >= 0
        tampered = bytearray(raw)
        tampered[idx + len('"summary":"sk-b'):idx + len('"summary":"sk-b') + 1] = b"X"
        Path(path).write_bytes(bytes(tampered))

        loaded, meta = load_with_fallback(path)
        # primary checksum no longer matches → fallback to .bak (the earlier save)
        assert meta["loaded_from"] == "bak"
        assert loaded[0].summary == "sk-a"


# ===========================================================================
# V1d — predicate invariance under symmetries (deferred)
# ===========================================================================


class TestV1dInvariance:
    def test_invariance_deferred(self) -> None:
        pytest.skip(
            "TODO_V4: predicate invariance under translation/rotation/color-perm "
            "is meaningful only on real ft09 frames; deferred to autoresearch V4 "
            "phase. See reports/plan_v651_skill_library_revB.md §3 V1d."
        )


# ===========================================================================
# V2a — query canary
# ===========================================================================


class TestV2aCanary:
    @pytest.mark.asyncio
    async def test_empty_library_query(self, stub_skill_lib: SkillLibrary) -> None:
        t0 = time.perf_counter()
        result = await stub_skill_lib.query(str, "anything")
        elapsed = time.perf_counter() - t0
        # Empty stack short-circuits via _empty_result — no agent spawn.
        assert result == "No stored skills yet."
        assert elapsed < 5.0, f"empty query took {elapsed:.2f}s, expected <5s"

    def test_single_skill_query_via_summaries(
        self, stub_skill_lib: SkillLibrary
    ) -> None:
        """Library-as-data hot path: summaries() must return the added skill quickly."""
        fix = _load_fixture("synthetic_skill_1.json")
        stub_skill_lib.add(**_skill_kwargs_from_fixture(fix))
        t0 = time.perf_counter()
        summaries = stub_skill_lib.summaries()
        elapsed = time.perf_counter() - t0
        assert len(summaries) == 1
        assert "3x3" in summaries[0]
        assert elapsed < 5.0

    def test_thousand_skill_query(self, tmp_path: Path) -> None:
        """1000 skills: summaries() + a manual category scan must complete <5s.

        We deliberately exercise the synchronous hot path here.  ``query()``
        would require an Agentica subagent that we do not spawn in tests; the
        "library-as-data" interface is what production caller paths hit when
        the library is large.
        """
        # avoid persistence I/O at 1000-scale for this microbench
        lib = SkillLibrary(model="stub")
        t0 = time.perf_counter()
        for i in range(1000):
            lib.add(
                summary=f"skill {i}",
                recipe=f"recipe {i}",
                evidence=[f"mem:{i}"],
                applicability_conditions=[],
                parent_hypothesis_ids=[],
                category="mechanic" if i % 2 == 0 else "strategy",
            )
        add_elapsed = time.perf_counter() - t0
        assert add_elapsed < 5.0, f"1000 adds took {add_elapsed:.2f}s"

        t0 = time.perf_counter()
        summaries = lib.summaries()
        mechanic = [s for s in lib.stack if s.category == "mechanic"]
        scan_elapsed = time.perf_counter() - t0
        assert len(summaries) == 1000
        assert len(mechanic) == 500
        assert scan_elapsed < 5.0, f"summaries+scan took {scan_elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_empty_library_query_int(self, stub_skill_lib: SkillLibrary) -> None:
        # Confirms _empty_result has a typed fallback for non-str return types.
        result = await stub_skill_lib.query(int, "how many?")
        assert result == 0


# ===========================================================================
# Consolidator — hypothesis gate + level-clear extraction + dedup
# ===========================================================================


class _FakeMem:
    """Minimal duck-typed memory record for consolidate_from_level_clear."""

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeMemories:
    def __init__(self, items: list[Any]) -> None:
        self.stack = items


class TestConsolidator:
    def test_consolidate_from_hypothesis_gate_pass(self) -> None:
        h = {
            "id": "hyp-1",
            "claim": "pressing ACTION1 advances level when region_count >= 3",
            "confirm_n": 2,
            "falsify_n": 0,
            "evidence_memory_ids": ["mem:1", "mem:2"],
            "conditions": ["region_count >= 3"],
            "category": "mechanic",
            "recipe": "press ACTION1",
        }
        skill = consolidate_from_hypothesis(h, memories=None)
        assert skill is not None
        assert isinstance(skill, Skill)
        assert skill.parent_hypothesis_ids == ["hyp-1"]
        assert skill.category == "mechanic"
        assert "ACTION1" in skill.summary

    def test_consolidate_from_hypothesis_gate_fail(self) -> None:
        h = {
            "id": "hyp-2",
            "claim": "anything",
            "confirm_n": 5,
            "falsify_n": 1,  # any falsify > 0 should block
        }
        skill = consolidate_from_hypothesis(h, memories=None)
        assert skill is None

        # Also fails at confirm_n < 2
        h2 = {"id": "hyp-3", "claim": "x", "confirm_n": 1, "falsify_n": 0}
        assert consolidate_from_hypothesis(h2, memories=None) is None

    def test_consolidate_from_level_clear_extracts_chain(self) -> None:
        memories = _FakeMemories(
            [
                _FakeMem(memory_id="m0", summary="level_start", details=""),
                _FakeMem(memory_id="m1", action="ACTION1", outcome="moved cursor"),
                _FakeMem(memory_id="m2", action="ACTION6@(4,4)", outcome="region toggled"),
                _FakeMem(memory_id="m3", action="ACTION3", outcome="confirm press"),
                _FakeMem(memory_id="m4", summary="level_clear", details="cleared"),
            ]
        )
        skills = consolidate_from_level_clear(memories, level_clear_memory_index=4)
        assert len(skills) >= 1
        skill = skills[0]
        assert skill.category == "mechanic"
        assert "ACTION1" in skill.recipe
        assert "ACTION6@(4,4)" in skill.recipe
        assert set(skill.evidence) == {"m1", "m2", "m3"}

    def test_dedup_keeps_higher_margin(self) -> None:
        a = Skill(
            skill_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            summary="press ACTION1 when region count is high",
            recipe="step 1: do thing\nstep 2: confirm",
            evidence=["m1"],
            posterior=(5, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=["h1"],
            category="mechanic",
        )
        b = Skill(
            skill_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            summary="press ACTION1 when region count is high",
            recipe="step 1: do thing\nstep 2: confirm",
            evidence=["m2"],
            posterior=(1, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=["h2"],
            category="mechanic",
        )
        merged = dedup_near_duplicate([a, b], threshold=0.85)
        assert len(merged) == 1
        kept = merged[0]
        # higher margin (5-0=5 vs 1-0=1) wins
        assert kept.posterior == (5, 0)
        # evidence + parent_hypothesis_ids are unioned
        assert set(kept.evidence) >= {"m1", "m2"}
        assert set(kept.parent_hypothesis_ids) >= {"h1", "h2"}

    def test_dedup_preserves_distinct_skills(self) -> None:
        a = Skill(
            skill_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            summary="click center of 3x3 region",
            recipe="centroid then ACTION6",
            evidence=[],
            posterior=(1, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
        )
        b = Skill(
            skill_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            summary="toggle all red cells with ACTION3",
            recipe="enumerate then press",
            evidence=[],
            posterior=(1, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="strategy",
        )
        merged = dedup_near_duplicate([a, b], threshold=0.85)
        assert len(merged) == 2


class TestCrossLoopGuard:
    """codex r3 locked guardrail: cross-loop misuse must raise SkillLoopError.

    The plan explicitly forbids a SkillLibrary instance bound to one event loop
    from being reused under a different loop. Detection happens in
    ``_ensure_agent`` via ``id(asyncio.get_running_loop())`` comparison.
    """

    def test_ensure_agent_raises_skill_loop_error_on_mismatch(
        self, tmp_persistence_path: str
    ) -> None:
        """Direct unit test of the loop-id mismatch branch."""
        lib = SkillLibrary(model="stub", persistence_path=tmp_persistence_path)
        # Simulate state after a first call from a *different* loop:
        # _skill_agent set (bypasses first-call branch), _loop_id is a fake id
        # that will not equal id(asyncio.get_running_loop()) in the new context.
        lib._skill_agent = asyncio.Future()  # type: ignore[assignment]
        lib._loop_id = -1  # impossible real id

        async def call_under_new_loop() -> None:
            lib._ensure_agent()

        with pytest.raises(SkillLoopError):
            asyncio.run(call_under_new_loop())

    @pytest.mark.asyncio
    async def test_query_propagates_skill_loop_error(
        self, stub_skill_lib: SkillLibrary
    ) -> None:
        """The public ``query`` path must NOT swallow SkillLoopError
        (codex r2/r3 locked guardrail; codex r4 CRITICAL flagged a prior
        bug where the try/except swallowed it as ``_empty_result``)."""
        stub_skill_lib.add(
            summary="cross-loop probe",
            recipe="noop",
            evidence=[],
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
        )
        # Force the loop-id mismatch path
        stub_skill_lib._skill_agent = asyncio.Future()  # type: ignore[assignment]
        stub_skill_lib._loop_id = -1

        with pytest.raises(SkillLoopError):
            await stub_skill_lib.query(str, "what skill applies?")


if __name__ == "__main__":  # pragma: no cover - convenience runner
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

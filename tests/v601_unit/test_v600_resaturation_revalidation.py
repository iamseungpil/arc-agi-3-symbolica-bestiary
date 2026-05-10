"""v600 fixture revalidation under v601/v602 saturation arm-key path.

Plan v602 §11 codex final-gate refinement #6: v600's 13 fixtures must be
explicitly re-run under v601's new ArmKey saturation_status dimension and
confirmed to still pass (backward-compat invariant).

Strategy:
  - Load each v600 fixture
  - Where the fixture exercises predicate_posterior, force-invoke the v601
    saturation-extending paths:
      • split_rasi_with_saturation() applied
      • select() / rank_top() called with state=None (no target_marker_id),
        which is the v600 fallback behavior
  - Assert original v600 expected_assertion still holds.

If any fixture's EXPECTED behavior must change under v601 logic, this test
flags it for plan refinement (target: 0 such fixtures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.fixtures import Fixture, load_fixtures  # noqa: E402
from agents.templates.agentica_lite.memory_journal import (  # noqa: E402
    EpisodeRecord, MemoryJournal,
)
from agents.templates.agentica_lite.predicate_library import (  # noqa: E402
    Predicate, PredicateLibrary, _stub_predicate,
)
from agents.templates.agentica_lite.predicate_posterior import (  # noqa: E402
    ArmKey, ArmStats, PredicatePosterior,
)
from agents.templates.agentica_lite.stalemate_trigger import (  # noqa: E402
    StalemateConfig, StalemateTrigger,
)

V600_FIXTURES_DIR = REPO_ROOT / "tests" / "v600" / "fixtures"


# ---------- helpers (mirror v600 driver but with saturation arm-key path) ----


def _eval_assertion(expr: str, output, fixture_input):
    g: dict = {"__builtins__": __builtins__, "output": output, "input": fixture_input}
    if isinstance(fixture_input, dict):
        if "real_predicate" in fixture_input:
            g["real_key"] = ArmKey(fixture_input["real_predicate"]["id"], "ANY")
        if "decoy_predicate" in fixture_input:
            g["decoy_key"] = ArmKey(fixture_input["decoy_predicate"]["id"], "ANY")
    return bool(eval(expr, g))  # noqa: S307


def _run_predicate_posterior_with_v601_path(fx: Fixture):
    """Same logic as v600 driver but with split_rasi_with_saturation() applied."""
    inp = fx.input

    if "decoy_predicate" in inp:
        post = PredicatePosterior()
        for label, side in (("real", "real_predicate"), ("decoy", "decoy_predicate")):
            spec = inp[side]
            key = ArmKey(spec["id"], "ANY")
            post.arms[key] = ArmStats(
                alpha=float(spec["alpha"]), beta=float(spec["beta"]),
                n_emit=int(spec.get("n_emit", 0)),
            )
        # ★ v601: extend arms with saturation sub-arms (preserves sentinel)
        post.split_rasi_with_saturation()
        return post

    if "static_arms" in inp and "extended_arms" in inp:
        post = PredicatePosterior()
        lib = PredicateLibrary()

        all_ids = [a["id"] for a in inp["static_arms"]] + [a["id"] for a in inp["extended_arms"]]
        lib.static = {pid: Predicate(pid, "fixture", "centroid", _stub_predicate, True)
                      for pid in all_ids}
        region_id = "R0"
        for a in inp["static_arms"]:
            key = ArmKey(a["id"], region_id)
            post.arms[key] = ArmStats(
                alpha=float(a["alpha"]), beta=float(a["beta"]), n_emit=1,
            )
        for a in inp["extended_arms"]:
            key = ArmKey(a["id"], region_id)
            post.arms[key] = ArmStats(
                alpha=float(a["alpha"]), beta=float(a["beta"]),
                n_emit=int(a.get("n_emit", 0)),
            )
        # ★ v601: split into saturation sub-arms (sentinel preserved -> v600 paths
        # using ArmKey(pid, region_id) without saturation_status still resolve)
        post.split_rasi_with_saturation()
        visible_regions = [{"id": region_id, "region_id": region_id, "bbox": {}}]
        # state=None -> v601 saturation_progress predicates skipped (good, since
        # v600 fixtures don't include P12).
        return post.select(visible_regions, lib, state=None)

    visible_regions = inp.get("visible_regions", [])
    post = PredicatePosterior()
    lib = PredicateLibrary()
    # state=None preserves v600 behavior: P12 saturation_progress arms skipped
    return post.rank_top(visible_regions, lib, k=10, state=None)


def _run_stalemate(fx: Fixture):
    """v600 stalemate fixtures: return the trigger; assertions exercise it.

    Mirrors the v600 driver in tests/v600/test_v600_fixtures.py exactly so the
    fixture-defined `expected_assertion` (which calls output.fires(...) etc.)
    stays compatible.
    """
    return StalemateTrigger(StalemateConfig())


def _run_library_install(fx: Fixture):
    inp = fx.input
    lib = PredicateLibrary()
    return lib.resolve_coord(inp["predicate_id"], inp["region"], state=None)


def _run_memory_journal(fx: Fixture, tmp_path: Path):
    n = int(fx.input.get("n_records", 10))
    journal = MemoryJournal("game_test", base_dir=tmp_path)
    for i in range(n):
        rec = EpisodeRecord(
            episode_id=f"ep_{i:02d}", seed=i,
            framework_version="v600", git_sha="abc123",
            ts_start=float(i), ts_end=float(i) + 1.0,
            turns=10, max_level=1,
        )
        journal.append(rec)
    return journal.load_all()


def _run_fixture(fx: Fixture, tmp_path: Path):
    if fx.expected_module == "stalemate_trigger":
        out = _run_stalemate(fx)
    elif fx.expected_module == "predicate_posterior":
        out = _run_predicate_posterior_with_v601_path(fx)
    elif fx.expected_module == "library_install":
        out = _run_library_install(fx)
    elif fx.expected_module == "memory_journal":
        out = _run_memory_journal(fx, tmp_path)
    else:
        pytest.skip(f"unknown expected_module: {fx.expected_module}")
    return _eval_assertion(fx.expected_assertion, out, fx.input)


# ---------- pytest entry point ----------------------------------------------


@pytest.fixture
def v600_corpus():
    return load_fixtures(V600_FIXTURES_DIR)


def test_v600_corpus_loaded(v600_corpus):
    """Sanity gate: v600 corpus has the documented 13 fixtures."""
    assert len(v600_corpus) >= 13, (
        f"expected >=13 v600 fixtures, got {len(v600_corpus)}"
    )


@pytest.mark.parametrize(
    "fixture_path",
    sorted(str(p) for p in V600_FIXTURES_DIR.rglob("*.json")),
)
def test_v600_fixture_under_v601_saturation_path(fixture_path, tmp_path):
    """Each v600 fixture passes when run through v601 saturation arm-key path.

    Backward-compat invariant: split_rasi_with_saturation() preserves the
    sentinel arm at saturation_status='n/a', so v600's ArmKey(pid, region_id)
    lookups still resolve. saturation_progress predicates are skipped at
    select()/rank_top() when state=None (v600's fallback path).
    """
    # Need to load just this single fixture
    with open(fixture_path, encoding="utf-8") as f:
        d = json.load(f)
    fx = Fixture(**d)
    ok = _run_fixture(fx, tmp_path)
    assert ok, (
        f"v600 fixture {fx.id} (failure_mode={fx.failure_mode}) regressed "
        f"under v601 saturation path. assertion={fx.expected_assertion!r}"
    )

"""Fixture suite for plan v600 minimal-fast framework.

Loads every JSON fixture under tests/v600/fixtures/ and asserts each
fixture's `expected_assertion` against the named module's output.

Pass criteria (plan §1.4): train >= 80%, val >= 75%, gap <= 15pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.fixtures import Fixture, load_fixtures, split_pass_rates  # noqa: E402
from agents.templates.agentica_lite.memory_journal import EpisodeRecord, MemoryJournal  # noqa: E402
from agents.templates.agentica_lite.predicate_library import PredicateLibrary  # noqa: E402
from agents.templates.agentica_lite.predicate_posterior import (  # noqa: E402
    ArmKey,
    ArmStats,
    PredicatePosterior,
)
from agents.templates.agentica_lite.stalemate_trigger import (  # noqa: E402
    StalemateConfig,
    StalemateTrigger,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- helpers


def _eval_assertion(expr: str, output, fixture_input):
    """Eval a fixture's expected_assertion.

    Names `output`, `input`, and (when applicable) `real_key`/`decoy_key`
    are placed in the globals dict so that they remain visible inside
    generator expressions and comprehensions.
    """
    g: dict = {"__builtins__": __builtins__, "output": output, "input": fixture_input}
    if isinstance(fixture_input, dict):
        if "real_predicate" in fixture_input:
            g["real_key"] = ArmKey(fixture_input["real_predicate"]["id"], "ANY")
        if "decoy_predicate" in fixture_input:
            g["decoy_key"] = ArmKey(fixture_input["decoy_predicate"]["id"], "ANY")
    return bool(eval(expr, g))  # noqa: S307


# --------------------------------------------------------------------------- runners


def _run_stalemate(fx: Fixture):
    return StalemateTrigger(StalemateConfig())


def _run_predicate_posterior(fx: Fixture):
    """Build the posterior in the shape that this fixture's assertion expects."""
    inp = fx.input
    fid = fx.id

    # F1 decoy: build a posterior with two seeded arms; output is the posterior itself.
    if "decoy_predicate" in inp:
        post = PredicatePosterior()
        for label, side in (("real", "real_predicate"), ("decoy", "decoy_predicate")):
            spec = inp[side]
            key = ArmKey(spec["id"], "ANY")
            post.arms[key] = ArmStats(
                alpha=float(spec["alpha"]),
                beta=float(spec["beta"]),
                n_emit=int(spec.get("n_emit", 0)),
            )
        return post

    # F1 invented chid: build static + extended arms; output is the result of select().
    if "static_arms" in inp and "extended_arms" in inp:
        post = PredicatePosterior()
        # build a fake library that exposes all the arm IDs
        lib = PredicateLibrary()

        from agents.templates.agentica_lite.predicate_library import Predicate, _stub_predicate
        all_ids = [a["id"] for a in inp["static_arms"]] + [a["id"] for a in inp["extended_arms"]]
        # clear the default static and supply only the fixture-defined ones
        lib.static = {pid: Predicate(pid, "fixture", "centroid", _stub_predicate, True)
                      for pid in all_ids}
        # seed posterior — static arms have n_emit=1 to make explore term finite
        region_id = "R0"
        for a in inp["static_arms"]:
            key = ArmKey(a["id"], region_id)
            post.arms[key] = ArmStats(
                alpha=float(a["alpha"]),
                beta=float(a["beta"]),
                n_emit=1,
            )
        for a in inp["extended_arms"]:
            key = ArmKey(a["id"], region_id)
            post.arms[key] = ArmStats(
                alpha=float(a["alpha"]),
                beta=float(a["beta"]),
                n_emit=int(a.get("n_emit", 0)),
            )
        # Force total_selections finite so log_N is well-defined.
        visible_regions = [{"id": region_id, "region_id": region_id, "bbox": {}}]
        return post.select(visible_regions, lib)

    # Replay-style fixtures: rank_top so assertions on `output[:3]` and `len(output) > 0` work.
    visible_regions = inp.get("visible_regions", [])
    post = PredicatePosterior()
    lib = PredicateLibrary()
    return post.rank_top(visible_regions, lib, k=10)


def _run_library_install(fx: Fixture):
    inp = fx.input
    lib = PredicateLibrary()
    return lib.resolve_coord(inp["predicate_id"], inp["region"], state=None)


def _run_memory_journal(fx: Fixture, tmp_path: Path):
    n = int(fx.input.get("n_records", 10))
    journal = MemoryJournal("game_test", base_dir=tmp_path)
    for i in range(n):
        rec = EpisodeRecord(
            episode_id=f"ep_{i:02d}",
            seed=i,
            framework_version="v600",
            git_sha="abc123",
            ts_start=float(i),
            ts_end=float(i) + 1.0,
            turns=10,
            max_level=1,
        )
        journal.append(rec)
    return journal.load_all()


# --------------------------------------------------------------------------- driver


def _run_fixture(fx: Fixture, tmp_path: Path):
    if fx.expected_module == "stalemate_trigger":
        out = _run_stalemate(fx)
    elif fx.expected_module == "predicate_posterior":
        out = _run_predicate_posterior(fx)
    elif fx.expected_module == "library_install":
        out = _run_library_install(fx)
    elif fx.expected_module == "memory_journal":
        out = _run_memory_journal(fx, tmp_path)
    else:
        pytest.skip(f"unknown expected_module: {fx.expected_module}")
    return _eval_assertion(fx.expected_assertion, out, fx.input)


@pytest.fixture
def fixture_corpus():
    return load_fixtures(FIXTURES_DIR)


def test_corpus_loaded(fixture_corpus):
    assert len(fixture_corpus) >= 13, f"expected >=13 fixtures, got {len(fixture_corpus)}"


@pytest.mark.parametrize(
    "fixture_path",
    sorted(str(p) for p in FIXTURES_DIR.rglob("*.json")),
)
def test_fixture(fixture_path, tmp_path):
    fx_list = load_fixtures(Path(fixture_path).parent)
    fx = next(f for f in fx_list if str(Path(fixture_path).resolve()).endswith(
        Path(fixture_path).name)
        and f.id == _id_from_path(fixture_path))
    ok = _run_fixture(fx, tmp_path)
    assert ok, f"fixture {fx.id} ({fx.failure_mode}) failed: {fx.expected_assertion}"


def _id_from_path(path: str) -> str:
    import json as _json
    with open(path) as f:
        return _json.load(f)["id"]


def test_pass_rates_meet_plan(tmp_path):
    """Aggregate: train >= 80%, val >= 75%, gap <= 15pt (plan §1.4)."""
    fx_list = load_fixtures(FIXTURES_DIR)
    results: list[tuple[Fixture, bool]] = []
    for fx in fx_list:
        try:
            ok = _run_fixture(fx, tmp_path / fx.id.replace("/", "_"))
        except Exception as e:  # noqa: BLE001
            print(f"[{fx.id}] EXCEPTION: {e}")
            ok = False
        results.append((fx, ok))
    rates = split_pass_rates(results)
    print(f"split_pass_rates: {rates}")
    train, val = rates.get("train", 1.0), rates.get("val", 1.0)
    assert train >= 0.80, f"train pass rate {train:.2%} < 80%"
    assert val >= 0.75, f"val pass rate {val:.2%} < 75%"
    assert abs(train - val) <= 0.15, f"split gap {train - val:.2%} > 15pt"

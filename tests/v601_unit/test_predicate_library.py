"""Per-module unit tests for predicate_library.py.

Plan v602 §11 addendum: 10 critical branch tests for static predicate set,
sandbox installer, and coord resolution.

Branches under test:
  1. static-P00-P11 (all base static predicates registered + callable)
  2. P12 with target (returns unclicked-slot regions)
  3. P12 no target (returns empty)
  4. sandbox eval-blocked
  5. sandbox import-blocked
  6. sandbox dunder-blocked
  7. sandbox non-deterministic rejected
  8. sandbox valid-installs
  9. install-runtime under 5s
  10. resolve-coord (centroid / corner_top_left / list-bbox)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.predicate_library import (  # noqa: E402
    PredicateLibrary, STATIC_PREDICATES, RegionRef,
)


# ---------- 1. static P00-P11 -------------------------------------------------

def test_static_predicates_p00_to_p11_all_registered():
    """All 12 base static predicates (P00-P11) registered + callable on a tiny state."""
    expected_ids = [
        "P00_fallback", "P01_dominant_transition", "P02_marker_align",
        "P03_sector_alignment", "P04_color_invariance", "P05_size_anchor",
        "P06_neighbor_pair", "P07_corner_align", "P08_unclicked_neighbor",
        "P09_compass_change", "P10_marker_no_recent", "P11_recent_response",
    ]
    for pid in expected_ids:
        assert pid in STATIC_PREDICATES, f"missing static predicate: {pid}"
        pred = STATIC_PREDICATES[pid]
        assert pred.is_static is True
        # callable on minimal state -> returns list (may be empty for stubs)
        out = pred.fn({"visible_regions": []}, 0)
        assert isinstance(out, list)


# ---------- 2. P12 with-target ------------------------------------------------

def test_p12_saturation_progress_with_target_returns_unclicked():
    """P12 returns RegionRefs for the target marker's unclicked compass slots."""
    state = {
        "target_marker_id": "M0",
        "marker_neighbor_states": [
            {"marker_id": "M0", "compass": {
                "N": {"region_id": "R_clicked", "clicks": 1},
                "E": {"region_id": "R_unclicked_a", "clicks": 0},
                "S": {"region_id": "R_unclicked_b", "clicks": 0},
                "W": {"region_id": "R_clicked2", "clicks": 2},
            }},
        ],
        "visible_regions": [
            {"region_id": "R_clicked", "bbox": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}},
            {"region_id": "R_unclicked_a", "bbox": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}},
            {"region_id": "R_unclicked_b", "bbox": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}},
            {"region_id": "R_other", "bbox": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}},
        ],
    }
    p12 = STATIC_PREDICATES["P12_saturation_progress"]
    out = p12.fn(state, 0)
    rids = sorted(r.region_id for r in out)
    assert rids == ["R_unclicked_a", "R_unclicked_b"]


# ---------- 3. P12 no-target --------------------------------------------------

def test_p12_saturation_progress_returns_empty_without_target():
    """Without target_marker_id, P12 returns []."""
    state = {
        "marker_neighbor_states": [
            {"marker_id": "M0", "compass": {"N": {"region_id": "R0", "clicks": 0}}},
        ],
        "visible_regions": [{"region_id": "R0", "bbox": {"min_x": 0, "min_y": 0, "max_x": 1, "max_y": 1}}],
    }
    p12 = STATIC_PREDICATES["P12_saturation_progress"]
    assert p12.fn(state, 0) == []


# ---------- 4. sandbox eval-blocked ------------------------------------------

def test_sandbox_blocks_eval_call():
    """`eval('...')` source is rejected with exec_blocked."""
    lib = PredicateLibrary()
    src = "def fn(state, t):\n    return list(eval('[]'))\n"
    res = lib.install(src)
    assert res.accepted is False
    assert "eval" in res.reason or "exec_blocked" in res.reason


# ---------- 5. sandbox import-blocked ----------------------------------------

def test_sandbox_blocks_import_statement():
    """`import os` source is rejected."""
    lib = PredicateLibrary()
    src = "import os\ndef fn(state, t):\n    return []\n"
    res = lib.install(src)
    assert res.accepted is False
    assert "import" in res.reason


# ---------- 6. sandbox dunder-blocked ----------------------------------------

def test_sandbox_blocks_dunder_attribute():
    """`x.__class__` access is rejected."""
    lib = PredicateLibrary()
    src = "def fn(state, t):\n    return state.__class__.__name__\n"
    res = lib.install(src)
    assert res.accepted is False
    assert "dunder" in res.reason or "__" in res.reason


# ---------- 7. sandbox non-deterministic rejected ----------------------------

def test_sandbox_rejects_non_deterministic():
    """A predicate that returns different output on identical input is rejected."""
    lib = PredicateLibrary()
    # Use a mutable cell to flip output across calls; sandbox calls fn 10 times.
    src = (
        "_state = {'flip': 0}\n"
        "def fn(state, t):\n"
        "    _state['flip'] = 1 - _state['flip']\n"
        "    if _state['flip'] == 1:\n"
        "        return [{'region_id': 'A'}]\n"
        "    return [{'region_id': 'B'}]\n"
    )
    res = lib.install(src)
    assert res.accepted is False
    assert "non_determ" in res.reason


# ---------- 8. sandbox valid-installs ----------------------------------------

def test_sandbox_accepts_clean_predicate():
    """A valid deterministic predicate installs and is queryable."""
    lib = PredicateLibrary()
    src = (
        "def fn(state, t):\n"
        "    out = []\n"
        "    for r in state['visible_regions']:\n"
        "        out.append({'region_id': r['region_id'], 'bbox': r.get('bbox', {})})\n"
        "    return out\n"
    )
    res = lib.install(src, predicate_id="L_test_id")
    assert res.accepted, f"unexpected rejection: {res.reason}"
    assert "L_test_id" in lib.installed
    assert lib.installed["L_test_id"].is_static is False


# ---------- 9. install-runtime ------------------------------------------------

def test_install_runtime_under_5s():
    """sandbox install completes well under 5s for a trivial predicate."""
    lib = PredicateLibrary()
    src = "def fn(state, t):\n    return []\n"
    res = lib.install(src)
    assert res.accepted
    # Hard cap at 5000 ms; in practice should be well under 100 ms.
    assert res.runtime_ms < 5000.0


# ---------- 10. resolve-coord (3 cases) --------------------------------------

def test_resolve_coord_centroid_corner_listbbox():
    """resolve_coord supports centroid + corner_top_left + list-bbox shapes."""
    lib = PredicateLibrary()
    # P00 -> centroid: bbox (10, 20, 30, 40) -> (20, 30)
    region_dict = {"region_id": "R0", "bbox": {"min_x": 10, "min_y": 20, "max_x": 30, "max_y": 40}}
    xy = lib.resolve_coord("P00_fallback", region_dict)
    assert xy == (20, 30)
    # P07 (corner_align) -> corner_top_left -> (10, 20)
    xy_c = lib.resolve_coord("P07_corner_align", region_dict)
    assert xy_c == (10, 20)
    # list-bbox shape
    region_list = {"region_id": "R1", "bbox": [4, 6, 14, 26]}
    xy_l = lib.resolve_coord("P00_fallback", region_list)
    assert xy_l == ((4 + 14) // 2, (6 + 26) // 2)

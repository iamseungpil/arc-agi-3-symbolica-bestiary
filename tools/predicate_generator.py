"""B18 plan v590 — Symbolica deterministic predicate induction.

Pure deterministic. No LLM. Replaces v589 B17 typed-role candidates.

Each predicate is a Python function `P(chain_state, t) -> list[RegionRef]`
that returns matching regions. Symbolica derives the click coord from
the predicate's anchor_region (centroid by default). M1 picks which
predicate-region pair to test by Beta-Bernoulli score.

Codex round-5 design + plan v590 self-critic round-3 frozen:
- 12 templates + P00 fallback (P00 score ≤0.3 cap)
- Beta-Bernoulli posterior; α=β=1 default; α=2 β=1 for simple templates
- Action-conditioned verdicts (clicked anchor_region required)
- Coord policy enum (centroid | sprite_center | corner_top_left)
- Whitelisted helpers, leak-safe by code review

Anti-leak: predicate library is FIXED at code level. V-LEAK extension
scans this file for forbidden vocab AND every emitted predicate_id
template prefix must be in PREDICATE_LIBRARY.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------
# Type defs.
# --------------------------------------------------------------------


@dataclass
class RegionRef:
    region_id: str
    bbox: dict | list
    color: int | None = None
    is_multicolor: bool = False
    kind: str = "non_marker"
    neighbors_3x3: dict = field(default_factory=dict)


@dataclass
class ChainState:
    visible_regions: list[dict]
    recent_turn_diffs: list[dict]
    marker_neighbor_states: list[dict]
    recent_clicks: list[str]


# --------------------------------------------------------------------
# Whitelisted helper functions (round-5 codex).
# --------------------------------------------------------------------


def _bbox_dict(bbox: Any) -> dict:
    if isinstance(bbox, dict):
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return {"min_x": int(bbox[0]), "min_y": int(bbox[1]),
                "max_x": int(bbox[2]), "max_y": int(bbox[3])}
    return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}


def _markers(chain_state: ChainState) -> list[dict]:
    return [r for r in (chain_state.visible_regions or [])
            if isinstance(r, dict) and r.get("is_multicolor")]


def _non_markers(chain_state: ChainState) -> list[dict]:
    return [r for r in (chain_state.visible_regions or [])
            if isinstance(r, dict) and not r.get("is_multicolor")]


def _to_region_ref(r: dict) -> RegionRef:
    return RegionRef(
        region_id=str(r.get("id", "?")),
        bbox=_bbox_dict(r.get("bbox")),
        color=r.get("color"),
        is_multicolor=bool(r.get("is_multicolor")),
        kind="marker_multicolor" if r.get("is_multicolor") else "non_marker",
        neighbors_3x3=r.get("neighbors_3x3") or {},
    )


def _click_count(chain_state: ChainState, region_id: str) -> int:
    return sum(1 for c in (chain_state.recent_clicks or []) if c == region_id)


def _last_compass_changes(chain_state: ChainState) -> list[dict]:
    diffs = chain_state.recent_turn_diffs or []
    if not diffs:
        return []
    return (diffs[-1] or {}).get("compass_changes") or []


def _recent_lp_regions(chain_state: ChainState, k: int = 5) -> list[str]:
    """Region ids from recent diffs whose level_delta ≥ 1."""
    out = []
    for d in (chain_state.recent_turn_diffs or [])[-k:]:
        if int((d or {}).get("level_delta") or 0) >= 1:
            rid = (d or {}).get("click_region_id")
            if rid and rid != "_outside_":
                out.append(rid)
    return out


def _compass_uniformity(compass: dict) -> bool:
    """All 8 compass cells the same color?"""
    if not isinstance(compass, dict):
        return False
    colors = [v.get("current_color") for d, v in compass.items()
              if isinstance(v, dict)]
    colors = [c for c in colors if c is not None]
    return len(colors) >= 6 and len(set(colors)) == 1


def _marker_compass_state(chain_state: ChainState, marker_id: str) -> dict:
    for m in (chain_state.marker_neighbor_states or []):
        if isinstance(m, dict) and m.get("marker_id") == marker_id:
            return m.get("compass") or {}
    return {}


# --------------------------------------------------------------------
# Coord policies (round-3 C-13).
# --------------------------------------------------------------------


COORD_POLICY_OPTIONS = ("centroid", "sprite_center", "corner_top_left")


def _coord_for(region: RegionRef, policy: str = "centroid") -> list[int]:
    bb = region.bbox if isinstance(region.bbox, dict) else _bbox_dict(region.bbox)
    if policy == "corner_top_left":
        return [int(bb["min_x"]), int(bb["min_y"])]
    if policy == "sprite_center":
        # mid of bbox; for multicolor markers this hits the marker body.
        return [int((bb["min_x"] + bb["max_x"]) / 2),
                int((bb["min_y"] + bb["max_y"]) / 2)]
    # centroid (default): same as sprite_center for our boxes.
    return [int((bb["min_x"] + bb["max_x"]) / 2),
            int((bb["min_y"] + bb["max_y"]) / 2)]


# --------------------------------------------------------------------
# Predicate templates (round-3 §12 step 2 — 12 + P00).
# --------------------------------------------------------------------


PREDICATE_LIBRARY: dict[str, dict] = {}


def _register(template_id: str, complexity: str, coord_policy: str = "centroid"):
    """Round-3 C-12: register decorator that records metadata + body."""
    def deco(fn):
        if not template_id.startswith("P"):
            raise ValueError(f"Template id must start with P: {template_id}")
        PREDICATE_LIBRARY[template_id] = {
            "fn": fn,
            "complexity": complexity,   # "simple" or "compound"
            "coord_policy": coord_policy,
        }
        return fn
    return deco


# --- P00: fallback, always at least one marker ---


@_register("P00_explore_uncovered_marker", complexity="simple",
           coord_policy="sprite_center")
def _p00(chain_state, t):
    markers = _markers(chain_state)
    if not markers:
        return []
    # Region with smallest click count.
    return [_to_region_ref(min(
        markers,
        key=lambda m: _click_count(chain_state, m.get("id", "?")),
    ))]


# --- P01-P12: leak-safe templates ---


@_register("P01_unclicked_neighbor_of_active_marker",
           complexity="simple", coord_policy="centroid")
def _p01(chain_state, t):
    out = []
    for m in _markers(chain_state):
        n3 = m.get("neighbors_3x3") or {}
        rid_set = {v for v in n3.values() if v and v != "_outside_"}
        for r in (chain_state.visible_regions or []):
            rid = r.get("id")
            if rid in rid_set and _click_count(chain_state, rid) == 0:
                out.append(_to_region_ref(r))
    return out


@_register("P02_compass_changed_neighbor_revisit",
           complexity="simple", coord_policy="centroid")
def _p02(chain_state, t):
    last = _last_compass_changes(chain_state)
    changed_compass_anchors = {(c.get("marker_id"), c.get("direction"))
                                for c in last if isinstance(c, dict)}
    if not changed_compass_anchors:
        return []
    out = []
    for marker_id, direction in changed_compass_anchors:
        compass = _marker_compass_state(chain_state, marker_id)
        cell = compass.get(direction)
        if not isinstance(cell, dict):
            continue
        rid = cell.get("region_id")
        for r in (chain_state.visible_regions or []):
            if r.get("id") == rid:
                out.append(_to_region_ref(r))
                break
    return out


@_register("P03_marker_compass_uniform_breaker",
           complexity="compound", coord_policy="centroid")
def _p03(chain_state, t):
    """For markers whose compass is uniform, return non-uniform-color
    visible non-marker regions (clicking would break uniformity)."""
    out = []
    for m in (chain_state.marker_neighbor_states or []):
        if not _compass_uniformity(m.get("compass") or {}):
            continue
        # Find non-marker regions whose color differs from compass-color.
        compass = m.get("compass") or {}
        sample = next((v.get("current_color") for v in compass.values()
                       if isinstance(v, dict)), None)
        for r in _non_markers(chain_state):
            if r.get("color") != sample:
                out.append(_to_region_ref(r))
                if len(out) >= 3:
                    return out
    return out


@_register("P04_marker_compass_uniformity_completer",
           complexity="compound", coord_policy="centroid")
def _p04(chain_state, t):
    """For markers whose compass is NOT uniform, find the odd-color
    cells. Click region MIGHT complete uniformity."""
    out = []
    for m in (chain_state.marker_neighbor_states or []):
        compass = m.get("compass") or {}
        if _compass_uniformity(compass):
            continue
        # Mode color (most common compass color).
        from collections import Counter
        colors = [v.get("current_color") for v in compass.values()
                  if isinstance(v, dict)]
        colors = [c for c in colors if c is not None]
        if not colors:
            continue
        cnt = Counter(colors)
        mode_color = cnt.most_common(1)[0][0]
        # Find compass cells NOT mode color.
        for direction, v in compass.items():
            if isinstance(v, dict) and v.get("current_color") != mode_color:
                rid = v.get("region_id")
                for r in (chain_state.visible_regions or []):
                    if r.get("id") == rid:
                        out.append(_to_region_ref(r))
                        if len(out) >= 3:
                            return out
                        break
    return out


@_register("P05_shared_neighbor_between_markers",
           complexity="compound", coord_policy="centroid")
def _p05(chain_state, t):
    markers = _markers(chain_state)
    if len(markers) < 2:
        return []
    out = []
    for i, m1 in enumerate(markers):
        n1 = set((m1.get("neighbors_3x3") or {}).values())
        for m2 in markers[i + 1:]:
            n2 = set((m2.get("neighbors_3x3") or {}).values())
            shared = (n1 & n2) - {None, "_outside_"}
            for rid in shared:
                for r in (chain_state.visible_regions or []):
                    if r.get("id") == rid:
                        out.append(_to_region_ref(r))
                        if len(out) >= 3:
                            return out
                        break
    return out


@_register("P06_recently_l_plus_region_kin",
           complexity="simple", coord_policy="centroid")
def _p06(chain_state, t):
    lp = _recent_lp_regions(chain_state, k=10)
    if not lp:
        return []
    out = []
    for r in (chain_state.visible_regions or []):
        rid = r.get("id")
        if rid and rid != "_outside_" and rid not in lp:
            # Same color or kind as a recent L+ region?
            for lp_rid in lp:
                lp_r = next((rr for rr in chain_state.visible_regions
                             if rr.get("id") == lp_rid), None)
                if lp_r and r.get("color") == lp_r.get("color") \
                        and r.get("is_multicolor") == lp_r.get("is_multicolor"):
                    out.append(_to_region_ref(r))
                    break
        if len(out) >= 3:
            return out
    return out


@_register("P07_repeat_click_parity_revert",
           complexity="simple", coord_policy="centroid")
def _p07(chain_state, t):
    """Region clicked exactly once in chain — clicking again may revert."""
    out = []
    for r in (chain_state.visible_regions or []):
        rid = r.get("id")
        if rid and _click_count(chain_state, rid) == 1:
            out.append(_to_region_ref(r))
            if len(out) >= 3:
                return out
    return out


@_register("P08_unique_color_neighbor_of_marker",
           complexity="compound", coord_policy="centroid")
def _p08(chain_state, t):
    out = []
    for m in _markers(chain_state):
        n3 = m.get("neighbors_3x3") or {}
        rid_set = {v for v in n3.values() if v and v != "_outside_"}
        # Group neighbors by color.
        from collections import Counter
        nbr_colors = []
        for r in (chain_state.visible_regions or []):
            if r.get("id") in rid_set:
                nbr_colors.append(r.get("color"))
        if not nbr_colors:
            continue
        cnt = Counter(c for c in nbr_colors if c is not None)
        unique_colors = {c for c, n in cnt.items() if n == 1}
        for r in (chain_state.visible_regions or []):
            if r.get("id") in rid_set and r.get("color") in unique_colors:
                out.append(_to_region_ref(r))
                if len(out) >= 3:
                    return out
    return out


@_register("P09_compass_changed_then_unchanged",
           complexity="compound", coord_policy="centroid")
def _p09(chain_state, t):
    """Marker whose compass changed last turn but NOT this turn — likely settling."""
    diffs = chain_state.recent_turn_diffs or []
    if len(diffs) < 2:
        return []
    last_cc = diffs[-1].get("compass_changes") or []
    prev_cc = diffs[-2].get("compass_changes") or []
    last_markers = {c.get("marker_id") for c in last_cc if isinstance(c, dict)}
    prev_markers = {c.get("marker_id") for c in prev_cc if isinstance(c, dict)}
    target_markers = prev_markers - last_markers
    if not target_markers:
        return []
    out = []
    for m in _markers(chain_state):
        if m.get("id") in target_markers:
            out.append(_to_region_ref(m))
            if len(out) >= 3:
                return out
    return out


@_register("P10_marker_no_recent_compass_change",
           complexity="simple", coord_policy="centroid")
def _p10(chain_state, t):
    """Markers whose compass has NOT changed in last 5 turns — may need probing."""
    recent_5 = (chain_state.recent_turn_diffs or [])[-5:]
    seen_markers: set[str] = set()
    for d in recent_5:
        for c in (d or {}).get("compass_changes") or []:
            if isinstance(c, dict):
                seen_markers.add(c.get("marker_id"))
    out = []
    for m in _markers(chain_state):
        if m.get("id") not in seen_markers:
            n3 = m.get("neighbors_3x3") or {}
            rid_set = {v for v in n3.values() if v and v != "_outside_"}
            for r in (chain_state.visible_regions or []):
                if r.get("id") in rid_set:
                    out.append(_to_region_ref(r))
                    break
            if len(out) >= 3:
                return out
    return out


@_register("P11_marker_with_two_uniform_compass_directions",
           complexity="compound", coord_policy="centroid")
def _p11(chain_state, t):
    """Two specific compass directions of a marker have the same color
    while others differ — probing one of them may push toward uniformity."""
    out = []
    for m in (chain_state.marker_neighbor_states or []):
        compass = m.get("compass") or {}
        from collections import Counter
        colors = [v.get("current_color") for d, v in compass.items()
                  if isinstance(v, dict)]
        colors = [c for c in colors if c is not None]
        if not colors:
            continue
        cnt = Counter(colors)
        mode = cnt.most_common(1)[0][0]
        if cnt[mode] < 2:
            continue
        for direction, v in compass.items():
            if isinstance(v, dict) and v.get("current_color") == mode:
                rid = v.get("region_id")
                for r in (chain_state.visible_regions or []):
                    if r.get("id") == rid:
                        out.append(_to_region_ref(r))
                        if len(out) >= 3:
                            return out
                        break
    return out


@_register("P12_recent_click_no_change_revisit",
           complexity="simple", coord_policy="centroid")
def _p12(chain_state, t):
    """Recent click → no compass change. Re-clicking may have effect now."""
    diffs = chain_state.recent_turn_diffs or []
    if not diffs:
        return []
    last = diffs[-1]
    rid = (last or {}).get("click_region_id")
    cc = (last or {}).get("compass_changes") or []
    if not rid or rid == "_outside_" or cc:
        return []
    for r in (chain_state.visible_regions or []):
        if r.get("id") == rid:
            return [_to_region_ref(r)]
    return []


# --------------------------------------------------------------------
# Anti-leak audit on PREDICATE_LIBRARY.
# --------------------------------------------------------------------


_FORBIDDEN_VOCAB = re.compile(
    r"per_neighbor_target|needs_toggle|marker_progress|joint_neighbors|"
    r"expected_neighbor_colors|target_color|win.?state|win.?condition|"
    r"target.?state|complete the (level|pattern|grid)|"
    r"correct color|right color|should be color|must be color|"
    r"match the indicator|match the pattern|the goal is|"
    r"satisfy",
    re.IGNORECASE,
)


def _audit_template_id(template_id: str) -> bool:
    if template_id not in PREDICATE_LIBRARY:
        return False
    if _FORBIDDEN_VOCAB.search(template_id):
        return False
    return True


# --------------------------------------------------------------------
# Beta-Bernoulli score (round-2 C-2 + round-3 C-11 P00 cap).
# --------------------------------------------------------------------


def _beta_bernoulli_score(template_id: str, template_history: dict) -> float:
    h = template_history.get(template_id, {}) if template_history else {}
    s = int(h.get("supported", 0))
    r = int(h.get("refuted", 0))
    meta = PREDICATE_LIBRARY.get(template_id, {})
    if meta.get("complexity") == "simple":
        alpha, beta = 2, 1
    else:
        alpha, beta = 1, 1
    score = (alpha + s) / (alpha + beta + s + r)
    # Round-3 C-11 P00 cap.
    if template_id == "P00_explore_uncovered_marker":
        score = min(score, 0.30)
    return round(score, 4)


# --------------------------------------------------------------------
# Top-level entry: generate_predicates.
# --------------------------------------------------------------------


def build_chain_state_from_inputs(
    *, visible_regions: list[dict],
    recent_turn_diffs: list[dict],
    marker_neighbor_states: list[dict],
    recent_clicks: list[str],
) -> ChainState:
    return ChainState(
        visible_regions=list(visible_regions or []),
        recent_turn_diffs=list(recent_turn_diffs or []),
        marker_neighbor_states=list(marker_neighbor_states or []),
        recent_clicks=list(recent_clicks or []),
    )


def generate_predicates(
    *,
    chain_state: ChainState,
    turn_index: int,
    template_history: dict | None = None,
    k_global: int = 12,
) -> list[dict]:
    """Returns up to k_global grounded predicate-region tests, each
    leak-safe and accompanied by Symbolica-derived suggested_coord."""
    template_history = template_history or {}
    out: list[dict] = []
    for template_id, meta in PREDICATE_LIBRARY.items():
        if not _audit_template_id(template_id):
            continue
        try:
            regions = meta["fn"](chain_state, turn_index) or []
        except Exception:
            continue
        for reg in regions[:3]:   # round-2 C-4: cap regions per predicate
            score = _beta_bernoulli_score(template_id, template_history)
            policy = meta.get("coord_policy") or "centroid"
            coord = _coord_for(reg, policy)
            pid = f"{template_id}:T{turn_index}:{reg.region_id}"
            out.append({
                "predicate_id": pid,
                "template_id": template_id,
                "anchor_region": {
                    "region_id": reg.region_id,
                    "bbox": reg.bbox,
                    "color": reg.color,
                    "is_multicolor": reg.is_multicolor,
                    "kind": reg.kind,
                },
                "score": score,
                "coord_policy": policy,
                "suggested_coord": coord,
            })
    out.sort(key=lambda c: -c.get("score", 0))
    return out[:k_global]


# --------------------------------------------------------------------
# Verdict update (round-5 codex action-conditioned).
# --------------------------------------------------------------------


def update_predicate_log_with_observation(
    *, predicate_log: list[dict],
    observation: dict,
    click_region_id: str | None,
    click_coord: list[int] | None,
    turn_now: int,
) -> list[dict]:
    """Action-conditioned verdict per codex round-5:
       P held + clicked matching region + L+ within K = supported
       P held + clicked matching region + no L+ = refuted
       P held + not clicked = untested (no verdict)
       P not held = inconclusive (no verdict)

    Round-2 C-5: accept clicks INSIDE anchor_region.bbox even if
    primary_region_id differs (game segmentation can shift)."""
    out = []
    if not isinstance(observation, dict):
        return out
    ld = int(observation.get("level_delta") or 0)
    obs_rid = observation.get("primary_region_id") or ""
    cx, cy = (click_coord or [None, None])[:2]

    for c in predicate_log:
        if c.get("verdict") in ("supported", "refuted"):
            continue
        anchor = c.get("anchor_region") or {}
        anchor_rid = anchor.get("region_id")
        anchor_bbox = anchor.get("bbox") or {}

        # Action-conditioned match.
        rid_match = (click_region_id == anchor_rid) or (obs_rid == anchor_rid)
        bbox_match = False
        if (cx is not None and cy is not None and
                isinstance(anchor_bbox, dict)):
            try:
                bbox_match = (
                    int(anchor_bbox["min_x"]) <= int(cx) <= int(anchor_bbox["max_x"])
                    and int(anchor_bbox["min_y"]) <= int(cy) <= int(anchor_bbox["max_y"])
                )
            except (KeyError, TypeError, ValueError):
                bbox_match = False
        action_matches = rid_match or bbox_match
        if not action_matches:
            continue

        emitted = int(c.get("emitted_at_turn", turn_now))
        turns_since = turn_now - emitted
        verdict = None
        if ld >= 1 and turns_since <= 3:
            verdict = "supported"
        elif ld == 0 and turns_since >= 1:
            verdict = "refuted"
        if verdict:
            c["verdict"] = verdict
            c["resolved_at_turn"] = turn_now
            out.append({
                "predicate_id": c.get("predicate_id"),
                "template_id": c.get("template_id"),
                "verdict": verdict,
                "resolved_at_turn": turn_now,
            })
    return out

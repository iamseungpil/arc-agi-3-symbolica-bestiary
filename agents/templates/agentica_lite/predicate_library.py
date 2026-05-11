"""Static predicate library + sandbox installer (plan §6.4).

Predicate ABI: P(chain_state, t) -> list[RegionRef], plus a coord_policy enum.
LLM-extended predicates pass through an AST-guarded sandbox before install.
"""

from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RegionRef:
    region_id: str
    bbox: dict | list
    color: int | None = None
    is_multicolor: bool = False
    kind: str = "non_marker"


@dataclass
class Predicate:
    id: str
    family: str
    coord_policy: str  # "centroid" | "sprite_center" | "corner_top_left"
    fn: Callable[[Any, int], list[RegionRef]]
    is_static: bool
    install_ts: float | None = None


def _regions_of(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return state.get("visible_regions", [])
    return getattr(state, "visible_regions", []) or []


def _to_region_ref(r: Any) -> RegionRef:
    if isinstance(r, RegionRef):
        return r
    if isinstance(r, dict):
        rid = r.get("region_id") or r.get("id") or "?"
        return RegionRef(rid, r.get("bbox", {}), r.get("color"),
                         bool(r.get("is_multicolor", False)),
                         r.get("kind", "non_marker"))
    return RegionRef(str(r), {})


def _p00_fallback(state: Any, t: int) -> list[RegionRef]:
    return [_to_region_ref(r) for r in _regions_of(state)]


def _p01_dominant_transition(state: Any, t: int) -> list[RegionRef]:
    regions = _regions_of(state)
    target = None
    if isinstance(state, dict):
        dt = (state.get("last_observation") or {}).get("dominant_transition") or {}
        target = dt.get("to")
    if target is None:
        return [_to_region_ref(r) for r in regions]
    return [_to_region_ref(r) for r in regions
            if (r.get("color") if isinstance(r, dict) else getattr(r, "color", None)) == target]


def _p02_marker_align(state: Any, t: int) -> list[RegionRef]:
    out = []
    for r in _regions_of(state):
        if isinstance(r, dict):
            if r.get("is_marker_neighbor") or r.get("is_primary_marker"):
                out.append(_to_region_ref(r))
        elif getattr(r, "is_marker_neighbor", False) or getattr(r, "is_primary_marker", False):
            out.append(_to_region_ref(r))
    return out


def _p03_sector_alignment(state: Any, t: int) -> list[RegionRef]:
    out = []
    for r in _regions_of(state):
        band = r.get("y_band") if isinstance(r, dict) else getattr(r, "y_band", None)
        if band == "play_zone":
            out.append(_to_region_ref(r))
    return out


def _stub_predicate(state: Any, t: int) -> list[RegionRef]:
    return [_to_region_ref(r) for r in _regions_of(state)]


def _p12_saturation_progress(state: Any, t: int) -> list[RegionRef]:
    """v601 §3.9 G20: unclicked compass slots of the target marker.

    state["target_marker_id"] is injected by Proposer or fallback (G6).
    Returns RegionRefs for visible regions whose region_id matches an
    unclicked compass slot (clicks == 0) of the target marker.
    """
    if isinstance(state, dict):
        target_marker_id = state.get("target_marker_id")
        markers = state.get("marker_neighbor_states") or []
        regions = state.get("visible_regions") or []
    else:
        target_marker_id = getattr(state, "target_marker_id", None)
        markers = getattr(state, "marker_neighbor_states", None) or []
        regions = getattr(state, "visible_regions", None) or []
    if target_marker_id is None:
        return []
    target_marker = None
    for m in markers:
        if (m.get("marker_id") if isinstance(m, dict) else getattr(m, "marker_id", None)) == target_marker_id:
            target_marker = m
            break
    if target_marker is None:
        return []
    compass = target_marker.get("compass") if isinstance(target_marker, dict) else getattr(target_marker, "compass", {})
    if not isinstance(compass, dict):
        return []
    unclicked_rids: set[str] = set()
    for slot in compass.values():
        if not isinstance(slot, dict):
            continue
        if int(slot.get("clicks", 0) or 0) == 0:
            rid = slot.get("region_id")
            if rid:
                unclicked_rids.add(rid)
    out: list[RegionRef] = []
    for r in regions:
        rid = r.get("region_id") or r.get("id") if isinstance(r, dict) else (
            getattr(r, "region_id", None) or getattr(r, "id", None)
        )
        if rid in unclicked_rids:
            out.append(_to_region_ref(r))
    return out


STATIC_PREDICATES: dict[str, Predicate] = {}


def _register(p: Predicate) -> None:
    STATIC_PREDICATES[p.id] = p


_register(Predicate("P00_fallback", "fallback", "centroid", _p00_fallback, True))
_register(Predicate("P01_dominant_transition", "dominant_transition", "centroid", _p01_dominant_transition, True))
_register(Predicate("P02_marker_align", "marker_align", "centroid", _p02_marker_align, True))
_register(Predicate("P03_sector_alignment", "sector_alignment", "centroid", _p03_sector_alignment, True))
for _pid, _fam in [
    ("P04_color_invariance", "color_invariance"),
    ("P05_size_anchor", "size_anchor"),
    ("P06_neighbor_pair", "neighbor_pair"),
    ("P07_corner_align", "corner_align"),
    ("P08_unclicked_neighbor", "unclicked_neighbor"),
    ("P09_compass_change", "compass_change"),
    ("P10_marker_no_recent", "marker_no_recent"),
    ("P11_recent_response", "recent_response"),
]:
    _register(Predicate(
        _pid, _fam,
        "corner_top_left" if "corner" in _pid else "centroid",
        _stub_predicate, True,
    ))
# v601 §3.9 G20: saturation-progress predicate (drives target marker to completion).
_register(Predicate(
    "P12_saturation_progress", "saturation_progress", "centroid",
    _p12_saturation_progress, True,
))
# Alias for proposer-emitted IDs (plan §3.9 P_saturation_progress).
STATIC_PREDICATES["P_saturation_progress"] = STATIC_PREDICATES["P12_saturation_progress"]


P_TEST_FRAME: dict[str, Any] = {
    "visible_regions": [{
        "id": "R_test", "region_id": "R_test",
        "bbox": {"min_x": 0, "min_y": 0, "max_x": 4, "max_y": 4},
        "color": 0, "is_multicolor": False, "y_band": "play_zone",
        "is_marker_neighbor": False, "is_primary_marker": False,
    }],
    "last_observation": {"dominant_transition": {"from": 1, "to": 0, "count": 1}},
}


@dataclass
class SandboxResult:
    accepted: bool
    reason: str
    runtime_ms: float
    fingerprint: str | None = None


_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "__builtins__", "__class__", "__bases__", "__subclasses__",
    "breakpoint", "memoryview",
}
_ALLOWED_BUILTINS = {
    "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "range", "len", "abs", "min", "max", "sum", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "any", "all", "round",
    "True", "False", "None",
}


class _ASTGuard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def visit_Import(self, node):  # noqa: N802
        self.error = "exec_blocked: import"

    def visit_ImportFrom(self, node):  # noqa: N802
        self.error = "exec_blocked: from-import"

    def visit_Attribute(self, node):  # noqa: N802
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.error = f"exec_blocked: dunder access {node.attr!r}"
        self.generic_visit(node)

    def visit_Name(self, node):  # noqa: N802
        if node.id in _FORBIDDEN_NAMES:
            self.error = f"exec_blocked: forbidden name {node.id!r}"
        self.generic_visit(node)

    def visit_Call(self, node):  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
            self.error = f"exec_blocked: call to {node.func.id!r}"
        self.generic_visit(node)


def _region_id_of(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("region_id")
    return getattr(item, "region_id", None)


class PredicateLibrary:
    def __init__(self) -> None:
        self.static: dict[str, Predicate] = dict(STATIC_PREDICATES)
        self.installed: dict[str, Predicate] = {}
        self.install_log: list[dict] = []

    def install(
        self,
        source_text: str,
        llm_provided_lambda: Any = None,
        predicate_id: str | None = None,
        family: str = "extended",
        coord_policy: str = "centroid",
    ) -> SandboxResult:
        t0 = time.monotonic()
        try:
            tree = ast.parse(source_text, mode="exec")
        except SyntaxError as e:
            return SandboxResult(False, f"type_error: {e.msg}", (time.monotonic() - t0) * 1000)
        guard = _ASTGuard()
        guard.visit(tree)
        if guard.error is not None:
            return SandboxResult(False, guard.error, (time.monotonic() - t0) * 1000)

        builtin_obj = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        safe_builtins = {n: builtin_obj[n] for n in _ALLOWED_BUILTINS if n in builtin_obj}
        ns: dict[str, Any] = {"__builtins__": safe_builtins, "RegionRef": RegionRef}
        try:
            exec(compile(tree, "<sandbox>", "exec"), ns)  # noqa: S102
        except Exception as e:
            return SandboxResult(False, f"exec_blocked: {type(e).__name__}: {e}",
                                 (time.monotonic() - t0) * 1000)

        fn = ns.get("fn") or llm_provided_lambda
        if not callable(fn):
            return SandboxResult(False, "type_error: no callable `fn`",
                                 (time.monotonic() - t0) * 1000)

        deadline = t0 + 5.0
        try:
            out = fn(P_TEST_FRAME, 0)
        except Exception as e:
            return SandboxResult(False, f"ABI_violation: {type(e).__name__}: {e}",
                                 (time.monotonic() - t0) * 1000)
        if time.monotonic() > deadline:
            return SandboxResult(False, "timeout", (time.monotonic() - t0) * 1000)
        if not isinstance(out, list):
            return SandboxResult(False, "ABI_violation: non-list return",
                                 (time.monotonic() - t0) * 1000)
        for item in out:
            if _region_id_of(item) is None:
                return SandboxResult(False, "ABI_violation: missing region_id",
                                     (time.monotonic() - t0) * 1000)

        try:
            sig0 = [_region_id_of(x) for x in out]
            for _ in range(9):
                if [_region_id_of(x) for x in fn(P_TEST_FRAME, 0)] != sig0:
                    return SandboxResult(False, "non_deterministic",
                                         (time.monotonic() - t0) * 1000)
        except Exception as e:
            return SandboxResult(False, f"ABI_violation: {type(e).__name__}: {e}",
                                 (time.monotonic() - t0) * 1000)

        runtime_ms = (time.monotonic() - t0) * 1000
        fingerprint = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]
        pid = predicate_id or f"L_ext_{fingerprint[:8]}"
        self.installed[pid] = Predicate(pid, family, coord_policy, fn, False, time.time())
        self.install_log.append({"id": pid, "fingerprint": fingerprint,
                                 "runtime_ms": runtime_ms, "reason": "ok"})
        return SandboxResult(True, "ok", runtime_ms, fingerprint)

    def resolve_coord(self, predicate_id: str, region: Any, state: Any = None) -> tuple[int, int]:
        pred = self.static.get(predicate_id) or self.installed.get(predicate_id)
        policy = pred.coord_policy if pred is not None else "centroid"
        bbox = region.get("bbox") if isinstance(region, dict) else getattr(region, "bbox", None)
        if isinstance(bbox, dict):
            x_min = int(bbox.get("min_x", 0)); y_min = int(bbox.get("min_y", 0))
            x_max = int(bbox.get("max_x", x_min)); y_max = int(bbox.get("max_y", y_min))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x_min, y_min, x_max, y_max = (int(v) for v in bbox[:4])
        else:
            return (0, 0)
        if policy == "corner_top_left":
            return (x_min, y_min)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        # v607 P3 (codex pivot 3): for UNKNOWN predicate (Reflector-emitted), rotate
        # through 9 positions of the region's bbox 3x3 grid based on turn_count.
        # Fixes the centroid-only fallback that caused 0 L+ across cycles 300-304.
        if pred is None and isinstance(state, dict):
            turn = state.get("turn_count")
            if isinstance(turn, int) and turn >= 0:
                # 3x3 grid offsets: indices 0..8
                # 0=NW, 1=N, 2=NE, 3=W, 4=center, 5=E, 6=SW, 7=S, 8=SE
                idx = turn % 9
                xs = [x_min, cx, x_max]
                ys = [y_min, cy, y_max]
                return (xs[idx % 3], ys[idx // 3])
        return (cx, cy)

    def all_predicates(self) -> dict[str, Predicate]:
        return {**self.static, **self.installed}

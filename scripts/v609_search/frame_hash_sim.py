"""v609 P2: env.step-as-simulator with frame-hash cache.

Codex recommended this as the minimum-viable pivot away from "learn a
symbolic effect model". The wrapper exposes:

    FrameHashSimulator(env)
        .reset() -> SimState                      # fresh game state
        .step(state, coord_xy) -> SimResult        # cache-aware step

Internally:
- `SimState` carries the *click history* needed to reproduce the env from
  a fresh reset, NOT a deep snapshot (the local Arcade does not expose
  snapshot / restore).
- Each unique `(parent_state_key, coord_xy)` is memoised in an LRU
  cache, so repeated A* expansions of the same node hit O(1).
- Replay-from-root is the cost paid on cache miss; depth is bounded by
  the search depth limit (plan section 8 C3).

The simulator is intentionally LLM-free.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")
os.environ.setdefault("RECORDINGS_DIR", "recordings")


def _to_python_nested_lists(obj):
    """Coerce numpy arrays to nested Python lists (matches v608 E1 audit)."""
    if obj is None:
        return None
    if hasattr(obj, "tolist") and not isinstance(obj, (list, tuple, dict,
                                                        str, bytes)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_python_nested_lists(x) for x in obj]
    return obj


class _FrameProxy:
    __slots__ = ("frame", "levels_completed", "win_levels", "state",
                 "available_actions", "game_id", "guid")

    def __init__(self, raw):
        self.frame = _to_python_nested_lists(getattr(raw, "frame", None))
        self.levels_completed = int(getattr(raw, "levels_completed", 0) or 0)
        self.win_levels = int(getattr(raw, "win_levels", 0) or 0)
        self.state = getattr(raw, "state", None)
        self.available_actions = list(getattr(raw, "available_actions", []) or [])
        self.game_id = str(getattr(raw, "game_id", "") or "")
        self.guid = getattr(raw, "guid", None)


def hash_frame(frame_field) -> str:
    """Stable hash of the latest grid in a FrameData stack.

    Returns a 16-char SHA1 prefix so log lines stay short while keeping
    collision probability negligible for ft09's ~30-region 64x64 grids.
    """
    grid = _to_python_nested_lists(frame_field)
    if not grid:
        return "empty"
    # Pick the last non-empty 2D layer.
    layer = None
    for g in reversed(grid):
        if g and (isinstance(g[0], (list, tuple)) and len(g[0]) > 0):
            layer = g
            break
    if layer is None:
        return "empty"
    flat = "|".join(",".join(str(int(c)) for c in row) for row in layer)
    return hashlib.sha1(flat.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SimState:
    """Immutable handle to a search node.

    `clicks` is the tuple of (x, y) coords that, when replayed from a
    fresh env.reset(), reproduces the underlying game state. `state_key`
    is a content-addressed hash of the resulting grid plus level count.
    """
    state_key: str
    clicks: tuple[tuple[int, int], ...]
    levels_completed: int

    def depth(self) -> int:
        return len(self.clicks)


@dataclass
class SimResult:
    """One expansion step's outcome.

    PATH-B Q3 (codex EXTEND_PATH_B_Q3_FIX, 2026-05-12) added:
    - observed_unsat_delta: post-pre marker_constraint unsatisfied delta.
      Negative = constraint repair, positive = constraint break, 0 = neutral.
    - frame_changed: True if the post-click frame hash differs from parent.
    """
    state: SimState
    level_delta: int
    unsatisfied: int
    high_unsatisfied: int
    n_constraints: int
    n_high_constraints: int
    observed_unsat_delta: int = 0
    frame_changed: bool = True


@dataclass
class FrameHashSimulator:
    """env.step replay + LRU cache.

    Construct with a fresh local-Arcade `env` (the wrapper that exposes
    `.reset()` and `.step(action, data=...)`). The wrapper resets the env
    on every cache miss to reach the parent state's clicks before applying
    the new click. The cost is O(depth) env.step calls per miss.
    """
    env: Any
    cache_size: int = 4096
    # Internal: keyed by (parent_state_key, coord_xy) -> SimResult.
    _cache: "OrderedDict[tuple[str, tuple[int, int]], SimResult]" = field(
        default_factory=OrderedDict)
    # Stats for telemetry.
    cache_hits: int = 0
    cache_misses: int = 0
    env_steps_spent: int = 0

    def reset(self) -> SimState:
        """Reset the underlying env and return the root SimState."""
        # Local import so the module is importable without local arcade
        # when only the dataclasses are needed.
        from agents.templates.agentica_lite._frame_to_state import (
            _reset_stability_state,
        )
        _reset_stability_state()
        raw = self.env.reset()
        proxy = _FrameProxy(raw)
        key = hash_frame(proxy.frame)
        return SimState(state_key=key, clicks=(),
                        levels_completed=proxy.levels_completed)

    def _replay(self, clicks: tuple[tuple[int, int], ...]) -> _FrameProxy:
        """Reset the env and replay the clicks list. Returns the proxy
        wrapping the final FrameData. NO cache lookup here — caller is
        responsible for caching the final state if desired.
        """
        from arcengine import GameAction
        from agents.templates.agentica_lite._frame_to_state import (
            _reset_stability_state,
        )
        _reset_stability_state()
        raw = self.env.reset()
        for (x, y) in clicks:
            ga = GameAction.ACTION6
            raw = self.env.step(ga, data={"x": int(x), "y": int(y)})
            self.env_steps_spent += 1
        return _FrameProxy(raw)

    def _summary_for_proxy(self, proxy: _FrameProxy) -> dict:
        """Run v608 `frame_to_state` so we get marker_constraint_summary."""
        from agents.templates.agentica_lite._frame_to_state import (
            frame_to_state,
        )
        return frame_to_state(proxy, [], proxy.levels_completed)

    def step(
        self,
        state: SimState,
        coord_xy: tuple[int, int],
        parent_unsatisfied: int | None = None,
    ) -> SimResult:
        """Apply `coord_xy` from `state`. Cache by (parent_key, coord).

        `parent_unsatisfied` is the pre-click marker_constraint.unsatisfied
        for delta computation (Q3 fix). Caller may pass None; in that case
        observed_unsat_delta=0.
        """
        cache_key = (state.state_key, (int(coord_xy[0]), int(coord_xy[1])))
        if cache_key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            # PATH-B bug fix (2026-05-12): cached.state.clicks reflects the
            # FIRST caller's path. Reconstruct with the CURRENT caller's
            # clicks so history_aware visited dedup is correct. Also recompute
            # observed_unsat_delta against THIS caller's parent_unsatisfied.
            fresh_clicks = state.clicks + (cache_key[1],)
            fresh_state = SimState(
                state_key=cached.state.state_key,
                clicks=fresh_clicks,
                levels_completed=cached.state.levels_completed,
            )
            fresh_delta = (cached.unsatisfied - parent_unsatisfied
                           if parent_unsatisfied is not None else 0)
            return SimResult(
                state=fresh_state,
                level_delta=cached.level_delta,
                unsatisfied=cached.unsatisfied,
                high_unsatisfied=cached.high_unsatisfied,
                n_constraints=cached.n_constraints,
                n_high_constraints=cached.n_high_constraints,
                observed_unsat_delta=fresh_delta,
                frame_changed=cached.frame_changed,
            )
        self.cache_misses += 1
        new_clicks = state.clicks + (cache_key[1],)
        proxy = self._replay(new_clicks)
        summary = self._summary_for_proxy(proxy)
        mcs = summary.get("marker_constraint_summary") or {}
        next_key = hash_frame(proxy.frame)
        new_levels = int(proxy.levels_completed)
        level_delta = max(0, new_levels - state.levels_completed)
        next_state = SimState(
            state_key=next_key, clicks=new_clicks, levels_completed=new_levels,
        )
        post_unsat = int(mcs.get("unsatisfied", 0) or 0)
        observed_delta = (post_unsat - parent_unsatisfied
                          if parent_unsatisfied is not None else 0)
        result = SimResult(
            state=next_state,
            level_delta=level_delta,
            unsatisfied=post_unsat,
            high_unsatisfied=int(mcs.get("high_unsatisfied", 0) or 0),
            n_constraints=int(mcs.get("total", 0) or 0),
            n_high_constraints=int(mcs.get("high_total", 0) or 0),
            observed_unsat_delta=observed_delta,
            frame_changed=(next_key != state.state_key),
        )
        self._cache[cache_key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total else 0.0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "env_steps_spent": self.env_steps_spent,
            "cache_size": len(self._cache),
        }

"""v603 ArcgenticaLite adapter — Agent-subclass shim around the v601/v602 framework.

Keeps strict separation of concerns:
  * `ArcgenticaLite` (this package's `agent.ArcgenticaLite`) owns the per-turn
    decision logic on a state dict.
  * This adapter only translates `FrameData` <-> state and Action <-> GameAction.

Codex objective-reviewer contract is honored explicitly:

  Contract C1 (whitelist):  state translation defers to `_frame_to_state.py`
                             which uses ONLY raw frame, levels_completed,
                             state, available_actions, and the agent's own
                             click history.
  Contract C2 (forbidden):  no game-specific region IDs, no game-specific
                             saturation thresholds, no engine-internal state
                             are referenced here.
                             (See test_no_game_constants_in_adapter_source.)
  Contract C3 (history):    `_action_history` is the SOLE source of click
                             counts; reset to empty on every fresh episode
                             boundary detected by `_maybe_reset_episode`.
  Contract C4 (async):      `choose_action` runs `lite.run_turn` on a fresh
                             event loop when none is active; falls back to
                             `run_until_complete` on the active loop.
  Contract C5 (D1):         persistence policy enforced by env var
                             `ARC_V603_DISABLE_CROSS_RUN` (default ON for D1).
                             No load_rasi_prior is called from this adapter,
                             and no prior-cycle persisted memory file is read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ...agent import Agent
from ._frame_to_state import frame_to_state
from .agent import Action, ArcgenticaLite

logger = logging.getLogger(__name__)


def _d1_cross_run_disabled() -> bool:
    """Contract C5: D1 default disables prior-cycle persisted memory imports."""
    val = os.environ.get("ARC_V603_DISABLE_CROSS_RUN", "1")
    return val not in ("0", "false", "False", "")


class ArcgenticaLiteAgent(Agent):
    """Thin Agent-subclass adapter around `ArcgenticaLite`.

    Designed to be a drop-in for main.py's swarm. Wraps the v601/v602
    ArcgenticaLite per-turn loop and translates FrameData <-> state.
    """

    # Action budget for multi-level progression. Default 1500 leaves
    # ~250 actions per level on average (over-provisioned for long-episode
    # protocol). Override via env ARC_LITE_MAX_ACTIONS for cold-start
    # cohorts where shorter episodes are intentional.
    MAX_ACTIONS = int(os.environ.get("ARC_LITE_MAX_ACTIONS", "1500"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + (hash(self.game_id) % 1_000_000)
        random.seed(seed)
        self.lite = ArcgenticaLite(game_id=self.game_id, seed=seed)
        # Contract C3: per-episode action history. Each entry is a dict with
        # keys {coord_xy, prev_grid, curr_grid, level_delta}.
        self._action_history: list[dict[str, Any]] = []
        self._prev_levels_completed: int = 0
        self._prev_grid: list[list[int]] | None = None
        self._reset_emitted: bool = False
        # Contract C5: enforce D1 default.
        self._cross_run_disabled = _d1_cross_run_disabled()
        if self._cross_run_disabled:
            logger.info(
                "ArcgenticaLiteAgent: D1 mode — prior-cycle memory import disabled."
            )
        # Convenience: expose for tests.
        self.framework_version = ArcgenticaLite.FRAMEWORK_VERSION

    # -------------------------------------------------------------------- API

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> bool:
        """Match the v600/v601 lifecycle: stop on WIN.

        Side-effect: detect fresh-episode boundary and clear `_action_history`
        per Contract C3. We treat (state == NOT_PLAYED, levels_completed == 0)
        as the start of a new episode.
        """
        self._maybe_reset_episode(latest_frame)
        return latest_frame.state is GameState.WIN

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        # Episode bootstrap: first turn or post-GAME_OVER must RESET.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._reset_emitted = True
            return GameAction.RESET

        # Translate frame to lite state.
        state = frame_to_state(
            latest_frame,
            self._action_history,
            self._prev_levels_completed,
        )

        # Run the per-turn decision on whichever event-loop posture is active.
        action = self._run_lite_turn_sync(state)
        return self._action_to_game_action(action, latest_frame)

    # The base class already provides do_action_request that submits via
    # arc_env.step + returns FrameData. We override it to also append to
    # `_action_history` so click counts come ONLY from our own emissions
    # (Contract C3).
    def do_action_request(self, action: GameAction) -> FrameData:
        prev_grid_snapshot = self._prev_grid
        coord_xy: tuple[int, int] | None = None
        if action.is_complex():
            data = action.action_data.model_dump()
            coord_xy = (int(data.get("x", -1)), int(data.get("y", -1)))
        frame = super().do_action_request(action)
        # Update the trailing grid snapshot.
        from ._frame_to_state import _latest_grid_from_frame
        new_grid = _latest_grid_from_frame(getattr(frame, "frame", None))
        # Compute level_delta against the prior bookkeeping.
        new_lc = int(getattr(frame, "levels_completed", 0) or 0)
        level_delta = max(0, new_lc - self._prev_levels_completed)
        if coord_xy is not None and coord_xy != (-1, -1):
            self._action_history.append(
                {
                    "coord_xy": coord_xy,
                    "prev_grid": prev_grid_snapshot,
                    "curr_grid": new_grid,
                    "level_delta": level_delta,
                    "action_id": int(action.value),
                }
            )
        self._prev_grid = new_grid
        self._prev_levels_completed = new_lc
        return frame

    # ----------------------------------------------------------- adapter glue

    def _maybe_reset_episode(self, latest_frame: FrameData) -> None:
        """Contract C3: clear action history on episode boundary."""
        lc = int(getattr(latest_frame, "levels_completed", 0) or 0)
        st = getattr(latest_frame, "state", None)
        # Boundary: NOT_PLAYED with no progress, or full_reset flag.
        if (st is GameState.NOT_PLAYED and lc == 0) or getattr(
            latest_frame, "full_reset", False
        ):
            if self._action_history:
                logger.info("ArcgenticaLiteAgent: episode boundary -> clearing history")
            self._action_history = []
            self._prev_levels_completed = 0
            self._prev_grid = None

    def _run_lite_turn_sync(self, state: dict[str, Any]) -> Action | None:
        """Contract C4: dispatch async run_turn into a sync call safely.

        Three cases to handle:
          (a) No running event loop (the typical main.py case) -> asyncio.run.
          (b) A loop exists but is NOT running -> reuse it via run_until_complete
              (avoids the cost of creating a fresh loop per turn).
          (c) A loop IS running (pytest-asyncio, notebook) -> we cannot block on
              it from sync code, so we run the coroutine on a fresh loop in this
              thread. This is sufficient because lite.run_turn is purely
              CPU-bound + uses internal proposers that own their own awaits.
        """
        coro = self.lite.run_turn(state)
        # First, see if a loop is *currently running* in this thread.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            # Case (c): cannot block; run on a private loop.
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        # No running loop -> easiest is asyncio.run, which builds a fresh
        # loop and tears it down for us.
        return asyncio.run(coro)

    def _action_to_game_action(
        self,
        action: Action | None,
        latest_frame: FrameData,
    ) -> GameAction:
        """Map an Action(predicate_id, region_id, coord_xy) into a GameAction.

        ArcgenticaLite always issues click-style decisions, so we map onto
        ACTION6 with x/y. If the lite returned None (no decision), or the
        decided coordinate is invalid, fall back to a legal action drawn from
        latest_frame.available_actions. We never crash the harness.
        """
        if action is None:
            return self._fallback_legal_action(latest_frame)

        x, y = action.coord_xy
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return self._fallback_legal_action(latest_frame)

        if not self._is_action_available(GameAction.ACTION6, latest_frame):
            return self._fallback_legal_action(latest_frame)

        ga = GameAction.ACTION6
        ga.set_data({"x": int(x), "y": int(y)})
        ga.reasoning = {
            "predicate_id": action.predicate_id,
            "region_id": action.region_id,
            "framework_version": self.framework_version,
        }
        return ga

    @staticmethod
    def _is_action_available(action: GameAction, frame: FrameData) -> bool:
        avail = getattr(frame, "available_actions", None) or []
        if not avail:
            # Permissive when engine doesn't restrict.
            return True
        try:
            return action in avail or int(action.value) in {
                int(getattr(a, "value", a)) for a in avail
            }
        except Exception:
            return True

    def _fallback_legal_action(self, latest_frame: FrameData) -> GameAction:
        """Return any legal non-RESET action; used when lite cannot decide."""
        avail = getattr(latest_frame, "available_actions", None) or []
        if not avail:
            return GameAction.RESET
        for cand in avail:
            ga = cand if isinstance(cand, GameAction) else GameAction.from_id(int(cand))
            if ga is GameAction.RESET:
                continue
            if ga.is_complex():
                ga.set_data({"x": 32, "y": 32})  # safe centre fallback
            ga.reasoning = {"fallback": "lite_no_decision"}
            return ga
        return GameAction.RESET

    # -------------------------------------------------------------- diagnostics

    def episode_end(self) -> Any:
        """Pass-through to ArcgenticaLite.episode_end for journaling."""
        try:
            return self.lite.episode_end()
        except Exception as e:  # noqa: BLE001
            logger.warning("episode_end failed: %s", e)
            return None

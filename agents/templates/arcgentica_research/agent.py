"""Arcgentica host with optional research guests.

This class keeps Arcgentica as the execution host and limits research changes to
hook dispatch, prompt overlays, and persistence sidecars.

Supports two backends:
  - agentica SDK (when AGENTICA_API_KEY or S_M_BASE_URL set)
  - TRAPI direct (Azure OpenAI Responses API, always available)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from research_extensions import (
    ModuleRegistry,
    ResearchRuntimeContext,
    load_research_config,
)
from research_extensions.grid_utils import current_grid, encode_row

from ...agent import Agent
from ...tracing import trace_agent_session
from ..agentica.prompts import GAME_REFERENCE
from ..agentica.scope.frame import Frame

logger = logging.getLogger(__name__)

# TRAPI config
_TRAPI_SCOPE = "api://trapi/.default"
_TRAPI_API_VERSION = "2025-04-01-preview"
_TRAPI_MODEL_MAP = {
    "gpt-5.3-codex": "gpt-5.3-codex_2026-02-24",
    "gpt-5.4-mini": "gpt-5.4-mini_2026-03-17",
    "gpt-5.4": "gpt-5.4_2026-03-05",
}

_trapi_client = None


def _resolve_shared_namespace(timestamp: int | None = None) -> str:
    explicit = os.environ.get("ARC_RESEARCH_SHARED_NAMESPACE")
    if explicit:
        return explicit
    if timestamp is None:
        timestamp = int(time.time())
    return f"ephemeral_{timestamp}_{os.getpid()}"

def _get_trapi_client():
    global _trapi_client
    if _trapi_client is not None:
        return _trapi_client
    from openai import AzureOpenAI
    from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(), ManagedIdentityCredential(),
    ), _TRAPI_SCOPE)
    _trapi_client = AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        azure_ad_token_provider=credential,
        api_version=_TRAPI_API_VERSION,
    )
    return _trapi_client

def _call_trapi(messages: list[dict], model: str = "gpt-5.3-codex") -> str:
    deployment = _TRAPI_MODEL_MAP.get(model, model)
    client = _get_trapi_client()
    input_items = [{"type": "message", "role": m["role"], "content": m["content"]} for m in messages]
    resp = client.responses.create(model=deployment, input=input_items, max_output_tokens=4096)
    for item in resp.output:
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    return block.text
    return ""


def _create_trapi_response(
    *,
    model: str = "gpt-5.3-codex",
    input_items: Any,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_output_tokens: int = 4096,
):
    deployment = _TRAPI_MODEL_MAP.get(model, model)
    client = _get_trapi_client()
    kwargs: dict[str, Any] = {
        "model": deployment,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["parallel_tool_calls"] = False
        kwargs["max_tool_calls"] = 32
    return client.responses.create(**kwargs)


class ArcgenticaResearch(Agent):
    """Arcgentica-style agent with research hooks, powered by TRAPI.

    Keeps a strict per-step host: each model turn sees only the current visible
    grid, state, available actions, and optional research overlays.
    """

    MAX_ACTIONS: int = int(os.environ.get("ARC_SMOKE_MAX_ACTIONS", "200"))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.research_config = load_research_config()
        self._registry = None
        super().__init__(*args, **kwargs)
        timestamp = int(time.time())
        self.shared_namespace = _resolve_shared_namespace(timestamp)
        log_root = Path(self.research_config.log_dir)
        self.research_workdir = log_root / f"{self.game_id}_{timestamp}"
        self.research_workdir.mkdir(parents=True, exist_ok=True)
        shared_dir = log_root / "shared" / self.game_id
        shared_dir.mkdir(parents=True, exist_ok=True)
        context = ResearchRuntimeContext(
            game_id=self.game_id,
            workdir=self.research_workdir,
            shared_dir=shared_dir / self.shared_namespace,
        )
        context.shared_dir.mkdir(parents=True, exist_ok=True)
        self._registry = ModuleRegistry(self.research_config, context)
        self._registry.load()
        self.messages: list[dict] = []
        self.action_history: list[tuple[str, Frame]] = []
        self.best_level = 0
        self.reset_count = 0
        self._memories = _LocalMemories(context.shared_dir / "memories.json")
        if self._registry:
            self._registry.on_memories_ready(self._memories)

    @property
    def name(self) -> str:
        active = self.research_config.active_modules() if self.research_config else []
        suffix = f".{'+'.join(active)}" if active else ".baseline"
        return f"{self.game_id}.arcgentica_research{suffix}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Defensive: on level-transition some frames come back with an
        # empty `.frame` list, which makes Frame() raise IndexError. Keep
        # the loop alive by falling back to the raw FrameData; downstream
        # tool handlers that need the rich Frame guard individually.
        try:
            rich_frame = self._to_rich_frame(latest_frame)
        except Exception:
            logger.exception("_to_rich_frame failed in choose_action; using raw FrameData")
            rich_frame = latest_frame
        available_names = [GameAction.from_id(a).name for a in latest_frame.available_actions]
        overlay = self._prompt_overlay()

        # Symbolica-style header (game + level + budget + action legend).
        # The single-line action legend is redundant with GAME_REFERENCE in
        # the system prompt, but reminding the agent every turn stops
        # direction-semantic drift (a common LLM failure mode on ls20).
        legend_parts = []
        for a in available_names:
            sem = {
                "ACTION1": "Up", "ACTION2": "Down",
                "ACTION3": "Left", "ACTION4": "Right",
                "ACTION5": "Space/Enter", "ACTION6": "Click(x,y)",
                "ACTION7": "Aux", "RESET": "restart level",
            }.get(a, "")
            legend_parts.append(f"{a}({sem})" if sem else a)
        action_legend = ", ".join(legend_parts)

        # Last-action effect summary (Symbolica's change_summary equivalent).
        last_action_line = ""
        if self.action_history:
            prev_name, prev_frame_rich = self.action_history[-1]
            try:
                prev_grid = current_grid(prev_frame_rich)
                cur_grid = current_grid(latest_frame)
                diff_cells = 0
                for r1, r2 in zip(prev_grid, cur_grid):
                    for a, b in zip(r1, r2):
                        if a != b:
                            diff_cells += 1
                lvl_change = (
                    getattr(prev_frame_rich, "levels_completed", latest_frame.levels_completed)
                    != latest_frame.levels_completed
                )
                last_action_line = (
                    f"Last action: {prev_name} → grid changed {diff_cells} cells"
                    f"{'; LEVEL UP!' if lvl_change else ''}\n"
                )
            except Exception:
                last_action_line = f"Last action: {prev_name}\n"

        user_msg = (
            f"Game: {self.game_id}  |  "
            f"Level {latest_frame.levels_completed}/{latest_frame.win_levels}  |  "
            f"Non-reset actions used: {self.action_counter}/{self.MAX_ACTIONS}\n"
            f"State: {latest_frame.state.name}  |  Available: {action_legend}\n"
            f"{last_action_line}"
            f"\n{self._frame_to_text(latest_frame)}\n\n"
            f"{overlay}\n\n"
            "Use the available helper tools as needed (frame_render, frame_diff, "
            "frame_change_summary, frame_find, frame_bounding_box, frame_color_counts, "
            "history, memories_*). End the turn by calling `submit_action` exactly "
            "once with your chosen action and optional side channels "
            "(`predict`, `world_update`, `propose_skill`, `env_note`)."
        )

        response = _create_trapi_response(
            model="gpt-5.3-codex",
            instructions=self._default_system_prompt(),
            input_items=[{"type": "message", "role": "user", "content": user_msg}],
            tools=self._tool_definitions(),
            max_output_tokens=4096,
        )

        finish: dict[str, Any] | None = None
        while True:
            tool_outputs: list[dict[str, Any]] = []
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not function_calls:
                text = getattr(response, "output_text", "") or ""
                if text:
                    action_name, side = self._parse_action(text, latest_frame.available_actions)
                    finish = {"action": action_name, **side}
                break

            finish_calls = [call for call in function_calls if call.name == "submit_action"]
            if finish_calls:
                finish = json.loads(finish_calls[0].arguments)
                break

            for call in function_calls:
                output = self._execute_tool(call.name, call.arguments, rich_frame)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": output,
                    }
                )

            response = _create_trapi_response(
                model="gpt-5.3-codex",
                previous_response_id=response.id,
                input_items=tool_outputs,
                tools=self._tool_definitions(),
                max_output_tokens=4096,
            )

        if finish is None:
            finish = {"action": available_names[0]}

        action_name = str(finish.get("action_name", finish.get("action", available_names[0]))).upper()
        side = {k: v for k, v in finish.items() if k not in {"action", "action_name"}}

        if self._registry:
            if side.get("predict") is not None:
                self._registry.record_agent_prediction(action_name, side["predict"])
            if side.get("env_note"):
                self._registry.record_agent_env_note(str(side["env_note"]))
            if isinstance(side.get("propose_skill"), dict):
                self._registry.record_agent_skill_proposal(side["propose_skill"])
            if side.get("world_update") is not None:
                self._registry.record_agent_world_update(side["world_update"])

        return GameAction.from_name(action_name)

    def _parse_action(
        self, response: str, available: list[int]
    ) -> tuple[str, dict[str, Any]]:
        """Return (chosen action name, dict of side channel values).

        Side channel may contain: predict, env_note, propose_skill, world_update.
        """
        available_names = {GameAction.from_id(a).name for a in available}
        if not available_names:
            available_names = {
                name
                for name in [
                    "RESET",
                    "ACTION1",
                    "ACTION2",
                    "ACTION3",
                    "ACTION4",
                    "ACTION5",
                    "ACTION6",
                    "ACTION7",
                ]
            }

        side: dict[str, Any] = {}
        action_name = "ACTION1"

        text = response
        if "{" in text:
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                raw_action = str(data.get("action_name", data.get("action", ""))).upper()
                if raw_action in available_names:
                    action_name = raw_action
                # Side channels (all optional)
                for key in ("predict", "env_note", "propose_skill", "world_update"):
                    if key in data:
                        side[key] = data[key]
            except Exception:
                pass

        if action_name not in available_names:
            # Fallback: any named action appearing in the response
            for candidate in [
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
                "ACTION7",
                "RESET",
            ]:
                if candidate in response.upper() and candidate in available_names:
                    action_name = candidate
                    break
            else:
                action_name = next(iter(sorted(available_names)))

        return action_name, side

    def _frame_to_text(self, frame: FrameData) -> str:
        """Symbolica-unified default render of the frame shown to the LLM.

        Uses the upstream `Frame.render()` with gap=' ' and y/x ticks so
        the grid reads as a picture (spatial pattern visible) rather than
        as a dense hex string. Matches exactly what Symbolica's
        arcgentica agent shows its LLM.
        """
        lines = [f"state: {frame.state.name}"]
        try:
            rich = self._to_rich_frame(frame)
            lines.append(
                f"grid ({rich.width}×{rich.height}):  "
                "(values 0-15 → COLOR_LEGEND from system prompt)"
            )
            # y_ticks + x_ticks = True so coordinates are explicit; gap=' '
            # so neighbouring cells are visually separated.
            lines.append(rich.render(y_ticks=True, x_ticks=True))
        except Exception:
            # Defensive: on level-transition edges Frame() can fail;
            # fall back to raw grid encoding so the turn isn't lost.
            logger.exception("Frame.render failed in _frame_to_text; using raw")
            grid = current_grid(frame)
            if grid:
                lines.append(f"grid ({len(grid)}×{len(grid[0]) if grid else 0}):")
                for i, row in enumerate(grid):
                    lines.append(f"y={i:>2d} | " + " ".join(
                        "0123456789abcdef"[c % 16] for c in row
                    ))
        return "\n".join(lines)

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            GAME_REFERENCE
            + "\n\nAdapter notes for this research host:\n"
            "- This is a direct player host, not the upstream top-level orchestrator. "
            "You do not have `spawn_agent`, but you do have the same core observation helpers.\n"
            "- Use tools instead of REPL syntax. `history` mirrors `history()`. "
            "`memories_summaries`, `memories_get`, `memories_add`, and `memories_query` mirror the shared `memories` object. "
            "`frame_render`, `frame_diff`, `frame_render_diff`, `frame_change_summary`, `frame_find`, "
            "`frame_bounding_box`, and `frame_color_counts` mirror Frame helpers.\n"
            "- Frame tools accept `source='current'` or `source='winning'`. "
            "Use `source='winning'` when the most recent action completed a level and you want to inspect `winning_frame`.\n"
            "- `submit_action` is this host's action commit tool. Call it exactly once per turn to choose the real environment action.\n"
            "- Ignore upstream REPL-only details such as `NOOP` or `submit_action.remaining`. "
            "Use helper tools to inspect without acting; the non-reset action budget is shown in the user turn message.\n"
            "- Helper tools only re-render or summarize already observed frames. There is no snapshot/restore, rollback, or hidden state API.\n"
            "- Any extra text blocks in the prompt are agent-side research overlays, not hidden environment API outputs.\n"
            "- You are free to invent your own vocabulary for skills, triggers, and environment notes. No names are forced.\n"
            "- When you predict what will happen next, say so in the `predict` field. If you want the prediction to be scored, prefix it with "
            "`EXPECT_CHANGE:` or `EXPECT_NO_CHANGE:`. You may also use a structured object like "
            "`{\"expect_change\": true, \"focus\": \"...\", \"note\": \"...\"}`.\n"
            "- If you are building a local rule model, revise it in `world_update` using markdown, code, or pseudocode.\n"
            "- When you notice a reusable pattern, you may propose a skill via `propose_skill`. You choose the shape of that object and the terms you use.\n"
            "- Form hypotheses by comparing states: reproduce an effect from a different starting state before trusting it; never trust a single-observation theory.\n"
            "- Colors are semantic, not positional: think 'A must reach B' rather than 'A must reach row 38'. If your hypothesis cites a specific coordinate as the goal, restate it as a relationship.\n"
            "- The grid is a picture, not a spreadsheet: `color_counts()` and `bounding_box()` are lossy summaries; regularly `frame_render` the grid (or a crop) before concluding anything.\n"
            "- After each action, a last-action effect line shows how many cells changed. Unexpected deltas are more important than raw diff magnitude — inspect any region that changed outside where you acted.\n"
            "- Do not retry the same action-sequence 3+ times without re-examining the grid. If two variations of an approach fail, report what you tried and switch tactics.\n"
            "- Efficient play: do not take exploratory actions you have already taken; prefer targeted experiments over exhaustive sweeps.\n"
        )

    def _prompt_overlay(self) -> str:
        overlay_parts = []
        if self._registry:
            # Exploration-phase notice: tell the LLM that the planner is
            # OFF and real-env signature novelty is being rewarded directly.
            exp_reason = self._registry.bridge.read_hint("exploration_reason", "")
            if exp_reason:
                overlay_parts.append(
                    "[Exploration Phase]\n"
                    f"- {exp_reason}. World coder is not yet reliable — prefer actions that\n"
                    "  reach an observation family you have not seen. Explicit bonus: MCTS\n"
                    "  (when enabled later) will reward simulator paths whose leaf\n"
                    "  `next_signature_hint` suggests an unseen family. While in\n"
                    "  exploration phase, the registry auto-picks the least-tried action\n"
                    "  for this state; you may still emit `predict`, `world_update`, and\n"
                    "  `propose_skill` to accelerate the world coder."
                )
            # Wake overlay (if surprise queued) comes first — it is the
            # most time-sensitive instruction to the LLM.
            if hasattr(self._registry, "wake_overlay"):
                wake = self._registry.wake_overlay()
                if wake:
                    overlay_parts.append(wake)
            registry_overlay = self._registry.prompt_overlay()
            if registry_overlay:
                overlay_parts.append(registry_overlay)
        summaries = self._memories.summaries()
        if summaries:
            # Category-balanced memory selection (fixes v3-full where
            # `summaries[-6:]` showed only the most recent surprise/score
            # events and missed consolidated observed laws). Priority:
            # Observed Law first (aggregate truth > individual anecdotes),
            # then recent Draft evolution, then a few Surprise/Score/Skill.
            def _filter(prefix: str, n: int) -> list[str]:
                # summaries() prepends "[<idx>] " so we substring-match.
                matches = [s for s in summaries if prefix in s]
                return matches[-n:]
            balanced: list[str] = []
            balanced.extend(_filter("[Observed Law]", 4))
            balanced.extend(_filter("[World Draft]", 2))
            balanced.extend(_filter("[World Surprise]", 2))
            balanced.extend(_filter("[World Score]", 1))
            balanced.extend(_filter("[Skill]", 2))
            # Deduplicate preserving order.
            seen: set[str] = set()
            picked: list[str] = []
            for s in balanced:
                if s in seen:
                    continue
                seen.add(s)
                picked.append(s)
            # Fallback: if balanced selection is empty (early game), use
            # the last 6 as before.
            if not picked:
                picked = summaries[-6:]
            memory_lines = ["[Persistent Notes]"]
            memory_lines.extend(f"- {line}" for line in picked)
            overlay_parts.append("\n".join(memory_lines))

            # v3.12: inject the FULL details of observed-law memories
            # so predict_effect can actually cite the per-action mean_diff
            # instead of guessing `expected_diff_cells=1`.
            law_details: list[str] = []
            for idx_m, mem in enumerate(self._memories.stack):
                if "[Observed Law]" in mem.summary:
                    law_details.append(f"- {mem.summary}: {mem.details[:240]}")
            if law_details:
                overlay_parts.append(
                    "[Observed-Law Details — use these numbers in predict_effect]\n"
                    + "\n".join(law_details[-6:])
                    + "\n→ If mean_diff for ACTIONk is ~42, your `expected_diff_cells`"
                    " must be ~42 (not 1) AND your `expected_next_grid` must differ"
                    " from the input in ~42 cells."
                )
        return "\n\n".join(overlay_parts)

    def _to_rich_frame(self, frame: FrameData) -> Frame:
        prev_levels_completed = None
        if self.action_history:
            prev_levels_completed = self.action_history[-1][1].levels_completed
        elif len(self.frames) >= 2:
            prev_levels_completed = self.frames[-2].levels_completed
        # Upstream Frame() indexes `data.frame[-1]`; if the engine returns
        # an empty grid stack (observed around level transitions), fall
        # back to the most recent non-empty frame in history instead of
        # crashing.
        if not getattr(frame, "frame", None):
            for cached in reversed(self.frames[:-1]):
                if getattr(cached, "frame", None):
                    frame = cached
                    break
            else:
                raise ValueError("no non-empty frame available for Frame()")
        return Frame(frame, prev_levels_completed=prev_levels_completed)

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "history",
                "description": "Return the last n observed (action, frame) entries, oldest first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1, "maximum": 50},
                        "wins_only": {"type": "boolean"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_summaries",
                "description": "List short summaries of stored shared memories.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_get",
                "description": "Retrieve a stored shared memory by index.",
                "parameters": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_add",
                "description": "Store a shared memory summary and details for later turns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["summary", "details"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "memories_query",
                "description": "Natural-language retrieval over shared memories. Mirrors memories.query(...).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "return_type": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_render",
                "description": (
                    "Render a frame as text (Symbolica Frame.render signature). "
                    "Look at the grid as a picture; this is the single most reliable "
                    "way to see shapes, walls, player, goals. Use `crop=(x1,y1,x2,y2)` "
                    "to zoom in on a region. Use `y_ticks=True, x_ticks=True` for axes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "keys": {"type": "string", "description": "16-char map int→glyph (default '0123456789abcdef')"},
                        "gap": {"type": "string", "description": "separator between cells (default ' ')"},
                        "crop": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "(x1, y1, x2, y2) exclusive end",
                        },
                        "y_ticks": {"type": "boolean", "description": "prefix each row with y=NN"},
                        "x_ticks": {"type": "boolean", "description": "header with x column numbers"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_diff",
                "description": (
                    "Diff a chosen frame against the previous or a chosen history frame. "
                    "Returns list[DiffRegion]; use each region's (x0,y0,x1,y1) to crop "
                    "frame_render_diff for zoom-in. `margin` merges changes within N "
                    "pixels into a single region (default 2, Symbolica default)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "margin": {"type": "integer", "minimum": 0, "description": "cluster radius (default 2)"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_change_summary",
                "description": "One-line region summaries of changes between a chosen frame and a previous frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "margin": {"type": "integer", "minimum": 0, "description": "cluster radius (default 2)"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_render_diff",
                "description": (
                    "Visual diff between a chosen frame and a previous or chosen history frame. "
                    "Changed cells show new value; unchanged cells show '.'. Use "
                    "`crop='auto'` to zoom on tight bbox of changes, `crop=[x1,y1,x2,y2]` "
                    "for explicit region, `crop=None` (omit) for full grid."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                        "keys": {"type": "string"},
                        "gap": {"type": "string"},
                        "crop": {
                            "oneOf": [
                                {"type": "string", "enum": ["auto"]},
                                {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                            ]
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_find",
                "description": "Find matching cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                        "history_index": {"type": "integer"},
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_bounding_box",
                "description": "Bounding box of matching cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "values": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                        "history_index": {"type": "integer"},
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "frame_color_counts",
                "description": "Count cell values in the current frame, winning_frame, or a history frame.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["current", "winning"]},
                        "history_index": {"type": "integer"},
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "submit_action",
                "description": "Commit this turn's real game action and optional side-channel outputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_name": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "predict": {},
                        "env_note": {"type": "string"},
                        "propose_skill": {"type": "object"},
                        "world_update": {},
                    },
                    "required": ["action_name"],
                    "additionalProperties": False,
                },
            },
        ]

    def _execute_tool(self, name: str, raw_arguments: str, current_frame: Frame) -> str:
        try:
            args = json.loads(raw_arguments or "{}")
        except Exception:
            args = {}

        if name == "history":
            n = int(args.get("n", 10))
            wins_only = bool(args.get("wins_only", False))
            entries = self.action_history
            if wins_only:
                entries = [(action, frame) for action, frame in entries if frame.winning_frame is not None]
            out = []
            for action, frame in entries[-n:]:
                out.append(
                    {
                        "action": action,
                        "has_winning_frame": frame.winning_frame is not None,
                        "state": frame.state.name,
                        "levels_completed": frame.levels_completed,
                        "available_actions": frame.available_actions,
                    }
                )
            return json.dumps(out)

        if name == "memories_summaries":
            return json.dumps(self._memories.summaries())
        if name == "memories_get":
            idx = int(args["index"])
            memory = self._memories.get(idx)
            return json.dumps(
                {
                    "summary": memory.summary,
                    "details": memory.details,
                    "timestamp": memory.timestamp.isoformat(),
                }
            )
        if name == "memories_add":
            self._memories.add(str(args["summary"]), str(args["details"]))
            return json.dumps({"ok": True, "count": len(self._memories.stack)})
        if name == "memories_query":
            result = self._memories.query(
                str(args.get("question", "")),
                return_type=str(args.get("return_type", "str")),
                limit=int(args.get("limit", 3)),
            )
            return json.dumps(result)

        frame = self._resolve_frame(current_frame, args)
        if name == "frame_render":
            crop = args.get("crop")
            crop_tuple = tuple(int(x) for x in crop) if crop is not None else None
            # Match Symbolica signature: default y_ticks=False, x_ticks=False,
            # gap=" ", keys="0123456789abcdef". The agent can override.
            return frame.render(
                keys=str(args.get("keys", "0123456789abcdef")),
                gap=str(args.get("gap", " ")),
                y_ticks=bool(args.get("y_ticks", False)),
                x_ticks=bool(args.get("x_ticks", False)),
                crop=crop_tuple,
            )
        if name == "frame_diff":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            margin = int(args.get("margin", 2))
            return repr(frame.diff(ref, margin=margin))
        if name == "frame_change_summary":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            margin = int(args.get("margin", 2))
            return frame.change_summary(ref, margin=margin)
        if name == "frame_render_diff":
            ref = self._resolve_diff_reference(frame, args.get("history_index"))
            crop = args.get("crop")
            if isinstance(crop, list):
                crop = tuple(int(x) for x in crop)
            return frame.render_diff(
                ref,
                keys=str(args.get("keys", "0123456789abcdef")),
                gap=str(args.get("gap", " ")),
                crop=crop,
            )
        if name == "frame_find":
            values = [int(v) for v in args.get("values", [])]
            return json.dumps(frame.find(*values))
        if name == "frame_bounding_box":
            values = [int(v) for v in args.get("values", [])]
            return json.dumps(frame.bounding_box(*values))
        if name == "frame_color_counts":
            return json.dumps(frame.color_counts())

        return json.dumps({"error": f"unknown tool {name}"})

    def _resolve_frame(self, current_frame: Frame, args: dict[str, Any]) -> Frame:
        source = str(args.get("source", "current"))
        history_index = args.get("history_index")
        if history_index is not None:
            base = self._resolve_history_frame(current_frame, history_index)
            if source == "winning" and base.winning_frame is not None:
                return base.winning_frame
            return base
        if source == "winning" and current_frame.winning_frame is not None:
            return current_frame.winning_frame
        return current_frame

    def _resolve_history_frame(self, current_frame: Frame, history_index: Any) -> Frame:
        if history_index is None:
            return current_frame
        idx = int(history_index)
        if not self.action_history:
            return current_frame
        return self.action_history[idx][1]

    def _resolve_diff_reference(self, frame: Frame, history_index: Any) -> Frame:
        if history_index is not None:
            return self._resolve_history_frame(frame, history_index)
        if len(self.action_history) >= 2:
            return self.action_history[-2][1]
        if self.action_history:
            return self.action_history[-1][1]
        return frame

    def _ensure_bootstrap_frame(self) -> None:
        latest = self.frames[-1]
        if latest.state is not GameState.NOT_PLAYED:
            return

        initial = self.take_action(GameAction.RESET)
        if initial:
            self.append_frame(initial)

    @trace_agent_session
    def main(self) -> None:
        self.timer = time.time()
        self._ensure_bootstrap_frame()
        while not self.is_done(self.frames, self.frames[-1]) and self.action_counter < self.MAX_ACTIONS:
            latest = self.frames[-1]
            available_names = [GameAction.from_id(a).name for a in latest.available_actions]

            # Plan v3 §4 P0: imagination + option-execution bypass.
            bypass_action: str | None = None
            if self._registry and hasattr(self._registry, "imagine_and_maybe_commit"):
                try:
                    bypass_action = self._registry.imagine_and_maybe_commit(latest, available_names)
                except Exception:
                    logger.exception("imagination phase failed; falling back to LLM")
                    bypass_action = None

            if bypass_action:
                # Synthesise a prediction so after_action can score the
                # transition against the simulator.
                try:
                    self._registry.synthesize_prediction(latest, bypass_action)
                except Exception:
                    logger.exception("synthesize_prediction failed; continuing")
                action = GameAction.from_name(bypass_action)
                action_name = action.name
                # Log bypass so the next LLM turn can read it in overlay.
                try:
                    committed = self._registry.bridge.current_skill_in_execution
                    self._registry.record_bypass_step(
                        skill=committed.skill_name if committed else "?",
                        action=bypass_action,
                        outcome="pending",  # after_action will not update; agent sees 'pending' = ran successfully
                        turn=self.action_counter,
                    )
                except Exception:
                    pass
                logger.info("[v3 bypass] skill-committed action %s (LLM skipped)", bypass_action)
            else:
                action = self.choose_action(self.frames, latest)
                action_name = action.name
                if self._registry:
                    override = self._registry.before_action(latest, action_name, available_names)
                    if override:
                        action = GameAction.from_name(override)

            frame = self.take_action(action)
            if frame:
                self.append_frame(frame)
                # Defensive: _to_rich_frame can crash if the engine returns
                # a frame with an empty `.frame` list (level-transition
                # edge case). Skip rich-frame wrapping for that step so
                # the main loop can continue and on_run_end still persists
                # state.
                try:
                    rich_frame = self._to_rich_frame(frame)
                    self.action_history.append((action.name, rich_frame))
                except Exception:
                    logger.exception(
                        "_to_rich_frame failed (likely empty frame.frame); "
                        "recording raw FrameData so the main loop can continue"
                    )
                    self.action_history.append((action.name, frame))
                self.best_level = max(self.best_level, frame.levels_completed)
                if action.name == "RESET":
                    self.reset_count += 1

                # Research hook: after_action
                if self._registry:
                    try:
                        self._registry.after_action(latest, action.name, frame)
                    except Exception:
                        logger.exception("registry.after_action failed; continuing")

                logger.info(f"{self.game_id} - {action.name}: L={frame.levels_completed}, count={self.action_counter}")

            if action.name != "RESET":
                self.action_counter += 1

        # Research hook: on_run_end (always run, even after exceptions).
        if self._registry:
            try:
                self._registry.on_run_end(history=self.action_history, finish_status=None)
            except Exception:
                logger.exception("registry.on_run_end failed; shared dir may be incomplete")
        if hasattr(self, "_memories") and self._memories is not None:
            try:
                self._memories.save()
            except Exception:
                logger.exception("memories.save failed")

        # Save research state
        unique_actions = sorted({name for name, _ in self.action_history})
        (self.research_workdir / "summary.json").write_text(json.dumps({
            "best_level": self.best_level,
            "actions": self.action_counter,
            "reset_count": self.reset_count,
            "unique_actions": unique_actions,
            "active_modules": self.research_config.active_modules() if self.research_config else [],
            "shared_namespace": self.shared_namespace,
        }, indent=2))
        logger.info(f"Done: L={self.best_level}, actions={self.action_counter}")
        self.cleanup()


class _LocalMemories:
    def __init__(self, state_path: Path) -> None:
        self._state_path = state_path
        self.stack: list[_Memory] = []
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                for item in raw:
                    self.stack.append(
                        _Memory(
                            summary=str(item.get("summary", "")),
                            details=str(item.get("details", "")),
                            timestamp=datetime.fromisoformat(str(item.get("timestamp"))),
                        )
                    )
            except Exception:
                self.stack = []

    def add(self, title: str, text: str) -> None:
        self.stack.append(_Memory(summary=title, details=text))

    def summaries(self) -> list[str]:
        return [f"[{i}] {m.summary}" for i, m in enumerate(self.stack)]

    def get(self, i: int) -> "_Memory":
        return self.stack[i]

    def query(self, question: str, *, return_type: str = "str", limit: int = 3) -> dict[str, Any]:
        tokens = {token.lower() for token in question.replace("_", " ").split() if token.strip()}
        ranked: list[tuple[int, int, _Memory]] = []
        for idx, memory in enumerate(self.stack):
            haystack = f"{memory.summary}\n{memory.details}".lower()
            score = sum(1 for token in tokens if token in haystack)
            if score > 0 or not tokens:
                ranked.append((score, idx, memory))
        ranked.sort(key=lambda item: (-item[0], -item[1]))
        matches = [
            {
                "index": idx,
                "summary": memory.summary,
                "details": memory.details,
                "timestamp": memory.timestamp.isoformat(),
            }
            for score, idx, memory in ranked[:limit]
        ]
        answer = "\n\n".join(
            f"[{item['index']}] {item['summary']}\n{item['details']}" for item in matches
        )
        return {
            "question": question,
            "return_type": return_type,
            "matches": matches,
            "answer": answer,
        }

    def save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(
                [
                    {
                        "summary": memory.summary,
                        "details": memory.details,
                        "timestamp": memory.timestamp.isoformat(),
                    }
                    for memory in self.stack
                ],
                indent=2,
            ),
            encoding="utf-8",
        )


class _Memory:
    def __init__(self, summary: str, details: str, timestamp: datetime | None = None) -> None:
        self.summary = summary
        self.details = details
        self.timestamp = timestamp or datetime.now()

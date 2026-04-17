from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class RuntimeSnapshot:
    action_name: str
    level_before: int
    level_after: int
    state_before: str
    state_after: str
    available_actions: list[str] = field(default_factory=list)


class ResearchModule(Protocol):
    name: str

    def prompt_overlay(self) -> str: ...

    def on_memories_ready(self, memories: Any) -> None: ...

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None: ...

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None: ...

    def on_run_end(
        self,
        *,
        history: list[tuple[str, Any]],
        finish_status: Any,
        workdir: Path,
    ) -> None: ...

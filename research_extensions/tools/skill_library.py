"""SkillLibrary — shared, persistable skill store (S6, plan v651 revB).

A Memories-shape clone specialized for ``Skill`` records.  Skills are
high-confidence behavioural recipes promoted from hypotheses (or directly
from level-clear traces) by the consolidator.  This module owns:

* The immutable :class:`Skill` dataclass (frozen + slots, codex r1 field order).
* The :class:`SkillLibrary` container with synchronous CRUD and an async
  natural-language ``query`` interface backed by an Agentica subagent.
* Posterior accounting and auto-quarantine on contamination (``falsify_n >
  confirm_n``) per the codex contamination guardrail.
* Persistence delegated to :mod:`skill_persistence`; sandbox enforcement
  delegated to :mod:`skill_sandbox`.

Design rules (see ``reports/plan_v651_skill_library_revB.md``):

* No edits to frozen Agentica templates.  ``spawn`` is imported **lazily**
  on first ``_ensure_agent`` call so module import is safe even before the
  Agentica server is running.
* The agent task is created lazily and bound to the event loop it was
  spawned in; reuse across event loops raises ``RuntimeError`` (codex r3).
* ``query`` falls back to ``_empty_result`` on timeout/agent failure so the
  caller never has to handle library outages in research code.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from . import skill_persistence
from .skill_sandbox import SkillSandboxError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentica import Agent

__all__ = ["Skill", "SkillLibrary", "SkillLoopError", "SkillQueryError"]


class SkillLoopError(RuntimeError):
    """Raised when a SkillLibrary instance is used from a different event loop
    than the one it was first bound to (codex r3 cross-loop guardrail).

    This is a hard failure — callers must propagate it. Tests may catch it to
    assert the guard fires."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill dataclass — required fields first, defaulted fields last (codex r1).
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Skill:
    """A confirmed, executable behavioural recipe.

    Fields:
        skill_id: stable UUID4 string assigned at creation.
        summary: one-line natural-language description (<=200 chars).
        recipe: step-by-step instructions for re-execution.
        evidence: memory-id references that justify this skill.
        posterior: ``(confirm_n, falsify_n)`` Beta-style counts.
        applicability_conditions: short natural-language preconditions.
        parent_hypothesis_ids: hypotheses this skill was promoted from.
        category: ``"mechanic"`` (env rule) or ``"strategy"`` (plan).
        predicate: optional sandboxed DSL string evaluated against frame
            features; ``None`` disables predicate gating.
        code: optional helper-code string executed OUT-OF-BAND by the
            orchestrator (never inside the DSL evaluator).
        schema_version: persistence schema version.
        predicate_dsl_version: predicate DSL grammar version.
        quarantined: True iff contamination guardrail tripped.
        quarantine_reason: human-readable reason for quarantine.
        timestamp: creation/replacement wall-clock time.
    """

    skill_id: str
    summary: str
    recipe: str
    evidence: list[str]
    posterior: tuple[int, int]
    applicability_conditions: list[str]
    parent_hypothesis_ids: list[str]
    category: Literal["mechanic", "strategy"]
    predicate: str | None = None
    code: str | None = None
    schema_version: str = "1"
    predicate_dsl_version: str = "1"
    quarantined: bool = False
    quarantine_reason: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class SkillQueryError(Exception):
    """Raised when a skill query cannot be answered in the requested format."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# SkillLibrary — Memories-shape clone.
# ---------------------------------------------------------------------------


_AGENT_PREMISE = (
    "You retrieve information from a shared `skill_library` object containing "
    "high-confidence behavioural skills promoted from hypotheses. You can call "
    "any of its methods: `skill_library.stack`, `skill_library.get(i_or_id)`, "
    "`skill_library.summaries()`. Skills with `quarantined=True` failed a "
    "contamination check; mention this when surfacing them. Do not invent "
    "skills -- only return what the stack supports. If the caller asks for "
    "`str`, return a concise plain-text answer rather than raising when the "
    "library is weak or empty. Only raise SkillQueryError when the requested "
    "return format itself is not appropriate for the query. NEVER call "
    "`skill_library.add` from inside a query -- writes happen only via the "
    "consolidator pipeline."
)


class SkillLibrary:
    """Shared, persistable skill store with async natural-language query.

    Construction is synchronous and safe to do at agent-boot time before any
    event loop exists.  The agent task is created lazily on first ``query``
    so the library survives event-loop teardown between rollouts.
    """

    stack: list[Skill]

    def __init__(
        self,
        model: str,
        *,
        persistence_path: str | None = None,
        query_timeout_s: float = 30.0,
    ) -> None:
        self._model = model
        self._persistence_path = persistence_path
        self._query_timeout_s = query_timeout_s
        self._lock = asyncio.Lock()
        self._skill_agent: asyncio.Task[Agent] | None = None
        self._loop_id: int | None = None
        self._last_seen = 0
        self._load_meta: dict[str, Any] = {"loaded_from": "fresh"}

        if persistence_path is not None:
            loaded, meta = skill_persistence.load_with_fallback(persistence_path)
            self.stack = loaded
            self._load_meta = meta
            self._last_seen = len(loaded)
        else:
            self.stack = []

    # ----- agent lifecycle -------------------------------------------------

    def _ensure_agent(self) -> None:
        """Idempotently start (and loop-pin) the backing query agent."""
        running_loop = asyncio.get_running_loop()
        if self._skill_agent is None:
            # Lazy import: do NOT touch agents.templates.agentica at module load.
            from agents.templates.agentica.compat import spawn  # type: ignore
            from agentica.logging.agent_listener import AgentListener  # type: ignore
            from agentica.logging.loggers.file_logger import FileLogger  # type: ignore

            self._loop_id = id(running_loop)
            self._skill_agent = asyncio.ensure_future(
                spawn(
                    model=self._model,
                    listener=lambda: AgentListener(
                        FileLogger("logs/", "skill-library-agent-")
                    ),
                    premise=_AGENT_PREMISE,
                    scope={
                        "skill_library": self,
                        "SkillQueryError": SkillQueryError,
                        "SkillSandboxError": SkillSandboxError,
                    },
                )
            )
            return

        if self._loop_id is not None and id(running_loop) != self._loop_id:
            raise SkillLoopError(
                "SkillLibrary instance used across event loops "
                f"(bound to loop {self._loop_id}, now {id(running_loop)})"
            )

    # ----- sync CRUD -------------------------------------------------------

    def add(
        self,
        *,
        summary: str,
        recipe: str,
        evidence: list[str],
        applicability_conditions: list[str],
        parent_hypothesis_ids: list[str],
        category: Literal["mechanic", "strategy"],
        posterior: tuple[int, int] = (0, 0),
        predicate: str | None = None,
        code: str | None = None,
    ) -> str:
        """Append a new skill; return its assigned ``skill_id``.

        Auto-snapshots when ``persistence_path`` is configured.
        """
        skill = Skill(
            skill_id=str(uuid.uuid4()),
            summary=summary,
            recipe=recipe,
            evidence=list(evidence),
            posterior=tuple(posterior),  # type: ignore[arg-type]
            applicability_conditions=list(applicability_conditions),
            parent_hypothesis_ids=list(parent_hypothesis_ids),
            category=category,
            predicate=predicate,
            code=code,
        )
        self.stack.append(skill)
        self._maybe_snapshot()
        return skill.skill_id

    def summaries(self) -> list[str]:
        """One-line summary per skill, suitable for prompt context."""
        out: list[str] = []
        for i, s in enumerate(self.stack):
            tail = " [QUARANTINED]" if s.quarantined else ""
            out.append(f"[{i}/{s.skill_id[:8]}] {s.summary}{tail}")
        return out

    def get(self, skill_id_or_index: int | str) -> Skill:
        """Resolve by integer index or by full ``skill_id`` string."""
        if isinstance(skill_id_or_index, int):
            return self.stack[skill_id_or_index]
        for s in self.stack:
            if s.skill_id == skill_id_or_index:
                return s
        raise KeyError(f"No skill with id {skill_id_or_index!r}")

    # ----- async posterior updates ----------------------------------------

    async def confirm(self, skill_id: str, evidence_ref: str) -> None:
        """Increment ``confirm_n`` and append a confirmation event."""
        async with self._lock:
            idx, old = self._find(skill_id)
            new_conf = old.posterior[0] + 1
            new_falsif = old.posterior[1]
            new = replace(
                old,
                posterior=(new_conf, new_falsif),
                evidence=[*old.evidence, evidence_ref],
            )
            self.stack[idx] = new
            self._append_event(
                {
                    "event": "confirm",
                    "skill_id": skill_id,
                    "evidence_ref": evidence_ref,
                    "new_posterior": [new_conf, new_falsif],
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self._maybe_snapshot()

    async def falsify(self, skill_id: str, reason: str, evidence_ref: str) -> None:
        """Increment ``falsify_n``; auto-quarantine if falsify_n > confirm_n."""
        async with self._lock:
            idx, old = self._find(skill_id)
            new_conf = old.posterior[0]
            new_falsif = old.posterior[1] + 1
            quarantined = new_falsif > new_conf
            new = replace(
                old,
                posterior=(new_conf, new_falsif),
                evidence=[*old.evidence, evidence_ref],
                quarantined=quarantined or old.quarantined,
                quarantine_reason=(
                    reason if quarantined and not old.quarantined else old.quarantine_reason
                ),
            )
            self.stack[idx] = new
            self._append_event(
                {
                    "event": "falsify",
                    "skill_id": skill_id,
                    "reason": reason,
                    "evidence_ref": evidence_ref,
                    "new_posterior": [new_conf, new_falsif],
                    "quarantined": new.quarantined,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self._maybe_snapshot()

    # ----- async query -----------------------------------------------------

    async def query[T](self, return_type: type[T], query: str) -> T:
        """Natural-language query over the skill stack.

        Mirrors ``Memories.query`` semantics: returns ``_empty_result`` when
        the stack is empty OR when the backing agent times out / errors.
        """
        if not self.stack:
            return self._empty_result(return_type)

        try:
            self._ensure_agent()
        except SkillLoopError:
            # codex r3 locked guardrail: cross-loop misuse must propagate,
            # not silently degrade to _empty_result.
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SkillLibrary._ensure_agent failed: %r", exc)
            return self._empty_result(return_type)

        assert self._skill_agent is not None  # for mypy

        try:
            agent = await asyncio.wait_for(
                asyncio.shield(self._skill_agent),
                timeout=self._query_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("SkillLibrary: agent spawn timed out")
            return self._empty_result(return_type)
        except Exception as exc:
            logger.warning("SkillLibrary: agent spawn failed: %r", exc)
            return self._empty_result(return_type)

        new_count = len(self.stack) - self._last_seen
        self._last_seen = len(self.stack)
        preamble = (
            f"There are {new_count} new skills since your last call.\n\n"
            if new_count > 0
            else ""
        )
        format_hint = (
            "If return_type is `str`, answer in plain text only and do not "
            "return a dict/JSON object."
        )
        try:
            return await asyncio.wait_for(
                agent.call(
                    return_type,
                    f"{preamble}Answer the following query. If return_type is "
                    f"`str`, return a concise plain-text summary and, if "
                    f"evidence is weak, explicitly say so instead of raising. "
                    f"Raise SkillQueryError only if the requested return "
                    f"format is not appropriate for the query.\n\n"
                    f"{format_hint}\n\nQuery: {query}",
                    skill_library=self,
                    stack=self.stack,
                ),
                timeout=self._query_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("SkillLibrary.query timed out: %r", query[:80])
            return self._empty_result(return_type)
        except Exception as exc:
            if isinstance(exc, SkillQueryError) or "SkillQueryError" in repr(exc):
                return self._empty_result(return_type)
            logger.warning("SkillLibrary.query failed: %r", exc)
            return self._empty_result(return_type)

    @staticmethod
    def _empty_result[T](return_type: type[T]) -> T:
        origin = get_origin(return_type)
        if return_type is str:
            return "No stored skills yet."  # type: ignore[return-value]
        if return_type is bool:
            return False  # type: ignore[return-value]
        if return_type is int:
            return 0  # type: ignore[return-value]
        if return_type is float:
            return 0.0  # type: ignore[return-value]
        if return_type is Skill:
            return Skill(
                skill_id="00000000-0000-0000-0000-000000000000",
                summary="No stored skills yet.",
                recipe="",
                evidence=[],
                posterior=(0, 0),
                applicability_conditions=[],
                parent_hypothesis_ids=[],
                category="mechanic",
            )  # type: ignore[return-value]
        if origin is list or return_type is list:
            return []  # type: ignore[return-value]
        if origin is dict or return_type is dict:
            return {}  # type: ignore[return-value]

        args = get_args(return_type)
        if args and len(args) == 2 and type(None) in args:
            return None  # type: ignore[return-value]

        raise SkillQueryError(
            f"No stored skills yet, and no empty fallback is defined for "
            f"return type {return_type!r}."
        )

    # ----- persistence helpers --------------------------------------------

    def snapshot(self, path: str | None = None) -> None:
        """Write a checksum-validated snapshot of the current stack."""
        target = path or self._persistence_path
        if target is None:
            raise ValueError("snapshot() requires a path or configured persistence_path")
        skill_persistence.save(target, self.stack)

    @classmethod
    def load(
        cls,
        path: str,
        model: str,
        **kwargs: Any,
    ) -> "SkillLibrary":
        """Load (or initialize-from-empty) a library bound to ``path``."""
        return cls(model, persistence_path=path, **kwargs)

    async def shutdown(self) -> None:
        """Cancel the background agent and flush a final snapshot."""
        if self._skill_agent is not None and not self._skill_agent.done():
            self._skill_agent.cancel()
            try:
                await asyncio.sleep(0)
            except Exception:  # pragma: no cover - defensive
                pass
        if self._persistence_path is not None:
            try:
                self.snapshot()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("SkillLibrary.shutdown final snapshot failed: %r", exc)

    # ----- internals -------------------------------------------------------

    def _find(self, skill_id: str) -> tuple[int, Skill]:
        for i, s in enumerate(self.stack):
            if s.skill_id == skill_id:
                return i, s
        raise KeyError(f"No skill with id {skill_id!r}")

    def _maybe_snapshot(self) -> None:
        if self._persistence_path is None:
            return
        try:
            skill_persistence.save(self._persistence_path, self.stack)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SkillLibrary auto-snapshot failed: %r", exc)

    def _append_event(self, event: dict[str, Any]) -> None:
        if self._persistence_path is None:
            return
        try:
            skill_persistence.append_event(self._persistence_path, event)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("SkillLibrary event-log append failed: %r", exc)

    def __repr__(self) -> str:
        return f"SkillLibrary(<{len(self.stack)} skills>)"

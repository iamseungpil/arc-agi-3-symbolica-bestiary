"""Skill consolidation — promote hypotheses / level-clears to Skills (S6).

Pure, stateless, synchronous functions.  No agent calls, no I/O, no
randomness, no time-dependent behaviour beyond ``Skill.timestamp`` (which
defaults at construction).  These functions are the ONLY sanctioned path
from low-confidence beliefs to high-confidence ``Skill`` records — keeping
them side-effect-free makes them trivially unit-testable and replayable.

Public API:
    * :func:`consolidate_from_hypothesis` — promote a single hypothesis
      that has cleared the ``confirm_n >= 2 and falsify_n == 0`` gate.
    * :func:`consolidate_from_level_clear` — mine a backward window from a
      level-clear memory for causally-linked action chains and emit one
      mechanic skill per chain.
    * :func:`dedup_near_duplicate` — Jaccard-merge skills whose
      ``summary + " " + recipe`` token sets exceed ``threshold``, keeping
      the skill with the higher posterior margin and unioning evidence /
      parent-hypothesis ids.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from .skill_library import Skill

__all__ = [
    "consolidate_from_hypothesis",
    "consolidate_from_level_clear",
    "dedup_near_duplicate",
]


# ---------------------------------------------------------------------------
# Hypothesis → Skill
# ---------------------------------------------------------------------------


def consolidate_from_hypothesis(
    hypothesis_dict: dict[str, Any],
    memories: Any,  # untyped — Memories shape, only used for evidence look-up callers may want
) -> Skill | None:
    """Promote a hypothesis dict to a :class:`Skill` if the gate is cleared.

    Gate (codex r4): ``confirm_n >= 2 AND falsify_n == 0``.  Any other
    combination returns ``None`` so callers can iterate the pending set
    without branching.

    The hypothesis dict shape is intentionally loose — only ``claim`` and
    ``id`` are required.  All other fields default to safe empties.
    """
    confirm_n = int(hypothesis_dict.get("confirm_n", 0))
    falsify_n = int(hypothesis_dict.get("falsify_n", 0))
    if not (confirm_n >= 2 and falsify_n == 0):
        return None

    claim = str(hypothesis_dict.get("claim", "")).strip()
    if not claim:
        return None
    hyp_id = hypothesis_dict.get("id")
    if hyp_id is None:
        return None

    category = hypothesis_dict.get("category", "mechanic")
    if category not in ("mechanic", "strategy"):
        category = "mechanic"

    return Skill(
        skill_id=str(uuid.uuid4()),
        summary=claim[:200],
        recipe=str(hypothesis_dict.get("recipe", "")),
        evidence=list(hypothesis_dict.get("evidence_memory_ids", [])),
        posterior=(0, 0),
        applicability_conditions=list(hypothesis_dict.get("conditions", [])),
        parent_hypothesis_ids=[str(hyp_id)],
        category=category,
    )


# ---------------------------------------------------------------------------
# Level-clear trace → Skill(s)
# ---------------------------------------------------------------------------


def _is_level_start_marker(memory: Any) -> bool:
    summary = _memory_field(memory, "summary", "")
    details = _memory_field(memory, "details", "")
    blob = f"{summary} {details}".lower()
    return "level_start" in blob or "level start" in blob


def _memory_field(memory: Any, name: str, default: Any = None) -> Any:
    if isinstance(memory, dict):
        return memory.get(name, default)
    return getattr(memory, name, default)


def _memory_id(memory: Any, fallback_index: int) -> str:
    explicit = _memory_field(memory, "memory_id", None) or _memory_field(memory, "id", None)
    if explicit is not None:
        return str(explicit)
    return f"mem:{fallback_index}"


def consolidate_from_level_clear(
    memories: Any,
    level_clear_memory_index: int,
) -> list[Skill]:
    """Mine a backward window for action chains and emit one Skill per chain.

    Walks ``memories.stack`` backwards from ``level_clear_memory_index``
    until it hits a ``level_start`` marker (or the bottom of the stack),
    grouping consecutive memories that carry both ``action`` and
    ``outcome`` fields into chains.  Each chain becomes a single
    mechanic skill.  Returns ``[]`` when no actionable chain is found.
    """
    stack = getattr(memories, "stack", None)
    if not stack:
        return []
    if not (0 <= level_clear_memory_index < len(stack)):
        return []

    window: list[tuple[int, Any]] = []
    for i in range(level_clear_memory_index - 1, -1, -1):
        memory = stack[i]
        if _is_level_start_marker(memory):
            break
        window.append((i, memory))
    window.reverse()  # chronological order

    if not window:
        return []

    # Group consecutive action+outcome memories into chains.
    chains: list[list[tuple[int, Any]]] = []
    current: list[tuple[int, Any]] = []
    for idx, memory in window:
        action = _memory_field(memory, "action", None)
        outcome = _memory_field(memory, "outcome", None)
        if action is not None and outcome is not None:
            current.append((idx, memory))
        elif current:
            chains.append(current)
            current = []
    if current:
        chains.append(current)

    if not chains:
        return []

    skills: list[Skill] = []
    for chain in chains:
        actions = [str(_memory_field(m, "action")) for _, m in chain]
        outcomes = [str(_memory_field(m, "outcome")) for _, m in chain]
        evidence = [_memory_id(m, idx) for idx, m in chain]
        actions_str = " -> ".join(actions)
        recipe_lines = [
            f"{step}. action={a}; outcome={o}"
            for step, (a, o) in enumerate(zip(actions, outcomes), start=1)
        ]
        skill = Skill(
            skill_id=str(uuid.uuid4()),
            summary=f"Skill for clearing level via {actions_str}"[:200],
            recipe="\n".join(recipe_lines),
            evidence=evidence,
            posterior=(0, 0),
            applicability_conditions=[],
            parent_hypothesis_ids=[],
            category="mechanic",
        )
        skills.append(skill)
    return skills


# ---------------------------------------------------------------------------
# Near-duplicate dedup
# ---------------------------------------------------------------------------


def _token_set(skill: Skill) -> frozenset[str]:
    blob = f"{skill.summary} {skill.recipe}".lower()
    # Cheap tokenization: split on whitespace + strip basic punctuation.
    raw = blob.replace(",", " ").replace(".", " ").replace(";", " ").replace(":", " ")
    return frozenset(tok for tok in raw.split() if tok)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _margin(skill: Skill) -> int:
    return skill.posterior[0] - skill.posterior[1]


def _union_lists(a: Iterable[str], b: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for x in a:
        seen.setdefault(x, None)
    for x in b:
        seen.setdefault(x, None)
    return list(seen.keys())


def dedup_near_duplicate(skills: list[Skill], threshold: float = 0.85) -> list[Skill]:
    """Merge near-duplicate skills (Jaccard >= ``threshold``).

    For each merged pair the higher-margin skill is kept; its ``evidence``,
    ``parent_hypothesis_ids``, and ``applicability_conditions`` are unioned
    with the dropped skill's.  Quarantine flags are preserved on the kept
    record (a dropped skill's quarantine status does not propagate).
    """
    if len(skills) <= 1:
        return list(skills)

    # Pre-tokenize once.
    tokens = [_token_set(s) for s in skills]
    kept: list[Skill] = []
    kept_tokens: list[frozenset[str]] = []
    dropped: set[int] = set()

    for i, skill in enumerate(skills):
        if i in dropped:
            continue
        merged = skill
        merged_tokens = tokens[i]
        for j in range(i + 1, len(skills)):
            if j in dropped:
                continue
            if _jaccard(merged_tokens, tokens[j]) < threshold:
                continue
            other = skills[j]
            if _margin(other) > _margin(merged):
                winner, loser = other, merged
            else:
                winner, loser = merged, other
            from dataclasses import replace  # local import keeps top clean

            merged = replace(
                winner,
                evidence=_union_lists(winner.evidence, loser.evidence),
                parent_hypothesis_ids=_union_lists(
                    winner.parent_hypothesis_ids, loser.parent_hypothesis_ids
                ),
                applicability_conditions=_union_lists(
                    winner.applicability_conditions, loser.applicability_conditions
                ),
            )
            merged_tokens = _token_set(merged)
            dropped.add(j)
        kept.append(merged)
        kept_tokens.append(merged_tokens)

    return kept

"""Slim validation helpers for ArcgenticaSimple v14.

The pre-v14 validators (validate_hypothesis_record, validate_action_sequence_record,
validate_skill_record, ...) were tied to HypothesisLedger / ActionSequenceBook /
SkillBook record schemas. v14 uses dataclasses in goal_board.py with from_dict
classmethods that do their own coercion, so most of the legacy validators are
dead code. This stub keeps a tiny surface for any external caller that imports
``ValidationResult`` and the small predicate helpers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    checks: list[str]
    failures: list[str]


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()) if len(tok) >= 3}


def _overlap_score(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _normalize_action_label(action: str) -> str:
    label = str(action).strip()
    if not label:
        return ""
    return label.split("(", 1)[0].strip().upper()


def _is_action_like(step: str) -> bool:
    return bool(re.fullmatch(r"(ACTION[0-9]+|RESET)(\(\d+,\d+\))?", str(step).strip().upper()))


def _coords_only_on_coordinate_action(step: str) -> bool:
    raw = str(step).strip().upper()
    if "(" not in raw:
        return True
    return raw.startswith("ACTION6(")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(text).lower()
    return any(keyword in lowered for keyword in keywords)


def validate_hypothesis_record(record: dict[str, Any]) -> ValidationResult:
    """Legacy stub: accepts any non-empty dict with a non-empty title field."""
    title = str(record.get("title", "")).strip() if isinstance(record, dict) else ""
    if not title:
        return ValidationResult(ok=False, checks=[], failures=["title_missing"])
    return ValidationResult(ok=True, checks=["title_present"], failures=[])


def validate_action_sequence_record(record: dict[str, Any]) -> ValidationResult:
    sequence = record.get("sequence") if isinstance(record, dict) else None
    if not sequence:
        return ValidationResult(ok=False, checks=[], failures=["sequence_missing"])
    return ValidationResult(ok=True, checks=["sequence_present"], failures=[])


def validate_skill_record(record: dict[str, Any]) -> ValidationResult:
    name = str(record.get("name", "")).strip() if isinstance(record, dict) else ""
    if not name:
        return ValidationResult(ok=False, checks=[], failures=["name_missing"])
    return ValidationResult(ok=True, checks=["name_present"], failures=[])

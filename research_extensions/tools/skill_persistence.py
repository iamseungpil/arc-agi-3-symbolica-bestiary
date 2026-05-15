"""Atomic, checksum-validated persistence for the SkillLibrary (S6).

Disk layout for a configured ``persistence_path = "x.json"``:

* ``x.json``          - primary snapshot (canonical-JSON wrapper).
* ``x.json.bak``      - previous primary, rotated on every successful save.
* ``x.json.tmp.<pid>`` - in-flight write target (renamed atomically).
* ``x.json.events.jsonl`` - append-only event log (confirm/falsify lines).

Snapshot wrapper schema::

    {
      "schema_version": "1",
      "checksum": "<sha256 hex of canonical skills array>",
      "skills": [ ...skill objects... ]
    }

The checksum is computed over the canonical-JSON encoding of the **skills
array only** so that bumping ``schema_version`` on the wrapper does not
silently invalidate every existing snapshot.

Durability contract (codex r5):
    1. open temp with ``O_WRONLY|O_CREAT|O_TRUNC``
    2. write canonical bytes
    3. ``os.fsync(fd)`` before close
    4. rotate previous primary to ``.bak``
    5. ``os.replace(tmp, primary)`` (POSIX-atomic)

``load_with_fallback`` never raises: on primary failure it tries ``.bak``,
and on double failure returns ``([], {"loaded_from": "empty"})``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .skill_library import Skill

logger = logging.getLogger(__name__)

__all__ = ["save", "load_with_fallback", "append_event"]


# ---------------------------------------------------------------------------
# Canonical JSON helpers
# ---------------------------------------------------------------------------


def _datetime_handler(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Unserializable type: {type(obj).__name__}")


def _canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=_datetime_handler,
        ensure_ascii=False,
    )


def _skill_to_dict(skill: "Skill") -> dict[str, Any]:
    raw = dataclasses.asdict(skill)
    # Normalize datetime and tuple here so the dict round-trips through
    # plain ``json.loads`` without needing the default= hook on read.
    raw["timestamp"] = (
        skill.timestamp.isoformat()
        if isinstance(skill.timestamp, datetime)
        else skill.timestamp
    )
    raw["posterior"] = list(skill.posterior)
    return raw


def _skill_from_dict(d: dict[str, Any]) -> "Skill":
    # Local import avoids a circular at module load.
    from .skill_library import Skill

    ts_raw = d.get("timestamp")
    if isinstance(ts_raw, str):
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            ts = datetime.now()
    elif isinstance(ts_raw, datetime):
        ts = ts_raw
    else:
        ts = datetime.now()

    posterior_raw = d.get("posterior", [0, 0])
    if isinstance(posterior_raw, list) and len(posterior_raw) == 2:
        posterior = (int(posterior_raw[0]), int(posterior_raw[1]))
    else:
        posterior = (0, 0)

    return Skill(
        skill_id=str(d["skill_id"]),
        summary=str(d.get("summary", "")),
        recipe=str(d.get("recipe", "")),
        evidence=list(d.get("evidence", [])),
        posterior=posterior,
        applicability_conditions=list(d.get("applicability_conditions", [])),
        parent_hypothesis_ids=list(d.get("parent_hypothesis_ids", [])),
        category=d.get("category", "mechanic"),
        predicate=d.get("predicate"),
        code=d.get("code"),
        schema_version=str(d.get("schema_version", "1")),
        predicate_dsl_version=str(d.get("predicate_dsl_version", "1")),
        quarantined=bool(d.get("quarantined", False)),
        quarantine_reason=d.get("quarantine_reason"),
        timestamp=ts,
    )


def _skills_checksum(skill_dicts: list[dict[str, Any]]) -> str:
    inner_bytes = _canonical_json(skill_dicts).encode("utf-8")
    return hashlib.sha256(inner_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save(path: str, skills: list["Skill"], schema_version: str = "1") -> None:
    """Atomically persist ``skills`` to ``path`` (rotating prior to .bak)."""
    skill_dicts = [_skill_to_dict(s) for s in skills]
    checksum = _skills_checksum(skill_dicts)
    wrapper = {
        "schema_version": schema_version,
        "checksum": checksum,
        "skills": skill_dicts,
    }
    payload = _canonical_json(wrapper).encode("utf-8")

    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp.{os.getpid()}"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)

    if os.path.exists(path):
        bak_path = f"{path}.bak"
        try:
            os.replace(path, bak_path)
        except OSError as exc:  # pragma: no cover - filesystem edge
            logger.warning("skill_persistence: .bak rotate failed: %r", exc)

    os.replace(tmp_path, path)


def load_with_fallback(path: str) -> tuple[list["Skill"], dict[str, Any]]:
    """Load primary, falling back to .bak, then to empty.  Never raises."""
    last_error: str = ""

    for candidate, label in ((path, "primary"), (f"{path}.bak", "bak")):
        if not os.path.exists(candidate):
            last_error = f"{label} not present"
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                wrapper = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            last_error = f"{label} read/parse: {exc!r}"
            logger.warning("skill_persistence: %s", last_error)
            continue

        if not isinstance(wrapper, dict):
            last_error = f"{label} wrapper not a dict"
            logger.warning("skill_persistence: %s", last_error)
            continue
        if wrapper.get("schema_version") != "1":
            last_error = f"{label} schema_version mismatch: {wrapper.get('schema_version')!r}"
            logger.warning("skill_persistence: %s", last_error)
            continue

        skill_dicts = wrapper.get("skills")
        if not isinstance(skill_dicts, list):
            last_error = f"{label} skills field not a list"
            logger.warning("skill_persistence: %s", last_error)
            continue

        expected_checksum = wrapper.get("checksum", "")
        actual_checksum = _skills_checksum(skill_dicts)
        if expected_checksum != actual_checksum:
            last_error = (
                f"{label} checksum mismatch "
                f"(expected {expected_checksum!r}, got {actual_checksum!r})"
            )
            logger.warning("skill_persistence: %s", last_error)
            continue

        try:
            skills = [_skill_from_dict(d) for d in skill_dicts]
        except (KeyError, TypeError, ValueError) as exc:
            last_error = f"{label} skill decode: {exc!r}"
            logger.warning("skill_persistence: %s", last_error)
            continue

        return skills, {
            "loaded_from": label,
            "checksum": actual_checksum,
            "count": len(skills),
        }

    return [], {"loaded_from": "empty", "reason": last_error or "no snapshot"}


def append_event(path: str, event: dict[str, Any]) -> None:
    """Append a JSON-line event record to ``<path>.events.jsonl``."""
    events_path = f"{path}.events.jsonl"
    parent = os.path.dirname(events_path) or "."
    os.makedirs(parent, exist_ok=True)

    line = (_canonical_json(event) + "\n").encode("utf-8")
    fd = os.open(events_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)

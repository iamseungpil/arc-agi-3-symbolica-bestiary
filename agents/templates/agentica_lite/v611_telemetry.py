"""v611 per-turn telemetry (Plan rev D §Step 2b Q6, codex round 12-13).

Single function `log_turn_event` writes ONE JSON line per emission
to `logs/v611_turn_telemetry.jsonl`. No buffering. fsync after each
write so the data survives mid-episode crashes.

Emission points (frozen by round 13):
  1. immediately after each role-runner returns (BEFORE validator)
  2. immediately after each validator returns
  3. immediately after env.step returns
  4. immediately after M4 SKILL.md patch applied
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


_LOCK = threading.Lock()


@dataclass
class TurnEvent:
    """Schema for one per-turn telemetry line (codex round 12 Q6)."""

    turn_id: int
    seed: int | None
    episode_id: str | None
    role: str                       # 'm1' | 'm2v' | 'm2e' | 'm4' | 'env' | 'm3'
    event: str                      # 'role_returned' | 'validator_ok' |
                                    #   'validator_fail' | 'env_step' |
                                    #   'm4_patch_applied'
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda:
                            _dt.datetime.now(_dt.timezone.utc).isoformat())


_DEFAULT_LOG_PATH = "logs/v611_turn_telemetry.jsonl"


def _resolve_log_path() -> Path:
    """Allow override via env var; default to logs/v611_turn_telemetry.jsonl."""
    p = os.environ.get("V611_TELEMETRY_PATH", _DEFAULT_LOG_PATH)
    path = Path(p)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_turn_event(
    turn_id: int,
    role: str,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    seed: int | None = None,
    episode_id: str | None = None,
    log_path: Path | None = None,
) -> None:
    """Append one TurnEvent to the telemetry log.

    fsync after each write (per round 12 spec — no buffered loss).
    """
    e = TurnEvent(
        turn_id=turn_id,
        seed=seed,
        episode_id=episode_id,
        role=role,
        event=event,
        payload=payload or {},
    )
    line = json.dumps(asdict(e), default=str, ensure_ascii=False)
    target = log_path or _resolve_log_path()
    with _LOCK:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass  # fsync not supported on some fs (best-effort)


def read_telemetry(log_path: Path | str | None = None) -> list[TurnEvent]:
    """Read back all events from the log (for tests / analysis)."""
    target = Path(log_path) if log_path else _resolve_log_path()
    if not target.exists():
        return []
    out: list[TurnEvent] = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(TurnEvent(
                turn_id=d.get("turn_id", -1),
                seed=d.get("seed"),
                episode_id=d.get("episode_id"),
                role=d.get("role", ""),
                event=d.get("event", ""),
                payload=d.get("payload", {}) or {},
                timestamp=d.get("timestamp", ""),
            ))
    return out

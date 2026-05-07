"""Plan v588 B16 — Symbolica action-state chain compression.

Pure deterministic functions. No LLM. Used at runtime by agent.py to
build the chain payload that M1/M3/M4 receive.

Entry point: compress_action_state_chain(...).

Two compression levels (post-critic round-3 C-16):
  - "full"    : all features. M3 input.
  - "compact" : tokens last 15 + features + errors last 5 +
                causal_table top-5. M1 input.

Anti-leak: vocabulary used is "click" / "non_marker" /
"marker_multicolor" / "outside" / "unknown_kind" — all already in
the v57 region_kind_pre vocabulary. No new game-specific terms.
The leak vocabulary list (per scripts/check_no_leak_prompts.py) is
NOT used anywhere in the output payload.
"""

from __future__ import annotations

import re
from typing import Iterable


_TOKEN_FIELD_SCRUBBER = re.compile(r"[|\n]")
_DEFAULT_MAX_CHAIN_LENGTH = 30


def _scrub(token_field: object) -> str:
    """Defensive: strip pipe and newline characters from any field that
    might appear in a token string. Per C-18."""
    s = "" if token_field is None else str(token_field)
    return _TOKEN_FIELD_SCRUBBER.sub("_", s)[:30]


# ---------------------------------------------------------------------
# 1. Tokenisation (M-CHAIN-1)
# ---------------------------------------------------------------------


def _format_transition(transition: object) -> str:
    if not isinstance(transition, dict):
        return "_"
    f, t = transition.get("from"), transition.get("to")
    if f is None or t is None:
        return "_"
    return f"{f}->{t}"


def _summarise_compass_changes(compass_changes: object) -> str:
    """Return a compact comma-joined string of compass deltas.
    Example: 'M12_N:9->2,M20_E:9->2'. Empty when no changes."""
    if not isinstance(compass_changes, list) or not compass_changes:
        return "_"
    parts = []
    for c in compass_changes[:6]:
        if not isinstance(c, dict):
            continue
        mid = _scrub(c.get("marker_id"))
        d = _scrub(c.get("direction"))
        f, t = c.get("from"), c.get("to")
        parts.append(f"{mid}_{d}:{f}->{t}")
    return ",".join(parts) if parts else "_"


def tokenise_turn(verbose: dict, diff: dict) -> str:
    """Format one turn entry as 'T<turn>|click|<region_kind>(<R-id>,<color>)|<transition>|<compass>|L+<delta>'.

    `verbose` is one entry of board.recent_verbose; `diff` is the
    matching entry from board.recent_turn_diffs. When fields are
    missing (legacy traces), substitute 'unknown_kind' or '_'.
    Per C-13."""
    turn = verbose.get("turn") if isinstance(verbose, dict) else None
    obs = (verbose or {}).get("observation") or {}
    rid = obs.get("primary_region_id") or (diff or {}).get("click_region_id") or "_outside_"
    transition = obs.get("dominant_transition") or {}
    transition_str = _format_transition(transition)
    region_kind = (diff or {}).get("region_kind_pre") or "unknown_kind"
    region_color = (diff or {}).get("region_color_pre")
    color_str = f"{region_color}" if region_color is not None else "_"
    compass_str = _summarise_compass_changes((diff or {}).get("compass_changes"))
    level_delta = int(obs.get("level_delta") or 0)
    return (
        f"T{turn}|click|{_scrub(region_kind)}({_scrub(rid)},{_scrub(color_str)})"
        f"|{_scrub(transition_str)}|{compass_str}|L+{level_delta}"
    )


def build_chain_tokens(
    recent_verbose: list[dict],
    recent_turn_diffs: list[dict],
    max_chain_length: int = _DEFAULT_MAX_CHAIN_LENGTH,
) -> list[str]:
    """Build one token per turn using BOTH verbose and diffs.

    BUGFIX (2026-05-07): originally iterated over recent_verbose, which
    is bounded by V57_VERBOSE_WINDOW=3 — so chain_tokens never grew
    past 3, and the M3 chain≥10 gate never fired. Fixed by iterating
    over the longer collection (recent_turn_diffs is up to
    _TURN_DIFF_MAX_RAW=24) and looking up verbose by turn for richer
    fields (transition, level_delta) where available; otherwise
    synthesising the token from diff fields alone.

    Returns last `max_chain_length` tokens."""
    diffs = list(recent_turn_diffs or [])
    verboses = list(recent_verbose or [])
    if not diffs and not verboses:
        return []
    verbose_by_turn = {}
    for v in verboses:
        if isinstance(v, dict) and "turn" in v:
            verbose_by_turn[v["turn"]] = v
    diffs_by_turn = {}
    for d in diffs:
        if isinstance(d, dict) and "turn" in d:
            diffs_by_turn[d["turn"]] = d
    # Union of all turns we have ANY data for; iterate ascending.
    all_turns = sorted(set(verbose_by_turn) | set(diffs_by_turn))
    tokens = []
    for t in all_turns:
        v = verbose_by_turn.get(t, {"turn": t, "observation": {}})
        d = diffs_by_turn.get(t, {"turn": t})
        tokens.append(tokenise_turn(v, d))
    return tokens[-max_chain_length:]


# ---------------------------------------------------------------------
# 2. Trajectory features (M-CHAIN-2)
# ---------------------------------------------------------------------


def _level_delta_curve_shape(verbose: list[dict]) -> str:
    """C-14: rename from 'marker_progress_monotonicity'. Returns
    'rising_only', 'flat', or 'oscillating' based on level_delta
    sequence."""
    seq = [int(((v or {}).get("observation") or {}).get("level_delta") or 0) for v in verbose]
    if not seq or all(s == 0 for s in seq):
        return "flat"
    rises = sum(1 for s in seq if s >= 1)
    if rises == 0:
        return "flat"
    # Treat as oscillating when ≥3 L+ events in window AND some 0s in
    # between (the actual ft09 failure mode).
    if rises >= 3:
        zeros_between = any(seq[i] == 0 and seq[i - 1] >= 1 for i in range(1, len(seq)))
        return "oscillating" if zeros_between else "rising_only"
    return "rising_only"


def compute_trajectory_features(
    recent_verbose: list[dict],
    recent_turn_diffs: list[dict],
) -> dict:
    """Aggregate stats over the chain. Pure deterministic.
    BUGFIX (review round 1, Q-H): iterate union-of-turns from
    BOTH verbose and diffs — was previously bounded by recent_verbose
    (=3 entries) so chain_length/unique_regions/lp_event_intervals
    were all degenerate."""
    verbose_by_turn = {v["turn"]: v for v in (recent_verbose or [])
                       if isinstance(v, dict) and v.get("turn") is not None}
    diffs_by_turn = {d["turn"]: d for d in (recent_turn_diffs or [])
                     if isinstance(d, dict) and d.get("turn") is not None}
    all_turns = sorted(set(verbose_by_turn) | set(diffs_by_turn))
    n = len(all_turns)
    if n == 0:
        return {
            "chain_length": 0,
            "unique_regions": 0,
            "repeat_click_max_count": 0,
            "compass_changes_per_turn": 0.0,
            "level_delta_curve_shape": "flat",
            "kind_distribution": {},
            "lp_event_intervals": [],
        }
    # Region ids — pull from verbose obs first, fall back to diff click_region_id.
    region_ids = []
    for t in all_turns:
        v = verbose_by_turn.get(t, {})
        d = diffs_by_turn.get(t, {})
        rid = ((v or {}).get("observation") or {}).get("primary_region_id") or (d or {}).get("click_region_id")
        region_ids.append(rid)
    valid_rids = [r for r in region_ids if r and r != "_outside_"]
    region_counts: dict[str, int] = {}
    for r in valid_rids:
        region_counts[r] = region_counts.get(r, 0) + 1
    repeat_max = max(region_counts.values()) if region_counts else 0
    unique_n = len(region_counts)
    # Compass-change density.
    cc_total = 0
    for t in all_turns:
        d = diffs_by_turn.get(t, {})
        cc = (d or {}).get("compass_changes")
        if isinstance(cc, list):
            cc_total += len(cc)
    cc_per_turn = round(cc_total / max(1, n), 3)
    # Kind distribution.
    kind_counts: dict[str, int] = {}
    for t in all_turns:
        d = diffs_by_turn.get(t, {})
        k = (d or {}).get("region_kind_pre") or "unknown_kind"
        kind_counts[k] = kind_counts.get(k, 0) + 1
    # L+ event intervals — read level_delta from verbose first, then diff.
    lp_turns = []
    for t in all_turns:
        v = verbose_by_turn.get(t, {})
        d = diffs_by_turn.get(t, {})
        ld = int(((v or {}).get("observation") or {}).get("level_delta") or
                 (d or {}).get("level_delta") or 0)
        if ld >= 1:
            lp_turns.append(t)
    intervals = [lp_turns[i] - lp_turns[i - 1] for i in range(1, len(lp_turns))]
    # Curve shape — synthesise level_delta sequence from union.
    ld_seq = []
    for t in all_turns:
        v = verbose_by_turn.get(t, {})
        d = diffs_by_turn.get(t, {})
        ld_seq.append({"observation": {"level_delta":
            int(((v or {}).get("observation") or {}).get("level_delta") or
                (d or {}).get("level_delta") or 0)}})
    # Round-3 reviewer fix #3: lp_attempt_rate aggregation. We
    # estimate lp_attempt = (verbose.expected.level_delta>=1 OR
    # verbose.observation.level_delta>=1) per turn. With chain_verbose
    # missing for many turns, use observed-only side as a proxy.
    lp_attempts = 0
    for t in all_turns:
        v = verbose_by_turn.get(t, {})
        action = v.get("action") if isinstance(v.get("action"), dict) else {}
        expected = action.get("expected_observation") if isinstance(
            action.get("expected_observation"), dict) else None
        exp_ld = int(expected.get("level_delta") or 0) if expected else 0
        obs_ld = int(((v or {}).get("observation") or {}).get("level_delta") or 0)
        if exp_ld >= 1 or obs_ld >= 1:
            lp_attempts += 1
    lp_attempt_rate = round(lp_attempts / max(1, n), 3)
    return {
        "chain_length": n,
        "unique_regions": unique_n,
        "repeat_click_max_count": repeat_max,
        "compass_changes_per_turn": cc_per_turn,
        "level_delta_curve_shape": _level_delta_curve_shape(ld_seq),
        "kind_distribution": kind_counts,
        "lp_event_intervals": intervals,
        "lp_attempt_rate": lp_attempt_rate,
    }


# ---------------------------------------------------------------------
# 3. Prediction-error chain (M-CHAIN-3)
# ---------------------------------------------------------------------


def compute_prediction_errors(
    recent_verbose: list[dict],
    recent_turn_diffs: list[dict] | None = None,
) -> list[dict]:
    """Per turn: compare the agent's expected_observation against the
    actual observation.

    Per round-2 reviewer D-1: stay VERBOSE-ONLY iteration (not union)
    because `expected_observation` only lives in verbose entries and a
    union iteration would create flood of null rows. Length of this
    channel must instead be raised by feeding a longer verbose buffer
    (chain_verbose, see V57Board).

    Adds `lp_attempt: bool` per round-1 Q-B counter — flag turns where
    either expected or observed level_delta >= 1 so consumers can
    stratify (turn-where-L+-was-attempted vs zero-zero confirmation)."""
    out: list[dict] = []
    for v in recent_verbose or []:
        if not isinstance(v, dict):
            continue
        action = v.get("action") or {}
        action_dict = action if isinstance(action, dict) else {}
        expected = (
            action_dict.get("expected_observation")
            if isinstance(action_dict.get("expected_observation"), dict)
            else None
        )
        obs = v.get("observation") or {}
        obs_ld = int(obs.get("level_delta") or 0)
        if not expected:
            out.append({
                "turn": v.get("turn"),
                "region_id_match": None,
                "transition_match": None,
                "level_delta_match": None,
                "surprise_score": 0,
                "lp_attempt": obs_ld >= 1,
            })
            continue
        # Region id.
        exp_rid = expected.get("primary_region_id")
        obs_rid = obs.get("primary_region_id")
        rid_match = (
            None if exp_rid is None
            else (exp_rid == obs_rid)
        )
        # Transition.
        exp_t = expected.get("dominant_transition") if isinstance(expected.get("dominant_transition"), dict) else None
        obs_t = obs.get("dominant_transition") if isinstance(obs.get("dominant_transition"), dict) else None
        if exp_t is None:
            t_match = None
        elif obs_t is None:
            t_match = False
        else:
            t_match = (exp_t.get("from") == obs_t.get("from")
                       and exp_t.get("to") == obs_t.get("to"))
        # Level delta.
        exp_ld = expected.get("level_delta")
        ld_match = (
            None if exp_ld is None
            else (int(exp_ld or 0) == obs_ld)
        )
        # Surprise score: count False matches.
        score = sum(
            1
            for m in (rid_match, t_match, ld_match)
            if m is False
        )
        # lp_attempt — turn worth attending to for diagnostic purposes
        # (per Q-B counter): either expected or observed L+ ≥ 1.
        exp_ld_int = int(exp_ld or 0) if exp_ld is not None else 0
        lp_attempt = (exp_ld_int >= 1) or (obs_ld >= 1)
        out.append({
            "turn": v.get("turn"),
            "region_id_match": rid_match,
            "transition_match": t_match,
            "level_delta_match": ld_match,
            "surprise_score": score,
            "lp_attempt": lp_attempt,
        })
    return out


# ---------------------------------------------------------------------
# 4. Causal frequency table (M-CHAIN-4)
# ---------------------------------------------------------------------


def compute_causal_table(
    recent_verbose: list[dict],
    recent_turn_diffs: list[dict],
    min_count: int | None = None,
) -> list[dict]:
    """For each (region_kind, transition) pair, count occurrences and
    compute next-1-turn (avg) AND next-3-turn (max) level_delta rates.

    Round-2 reviewer D-2 push-back: ft09 marker-progress can be
    multi-step (animation/state propagation), so pure lag=1 throws
    away signal. Emit BOTH lag1_avg AND lag3_max so the LLM can
    triangulate.

    Round-2 reviewer N-1: min_count fixed at 4 produces zero rows for
    short games. Adaptive: max(2, min(4, chain_length // 15)).

    BUGFIX history:
    - round-1 Q-H: iterate union-of-turns from BOTH verbose and diffs
      (was bounded by recent_verbose=3).
    - round-1 Q-C: original used SUM over 3-turn window with
      overlapping samples — single L+ inflated up to 3 rows. Now
      separated into lag1_avg (non-overlapping) and lag3_max
      (overlap-tolerant but doesn't double-count).
    - round-1 Q-C: added n_distinct_lp_event_turns so LLM can
      discount when only 1 distinct event drove the count.
    """
    diffs_by_turn: dict[int, dict] = {}
    for d in recent_turn_diffs or []:
        if isinstance(d, dict) and d.get("turn") is not None:
            diffs_by_turn[int(d["turn"])] = d
    verbose_by_turn: dict[int, dict] = {}
    for v in recent_verbose or []:
        if isinstance(v, dict) and v.get("turn") is not None:
            verbose_by_turn[int(v["turn"])] = v

    # Iterate union ascending (Q-H fix).
    turns = sorted(set(verbose_by_turn) | set(diffs_by_turn))
    if not turns:
        return []
    # Adaptive min_count (N-1): scale with chain length.
    if min_count is None:
        min_count = max(2, min(4, len(turns) // 15))

    bucket: dict[tuple[str, str], dict] = {}
    for i, t in enumerate(turns):
        v = verbose_by_turn.get(t, {})
        d = diffs_by_turn.get(t, {})
        kind = (d or {}).get("region_kind_pre") or "unknown_kind"
        obs = v.get("observation") or {}
        transition_str = _format_transition(obs.get("dominant_transition"))
        if transition_str == "_":
            continue
        # Lag-1 next turn (D-2 lag1).
        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        if next_turn is None:
            continue
        next_v = verbose_by_turn.get(next_turn, {})
        next_d = diffs_by_turn.get(next_turn, {})
        next_ld = int((next_v.get("observation") or {}).get("level_delta") or
                      (next_d or {}).get("level_delta") or 0)
        # Lag-3 max (D-2 lag3_max — keeps multi-step marker-progress signal).
        lag3_window = turns[i + 1: i + 4]
        lag3_max = 0
        lag3_lp_turns: set[int] = set()
        for tt in lag3_window:
            tv = verbose_by_turn.get(tt, {})
            td = diffs_by_turn.get(tt, {})
            ld = int((tv.get("observation") or {}).get("level_delta") or
                     (td or {}).get("level_delta") or 0)
            if ld > lag3_max:
                lag3_max = ld
            if ld >= 1:
                lag3_lp_turns.add(tt)
        key = (kind, transition_str)
        b = bucket.setdefault(key, {
            "count": 0, "next_ld_sum": 0,
            "lag1_lp_turns": set(),
            "lag3_max_ld_sum": 0, "lag3_lp_turns": set(),
        })
        b["count"] += 1
        b["next_ld_sum"] += next_ld
        if next_ld >= 1:
            b["lag1_lp_turns"].add(next_turn)
        b["lag3_max_ld_sum"] += lag3_max
        b["lag3_lp_turns"].update(lag3_lp_turns)

    rows: list[dict] = []
    for (kind, transition), b in bucket.items():
        if b["count"] < min_count:
            continue
        rows.append({
            "region_kind": kind,
            "transition": transition,
            "count": b["count"],
            "lag1_avg_level_delta": round(b["next_ld_sum"] / b["count"], 3),
            "lag3_avg_max_level_delta": round(b["lag3_max_ld_sum"] / b["count"], 3),
            "n_distinct_lp_lag1": len(b["lag1_lp_turns"]),
            "n_distinct_lp_lag3": len(b["lag3_lp_turns"]),
        })
    rows.sort(key=lambda r: -r["count"])
    return rows


# ---------------------------------------------------------------------
# Top-level entry: full vs compact per C-16.
# ---------------------------------------------------------------------


def compress_action_state_chain(
    recent_verbose: list[dict],
    recent_turn_diffs: list[dict],
    *,
    max_chain_length: int = _DEFAULT_MAX_CHAIN_LENGTH,
    level: str = "full",
) -> dict:
    """Build the chain payload.

    `level="full"` returns all four channels (M3 input).
    `level="compact"` keeps tokens last 15, features full, errors last 5,
    causal_table top-5 rows (M1 input).

    Empty input returns the four keys with empty/zero values.
    """
    tokens = build_chain_tokens(
        recent_verbose, recent_turn_diffs, max_chain_length=max_chain_length
    )
    features = compute_trajectory_features(recent_verbose, recent_turn_diffs)
    errors = compute_prediction_errors(recent_verbose)
    causal = compute_causal_table(recent_verbose, recent_turn_diffs)

    if level == "compact":
        tokens = tokens[-15:]
        errors = errors[-5:]
        causal = causal[:5]

    return {
        "chain_tokens": tokens,
        "trajectory_features": features,
        "prediction_errors": errors,
        "causal_table": causal,
    }

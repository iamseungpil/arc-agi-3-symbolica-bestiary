"""v607 Phase 2 — Anti-leak validation for Reflector-emitted chid_templates.

Plan v607 rev E §5.1 U5 (codex r5 explicit set):
  HARD validators applied at Reflector emit time AND Proposer schema time:
    1. V-LEAK regex: forbidden ft09-specific tokens
       (R28, R31, R12, gqb, bsT, Hkx, NTi, kCv, cwU, elp, Ycb)
    2. cycle237 4-char substring ban: any 4-char substring from cycle237 trace
       chid strings is forbidden (prevents paraphrase leak)
    3. region_hint ∈ visible_region_ids — enforced at runtime in Proposer
       schema (already in proposer.py), not in this module
    4. TF-IDF cosine > 0.8 vs forbidden corpus → reject (char-3gram, pure
       Python implementation with optional sklearn fallback)

Returns (ok: bool, reason_code: str) where reason_code ∈ {
    "ok",
    "v_leak_token",
    "cycle237_4char",
    "tfidf_cosine_high",
    "empty_template",
}.

Used by:
  - reflector.py:emit_new_chid (post-LLM, before skill_state mutation)
  - proposer.py:_validate_schema (defense-in-depth on Proposer output)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Tuple

# =============================================================================
# Constants — explicit anti-leak token set (codex r5 R2)
# =============================================================================

# V-LEAK tokens: ft09-specific identifiers extracted by the existing leak
# checker (scripts/check_no_leak_prompts.py). Reflector chid_templates must
# never contain any of these as substrings.
V_LEAK_TOKENS: tuple[str, ...] = (
    "R28", "R31", "R12",
    "gqb", "bsT", "Hkx", "NTi", "kCv", "cwU", "elp", "Ycb",
)

# Plan v607 says "cycle237 4-char substring ban". Loaded lazily from trace
# file at first call; cached in module global. Tests can override via
# `_set_forbidden_corpus(...)` for isolation.
_FORBIDDEN_CHIDS: list[str] | None = None
_FORBIDDEN_4GRAMS: set[str] | None = None

CYCLE237_TRACE_PATH = Path(
    "simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl"
)

# TF-IDF cosine threshold (plan v607 rev E §5.1 U5 HARD gate).
TFIDF_COSINE_THRESHOLD = 0.8


# =============================================================================
# Forbidden corpus loader (cycle237 chids → 4-grams)
# =============================================================================

# Common English fragments / generic predicate skeleton tokens to ignore.
# These appear in cycle237 chids but are NOT ft09-distinguishing.
_GENERIC_TOKEN_STOPLIST: set[str] = {
    "of", "on", "in", "to", "at", "or", "and", "the", "a",
    "marker", "active", "recent", "parity", "click", "revert",
    "repeat", "change", "neighbor", "action", "static", "complete",
    "kin", "no", "non", "fresh", "trigger", "prior", "plan", "probe",
    "replay", "relation", "overlap", "T", "R",
}


def _load_forbidden_corpus() -> tuple[list[str], set[str]]:
    """Read cycle237 trace.jsonl chids; return (chid list, distinguishing token set).

    Codex r5 R2 intent revised: ban cycle237 SEMANTIC tokens (crop, compass,
    sector, alignment, sweep, etc.) that are ft09-distinguishing. Common
    English (action, click, change) is in stoplist to avoid over-blocking.
    """
    chids: list[str] = []
    if CYCLE237_TRACE_PATH.exists():
        try:
            with CYCLE237_TRACE_PATH.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    act = row.get("action")
                    if not isinstance(act, dict):
                        continue
                    chid = act.get("chosen_hypothesis_id")
                    if chid and isinstance(chid, str):
                        chids.append(chid)
        except Exception:
            pass
    # Tokenize by `_` and `:` (cycle237 chids use both separators).
    tokens: set[str] = set()
    for c in chids:
        for tok in re.split(r"[_:]+", c):
            if (
                len(tok) >= 4
                and tok.isalpha()
                and tok.lower() not in _GENERIC_TOKEN_STOPLIST
            ):
                tokens.add(tok.lower())
    return chids, tokens


def _ensure_corpus() -> tuple[list[str], set[str]]:
    global _FORBIDDEN_CHIDS, _FORBIDDEN_4GRAMS
    if _FORBIDDEN_CHIDS is None or _FORBIDDEN_4GRAMS is None:
        _FORBIDDEN_CHIDS, _FORBIDDEN_4GRAMS = _load_forbidden_corpus()
    return _FORBIDDEN_CHIDS, _FORBIDDEN_4GRAMS


def _set_forbidden_corpus(chids: list[str]) -> None:
    """Test hook: override corpus for fixture isolation."""
    global _FORBIDDEN_CHIDS, _FORBIDDEN_4GRAMS
    _FORBIDDEN_CHIDS = list(chids)
    tokens: set[str] = set()
    for c in chids:
        for tok in re.split(r"[_:]+", c):
            if (
                len(tok) >= 4
                and tok.isalpha()
                and tok.lower() not in _GENERIC_TOKEN_STOPLIST
            ):
                tokens.add(tok.lower())
    _FORBIDDEN_4GRAMS = tokens  # name retained for back-compat; now stores tokens


def _clear_forbidden_corpus() -> None:
    """Test hook: revert to lazy-load behaviour."""
    global _FORBIDDEN_CHIDS, _FORBIDDEN_4GRAMS
    _FORBIDDEN_CHIDS = None
    _FORBIDDEN_4GRAMS = None


# =============================================================================
# TF-IDF char-3gram cosine (pure Python, no sklearn dep)
# =============================================================================

def _char_3gram_vector(s: str) -> Counter[str]:
    grams: Counter[str] = Counter()
    s2 = s.lower()
    for i in range(len(s2) - 2):
        grams[s2[i : i + 3]] += 1
    return grams


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[g] * b[g] for g in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _max_cosine_vs_corpus(s: str) -> float:
    """Pairwise max cosine (codex r5 R2 decision: pairwise max not aggregate)."""
    chids, _ = _ensure_corpus()
    if not chids:
        return 0.0
    s_vec = _char_3gram_vector(s)
    best = 0.0
    for c in chids:
        cv = _char_3gram_vector(c)
        sc = _cosine(s_vec, cv)
        if sc > best:
            best = sc
    return best


# =============================================================================
# Public API
# =============================================================================

def validate_chid_template(s: str) -> Tuple[bool, str]:
    """Validate a Reflector-emitted chid_template against anti-leak rules.

    Returns (ok, reason_code). On reject, reason_code identifies the specific
    rule violated (V_LEAK_TOKENS / cycle237 4-gram / TF-IDF cosine).

    Empty template short-circuits to empty_template reject.

    v607 Phase 9 (P4 hybrid): ARC_LITE_ANTI_LEAK_MODE env var controls strictness:
      "strict" (default): all 3 rules active (V-LEAK + cycle237 + TF-IDF)
      "v_leak_only":      only V-LEAK ft09-identifier tokens; cycle237 vocab allowed
      "off":              all rules disabled (debug only)
    """
    if not s or not s.strip():
        return False, "empty_template"
    import os as _os
    mode = _os.environ.get("ARC_LITE_ANTI_LEAK_MODE", "strict").lower()
    if mode == "off":
        return True, "ok"
    # Rule 1: V-LEAK forbidden tokens (always active unless mode=off).
    for tok in V_LEAK_TOKENS:
        if tok in s:
            return False, "v_leak_token"
    if mode == "v_leak_only":
        return True, "ok"
    # Rule 2: cycle237 distinguishing-token ban (case-insensitive substring match).
    _, tokens = _ensure_corpus()
    s_lower = s.lower()
    for tok in tokens:
        if tok in s_lower:
            return False, f"cycle237_token:{tok}"
    # Rule 3: TF-IDF cosine (paraphrase bypass mitigation).
    cos = _max_cosine_vs_corpus(s)
    if cos > TFIDF_COSINE_THRESHOLD:
        return False, "tfidf_cosine_high"
    return True, "ok"

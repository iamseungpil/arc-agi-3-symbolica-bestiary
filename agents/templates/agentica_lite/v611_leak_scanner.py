"""v611 leak-audited scanner (Plan rev B §canary leak test, FROZEN).

Codex 7-round adversarial review converged to this exact spec.

Three responsibilities:
1. `canon(text)` — canonicalization pipeline shared by artifact text
   AND phrase library before n-gram matching.
2. `scan_artifact(text, phrase_lib)` — per-source independent scan;
   returns list of n-gram hits.
3. `validate_confirmed_skill(skill)` — hygiene validator for SKILL.md
   `confirmed_skills` (no coords, no region/level IDs, no ft09 vocab).

The canary phrase library is a deliberately FALSE statement about
ft09 ('Quadrant 4 always wraps to quadrant 0 after 2-step rotation').
If episode-2 artifacts reference any phrase from this library, that
proves cross-episode contamination from inherited SKILL.md.

Acceptance gate (used in v611 Step 4 canary leak test):
  scanner_hits_real <= B  (B = baseline noise from no-canary control)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# canon() — shared by artifact and phrase library
# ─────────────────────────────────────────────────────────────────


def canon(text: str) -> list[str]:
    """Canonicalize text into a token list.

    Pipeline (FROZEN — Plan v611 rev B §scanner spec):
    1. lowercase
    2. strip non-alphanumeric (keep digits + whitespace)
    3. collapse multiple whitespace
    4. split on whitespace
    """
    if not text:
        return []
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    return s.split()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


# ─────────────────────────────────────────────────────────────────
# Canary phrase library (FROZEN)
# ─────────────────────────────────────────────────────────────────
#
# These phrases canonicalize to specific n-grams that must NOT appear
# in episode-2 artifacts if the FALSE canary card was inherited.

CANARY_RAW_PHRASES = (
    "quadrant",
    "rotate",
    "rotation",
    "wrap",
    "sector",
    "quad",
    "2-step",
    "two step",
    "wrap around",
    "sector rotate",
    "q4 wraps to",
    "two step rotation",
    "quadrant cycle",
)


def _compile_phrase_library(
    raw_phrases: tuple[str, ...] = CANARY_RAW_PHRASES,
) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str, str]]]:
    """Compile raw canary phrases into (unigrams, bigrams, trigrams)."""
    unigrams: set[str] = set()
    bigrams: set[tuple[str, str]] = set()
    trigrams: set[tuple[str, str, str]] = set()
    for raw in raw_phrases:
        toks = canon(raw)
        if len(toks) == 1:
            unigrams.add(toks[0])
        elif len(toks) == 2:
            bigrams.add((toks[0], toks[1]))
        elif len(toks) >= 3:
            for i in range(len(toks) - 2):
                trigrams.add((toks[i], toks[i + 1], toks[i + 2]))
    return unigrams, bigrams, trigrams


CANARY_UNIGRAMS, CANARY_BIGRAMS, CANARY_TRIGRAMS = _compile_phrase_library()


# ─────────────────────────────────────────────────────────────────
# scan_artifact — per-source independent (no cross-source n-grams)
# ─────────────────────────────────────────────────────────────────


@dataclass
class SourceHit:
    source_name: str
    unigram_hits: set[str]
    bigram_hits: set[tuple[str, str]]
    trigram_hits: set[tuple[str, str, str]]

    def total(self) -> int:
        return (len(self.unigram_hits)
                + len(self.bigram_hits)
                + len(self.trigram_hits))


def scan_artifact(
    source_name: str,
    text: str,
    canary_unigrams: set[str] = CANARY_UNIGRAMS,
    canary_bigrams: set[tuple[str, str]] = CANARY_BIGRAMS,
    canary_trigrams: set[tuple[str, str, str]] = CANARY_TRIGRAMS,
) -> SourceHit:
    """Single-source independent scan. No cross-source n-grams.

    Returns SourceHit with per-source violation counts.
    """
    toks = canon(text)
    uni = set(toks)
    bi = ngrams(toks, 2)
    tri = ngrams(toks, 3)
    return SourceHit(
        source_name=source_name,
        unigram_hits=uni & canary_unigrams,
        bigram_hits=bi & canary_bigrams,
        trigram_hits=tri & canary_trigrams,
    )


def scan_all_sources(
    sources: dict[str, str],
) -> tuple[int, list[SourceHit]]:
    """Scan multiple sources independently. Returns (total_hits, per_src)."""
    per_source: list[SourceHit] = []
    for name, text in sources.items():
        hit = scan_artifact(name, text)
        if hit.total() > 0:
            per_source.append(hit)
    total = sum(h.total() for h in per_source)
    return total, per_source


# ─────────────────────────────────────────────────────────────────
# validate_confirmed_skill — SKILL.md hygiene gate
# ─────────────────────────────────────────────────────────────────
#
# Plan v611 rev B §memory hygiene rules:
# - confirmed_skills: NO `\d` patterns, NO `[CR]\d+`, NO `L[1-6]`,
#   NO ft09 vocab (bsT, gqb, Hkx, NTi, kCv, cwU, elp, Ycb).

_FORBIDDEN_PATTERNS = [
    re.compile(r"\b\d{1,4}\b"),         # raw coords / level numbers
    re.compile(r"\b[CR]\d+\b"),         # region IDs C1/R12 etc
    re.compile(r"\bL[1-6]\b"),          # level identifiers
]
_FORBIDDEN_FT09_VOCAB = (
    "bsT", "gqb", "Hkx", "NTi", "kCv", "cwU", "elp", "Ycb",
    "AcT", "aIV", "ajm",  # ft09 sprite names (env_files)
)


@dataclass
class HygieneResult:
    skill_id: str | None
    ok: bool
    violations: list[str]


def validate_confirmed_skill(skill: dict) -> HygieneResult:
    """Validate a single `confirmed_skill` card.

    Plan v611 rev B §SKILL.md no-leak rules. Returns ok=False on first
    violation; collects ALL violations for reporting.
    """
    sid = skill.get("id") or skill.get("skill_id")
    text_parts = []
    for field in ("nl_description", "claim", "abstract_precondition",
                   "expected_observed_effect"):
        v = skill.get(field)
        if isinstance(v, str):
            text_parts.append(v)
        elif isinstance(v, list):
            text_parts.extend(s for s in v if isinstance(s, str))
    bundle = " ".join(text_parts)
    violations: list[str] = []
    for pat in _FORBIDDEN_PATTERNS:
        for match in pat.finditer(bundle):
            violations.append(f"forbidden pattern '{match.group()}' "
                              f"(rule={pat.pattern})")
    for tok in _FORBIDDEN_FT09_VOCAB:
        if tok in bundle:
            violations.append(f"ft09 vocab '{tok}' in confirmed_skill")
    return HygieneResult(
        skill_id=sid,
        ok=(len(violations) == 0),
        violations=violations,
    )


def filter_inheritable_skills(skills: list[dict]) -> tuple[list[dict], list[HygieneResult]]:
    """Filter `confirmed_skills` for episode inheritance.

    Returns (clean_skills, rejected_results). Rejected skills are NOT
    propagated to next episode SKILL.md.
    """
    clean: list[dict] = []
    rejected: list[HygieneResult] = []
    for s in skills:
        res = validate_confirmed_skill(s)
        if res.ok:
            clean.append(s)
        else:
            rejected.append(res)
    return clean, rejected

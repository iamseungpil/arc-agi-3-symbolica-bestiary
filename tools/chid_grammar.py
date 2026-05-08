"""B19 v591 — Invented chid grammar + leak scan + region anchoring.

TIER-B chids let M1 propose a hypothesis that the deterministic predicate
library cannot express (cross-region relation, sprite-internal geometry,
negation, sequence). The grammar is intentionally permissive to cover
cycle237's empirical success patterns, but every emission is gated by:
  (1) a regex shape check that disallows free-form text;
  (2) a leak-vocab scan reusing the FORBIDDEN_VOCAB constant from prompts;
  (3) at least one R<id> referencing a CURRENTLY visible region;
  (4) optional direction suffix from a closed compass set.

cycle237 reference chids that MUST validate (extracted from
simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl):

  H_static_nonmarker_R19            -> rationale=static_nonmarker, R19
  H_crop_align_R31_NW               -> rationale=crop_align, R31, dir=NW
  H_crop_align_R31_W                -> rationale=crop_align, R31, dir=W
  P_crop_compass_sweep_R31          -> rationale=crop_compass_sweep, R31
  H_fresh_neighbor_toggle_R16       -> rationale=fresh_neighbor_toggle
  H_R15_lower_S                     -> rationale=lower (mid-string R-id)
  P_R12_crop_sector_alignment       -> rationale=crop_sector_alignment
  H_crop_overlap_probe              -> NO R-id (history-anchored variant)

The grammar is therefore:
  ^(H|P)_<token-sequence-with-at-least-one-Rdigits>$

where token-sequence may have R<digits> appearing anywhere, and an
optional 1- or 2-letter compass direction at the end.
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple

INVENTED_CHID_RE = re.compile(
    r"^(?P<prefix>H|P)_"
    r"(?P<body>[A-Za-z][A-Za-z0-9_]{2,80})$"
)

R_ID_RE = re.compile(r"R(\d+)")
DIRECTION_TAIL_RE = re.compile(r"_(NE|NW|SE|SW|N|S|E|W)$")
HISTORY_ANCHOR_RE = re.compile(r"(?<![A-Za-z0-9])T\d+(?![A-Za-z0-9])")

# History-anchored rationale keywords: TIER-B chids that reference past
# behaviour rather than a current region. cycle237 emitted bare bodies
# like `H_replay_prior_trigger`, `H_crop_overlap_probe`,
# `H_crop_relation_plan`, `P_complete_upper_crop2` — none carry an R-id
# or T-token but every one references a temporal artefact (prior, trigger,
# replay, relation, overlap, plan, complete) and was meaningful relative
# to the run summary. Accept these as anchored-by-history.
_HISTORY_KEYWORDS = re.compile(
    r"(replay|trigger|prior|relation|overlap|history|recall|"
    r"plan|complete|continue|resume|recover)"
)

# Non-trivial markers (V-H4): cross-region relation, sprite-internal
# geometry, negation, sequence. A TIER-B chid is "non-trivial" when
# either the body contains one of these substrings OR a direction suffix
# is present.
_NON_TRIVIAL_BODY = re.compile(
    r"(shared|align|sweep|overlap|alignment|relation|sector|"
    r"static|inert|nonmarker|compass|crop|corner|edge|probe|"
    r"fresh|neighbor|toggle|replay|trigger|prior|upper|lower|"
    r"plan|complete|recover)"
)


def _load_forbidden_vocab() -> Tuple[str, ...]:
    """Single source of truth — import from prompts.py at runtime.

    We tolerate import failure (e.g. when running pure unit tests outside
    the agent package) by falling back to the historical literal list.
    """
    try:  # pragma: no cover - import guard
        from agents.templates.agentica_v57.prompts import FORBIDDEN_VOCAB

        return tuple(FORBIDDEN_VOCAB)
    except Exception:
        return (
            "per_neighbor_target",
            "target_color",
            "needs_toggle",
            "marker_progress",
            "joint_neighbors",
            "expected_neighbor_colors",
            "win_state",
            "goal_state",
        )


def is_tier_a_predicate_id(chid: str) -> bool:
    """TIER-A predicate_ids look like 'P01_<template>:T<turn>:R<region>'.

    Accepts 2-3 digit template numbers to future-proof against P100+.
    """
    return bool(re.match(r"^P\d{2,3}_[a-z0-9_]+:T\d+:R\d+$", chid or ""))


def parse_invented_chid(chid: str) -> dict | None:
    """Return parsed fields if grammar matches, else None.

    Fields:
      prefix        : 'H' (hypothesis) or 'P' (proposal/sequence)
      body          : the full body (used for template_id)
      region_ids    : list of R<id> tokens found in body
      direction     : compass direction at end of body if present
      history_anchor: True iff body contains a T<digits> token
      template_id   : prefix + body with R<id>->Rx (for cross-run aggregation)
    """
    m = INVENTED_CHID_RE.match(chid or "")
    if not m:
        return None
    body = m.group("body")
    region_ids = [f"R{n}" for n in R_ID_RE.findall(body)]
    direction = None
    dm = DIRECTION_TAIL_RE.search(body)
    if dm:
        direction = dm.group(1)
    history_anchor = bool(HISTORY_ANCHOR_RE.search(body))
    template_body = R_ID_RE.sub("Rx", body)
    return {
        "prefix": m.group("prefix"),
        "body": body,
        "region_ids": region_ids,
        "direction": direction,
        "history_anchor": history_anchor,
        "template_id": f"{m.group('prefix')}_{template_body}",
    }


def validate_invented_chid(
    chid: str,
    visible_region_ids: Iterable[str],
    *,
    forbidden_vocab: Iterable[str] | None = None,
) -> Tuple[bool, str]:
    """Return (ok, reason). reason is 'ok' on success, else short tag."""
    parsed = parse_invented_chid(chid)
    if parsed is None:
        return (False, "ungrammatical")
    body_lower = parsed["body"].lower()
    vocab = tuple(forbidden_vocab) if forbidden_vocab is not None else _load_forbidden_vocab()
    for term in vocab:
        if term and term.lower() in body_lower:
            return (False, f"leak_vocab:{term}")
    has_history_keyword = bool(_HISTORY_KEYWORDS.search(body_lower))
    if (
        not parsed["region_ids"]
        and not parsed["history_anchor"]
        and not has_history_keyword
    ):
        return (False, "no_anchor")
    visible_set = {str(r) for r in visible_region_ids if r}
    if parsed["region_ids"]:
        # At least ONE referenced region must be currently visible.
        if not any(rid in visible_set for rid in parsed["region_ids"]):
            return (False, "region_not_visible")
    return (True, "ok")


def is_non_trivial_invention(chid: str) -> bool:
    """V-H4: invention contains direction OR rich rationale token."""
    parsed = parse_invented_chid(chid)
    if parsed is None:
        return False
    if parsed["direction"]:
        return True
    if _NON_TRIVIAL_BODY.search(parsed["body"]):
        return True
    return False


def template_id(chid: str) -> str | None:
    """Stable cross-run identifier (R<id> stripped to Rx)."""
    parsed = parse_invented_chid(chid)
    return parsed["template_id"] if parsed else None


def cited_region_id(chid: str) -> str | None:
    """Return the FIRST R<id> mentioned in a TIER-B chid, else None.

    Used to cross-check that the click coord lies inside the cited
    region's bbox (plan v591 §4.4 requirement; review issue W7).
    """
    parsed = parse_invented_chid(chid)
    if parsed and parsed["region_ids"]:
        return parsed["region_ids"][0]
    return None


_NEIGHBOR_RE = re.compile(r"(?:^|_)neighbor", re.IGNORECASE)


def chid_rationale_intent(chid: str) -> dict:
    """Extract rationale-coord intent from a TIER-B chid.

    Returns:
      {
        "neighbor_of": True if rationale says "neighbor of cited
                       region" → coord should land in
                       cited.neighbors_3x3 not in cited.bbox itself,
        "direction": "N|S|E|W|NE|NW|SE|SW" or None — coord should
                     be the corner / edge midpoint of cited.bbox,
        "corner": True if direction is 2-letter (corner pixel),
                  False if 1-letter (edge midpoint),
      }

    Used by agent.py runtime to redirect M1's coord so the click
    actually tests the rationale (review issue C-2 round-2).
    """
    parsed = parse_invented_chid(chid)
    if parsed is None:
        return {"neighbor_of": False, "direction": None, "corner": False}
    body_lower = parsed["body"].lower()
    is_neighbor = bool(_NEIGHBOR_RE.search(body_lower))
    direction = parsed["direction"]
    return {
        "neighbor_of": is_neighbor,
        "direction": direction,
        "corner": direction is not None and len(direction) == 2,
    }

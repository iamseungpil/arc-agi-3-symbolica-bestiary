"""Goal Board: schemas, regexes, scoring, validation, falsifier, parse_or_retry, GoalBoard.

Single co-located module per plan v14 §4.0. Holds all hypothesis-card / chosen-action /
abstract-skill dataclasses plus the precision_score, validate_card, evaluate_falsifier
helpers, the parse_or_retry async wrapper, the _parse_action_str helper, and the
defensive _NullTrace shim that SemanticPackets uses when no real trace store exists.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

# ---------- Regex constants (plan v13 §4.4 inline) ----------

CONNECTOR = r"(?:->|→|to|becomes?|changes?\s+to|transitions?\s+to|=>)"
COLOR_NAMES = (
    r"(?:black|blue|red|green|yellow|teal|orange|purple|gray|grey|"
    r"brown|pink|cyan|magenta|white|dark|light)"
)

_R_ID_RE = re.compile(r"r\d+", re.IGNORECASE)
BIND_DIGIT_RE = re.compile(rf"r\d+[^.]{{0,80}}?\b\d{{1,2}}\s*{CONNECTOR}\s*\d{{1,2}}", re.IGNORECASE)
BIND_NAME_RE = re.compile(
    rf"r\d+[^.]{{0,200}}?\b{COLOR_NAMES}\b\s*{CONNECTOR}\s*\b{COLOR_NAMES}\b",
    re.IGNORECASE,
)
ACTION_COORD_RE = re.compile(r"action\d+\s*\(\s*\d+\s*,\s*\d+\s*\)", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json|python)?\s*|\s*```$", re.MULTILINE)
_ACTION_PARSE_RE = re.compile(
    r"(ACTION\d+|RESET)\s*(?:\(\s*(\d+)\s*,\s*(\d+)\s*\))?", re.IGNORECASE
)


# ---------- v20 helpers: diversity + info-density ----------


_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")
_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "of", "to", "by", "for", "with",
    "is", "be", "are", "was", "were", "this", "that", "these", "those",
    "where", "when", "must", "should", "would", "could", "will", "can",
    "and", "or", "but", "as", "from", "into", "than", "then",
})


def _word_set(text: str) -> set[str]:
    """Lowercase + strip non-alphanumerics + drop stopwords + drop short tokens."""
    s = _NON_WORD_RE.sub(" ", str(text).lower())
    return {w for w in s.split() if len(w) > 2 and w not in _STOP_WORDS}


def jaccard_words(a: str, b: str) -> float:
    """Token-jaccard over predicate strings. Used to detect near-paraphrases."""
    A, B = _word_set(a), _word_set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


# Coarse archetype-alignment heuristic: maps a card to one of the 5 schema ids
# by mention pattern. Used so non-archetype cards (mechanism / target_state)
# can be attributed to an archetype umbrella for stagnation accounting.
_ARCHETYPE_TERMS = {
    "indicator_match": ("indicator",),
    "constraint_satisfaction": ("constraint", "mask"),
    "lights_out": ("lights", "uniform", "all-off", "all-on"),
    "color_cycle": ("cycle", "modular"),
    "marker_constraint": ("marker", "fixed sprite"),
}


def archetype_alignment(card: "HypothesisCard") -> str | None:
    """Best-guess archetype_id for a card (explicit on archetype cards;
    inferred from predicate+recipe text on others). Returns None if no signal."""
    sig = card.expected_signature or {}
    explicit = sig.get("archetype_id")
    if explicit:
        return str(explicit)
    text = (str(card.predicate or "") + " " + str(card.abstract_recipe or "")).lower()
    for aid, terms in _ARCHETYPE_TERMS.items():
        if any(t in text for t in terms):
            return aid
    return None


def skill_density_score(skill: "AbstractSkill") -> int:
    """Info-density: longer + multi-region anchor = higher. Used as LRU key."""
    if skill is None:
        return 0
    base = len(skill.causal_mapping or "") + len(skill.applies_when or "")
    base += sum(len(s or "") for s in (skill.schema_steps or []))
    anchor_rids = len(set(_R_ID_RE.findall(str(skill.concrete_anchor or ""))))
    return base + 5 * anchor_rids


# ---------- Schemas ----------


@dataclass
class HypothesisCard:
    id: str
    predicate: str
    abstract_recipe: str = ""
    expected_signature: dict = field(default_factory=dict)
    prior_plausibility: Literal["low", "med", "high"] = "low"
    evidence_quote: str = ""
    precision_score: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisCard":
        valid_keys = {
            "id",
            "predicate",
            "abstract_recipe",
            "expected_signature",
            "prior_plausibility",
            "evidence_quote",
        }
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        # Defensive coercion
        kwargs.setdefault("id", str(d.get("id", "")).strip() or "C0")
        kwargs.setdefault("predicate", str(d.get("predicate", "")).strip())
        kwargs["expected_signature"] = (
            dict(kwargs.get("expected_signature") or {})
            if isinstance(kwargs.get("expected_signature"), dict)
            else {}
        )
        plaus = str(kwargs.get("prior_plausibility", "low")).lower().strip()
        if plaus not in {"low", "med", "high"}:
            plaus = "low"
        kwargs["prior_plausibility"] = plaus
        return cls(**kwargs)


@dataclass
class ChosenAction:
    card_id: str
    action_sequence: list[str] = field(default_factory=list)
    expected_diff_signature: dict = field(default_factory=dict)
    falsification_criterion: str = ""
    # v35 framework: explicit binding from M2 to a skill. "S1".."S5" or
    # "none". Forces M2 to consult active_skills and creates audit trail
    # for skill usage analysis. M3 sees per-skill usage histogram via
    # choice_history → identifies dead skills → emits diverse skill types.
    skill_anchor: str = "none"
    # v40: per-step predicted diffs for multi-step plans. Length must
    # equal action_sequence length when action_sequence > 1; orchestrator
    # aborts the plan early if observed step diff mismatches.
    expected_step_diffs: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ChosenAction":
        # Defensive: M2 may emit a JSON list/scalar instead of an object;
        # mirror AbstractSkill.from_dict so callers (parse_or_retry) get
        # a clean None rather than an AttributeError on .items().
        if not isinstance(d, dict):
            return None  # type: ignore[return-value]
        valid_keys = {
            "card_id",
            "action_sequence",
            "expected_diff_signature",
            "falsification_criterion",
            "skill_anchor",
            "expected_step_diffs",
        }
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        seq = kwargs.get("action_sequence", [])
        if not isinstance(seq, list):
            seq = [str(seq)] if seq else []
        kwargs["action_sequence"] = [str(s).strip() for s in seq if str(s).strip()]
        kwargs["card_id"] = str(kwargs.get("card_id", "")).strip()
        if not kwargs.get("expected_diff_signature") or not isinstance(
            kwargs.get("expected_diff_signature"), dict
        ):
            kwargs["expected_diff_signature"] = {}
        kwargs["falsification_criterion"] = str(
            kwargs.get("falsification_criterion", "")
        ).strip()
        kwargs["skill_anchor"] = str(kwargs.get("skill_anchor", "none")).strip() or "none"
        # v40: normalise expected_step_diffs into list of dicts.
        steps = kwargs.get("expected_step_diffs", [])
        if not isinstance(steps, list):
            steps = []
        kwargs["expected_step_diffs"] = [s if isinstance(s, dict) else {} for s in steps]
        if not kwargs["card_id"]:
            return None  # type: ignore[return-value]
        return cls(**kwargs)


@dataclass
class LessonCard:
    """v21: per-turn experience reflection emitted by M4.

    Sits between board.update (raw verdict) and M3 (skill compression).
    Each field is a single sentence (≤200 chars). At least 4 of 5 fields
    must be non-empty for the lesson to be accepted into lesson_log.
    """
    what_happened: str = ""
    delta: str = ""
    lesson: str = ""
    retry_modification: str = ""
    skill_seed: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Optional["LessonCard"]:
        if not isinstance(d, dict):
            return None
        valid = {"what_happened", "delta", "lesson", "retry_modification", "skill_seed"}
        kwargs = {k: str(d.get(k, "") or "").strip()[:200] for k in valid}
        non_empty = sum(1 for v in kwargs.values() if v)
        if non_empty < 4:
            return None
        return cls(**kwargs)


def parse_lesson_card(payload) -> Optional[LessonCard]:
    """M4 parser: requires a JSON object with at least 4 of 5 fields populated."""
    if isinstance(payload, list) and payload:
        # defensive: M4 might wrap in a list
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    return LessonCard.from_dict(payload)


@dataclass
class AbstractSkill:
    goal_phrase: str
    causal_mapping: str = ""
    concrete_anchor: str = ""
    schema_steps: list[str] = field(default_factory=list)
    applies_when: str = ""
    # v36: skill type taxonomy + per-emission reflexion
    skill_type: str = "mechanic"  # mechanic|strategic|discovery_method|meta_cognitive
    novelty_diff: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Optional["AbstractSkill"]:
        if not isinstance(d, dict):
            return None
        if d.get("skip"):
            return None
        valid_keys = {
            "goal_phrase",
            "causal_mapping",
            "concrete_anchor",
            "schema_steps",
            "applies_when",
            "skill_type",
            "novelty_diff",
        }
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        gp = str(kwargs.get("goal_phrase", "")).strip()
        if not gp:
            return None
        kwargs["goal_phrase"] = gp
        kwargs["causal_mapping"] = str(kwargs.get("causal_mapping", "")).strip()
        kwargs["concrete_anchor"] = str(kwargs.get("concrete_anchor", "")).strip()
        steps = kwargs.get("schema_steps", [])
        if not isinstance(steps, list):
            steps = []
        kwargs["schema_steps"] = [str(s).strip() for s in steps if str(s).strip()][:5]
        kwargs["applies_when"] = str(kwargs.get("applies_when", "")).strip()
        st = str(kwargs.get("skill_type", "mechanic") or "mechanic").strip()
        if st not in {"mechanic","strategic","discovery_method","meta_cognitive"}:
            st = "mechanic"
        kwargs["skill_type"] = st
        kwargs["novelty_diff"] = str(kwargs.get("novelty_diff", "") or "").strip()
        return cls(**kwargs)


# ---------- Scoring + validation ----------


def precision_score(
    card: HypothesisCard,
    current_frame_region_ids: set[str],
    falsified_predicates: list[str] | None = None,
    fresh_facts: set[tuple] | None = None,
) -> float:
    """Plan v13 §4.4 precision score."""
    text = card.predicate.lower()
    region_hits = [
        m for m in _R_ID_RE.findall(text) if m.upper() in current_frame_region_ids
    ]
    bind_digit = BIND_DIGIT_RE.findall(text)
    bind_name = BIND_NAME_RE.findall(text)
    coord_hits = ACTION_COORD_RE.findall(text)
    sig = card.expected_signature or {}
    sig_region = sig.get("region_id")
    # Goal-priority boost: cards that predict level_delta != 0 get a large
    # additive bonus so M2 doesn't drown them under high-grounded mechanism
    # cards. Once mechanism is known, goal cards are the only path to
    # level progression — we explicitly bias M2 to test them.
    level_predict = sig.get("level_delta") not in (None, 0)
    grounded = (
        min(len(region_hits), 3)
        + 5 * len(bind_digit)
        + 3 * len(bind_name)
        + 3 * len(coord_hits)
        + (3 if (isinstance(sig_region, str) and sig_region in current_frame_region_ids) else 0)
        + (2 if sig.get("dominant_transition") else 0)
        + (8 if level_predict else 0)  # GOAL boost: was 2, raised to 8
    )
    # v20: multi-region bonus — predicate naming >=2 distinct R-ids that
    # appear in the current frame gets +4 (rewards cross-region cards that
    # imply M2 will probe a new region). C2.2: only count R-ids in the same
    # sentence as a transition, to avoid spurious R-id spamming.
    multi_region_bonus = 0
    distinct_visible_rids = {
        m.upper() for m in _R_ID_RE.findall(text) if m.upper() in current_frame_region_ids
    }
    if len(distinct_visible_rids) >= 2 and (
        BIND_DIGIT_RE.search(text) or BIND_NAME_RE.search(text)
    ):
        multi_region_bonus = 4

    # v20: fresh-discovery bonus — card cites a (region/transition/color)
    # tuple that landed in discovered_facts within the last 5 turns.
    fresh_bonus = 0
    if fresh_facts:
        for kind, val in fresh_facts:
            if kind == "region" and str(val).lower() in text:
                fresh_bonus = 3
                break
            if kind == "transition" and isinstance(val, tuple):
                f, t = val
                if (
                    re.search(rf"\b{f}\s*(?:->|to|becomes?)\s*{t}\b", text)
                    or re.search(rf"\b{t}\s*(?:->|to|becomes?)\s*{f}\b", text)
                ):
                    fresh_bonus = 3
                    break

    # v20: soft jaccard penalty (-3 per high-overlap falsified predicate
    # in the 0.5-0.7 band). Hard-filter at >0.7 happens upstream in agent.py.
    jaccard_penalty = 0
    if falsified_predicates:
        for fp in falsified_predicates:
            j = jaccard_words(card.predicate, fp)
            if 0.5 <= j <= 0.7:
                jaccard_penalty -= 3
                break  # cap penalty at -3 regardless of #matches

    plausibility_w = {"low": 1, "med": 2, "high": 3}.get(card.prior_plausibility, 1)
    return float((grounded + multi_region_bonus + fresh_bonus) * plausibility_w + jaccard_penalty)


def validate_card(card: HypothesisCard, current_frame_region_ids: set[str]) -> bool:
    """A card is valid iff predicate names a real R-id AND a transition AND
    expected_signature.region_id is a visible R-id (v41 strict).

    Archetype cards (sig.is_archetype=true) still need region_id (v41) so
    the click that tests the archetype claim has a concrete target — but
    they are exempt from the transition requirement.

    v41: region_id is REQUIRED. Without it, M2 cannot target a coord and
    falls back to body-bbox-center which lands _outside_ in 50%+ of L4
    cases (cycle33 audit data).
    """
    sig = card.expected_signature or {}
    sig_region = str(sig.get("region_id", "") or "").strip()
    # v41: region_id strict check applies to every card.
    if not sig_region or sig_region not in current_frame_region_ids:
        return False
    if sig.get("is_archetype"):
        return bool(str(sig.get("archetype_id", "")).strip())
    text = card.predicate.lower()
    has_r = any(m.upper() in current_frame_region_ids for m in _R_ID_RE.findall(text))
    has_transition = bool(BIND_DIGIT_RE.search(text) or BIND_NAME_RE.search(text))
    return has_r and has_transition


def evaluate_falsifier(
    card: HypothesisCard,
    choice: ChosenAction,
    observed: dict,
    archetype_stagnation_window: int = 0,
    archetype_falsified_count: int = 0,
    archetype_confirm_count: int = 0,
    marker_progress_delta: int | None = None,
) -> Literal["confirm", "falsify", "inconclusive"]:
    """Plan v11 §4.5 4-axis verdict; v16+ adds is_target_state escape.

    is_target_state=True means the card describes a TARGET CONFIGURATION
    (win-state hypothesis), not a per-click prediction. For these cards we
    DO NOT use the level_delta axis to falsify on a single click producing
    level_delta=0 — that would kill long-horizon hypotheses. We only confirm
    them when level actually advances. They stay in 'inconclusive' state
    until many actions accumulate, then either confirm (level rose) or stay
    open (architecture stops them via skill-driven re-evaluation in M3).
    """
    sig = card.expected_signature or {}
    expected_min_cells = sig.get("min_cell_count")
    expected_transition = sig.get("dominant_transition")
    expected_region = sig.get("region_id")
    expected_level_delta = sig.get("level_delta")
    is_target_state = bool(sig.get("is_target_state", False))
    is_archetype = bool(sig.get("is_archetype", False))

    obs_level_delta_int = int(observed.get("level_delta", 0) or 0)
    if is_archetype:
        # Archetype cards are meta-hypotheses about the puzzle TYPE.
        # Confirm only when level actually advances; otherwise inconclusive,
        # UNLESS the H3 stagnation guard fires:
        #   (a) >=8 turns of level_delta=0 attributed to this archetype, AND
        #   (b) >=10 cards under this archetype umbrella already falsified, AND
        #   (c) <2 confirms under this archetype.
        # Then we falsify the archetype itself so M1 must pivot to a new one.
        if obs_level_delta_int > 0:
            return "confirm"
        if (
            archetype_stagnation_window >= 8
            and archetype_falsified_count >= 10
            and archetype_confirm_count < 2
        ):
            return "falsify"
        return "inconclusive"

    obs_changed = int(observed.get("changed_cells", 0) or 0)
    obs_dom = observed.get("dominant_transition") or {}
    obs_region = observed.get("primary_region_id", "")
    obs_level_delta = int(observed.get("level_delta", 0) or 0)

    # Axis 1: cell count
    cell_axis = None
    if expected_min_cells is not None:
        try:
            cell_axis = obs_changed >= int(expected_min_cells)
        except (TypeError, ValueError):
            cell_axis = None

    # Axis 2: dominant transition
    def _safe_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    trans_axis = None
    if expected_transition:
        if isinstance(expected_transition, dict):
            ef = _safe_int(expected_transition.get("from"))
            et = _safe_int(expected_transition.get("to"))
            if ef is not None and et is not None and isinstance(obs_dom, dict):
                of = _safe_int(obs_dom.get("from"))
                ot = _safe_int(obs_dom.get("to"))
                if of is not None and ot is not None:
                    trans_axis = (of == ef and ot == et)
        elif isinstance(expected_transition, str) and expected_transition.strip():
            trans_axis = expected_transition.strip().lower() in str(obs_dom).lower()

    # Axis 3: region
    region_axis = None
    if expected_region:
        region_axis = str(obs_region) == str(expected_region)

    # Axis 4: level_delta — SKIPPED entirely for target-state cards.
    # Target-state cards predict the END configuration, not the next click.
    # Falsifying them on level_delta=0 single click is a category error.
    level_axis = None
    if expected_level_delta is not None and not is_target_state:
        try:
            level_axis = obs_level_delta == int(expected_level_delta)
        except (TypeError, ValueError):
            level_axis = None

    # Target-state confirm path: if level actually rose, confirm regardless
    # of other axes (the win-state hypothesis was on the right track).
    if is_target_state and obs_level_delta > 0:
        return "confirm"

    # v39: 5th axis — actual marker-satisfaction progress (the win condition).
    # marker_progress_delta = total_unsatisfied(before) - total_unsatisfied(after).
    # Positive = real progress toward win, negative = regression. This signal is
    # independent of the card's prediction axes; it reflects whether the click
    # actually moved the puzzle closer to win. Crucial for L4+ where joint
    # constraints make signature-based axes inconclusive even when progress
    # genuinely happened.
    progress_axis = None
    if marker_progress_delta is not None:
        try:
            mpd = int(marker_progress_delta)
            if mpd >= 1:
                progress_axis = True   # made real progress
            elif mpd <= -1:
                progress_axis = False  # regressed
            # mpd == 0 → no info; leave None
        except (TypeError, ValueError):
            progress_axis = None

    axes = [a for a in (cell_axis, trans_axis, region_axis, level_axis, progress_axis) if a is not None]
    if not axes:
        return "inconclusive"
    if all(axes):
        return "confirm"
    if any(a is False for a in axes):
        return "falsify"
    return "inconclusive"


# ---------- Parsing helpers ----------


def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s.strip()).strip()


def _parse_action_str(action_str: str) -> tuple[str, int, int]:
    """'ACTION6(38,46)' -> ('ACTION6', 38, 46); 'ACTION1' -> ('ACTION1', 0, 0)."""
    if not isinstance(action_str, str):
        return ("RESET", 0, 0)
    m = _ACTION_PARSE_RE.search(action_str.strip())
    if not m:
        return ("RESET", 0, 0)
    name = m.group(1).upper()
    x = int(m.group(2)) if m.group(2) is not None else 0
    y = int(m.group(3)) if m.group(3) is not None else 0
    return (name, x, y)


def parse_card_list(payload) -> Optional[list[HypothesisCard]]:
    """M1 parser: requires a non-empty list of card dicts."""
    if not isinstance(payload, list):
        # Allow {"cards": [...]} envelope as a defensive convenience
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            payload = payload["cards"]
        else:
            return None
    cards = []
    for d in payload:
        if not isinstance(d, dict):
            continue
        try:
            cards.append(HypothesisCard.from_dict(d))
        except Exception:  # noqa: BLE001
            continue
    return cards if cards else None


async def parse_or_retry(agent, prompt: str, parser_fn, retries: int = 1):
    """Parse LLM output as JSON, retry once with corrective re-prompt on parse failure.

    `agent` is a spawn_agent handle; `agent.call(str, task)` is the runtime contract.
    """
    task = prompt
    for attempt in range(retries + 1):
        try:
            raw = await agent.call(str, task)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse_or_retry: agent.call raised %s", exc)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(_strip_code_fences(str(raw)))
            result = parser_fn(payload)
            if result is not None:
                return result
            raise ValueError("parser_fn rejected payload")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
            if attempt < retries:
                task = (
                    f"Your previous output failed to parse: "
                    f"{type(exc).__name__}: {str(exc)[:200]}. "
                    f"Re-emit ONLY a single JSON object/list. "
                    f"No prose, no markdown, no code fences."
                )
                continue
            logger.info("parse_or_retry: giving up after %d attempts", attempt + 1)
            return None
    return None


async def parse_with_critic(agent, prompt: str, parser_fn, num_passes: int = 1, retries: int = 1):
    """v37 inner-loop: each module does draft → self-critic → revise inside
    one logical call. Falls back to draft if critic call fails or rejects.

    Pipeline: parse_or_retry (draft) → for N passes: emit critic-revise prompt
    with draft inlined, parse → if valid revision, replace draft. Final draft
    is returned.

    This adds 1 extra LLM call per module per turn (M1/M2/M3) but forces
    each module to actually iterate before passing to the next module."""
    draft = await parse_or_retry(agent, prompt, parser_fn, retries=retries)
    if draft is None:
        return None

    for pass_idx in range(num_passes):
        try:
            if hasattr(draft, "__dataclass_fields__"):
                draft_payload = asdict(draft)
            elif isinstance(draft, list):
                draft_payload = [
                    asdict(c) if hasattr(c, "__dataclass_fields__") else c
                    for c in draft
                ]
            else:
                draft_payload = draft
            draft_json = json.dumps(draft_payload, default=str)[:6000]
        except (TypeError, ValueError):
            return draft

        critic_task = (
            f"DRAFT (your previous output, pass {pass_idx + 1}):\n"
            f"{draft_json}\n\n"
            f"CRITIQUE INSTRUCTIONS — review your draft against these:\n"
            f"  1. Are required schema fields populated and meaningful?\n"
            f"  2. Does the draft contradict any INPUT data above?\n"
            f"  3. Is the draft generic / a duplicate of an earlier output?\n"
            f"     If so, sharpen it (more concrete IDs, less paraphrase).\n"
            f"  4. Reflexion fields (prior_reflection / novelty_diff / "
            f"     skill_anchor) — are they specific and grounded?\n"
            f"Now emit your REVISED final JSON using the SAME schema as the "
            f"draft. If draft is already optimal, repeat it verbatim. Output "
            f"ONLY the JSON object/list, no prose."
        )
        try:
            raw = await agent.call(str, critic_task)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse_with_critic: critic-revise raised %s", exc)
            return draft
        if raw is None:
            return draft
        try:
            payload = json.loads(_strip_code_fences(str(raw)))
            revised = parser_fn(payload)
            if revised is not None:
                draft = revised
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    return draft


# ---------- _NullTrace shim (R8-#S1) ----------


class _NullTrace:
    """Defensive null-object: SemanticPackets uses .list(kind=...); other shims pass through."""

    def list(self, **kwargs):  # SemanticPackets calls this
        return []

    def event(self, *args, **kwargs) -> None:
        return None

    def plan(self, *args, **kwargs) -> None:
        return None

    def record_action(self, *args, **kwargs) -> None:
        return None

    def __getattr__(self, name):
        # Catch-all for any other unexpected callsites
        return lambda *args, **kwargs: None


# ---------- GoalBoard ----------


_MAX_OBS_LOG = 64
_MAX_SKILLS = 5  # LRU cap


class GoalBoard:
    """Active hypothesis cards + falsified cards + observation log + LRU skills."""

    def __init__(self, simple_workdir: Path | str) -> None:
        self.simple_workdir = Path(simple_workdir)
        self.simple_workdir.mkdir(parents=True, exist_ok=True)
        self.cards: list[HypothesisCard] = []
        self.falsified_cards: list[dict] = []
        self.observation_log: list[dict] = []
        self.skills: "OrderedDict[str, AbstractSkill]" = OrderedDict()
        self.tried_coords: set[tuple] = set()
        self.choice_history: list[dict] = []
        # Per-clicked-region response stats — lets M1 distinguish marker
        # regions (clicks=N, responses=0) from target regions (responses>0).
        # Key: region_id (the region that the click coord fell into); Value:
        # {"clicks": int, "responses": int}. Updated on every board.update().
        self.region_click_stats: dict[str, dict] = {}
        # v20: discovered_facts maps (kind, value) -> last_turn_index. Used
        # by M1 prompt to highlight which signals are NEW vs already-known.
        # kind in {"region", "transition", "color"}.
        self.discovered_facts: dict[tuple, int] = {}
        # v20: per-archetype stagnation counter — consecutive turns since
        # this archetype became dominant during which level_delta stayed 0.
        # When an archetype hits the H3 stagnation threshold, evaluate_falsifier
        # demotes any future card under it to "falsify".
        self.archetype_stagnation: dict[str, int] = {}
        self.stagnation_window: int = 0  # consecutive turns of level_delta=0
        self.turn_index: int = 0  # incremented per board.update call
        # v21: per-turn LessonCards emitted by M4. Capped at 30 entries
        # (LRU). Feeds back into M1 (recent 5) and M3 (full log).
        self.lesson_log: list[LessonCard] = []
        # v32 Fix D: cross-level confirmed predicates. Survives level
        # transitions (Fix C used to clear everything). Used as positive
        # transfer signal — M1 sees "this kind of hypothesis worked at L0,
        # try analogous one at L4". Cap at 20.
        self.cross_level_confirmed: list[dict] = []
        self.cross_level_confirmed_total: int = 0
        # v41: reflexion buffer. A short corrective sentence emitted by the
        # Reflexion module when the agent stagnates. Routed directly into
        # the M2 prompt as prepended text, decays on level_rise OR after
        # reflexion_decay_turns turns.
        self.reflexion_buffer: str = ""
        # v42 (Q3): per-module memory of self. Each module sees ITS OWN
        # previous turn output + the actual outcome that followed, so M1
        # can refine cards based on which of its own cards landed, and M2
        # can refine plans based on whether its own predictions held.
        self.last_m1_cards: list[dict] = []   # last batch of M1 cards (id+predicate+region)
        self.last_m2_choice: dict | None = None  # last M2 ChosenAction (card_id, plan, expected_step_diffs)
        self.last_outcome: dict | None = None  # observed result of last M2 plan: per-step regions, transitions, verdicts

    def record_m1_emit(self, cards: list) -> None:
        """v42: snapshot M1's current emission for next turn's REFLECT.

        Stores a compact summary (id, predicate, region_id) — full schema not
        needed by M1's self-reflection.
        """
        self.last_m1_cards = [
            {
                "id": c.id,
                "predicate": (c.predicate or "")[:160],
                "region_id": (c.expected_signature or {}).get("region_id"),
                "expected_transition": (c.expected_signature or {}).get("dominant_transition"),
                "is_archetype": bool((c.expected_signature or {}).get("is_archetype")),
            }
            for c in cards
        ][:6]

    def record_turn_outcome(
        self,
        choice,
        plan_executed: list,
        per_step_observed: list[dict],
        per_step_verdict: list[str],
        plan_aborted: bool,
    ) -> None:
        """v42: snapshot M2's plan + actual outcome at end of step-loop.

        plan_executed: action_str list actually submitted (≤ original plan len)
        per_step_observed: list of observed dicts per step
        per_step_verdict: list of verdicts per step
        """
        if choice is None:
            return
        self.last_m2_choice = {
            "card_id": choice.card_id,
            "skill_anchor": getattr(choice, "skill_anchor", "none"),
            "plan_emitted": list(choice.action_sequence),
            "expected_step_diffs": list(choice.expected_step_diffs or []),
        }
        self.last_outcome = {
            "plan_executed": plan_executed,
            "plan_aborted": plan_aborted,
            "steps": [
                {
                    "action": plan_executed[i] if i < len(plan_executed) else None,
                    "observed_region": (per_step_observed[i] or {}).get("primary_region_id"),
                    "observed_transition": (per_step_observed[i] or {}).get("dominant_transition"),
                    "changed_cells": (per_step_observed[i] or {}).get("changed_cells"),
                    "level_delta": (per_step_observed[i] or {}).get("level_delta", 0),
                    "verdict": per_step_verdict[i] if i < len(per_step_verdict) else None,
                }
                for i in range(len(per_step_observed))
            ],
        }
        self.reflexion_set_at_turn: int = -1
        self.reflexion_decay_turns: int = 8

    def should_reflect(self, falsify_window: int = 4) -> bool:
        """v41: trigger condition for Reflexion call.

        True when ANY of:
          * stagnation_window >= 6 AND no active reflexion buffer
          * last `falsify_window` verdicts are all in {falsify, inconclusive}
            (no confirm)  AND no active buffer
        Skip if a buffer is already set (avoid thrashing).
        """
        if self.reflexion_buffer:
            return False
        if self.stagnation_window >= 6:
            return True
        if len(self.observation_log) >= falsify_window:
            tail = self.observation_log[-falsify_window:]
            if all(e.get("verdict") in {"falsify", "inconclusive"} for e in tail):
                return True
        return False

    def set_reflection(self, text: str) -> None:
        """v41: store corrective text from Reflexion module."""
        if not text or not isinstance(text, str):
            return
        cleaned = text.strip()[:220]
        if not cleaned:
            return
        self.reflexion_buffer = cleaned
        self.reflexion_set_at_turn = self.turn_index

    def clear_reflection_if_stale(self) -> None:
        """v41: decay reflexion buffer when stale (level rise resets via
        update(), this guards age-based decay)."""
        if not self.reflexion_buffer:
            return
        if self.turn_index - self.reflexion_set_at_turn >= self.reflexion_decay_turns:
            self.reflexion_buffer = ""
            self.reflexion_set_at_turn = -1

    def update(
        self,
        cards: list[HypothesisCard],
        choice: ChosenAction,
        observed: dict,
        verdict: str,
        clicked_region: str | None = None,
    ) -> None:
        # Track observation
        entry = {
            "card_id": choice.card_id,
            "predicate": next(
                (c.predicate for c in cards if c.id == choice.card_id), ""
            ),
            "verdict": verdict,
            "action": observed.get("action"),
            "changed_cells": observed.get("changed_cells"),
            "dominant_transition": observed.get("dominant_transition"),
            "primary_region_id": observed.get("primary_region_id"),
            "level_delta": observed.get("level_delta"),
        }
        self.observation_log.append(entry)
        if len(self.observation_log) > _MAX_OBS_LOG:
            self.observation_log = self.observation_log[-_MAX_OBS_LOG:]

        # Marker-vs-target labelling: count clicks that landed in each region
        # and how many of those produced any cell change. A region with many
        # clicks and zero changed_cells is a constraint marker; M1 sees this
        # via visible_regions[i].click_response on the next turn.
        if clicked_region:
            stats = self.region_click_stats.setdefault(
                clicked_region, {"clicks": 0, "responses": 0}
            )
            stats["clicks"] += 1
            if int(observed.get("changed_cells", 0) or 0) > 0:
                stats["responses"] += 1

        # v20: discovered_facts ledger. Add (kind, value) -> turn_index for
        # every region/transition/color tuple seen this turn.
        self.turn_index += 1
        if clicked_region:
            self.discovered_facts[("region", clicked_region)] = self.turn_index
        prim = observed.get("primary_region_id")
        if prim and prim != "_outside_":
            self.discovered_facts[("region", str(prim))] = self.turn_index
        dom = observed.get("dominant_transition")
        if isinstance(dom, dict):
            try:
                f, t = int(dom.get("from")), int(dom.get("to"))
                self.discovered_facts[("transition", (f, t))] = self.turn_index
                self.discovered_facts[("color", f)] = self.turn_index
                self.discovered_facts[("color", t)] = self.turn_index
            except (TypeError, ValueError):
                pass

        # v20: stagnation tracking. stagnation_window counts consecutive turns
        # with level_delta=0. archetype_stagnation tracks per-archetype tenure
        # of the dominant card's archetype id during the stagnation window.
        level_delta = int(observed.get("level_delta", 0) or 0)
        if level_delta > 0:
            self.stagnation_window = 0
            self.archetype_stagnation.clear()
            # v43: instead of clearing reflexion on level rise, INJECT a
            # discovery-mode corrective so the next 5 turns explore new
            # sprites BEFORE applying L0-L3 mechanism rules. Without this,
            # cross_level_confirmed dominates and discovery never fires
            # at L4 (where NTi is structurally new) or L5 (where ZkU is
            # structurally new). The buffer decays after 8 turns or on
            # next level rise.
            self.reflexion_buffer = (
                "Level just rose to a new level. Spend the next 5 turns "
                "in DISCOVERY MODE: M1 must emit >=3 observation cards "
                "exploring atypical sprite behavior in this new frame; "
                "M2 must click a coord that could plausibly be a NEW "
                "sprite type (not just the largest visible region). If "
                "any click yields changed_cells > 50, hypothesise a "
                "multi-toggle (NTi-like) sprite distinct from prior Hkx."
            )
            self.reflexion_set_at_turn = self.turn_index
        else:
            self.stagnation_window += 1
            chosen = next((c for c in cards if c.id == choice.card_id), None)
            if chosen is not None:
                aid = archetype_alignment(chosen)
                if aid:
                    self.archetype_stagnation[aid] = (
                        self.archetype_stagnation.get(aid, 0) + 1
                    )

        self.choice_history.append(
            {
                "card_id": choice.card_id,
                "action_sequence": list(choice.action_sequence),
                "verdict": verdict,
                "skill_anchor": getattr(choice, "skill_anchor", "none"),
            }
        )

        # v32 Fix D: level transition expires ONLY cards bound to stale colors
        # (those whose expected_signature.dominant_transition references a
        # color that we observed in the prior level but not via this transition
        # event). CONFIRMED cards (cards that have ever produced a level rise
        # OR whose archetype is generic/mechanic-style without color binding)
        # are KEPT and propagate the analogy across levels. Without this M3 has
        # no positive transfer signal across levels.
        # Also: append confirmed predicates to a cross-level memory so M1 can
        # see "what worked before" even if the card itself dies later.
        if level_delta > 0:
            confirms_this_level = {
                e.get("card_id") for e in self.observation_log
                if e.get("verdict") == "confirm"
            }
            survivors: list = []
            for c in self.cards:
                sig = c.expected_signature or {}
                dt = sig.get("dominant_transition") or {}
                # Mechanic / archetype cards (no color-bound transition) survive
                color_bound = isinstance(dt, dict) and dt.get("from") is not None
                ever_confirmed = c.id in confirms_this_level
                if ever_confirmed or not color_bound:
                    survivors.append(c)
                    # Promote to cross_level confirmed memory
                    if ever_confirmed:
                        self.cross_level_confirmed.append(
                            {
                                "id": c.id,
                                "predicate": c.predicate,
                                "expected_signature": dict(sig),
                                "confirmed_at_level_delta_total": (
                                    self.cross_level_confirmed_total + level_delta
                                ),
                            }
                        )
                else:
                    self.falsified_cards.append(
                        {
                            "id": c.id,
                            "predicate": c.predicate,
                            "falsified_by": {
                                "reason": "level_changed_color_bound",
                                "level_delta": level_delta,
                            },
                        }
                    )
            self.cards = survivors
            self.cross_level_confirmed_total += level_delta
            self._persist()
            return

        # Promote cards into self.cards (deduped on id)
        existing = {c.id: c for c in self.cards}
        for c in cards:
            existing[c.id] = c
        self.cards = list(existing.values())

        # If falsified, move chosen card off active list
        if verdict == "falsify":
            chosen = next((c for c in self.cards if c.id == choice.card_id), None)
            if chosen is not None:
                self.falsified_cards.append(
                    {
                        "id": chosen.id,
                        "predicate": chosen.predicate,
                        "falsified_by": entry,
                    }
                )
                self.cards = [c for c in self.cards if c.id != chosen.id]

        self._persist()

    def add_skill(self, skill: AbstractSkill) -> None:
        if skill is None:
            return
        key = skill.goal_phrase
        if key in self.skills:
            del self.skills[key]
        self.skills[key] = skill
        # v20: density-aware LRU. When at cap, evict the LOWEST-density skill
        # (not the oldest). Density rewards verbose causal_mapping, populated
        # applies_when, and multi-region anchor — proxies for discriminative
        # info content. Ties broken by FIFO (the oldest entry wins eviction).
        while len(self.skills) > _MAX_SKILLS:
            min_key = min(
                self.skills.keys(),
                key=lambda k: (skill_density_score(self.skills[k]),),
            )
            del self.skills[min_key]
        self._persist()

    def add_lesson(self, lesson: LessonCard | None) -> None:
        """v21: append a M4 LessonCard, FIFO-cap at 30 entries."""
        if lesson is None:
            return
        self.lesson_log.append(lesson)
        if len(self.lesson_log) > 30:
            self.lesson_log = self.lesson_log[-30:]
        self._persist()

    def evict_skills_matching_archetype(self, archetype_id: str) -> int:
        """v20: when an archetype is stagnation-falsified, drop any promoted
        skill whose goal_phrase or causal_mapping references that archetype.
        Returns count of evicted skills."""
        if not archetype_id:
            return 0
        terms = _ARCHETYPE_TERMS.get(archetype_id, (archetype_id,))
        evict_keys = []
        for k, sk in self.skills.items():
            blob = (
                str(sk.goal_phrase or "")
                + " "
                + str(sk.causal_mapping or "")
                + " "
                + str(sk.applies_when or "")
            ).lower()
            if any(t in blob for t in terms):
                evict_keys.append(k)
        for k in evict_keys:
            del self.skills[k]
        if evict_keys:
            self._persist()
        return len(evict_keys)

    def top_skills(self, n: int = 5) -> list[dict]:
        items = list(self.skills.values())[-n:]
        return [asdict(s) for s in items]

    def observation_log_summary(self) -> list[dict]:
        return list(self.observation_log)

    def summary(self) -> dict:
        return {
            "active_cards": [asdict(c) for c in self.cards],
            "falsified_cards": list(self.falsified_cards),
            "observation_log": list(self.observation_log),
            "skills": [asdict(s) for s in self.skills.values()],
            "cross_level_confirmed": list(self.cross_level_confirmed[-20:]),
            "tried_coords": [list(t) for t in self.tried_coords],
            "choice_history": list(self.choice_history),
            "region_click_stats": dict(self.region_click_stats),
            # v20 fields for offline analysis
            "turn_index": self.turn_index,
            "stagnation_window": self.stagnation_window,
            "archetype_stagnation": dict(self.archetype_stagnation),
            "discovered_facts": [
                {"kind": k, "value": list(v) if isinstance(v, tuple) else v, "last_turn": t}
                for (k, v), t in self.discovered_facts.items()
            ],
            "lesson_log": [asdict(L) for L in self.lesson_log],
        }

    def _persist(self) -> None:
        try:
            (self.simple_workdir / "goal_board.json").write_text(
                json.dumps(self.summary(), indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("GoalBoard persist failed: %s", exc)

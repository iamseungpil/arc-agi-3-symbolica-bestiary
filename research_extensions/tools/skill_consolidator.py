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

Two modes for :func:`consolidate_from_level_clear`
--------------------------------------------------

**Default path (no env gate) — deterministic transcript, byte-identical.**
With the env gate absent, ``consolidate_from_level_clear`` returns exactly
the per-level *action transcript* Skills it always has: ``summary`` =
``"Skill for clearing level via <a -> b -> ...>"``, ``recipe`` = the
numbered ``"i. action=...; outcome=..."`` log.  This is the codex-required
ABLATION arm and the "NO LLM" contract holds for it unconditionally: no
import, no network, no randomness, no time read.  Every existing
``TestConsolidator`` case sets no env and therefore exercises this exact
byte-identical path.

**Abstract path (gated) — out-of-band, trace-only LLM abstraction.**
ONLY when ``A3_EXT == "trace2skill"`` AND ``T2S_SKILL_MODE == "abstract"``,
each cleared level's ``{action, outcome}`` chain is sent — out of band, in
a single pinned HTTP call to the proxy the frozen runner already booted
(model from ``S0_MODEL_PRESET``, default ``gpt-5.5``; ``temperature=0``) —
to be ABSTRACTED into a Skill whose ``summary`` is the inferred game
*mechanism* and whose ``recipe`` is the *meta-reasoning* a fresh agent
should follow (what to observe / how to decide), NOT a verbatim action
replay.  The estimand this serves is **"trace-derived abstract skill
induction + read-only reuse"** — explicitly NOT "the agent merely
accumulates its own experience verbatim" (that remains the v1 transcript
estimand, retained as the ablation arm).  Trace-only contract: the LLM is
given ONLY an ordered list of ``{step, action, outcome}`` — no game name,
no goal, no task, no future / outcome leakage.  The whole abstract path is
defensive: ANY ``BaseException`` is logged and falls back to the transcript
Skills, and individual per-chain failures fall back per chain (no chain is
ever dropped).  Frame / shape grounding is a deferred P2 and is NOT added
here (P1 = trace-only).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .skill_library import Skill

__all__ = [
    "consolidate_from_hypothesis",
    "consolidate_from_level_clear",
    "dedup_near_duplicate",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# trace2skill abstract-mode constants / injectable seam
# ---------------------------------------------------------------------------

# Frozen prompt.  EXACTLY ONE interpolation slot: ``{trace_json}``.  Contains
# NO game name / goal / task words — the model is told it is an *unknown*
# grid game and must infer mechanism + meta-reasoning from the action->outcome
# chain alone (trace-only contract, structurally enforced).
_T2S_ABSTRACT_PROMPT = (
    "You are given ONLY an ordered list of {{step,action,outcome}} from one "
    "playthrough segment of an unknown grid game. You are NOT told the game, "
    "the goal, or the task. Infer (a) the underlying MECHANISM the "
    "action->outcome chain reveals, (b) the META-REASONING a fresh agent "
    "should follow to reproduce the clear — what to observe and how to "
    "decide, not a verbatim action list, (c) applicability_conditions, (d) "
    "the expected_goal. Output STRICT JSON only: "
    '{{"summary":str,"recipe":str,"applicability_conditions":[str],'
    '"category":"mechanic"|"strategy","expected_goal":str}}\n\n'
    "TRACE:\n{trace_json}"
)

# Injectable seam for tests / callers: if set to a callable ``(prompt:str)
# -> str`` it is used as the LLM client (highest precedence).  Module-level
# so monkeypatch can replace it without env or network.
_T2S_ABSTRACT_CLIENT_HOOK: Callable[[str], str] | None = None

# Deterministic built-in stub (env ``T2S_ABSTRACT_CLIENT=stub``): a fixed,
# valid JSON string so the gated path is exercisable offline / in CI.
_T2S_STUB_JSON = json.dumps(
    {
        "summary": "Toggling the target region then confirming advances the game",
        "recipe": (
            "Observe which region the cursor is over and whether it matches "
            "the highlighted target; decide to toggle that region only when "
            "it is the target, then issue the confirm input once the region "
            "state changes."
        ),
        "applicability_conditions": [
            "a distinct target region is visible",
            "a confirm input is available",
        ],
        "category": "mechanic",
        "expected_goal": "advance to the next level by confirming the target region",
    },
    ensure_ascii=True,
)


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

    # --- gated, out-of-band, trace-only abstract mode (Option B) ----------
    # Default (gate absent) returns the byte-identical transcript ``skills``
    # built above; the abstract path is purely additive and ANY failure
    # falls back to the transcript skills (never raises).
    if (
        os.environ.get("A3_EXT") == "trace2skill"
        and os.environ.get("T2S_SKILL_MODE", "transcript") == "abstract"
        and skills
    ):
        try:
            abstracted = _abstract_skills_from_chains(chains, skills)
            if abstracted:
                return abstracted
        except BaseException:  # noqa: BLE001 — never break the frozen runner
            logger.exception(
                "trace2skill abstract mode failed; transcript fallback"
            )
    return skills


# ---------------------------------------------------------------------------
# trace2skill abstract mode — helpers (all lazy / defensive)
# ---------------------------------------------------------------------------


def _t2s_abstract_log_path() -> Path | None:
    """Resolve the JSONL artifact path, or ``None`` if logging is disabled."""
    explicit = os.environ.get("T2S_ABSTRACT_LOG_PATH")
    if explicit:
        return Path(explicit)
    persistence = os.environ.get("S0_SKILL_PERSISTENCE_PATH", "")
    if persistence:
        return Path(persistence).with_suffix(".abstract.jsonl")
    return None


def _t2s_log_record(record: dict[str, Any]) -> None:
    """Append one JSONL record.  Fully wrapped — logging never breaks a run."""
    try:
        path = _t2s_abstract_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=True, default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except BaseException:  # noqa: BLE001 — artifact logging is best-effort
        logger.debug("trace2skill abstract log write failed", exc_info=True)


def _t2s_stub_client(prompt: str) -> str:
    """Deterministic offline client — returns a fixed valid JSON string."""
    return _T2S_STUB_JSON


def _resolve_abstract_client() -> Callable[[str], str] | None:
    """Resolve the LLM client for abstract mode.

    Resolution order:
      (i)   module ``_T2S_ABSTRACT_CLIENT_HOOK`` if callable (test/inject seam);
      (ii)  ``T2S_ABSTRACT_CLIENT == "stub"`` → built-in deterministic stub;
      (iii) live: ONE HTTP call to the proxy the frozen runner already booted
            (NOT a new framework, NOT importing any handler / store / proposer).

    Returns ``None`` only if no client can be resolved (caller then keeps the
    transcript skill for that chain).
    """
    hook = _T2S_ABSTRACT_CLIENT_HOOK
    if callable(hook):
        return hook

    if os.environ.get("T2S_ABSTRACT_CLIENT") == "stub":
        return _t2s_stub_client

    def _live_client(prompt: str) -> str:
        import httpx  # lazy — no module-top network dependency

        port = os.environ.get("A3_PROXY_PORT", "9093")
        base = f"http://127.0.0.1:{port}/v1"
        model = os.environ.get("S0_MODEL_PRESET", "gpt-5.5")
        resp = httpx.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=float(os.environ.get("T2S_ABSTRACT_TIMEOUT_S", "30")) + 5.0,
        )
        return resp.json()["choices"][0]["message"]["content"]

    return _live_client


def _t2s_validate_parsed(parsed: Any) -> dict[str, Any] | None:
    """Schema-validate the parsed LLM JSON.  Returns a clean dict or ``None``.

    Rejects (→ per-chain transcript fallback) when ``summary`` is
    missing/empty OR is itself a transcript-marker string.
    """
    if not isinstance(parsed, dict):
        return None
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    if summary.startswith("Skill for clearing level via"):
        return None
    recipe = parsed.get("recipe")
    if not isinstance(recipe, str) or not recipe.strip():
        return None
    raw_conditions = parsed.get("applicability_conditions", [])
    if isinstance(raw_conditions, (list, tuple)):
        conditions = [str(c) for c in raw_conditions if str(c).strip()]
    elif raw_conditions:
        conditions = [str(raw_conditions)]
    else:
        conditions = []
    category = parsed.get("category", "mechanic")
    if category not in ("mechanic", "strategy"):
        category = "mechanic"
    expected_goal = parsed.get("expected_goal", "")
    expected_goal = str(expected_goal) if expected_goal else ""
    return {
        "summary": summary.strip(),
        "recipe": recipe,
        "applicability_conditions": conditions,
        "category": category,
        "expected_goal": expected_goal,
    }


def _abstract_skills_from_chains(
    chains: list[list[tuple[int, Any]]],
    transcript_skills: list[Skill],
) -> list[Skill]:
    """Abstract each action/outcome chain into a Skill (trace-only, gated).

    1:1 with ``transcript_skills`` (same order).  On ANY per-chain failure
    the ORIGINAL transcript skill is kept (no chain is ever dropped); the
    returned list always has ``len(chains)`` entries.  Reads ONLY
    ``.action`` / ``.outcome`` from each memory — never summary / details /
    memory_id — so no game / goal / future information can leak.
    """
    if len(chains) != len(transcript_skills):
        return []  # shape mismatch → full transcript fallback

    timeout_s = float(os.environ.get("T2S_ABSTRACT_TIMEOUT_S", "30"))
    model = os.environ.get("S0_MODEL_PRESET", "gpt-5.5")
    client = _resolve_abstract_client()
    out: list[Skill] = []

    for ordinal, (chain, transcript_skill) in enumerate(
        zip(chains, transcript_skills), start=1
    ):
        # TRACE-ONLY serialize: read ONLY .action / .outcome.
        trace = [
            {
                "step": i,
                "action": str(_memory_field(m, "action")),
                "outcome": str(_memory_field(m, "outcome")),
            }
            for i, (idx, m) in enumerate(chain, 1)
        ]
        prompt = _T2S_ABSTRACT_PROMPT.format(
            trace_json=json.dumps(trace, ensure_ascii=True)
        )
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        raw: str | None = None
        parsed_skill_dict: dict[str, Any] | None = None
        fallback_used = True
        fallback_reason: str | None = None

        if client is None:
            fallback_reason = "no_client_resolved"
            out.append(transcript_skill)
            _t2s_log_record(
                {
                    "ts": time.time(),
                    "chain_ordinal": ordinal,
                    "model": model,
                    "temperature": 0,
                    "prompt_sha": prompt_sha,
                    "raw_response": None,
                    "parsed_skill": None,
                    "fallback_used": True,
                    "fallback_reason": fallback_reason,
                }
            )
            continue

        try:
            from research_extensions.abstraction_handler import (  # lazy
                _invoke_with_timeout,
            )
            from research_extensions.abstraction import _parse_llm_json  # lazy

            raw = _invoke_with_timeout(client, prompt, timeout_s=timeout_s)
            parsed = _parse_llm_json(raw)
            validated = _t2s_validate_parsed(parsed)
            if validated is None:
                fallback_reason = "schema_invalid_or_unparseable"
                out.append(transcript_skill)
            else:
                expected_goal = validated["expected_goal"]
                recipe = validated["recipe"]
                if expected_goal:
                    recipe = recipe + "\n\nEXPECTED GOAL: " + expected_goal
                abstracted = Skill(
                    skill_id=str(uuid.uuid4()),
                    summary=validated["summary"][:200],
                    recipe=recipe,
                    # SAME evidence ids → runner t2s_prov:: tag + D4 guard
                    # keep operating unchanged.
                    evidence=list(transcript_skill.evidence),
                    posterior=(0, 0),
                    applicability_conditions=[
                        str(c)
                        for c in validated["applicability_conditions"]
                        if str(c).strip()
                    ],
                    parent_hypothesis_ids=[],
                    category=validated["category"],
                )
                parsed_skill_dict = {
                    "skill_id": abstracted.skill_id,
                    "summary": abstracted.summary,
                    "recipe": abstracted.recipe,
                    "applicability_conditions": list(
                        abstracted.applicability_conditions
                    ),
                    "category": abstracted.category,
                }
                fallback_used = False
                fallback_reason = None
                out.append(abstracted)
        except BaseException as exc:  # noqa: BLE001 — per-chain fallback
            fallback_reason = f"{type(exc).__name__}: {exc}"
            out.append(transcript_skill)

        _t2s_log_record(
            {
                "ts": time.time(),
                "chain_ordinal": ordinal,
                "model": model,
                "temperature": 0,
                "prompt_sha": prompt_sha,
                "raw_response": raw,
                "parsed_skill": parsed_skill_dict,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
            }
        )

    return out


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

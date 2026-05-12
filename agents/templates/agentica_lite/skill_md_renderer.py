"""v602 SKILL.md deterministic renderer (plan v602 §3).

Renders SkillState to Markdown with stable ordering:
  1A: confirmation_count desc, first_seen_run+turn asc
  1B: proposed_at_turn asc
  F:  (run_id, turn) desc, capped at top 30
  S.static: id asc
  S.dynamic: confirmed_count desc, status active first

Output is byte-identical given identical input (idempotent property INT07).
SKILL.md is generated-only; never hand-edited (auto-generated header banner).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from .skill_state import (
    ActiveHypothesis, ConfirmedMechanic, Falsification, SkillMetrics,
    SkillCard, SkillRecord, SkillState,
)

AUTO_GENERATED_HEADER = (
    "<!-- AUTO-GENERATED FROM skill_state.json - DO NOT EDIT -->"
)

# Render caps (plan v602 §3 + §5 INT09)
F_TOP_CAP = 30  # falsifications: top 30 by recency


def _render_metadata_yaml(state: SkillState) -> str:
    md = state.metadata
    metrics = state.metrics
    lines = [
        "---",
        f"schema_version: {md.schema_version}",
        f"game_id: {md.game_id}",
        f"last_updated: {md.last_updated}",
        "metrics:",
        f"  skills_proposed: {metrics.skills_proposed}",
        f"  skills_confirmed: {metrics.skills_confirmed}",
        f"  skills_retired: {metrics.skills_retired}",
        f"  proposal_to_confirmation_rate: {metrics.proposal_to_confirmation_rate}",
        f"  median_episodes_to_confirmation: {metrics.median_episodes_to_confirmation}",
        f"  skill_reuse_rate: {metrics.skill_reuse_rate}",
        f"  successful_L_plus_with_discovered_skill_rate: {metrics.successful_L_plus_with_discovered_skill_rate}",
        f"  skill_promotion_to_1A_rate: {metrics.skill_promotion_to_1A_rate}",
        "---",
    ]
    return "\n".join(lines)


def _sorted_1A(items: Iterable[ConfirmedMechanic]) -> list[ConfirmedMechanic]:
    return sorted(
        items,
        key=lambda c: (-c.confirmation_count, c.first_seen_run, c.first_seen_turn, c.id),
    )


def _sorted_1B(items: Iterable[ActiveHypothesis]) -> list[ActiveHypothesis]:
    return sorted(items, key=lambda h: (h.proposed_at_turn, h.id))


def _sorted_F(items: Iterable[Falsification]) -> list[Falsification]:
    """Sort by (run_id, turn) descending; cap at F_TOP_CAP."""
    rows = sorted(items, key=lambda f: (f.run_id, f.turn, f.id), reverse=True)
    return rows[:F_TOP_CAP]


def _sorted_static_skills(items: Iterable[SkillRecord]) -> list[SkillRecord]:
    return sorted([s for s in items if s.is_static], key=lambda s: s.id)


def _sorted_dynamic_skills(items: Iterable[SkillRecord]) -> list[SkillRecord]:
    """status active first; then by confirmed_count desc, id asc."""
    dyn = [s for s in items if not s.is_static]
    status_rank = {"active": 0, "pending_eviction": 1, "retired": 2}
    return sorted(
        dyn,
        key=lambda s: (status_rank.get(s.status, 99), -s.confirmed_count, s.id),
    )


def _render_1A_section(items: list[ConfirmedMechanic]) -> str:
    if not items:
        return "## 1A — Confirmed Abstract Mechanics\n_(none yet)_"
    lines = ["## 1A — Confirmed Abstract Mechanics"]
    for c in items:
        lines.append(
            f"- [{c.id}] {c.natural_language}"
        )
        if c.evidence_runs:
            lines.append(f"  evidence_runs: {', '.join(c.evidence_runs)}")
        lines.append(
            f"  first_seen: {c.first_seen_run} T{c.first_seen_turn}"
        )
        lines.append(f"  confirmation_count: {c.confirmation_count}")
        if c.paired_cf_offsets:
            lines.append(f"  pcf_offsets: {c.paired_cf_offsets}")
    return "\n".join(lines)


def _render_1B_section(items: list[ActiveHypothesis]) -> str:
    if not items:
        return "## 1B — Active Hypotheses (this episode)\n_(none active)_"
    lines = ["## 1B — Active Hypotheses (this episode)"]
    for h in items:
        lines.append(
            f"- [{h.id}] {h.predicate_id} on {h.region_hint}"
        )
        if h.required_pre_state:
            ps = ", ".join(f"{k}: {v}" for k, v in sorted(h.required_pre_state.items()))
            lines.append(f"  required_pre_state: {{{ps}}}")
        lines.append(f"  confidence: {h.confidence}")
        lines.append(f"  status: {h.status}")
    return "\n".join(lines)


def _render_F_section(items: list[Falsification]) -> str:
    if not items:
        return "## F — Falsified Hypotheses\n_(none yet)_"
    lines = ["## F — Falsified Hypotheses"]
    for f in items:
        lines.append(f"- [{f.id}] {f.natural_language}")
        lines.append(f"  failure_mode: {f.failure_mode}")
        if f.evidence:
            ev = ", ".join(f"{k}: {v}" for k, v in sorted(f.evidence.items()))
            lines.append(f"  evidence: {{{ev}}}")
    return "\n".join(lines)


def _render_S_section(items: list[SkillRecord]) -> str:
    static = _sorted_static_skills(items)
    dynamic = _sorted_dynamic_skills(items)
    lines = ["## S — Skill Library", "### S.static"]
    if static:
        for s in static:
            lines.append(
                f"- [{s.id}] family={s.family} ; "
                f"confirmed={s.confirmed_count} / falsified={s.falsified_count} "
                f"/ used in L+: {s.used_in_successful_L_plus}"
            )
    else:
        lines.append("_(none)_")
    lines.append("### S.dynamic")
    if dynamic:
        for s in dynamic:
            install = s.install_run or "?"
            lines.append(
                f"- [{s.id}] {s.description} family={s.family} "
                f"(installed {install})"
            )
            lines.append(
                f"  confirmed={s.confirmed_count} / falsified={s.falsified_count} "
                f"/ used in L+: {s.used_in_successful_L_plus}"
            )
            lines.append(f"  status: {s.status}")
    else:
        lines.append("_(none)_")
    return "\n".join(lines)


def _sorted_cards(items: Iterable[SkillCard], card_type: str) -> list[SkillCard]:
    status_rank = {"active": 0, "draft": 1, "narrowed": 2, "falsified": 3, "retired": 4}
    return sorted(
        [c for c in items if c.card_type == card_type],
        key=lambda c: (
            status_rank.get(c.status, 99),
            -(c.support_count - c.refute_count),
            c.id,
        ),
    )


def _render_card_group(title: str, items: list[SkillCard]) -> list[str]:
    lines = [f"### {title}"]
    if not items:
        lines.append("_(none)_")
        return lines
    for c in items:
        lines.append(
            f"- [{c.id}] status={c.status} support={c.support_count} "
            f"refute={c.refute_count}"
        )
        lines.append(f"  claim: {c.claim}")
        if c.state_features:
            lines.append(f"  state_features: {', '.join(c.state_features)}")
        if c.predicts:
            lines.append(f"  predicts: {' | '.join(c.predicts)}")
        if c.falsifiers:
            lines.append(f"  falsifiers: {' | '.join(c.falsifiers)}")
        if c.policy_hooks:
            lines.append(f"  policy_hooks: {', '.join(c.policy_hooks)}")
    return lines


def _render_cards_section(items: list[SkillCard]) -> str:
    lines = ["## Cards — Mechanic / Strategy / Hypothesis Ledger"]
    lines.extend(_render_card_group("Mechanics", _sorted_cards(items, "mechanic")))
    lines.extend(_render_card_group("Strategies", _sorted_cards(items, "strategy")))
    lines.extend(_render_card_group("Hypotheses", _sorted_cards(items, "hypothesis")))
    return "\n".join(lines)


def render(state: SkillState) -> str:
    """Render SkillState to a deterministic Markdown string."""
    sections = [
        AUTO_GENERATED_HEADER,
        _render_metadata_yaml(state),
        "",
        f"# SKILL.md — {state.metadata.game_id}",
        "",
        _render_1A_section(_sorted_1A(state.confirmed_mechanics)),
        "",
        _render_1B_section(_sorted_1B(state.active_hypotheses)),
        "",
        _render_F_section(_sorted_F(state.falsifications)),
        "",
        _render_cards_section(state.cards),
        "",
        _render_S_section(state.skill_lifecycle),
        "",
    ]
    return "\n".join(sections)


# =============================================================================
# Capped prompt-context view (plan v602 §3 caps + §5 INT09)
# =============================================================================

PROMPT_CONTEXT_BUDGET_BYTES = 4 * 1024  # 4 KB

# Plan v602 §3 caps:
CAP_1A_TOP = 8           # top 8 confirmed by confirmation_count
CAP_1A_RECENT = 4        # plus 4 most-recent (first_seen_turn desc)
CAP_F_RELEVANT = 8       # top 8 by relevance (using natural_language len as a proxy weight)
CAP_F_RECENT = 4         # plus 4 most-recent
CAP_S_DYNAMIC = 8        # top 8 active S.dynamic


def render_prompt_context(state: SkillState) -> str:
    """Build a capped context string for inclusion in the Proposer prompt.

    Caps (per plan v602 §3 + §5 INT09):
      1A: top 8 by confirmation_count + 4 most-recent
      F: top 8 by relevance + 4 most-recent
      S.dynamic: top 8 active
      S.static: counter-only summary

    Total is bounded at PROMPT_CONTEXT_BUDGET_BYTES (4 KB); if overshoot, the
    F section is truncated (lowest-priority).
    """
    # 1A — top by confirmation_count + recent
    sorted_1A = _sorted_1A(state.confirmed_mechanics)
    top_1A = sorted_1A[:CAP_1A_TOP]
    recent_1A = sorted(
        state.confirmed_mechanics,
        key=lambda c: c.first_seen_turn, reverse=True,
    )[:CAP_1A_RECENT]
    # union, preserving order: top first, then recent (de-dup)
    seen_ids: set[str] = set()
    one_A: list[ConfirmedMechanic] = []
    for c in list(top_1A) + list(recent_1A):
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        one_A.append(c)

    # F — top by length (relevance proxy) + recent
    f_by_len = sorted(state.falsifications, key=lambda f: -len(f.natural_language or ""))
    top_F = f_by_len[:CAP_F_RELEVANT]
    recent_F = sorted(state.falsifications, key=lambda f: f.turn, reverse=True)[:CAP_F_RECENT]
    seen_F: set[str] = set()
    F_list: list[Falsification] = []
    for f in list(top_F) + list(recent_F):
        if f.id in seen_F:
            continue
        seen_F.add(f.id)
        F_list.append(f)

    # S.dynamic — active only, top 8
    dynamic_active = [s for s in state.skill_lifecycle
                      if not s.is_static and s.status == "active"]
    dynamic_top = sorted(dynamic_active, key=lambda s: -s.confirmed_count)[:CAP_S_DYNAMIC]

    # S.static — counter summary
    static = [s for s in state.skill_lifecycle if s.is_static]

    lines: list[str] = []
    lines.append("# SKILL.md (capped view for proposer)")
    lines.append("## 1A (confirmed mechanics)")
    for c in one_A:
        lines.append(f"- [{c.id}] {c.natural_language} (n={c.confirmation_count})")
    lines.append("## F (falsified)")
    for f in F_list:
        lines.append(f"- [{f.id}] {f.natural_language[:120]}")
    lines.append("## S.dynamic (active)")
    for s in dynamic_top:
        lines.append(
            f"- [{s.id}] {s.description[:80]} family={s.family} "
            f"confirmed={s.confirmed_count}"
        )
    lines.append("## S.static (counters only)")
    if static:
        # roll-up summary
        confirmed_total = sum(s.confirmed_count for s in static)
        falsified_total = sum(s.falsified_count for s in static)
        lines.append(
            f"- {len(static)} static skills; "
            f"confirmed_total={confirmed_total}, falsified_total={falsified_total}"
        )

    text = "\n".join(lines)

    # Hard byte budget — chop F section first if oversize
    if len(text.encode("utf-8")) > PROMPT_CONTEXT_BUDGET_BYTES:
        # find & truncate F section
        marker = "## F (falsified)"
        next_section = "## S.dynamic (active)"
        try:
            i_F = text.index(marker)
            i_next = text.index(next_section)
            # keep up to budget by trimming F entries
            head = text[:i_F + len(marker) + 1]
            tail = text[i_next:]
            # re-add just enough F entries to stay under budget
            allowed = PROMPT_CONTEXT_BUDGET_BYTES - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
            f_lines: list[str] = []
            running = 0
            for f in F_list:
                line = f"- [{f.id}] {f.natural_language[:80]}\n"
                if running + len(line.encode("utf-8")) > allowed:
                    break
                f_lines.append(line)
                running += len(line.encode("utf-8"))
            text = head + "".join(f_lines) + tail

        except ValueError:
            # marker not found; truncate from end
            byte_text = text.encode("utf-8")[:PROMPT_CONTEXT_BUDGET_BYTES]
            text = byte_text.decode("utf-8", errors="ignore")

    return text

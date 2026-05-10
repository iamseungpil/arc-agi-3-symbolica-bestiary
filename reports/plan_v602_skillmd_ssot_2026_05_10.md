# Plan v602 — SKILL.md SSOT for Skill Discovery

**Date**: 2026-05-10
**Predecessor**: v601 (3-role + I-saturation-bundle, frozen tag `v601-3role-saturation-d1`, 40/40 tests pass)
**Trigger**: User clarified after v601 freeze that this research's primary angle is **skill discovery**; SKILL.md should be the SSOT for hypotheses + skills; raw memories (paired_cf, episode_journal) stay separate.
**Codex review-counter**: 1 round complete (rev A → this rev B incorporates codex direction)

---

## 0. Top-level I/H/V

**Intent (I)**. Restore SKILL.md as the human-readable single source of truth for hypotheses and skills, but keep it as a **deterministic rendered view** of an underlying typed `skill_state.json` so v601's deterministic-Memory-writer / conditional-Reflector role separation is preserved. This makes skill discovery measurable and publishable without re-introducing the v52-era multi-writer drift.

**Hypothesis (H)**. If hypotheses + skill lifecycle are tracked in a typed canonical store, then (a) Reflector promotion thresholds become statistically defensible, (b) skill reuse can be measured (`successful_L_plus_with_discovered_skill_rate`), (c) the paper has a tight skill-discovery narrative tying paired_cf evidence → skill induction → improved L+ rate.

**Verification (V)**. Two tiers:
- **Pre-launch (offline)**: 22 v601 INT + 13 v600 fixtures still pass + 3 new INT (INT07 render integrity, INT08 cross-run merge/promotion, INT09 prompt-injection cap) at 100%.
- **Post-launch (live, Phase D1+)**: 8 publication metrics tracked per episode (codex-converged set in §6).

---

## 1. Architecture (codex-converged: typed substrate + rendered view)

```
┌──────────────────── per-game memory layout ────────────────────┐
│                                                                  │
│   paired_cf_memory.jsonl   (raw evidence, append-only)           │
│   episode_journal.jsonl    (raw observations, append-only)       │
│   skill_state.json         ← TYPED canonical write target        │
│        ↑                                                          │
│        │  applied via structured patches:                        │
│        │     • Memory writer (deterministic): counter increments,│
│        │       1B rewrites on Proposer call, F-section appends,  │
│        │       static skill confirmed/falsified counters         │
│        │     • Reflector (LLM, conditional, severity > 0.5):     │
│        │       structured ops {promote_1A, propose_dynamic_skill,│
│        │                        retire_skill, falsify, merge_h}  │
│        ▼                                                          │
│   SKILL.md          ← DETERMINISTIC rendered view (read-only)    │
│        ↑                                                          │
│        │  re-rendered every persist; injected into Proposer      │
│        │  prompt with top-K caps                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**No hand-edit of SKILL.md.** Markdown drift = banned. All writers go through typed patches → applier → re-render.

---

## 2. Typed `skill_state.json` schema

```python
@dataclass
class SkillState:
    schema_version: int = 1
    game_id: str
    last_updated: str  # ISO-8601

    # 1A: cross-run confirmed abstract mechanics
    confirmed_mechanics: list[ConfirmedMechanic]

    # 1B: this-episode active hypothesis board (rewritten per Proposer call)
    active_hypotheses: list[ActiveHypothesis]

    # F: cross-run falsified hypotheses
    falsifications: list[Falsification]

    # S: skill library lifecycle (static + dynamic)
    skill_lifecycle: list[SkillRecord]

    # publication metrics rolling counts
    metrics: SkillMetrics

@dataclass
class ConfirmedMechanic:
    id: str  # A001, A002, ...
    natural_language: str
    evidence_runs: list[str]  # cycle_ids
    first_seen: tuple[str, int]  # (run_id, turn)
    confirmation_count: int
    paired_cf_offsets: list[int]  # references into paired_cf_memory.jsonl

@dataclass
class ActiveHypothesis:
    id: str  # H_<run_id>_<turn>
    predicate_id: str
    region_hint: str
    required_pre_state: dict
    confidence: float
    status: Literal["under_test", "confirmed", "refuted", "inconclusive"]
    proposed_at_turn: int

@dataclass
class Falsification:
    id: str  # F_<run_id>_<turn>
    natural_language: str
    failure_mode: Literal["coord_miss", "mechanic_absence", "wrong_state", "schema_violation"]
    evidence: dict  # {coord, primary_region, observed_dt, expected}
    run_id: str
    turn: int

@dataclass
class SkillRecord:
    id: str  # S00-S12 static, Sext-<sha> dynamic
    family: str
    description: str
    is_static: bool
    install_run: str | None
    confirmed_count: int
    falsified_count: int
    last_used_run: str | None
    used_in_successful_L_plus: int  # credit assignment counter
    status: Literal["active", "pending_eviction", "retired"]

@dataclass
class SkillMetrics:
    skills_proposed: int
    skills_confirmed: int
    skills_retired: int
    proposal_to_confirmation_rate: float  # rolling
    median_episodes_to_confirmation: float | None
    skill_reuse_rate: float
    successful_L_plus_with_discovered_skill_rate: float
    skill_promotion_to_1A_rate: float
```

---

## 3. SKILL.md render contract

Format frozen so diffs are minimal:

```markdown
---
schema_version: 1
game_id: ft09-9ab2447a
last_updated: 2026-05-10T18:00:00Z
metrics:
  skills_proposed: 12
  skills_confirmed: 3
  skills_retired: 2
  ...
---

# SKILL.md — ft09-9ab2447a

## 1A — Confirmed Abstract Mechanics
- [A001] Marker compass saturation gates L+ (8/8 clicks ≥ 1)
  evidence_runs: cycle237, cycle289
  first_seen: cycle237 T27
  confirmation_count: 3
  pcf_offsets: [42, 91, 117]

## 1B — Active Hypotheses (this episode)
- [H_cycle291_T15] P12_saturation_progress on R36
  required_pre_state: {marker_id: R31, sat_threshold: 7, sat_denom: 8}
  confidence: 0.7
  status: under_test

## F — Falsified Hypotheses
- [F_cycle291_T8] Sector_alignment R31 → click R16 → expected level_delta+1, observed 0
  failure_mode: wrong_state

## S — Skill Library
### S.static
- [S03] P03_sector_alignment family=sector_alignment
  confirmed: 2 / falsified: 5 / used in L+: 1

### S.dynamic
- [Sext-a3f8] L_ext_a3f8 family=extended (installed cycle274 T15)
  description: "click center of dominant 9→12 transition region"
  confirmed: 0 / falsified: 2 / used in L+: 0
  status: pending_eviction
```

Render order is **deterministic**: 1A sorted by `confirmation_count desc, first_seen asc`; 1B by `proposed_at_turn asc`; F by `(run_id, turn) desc, capped at top 30`; S.static by id; S.dynamic by `confirmed_count desc, status active first`.

---

## 4. Writer policies (codex-converged)

### Memory writer (deterministic, runs every turn)

```python
def memory_writer.handle_turn(turn_record):
    # 1. Append paired_cf entry if same-coord-different-outcome detected (v601 carry)
    paired_cf.maybe_append(turn_record)

    # 2. Rewrite 1B if Proposer fired this turn
    if turn_record.proposer_output is not None:
        skill_state.active_hypotheses = build_1B_from_proposer(turn_record.proposer_output)

    # 3. Update verdict status of last hypothesis
    if turn_record.last_hypothesis_id is not None:
        skill_state.update_status(last_hypothesis_id, turn_record.verdict)

    # 4. Append F-section entry if verdict == refuted
    if turn_record.verdict == "refuted":
        skill_state.falsifications.append(build_F_entry(turn_record))

    # 5. Update S.static counters from observation outcome
    skill_state.skill_lifecycle.update_counter(turn_record.predicate_id, turn_record.verdict)

    # 6. If level_delta > 0 and last_used_skill_id known: increment used_in_successful_L_plus
    if turn_record.level_delta > 0:
        skill_state.skill_lifecycle.credit_L_plus(turn_record.last_used_skill_id)

    # 7. Re-render SKILL.md from skill_state
    skill_md_renderer.render(skill_state)
```

### Reflector (LLM, conditional, severity > 0.5 + cooldown ≥ 5)

Reflector emits ONE of these structured patches (not free-form text):

```python
@dataclass
class PromoteA1Patch:
    op: Literal["promote_1A"]
    natural_language: str
    pcf_offsets: list[int]  # ≥ 2 paired_cf entries supporting

@dataclass
class ProposeDynamicSkillPatch:
    op: Literal["propose_dynamic_skill"]
    family: str
    description: str
    lambda_body: str  # Python code, sandbox-validated
    coord_policy: Literal["centroid", "sprite_center", "corner_top_left"]
    install_run: str

@dataclass
class RetireSkillPatch:
    op: Literal["retire_skill"]
    skill_id: str
    reason: str
    evidence: list[int]  # paired_cf_offsets

@dataclass
class FalsifyPatch:
    op: Literal["falsify"]
    hypothesis_id: str
    natural_language: str
```

**Applier validates** each patch (sandbox `lambda_body`, check confirmation_count threshold, etc.) before mutating `skill_state.json`. Validation failures logged with `reflector_patch_rejected_reason`.

---

## 5. Three new integration fixtures

### INT07 — Render integrity round-trip
Build `skill_state.json` programmatically with each section populated; render SKILL.md; parse back; assert no section loss, IDs preserved, ordering stable.

### INT08 — Cross-run merge + promotion
Inject 2 paired_cf entries for same discriminator across simulated cycle_a + cycle_b; run Reflector applier (mocked patch emission); assert (a) `confirmed_mechanics` gains 1 A_id with `confirmation_count == 2`, (b) SKILL.md renders new 1A entry, (c) `paired_cf_offsets` cite both source entries.

### INT09 — Prompt injection cap
Generate large `skill_state.json` (50 1A entries, 200 F entries, 30 S.dynamic). Build proposer-prompt-context view. Assert ≤ 8 1A + ≤ 4 recent + ≤ 8 F by relevance + ≤ 4 recent F + ≤ 8 active S.dynamic + counter-only S.static. Assert total prompt context ≤ 4 KB.

All 3 deterministic, ≤ 100 LOC each pytest, 100% pass required.

---

## 6. Publication metrics (codex-converged 8)

| Metric | Definition | Computed where |
|---|---|---|
| `skills_proposed_per_episode` | count of `propose_dynamic_skill` patches per episode | Reflector + applier |
| `skills_confirmed_per_episode` | count of S.dynamic transitions to `confirmed >= 2` per episode | Memory writer counter |
| `skills_retired_per_episode` | count of `retire_skill` patches per episode | Reflector + applier |
| `proposal_to_confirmation_rate` | rolling: `skills_confirmed / skills_proposed` | applier rolling counter |
| `median_episodes_to_confirmation` | median across S.dynamic of `(confirmation_run - install_run)` | Memory writer |
| `skill_reuse_rate` | `(unique skills used ≥ 2 times) / (total unique skills)` | Memory writer |
| `successful_L_plus_with_discovered_skill_rate` | **most important** — `(L+ events crediting an Sext-* skill) / (total L+ events)` | Memory writer credit assignment |
| `skill_promotion_to_1A_rate` | `(promote_1A patches accepted) / (S.dynamic confirmed)` | applier |

All emit to `skill_state.metrics` field; logged per-episode in `episode_journal.jsonl`.

---

## 7. Backward compatibility (codex 4-step migration)

1. **Introduce** `skill_state.json` (NEW).
2. **One-time importer**: on first load, if `skill_state.json` missing AND `cross_run_memory.json` exists, import 1A entries → `skill_state.confirmed_mechanics`. Log import.
3. **Render** `SKILL.md` from `skill_state.json` after import.
4. **Mark** `cross_run_memory.json` deprecated; v602 only writes to `skill_state.json`. v603 removes `cross_run_memory.json` entirely.

No dual-write authority. Avoids divergence (codex killer concern #5).

---

## 8. Code layout

```
agents/templates/agentica_lite/
  skill_state.py        (NEW, ~250 LOC) — dataclasses + JSON serialization + applier
  skill_md_renderer.py  (NEW, ~200 LOC) — deterministic Markdown render
  reflector.py          (extend +50 LOC) — emit structured patches instead of free text
  memory_writer.py      (extend +80 LOC) — call applier on each turn, render SKILL.md
  proposer_prompt.py    (extend +40 LOC) — inject capped SKILL.md view per §3 caps
tests/v602/
  fixtures/
    INT07_render_integrity.json
    INT08_cross_run_merge.json
    INT09_prompt_injection_cap.json
  test_v602_integration.py (~250 LOC)
tools/
  migrate_cross_run_memory.py (NEW, ~80 LOC) — one-time JSON → skill_state.json importer
```

Total v602 increment: ~700 LOC. v601's 2,264 → v602 ~2,964 LOC.

---

## 9. Implementation sequence

| Day | Deliverable | Verification |
|---|---|---|
| 1 | `skill_state.py` dataclasses + serializer + applier; `skill_md_renderer.py` deterministic render | unit test: round-trip ok, deterministic ordering |
| 2 | extend Memory writer with applier calls + `skill_state.json` writes; INT07 fixture | INT07 + 22 v601 + 13 v600 still green |
| 3 | extend Reflector with structured patch emission; `migrate_cross_run_memory.py` | INT08 fixture green |
| 4 | extend `proposer_prompt.py` with capped SKILL.md view injection; INT09 fixture | INT09 green |
| 5 | iterative-code-review smoke-critic-fix | ALL 22+13+3 = 38 fixtures + smoke green |
| 6 | Codex final gate on v602 code | tag `v602-skill-md-ssot` after [CONVERGED] |

---

## 10. Killer concerns (codex round 1)

| Concern | Mitigation |
|---|---|
| Markdown drift | SKILL.md is generated, not edited; lint test checks render hash |
| Prompt poisoning by stale F | top-K cap by relevance (§5 INT09) |
| False promotion from weak evidence | Reflector applier requires `confirmation_count ≥ 2` AND `pcf_offsets length ≥ 2` |
| Credit assignment inflation | `used_in_successful_L_plus` counter only increments when `last_used_skill_id` was actually selected by Policy at that turn |
| Dual-source divergence | 4-step migration deprecates `cross_run_memory.json` after import (§7) |

---

## 11. Per-module strict unit test addendum (rev B addition, user-requested 2026-05-10)

User-confirmed priority: strict per-module unit tests, especially LLM-call branches in `proposer.py` and prose validation in `reflector.py`. To be implemented alongside v602 INT07-INT09.

### 9 modules × ~75 unit tests target

| Module | LOC | Unit tests target | Critical branches |
|---|---|---|---|
| `agent.py` | 351 | 5 | proposer-fired / proposer-skipped / episode-end / budget-breach / missing-obs |
| `proposer.py` | 184 | 10 | **schema valid full / missing-field / extra-field / confidence-OOR / predicate-blacklist / region-unknown / timeout / parse-error / llm-no-client / tool-call-blocked** |
| `proposer_prompt.py` | 114 | 5 | template-lint / renders-with-markers / renders-without-markers (n/a) / R-aliasing / top-K-cap (post-v602) |
| `policy.py` | 300 | 14 | target-priority ×4 / sat-threshold ×3 / ucb1-floor / confidence-override ×4 / marker-validation / resolver ×2 / verdict ×3 |
| `memory_writer.py` | 263 | 14 | paired-cf-trigger ×3 / severity ×3 / extract-diff ×2 / reflector-spawn ×3 / cooldown ×2 / n=2 vs n≥3 ×1 |
| `reflector.py` | 136 | 6 | **structured-patch-emit / invalid-arm-dropped / text-truncation-256 / boost-clamped-0-0.3 / timeout / parse-error** |
| `predicate_library.py` | 325 | 10 | static P00-P11 / P12 with-target / P12 no-target / sandbox ×4 (eval/import/dunder/non-det) / install-runtime / resolve-coord ×3 |
| `predicate_posterior.py` | 299 | 11 | arm-key default / arm-key sat / repr / RASI floor / RASI no-split-n/a / co-decay / record-emission / select skip-no-target / select with-target / rank-top / load-rasi 7 modes |
| `stalemate_trigger.py` | 55 | 6 | fires-after-K / not-fires-below-K / not-fires-high-posterior / warm-up-first / warm-up-once / once-per-episode |
| **Total** | | **~81** | |

### Acceptance
- 81 = **coverage budget, not sacred number** (codex final gate refinement). Mechanical implementation-mirroring tests prunable. Priority order: (1) LLM branches in proposer/reflector → (2) saturation arm-key and posterior update → (3) SKILL.md SSOT renderer/idempotency/migration → (4) policy edge cases → (5) memory writer atomicity → (6) simple modules.
- LLM-call paths use **two-layer mocking**: ~8-10 JSON/schema validator isolation tests + ~4-5 mocked async LLM-call path tests (codex Q2 refinement). Don't duplicate suites.
- Existing 40 (v601 INT + property + smoke + v600) tests still pass
- **v600 13 fixtures explicitly re-run under v601/v602 saturation arm-key path** (codex final-gate refinement; addresses user audit gap "existing v600 fixtures not re-validated under v601 new logic")
- Combined wall-clock < 30 s

### File layout
```
tests/v601_unit/                  (NEW)
  test_agent.py                   (5 tests)
  test_proposer.py                (10 tests)
  test_proposer_prompt.py         (5 tests)
  test_policy.py                  (14 tests)
  test_memory_writer.py           (14 tests)
  test_reflector.py               (6 tests)
  test_predicate_library.py       (10 tests)
  test_predicate_posterior.py     (11 tests)
  test_stalemate_trigger.py       (6 tests)
```

### Implementation sequence
| Day | Deliverable |
|---|---|
| 1 | Quick wins: stalemate_trigger.py (6) + predicate_posterior.py (11) + predicate_library.py (10) |
| 2 | proposer.py (10, all LLM branches with mock) + proposer_prompt.py (5) |
| 3 | policy.py (14, target priority + confidence override critical) |
| 4 | memory_writer.py (14) + reflector.py (6, prose validation critical) |
| 5 | agent.py (5 orchestrator) + v602 INT07-INT09 implementation |
| 6 | iterative-code-review smoke-critic-fix until clean + codex final gate |

---

## 12. Convergence record

- 2026-05-10 codex round 1 (low effort): converged on 4-file layout (paired_cf + journal + skill_state + SKILL.md), structured Reflector patches, top-K prompt cap, 8 publication metrics, INT07/08/09 fixtures.
- 2026-05-10 user audit revealed unit test gap; §11 addendum addresses ~81 unit tests.

Pending: codex final gate on rev B (this doc) before implementation launch.

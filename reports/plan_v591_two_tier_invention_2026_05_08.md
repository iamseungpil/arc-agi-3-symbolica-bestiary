# Plan v591 — Two-Tier Predicate-Plus-Invention for M1 (B19)

**Status:** DRAFT round 1
**Date:** 2026-05-08
**Owner:** autoresearch
**Supersedes:** plan_v590_predicate_induction_2026_05_07.md (FROZEN-ROUND-3 but architecturally falsified by cycle237 vs cycle259-263 evidence)
**Scope:** Two-tier hypothesis system. TIER-A: Symbolica-emitted predicates (deterministic, advisory). TIER-B: M1-invented chids constrained by grammar. Snap-fallback ONLY on coord-fail. M4 promotes successful invented chids to library (DreamCoder-style abstraction).
**Rejected if:** any leak vocabulary in invented chid; coord references future state or hidden labels; M1 abandons the predicate library entirely (TIER-A retention <30%); cycle237 success pattern not reproducible in smoke.

---

## 0. Empirical diagnosis (why v590 → v591)

Concrete trace evidence (verified 2026-05-08):

cycle237 (39 turns, REACHED L+2) — `simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl`:

| turn | coord | snap | chid | observation | level_delta |
|------|-------|------|------|-------------|-------------|
| T0 | [6,38] | True | `P01_unclicked_neighbor_of_active_marker:T0:R21` | _outside_ | 0 |
| T1 | [34,34] | False | `H_static_nonmarker_R19` | _outside_ | 0 |
| T3 | [38,38] | False | `H_crop_align_R31_NW` | R24 | 0 |
| T6 | [38,54] | False | `P_crop_compass_sweep_R31` | R36 | **+1 (L+1)** |
| T8 | [38,48] | False | `H_fresh_neighbor_toggle_R16` | R16 | 0 |
| T11 | [38,32] | False | `P_shared_blank_sweep_R12` | R10 | 0 |
| T22 | [30,48] | False | `H_R15_lower_S` | R15 | 0 |
| T26 | [22,40] | False | `P_R12_crop_sector_alignment` | R11 | 0 |
| T27 | [38,48] | False | `P_R12_crop_sector_alignment` | R16 | **+1 (L+2)** |

Counts: 17/39 snap=True (real predicate_id from PREDICATE_LIBRARY) → 0 level rise events.
22/39 snap=False (M1-INVENTED chid) → BOTH L+1 and L+2 events.

cycle259/260 (regression) — same pattern but 100% strict-binding compliance, 67-80% _outside_, 0 level rise. v590 H2/H3 successfully forced M1 to obey the library and obedience killed exploration.

**Diagnosis:** v590 plan §3 H2 ("≥80% predicate_id reference") + §4.3 binding ("coord = predicate suggested_coord, stay inside anchor_region.bbox") MECHANICALLY block the cycle237 success pattern.

The structural properties M1 invented that the library lacks:
1. **Region-direction triple**: `R31_NW` = NW corner of R31's bbox (library only emits centroid).
2. **Cross-region relation**: `shared_blank_sweep_R12` = sweep cells shared between R12 and another region (library is single-region).
3. **Sprite-internal geometry**: `crop_align_NW` = test alignment within R31's 3×3 crop (library is bbox-level, no sprite).
4. **Negation hypothesis**: `static_nonmarker_R19` = test R19 is NON-INTERACTIVE (library only generates positive tests).
5. **Sequence hypothesis**: `compass_sweep_R31` = ordered click pattern across compass (library is single-click).

---

## 1. Survey of related work (theory-grounded)

### 1.1 DreamCoder Wake-Sleep-Abstraction (Ellis 2021)
Programs as compressions of solved tasks. Wake proposes; Sleep compresses; Abstraction promotes reusable subroutines. **Direct relevance**: TIER-B's invented chids are wake-phase programs; M4's promotion to library = sleep-abstraction.

### 1.2 Library Learning (Bowers 2022, Stitch / DreamCoder-DSL)
Programs cluster into a refactored library by minimum description length. **Relevance**: defines the formal procedure for promoting an invented chid to a library template.

### 1.3 Voyager (Wang 2023)
LLM agent in Minecraft maintains a SKILL LIBRARY of Python functions, adds new skills when novel sub-task succeeds. **Relevance**: closest analogue — TIER-B chids are atomic skills; successful chids promote.

### 1.4 ReAct-Reflexion (Shinn 2023, Yao 2023)
Reasoning + acting + self-critique loop. **Relevance**: M1 reasoning + M4 reflexion = ReAct + Reflexion. v591 keeps this and adds chid grammar for invented hypotheses.

### 1.5 ILP / Popper (Cropper 2021)
Inductive logic programming finds Horn clauses. **Relevance**: TIER-A library defines a small ILP-style hypothesis class; we don't search but enumerate.

### 1.6 Compositional generalisation (Lake & Baroni 2018)
Models that generalise to NEW combinations by composing primitives. **Relevance**: cycle237's success was M1 COMPOSING primitives (region × direction × relation × negation) at INFERENCE TIME. v591 must allow this composition.

### 1.7 Toolformer / Tool augmentation (Schick 2023)
LLM learns when to call tools vs reason directly. **Relevance**: M1 should choose between TIER-A predicate (tool) and TIER-B invention (free reasoning).

### 1.8 Relation to v590
v590 = DreamCoder library only, no Wake invention. v591 = Wake + library + Sleep promotion. Theory predicts v591 generalises better when single puzzle has out-of-library structure (which ft09 demonstrably does per cycle237).

---

## 2. Method candidates (theory-grounded, multiple distinct)

### 2.1 M-INV-A — Predicates ADVISORY, M1 free invention (REJECTED-naive)
M1 may invent any chid. Pros: restores cycle237. Cons: undisciplined, regresses to v55/v56 garbage chids; library becomes vestigial. **Rejected** because v55-v56 era already demonstrated unconstrained invention is noisy.

### 2.2 M-INV-B — Library extension to compositional templates (REJECTED-bloat)
Add P13_corner, P14_compass_NW, P15_compass_NE, ... to library. Pros: stays in v590 architecture. Cons: enumerative blow-up (5 directions × 4 corners × 12 regions = 240 templates); cycle237's `P_R12_crop_sector_alignment` cross-region pattern is NOT enumerable. **Rejected** because compositional structure cannot be exhaustively enumerated.

### 2.3 M-INV-C — Two-tier with chid grammar + abstraction promotion (ADOPTED)
- TIER-A: PREDICATE_LIBRARY (12 templates from v590) emits ≤8 candidate predicates per turn. Score = Beta-Bernoulli posterior.
- TIER-B: M1 may invent chid following GRAMMAR `^(H|P)_[a-z][a-z0-9_]*_R\d+(_[NSEW]{1,2})?$`. Coord MUST lie inside SOME visible region's bbox. M1 cites which region in thought.
- M1 selects ONE chid per turn: TIER-A predicate_id OR TIER-B invented chid.
- M4 reflexion observes invented-chid outcomes; if any TIER-B chid template name reaches `supported≥3` across runs, promote it to TIER-A library (DreamCoder-Sleep abstraction).
- Snap-fallback fires ONLY when M1's coord is outside ALL visible regions AND no chid supplied.

**Distinction from v590:** v590 forces TIER-A. v591 makes both tiers first-class; M4 sleep-promotes successful inventions.

**Distinction from M-INV-A:** TIER-B is grammar-constrained, leak-checked, region-anchored — not free-form garbage.

### 2.4 M-INV-D — Differentiable predicate inducer (DEFERRED to B20)
Train a neural module to score predicate compositions. Reserved if v591 fails to reach L+3.

### 2.5 Down-select rationale
**Adopted: M-INV-C.** Cycle237 success pattern is empirically a TWO-TIER pattern (used both real predicates and invented chids); v591 codifies this. Theory: Voyager + DreamCoder-Sleep. Verification: H1-H6 below.

---

## 3. Intent / Hypothesis / Verification (I/H/V)

### Intent

Reproduce cycle237's L+2 success pattern by allowing M1 to use BOTH the deterministic predicate library AND structurally-grammared invented hypotheses. M4 reflexion observes invention outcomes and promotes successful ones. Snap-fallback exists only as safety net for coord-fail, not as enforcement of binding.

### Hypotheses (each falsifiable, with explicit V-gate)

**H1 (Predicate generation works).** TIER-A generator emits ≥5 grounded predicates per turn at chain length ≥10. ≥3 distinct templates per cycle.
- **V-H1**: trace scan: `predicate_tests_emitted` length ≥5 in ≥80% of turns past T10. Distinct `template_id` count ≥3 per cycle.
- **Falsifiable**: <5 predicates in any turn past T10, or <3 distinct templates in 200 turns.

**H2 (Two-tier selection).** M1 cites EITHER a `predicate_id` from TIER-A OR a TIER-B invented chid following the grammar. Both are valid.
- **V-H2**: trace scan: ≥95% of turns past T5 have `chosen_hypothesis_id` matching `^(P\d{2}_[a-z_]+:T\d+:R\d+|(H|P)_[a-z][a-z0-9_]*_R\d+(_[NSEW]{1,2})?)$`. Empty/null chid <5%.
- **Falsifiable**: ≥10% of qualified turns have null chid, or ≥10% have ungrammatical chid.

**H3 (Region-anchored coord).** Click coord lies inside SOME visible region's bbox in ≥85% of qualified turns. Out-of-region clicks fall back to chosen-card region center.
- **V-H3**: trace scan: `observation.primary_region_id != "_outside_"` in ≥85% of clicks past T5. (Note: ft09 has _outside_ even with valid coord, so we also accept `coord ∈ visible_regions.bbox.union`.)
- **Falsifiable**: <70% in-region rate.

**H4 (Invention is non-trivial).** Of TIER-B chids, ≥30% are NOT direct paraphrases of TIER-A predicates (i.e., chid names contain at least one of: corner-direction (`_N|_S|_E|_W|_NE|_SE|_NW|_SW`), cross-region marker, sprite-internal vocab, or negation prefix `static_`).
- **V-H4**: post-cycle script `scripts/measure_v591_invention.py` reports `non_trivial_invention_rate ≥ 0.30`.
- **Falsifiable**: <0.15.

**H5 (Posterior signal accumulates).** ≥1 TIER-A template OR TIER-B chid-template reaches `supported_count ≥ 3` within 100 turns.
- **V-H5**: `template_history.json` + `invented_chid_history.json` scan.
- **Falsifiable**: all entries stuck at supported≤1 after 100 turns.

**H6 (L+2 reach — integration target).** ≥1 of cycle264/265/266 reaches an L+2 event within 400 turns.
- **V-H6**: trace scan `level_delta ≥ 1` accumulator reaches level=2.
- **Falsifiable**: zero L+2 across all three.

### Verification gates

| Gate | Method | Pass condition |
|---|---|---|
| V-LEAK | `python3 scripts/check_no_leak_prompts.py` + scan TIER-A library + grammar-check TIER-B emitted chids | 5/5 PASS, zero forbidden vocab |
| V-TESTS | `pytest tests/test_v591_*.py tests/test_v590_*.py tests/test_v589_*.py tests/test_v588_*.py -v` | all PASS |
| V-FIXTURE | `tests/test_v591_invention_grammar.py` train ≥80%, val ≥75%, gap ≤15pt | per project memory `feedback_module_fixture_first.md` |
| V-SMOKE | 5-turn run with REAL TRAPI gpt-5.4-pro; verify ≥1 invented chid emitted, ≥1 TIER-A predicate selected, no exceptions | trace.jsonl shows both tiers |
| V-H1..V-H6 | per §3 above | as listed |

V-LEAK + V-TESTS + V-FIXTURE + V-SMOKE are **launch-blocking**.
V-H1..H4 are wiring; failure = bug.
V-H5 measures functional accumulation.
V-H6 is the integration target.

---

## 4. Implementation specification

### 4.1 TIER-A: keep v590 library

`tools/predicate_generator.py` UNCHANGED from v590. Beta-Bernoulli scoring, 12 templates + P00 fallback.

### 4.2 TIER-B: invention grammar

New file `tools/chid_grammar.py`:

```python
import re
INVENTED_CHID_RE = re.compile(
    r"^(H|P)_"
    r"(?P<rationale>[a-z][a-z0-9_]{2,40})"
    r"_R(?P<region>\d+)"
    r"(_(?P<direction>N|S|E|W|NE|NW|SE|SW))?$"
)

LEAK_VOCAB = {  # mirror prompts.py forbidden set
    "per_neighbor_target","target_color","needs_toggle","marker_progress",
    "joint_neighbors","expected_neighbor_colors","win_state","goal_state",
    "is_target_state","precision_score",
}

def validate_invented_chid(chid: str, visible_region_ids: set[str]) -> tuple[bool, str]:
    m = INVENTED_CHID_RE.match(chid or "")
    if not m: return (False, "ungrammatical")
    if any(v in chid.lower() for v in LEAK_VOCAB): return (False, "leak_vocab")
    rid = f"R{m.group('region')}"
    if rid not in visible_region_ids: return (False, "region_not_visible")
    return (True, "ok")
```

Tests: `tests/test_v591_chid_grammar.py` ≥12 tests covering accept/reject pairs.

### 4.3 agent.py changes (file:line)

Locate `run_turn` near line 1700. Modify post-action-decision section:

**REMOVE** these snap-fallback paths added in rounds 1-7:
- L1828 multicolor-corner snap (gated by EXPLORATION_TRIGGER=999, effectively dead, but DELETE to reduce surface).
- L1881 chosen_card region-center snap on `not in_some_region` — KEEP but only if BOTH chid is null AND coord is outside.
- L1909-L1927 `_b18_cold_start` snap — REMOVE entirely (round-3..6 added it; round-7 tightened it; v591 deletes it because TIER-B grammar handles cold-start).
- L1938-L1987 anti-oscillation tier-1/2/3 — KEEP as safety; only triggers on identical-coord 2-repeat.

**ADD** TIER-B chid validation (pseudocode after `aresp` is parsed):
```python
chid = aresp.get("chosen_hypothesis_id")
visible_rids = {r.get("id") for r in visible_regions if r.get("id")}
if chid and not _is_tierA(chid):
    ok, reason = validate_invented_chid(chid, visible_rids)
    if not ok:
        # invalid invention: log, fall through to snap (which will only fire on coord-fail).
        board.invented_invalid_count += 1
        aresp["chosen_hypothesis_id"] = None
```

**ADD** template_history equivalent for invented chids:
```python
def _invented_chid_template(chid: str) -> str | None:
    m = INVENTED_CHID_RE.match(chid)
    return f"{m.group(1)}_{m.group('rationale')}" if m else None
# Persist to simple_logs/<game>/invented_chid_history.json (race-safe per B9 v2 pattern).
```

### 4.4 prompts.py M1 changes (file:line)

Modify ACTION_SYSTEM_PROMPT around L353-L392 (the HARD BINDING block):

**Replace** strict binding block with two-tier selection:

```
- CANDIDATE_TESTS (B18 PREDICATE_TESTS): list (≤8) of falsifiable
  predicate-region tests. Each entry has predicate_id, anchor_region, suggested_coord,
  score (Beta-Bernoulli posterior), coord_policy.

  TWO-TIER SELECTION — choose one of:

  TIER-A: pick a predicate_id from CANDIDATE_TESTS by information gain
          (highest score × novelty). Set chosen_hypothesis_id to the
          predicate_id. Coord starts at suggested_coord; you may adjust
          ±2 px or pick a different cell INSIDE anchor_region.bbox.

  TIER-B: invent a structurally-justified hypothesis chid. Grammar:
          ^(H|P)_<rationale>_R<region_id>[_<direction>]$
          where <rationale> is a snake_case noun phrase you choose
          (e.g. crop_align, sector_alignment, fresh_neighbor_toggle,
          shared_blank_sweep, static_nonmarker, compass_sweep,
          corner_probe), <region_id> is one currently in
          VISIBLE_REGIONS, and <direction> ∈ {N,S,E,W,NE,NW,SE,SW}
          if the hypothesis tests a specific corner/edge.

          You MAY use TIER-B when:
          (a) the predicate library has no entry that matches the
              structural relation you want to test (e.g. cross-region
              shared blank, sprite-internal alignment, corner-of-marker);
          (b) you want to test a NEGATION (region X is NON-interactive);
          (c) you want to test a SEQUENCE (compass sweep across N→E→S→W).

          Coord MUST lie inside the bbox of the cited R<region_id>.
          For corner/direction tests, compute the corner pixel of the
          region's bbox. For sprite-internal tests, sample the appropriate
          3×3 crop cell.

          Forbidden vocab in chid: per_neighbor_target, target_color,
          needs_toggle, marker_progress, joint_neighbors,
          expected_neighbor_colors, win_state, goal_state, is_target_state,
          precision_score.

  YOUR THOUGHT MUST:
  (1) State whether you chose TIER-A or TIER-B and WHY.
  (2) If TIER-A, cite the chosen predicate_id and its score.
  (3) If TIER-B, cite the invented chid AND the structural property
      it tests that is missing from CANDIDATE_TESTS.
  (4) Show the coord computation (region.bbox + offset/corner/direction).
```

### 4.5 M4 reflexion: invention promotion (sleep-abstraction)

Modify `m4_prompt.py` and `goal_board.apply_m4_reflexions`:

After M4 emits its `m4_output`, agent.py scans `invented_chid_history.json` for any TIER-B template that has `supported_count ≥ 3`. Each such template gets logged as a candidate for `cross_run_memory.json` Section 1A (Abstract Mechanics) with text "When tested at <region_kind>, hypothesis pattern <template> succeeded ≥3×". M4 may explicitly add it via `promote_to_1A` with provenance.

This is the **DreamCoder-Sleep abstraction step**: invented programs that demonstrate value enter the library.

### 4.6 Latency mitigation

Switch model: M1 → `gpt-5.4-pro` (per memory `project_skilldiscovery.md` GPT 5.4 study confirmed). M4 reflexion → keep gpt-5.5 (rare, ≤every 5 turns).

**File:line changes:**
- `agents/templates/agentica_v57/agent.py` near `_make_llm_client` or model spawn call: detect call type, route accordingly. Concretely add `MODEL_BY_MODULE = {"M1":"gpt-5.4-pro","M3":"gpt-5.4-pro","M4":"gpt-5.5_2026-04-24"}` and use it.
- Validate via smoke that gpt-5.4-pro is available on TRAPI deployment.

Estimated cycle wall-clock: 60s × 400 + 185s × 80 = 24000+14800 = ~11 hours. Achievable with detached overnight launch.

### 4.7 Detached launch (Q3)

`linger=yes` confirmed for user. systemd-run --user nonetheless got SIGINT — root cause likely python signal handler in launcher. Use:

```bash
LOG=simple_logs/ft09-9ab2447a/cycle264_v591/launch.log
mkdir -p $(dirname "$LOG")
setsid -f bash -c '
  exec </dev/null
  exec >>'"$LOG"' 2>&1
  cd /home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate llm-addiction || true
  export ARC_NO_GOAL_LEAK=1
  export ARC_SMOKE_MAX_ACTIONS=400
  export V591_M1_MODEL=gpt-5.4-pro
  export V591_M4_MODEL=gpt-5.5_2026-04-24
  exec python -u launch_v57.py --game ft09-9ab2447a --max_actions 400 \
    --namespace cycle264_v591_${RANDOM}
'
```

`setsid -f` forks into a new session, fully untethered from the parent terminal/PAM session. `exec` replaces the shell so the only running process is python. Backup: `tmux new-session -d -s cycle264 'CMD'`.

Smoke first: 5-turn run with `ARC_SMOKE_MAX_ACTIONS=5` to verify TRAPI + both tiers + no exceptions. Only after smoke green, full launch.

### 4.8 Anti-leak

- TIER-B grammar regex enforces snake_case rationale + R-id + optional direction; rejects free-form text.
- LEAK_VOCAB scan on every emitted chid before logging.
- M4 promotion preserves only the `(H|P)_<rationale>` prefix as template_id; specific R-id stripped (so `H_crop_align_R31_NW` and `H_crop_align_R12_NW` collapse to template `H_crop_align`).
- V-LEAK extended to scan `invented_chid_history.json` and `cross_run_memory.json` Section 1A entries.

### 4.9 Tests (≥18 tests)

`tests/test_v591_chid_grammar.py` (≥12):
- T-G-1..6: accept valid grammar examples (cycle237's chids).
- T-G-7..10: reject ungrammatical (no R-id, leak vocab, non-snake_case, missing prefix).
- T-G-11: reject region not in visible_regions.
- T-G-12: template_id extraction strips R-id.

`tests/test_v591_two_tier.py` (≥6):
- T-2T-1: M1 returning predicate_id passes validation.
- T-2T-2: M1 returning invented chid passes validation.
- T-2T-3: M1 returning ungrammatical chid → fallback.
- T-2T-4: snap-fallback fires only on coord-fail (not on invented chid).
- T-2T-5: invented_chid_history.json updates on supported verdict.
- T-2T-6: M4 promotion to 1A fires when template_id supported≥3.

`tests/test_v591_invention_grammar.py` (fixture coverage ≥30):
- Use cycle237's 22 invented chids + 8 synthetic edges → train ≥80%, val ≥75%, gap ≤15pt.

### 4.10 Backward compatibility

- v590 PREDICATE_LIBRARY kept verbatim.
- Old traces with `chosen_hypothesis_id` matching v590 format still pass H2.
- New trace fields: `chid_tier` ("A"|"B"|"null"), `invented_chid_template`.
- Visualizer: dual-render TIER-A and TIER-B counts.

---

## 5. Implementation order (mechanical)

1. `tools/chid_grammar.py` — regex + leak scan + validate function.
2. `tests/test_v591_chid_grammar.py` — 12 tests; run.
3. `agents/templates/agentica_v57/agent.py` — REMOVE _b18_cold_start (L1909-1927); REMOVE L1828 dead path; KEEP L1881 (chosen_card region snap) but only when chid null; KEEP anti-osc tiers.
4. `agents/templates/agentica_v57/agent.py` — add `validate_invented_chid` call + invented_chid_history persistence.
5. `agents/templates/agentica_v57/prompts.py` — replace HARD BINDING block (L353-L392) with TWO-TIER SELECTION block from §4.4.
6. `agents/templates/agentica_v57/agent.py` — model routing per §4.6.
7. M4: add `promote_to_1A` provenance for invented templates with supported≥3.
8. `tests/test_v591_two_tier.py` — 6 tests; run.
9. `tools/extract_v591_invention_fixtures.py` — extract from cycle237 trace.
10. `tests/test_v591_invention_grammar.py` — 30+ fixtures; run.
11. V-LEAK + V-TESTS + V-FIXTURE gates; fix any failures.
12. V-SMOKE: 5-turn detached launch with `ARC_SMOKE_MAX_ACTIONS=5`. Verify both tiers emit, no exceptions, gpt-5.4-pro responds <90s.
13. iterative-code-review skill: 3-round critic-fix loop on §4.3 + §4.4 changes.
14. Detached launch cycle264-266 per §4.7.
15. Monitor V-H1..V-H6 over 400 actions × 3 cycles.

After each step 1-12: run tests. Do not proceed if regressing.

---

## 6. Risks and mitigations

- **R-1 (HIGH): TIER-B regresses to garbage.** M1 might invent nonsense chids. Mitigation: grammar regex + leak scan + post-cycle non-trivial-rate check (V-H4 ≥0.30).
- **R-2 (HIGH): cycle237 was a fluke.** Single trace might not represent the true success path. Mitigation: V-H6 requires ≥1 of 3 cycles; if all fail, escalate to B20 (differentiable inducer).
- **R-3 (MED): coord computation for invented chids.** M1 might cite `R31_NW` but emit centroid coord. Mitigation: prompts.py §4.4 explicitly demands "show the coord computation"; agent.py post-checks coord ∈ region.bbox.
- **R-4 (MED): gpt-5.4-pro reasoning quality drop.** Bulk model swap might lose cycle237's quality. Mitigation: smoke compares M1 thought quality (length + chain-of-reasoning citations); if degraded, revert to gpt-5.5 with longer cycle wall-clock.
- **R-5 (MED): M4 promotion noise.** Promoting `static_nonmarker` to 1A might bias future M1 against exploration. Mitigation: 1A entries marked as `confirmed_runs ≥ 1`; M1 prompt treats 1A as advisory not prescriptive.
- **R-6 (LOW): backward-compat break in trace fields.** Add `chid_tier` field; visualizer handles both. Documented.
- **R-7 (LOW): linger + setsid still SIGINTs.** Backup tmux server. If both fail, run inside Azure ML compute target with explicit job lifecycle.

---

## 7. Self-critic round 1 — findings

**C-1 (TIER-B grammar too restrictive on direction).** cycle237 has chids like `H_crop_align_R31_NW`, `H_R15_lower_S` (note `_lower_S` is a 2-token rationale + direction). Grammar regex `^(H|P)_<rationale>_R\d+(_[NSEW]{1,2})?$` requires direction at end and rationale before R-id. Check cycle237 chids:
  - `H_static_nonmarker_R19` ✓ (rationale=static_nonmarker, R19)
  - `H_crop_align_R31_NW` ✓
  - `H_R15_lower_S` ✗ — R-id appears MID-string, direction at end. FAIL grammar.
  - `P_R12_crop_sector_alignment` ✗ — R-id mid-string, no direction.
  - `H_replay_prior_trigger` ✗ — no R-id at all.

**Mitigation**: relax grammar to permit R-id anywhere in the chid:
```python
INVENTED_CHID_RE = re.compile(r"^(H|P)_(?=.*R\d+)[a-z0-9_]{3,80}$")
```
Then extract first `R\d+` as region_id, last `_(NS|EW|NE|...)` as direction (optional). Validate region exists in visible_regions.

**C-2 (Invented chid `H_replay_prior_trigger` has no R-id).** Some chids reference behavior, not region. Allow these but require trace evidence (last verbose turn region) to anchor.
- Mitigation: add CHID_KIND={anchored_to_region, anchored_to_history}. History-anchored chids must reference `last_T<n>` in chid.

**C-3 (V-H3 in-region rate definition).** v590 H3 uses `primary_region_id != "_outside_"` which on ft09 is too strict — ft09 reports _outside_ frequently due to game segmentation. cycle237 has many _outside_ on snap=False turns too (T1 [34,34] is _outside_).
- Mitigation: change V-H3 metric to `coord ∈ ⋃ visible_regions.bbox` (geometric, not game-reported).

**C-4 (M4 promotion granularity).** Promoting `H_crop_align` template might be too coarse if `H_crop_align_R31_NW` worked but `H_crop_align_R12_E` didn't.
- Mitigation: template_id includes direction if present. Promote `H_crop_align_NW` not `H_crop_align`.

**C-5 (Smoke test calls TRAPI).** Smoke is 5 calls = 5 × 60s gpt-5.4-pro = 5 minutes. Acceptable but blocking.
- Mitigation: smoke can use stub LLM (return canned thought + chid) for V-TESTS pass, then ONE real TRAPI call for V-SMOKE-LIVE.

**C-6 (Forbidden vocab list incomplete).** Plan §4.4 lists 9 leak terms; check prompts.py for full list.
- Mitigation: §4.4 LEAK_VOCAB will be sourced from `agents/templates/agentica_v57/prompts.py` LEAK_TERMS module-level constant (or whatever exists). Audit before launch.

**C-7 (cycle237 trace evidence might not generalise to cycle264 puzzle layout).** ft09 might have changed since cycle237. Mitigation: re-run cycle263's level layout against cycle237's; if different, re-anchor invention examples.

**C-8 (TRAPI gpt-5.4-pro existence not verified).** Plan assumes deployment exists.
- Mitigation: V-SMOKE-LIVE first verifies model availability with a single call.

## 8. Round-2 corrections (applied)

- **C-1 → §4.2:** grammar relaxed to lookahead `(?=.*R\d+)`. Re-test on all cycle237 chids.
- **C-2 → §4.2:** add CHID_KIND with `anchored_to_history` variant (must contain `T\d+` token).
- **C-3 → §3 V-H3:** redefined as geometric in-bbox check.
- **C-4 → §4.5:** template_id retains direction suffix.
- **C-5 → §5 step 12:** split V-SMOKE into V-SMOKE-STUB (CI-fast) and V-SMOKE-LIVE (one TRAPI call).
- **C-6 → §4.8:** import LEAK_VOCAB from prompts.py at runtime, not duplicate.
- **C-7 → §5 step 9:** re-extract fixtures from BOTH cycle237 AND a recent cycle263 visible_regions snapshot.
- **C-8 → §5 step 12:** V-SMOKE-LIVE first call verifies gpt-5.4-pro availability.

## 9. Self-critic round 2 — findings

**C-9 (Two-tier introduces M1 prompt complexity).** §4.4 prompt is ~50 lines of new instructions. M1 might pick TIER-A always (easier) or TIER-B always (more freedom). Need a TIER BIAS knob to encourage diversity.
- Mitigation: in §4.4 prompt, add "If you have used TIER-A in 4 of last 5 turns, the 5th turn YOU MUST use TIER-B (or justify why no TIER-B fits)." Symmetric counter logic for TIER-B run-in.

**C-10 (V-H4 non-trivial-rate measures the WRONG thing).** A trivial TIER-B chid like `H_click_R5` (no direction, no relation) passes grammar but is just a TIER-A primitive. V-H4 should reject these.
- Mitigation: V-H4 accepts only TIER-B chids with EITHER (a) direction suffix, OR (b) cross-region rationale (regex `_shared_|_align_|_sweep_|_overlap_|_alignment_|_relation_`), OR (c) negation prefix (`static_|inert_|nonmarker_`).

**C-11 (M4 promotion threshold supported≥3 too lax).** With 400 turns × 3 cycles, many trivial templates might pass.
- Mitigation: `supported≥3 AND (supported - refuted) ≥ 2`. Beta-Bernoulli posterior expected≥0.6.

**C-12 (Detached launch needs keepalive).** §4.7 setsid+exec is correct but if TRAPI hangs >2min, the python process might have its own SIGINT handler. Need to confirm or override. Per memory `feedback_bsc_idle_suspend.md`: "BSC H200 idle-suspend ~17min; run gpu_keeper.py immediately before bootstrap finishes". Azure VM may have similar idle reaper.
- Mitigation: launcher script writes a heartbeat file every 60s; separate watchdog touches a noop GPU compute call if needed.

**C-13 (V-SMOKE-LIVE one-call cost).** One gpt-5.4-pro call ~60s. Three smoke calls (M1 stub + M1 live + M4 live) ~5 min. OK.

**C-14 (Cycle namespace collision).** `cycle264_v591_${RANDOM}` might collide with abandoned cycle262/263. Use UUID or timestamp.
- Mitigation: `cycle264_v591_$(date +%s)`.

## 10. Round-3 corrections (applied)

- **C-9 → §4.4 prompt:** add tier-diversity rule (5-turn sliding window).
- **C-10 → §4.2 + §3 V-H4:** non-trivial chid definition tightened.
- **C-11 → §4.5:** promotion threshold to `supported≥3 AND s-r≥2`.
- **C-12 → §4.7:** add heartbeat file write every 60s in launcher; document Azure VM idle behavior in §6 R-7.
- **C-14 → §4.7:** namespace uses `$(date +%s)`.

## 11. Self-critic round 3 — findings

**C-15 (Tier-diversity rule may force suboptimal moves).** "MUST use TIER-B if 4 of last 5 were TIER-A" can backfire if TIER-A genuinely fits. Soften to "consider TIER-B" + log a `tier_balance_score` for V-monitor.
- Applied: §4.4 prompt softened to "If you have used TIER-A in 4 of last 5 turns, REFLECT on whether TIER-B might offer a structural property you haven't tested yet; cite that reflection."

**C-16 (Cross-cycle invented_chid_history collision).** Multiple concurrent cycles writing same JSON.
- Applied: per-cycle scratch + post-cycle merge. §4.3 file `simple_logs/<game>/cycle<N>/invented_chid_history.json`; merge to `simple_logs/<game>/invented_chid_history.json` at cycle end.

**C-17 (No-leak scan on TIER-B does not check region-existence at chid emission).** prompts.py defers to agent.py; agent.py validates. OK; explicit in §4.3.

**C-18 ✓** No new structural issues. Plan FROZEN at round 3.

## 12. Final implementation order (canonical)

1. `tools/chid_grammar.py` (regex + leak scan + validate; relaxed grammar from §4.2).
2. `tests/test_v591_chid_grammar.py` 12 tests; run.
3. `agents/templates/agentica_v57/agent.py` snap-fallback cleanup (§4.3 REMOVE/KEEP).
4. `agents/templates/agentica_v57/agent.py` invented-chid validation + history persistence (§4.3 ADD).
5. `agents/templates/agentica_v57/prompts.py` two-tier selection block (§4.4) replacing L353-L392.
6. `agents/templates/agentica_v57/agent.py` model routing (§4.6).
7. M4 reflexion: invention promotion (§4.5).
8. `tests/test_v591_two_tier.py` 6 tests; run.
9. `tools/extract_v591_invention_fixtures.py` extract cycle237 + cycle263 chids.
10. `tests/test_v591_invention_grammar.py` 30+ fixtures; run.
11. V-LEAK + V-TESTS + V-FIXTURE; fix.
12. V-SMOKE-STUB (no TRAPI) then V-SMOKE-LIVE (3 TRAPI calls).
13. iterative-code-review smoke-critic-fix 3 rounds.
14. Detached launch cycle264-266 per §4.7 with namespace `cycle264_v591_$(date +%s)`.
15. Monitor V-H1..V-H6 over 400 actions × 3 cycles.

Plan v591 frozen at end of round 3. Implementation proceeds from §12 step 1.

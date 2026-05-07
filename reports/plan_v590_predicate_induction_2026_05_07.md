# Plan v590 — Symbolica Predicate Induction for M1 (B18)

**Status:** DRAFT round 1
**Date:** 2026-05-07
**Owner:** autoresearch
**Scope:** Replace v589 B17 typed-candidate roles with deterministic Python predicates over the trace. M1 selects which predicate-region pair to test (info gain). Symbolica derives the click coord from the predicate premise. M3 stays dormant. Architecture = M1 + Symbolica predicate generator/evaluator + M4 reflexion.
**Rejected if:** any leak vocabulary appears in predicate body or scoring output; coord derivation references future state or hidden labels; M1 textual compliance only without action consumption ≥0.4.

---

## 0. Diagnosis (why this plan exists — empirical)

v589 B17 ran cycle224-226 with full fixture-validated typed candidates:

- 257 candidates emitted across 3 cycles
- 203 verdicts resolved
- **0 supported (0.0%)**, 65 refuted (32%), 138 ignored (68%)
- L+1: 11 events across 3 cycles (vs ~36 at same point with v588 B16)
- L+2: **0** (unchanged — same ceiling)

Diagnosis: typed *roles* are too abstract for M1 to translate to specific click coords. The "knowing-doing gap" (codex round-1, GP-reviewer round-3) is unbridgeable when the agent is asked to invent the role-to-coord mapping. The role abstraction layer DELETES information that M1 needs. Even role anchor_marker_id doesn't cross the gap.

Codex round-1 + round-4 named this Risk-1 (HIGH) and reserved program induction as fallback. Round-5 pinned B18 design: predicates that RETURN matching regions (not booleans), with Symbolica deriving coords from the premise.

The architectural shift: **M1 no longer translates abstract concepts to clicks.** Symbolica gives M1 ready-to-execute predicate-region pairs; M1's only job is information-gain selection.

---

## 1. Survey of program induction methods

### 1.1 Inductive Logic Programming (ILP, Muggleton 1991+)
Induce Horn clauses from positive/negative examples. **Popper** (Cropper 2021) uses constraint-based search. **Metagol** uses meta-rules. Strength: provable correctness on first-order rules. Weakness: NP-hard search.

### 1.2 DreamCoder Wake-Sleep-Abstraction (Ellis 2021)
Solves tasks via program search; Sleep compresses observed solutions; Abstraction extracts reusable sub-routines. Strength: end-to-end compositional. Weakness: needs many tasks; we have ONE puzzle.

### 1.3 Bayesian Program Learning (Lake, Salakhutdinov, Tenenbaum 2015)
Programs as compositional structure with Bayesian priors. Posterior inference over program space. Foundation for our Beta-Bernoulli scoring.

### 1.4 FlashFill / DeepCoder (Microsoft / Balog 2017)
Synthesize programs from input-output examples. Lighter than full ILP. Useful for predicate body synthesis but not posteriors.

### 1.5 Neuro-symbolic concept learning (Mao 2019, Mao & Tenenbaum)
Combine neural perception with symbolic execution. Strength: handles raw observations. Weakness: too heavy; needs trained neural module.

### 1.6 Differential ILP (dILP, Evans & Grefenstette 2018)
Embeds ILP search in a differentiable network. Strength: gradient signal. Weakness: training cost.

### 1.7 LLM-driven program synthesis (Chen 2021, Codex)
LLM generates program text. Risk: leak vocabulary in generation. Codex round-2 rejected this for our setting.

### 1.8 Where B18 lands
B18 sits closest to Bayesian Program Learning (predicate priors + posteriors) + a restricted FlashFill-like enumeration. We do NOT use full ILP search (too slow) or LLM-driven synthesis (leak risk). Symbolica deterministically enumerates a bounded predicate library; posteriors update from action-conditioned trace observations.

---

## 2. Method candidates (theory-grounded, multiple distinct)

### 2.1 M-PRED-1 — Bounded enumerated predicate library (ADOPTED)

A fixed library of ~20 parametrised predicate templates. Symbolica instantiates each over current state's regions/markers, returning at runtime up to ~50 grounded predicates per turn. Each predicate `P(chain, t) → list[RegionRef]` returns matching regions.

**Distinction from B17:** B17 emits abstract role *names*. B18 emits *executable Python predicates that produce concrete RegionRef lists*. M1's gap-bridging job vanishes — Symbolica supplies the regions.

### 2.2 M-PRED-2 — LLM-generated predicates (REJECTED)

LLM emits predicate body as text. Round-2 codex flagged leak risk. Also: LLM-generated bodies are unverifiable; would need a sandbox + V-LEAK on each emission. Higher cost, lower trust.

### 2.3 M-PRED-3 — Differentiable predicate inducer (REJECTED for B18)

Train a small MLP that learns predicate compositions from trace. Reserved for B19+ if B18's bounded library is too rigid.

### 2.4 M-PRED-4 — Free-form Datalog rules (REJECTED)

Datalog adds inference engine + rule grammar overhead. Bounded Python is simpler.

### 2.5 Down-select rationale

Adopted: M-PRED-1.
Reason: deterministic, leak-safe by code-review, fast (no LLM in scoring loop), composable with existing chain payload. Round-5 codex agreed this is the correct B18 entry point.

---

## 3. Intent / Hypothesis / Verification (I/H/V)

### Intent

Bridge "knowing-doing gap" by **eliminating M1's role-to-coord translation step.** Symbolica predicates return concrete regions; M1 picks which predicate-region pair to test by information gain. Predicate posteriors accumulate evidence across cycles. Architecture simplifies to M1 + Symbolica + M4 (no M3, no role layer).

### Hypotheses (each falsifiable)

**H1 (Predicate generation works).** Generator emits ≥5 grounded predicates per turn at chain length ≥10. ≥3 distinct predicate templates per cycle.
**Falsifiable:** zero grounded predicates over a 50-turn window OR <2 distinct templates.

**H2 (M1 selects predicates by id).** ≥80% of M1 thoughts past turn 5 contain a `predicate_id` reference + `selected_region_ref`.
**Falsifiable:** <50% rate.

**H3 (Action-region match).** Click coord is INSIDE the bbox of the selected region in ≥80% of qualified turns.
**Falsifiable:** <50% match.

**H4 (Posterior signal accumulates).** ≥1 predicate reaches `supported_count ≥ 3` within 100 turns.
**Falsifiable:** all predicates stuck at supported=0/1.

**H5 (L+2 reach — integration target).** ≥1 of cycle231/232/233 reaches an L+2 event within 400 turns.
**Falsifiable:** zero L+2 across all three.

### Verification gates

| Gate | Method | Pass condition |
|---|---|---|
| V-LEAK | `python3 scripts/check_no_leak_prompts.py` + scan predicate library + scan emitted grounded predicates | 5/5 PASS, zero forbidden vocab |
| V-TESTS | `pytest tests/test_v590_*.py tests/test_v589_*.py tests/test_v588_*.py tests/test_v587_*.py tests/test_v586_*.py -v` | all PASS |
| V-FIXTURE | `tests/test_v590_predicate_coverage.py` train ≥80%, val ≥75%, gap ≤15pt | per project memory |
| V-H1 | trace scan: `predicate_generation` field per turn after t10 | ≥5 grounded |
| V-H2 | grep M1 thought | ≥80% qualified |
| V-H3 | trace scan: click coord vs selected_region_ref bbox | ≥80% |
| V-H4 | role_history scan | ≥1 predicate cfr≥3 within 100 turns |
| V-H5 | trace.jsonl `level_delta == 2` | ≥1 in 3 cycles |

V-H1, V-H2, V-H3 are wiring gates and must pass.
V-H4 measures functional posterior accumulation.
V-H5 is the integration target; failure with H1-H4 PASS triggers B19 escalation (differentiable predicate inducer).

---

## 4. Implementation specification

### 4.1 New file — `tools/predicate_generator.py`

Pure deterministic. No LLM. Whitelisted predicate library:

**Predicate template signature:**
```python
PredicateFn = Callable[[ChainState, int], list[RegionRef]]
```

`ChainState` = typed wrapper around `{visible_regions, recent_turn_diffs,
marker_neighbor_states, recent_clicks}`.

`RegionRef` = `{region_id, bbox, color, is_multicolor, kind}` — minimal subset for coord materialisation.

**Library (round-5 codex-aligned):**

```python
PREDICATE_LIBRARY: dict[str, PredicateFn] = {
    "P01_unclicked_neighbor_of_active_marker": ...,
    "P02_compass_changed_neighbor_revisit":     ...,
    "P03_marker_compass_uniform_breaker":       ...,  # find region whose click would BREAK uniformity
    "P04_marker_compass_uniformity_completer":  ...,
    "P05_shared_neighbor_between_markers":      ...,
    "P06_recently_l_plus_region_kin":           ...,
    "P07_repeat_click_parity_revert":           ...,
    "P08_unique_color_neighbor_of_marker":      ...,
    "P09_compass_changed_then_unchanged":       ...,
    "P10_marker_no_recent_compass_change":      ...,
    # … target ~12-20 templates total
}
```

Each template uses ONLY the whitelisted helpers:
```python
HELPERS = {
    "regions_at": lambda chain, t: [...],
    "marker_compass_state": lambda chain, t, marker_id: {...},
    "click_count_in_chain": lambda chain, region_id: int,
    "color_transition_at": lambda chain, t: {from, to} | None,
    "is_marker": lambda r: r.is_multicolor,
    "is_neighbor": lambda r1, r2: r2.region_id in r1.neighbors_3x3.values(),
    "compass_uniformity": lambda compass: all_same_color(compass),
}
```

**Generator entry:**
```python
def generate_predicates(
    *,
    chain_state: ChainState,
    turn_index: int,
    role_history: dict,
    k_global: int = 12,
) -> list[GroundedPredicate]:
    out = []
    for pid, fn in PREDICATE_LIBRARY.items():
        regions = fn(chain_state, turn_index)
        if not regions:
            continue
        for reg in regions[:3]:  # cap regions per predicate
            out.append({
                "predicate_id": f"{pid}:T{turn_index}:{reg.region_id}",
                "template_id": pid,
                "anchor_region": reg,                 # leak-safe — observable
                "score": _beta_bernoulli_score(pid, role_history),
                "coord_policy": "centroid",            # codex round-5
                "suggested_coord": _centroid(reg.bbox),
            })
    out.sort(key=lambda c: -c["score"])
    return out[:k_global]
```

**Score (codex round-5 Beta-Bernoulli):**
```python
def _beta_bernoulli_score(template_id, role_history):
    h = role_history.get(template_id, {})
    s = int(h.get("supported", 0))
    r = int(h.get("refuted", 0))
    return (1 + s) / (2 + s + r)   # alpha=beta=1
```

### 4.2 `agent.py` changes

**Replace** `tools.candidate_generator.generate_candidates` import with `tools.predicate_generator.generate_predicates`. (Keep B17 module file for backward-compat reading of old traces.)

`run_turn`:
```python
chain_state = build_chain_state(board, visible_regions, marker_neighbor_states)
new_preds = generate_predicates(
    chain_state=chain_state,
    turn_index=board.turn_index,
    role_history=board.role_history,
    k_global=_PRED_K_GLOBAL,
)
board.register_candidate_log(new_preds)   # reuse existing log infra
preds_for_m1 = board.serialised_candidates_for_m1()
```

`spawn_action` payload key renamed `candidate_tests` → `predicate_tests` (legacy `candidate_tests` kept readable for trace).

**Verdict update (codex round-5 action-conditioned)**:
```python
def update_predicate_log_with_observation(
    predicate_log, observation, click_region_id, turn_now
):
    for c in predicate_log:
        anchor_rid = c["anchor_region"]["region_id"]
        if click_region_id != anchor_rid:
            continue   # not action-conditioned to this predicate
        ld = obs.level_delta
        if ld >= 1 and turns_since <= 3: c.verdict = "supported"
        elif ld == 0 and turns_since >= 3: c.verdict = "refuted"
```

`role_history` is now `template_history` keyed by `template_id`.

### 4.3 `prompts.py` M1 changes

```
- PREDICATE_TESTS: list (≤12) of grounded predicate-region tests.
  Each entry:
    {predicate_id, template_id, anchor_region:{region_id, bbox, color},
     score, suggested_coord:[x,y], coord_policy}
  Symbolica has already SELECTED the region for you. Your only choice
  is which predicate to test next.

  BINDING — before choosing an action, you MUST:
    (a) select one PREDICATE_TEST by predicate_id;
    (b) state in your thought why testing this predicate is high
        information-gain (cite the score);
    (c) emit action coord = the predicate's suggested_coord (you may
        adjust ±2 pixels for sprite alignment, but stay inside
        anchor_region.bbox);
    (d) cite predicate_id explicitly in your thought.
  Do NOT invent a coord that doesn't match an entry's anchor_region.
```

### 4.4 Anti-leak

- Predicate library is FIXED at code level — V-LEAK extension scans `tools/predicate_generator.py` for forbidden vocab.
- Helper function names cannot include leak vocab.
- Per-emission audit: every predicate_id template prefix must be in `PREDICATE_LIBRARY` keys.
- No predicate body uses future-turn state.
- Coord derivation: only from current observable region geometry (bbox centroid).

### 4.5 Tests

`tests/test_v590_predicates.py` ≥14 tests:

- T-PRED-1..3: each template instantiates correctly on synthetic state.
- T-PRED-4: V-LEAK on PREDICATE_LIBRARY source.
- T-PRED-5: each emitted predicate_id template prefix ∈ library.
- T-PRED-6: anchor_region is observable (no future leak).
- T-PRED-7: suggested_coord ∈ anchor_region.bbox.
- T-PRED-8: Beta-Bernoulli score formula correct.
- T-PRED-9: empty input → empty list.
- T-PRED-10: per-turn cap = 12 enforced.
- T-PRED-11: action-conditioned verdict (clicked matching region only).
- T-PRED-12: M1 payload contains `predicate_tests` key.
- T-PRED-13: backward-compat — old traces with `candidate_tests` still load.
- T-PRED-14: no two predicate_ids same per turn.

`tests/test_v590_predicate_coverage.py` ≥30 fixtures from prior traces:
- Train coverage ≥80% (per fixture: ≥1 grounded predicate emitted, no leak)
- Val coverage ≥75%
- Gap ≤15pt

### 4.6 Backward compatibility

- B17 candidate_generator kept as legacy module (read-only).
- Trace fields `candidate_tests_emitted` etc. still parsable by visualizer.
- B18 writes `predicate_tests_emitted`, `predicate_verdicts_this_turn` alongside.
- visualizer renders both panels (legacy collapsed).

---

## 5. Implementation order

1. `tools/predicate_generator.py` — type defs (ChainState, RegionRef, GroundedPredicate).
2. `tools/predicate_generator.py` — helper functions (whitelisted set).
3. `tools/predicate_generator.py` — 12 predicate templates.
4. `tools/predicate_generator.py` — Beta-Bernoulli score + generate entry.
5. agent.py: build_chain_state helper.
6. agent.py: replace generate_candidates → generate_predicates.
7. agent.py: rename payload key + update_predicate_log_with_observation.
8. agent.py: template_history persistence.
9. prompts.py: PREDICATE_TESTS input doc + binding line.
10. tests/test_v590_predicates.py — 14 tests.
11. tools/extract_v590_fixtures.py — extract from prior traces.
12. tests/test_v590_predicate_coverage.py — ≥30 fixtures.
13. V-LEAK + V-TESTS gates.
14. Smoke 5-turn run.
15. Kill old cycles + launch cycle231-233.

After each step 1-13: run pytest. Do not proceed if regressing.

---

## 6. Risks and mitigations

- **R-1 (HIGH): predicate library too narrow.** The 12-20 templates miss the actual ft09 rule (XOR parity over compass). Mitigation: include high-coverage templates (P03/P04 about compass uniformity, P07 about repeat-click parity) — directly probing the rule shape.
- **R-2 (M-H): coord centroid never lands on the right pixel.** ft09 sometimes needs corner clicks. Mitigation: `coord_policy` field allows `centroid|interior_sample|corner` selection by template.
- **R-3 (M): overfitting predicates to trace quirks.** Codex round-5 raised this. Mitigation: minimum_support filter (template_history.supported ≥ 1 OR n_evals < 10) + complexity bound (no nested loops).
- **R-4 (M): coord leakage.** Coord includes anchor_region.bbox which is observable. No future state used. V-LEAK extended to per-emission scan.
- **R-5 (M): M1 textual compliance.** Same risk as B17. Mitigation: H3 metric (action-region bbox match) catches it; M4 marks ignored; same re-emit-up-to-3 pattern.
- **R-6 (L): backward-compat break.** Visualizer needs new section. Mitigation: §4.6 dual-rendering.

---

## 7. Self-critic round 1 — findings

**C-1 (Library size unbound).** §4.1 says "12-20 templates". Pin to exactly 12 (round numbers vs concrete). I'll lock at 12.

**C-2 (Score formula cold-start.)** Beta-Bernoulli with α=β=1 means every untested predicate scores 0.5. With 12 templates × 3 regions = 36 predicates per turn, M1 sees 12 ranked by score but all have score=0.5 at turn 0. M1 has no signal. Mitigation: add complexity-prior — simpler predicates start with α=2, β=1 (slightly favoured) so M1 has a deterministic tie-break.

**C-3 (Per-turn predicate count vs M1 prompt budget.)** 12 predicates × ~200 chars = 2.4KB. M1 prompt already at 14KB. +20% is OK.

**C-4 (Region cap 3 per predicate too tight or loose?)** With 5 markers visible, P05 (shared neighbor) might emit 0 or 5+ pairs. Cap=3 is arbitrary. Lock with rationale: 3 = balance between coverage and prompt bloat.

**C-5 (Action-conditioned verdict requires click_region_id == anchor_region_id.)** Strict match means click on a near-miss region (off by one bbox cell) doesn't count. Mitigation: also accept clicks INSIDE anchor_region.bbox even if obs.primary_region_id differs (which can happen due to game's region segmentation).

**C-6 (template_history persistence vs role_history namespace.)** B17's role_history.json schema reused or new? Mitigation: rename to `template_history.json` to avoid confusion.

**C-7 (V-FIXTURE coverage requires fixtures BEFORE launch.)** Per project memory `feedback_module_fixture_first.md`. Plan §4.5 has this. Implementation order §5 step 12 = before step 15 (launch). Correct.

**C-8 (No fallback for total predicate-emission failure.)** What if turn 5 has chain_state too sparse → 0 predicates? M1 has no input. Mitigation: emit a degenerate predicate "P00_explore_uncovered_marker" that always matches at least one marker if any visible.

**C-9 (M3 dormancy already in B17, kept here.)** Confirmed. ✓

**C-10 (R-1 HIGH risk on library coverage.)** If P03/P04/P07 don't capture XOR parity, B18 fails the same way as B17. Argues for B19 differentiable inducer earlier rather than waiting.

## 8. Round-2 corrections (applied)

**C-1 → §4.1 + §5:** lock at 12 templates. Names listed inline.
**C-2 → §4.1:** complexity-prior — simpler predicates (no compass joins) start α=2 β=1; complex predicates start α=1 β=1.
**C-4 → §4.1:** region cap 3 documented.
**C-5 → §4.2:** verdict logic accepts coord IN anchor_region.bbox OR primary_region_id == anchor_region.region_id.
**C-6 → §4.2:** rename to template_history.json.
**C-8 → §4.1:** add P00_explore_uncovered_marker as guaranteed fallback.

## 9. Self-critic round 2 — findings

**C-11 (P00 fallback and score weight).** P00 always returns ≥1 region. If its score is high, M1 always picks it → no exploration of other predicates. Mitigation: P00 score capped at 0.3 so it only wins when nothing else applies.

**C-12 (Helper whitelist enforcement.)** Templates can technically import other Python — code review only catches if reviewer is thorough. Mitigation: each template registered via decorator that AST-checks body imports. Or simpler: tests/test_v590_predicates.py T-PRED-15 imports the library and asserts each template is decorated.

**C-13 (Coord centroid for non-rectangular regions.)** ft09 multicolor markers are 8×8 with internal pattern. Centroid might miss the visually-obvious sprite. Mitigation: `coord_policy` enum {centroid, sprite_center, corner_top_left}; templates pick.

**C-14 (Backward-compat trace key naming.)** §4.6 says `predicate_tests_emitted` field. Old `candidate_tests_emitted` is also still emitted — visualizer renders both. Audit there's no field-name conflict.

## 10. Round-3 corrections

**C-11 → §4.1:** P00 score ≤ 0.3 cap.
**C-12 → §4.5:** T-PRED-15 — registry decorator audit.
**C-13 → §4.1:** coord_policy enum + per-template default.
**C-14 → §4.6:** confirm field names disjoint.

## 11. Self-critic round 3 — findings (final)

**C-15 ✓** No new issues. Plan FROZEN at round 3.

## 12. Final implementation order

1. `tools/predicate_generator.py` — type defs + register decorator (round-3 C-12).
2. `tools/predicate_generator.py` — 12 templates + P00 fallback (round-2 C-8 + round-3 C-11).
3. `tools/predicate_generator.py` — score + complexity-prior (round-2 C-2).
4. `tools/predicate_generator.py` — coord_policy enum (round-3 C-13).
5. simple_logs/<game>/template_history.json schema + race-safe persistence.
6. agent.py: build_chain_state.
7. agent.py: replace generator + payload + verdict logic (round-2 C-5 enriched).
8. agent.py: chain_rule_log pruning preserved; B17 candidate_log kept readable for legacy.
9. prompts.py: PREDICATE_TESTS doc + binding line.
10. tests/test_v590_predicates.py — 15 tests (T-PRED-1..15).
11. tools/extract_v590_fixtures.py — re-use B17 extractor pattern.
12. tests/test_v590_predicate_coverage.py — ≥30 fixtures train ≥80% / val ≥75% / gap ≤15pt.
13. V-LEAK + V-TESTS gates.
14. Smoke 5-turn with stub LLM.
15. Kill old cycles + launch cycle231-233 (budget 400).
16. Monitor V-H1..V-H5.

Plan v590 frozen at end of round 3. Implementation begins from §12 step 1.

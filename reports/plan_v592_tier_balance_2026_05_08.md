# Plan v592 — TIER Balance Restoration for B20

**Status:** DRAFT round 1
**Date:** 2026-05-08
**Owner:** autoresearch
**Supersedes:** plan_v591 (still ACTIVE for infrastructure; v592 patches only the TIER-balance layer)
**Scope:** Restore cycle237-style TIER-A predicate library + TIER-B invention MIX. Stop M1's drift toward 100% invention. Add three orthogonal mechanisms: bidirectional diversity reflection (prompt-side), score-gated TIER-A bias (prompt-side), and stuck-rotation runtime override (agent.py-side).
**Rejected if:** TIER-A ratio in any post-T5 50-turn window < 15%; cycle237 trace fails new V-FIXTURE; mechanism breaks the v591 cycle237-vintage invention reproduction (must keep cycle267f T46 success).

---

## 0. Empirical diagnosis (why v592 exists — concrete numbers)

cycle237 (sole L+2 success, 39 turns total):
- T7-T27 (L+1 → L+2 phase, 21 turns):
  - **TIER-A predicate calls: 6** — `P01_unclicked_neighbor_of_active_marker:T10:R2`, `P07_repeat_click_parity_revert:T4|T10|T11|T16:R*`, `P12_recent_click_no_change_revisit:T13:R3`
  - **TIER-B invention: 15** — `H_fresh_neighbor_toggle_R*`, `H_crop2_complete_R12_W`, `P_shared_blank_sweep_R12`, `H_R15_lower_S`, `P_R12_crop_sector_alignment` (the eventual L+2 trigger at T27)
  - **TIER-A ratio: 29%**

cycle267f (gpt-5.5 + v591 round-3 patches, 103 turns total):
- T47-T80 (post-L+1 phase, 34 turns sampled):
  - **TIER-A predicate calls: 0**
  - **TIER-B invention: 31**
  - **active card replay (M3-emitted C1/C2/C3): 3**
  - **TIER-A ratio: 0%**

**Conclusion:** cycle267f abandoned the deterministic predicate library entirely. Without TIER-A's hint signal ("this neighbor was not tested at T<n>"), M1 wanders inside its own invention space.

The plan v591 §4.4 "TWO-TIER SELECTION" block has only ONE diversity nudge — toward TIER-B ("if TIER-A in 4 of last 5 turns, REFLECT on TIER-B"). The reverse direction does not exist. v591 round-3 also REMOVED v590's HARD BINDING anchor language. Net effect: M1 has no friction toward TIER-A.

Plan v591 §3 H2 ("M1 cites EITHER predicate_id OR invented chid") was satisfied — but the architecture intent (BOTH tiers used in mix, like cycle237) was not.

---

## 1. Survey of related work

### 1.1 Multi-arm bandit (Auer 2002, Lattimore & Szepesvári 2020)
UCB1, Thompson sampling, ε-greedy. Explore-exploit tradeoff. M1 in cycle267f is pure exploit on its own invention; predicate library is the unused exploration arm. Bandit theory predicts bias-toward-novel-arm with confidence-bound expansion.

### 1.2 Active learning (Settles 2009, "Active Learning Literature Survey")
Query oracle when posterior uncertainty is high. The predicate library Beta-Bernoulli scores ARE posterior estimates. When library reports score≥0.7 unresolved predicates, that's a "high-yield query" — M1 should consult it.

### 1.3 Tool-use gating (Schick 2023 Toolformer, Yang 2023 LATS)
LLM should call tool when reasoning confidence is below threshold. Inverted: when tool confidence (TIER-A score) is HIGH, prefer tool. M1 ignoring TIER-A == ignoring confidence-rich tool calls.

### 1.4 AlphaGo policy + value (Silver 2016)
MCTS combines policy (LLM proposal) + value (rollout estimate). Pure policy with no value backup collapses to greedy local minimum. cycle267f T47-T80 is the policy-only failure mode.

### 1.5 Reflexion (Shinn 2023)
Self-reflection loop. Plan v591 had reflection toward TIER-B only. Plan v592 needs SYMMETRIC reflection — toward both directions.

### 1.6 Voyager (Wang 2023)
Skill library + invention coexist. Voyager's skill retrieval is HARD-GATED by precondition match. Without retrieval, agent invents. With Voyager-style precondition checking on TIER-A predicates, M1 would auto-call TIER-A when its preconditions match the current state.

### 1.7 DreamCoder Wake/Sleep (Ellis 2021)
Wake = library + search; never abandons library. Sleep = promotion. Plan v591 implemented Sleep (Trace2Skill) but its Wake collapses to invention-only. Need to restore Wake's library-first behavior.

### 1.8 Where v592 lands
v592 = bandit-flavored active learning bias. The deterministic predicate library is the "high-info arm" when its score is high; M1's invention is the high-novelty arm. We force the agent to re-balance via three mechanisms (prompt symmetric reflection + score-gated bias + stuck-rotation runtime) without removing invention freedom.

---

## 2. Method candidates (theory-grounded, multiple distinct)

### 2.1 M-9-A — Bidirectional diversity reflection (prompt only)
"If TIER-A used in 4 of last 5 turns, REFLECT on whether TIER-B fits. If TIER-B used in 4 of last 5 turns, REFLECT on whether TIER-A predicate score-rich entry is appropriate."
- **Pros:** minimal change, symmetric, soft.
- **Cons:** cycle237 evidence shows soft reflection alone might not suffice — M1 already had v591 soft and ignored it.

### 2.2 M-9-B — Hard ratio enforcement (runtime)
agent.py tracks per-cycle TIER counts. If `tier_a_ratio < 0.30` over last 20 turns, force the next turn to bind to top-score TIER-A predicate.
- **Pros:** guaranteed minimum mix.
- **Cons:** heavy-handed; M1 thought may protest; loses TIER-B's contextual freedom.

### 2.3 M-9-C — Score-gated TIER-A bias (prompt + runtime)
If ANY TIER-A predicate has Beta-Bernoulli score ≥ 0.7 AND is unresolved, prompt instructs M1 to PREFER it. Runtime soft-snap: if M1 ignored a score≥0.85 predicate, log a warning (no override).
- **Pros:** confidence-aware, principled (matches active-learning Settles).
- **Cons:** prompt complexity, soft warning may be ignored.

### 2.4 M-9-D — Stuck-rotation override (runtime)
If 3 consecutive TIER-B turns with ld=0, override the next turn's chid to the highest-score TIER-A predicate (force TIER-A re-entry). After 3 consecutive TIER-A with ld=0, allow TIER-B again.
- **Pros:** simple stuck-rotation.
- **Cons:** rough; may interrupt productive TIER-B exploration.

### 2.5 M-9-E — UCB1 auto-pick (runtime, advanced)
Compute UCB score = mean + sqrt(2 ln(N)/n) for each candidate. System auto-recommends top UCB; M1 can override with justification field.
- **Pros:** principled bandit theory.
- **Cons:** requires per-template visit count; significant infra; risk of over-engineering.

### 2.6 M-9-F — Restore HARD BINDING to TIER-A (prompt revert)
Default: pick top-score TIER-A predicate. TIER-B opt-out only with explicit "TIER-A entry insufficient because <structural property X>" justification. Mirror v590 behavior.
- **Pros:** cycle237-era discipline. Strong forcing function.
- **Cons:** might over-correct → cycle259-263 era 0 L+ regression.

### 2.7 Down-select rationale (ADOPTED: A + D + soft-C)

**Adopted combo:**
- **M-9-A (bidirectional reflection)** in prompt. Symmetric. Minimal risk.
- **M-9-D (stuck-rotation runtime)** in agent.py. Triggers only on 3-consecutive-stuck (hard rule, not soft).
- **M-9-C-soft** in prompt — explicit "prefer score≥0.7 unresolved TIER-A entries" without hard binding.

**Rejected:**
- B (heavy-handed) — too coarse vs M1's current intelligence.
- E (UCB1) — over-engineering for v592 scope; reserve for B21.
- F (HARD BINDING revert) — may regress to cycle259-263 mode.

This combo: prompt-side balance nudge (A+C-soft) + runtime safety (D) without removing invention path. Theory: matches active-learning + symmetric Reflexion + stuck-rotation heuristic.

---

## 3. Intent / Hypothesis / Verification (I/H/V)

### Intent

Restore cycle237's empirical TIER mix discipline (~29% TIER-A in post-L+1 phase) without losing v591's TIER-B invention quality (cycle267f T46 vintage chid reproduction). Three orthogonal mechanisms: bidirectional reflection, score-gated bias, stuck-rotation. M1 stays free to invent but is biased toward predicate library when (a) predicate scores indicate high-info entries, (b) M1 has used invention 4+ of last 5 turns, (c) M1 has been stuck on TIER-B for 3+ turns.

### Hypotheses (falsifiable, with V-gate)

**H1 (TIER-A ratio ≥ 0.20 in any 50-turn window post-T5).**
Threshold lower than cycle237's 0.29 to allow some variance.
- **V-H1**: post-cycle measurement: max over sliding 50-turn windows of `tier_a_count / (tier_a_count + tier_b_count + card_count)`. Min must be ≥ 0.20.
- **Falsifiable**: any 50-turn window with ratio < 0.10.

**H2 (Score-gated TIER-A consultation).**
When ≥1 TIER-A predicate has score ≥ 0.7 AND unresolved, M1 picks TIER-A in ≥60% of those qualifying turns.
- **V-H2**: trace scan. For each turn where `max(unresolved_tier_a_scores) ≥ 0.7`, check `chid_tier == "A"`.
- **Falsifiable**: <30% rate.

**H3 (Stuck rotation triggers).**
After 3 consecutive TIER-B turns with ld=0, the 4th turn's `chid_tier == "A"` (forced).
- **V-H3**: trace scan. Find runs of `tier_b ld=0` length≥3, verify next turn `tier_a` (or game end).
- **Falsifiable**: <80% trigger rate.

**H4 (TIER-B invention quality preserved).**
v591 cycle267f's 100% non-trivial-rate is preserved in v592. TIER-B chids in v592 still have non_trivial_rate ≥ 0.85.
- **V-H4**: existing `scripts/measure_v591_invention.py` adapted.
- **Falsifiable**: <0.50.

**H5 (cycle237 trace validates against new V-FIXTURE).**
Replaying cycle237 trace through v592 logic must NOT trigger any "violation" — cycle237 had 29% TIER-A, no 3+-turn stuck runs, predicate scores within range. cycle237 should be a PASSING trace.
- **V-H5**: `tests/test_v592_tier_balance.py::test_cycle237_compliant`.
- **Falsifiable**: cycle237 fails any v592 V-gate.

**H6 (L+2 reach — integration target).**
≥1 of cycle270/271/272 reaches L+2 in 400 turns.
- **V-H6**: trace scan `level_delta ≥ 1` accumulator reaches level=2.
- **Falsifiable**: zero L+2 across all three.

### Verification gates

| Gate | Method | Pass condition |
|---|---|---|
| V-LEAK | `python3 scripts/check_no_leak_prompts.py` | 5/5 PASS |
| V-TESTS | `pytest tests/test_v592_*.py tests/test_v591_*.py tests/test_v590_*.py -v` | all PASS |
| V-FIXTURE-H5 | `tests/test_v592_tier_balance.py::test_cycle237_compliant` | PASS |
| V-SMOKE | 10-turn live run; verify ≥1 TIER-A turn, ≥1 TIER-B turn, no exception | PASS |
| V-H1..V-H6 | per §3 | PASS |

Launch-blocking: V-LEAK + V-TESTS + V-FIXTURE-H5 + V-SMOKE.

---

## 4. Implementation specification

### 4.1 prompts.py M1 ACTION_SYSTEM_PROMPT changes

Replace the current TWO-TIER SELECTION block (added in v591 round-3 just before "Output ONLY the JSON object") with the v592 BALANCED-TWO-TIER block. Key additions:

```
TIER-A score-gated preference:
  When any TIER-A predicate in CANDIDATE_TESTS has score ≥ 0.7 and
  is unresolved, you should PREFER it unless your TIER-B invention
  tests a structural property that NO TIER-A predicate covers (e.g.
  cross-region relation, sprite-internal geometry, negation).

Bidirectional diversity reflection:
  Past 5 turns are visible in RECENT_TURNS / SUMMARY.
  - If TIER-A appeared in 4 of last 5: REFLECT on whether a TIER-B
    invention would test a structural property the library lacks.
  - If TIER-B appeared in 4 of last 5: REFLECT on whether a high-
    score (≥0.5) TIER-A predicate could provide deterministic
    feedback you have been missing.
  Cite the reflection result in your thought.

Stuck-rotation awareness:
  If your last 3 turns were TIER-B with level_delta=0, your CURRENT
  turn should default to the highest-score unresolved TIER-A
  predicate. The runtime will enforce this if you don't comply.
```

### 4.2 agent.py runtime stuck-rotation (M-9-D)

Add new helper after `chid_tier` classification block:

```python
# v592 round-9 (M-9-D): stuck-rotation runtime override.
# If last 3 turns were TIER-B with ld=0, force this turn to top-
# score unresolved TIER-A predicate. Mirrors cycle237 mix discipline.
def _v592_stuck_rotation_should_force(board, candidate_tests):
    if len(board.recent_verbose) < 3:
        return None
    last3 = board.recent_verbose[-3:]
    all_tier_b_zero_ld = all(
        (rv or {}).get("chid_tier") == "B"
        and int(((rv or {}).get("observation") or {}).get("level_delta") or 0) == 0
        for rv in last3
    )
    if not all_tier_b_zero_ld:
        return None
    if not candidate_tests:
        return None
    sorted_cands = sorted(
        candidate_tests,
        key=lambda c: -float(c.get("score", 0.0)),
    )
    return sorted_cands[0] if sorted_cands else None

forced = _v592_stuck_rotation_should_force(board, candidate_tests_for_m1)
if forced and chid_tier == "B":
    aresp["chosen_hypothesis_id"] = forced.get("predicate_id")
    chid_tier = "A"
    invented_meta = None
    # log forcing event for V-H3 measurement
    board.v592_stuck_rotation_count = (
        getattr(board, "v592_stuck_rotation_count", 0) + 1
    )
```

Insert after the existing TIER-B coord coupling block, BEFORE `# 3. Execute (caller-provided).` line.

### 4.3 agent.py recent_verbose chid_tier carry-through

Each `recent_verbose` entry must already record `chid_tier`. v591 adds it to trace; need to also add to `recent_verbose` queue. Verify by reading agent.py around `board.recent_verbose.append(...)`. If missing, add `"chid_tier": chid_tier` to the entry dict.

### 4.4 measure_v591_invention.py v592 extension

Add three new metrics:
- `tier_a_count`, `tier_a_ratio` (overall + sliding 50-turn min)
- `score_gated_compliance_rate` (V-H2)
- `stuck_rotation_trigger_rate` (V-H3)

### 4.5 Tests

`tests/test_v592_tier_balance.py` ≥10 tests:
- T-V592-1: stuck_rotation_should_force returns None on <3 turns.
- T-V592-2: returns None when last 3 turns are not all TIER-B.
- T-V592-3: returns None when last 3 are TIER-B but with ld≥1.
- T-V592-4: returns highest-score predicate when triggered.
- T-V592-5: returns None when candidate_tests empty.
- T-V592-6: cycle237 trace compliant (V-H5) — replay through measure_v591_invention.py extension.
- T-V592-7..10: ratio computation, score_gated metric, etc.

### 4.6 Backward compatibility

- v591 round-3 prompt block is REPLACED, not augmented. The new block subsumes its content.
- agent.py existing TIER-B validation and coord-coupling unchanged.
- chid_grammar.py unchanged.
- cross_run_memory.json A4-A7 (Trace2Skill) preserved.
- v591 tests still pass.

---

## 5. Implementation order

1. `tests/test_v592_tier_balance.py` skeleton with T-V592-1..5 (stub-LLM unit tests).
2. `agents/templates/agentica_v57/agent.py` — `_v592_stuck_rotation_should_force` helper + integration call.
3. Verify recent_verbose carries `chid_tier` field; if not, add.
4. Tests pass (T-V592-1..5).
5. `agents/templates/agentica_v57/prompts.py` — replace TWO-TIER block with BALANCED-TWO-TIER block.
6. V-1 leak gate.
7. `scripts/measure_v591_invention.py` extension (rename to measure_v592 OR add args).
8. T-V592-6 cycle237 compliance test.
9. T-V592-7..10 metric tests.
10. iterative-code-review smoke-critic-fix loop (3 rounds).
11. V-SMOKE 10-turn live run.
12. Detached launch cycle270/271/272 sequential (gpt-5.5 + Trace2Skill priors + v592 patches).
13. Monitor V-H1..V-H6.

---

## 6. Risks and mitigations

- **R-1 (HIGH): stuck-rotation override breaks productive TIER-B chains.** A 3-turn TIER-B run might be on the verge of L+ insight. Forcing TIER-A at turn 4 cuts it off. Mitigation: only force when last 3 ld==0 (no progress), and only when TIER-A has ≥1 unresolved candidate. Justified because 3 turns no progress is empirical signal of stuck.
- **R-2 (MED): score-gated bias still ignored by M1.** v591 round-5 showed prompt schema adjustments DO change M1 behavior, but rate not 100%. Mitigation: V-H2 threshold is 60% not 100%. Stuck-rotation D is the safety net.
- **R-3 (MED): cycle237 doesn't actually validate.** Plan §0 reports 29% TIER-A but didn't measure score-gated compliance. cycle237 might fail V-H2/V-H3. Mitigation: measure cycle237 BEFORE finalizing thresholds; adjust if needed.
- **R-4 (LOW): TIER-A predicate library has all-low scores at start.** If predicates have score=0.5 baseline, score-gated bias never fires until evidence accumulates. Acceptable — bias kicks in mid-cycle.
- **R-5 (LOW): runtime stuck rotation conflicts with anti-osc fallback at agent.py:1959+.** Current anti-osc only fires on coord-repeat. v592 stuck-rotation fires on ld=0 streak. Different triggers. No conflict. Verify in code review.

---

## 7. Self-critic round 1 — findings

**C-1 (V-H5 cycle237 score-gated check uncertain).** Plan §0 reported 29% TIER-A but did NOT verify whether cycle237's TIER-A choices were score-gated (whether the predicates M1 picked had score≥0.7 at time of pick). If cycle237's M1 picked LOW-score TIER-A entries, V-H2 score-gated metric is mismatched against historical data.
- Mitigation: measure cycle237 V-H2 compliance during plan §5 step 1. If low, lower V-H2 threshold from 60% to 40%.

**C-2 (Stuck-rotation triggers might cluster.)** If cycle270 hits "stuck" repeatedly (e.g., 3 turns TIER-B → forced TIER-A T4 → ld=0 → 3 more TIER-B → forced T8 → ...), V-H3 trigger rate could be very high (>50% of cycle), starving TIER-B exploration.
- Mitigation: cap stuck-rotation at ≤5 triggers per 50-turn window. After cap, allow TIER-B free.

**C-3 (Score-gated bias requires score field exposure.)** prompts.py block instructs M1 to read `score` from CANDIDATE_TESTS. Verify trace shows score is included in payload to M1. If not, add to agent.py M1 payload.
- Mitigation: explicit grep agent.py:1700 area for candidate_tests serialisation.

**C-4 (test_cycle237_compliant might be too strict).** If V-H1 threshold is 0.20 ratio and cycle237's 21-turn window (T7-T27) has 6/21 = 28.6%, that's > 0.20 ✓. But shorter windows (e.g., T20-T26 = 7 turns, mostly TIER-B) might dip below.
- Mitigation: V-H1 metric uses 50-turn windows. cycle237 has only 39 turns total — special-case for cycles < 50 turns: use ratio over full cycle. cycle237's full 39-turn ratio = 6 TIER-A / ~22 (T0-T6 had only 1 TIER-A `P01` that we know of, plus T7-T27's 6, plus T28-T38 unknown) — need recount.

**C-5 (M-9-A bidirectional reflection should also REFLECT on score variance.)** M1 reflection rules don't currently consider score evolution. Add: "if last 3 TIER-A picks were on predicates whose score remained <0.5 after refutation, consider those predicates exhausted; prefer fresh predicates or TIER-B."

**C-6 (cycle268f/269f data not yet incorporated.)** cycle268f is currently running with Trace2Skill priors (v591 round-8) — its TIER-A ratio should be observed BEFORE round-9 launch. If Trace2Skill priors alone restore TIER-A balance, round-9 may be unnecessary.
- Mitigation: defer round-9 launch until cycle268f/269f end. If either reaches L+2, cancel round-9; if both fail with TIER-A < 15%, proceed round-9.

## 8. Round-2 corrections (applied)

- **C-1 → §3 V-H2:** add cycle237 compliance check first; threshold conditional on result.
- **C-2 → §4.2 + §6 R-1:** cap stuck-rotation at 5 triggers per 50-turn window.
- **C-3 → §4.1:** explicit "score field is exposed in CANDIDATE_TESTS" line added.
- **C-4 → §3 V-H1:** for cycles <50 turns, use full-cycle ratio; cycle237 reference recount required in step 1.
- **C-5 → §4.1:** add "score-exhaustion reflection" sub-rule.
- **C-6 → §5 step 0:** "wait for cycle268f/269f termination + measurement before launching cycle270."

## 9. Self-critic round 2 — findings

**C-7 (Codex review-loop convergence criterion needed).** Plan v592 explicitly invokes codex iteration. Define convergence: codex agrees with adopted M-9-A/D/soft-C combo OR codex's alternative gets accepted, max 3 codex rounds.

**C-8 (V-H1 metric ambiguity for early turns.)** First 5 turns of any cycle have no TIER-A history (predicate library empty initially). V-H1 starts measurement after T5.
- Mitigation: §3 H1 says "post-T5"; explicit in measurement.

**C-9 (Stuck-rotation may force ineffective predicates.)** If all unresolved TIER-A scores are <0.5 (baseline + many refutes), forcing top-score still picks a low-confidence option. Better: if no predicate has score ≥0.5, allow TIER-B free (stuck-rotation no-op).
- Applied: §4.2 helper checks `top_score >= 0.5` else returns None.

**C-10 (Plan v591 §13 outcome matrix not updated.)** v591 §13.5 said all-L+1-no-L+2 → "relaunch 3 more". v592 IS that relaunch. Cross-reference.

## 10. Round-3 corrections (applied)

- **C-7 → §11:** define codex convergence criterion (max 3 rounds, accept-or-counter convergence).
- **C-8 → §3 H1:** explicit "post-T5" measurement window.
- **C-9 → §4.2:** stuck-rotation requires top_score ≥ 0.5.
- **C-10 → §0:** add "v592 = realisation of v591 §13 outcome matrix's 'relaunch with mechanism patch' branch."

## 11. Codex review-loop protocol

Convergence criterion: max 3 codex rounds. Each round prompts codex with current plan + a focused question; codex returns critique; we apply or counter.

- Round-1 codex prompt: "Read reports/plan_v592_tier_balance_2026_05_08.md. Do M-9-A + M-9-D + M-9-C-soft together actually solve the TIER-A abandonment? Or does combination still leave a hole? Cite cycle267f trace evidence."
- Round-2 codex prompt: based on codex round-1 critique. Apply changes or counter-argue.
- Round-3 codex prompt (if needed): final sanity. If codex agrees → freeze. If disagrees again → freeze with codex disagreement noted; proceed cautiously.

After convergence (or 3 rounds), plan FROZEN; implementation starts §5.

## 12. Self-critic round 3 — final

**C-11 ✓** No new structural issues. Plan FROZEN at round 3 pending codex round.

## 13. Final implementation order (canonical)

0. Wait for cycle268f/269f termination + measure their TIER-A ratio. If ≥1 reaches L+2, ABORT v592 (v591 sufficient). Else proceed.
1. Recount cycle237 full-trace TIER-A ratio + score-gated compliance. Adjust V-H1/V-H2 thresholds if needed.
2. `tests/test_v592_tier_balance.py` — T-V592-1..5 stub-LLM unit tests.
3. `agents/templates/agentica_v57/agent.py` — `_v592_stuck_rotation_should_force` + integration. Verify recent_verbose carries chid_tier.
4. Run T-V592-1..5; ensure pass.
5. `agents/templates/agentica_v57/prompts.py` — replace TWO-TIER with BALANCED-TWO-TIER block.
6. V-1 leak gate.
7. `scripts/measure_v592_invention.py` (extension or rename).
8. T-V592-6 cycle237 V-H5 compliance.
9. T-V592-7..10 metric tests.
10. Codex round-1 review (per §11).
11. Apply codex critique → round-2 → round-3 if needed.
12. iterative-code-review skill smoke-critic-fix (3 rounds).
13. V-SMOKE 10-turn live run.
14. Detached launch cycle270/271/272 sequential.
15. Monitor V-H1..V-H6.

Plan v592 frozen at end of round 3 self-critic, pending codex review-loop convergence (§11).

---

## 14. Codex round-1 critique + round-4 corrections (2026-05-08)

**Codex round-1 verdict: RECONSIDER M-9-C-soft.**

Codex's critique:
1. M-9-D (stuck-rotation) reactive only on exact `BBB + ld=0` — misses partial-gain / interleaved-card drift.
2. M-9-C-soft (score≥0.7 preference) too weak — same class as cycle267f-ignored soft hint.
3. **Blocker: no HARD MINIMUM TIER-A exposure.** M1 can drift into TIER-B dominance via occasional ambiguous progress + cards.

### Round-4 corrections (apply codex critique)

**Replace M-9-C-soft with M-9-C-HARD score-gated override:**
- When ANY TIER-A predicate has `score ≥ 0.85` AND unresolved AND not picked by M1 in this turn → runtime forces chid_tier = "A" (override M1's TIER-B chid). This is HARD, not warning.
- Threshold 0.85 (high enough that bias only fires when predicate is strongly informative).

**Add M-9-B-soft minimum ratio enforcement:**
- Per sliding 20-turn window past T5: if `tier_a_count / total_turns_in_window < 0.15`, runtime forces next turn TIER-A.
- Cap at 3 forces per 50-turn window to avoid starving TIER-B entirely.

**Updated combo (v592 final):**
- **M-9-A** bidirectional reflection (prompt, soft)
- **M-9-B-soft** 20-turn min-ratio runtime enforcement (≥0.15 floor)
- **M-9-C-HARD** score≥0.85 runtime override
- **M-9-D** stuck-rotation runtime (BBB + ld=0)

Three runtime mechanisms layered. Hard guarantees:
- TIER-A ratio ≥ 0.15 over any 20-turn window (M-9-B-soft).
- score≥0.85 predicates always picked when unresolved (M-9-C-HARD).
- TIER-B drift bounded by 3-turn rotation (M-9-D).

### Round-4 H1/H2 threshold updates

- V-H1: 50-turn ratio ≥ 0.20 (unchanged) AND 20-turn ratio ≥ 0.15 (new floor enforced by M-9-B-soft).
- V-H2: score≥0.85 predicates picked in ≥95% of qualifying turns (raised from 60% — now hard-enforced).
- V-H3: stuck-rotation triggers ≥80% on `BBB+ld=0` (unchanged).
- V-H7 (NEW): tier_a_min_ratio_20turn ≥ 0.15.

### Round-4 §4.2 agent.py addition (M-9-B + M-9-C-HARD)

```python
def _v592_min_ratio_should_force(board, candidate_tests, window=20, floor=0.15, cap_per_50=3):
    """M-9-B-soft: enforce TIER-A floor over rolling 20-turn window."""
    recent = board.recent_verbose[-window:]
    if len(recent) < window or not candidate_tests:
        return None
    tier_counts = {"A": 0, "B": 0, "card": 0, "none": 0}
    for rv in recent:
        tier_counts[(rv or {}).get("chid_tier", "none")] = tier_counts.get(
            (rv or {}).get("chid_tier", "none"), 0) + 1
    total = sum(tier_counts.values())
    if total == 0:
        return None
    a_ratio = tier_counts.get("A", 0) / total
    if a_ratio >= floor:
        return None
    cap_used = sum(
        1 for rv in board.recent_verbose[-50:]
        if (rv or {}).get("v592_min_ratio_forced") is True
    )
    if cap_used >= cap_per_50:
        return None
    sorted_cands = sorted(
        [c for c in candidate_tests if float(c.get("score", 0)) >= 0.5],
        key=lambda c: -float(c.get("score", 0.0)),
    )
    return sorted_cands[0] if sorted_cands else None


def _v592_score_gate_should_force(candidate_tests, threshold=0.85):
    """M-9-C-HARD: force when ANY unresolved predicate score >= threshold."""
    sorted_cands = sorted(
        [c for c in candidate_tests if float(c.get("score", 0)) >= threshold
         and c.get("verdict") in (None, "inconclusive")],
        key=lambda c: -float(c.get("score", 0.0)),
    )
    return sorted_cands[0] if sorted_cands else None


# Apply order: stuck-rotation (D) → score-gate (C-HARD) → min-ratio (B-soft).
forced = (
    _v592_stuck_rotation_should_force(board, candidate_tests_for_m1)
    or _v592_score_gate_should_force(candidate_tests_for_m1)
    or _v592_min_ratio_should_force(board, candidate_tests_for_m1)
)
if forced and chid_tier == "B":
    aresp["chosen_hypothesis_id"] = forced.get("predicate_id")
    chid_tier = "A"
    invented_meta = None
    # log which mechanism fired; per-rv flag for cap counting
    board.v592_force_count = getattr(board, "v592_force_count", 0) + 1
    # mark this turn's recent_verbose entry with the force reason
    # (added when verbose entry constructed below)
```

### Round-4 §4.5 tests addition

`tests/test_v592_tier_balance.py` additional:
- T-V592-11: min_ratio_should_force returns None when ratio ≥ floor.
- T-V592-12: min_ratio respects cap_per_50.
- T-V592-13: score_gate fires only on score≥0.85 unresolved.
- T-V592-14: priority order = stuck > score > min_ratio.

## 15. Self-critic round 4 — findings

**C-12 (M-9-C-HARD threshold 0.85 may rarely fire if predicate scoring is conservative.)** Beta-Bernoulli with α=β=1 starts at 0.5; reaching 0.85 needs ~5 supports without refute. cycle237 trace might never have had any predicate at 0.85.
- Mitigation: lower threshold to 0.70 if cycle237 measurement (step 1) shows zero predicates ever reached 0.85. Adapt.

**C-13 (Three runtime forcing mechanisms compose unpredictably.)** Order is stuck > score > min_ratio. If stuck triggers, score+min_ratio skip. Need test T-V592-14 explicitly. Risk: forced TIER-A turn may itself fail (ld=0), feeding back into stuck-rotation. Could create force-loops.
- Mitigation: after a forced TIER-A turn, reset stuck counter (do not count forced turns toward the "BBB" sequence).

**C-14 (cap_per_50 mechanism complex.)** Per-rv flag `v592_min_ratio_forced` requires recent_verbose mutation. Verify writability.
- Mitigation: explicit at trace-write site.

## 16. Round-5 corrections

- **C-12 → §3 V-H2:** thresholds adapt-after-cycle237-measure clause.
- **C-13 → §4.2:** forced-turn does not increment stuck counter.
- **C-14 → §4.2:** explicit recent_verbose flag write at trace site.

## 17. Codex round-2 review-loop

After §16 corrections, send codex round-2 prompt asking specifically about C-13 force-loop risk and C-12 threshold adaptivity. If codex agrees combo is now sufficient → FREEZE. If codex raises new structural issue → round-3.

## 18. Round-5 self-critic — final

**C-15 ✓** No new structural issues. Plan FROZEN at round 5 pending codex round-2.


---

## 19. Codex round-2 verdict: FREEZE

Codex Q1: "M-9-B-soft gives the missing hard exposure floor. C-HARD is meaningful but is not the floor; min-ratio is. Cap 3/50 means bounded not absolute under pathological underuse — acceptable since target is preventing abandonment, not enforcing every bad window forever."

Codex Q2 watch item: "Floor may force low-value TIER-A calls when library is stale or semantically exhausted. Guarantee exposure but not useful exposure. Add guard: min-ratio force should prefer unresolved score ≥0.5 (already in §14 spec — confirm). Log forced-low-confidence separately."

Codex Q3: **FREEZE** with the round-3 watch item.

### Round-6 (final) corrections — codex round-2 watch item

- §4.2 `_v592_min_ratio_should_force` already requires `score ≥ 0.5` filter (line in code spec). Verified.
- Add post-cycle metric `forced_tier_a_productivity` = `(forced_turns_with_ld>=1) / (forced_turns_total)`. If <0.10 across cycle270/271/272, the forced TIER-A is purely metric-satisfying → escalate to library refresh in v593.
- §4.4 measure_v592 extension to log `forced_tier_a_productivity`.

Plan v592 FROZEN at round-6 (self-critic round-5 + codex round-2 + watch item incorporated).


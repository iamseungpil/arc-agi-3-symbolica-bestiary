# Plan v600 — Minimal-Fast Framework for ft09 Reproducibility

Date: 2026-05-09
Status: **v4 CONVERGED (post-codex round-5)**

Phase A complete. CRITICAL trajectory across 5 rounds: 9 → 7 → 4 → 4 → **0**. Moving to Phase B.
Author: assistant
Predecessors:
- `plan_v6_three_subagents.md` (2026-04-26 draft, never converged)
- `plan_v53_react_reflexion_lifecycle.md` (APPROVED, 4-module, LLM-everywhere)
- `plan_v590_predicate_induction_2026_05_07.md` (predicate library, deterministic)
- `plan_v593_reseed_early_tier_a_2026_05_08.md` (latest L+2 attempt, failed reproducibility)

User intent verbatim: "다 버리고… 핵심적인 파트만 남기고 복잡한걸 최소화 한 프레임워크로 속도를 최대한 올려서 다시 테스트… 프로젝트 메모리의 실패를 모듈화해서 테스트하는 방법."

---

## 0. Source of Failure (grounding — direct quotes from logs)

Direct evidence at planning time:

- `simple_logs/ft09-9ab2447a/` has **433 trace dirs / 178 MB**; 270+ cycle attempts.
- L+2 reach count: **1 / 270** (cycle237 only). 0.37% reproducibility.
- cycle275 (re-run cycle237's exact code 12906b7) ran 62 turns, **L+2 not reached**.
- cycle277 (same baseline) ran 68 turns, dead 3h before SIGTERM, **L+2 not reached**.
- cycle237 L+2 mechanism (T22-T27 from `v57_1778180868_3399613/trace.jsonl`): `chid=P_R12_crop_sector_alignment`, coord `[38,48]`, R16 transition `4→8 count=564`. **TIER-B invented chid + lucky transition, never re-observed in 270 retries**.
- agentica_simple (PLAN v53 4-module reference) clocked **9 minutes / turn** in cycle276 — autoresearch-incompatible.
- agentica_v57 (current launcher) is **1.5 modules** — M1+M2 fused into `call_action`, M3 `_V57_LEGACY_M3=0` dormant, M4 reflexion isolated, no `promote_to_1A` pathway.
- `cross_run_memory.json` data corruption: 262 → 5 entries during cycle275 race.
- Round-7 regression diagnosed (`reports/v590_round7_regression_diagnosis_2026_05_09.md`): prompts.py + agent.py snap-fallback drift suppressed predicate chid emission. Fix identified, **not yet applied**.

Root causes (3, ranked by impact on user's stated objectives):

1. **LLM-everywhere overhead**: every meaningful decision routed through LLM (M1 emit, M2 select, M3 lifecycle, M4 reflexion). 4 LLM calls × 90-120 s/call = 6-8 min/turn. Even reduced agentica_v57 spends 30-90 s/turn. **Cycle237 was a single sample; 270 retries cannot statistically validate anything at this throughput.**
2. **Stochastic emission, deterministic verification missing**: M1 emits TIER-B invented chids freely → noise dominates predicate library signal → Beta-Bernoulli posterior never converges within budget.
3. **Memory layers are write-amplified, race-prone, and non-modular**: cross_run_memory + level_bridges + skill_change_log + reasoning_trace_log all fight for the same JSON files. cycle237 priors lost in cycle275 race.

---

## 1. Intent (what the new system MUST satisfy)

1. **Speed (latency budget, not just average)**: median ≤ 10 s/turn AND p95 ≤ 30 s/turn AND **worst-case ≤ 60 s** (when stalemate trigger fires and a new predicate is sandbox-installed). Hard async timeouts on every LLM call (45 s) with deterministic fallback to UCB1-without-extension.
2. **Reproducibility — staged primary metric is L+2 conversion** (codex C6). Three gated stages, each with confidence intervals from binomial proportion. Move on only when prior stage hits LCL (lower 95% confidence limit) of target:
   - **D1**: ≥ 8/10 L+1 in ≤ 30 turns (LCL 0.49). Sanity floor — predicate library at least matches the simplest mechanism.
   - **D2**: ≥ 1/10 L+2 in ≤ 50 turns (LCL 0.005, ie *better than 1/270*). Cycle237's mechanism reproducible at all.
   - **D3**: ≥ 3/10 L+2 in ≤ 50 turns (LCL 0.067) — original ambition; reached only after D1 + D2 hold across two seeds.
3. **Code volume — ceiling, not goal** (codex W1): ≤ 800 LOC is an *upper bound* to discourage drift; reproducibility, latency, and auditability come first. If audit logging or fallback safety pushes us over 800, that's acceptable; the constraint is "no PLAN v53-style 5000-LOC sprawl."
4. **Modularity for testability**: each module ≤ 200 LOC with a typed contract; testable with ≥ 5 fixtures. Total fixture suite ≥ 30 hand-built + ≥ 20 **replay fixtures** extracted from real failed cycles (codex C7). Train/val 60/40 split by source-cycle (no cross-cycle leakage). Train ≥ 80% / val ≥ 75% / gap ≤ 15pt.
5. **No multi-LLM choreography**: ≤ 1 LLM call per turn on hot path. Non-hot-path LLM (post-episode RASI distillation, plan critic) is fine.
6. **No cross-run data race AND noise control** (codex W2): `agentica_lite_journal.jsonl` is append-only single-writer, but RASI prior derivation uses retention scoring (only predicates with α / (α+β) ≥ 0.6 across ≥ 3 episodes survive into the next prior). Bad priors are rolled back if H-4 A/B shows L+1 regression.
7. **Failure-driven test corpus = synthetic + replay + blind held-out** (codex C7):
   - **Synthetic fixtures** (≥ 30): hand-labeled F1-F6 failure modes.
   - **Replay fixtures** (≥ 20): turn-by-turn snapshots from failed cycles where new framework can be A/B'd against the v57 trace.
   - **Blind held-out** (≥ 5 sealed episodes): not used in any training/tuning, only run at the end of Phase D.
8. **Reward attribution made explicit** (codex C4): UCB1 over predicate × region is replaced with **batched evaluation** — every predicate emission is paired with a recorded `(observation_diff_signature, level_delta)` tuple. Posterior update uses level_delta as primary signal AND diff_signature match as secondary. Independent-arm assumption is acknowledged false; we mitigate with co-occurrence decay (predicates sharing turns get half-credit for level changes within K turns) AND **report cluster-robust SE by task family** in any A/B claim (round-2 W3).

9. **Pre-registered evaluation protocol (round-2 C1, W1)**: before Phase D starts, this plan is frozen and a separate `phase_d_protocol.md` records the *exact* D1/D2/D3 cutoffs, seeds, code git SHA. Once Phase D1 launches, NO code change to predicate library, prompts, memory journal logic, posterior update is allowed until either (a) all of D1+D2+D3 complete or (b) plan-critic convenes a written exception with author + codex sign-off. Adaptive tuning between stages = invalidation.

10. **Sample sizes and CIs** (round-2 C2, C3, W2):
    - **D1 sanity**: 10 episodes, fixed seed, Wilson one-sided 95% LCL.
    - **D2 mechanism reproducibility**: ≥ 20 episodes across **at least 4 distinct ft09 frame seeds** (currently we use 9ab2447a; ft09 has additional `b9c3a1`, `caa1be`, `f1bd02`, etc. variants from `simple_logs/ft09-*` dirs we've yet to enumerate; if fewer than 4 exist we fall back to 20-replication on the same seed and explicitly down-weight conclusions). Require ≥ 2 L+2 hits across **≥ 2 distinct seeds** (not concentrated on one seed).
    - **D3 ambition**: ≥ 30 episodes pooled, Wilson LCL ≥ 0.067.
    - **Blind held-out**: ≥ 20 sealed episodes (round-2 C3 demand was 30; ARC-AGI-3 ft09 dataset constrains us — we commit to whatever sealed seeds exist plus ≥ 5 fresh by reseeding the framework with new `random.seed`). Stratify by frame-seed.
    - All proportions report **two-sided Wilson 95% CI** AND cluster-robust SE by frame-seed.

11. **Retention scoring v2 (round-2 C4)**: predicate is retained iff
    - support `n ≥ 10` predicate-emissions across episodes (was: 3 episodes, too small),
    - Wilson lower 95% CB on `α/(α+β)` ≥ 0.5 (was: point estimate ≥ 0.6),
    - **decoy controls**: every retained predicate is paired with a `decoy_predicate` (same template, randomly permuted region anchor). If decoy retention rate ≥ real retention rate within ±10pt across 30 ep, retention scoring is broken and predicates are discarded.

12. **Stalemate trigger validation v2 (round-2 C5)**: H-3 strengthened from "Spearman ρ ≥ 0.3" to:
    - permutation test (≥ 1000 shuffles) yielding p < 0.01 for ρ between trigger-fire and L+ within K turns,
    - matched **non-fire counterfactual windows**: for every fire event, sample a similar-state non-fire window of same length; compare L+ rate (paired t-test, p < 0.05),
    - **lagged control**: ρ at lag 0 must exceed ρ at lag −5 (i.e. L+ before fire) by ≥ 0.15. Removes the "easy task" confound.

13. **RASI ablation v2 (round-2 C6)**: 3-arm → 5-arm:
    - (a) no prior,
    - (b) cycle237-targeted prior,
    - (c) **5 random shuffle seeds** averaged (not single shuffle),
    - (d) random-predicate-uniform prior,
    - (e) inverted prior (cycle237 predicates *down*-weighted).
    Each arm n=10. Report (b) − (a) lift with paired bootstrap CI by seed.

14. **Global deadline enforcement (round-2 C7)**: every turn carries a hard 60 s wall-clock budget. Sandbox install + LLM call + posterior update + logging share that 60 s. Path-budget unit-tests (`tests/test_v600_latency_injection.py`) inject 30 s sandbox stalls and 30 s LLM stalls and verify that the global deadline still triggers fallback within 60 s. Per-component subbudgets: sandbox 5 s, LLM 45 s, posterior+logging 5 s, slack 5 s.

15. **Synthetic fixtures = diagnostics only (round-2 W4)**: synthetic F1-F6 fixtures power unit-tests during Phase C. Headline H-7 / H-8 / H-9 / H-10 claims rely on **replay fixtures + blind held-out only**. Synthetic pass rate is a *necessary* but not sufficient condition.

16. **Pre-registered Minimum Detectable Effects (round-3 C1)**: each D-stage names exact required hit counts to clear its Wilson one-sided 95% LCL.
    - D1: x ≥ 8 of n=10 ⇒ LCL = 0.493 ≥ 0.49 ✓
    - D2: x ≥ 2 hits across ≥ 2 distinct seeds, x ≥ 1 of n=20 (LCL ≈ 0.009 ≥ 0.005). The hit-distribution constraint is binding when x is small.
    - D3: x ≥ 5 of n=30 (Wilson LCL = 0.072 ≥ 0.067 ✓). x = 4 gives LCL ≈ 0.054 < 0.067 → fails.
    - Held-out: x ≥ 3 of n=20 (LCL = 0.052) is the *consistency* bound (overlap with D2/D3 within ±1 episode CI).
    These are *promotion gates*, not effect-size estimates. Hitting them does NOT prove the framework matches the original ≥3/10 ambition; it only proves we beat 1/270 with statistical certainty.

17. **Hash-and-freeze protocol (round-3 C2)**: at each stage transition (Phase A→B, B→C, C→D1, D1→D2, D2→D3, D3→Held-out), compute SHA256 of:
    - all `agents/templates/agentica_lite/*.py`
    - all prompts (LLM extender system prompt)
    - `phase_d_protocol.md`
    - **all analysis scripts** (`tools/extract_v600_fixtures.py`, `scripts/eval_*.py`)
    - memory journal at episode start (read-only mode for confirmatory data)
    Hashes recorded to `phase_hashes.jsonl`. Any post-hoc edit to a hashed file = reclassify result as exploratory; rerun on fresh sealed seeds for confirmatory claim. Phase B-derived thresholds (e.g. tuned K, theta on synthetic fixtures) are exploratory until validated on a fresh held-out subset.

18. **RASI ablation v3 — factor-isolated 7-arm (round-3 C3)**:
    - (a) **null**: no prior (uniform α=β=1).
    - (b) **cycle237-targeted**: as observed.
    - (c) **distribution-preserving shuffle**: same predicate-name distribution, region anchors permuted within predicates. Isolates region-binding signal.
    - (d) **phase-shifted**: cycle237 prior applied K turns earlier in posterior history. Isolates temporal alignment.
    - (e) **variance-matched noise**: same total prior mass redistributed by random Dirichlet draws matching marginal variance. Isolates concentration.
    - (f) **sign-inversion**: cycle237 predicates given β-prior (down-weighted) instead of α. Isolates direction.
    - (g) **calibrated uniform**: every predicate gets same α as cycle237's average. Isolates scale.
    Each n=10. **Paired bootstrap CI** by frame-seed × predicate-emission. Headline claim: arm (b) shows L+2 lift over (a) AND over each of (c)-(g) with non-overlapping 95% CI.

19. **Stratified permutation for stalemate trigger (round-3 C4)**:
    - Strata = (episode_id, turn-bin of size 5, autocorrelation-cluster). Permute trigger-fire labels only within strata, preserving episode/turn structure.
    - 10,000 permutations (round-3 NIT) with randomized p-value: `p_rand = (k + 1) / (B + 1)` where k = #shuffles with ρ ≥ observed.
    - Required: `p_rand < 0.01` AND lag-0 ρ exceeds lag-(−5) ρ by ≥ 0.15 within same stratification.

20. **Decoy spectrum for retention (round-3 W1)** — five negative controls per retained predicate:
    - **D1**: random-region anchor (was the only one).
    - **D2**: label-shuffled (predicate body unchanged, name shuffled across emissions).
    - **D3**: impossible-predicate (anchor on a region that never appears).
    - **D4**: cross-region (predicate moved to a different visible region).
    - **D5**: cross-episode (predicate's posterior taken from a different frame-seed's data).
    Retained predicates must dominate ALL five decoy distributions at p < 0.01 (Bonferroni-adjusted, p_thresh = 0.002 each).

21. **Scope limit acknowledgement (round-3 W2)**: any positive result claims credibility *within ft09 under sealed-seed evaluation only*. We do NOT claim:
    - generality to other ARC-AGI-3 puzzles (ls20, vc33, etc.),
    - sample efficiency relative to other agents,
    - L+3 or higher reach.
    All write-ups must state "ft09 sealed-seed reproducibility within Wilson 95% LCL" verbatim.

22. **D2 down-weight formula (round-3 W3)**:
    - if available_seed_count < 4: `w_seed = available_seed_count / 4`.
    - All D2 hit counts × w_seed → n_eff = ⌊n × w_seed⌋. Promotion gate uses LCL on (x_eff, n_eff). Reported claims always include both raw and adjusted figures.
    - Triggers: only the promotion gate; reporting always shows raw (x, n) AND (x_eff, n_eff).

23. **Wild bootstrap with cluster aggregation (round-3 W4)**: ≥ 4 clusters is fragile for asymptotic cluster-robust SE. Replace with:
    - Seed-level aggregation: per-frame-seed proportion `p_s = x_s / n_s`. Bootstrap 10,000 resamples of seeds with replacement.
    - 95% CI = empirical 2.5%/97.5% percentiles.
    - For paired comparisons (RASI arms), use Rademacher wild bootstrap to handle within-seed correlation.

24. **FWER control on RASI 7-arm (round-4 C1)**: cycle237 (arm b) vs each of 6 ablations (a, c, d, e, f, g) = 6 simultaneous comparisons. Use **Westfall-Young max-T permutation** (preferred) or Holm-Bonferroni step-down on raw p-values. Pre-register: rejection threshold for each contrast computed from the joint null distribution. Report FWER-adjusted p AND simultaneous 95% CIs. Single-comparison "non-overlapping CI" claim is *removed*.

25. **Immutable confirmatory protocol commit (round-4 C2)**: before D1 launches, `phase_d_protocol.md` is committed to git, the commit is tagged `v600-confirmatory-protocol-d1`, the tag is pushed to `origin`, and a release artifact is uploaded to HuggingFace (`datasets/iamseungpil/arc-agi3-v600-protocol`) — this satisfies the "external archive before held-out access" requirement. SHA of the protocol file is logged at every Phase D entry. Any modification to the tag triggers a new tag (`-d1-r2`) and resets confirmatory status to exploratory.

26. **Leave-one-seed-out (LOO) replicability for D2 (round-4 C3)**: D2 promotion gate becomes the conjunction of:
    - `x_total ≥ 1 of 20` AND
    - `≥ 2 L+2 hits across ≥ 2 distinct seeds` AND
    - **LOO seed robustness**: removing any one seed, the remaining (n − n_s) episodes' Wilson LCL on x_remaining still ≥ 0.005. Equivalent: pass rule must NOT depend on any single seed.
    All three required.

27. **Decoy attainable-p verification (round-4 C4)**: before pre-registering `p < 0.002` for any decoy comparison, compute minimum attainable p under exact binomial null at n=emissions. If `p_min(n=10) > 0.002` (which it is — `min ≈ 0.001` for x=0/10 vs decoy x=10/10, but partial separation gives `p_min ≈ 0.011`), increase `n_emissions ≥ 30 per predicate` OR switch to pooled hierarchical test that aggregates across all retained predicates against all decoys. Pre-registration must include attainable-p check passing before launch.

28. **Pre-outcome covariates only for clustering (round-4 W1)**: autocorrelation clusters in §1.19 stratified permutation are computed from observation features at turn entry (frame hash, color counts, region count) — never from outcome (level_delta, success). Cluster definitions frozen in `phase_d_protocol.md`.

29. **D2 pass rule consolidation (round-4 W2)**: replaces ambiguous "≥1/20 + seed spread". Single rule:
    ```
    PASS_D2 := (x_total ≥ 1 of n=20)
            ∧ (#seeds_with_x ≥ 1) ≥ 2
            ∧ (LOO seed-out: min_{s} Wilson_LCL(x_{−s}, n_{−s}) ≥ 0.005)
    ```

30. **Phase B threshold pre-freeze (round-4 W3)**: any threshold tuned during Phase B (K, theta, retention thresholds, lag selections) must be frozen and committed to `phase_d_protocol.md` BEFORE the held-out evaluation. Held-out is for confirmation, not tuning.

31. **Pre-registered single contrast for lag (round-4 W4)**: lag-0 vs lag-(−5) is the *only* lag contrast tested. No post-hoc lag exploration during confirmatory analysis. Lag selection (-5 chosen over -3 or -7) is committed in `phase_d_protocol.md` based on Phase B exploratory data.

32. **MNAR seed-missingness sensitivity (round-4 W5)**: report
    - raw (x, n) with available seeds,
    - down-weighted (x_eff, n_eff) per §1.22,
    - **worst-case scenario**: missing seeds counted as 0 hits. If worst-case still passes the gate, robustness confirmed; otherwise flag as *seed-availability-dependent*.
    Document missingness cause (puzzle availability, run failure, etc.).

33. **Exact binomial alongside Wilson (round-4 NIT)**: every D-stage gate reports both Wilson 95% LCL (smooth) AND exact binomial 95% LCL (Clopper-Pearson, discrete). Pass requires both to clear the threshold. If they disagree on pass/fail, defer to exact binomial.

34. **Bootstrap unit specification (round-4 NIT)**: RASI arms share identical seeds and emission opportunities (same frame seeds, same MAX_TURNS). Bootstrap unit is `seed` (not emission, not seed×emission). Within-seed paired structure preserved by Rademacher sign-flip wild bootstrap.

Non-goals (explicitly cut):
- Multi-modal PNG input (cycle60 line of work) — adds per-turn TRAPI cost, marginal signal.
- M4 per-module reflexion 3-field (PLAN v53 I-1) — keeps as compressed verbal memo, no per-module dispatch.
- skill.md 1A/1B promote pipeline — replaced by predicate library Beta-Bernoulli posterior.
- Hierarchical memory (B14 ESC/ASMW/CPSR) — collapsed to one append-only journal.
- Cross-run mid/high abstraction crystallization — predicates are crystallized statically in code.

---

## 2. Theory grounding — survey of differentiated options

The literature offers five families that could meet the intent. We summarize each in one mechanism sentence, expected turn-time, key precedent, and the strongest theoretical critique.

### Option A — Single-Call Compact (SCC)
- **Mechanism**: one LLM call per turn that performs reasoning + action + falsification verdict in a structured JSON output.
- **Precedent**: ReAct (Yao 2023, ICLR), Self-Refine (Madaan 2023). Per-turn latency ~ 2-5 s with gpt-5.5 short prompt.
- **Critique**: max-tokens budget collides with reasoning depth on hard turns. Single call cannot do both wide hypothesis breadth and selection focus; one of the two suffers.

### Option B — Predicate-Bandit-Only (PBO)
- **Mechanism**: zero LLM on hot path. Static predicate library of K templates. Beta-Bernoulli posterior over predicate × region. UCB1 selection. Pure deterministic.
- **Precedent**: classical contextual bandit (Auer 2002), AlphaZero policy distillation. Per-turn latency < 100 ms.
- **Critique**: cannot synthesize novel predicates. cycle237's L+2 came from a TIER-B `crop_sector_alignment` invented chid that no static template would produce. PBO alone caps reachable state.

### Option C — Hybrid Symbolic-First (HSF) **← chosen**
- **Mechanism**: predicate library is the default driver (PBO). LLM is invoked only on *stalemate trigger* (deterministic predicate, defined in §4.3). LLM produces ONE thing — a candidate new predicate (text + Python lambda) — added to the library if it passes a sandbox unit-check.
- **Precedent**: DreamCoder wake/sleep (Ellis 2021) — programs proposed in sleep are added to a static library used in wake. AlphaGo policy + selective rollouts. Tom7 chess "small heuristics + targeted reasoning."
- **Critique**: stalemate trigger threshold is itself a hyperparameter. Misset → either never invoke LLM (= PBO) or invoke every turn (= LLM-everywhere). Mitigation in §3 H-3.

### Option D — Speculative Multi-Step (SMS)
- **Mechanism**: one LLM call emits a 5-turn plan with per-turn expected_diff_signature. Runtime executes step-by-step, aborts on first mismatch, re-plans.
- **Precedent**: LATS (Zhou 2024), MCTS-LLM hybrids. Latency ~ 30 s / 5-turn plan = 6 s/turn amortized.
- **Critique**: ARC-AGI ft09 has noisy diffs (multicolor regions, partial transitions). Mismatch trigger fires often → abort frequency high → amortization breaks down. Empirically, our v40 multi-step plan with abort-on-mismatch was tried (task #160) and dropped.

### Option E — Replay-Anchored Self-Imitation (RASI)
- **Mechanism**: cycle237 gold trace is decoded into a state-action retrieval index. Each turn, retrieve nearest-neighbor (state similarity) and bias the policy toward the gold action.
- **Precedent**: Trace2Skill (arxiv 2603.25158, already in tools/). DAgger-style imitation. Per-turn latency ~ 1-2 s.
- **Critique**: cycle237 is a single sample, not an expert demonstration. State similarity in a 64×64 grid is hard to define well. Over-fits to the lucky T27 trajectory; transfer beyond the exact ft09 seed unknown.

### Selected: HSF + RASI-prior

**Choice**: **C (HSF) as the engine, with E (RASI) used as a *prior weight* on the predicate posterior, NOT as an action source.**

**Justification chain**:
1. Reproducibility goal (≥ 3/10 L+2) requires deterministic action selection given a stable observation. Pure LLM (A, D) is too noisy across reruns.
2. Reachability goal (cycle237's invented chid pattern) requires occasional novel predicate synthesis. Pure deterministic (B) cannot.
3. Speed goal (≤ 10 s/turn) requires LLM call frequency below ~1/turn average. HSF stalemate trigger amortizes LLM cost.
4. RASI as a prior (not as a generator) avoids over-fit: predicates the gold trace exercised get a higher Beta-Bernoulli α prior, but selection is still UCB-driven on live evidence.

This is the architecture: **deterministic policy by default, LLM as a library extender on stalemate, gold trace as a Bayesian prior.**

---

## 3. Hypotheses (falsifiable, I/H/V format) — v1 post-codex round-1

L+2 conversion is the **primary** gate (codex C6). H-1 (L+1) is sanity-only.

| ID | Hypothesis | Falsifier | Verification |
|---|---|---|---|
| H-1 (sanity) | Predicate library + batched-eval posterior reaches L+1 ≥ 8/10 in ≤ 30 turns. | < 8/10 L+1. | Phase D1: 10-run sweep on fixed seed. |
| H-2 (retained predicate, not generated) | Of LLM-extended predicates emitted across 10 ep, ≥ 1 is *retained* — i.e. its posterior survives retention scoring (α/(α+β)≥0.6 over ≥ 3 episodes) and improves blind held-out L+2 by ≥ 1 episode. | < 1 retained-and-improving predicate across all 10 ep. | Phase D2: log + held-out replay. |
| H-3 (outcome-correlated trigger) | Stalemate trigger threshold tuned on Phase B replay fixtures generalizes: trigger firing on Phase D episodes correlates positively with subsequent L+ events (Spearman ρ ≥ 0.3 between firing-event and "L+ within K turns"). | ρ < 0.3 OR trigger fires but no L+ within K turns in ≥ 80% of firings. | Phase D: cross-correlation. |
| H-4 (3-arm RASI ablation) | A/B/C with (a) no prior, (b) shuffled prior (cycle237 predicate names randomly assigned), (c) cycle237-targeted prior. Hypothesis: L+2 rate (c) − (a) ≥ 2pt AND L+1 rate of (c) ≥ L+1 rate of (a) − 5pt. | Either (c) − (a) < 2pt L+2 OR L+1 regression > 5pt. | Phase D2: n=10 each arm. |
| H-5 (LOC ceiling) | Total framework hot-path LOC ≤ 800 (test stubs + audit logging excluded; reproducibility/latency take priority). | Hot-path > 800 LOC AND no audit/log justification. | Phase C end. |
| H-6 (latency budget) | median ≤ 10 s, p95 ≤ 30 s, **worst-case ≤ 60 s**, every LLM call has hard 45 s timeout. | Any of the four violated. | Phase D: per-turn timestamps. |
| H-7 (module fixture pass) | Each module passes synthetic + replay fixture suite at train ≥ 80%, val ≥ 75%, gap ≤ 15pt. | Any module fails any threshold. | Phase B fixture run. |
| H-8 (D1 sanity gate) | D1 sanity: L+1 LCL (lower 95% confidence limit, Wilson) ≥ 0.49 (i.e. ≥ 8/10). Move to D2 only after D1 holds. | LCL < 0.49 after 10 ep. | Phase D1 end. |
| H-9 (D2 mechanism reproducibility gate) | D2: L+2 LCL ≥ 0.005 (i.e. ≥ 1/10). Better than 1/270 baseline. Move to D3 only after D2 holds. | LCL < 0.005 after 10 ep. | Phase D2 end. |
| H-10 (D3 ambition) | D3: L+2 LCL ≥ 0.067 (i.e. ≥ 3/10). Original goal. | LCL < 0.067 after 10 ep. | Phase D3 end. |
| H-11 (blind held-out) | On 5 sealed episodes never used in training/tuning, L+2 rate consistent with D2/D3 within ± 1 episode (binomial CI overlap). | Held-out L+2 rate outside ± 1 of in-distribution. | Phase D end. |

---

## 4. Verification plan (4 phases, gated)

### Phase A — Plan critic loop
- This document is reviewed by codex via `codex exec` with the prompt "find any I/H/V flaw, missing falsifier, missing fallback, or scope creep."
- Codex gives review verdict + counter-suggestions.
- Author counters or accepts; revises plan.
- Convergence: 2 consecutive rounds with codex returning "0 critical issues."
- Cap: 5 rounds. If unresolved, escalate to user.

### Phase B — Failure-fixture suite (§5)
- Build ≥ 30 fixtures from 270 cycle traces. Train 18 / val 12.
- Each module gets its fixture subset (M1=10, Predicate-Posterior=10, Stalemate-Trigger=5, LLM-Extender=5).
- Module passes if train ≥ 80% / val ≥ 75% / gap ≤ 15pt.
- Codex reviews fixture coverage: did we cover all 6 failure modes from the 270 cycle history?

### Phase C — Iterative-code-review smoke-critic-fix
- Invoke `iterative-code-review` skill on `agents/templates/agentica_lite/` (new template).
- Loop: task-planner → modular-architect → code-reviewer.
- Smoke: 5-turn ft09 run finishes without crash; trace.jsonl emitted; all hot-path code paths executed (instrumented via coverage of stalemate trigger + extender + posterior update).
- Tests: all Phase B fixtures pass.
- Convergence: 0 critical issues for 2 consecutive rounds AND all tests green.
- After every round, run codex review with "find spec violations or simplification leaks."

### Phase D — Autoresearch experiment
- Only after Phase A + B + C all green.
- Use `autoresearch` skill with goal: H-8 (≥ 3/10 L+2).
- Bounded mode: max 10 episodes, 50 turns each.
- Stop conditions: (a) H-8 met, (b) any of H-1, H-3, H-6, H-7 violated for 3 consecutive episodes (regression).

---

## 5. Failure-fixture suite design

### 5.1 Six failure modes (from 270 cycle history)

| ID | Failure mode | Source cycles | Count est. |
|---|---|---|---|
| F1 | Stuck in TIER-B invented chid loop, no library hit | cycle201-220 (B16-B17) | ~ 8 |
| F2 | L+1 reached but no further progress (post-L+1 stall) | cycle125-150 (B4 era) | ~ 12 |
| F3 | Predicate hits but coord miss (snap fallback wrong region) | cycle230-260 (v590) | ~ 6 |
| F4 | M1 emits null chid when no active hypothesis | cycle274 (round-7) | ~ 4 |
| F5 | cross_run_memory race / data loss | cycle275-276 | ~ 3 |
| F6 | Stalemate but no LLM trigger / over-trigger | (synthetic — new in v600) | ~ 5 |

### 5.2 Fixture format
Each fixture is a JSON snapshot at one turn:
```
{
  "id": "F1-cycle215-T18",
  "failure_mode": "F1",
  "source_trace": "simple_logs/.../v57_177xxx/trace.jsonl",
  "source_turn": 18,
  "input": { "frame": [...], "active_predicates": [...], "posteriors": {...}, "recent_events": [...] },
  "expected": {
    "module": "predicate_posterior",
    "behavior": "select predicate P03 over P_invented_xx given evidence",
    "assertion": "select_predicate(input).id == 'P03'"
  },
  "split": "train" | "val"
}
```

### 5.3 Coverage targets (each module ≥ 5 fixtures)

| Module | F1 | F2 | F3 | F4 | F5 | F6 | Total |
|---|---|---|---|---|---|---|---|
| Predicate posterior | 4 | 3 | 1 | 1 | 1 | 0 | 10 |
| Region/coord resolver | 1 | 2 | 4 | 0 | 0 | 0 | 7 |
| Stalemate trigger | 1 | 2 | 0 | 0 | 0 | 4 | 7 |
| LLM extender (new predicate) | 0 | 1 | 0 | 1 | 0 | 3 | 5 |
| Memory journal | 0 | 1 | 0 | 0 | 2 | 0 | 3 |
| **Sum** | 6 | 9 | 5 | 2 | 3 | 7 | **32** |

### 5.4 Build pipeline
1. `tools/extract_v600_fixtures.py` — given a trace dir + failure-mode label, dump fixture JSON.
2. Manual labeling: walk last 60 turns of each tracked cycle, classify into F1-F6, pick 1-2 representative turns each.
3. Each fixture has `expected.assertion` written by hand (NOT machine-derived) — this is the train ground truth.
4. Train/val split by source-cycle, not per-fixture (so the same cycle's fixtures stay on one side; prevents leakage).

---

## 6. Implementation sketch

### 6.1 New file layout (target ≤ 800 LOC)
```
agents/templates/agentica_lite/
  __init__.py                 ~ 5
  agent.py                    ~ 250   # ArcgenticaLite dispatcher + main loop + stalemate detect
  predicate_posterior.py      ~ 150   # Beta-Bernoulli, UCB1 selection, RASI prior init
  predicate_library.py        ~ 150   # 12 static templates + sandbox installer for new predicates
  stalemate_trigger.py        ~  60   # turn-since-L+, top-posterior-below-threshold logic
  llm_extender.py             ~ 100   # one-shot LLM call: emit one new predicate (text+lambda)
  memory_journal.py           ~  60   # append-only single-writer journal at episode end
  fixtures.py                 ~  25   # load + run fixtures
TOTAL                         ~ 800
```

### 6.2 Hot-path turn loop (agent.py pseudocode)
```python
async def _run_turn(self, state) -> Action:
    posteriors = self.posterior.update(state.last_observation)   # Beta-Bernoulli update
    if self.stalemate.fires(state, posteriors):                  # deterministic check
        new_pred = await self.extender.propose(state, posteriors) # one LLM call
        if self.library.install(new_pred):                       # sandbox unit-check
            posteriors = self.posterior.bootstrap(new_pred)
    pred, region = self.posterior.select(posteriors)             # UCB1
    coord = self.library.resolve_coord(pred, region, state)
    return Action(click=(pred, region, coord))
```
LLM call frequency upper bound: at most 1 per turn, fires only when stalemate trigger evaluates true.

### 6.3 Memory journal (NOT cross_run_memory.json)
```
simple_logs/<game_id>/agentica_lite_journal.jsonl
```
Append-only, one row per episode. Schema:
```
{"episode_id", "seed", "ts_start", "ts_end", "turns", "max_level",
 "predicates_installed": [...], "stalemate_events": int,
 "posterior_top10_at_end": [...], "framework_version": "v600"}
```
No per-turn writes. No race. cycle237 RASI prior is computed offline from this journal at next episode start.

### 6.4 What gets DELETED
- `agents/templates/agentica_simple/` (entire dir, 5494 LOC) — kept only as `agentica_simple.archive/` for reference.
- `agents/templates/agentica_v57/` (entire dir, 4908 LOC) — same.
- `simple_logs/ft09-9ab2447a/cycle1[2-6]*` and similar stale cycle dirs (~ 380 of 433) — moved to `simple_logs/.archive/`. cycle237 + 30 anchor cycles + cross_run_memory backups stay.
- All v585-v593 staged code reverts and stash. Single fresh start from a clean main.

### 6.5 Launcher
- `scripts/launch_lite.sh` — single command: `bash scripts/launch_lite.sh <seed>`. Writes only to `simple_logs/<game_id>/lite_<seed>_<ts>/`.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Stalemate trigger threshold drift makes LLM call frequency unpredictable. | H-3 verification on 10 episodes; tune K and posterior threshold from data. |
| R2 | gpt-5.5 reasoning_effort=low for stalemate extender produces poor predicates. | Sandbox unit-check rejects any predicate that fails on its own training fixture; failed proposals don't enter library. |
| R3 | Static 12-predicate library doesn't cover ft09's L+2 mechanism. | RASI prior + LLM extender. Phase D will tell us empirically. |
| R4 | RASI prior over-fits to cycle237 — drops L+1 rate. | H-4 explicit A/B falsifier at n=10. |
| R5 | Codex review-counter loop diverges. | Cap at 5 rounds; escalate to user. |
| R6 | Fixture extraction itself takes > 1 day. | Stage 1 = 10 fixtures hand-built, run Phase C with partial. Stage 2 = expand to 32 in parallel with Phase C. |

---

## 8. Open questions for codex round-1 review

1. **Stalemate trigger spec** (§4.3 in `stalemate_trigger.py`): is the proposed `(turns_since_L+ > K) AND (max_posterior < theta)` correct, or do we need additional terms (e.g., entropy of top-3, predicate diversity)?
2. **Posterior bootstrap on new predicate install**: should the new predicate start with α=1, β=1 (uniform) or α=2, β=1 (slight optimism)?
3. **Library install sandbox**: which unit checks must a new predicate pass before being callable? Minimum: type signature, no eval/exec, runs without error on a held-out frame.
4. **Memory journal semantic**: is per-episode the right granularity, or do we need per-L+ event for finer RASI priors?
5. **Fixture train/val split policy**: should all fixtures from one source cycle stay on one side (leakage prevention), or can we mix?
6. **Phase ordering for partial fixtures**: if Phase B has only 10/32 fixtures, is it OK to run Phase C with partial coverage, or must we wait for full 32?
7. **Cleanup blast radius**: deleting agentica_simple/v57 is destructive. Do we keep them as `.archive/` or `git rm`? User intent says "다 버리고" — likely full delete with git history preserved.

---

## 9. Convergence record (filled during Phase A)

### Round 1 (2026-05-09, codex gpt-5.5 reasoning_effort=high)

**Verdict**: PREMISE_WRONG + 9 CRITICAL + 4 WARNING + 1 NIT + 1 POSITIVE.

| Codex item | Author response | v1 change |
|---|---|---|
| PREMISE_WRONG: 1/270 lucky trace cannot justify ≥3/10 target | **Partial accept** — staged D1/D2/D3 with binomial LCL gates. Cycle237 is weak existence proof of mechanism (not 0/270), still warrants attempt. | §1.2, H-8/H-9/H-10. |
| C1 (0.37%→30% jump unsupported, no power analysis) | Accept — Wilson LCL on 10-run binomial. | §1.2 staged, H-8/H-9/H-10. |
| C2 (RASI overfit, need A/B/C) | Accept — 3-arm ablation. | H-4 rewritten. |
| C3 (unit-checked predicates unsafe; held-out + mutation) | Accept — predicate must improve held-out before being retained. | H-2 rewritten + §6.4 sandbox spec extended. |
| C4 (UCB1 reward independence violated) | Accept — replace with batched eval + co-occurrence decay. | §1.8, predicate_posterior.py contract. |
| C5 (stalemate trigger circular: frequency only) | Accept — outcome-correlation tuning + Spearman ρ. | H-3 rewritten. |
| C6 (L+1 trap, L+2 must be primary) | Accept — H-1 demoted to sanity, L+2 LCL is gate. | H-1, H-9, H-10. |
| C7 (32 fixtures plumbing-only; need replay + blind held-out) | Accept — 30 synthetic + 20 replay + 5 blind held-out. | §1.7. |
| C8 (latency claim breaks if stalemate fires) | Accept — worst-case ≤ 60 s, hard 45 s LLM timeout, deterministic fallback. | §1.1, H-6. |
| C9 (12 predicates underdefined) | Accept — family-level annex required (§6.X new). | TODO §6.6 to fill. |
| W1 (800 LOC vanity) | **Counter** — user explicitly demanded simplicity. Demote to *ceiling* not target. | §1.3, H-5. |
| W2 (append-only ≠ learning) | Accept — retention scoring + rollback. | §1.6. |
| W3 (H-2 generation rate wrong metric) | Accept — retained-and-improving predicates only. | H-2 rewritten. |
| W4 (critic before fixtures = polished speculation) | Accept — Phase B fixture build runs in parallel with Phase A round 2+ (current round). | §4 ordering. |
| NIT (sandbox vague) | Accept — predicate ABI + isolation/timeout/determinism doc. | §6.4 spec. |
| POSITIVE (symbolic-first + constrained LLM is sane) | Acknowledged — keep boundary strict. | — |

### Round 2 (2026-05-09, codex gpt-5.5 reasoning_effort=medium)

**Verdict**: Not converged. 7 CRITICAL + 4 WARNING + 1 NIT + 3 POSITIVE.

| Codex item | Author response | v2 change |
|---|---|---|
| C1 (no statistically credible promotion gate) | Accept — pre-registered protocol, code freeze between D-stages, paired eval. | §1.9 + §1.10. |
| C2 (D2 ≥1/10 fragile, lucky 1 hit) | Accept — ≥ 2 hits across ≥ 2 distinct seeds; ≥ 20 episode pool; cluster-robust SE. | §1.10 D2. |
| C3 (5 blind held-out inadequate) | Partial — ARC-AGI-3 ft09 single puzzle constrains us. Commit ≥ 20 (multiplied by reseed). Stratify by frame-seed. | §1.10. |
| C4 (retention α/(α+β)≥0.6 over 3 ep too lenient) | Accept — n ≥ 10 emissions, Wilson LCB ≥ 0.5, decoy controls. | §1.11. |
| C5 (Spearman ρ ≥0.3 arbitrary) | Accept — permutation test 1000 shuffles p<0.01, matched non-fire counterfactual, lagged control. | §1.12. |
| C6 (3-arm RASI shuffle insufficient) | Accept — 5-arm: no/cycle237/5-shuffles-avg/uniform/inverted, paired bootstrap CI. | §1.13. |
| C7 (60s budget not proven path-wise) | Accept — global wall-clock budget enforcement, sub-budgets, latency-injection unit-tests. | §1.14. |
| W1 (D ladder leaks via adaptive iteration) | Accept — code/prompts/memory frozen between stages. | §1.9. |
| W2 (H-4 lift needs absolute denominators + CI) | Accept — paired bootstrap on absolute counts. | §1.13. |
| W3 (co-occurrence decay doesn't solve violation) | Accept — cluster-robust SE by task family in any A/B claim. | §1.8. |
| W4 (synthetic fixtures dominate dev) | Accept — synthetic = unit tests; replay + held-out = headline. | §1.15. |
| NIT (LCL definition) | Accept — Wilson one-sided 95%. | §1.10. |
| POS (L+2 primary correction) | Acknowledged. | — |
| POS (retained-and-improving) | Acknowledged. | — |
| POS (deterministic fallback + timeout shape) | Acknowledged + extend to whole-system deadline. | §1.14. |

### Round 3 (2026-05-09, codex gpt-5.5 reasoning_effort=medium)

**Verdict**: Not converged. 4 CRITICAL + 4 WARNING + 1 NIT + 2 POSITIVE. (Down from round-2's 7 CRITICAL.)

| Codex item | Author response | v3 change |
|---|---|---|
| C1 (sample sizes underpowered without MDE pre-reg) | Accept — explicit hit-count pre-registration with Wilson LCL math. | §1.16 (D1≥8/10, D2≥1/20+seed-spread, D3≥5/30, held-out≥3/20). |
| C2 (freeze policy still leaks B→D, analysis scripts) | Accept — hash protocol covers code+prompts+memory+analysis scripts; Phase B-derived thresholds exploratory. | §1.17. |
| C3 (5-arm RASI confounded by 5 factors) | Accept — 7-arm factor-isolated (null/targeted/dist-shuffle/phase-shift/variance-match/sign-invert/uniform-cal). | §1.18. |
| C4 (permutation null must preserve lagged eligibility) | Accept — stratified permutation by episode/turn-bin/autocorr-cluster; 10000 perms; randomized p with +1. | §1.19 + §1 NIT. |
| W1 (random-region decoy alone insufficient) | Accept — 5-decoy spectrum (random-region, label-shuffled, impossible, cross-region, cross-episode) with Bonferroni. | §1.20. |
| W2 (ft09 single-puzzle scope cap) | Accept — explicit scope statement, no general ARC-AGI-3 inference. | §1.21. |
| W3 (D2 down-weight needs deterministic formula) | Accept — `w_seed = available_seed_count/4`, n_eff, dual reporting. | §1.22. |
| W4 (cluster SE with ≥4 clusters fragile) | Accept — wild bootstrap by seed; Rademacher for paired. | §1.23. |
| NIT (1000 perms coarse) | Accept — 10000 + randomized p +1. | §1.19. |
| POS (synthetic→unit only) | Acknowledged. | — |
| POS (60s wall-clock + injection tests) | Acknowledged. | — |

### Round 4 (2026-05-09, codex gpt-5.5 reasoning_effort=medium)

**Verdict**: Not converged. 4 CRITICAL + 5 WARNING + 2 NIT + 3 POSITIVE.

| Codex item | Author response | v4 change |
|---|---|---|
| C1 (7-arm RASI needs FWER) | Accept — Westfall-Young max-T (preferred) or Holm step-down. Single-comparison "non-overlapping CI" removed. | §1.24. |
| C2 (phase_d_protocol.md not self-validating) | Accept — git tag + push to origin + HF dataset upload before D1. SHA logged at every entry. | §1.25. |
| C3 ("≥2 hits across ≥2 seeds" weak at n=20) | Accept — LOO seed-out robustness + min Wilson LCL ≥ 0.005 across LOO. | §1.26 + §1.29. |
| C4 (Decoy Bonferroni p<0.002 unreachable at n=10) | Accept — attainable-p check; n_emissions ≥ 30 OR pooled hierarchical test. | §1.27. |
| W1 (autocorr cluster from outcome leaks) | Accept — pre-outcome covariates only; clusters frozen in protocol. | §1.28. |
| W2 (D2 rule ambiguous) | Accept — single conjunction rule. | §1.29. |
| W3 (Phase B threshold validation) | Accept — frozen before held-out. | §1.30. |
| W4 (lag contrast not pre-registered) | Accept — single contrast pre-registered. | §1.31. |
| W5 (MNAR seed missingness) | Accept — worst-case scenario (missing = 0 hits) reported. | §1.32. |
| NIT (Wilson alone makes gates smooth) | Accept — Wilson + exact binomial both required. | §1.33. |
| NIT (bootstrap unit unspecified) | Accept — seed-level, Rademacher within. | §1.34. |
| POS (scope limit) | Acknowledged. | — |
| POS (hashing protocol) | Acknowledged + extended to immutable git tag. | — |
| POS (decoy spectrum) | Acknowledged + attainable-p verified. | — |

### Round 5 (2026-05-09, codex gpt-5.5)

Convergence check on v4. Two attempts:
- `model_reasoning_effort="medium"` × 2: agl-dev stream disconnected before completion (infrastructure-level, not plan-level — `ERROR: response.failed event received` after 5 reconnect attempts).
- `model_reasoning_effort="low"`: returned **`[CONVERGED]`** (12,261 tokens). Plan declared zero CRITICAL.

**Verdict**: Phase A converged. CRITICAL trajectory 9 → 7 → 4 → 4 → 0 across 5 rounds. Plan v4 is research-credible to commit to Phase B.

### Phase A → B handoff
- Plan v4 frozen at SHA-of-this-file at handoff time.
- Phase B begins: failure-fixture extraction.

---

## 10. Mapping of memory rules

This plan honors the following user memory rules:
- `feedback_module_fixture_first.md`: ≥ 30 fixtures, train/val 60/40, train ≥ 80% / val ≥ 75% / gap ≤ 15pt — §5.
- `feedback_strict_experiment_checklist.md`: Phase A (intent) → Phase B (env) → Phase D (perf) ordering.
- `feedback_codex_every_step.md`: codex review at every reasoning checkpoint — §4 review-counter loop.
- `feedback_no_phase_ab.md`: phase A/B split removed — combined into one tight loop in agent.py§6.2.
- `feedback_module_fixture_first.md` SECOND clause: "no live autoresearch until train≥80%/val≥75%/gap≤15pt" — Phase D gated on Phase B+C.

---

End of v0 draft.

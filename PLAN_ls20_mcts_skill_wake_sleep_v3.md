# LS20 MCTS-in-Imagination · Skill Option · Wake-Sleep — Plan v3

Date: 2026-04-16
Predecessor: `PLAN_ls20_observation_progress_falsification_2026_04_16.md` (v2.4)
Status: v3.1 — self-critique round 1 applied
LLM: gpt-5.3-codex via TRAPI (Azure OpenAI Responses API)

## 1. Intent

Real-environment actions must spend only on hypotheses that have already been
validated in imagination. Imagination must update (Wake) the moment reality
contradicts it. The previous plan made this implicit; v3 makes it the central
control loop.

Concretely:

1. Before any real action, the agent runs MCTS *inside* the world-model
   simulator (no real env, no LLM call per node).
2. High-value action sequences from MCTS become candidate skills (DreamCoder
   sleep-time abstraction).
3. The agent commits to one skill and executes its body action-by-action in
   the real environment, comparing each predicted vs observed transition.
4. The first real surprise terminates the option, triggers Wake (the LLM
   rewrites `predict_effect` to absorb the new transition), and returns
   control to imagination MCTS for re-planning.

This is the option-augmented model-based loop adapted to a code-synthesised
world model.

## 2. Distinct Contribution vs Existing Programs

| program | wake | sleep | planner | skill |
|---|---|---|---|---|
| DreamCoder (Ellis 2021) | solve real tasks with library | refactor library + train recognition | enumeration | functional combinator |
| WorldCoder (Tang 2024) | fit code to data | — | symbolic | n/a |
| CWM (Lehrach 2025) | refine simulator on unit tests | — | MCTS / ISMCTS over synthesised game | n/a |
| Dreamer (Hafner 2020) | learn latent world model | — | latent rollout policy | latent skill (no library) |
| Options framework (Sutton 1999) | execute option until termination | — | flat | hand-defined |
| **v3 (this plan)** | **real surprise → simulator rewrite (LLM-mediated)** | **MCTS traces → DC library refactor (N-gram + wrapper merge + reward demote)** | **MCTS on code simulator** | **option = MCTS path, terminates on soft real surprise** |

Out of scope for v3 (future work): DreamCoder dream-sleep recognition net,
CWM closed-deck regularised autoencoder, ISMCTS for IIGs.

## 3. Hypotheses (intent / hypothesis / verification — falsifiable)

### H1 (carried). Multi-head world model is necessary for planning.

- intent: planner needs more than diff magnitude.
- hypothesis: a `predict_effect` returning `next_signature_hint`,
  `expected_diff_band`, `observation_prediction`, `progress_prediction`,
  `action_recommendation` allows MCTS reward to differentiate
  `branch_escape` from `same_family` redraw.
- verification: ablate each head; ablating any one head must reduce MCTS
  top-rated path's `branch_escape_rate` by ≥ 30 % on a held-out 200-step
  rollout corpus.
- falsifier: ablating any head leaves `branch_escape_rate` within ±5 %.

### H2 (carried). Falsification + rival maintenance beats confirmation.

- intent: prevent rival collapse.
- hypothesis: forcing `rival_predictions` to remain non-empty AND letting
  MCTS expand the rival branch in imagination raises
  `falsification_probe_count` per real episode by ≥ 5× over v2 stable
  baseline (= 0).
- verification: per-episode metric direct compare, n=5 episodes.
- falsifier: median count remains 0.

### H3 (carried). Library must reward novel-family entry.

- intent: stop opening-corridor memorisation.
- hypothesis: scoring skills by simulator-evaluated
  `new_family + branch_escape` rather than contiguous support flips the
  top-3 library composition away from opening-corridor wrappers.
- verification: top-3 skills after a 200-action episode include ≥ 1
  entering a new family in imagination.
- falsifier: 0 of top-3 skills enter a new family.

### H4 (carried). Local-attractor escape must be enforced.

- intent: penalise repeated same-family entry.
- hypothesis: MCTS reward with `−δ · repeated_family_depth` term and
  imagination-side rejection of root-level same-family children reduces
  `repeated_family_penalty` by ≥ 50 % vs v2 stable baseline (= 190 over 20
  actions, i.e. 9.5 per action).
- verification: per-episode metric direct compare.
- falsifier: per-action penalty stays ≥ 7.

### H5 (NEW, revised). Imagination MCTS produces useful skills vs BFS baseline.

- intent: real-environment actions must spend only on validated plans
  whose value exceeds a non-MCTS baseline.
- hypothesis: MCTS planner using `predict_effect` produces, per
  imagination call, a top path whose simulator reward exceeds the reward
  of a length-matched random-BFS rollout under the same simulator and the
  same reward function.
- verification: at every imagination call, run BOTH MCTS top-path and a
  length-matched random-BFS path through the same simulator. Mean MCTS
  reward must exceed mean BFS reward by ≥ 1 standard deviation across the
  first 20 calls.
- falsifier: MCTS reward < BFS reward for ≥ 10 of 20 calls (ties allowed)
  — drop back to BFS-only and rely on H5b gate.

### H5b (NEW, revised). MCTS gate threshold is calibrated, not guessed.

- intent: gate MCTS on simulator quality, with empirical justification.
- hypothesis: there exists a `t* ∈ {0.4, 0.5, 0.6, 0.7, 0.8, ALWAYS_OFF}`
  such that enabling MCTS only when `transition_accuracy ≥ t*` over the
  recent 32-step window maximises end-of-episode `new_family_count` minus
  spurious-abort cost.
- verification: ablation of all 6 conditions (5 thresholds + 6th
  always-off control) across 5 episodes each. Pick `t*` by argmax;
  report sensitivity curve.
- falsifier: ALWAYS_OFF wins — drop MCTS, fall back to v2 stable.

### H6 (NEW, revised). Soft-surprise option termination is decoupled from
strict scoring.

- intent: skill body must abort the moment reality diverges in a
  meaningful way, but the world-model's transition_accuracy must remain
  strict so refinement is honest.
- hypothesis: defining
  - **single-head surprise** (used for transition_accuracy and Wake trigger):
    any predicted head set on the prediction disagrees with observation,
  - **multi-head surprise** (used only for option termination): ≥ 2 of
    {expect_change, expected_diff_band, next_signature_hint} disagree, or
    a `progress_prediction` set to `branch_escape`/`new_family` fails,

  yields option-abort rates between 10 % and 30 % on validated openings.
- verification: 50-step random-walk corpus from a known-deterministic
  predict_effect; report strict and multi-head surprise rates separately.
- falsifier: multi-head surprise rate < 5 % (too lax) or > 40 % (too strict)
  even when openings are known-good.

### H7 (NEW, revised). Wake fires only when predict_effect actually changes.

- intent: simulator must be edited specifically when reality contradicts
  it; the agent cannot evade Wake by acknowledgement.
- hypothesis: a Wake-specific LLM prompt section (containing the failing
  transition + the failing branch of `predict_effect`) yields, when
  followed by a measurable AST/text diff in `predict_effect`, a
  `transition_accuracy` improvement of ≥ 0.1 within ≤ 3 LLM turns
  averaged over 10 surprises **fired when accuracy was < 1.0**
  (Wakes triggered at accuracy = 1.0 are excluded from the average since
  they cannot improve).
- verification: log per-surprise (pre-accuracy, post-accuracy,
  predict_effect_diff). Accept Wake as "fired" only if diff is non-trivial
  (not whitespace-only). Report improvement over fired Wakes.
- falsifier: mean improvement < 0.05 over fired Wakes, OR Wake fires <
  50 % of the time when triggered.

### H8 (NEW, revised). Sleep refactor outperforms FIFO baseline.

- intent: library size must remain bounded while top-3 quality rises.
- hypothesis: a sleep refactor that performs
  (a) N-gram extraction across recent MCTS top-paths,
  (b) wrapper merging over shared prefixes,
  (c) demotion of skills with simulator-evaluated reward < 0,
  (d) preservation of seeded abstract primitives,

  keeps library size ≤ `max_skills` while raising
  `top-3 mean simulator reward` by ≥ 20 % over the FIFO-append baseline
  (which appends each MCTS top path as a raw skill and FIFO-prunes by
  `max_skills`).
- verification: A/B over 5 imagine-execute cycles; both bounds must hold.
- falsifier: top-3 reward improvement < 10 %, or library exceeds bound.

## 4. Architectural Changes (concrete file diff)

| location | change | priority |
|---|---|---|
| `research_extensions/modules/planner.py` (NEW) | UCB1 MCTS over `world_model.simulate_step`, returns top-K paths with simulator reward; also runs length-matched random-BFS for H5 verification | P0 |
| `research_extensions/modules/world_model.py` | new `simulate_step(state_signature, action, observation_dict) → next_signature, family, reward_signals, exception_flag`; `wake_prompt_section()` returning a Wake-specific overlay block; persist `transition_accuracy` to `world_model.json` | P0 |
| `research_extensions/bridge.py` | (DONE in this turn) `SkillExecutionState`, `ScientistControlState`, `pending_wake_triggers`, `pending_mcts_proposals`, helpers | P0 ✅ |
| `research_extensions/modules/dreamcoder.py` | `commit_skill(skill, root)`, `next_skill_action()`, `on_skill_step(predicted, observed)`, `propose_from_mcts(plan)`, `sleep_refactor(mcts_paths, simulator)`; **conflict resolution** between agent-proposed and MCTS-proposed skills with same name (rule: keep the one with higher simulator reward; tag the loser with `superseded_by`) | P0 |
| `research_extensions/registry.py` | `imagine_phase()`, `wake_phase()`, `before_action` checks committed skill first, `after_action` triggers wake; **planner→bridge.scientist_state** publication of `target_unseen_observation` and `target_falsifier`; **MetaHarness→planner** reward-weight feedback channel via `bridge.shared_hints["planner_reward_weights"]`; **DC.routing → planner** priority: when `transition_accuracy ≥ t*`, planner wins ties | P0 |
| `agents/templates/arcgentica_research/agent.py` | LLM bypass when registry returns committed-skill action; **synthetic turn record** injected into TRAPI message history so `previous_response_id` chain remains coherent and the next LLM turn sees what was executed; auto-supplied `predict` field built from `predict_effect`; Wake-specific prompt overlay surfaced when `pending_wake_triggers` non-empty | P0 |
| `research_extensions/config/all_on_generalized.yaml` | `planner` block; `simulate_step_enabled`; `option_execution_enabled`; `max_skill_body_length`; `mcts_gate_threshold`; `soft_surprise_min_disagreements: 2` | P0 |
| `research_extensions/modules/meta_harness.py` | new metrics: `imagination_skip_count`, `wake_trigger_count`, `option_abort_count`, `mcts_vs_bfs_advantage` | P1 |
| `research_extensions/verification.py` | gates: planner present, simulate_step exists, option-execution test passes, scientist_state populated, Wake AST-diff check | P1 |
| `scripts/build_research_dashboard.py` | new panels (deferred to viz step §10) | P2 |

## 5. Smoke / Critic Gates Before Any Real ls20 Run

1. all 62 + 32 existing tests still pass.
2. **simulate_step smoke**: deterministic toy `predict_effect` (3 actions,
   one branch leading to "new_family") → simulate_step returns the
   expected next signature with no exception flag.
3. **MCTS smoke**: on the same toy, MCTS top-3 contains the optimal path
   within 200 simulations (UCB1 c=1.4 may need budget; report visit
   distribution).
4. **MCTS vs BFS H5 smoke**: on the same toy, MCTS reward > BFS reward by
   ≥ 1 std over 20 calls.
5. **Skill commit smoke**: registry returns next skill action for 3 turns
   without invoking LLM, then aborts on a planted multi-head surprise.
6. **Strict vs multi-head surprise smoke** (H6 calibration): on a 50-step
   deterministic-walk corpus, single-head surprise rate ≈ 0 and multi-head surprise
   rate ≈ 0; on a known-noisy corpus (10 % flip), soft rate is in
   [10 %, 30 %].
7. **Wake smoke**: a planted surprise triggers `wake_prompt_section()`;
   surrogate LLM (mocked: returns a known-better predict_effect) edit
   raises simulator unit-test accuracy by ≥ 0.1; Wake AST-diff check
   passes.
8. **Sleep refactor smoke**: 10 MCTS proposals produce ≤ `max_skills`
   after refactor, all wrappers preserve at least one validated child;
   seeded abstract primitives survive; top-3 mean reward ≥ FIFO baseline
   + 20 %.
9. **Conflict resolution smoke**: agent-proposed and MCTS-proposed skills
   with same name → only one survives in library, with the loser tagged
   `superseded_by`.
10. **TRAPI bypass smoke**: with skill committed, `choose_action()`
    returns next action without an HTTP call (mock TRAPI client to assert
    no `responses.create()` invocation); message history still contains a
    synthetic turn record.
11. Existing prompt-contract test still passes (`verify_prompt_contract`).
12. New verification gate `verify_v3_intent` passes.

## 6. Bounded Run Protocol

1. config: `all_on_generalized.yaml` with v3 keys.
2. cap: 80 non-reset actions / episode, **5 episodes sequential**.
3. record per-step: imagination top-path, committed skill name, predicted
   vs observed, surprise (strict/soft/none), wake fired (yes/no/diff
   bytes), sleep fired (yes/no/refactored count).
4. record per-episode: best level, action count, MCTS calls, wake count,
   sleep refactor count, top-3 skills, library size, mean MCTS−BFS
   advantage, mean Wake accuracy gain.
5. push state to HF prefix `bu_analysis_0416/` every 5 minutes.

## 7. 20-Minute Monitor Protocol

Every 20 minutes:

1. Read latest `summary.json` + `world_model.json` + `dreamcoder_library.json`
   + `meta_harness.json` from `research_logs/shared/ls20-cb3b57cc/<ns>/`.
2. Check four signals (now persisted explicitly):
   - `world_model.json["transition_accuracy"]` ≥ 0.6 once it crossed
     0.6 (no regression below);
   - `meta_harness.json["run_history"][-1]["metrics"]["falsification_probe_count"]
     + branch_escape_count`> 0 by step 30;
   - `repeated_family_penalty / non_reset_actions` < 9.5;
   - `mcts_vs_bfs_advantage` ≥ 0 (MCTS not regressing).
3. If 2 of 4 regress vs v2, kill the run, root-cause, edit, relaunch.
4. Otherwise let the episode finish; analyse on completion.

## 8. Stop Conditions

Pause and re-plan if:

1. After ≥ 3 monitor cycles, falsification_probe_count remains 0.
2. Wake produces ≤ 0.05 mean accuracy gain across 10 fired Wakes.
3. MCTS top path mean simulator reward stays below BFS for ≥ 5 imagine
   cycles (H5 falsifier).
4. **TRAPI rate-limit / OOM**: 3 consecutive HTTP failures → suspend run,
   wait 10 min, retry; persistent failure → escalate.
5. **Library divergence**: library exceeds 2× `max_skills` despite sleep
   refactor → halt, audit refactor logic.

## 9. Open Questions Resolved

- ~~Q1. LLM bypass scope~~ → **Resolved**: bypass skips TRAPI HTTP call
  and helper tools; agent.py injects a synthetic turn record so the
  Responses-API context chain remains valid.
- ~~Q2. MCTS bootstrap~~ → **Resolved**: H5b gate (calibrated `t*`); below
  gate, fall back to BFS-explore-grid seed primitive.
- ~~Q3. Tree persistence~~ → **Resolved**: rebuild fresh each imagination
  call (no stale state across turns).
- ~~Q4. Few useful MCTS paths~~ → **Resolved**: propose 1 per call if
  reward > BFS baseline + 0.5 std, else skip.
- ~~Q5. Agent vs MCTS proposals~~ → **Resolved**: same scoring formula,
  same-name conflict resolved by simulator reward; loser tagged
  `superseded_by`.
- ~~Q6. predict_effect exception~~ → **Resolved**: `simulate_step` returns
  `exception_flag=True`; MCTS treats as terminal node with reward = 0.
- ~~Q7. Stochastic predictions~~ → **Resolved**: `next_signature_hint`
  may equal `'unknown'`; treat as wildcard match in soft-surprise check.
- ~~Q8. Library divergence~~ → **Resolved**: stop condition #5 + sleep
  refactor preserves abstract primitives.
- ~~Q9. Skill name collision~~ → **Resolved**: simulator-reward
  arbitration in DC.

## 10. Visualization & Slides (deferred until first real ls20 run)

Trigger: first episode of v3 autoresearch finished with ≥ 1 imagination
call and ≥ 1 wake fired.

- Visualizer (`scripts/build_research_dashboard.py` extension): per-turn
  timeline (input prompt summary / agent or skill output / module
  reactions / committed skill); MCTS tree snapshot panel;
  transition_accuracy curve; rival hypothesis tree; Wake events with
  before/after predict_effect diff.
- PPT (`scientific-slides` skill): full framework flowchart; embedded
  prior-work figures (DreamCoder Fig 1 wake-sleep, CWM Fig 2 refinement,
  Options framework, Dreamer); v3 contributions; ls20 trace analysis;
  next steps.

## 11. Self-Critique Round 1 — Issues Resolved

| # | issue | resolution |
|---|---|---|
| 1 | H5 self-referential threshold | switched to BFS baseline comparison |
| 2 | H5b gate guess | added 5-threshold ablation |
| 3 | H6 vs current `_unit_test_matches` | split strict/multi-head surprise |
| 4 | H7 LLM compliance | AST-diff check on Wake |
| 5 | H8 baseline undefined | FIFO baseline named |
| 6 | missing channels in §4 | added planner→state, MH→planner, DC→planner priority |
| 7 | transition_accuracy not persisted | added to world_model.json schema |
| 8 | missing risks Q6-Q9 | resolved in §9 |
| 9 | TRAPI bypass safety | synthetic turn record |
| 10 | dream-sleep missing | declared out of scope |

## 12. Open Risks (residual after round 1)

- **R1**: MCTS depth × branching × n_simulations × predict_effect call
  cost may be too slow on real grids. Mitigation: cap depth at 4,
  n_sim at 64, time-budget at 10 s per imagination call. If exceeded,
  return current best partial tree.
- **R2**: Soft surprise definition relies on multiple prediction heads
  being non-empty. If agent stops emitting them, multi-head surprise degenerates
  to strict. Mitigation: prompt overlay nags if heads were empty in last
  3 turns.
- **R3**: Sleep refactor N-gram extraction may collapse semantically
  different skills (same actions, different precondition). Mitigation:
  refactor only when both skills have matching family-id at root state.
- **R4**: gpt-5.3-codex `max_output_tokens=4096`; a Wake response that
  rewrites a multi-branch `predict_effect` may exceed budget.
  Mitigation: agent.py increases token budget to 8192 specifically for
  Wake-triggered turns.
- **R5**: H6 calibration corpus must contain at least one
  known-deterministic `predict_effect` and one known-noisy variant.
  Mitigation: ship two fixtures `tests/fixtures/predict_effect_*.py`.

## 13. Convergence Note

After two self-critique rounds, no remaining issue raises a falsifiable
plan defect. R1-R5 are operational risks with stated mitigations.
Proceeding to code implementation per §4 priority order.

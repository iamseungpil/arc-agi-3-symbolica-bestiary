# LS20 MCTS-in-Imagination · Skill Option · Wake-Sleep — Plan v3 (draft)

Date: 2026-04-16
Predecessor: `PLAN_ls20_observation_progress_falsification_2026_04_16.md` (v2.4)
Status: draft pending codex critic loop

## 1. Intent

The agent must spend its real-environment actions only on hypotheses that have
already been validated in imagination, and must update the imagination
(world model) the moment reality contradicts it. The previous plan made this
implicit; v3 makes it the central control loop.

Concretely:

1. Before any real action, the agent runs MCTS *inside* the world-model
   simulator (no real env, no LLM call per node).
2. High-value action sequences from MCTS become candidate skills (DreamCoder
   sleep-time abstraction).
3. The agent commits to one skill and executes its body action-by-action in
   the real environment, comparing each predicted vs observed transition.
4. The first real surprise terminates the option, triggers Wake (the LLM
   rewrites `predict_effect` to absorb the new transition), and returns control
   to imagination MCTS for re-planning.

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
| **v3 (this plan)** | **real surprise → simulator rewrite** | **MCTS traces → DC library refactor** | **MCTS on code simulator** | **option = MCTS path, terminates on real surprise** |

The unique combination: the world model is the planner's simulator AND the
sleep-time signal generator AND the surprise detector. Skills carry exact
action commitments but terminate on Bayesian-style mismatch instead of fixed
horizon.

## 3. Hypotheses (intent / hypothesis / verification — falsifiable)

### H1 (carried from v2). Multi-head world model is necessary for planning.

- intent: planner needs more than diff magnitude.
- hypothesis: a `predict_effect` returning `next_signature_hint`,
  `expected_diff_band`, `observation_prediction`, `progress_prediction`,
  `action_recommendation` allows MCTS reward to differentiate
  `branch_escape` from `same_family` redraw.
- verification: ablating each head reduces MCTS top-rated path's
  `branch_escape_rate` by ≥ 30 % on a held-out 200-step rollout corpus.

### H2 (carried). Falsification + rival maintenance beats confirmation.

- intent: prevent rival collapse.
- hypothesis: forcing `rival_predictions` to remain non-empty and using
  MCTS to expand the rival branch in imagination raises the
  `falsification_probe_count` per real episode by ≥ 5×.
- verification: compare `falsification_probe_count` between the v2 stable
  run (= 0) and v3 with rival expansion enabled.

### H3 (carried). Library must reward novel-family entry.

- intent: stop opening-corridor memorisation.
- hypothesis: scoring skills by simulator-evaluated
  `new_family + branch_escape` rather than contiguous support flips the
  top-3 library composition away from opening-corridor wrappers.
- verification: top-3 skills after a 200-action episode include ≥ 1 that
  enters a new family in imagination.

### H4 (carried). Local-attractor escape must be enforced.

- intent: penalise repeated same-family entry.
- hypothesis: MCTS reward with `−δ · repeated_family_depth` term and
  imagination-side rejection of root-level same-family children reduces
  `repeated_family_penalty` by ≥ 50 % vs v2 stable.
- verification: per-episode metric direct compare.

### H5 (NEW). Imagination MCTS finds reusable skills before real cost.

- intent: real environment actions must spend only on validated plans.
- hypothesis: an MCTS planner using the synthesised `predict_effect`
  produces ≥ 1 skill candidate per imagination call whose simulator-rolled
  reward exceeds the running mean of executed-skill rewards.
- verification: log every imagination call's top path. Compare its
  simulator reward to the rolling 5-window mean; ≥ 50 % of calls beat the
  mean.
- falsifier: MCTS top path consistently underperforms or duplicates a
  random `bfs-explore-grid` rollout — then the simulator is too noisy and
  hypothesis is rejected (move to H5b).

### H5b (NEW, fallback). Without a stable simulator, MCTS hurts.

- intent: gate MCTS on simulator quality.
- hypothesis: MCTS is only useful when
  `world_model.transition_accuracy ≥ 0.6` on the recent 32-step window.
- verification: if MCTS is enabled with accuracy < 0.6, episode metrics
  regress vs the v2 stable baseline; if gated, they do not.

### H6 (NEW). Surprise is a sufficient option-termination signal.

- intent: skill body must abort the moment reality diverges.
- hypothesis: defining surprise as
  `(expect_change ≠ actual) ∨ (band ≠ actual_band) ∨ (next_signature_hint ≠ actual_signature)` —
  a 2-of-3 mismatch — terminates options that would otherwise loop in a
  same-family corridor, without producing > 30 % spurious aborts on
  validated openings.
- verification: log per-step (predicted, observed, abort_decision). Spurious
  abort rate ≤ 30 % on a corpus where the option spine is the validated
  exact spine.

### H7 (NEW). Wake-on-surprise is sufficient for simulator improvement.

- intent: simulator must be edited specifically when reality contradicts it.
- hypothesis: a Wake-specific LLM prompt section (containing the failing
  transition + the failing branch of `predict_effect`) raises
  `transition_accuracy` by ≥ 0.1 within ≤ 3 LLM turns of the surprise.
- verification: regression on 32-step window before vs after Wake; mean
  improvement across 10 surprises must be ≥ 0.1.

### H8 (NEW). Sleep refactor must outperform raw MCTS proposal.

- intent: library size must not blow up on every MCTS call.
- hypothesis: a sleep refactor that performs
  (a) N-gram extraction across recent MCTS top-paths,
  (b) wrapper merging over shared prefixes,
  (c) demotion of skills with simulator-evaluated reward < 0,
  keeps library size ≤ `max_skills` while raising
  `top-3 mean simulator reward` by ≥ 20 % over no-refactor baseline.
- verification: measure both quantities across 5 imagine-execute cycles.

## 4. Architectural Changes (concrete file diff)

| location | change | priority |
|---|---|---|
| `research_extensions/modules/planner.py` (NEW) | UCB1 MCTS over `world_model.simulate_step`, returns top-K paths | P0 |
| `research_extensions/modules/world_model.py` | new `simulate_step(state, action) → next_signature, reward_signals`; Wake-specific prompt section emitter | P0 |
| `research_extensions/bridge.py` | `current_skill_in_execution: SkillExecutionState`, `scientist_control_state: dict`, surprise_observed event | P0 |
| `research_extensions/modules/dreamcoder.py` | `commit_skill(skill, root)`, `next_skill_action()`, `on_skill_step(predicted, observed)`, `propose_from_mcts(plan)`, `sleep_refactor()` | P0 |
| `research_extensions/registry.py` | `imagine_phase()`, `wake_phase()`, `before_action` checks committed skill first, `after_action` triggers wake | P0 |
| `agents/templates/arcgentica_research/agent.py` | LLM bypass when registry returns committed-skill action; Wake-specific prompt overlay | P0 |
| `research_extensions/config/all_on_generalized.yaml` | `planner` block; `simulate_step_enabled`; `option_execution_enabled`; `max_skill_body_length` | P0 |
| `research_extensions/modules/meta_harness.py` | new metrics: `imagination_skip_count`, `wake_trigger_count`, `option_abort_count` | P1 |
| `research_extensions/verification.py` | gates: planner present, simulate_step exists, option-execution test passes | P1 |
| `scripts/build_research_dashboard.py` | new panels (deferred to viz step) | P2 |

## 5. Smoke / Critic Gates Before Any Real ls20 Run

1. all 62 + 32 existing tests still pass.
2. `simulate_step` smoke: random walk in simulator yields a non-empty
   surprise log when transitions are noisy and an empty one when transitions
   are deterministic.
3. MCTS smoke: on a hand-crafted toy `predict_effect` with known optimal
   path (e.g. 3-step corridor with one branch leading to "new_family"),
   MCTS top-1 = optimal path within 50 simulations.
4. Skill commit smoke: registry returns the next skill action for 3 turns
   without invoking LLM, then aborts on a planted surprise.
5. Wake smoke: a planted surprise triggers the Wake prompt section and the
   surrogate LLM (mocked) edit raises simulator unit-test accuracy.
6. Sleep refactor smoke: 10 MCTS proposals produce ≤ `max_skills` after
   refactor, all wrappers preserve at least one validated child.
7. Existing prompt-contract test still passes (`verify_prompt_contract`).
8. New verification gate `verify_v3_intent` passes:
   `planner_active`, `simulate_step_present`, `option_execution_runs`,
   `wake_section_present`, `sleep_refactor_runs`.

## 6. Bounded Run Protocol

1. config: `all_on_generalized.yaml` with v3 keys.
2. cap: 80 non-reset actions / episode, 5 episodes.
3. record per-step: imagination top-path, committed skill name, predicted vs
   observed, surprise (yes/no), wake fired, sleep fired.
4. record per-episode: best level, action count, MCTS calls, wake count,
   sleep refactor count, top-3 skills, library size.
5. push state to HF prefix `bu_analysis_0416/` every 5 minutes.

## 7. 20-Minute Monitor Protocol

Every 20 minutes:

1. Read latest `summary.json` + `world_model.json` + `dreamcoder_library.json`.
2. Check three signals:
   - `transition_accuracy` trend (must rise or hold above 0.6 once it hits)
   - `falsification_probe_count + branch_escape_count` (must be > 0 by step 30)
   - `repeated_family_penalty / non_reset_actions` (must drop below v2's
     baseline of 9.5)
3. If two of three regress vs v2, kill the run, root-cause, edit, relaunch.
4. Otherwise let the episode finish; analyse on completion.

## 8. Stop Conditions

Pause and re-plan if:

1. After ≥ 3 monitor cycles, falsification_probe_count remains 0.
2. Wake produces ≤ 0.05 mean accuracy gain across 10 surprises.
3. MCTS top path mean simulator reward stays below random baseline for
   ≥ 5 imagination cycles.

## 9. Open Questions Pending Codex Critique

- Q1. Should the option-execution scope skip LLM tool calls entirely
  (history, frame_render, memories) or still allow read-only inspection?
- Q2. How to bootstrap MCTS on turn 1 when `predict_effect` is empty?
  Current proposal: H5b gate (require accuracy ≥ 0.6 first).
- Q3. Should imagination phase persist trees across turns or rebuild fresh?
  Current proposal: rebuild fresh per imagination call to avoid stale state.
- Q4. What should `propose_from_mcts` do when MCTS finds < K useful paths?
  Current proposal: propose 1 if reward > 0, else skip.
- Q5. How do we weigh agent-proposed skills (LLM `propose_skill`) vs
  MCTS-proposed skills? Current proposal: same scoring formula, but agent
  proposals get +1 prior for "depth>0 abstract framing".

# LS20 Observation-Progress-Falsification Plan

Date: 2026-04-16
Status: draft under codex review
Workspace: `/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research`
Primary game: `ls20`

## 1. Intent

The immediate research goal is no longer merely to prevent shallow early stops.
That host-level repair work has already improved runs from 2-3 non-reset actions
to substantially longer traces. The next goal is to make the agent spend those
actions on the right things:

1. discover unseen observation families rather than repeatedly validating the
   same opening corridor
2. falsify the current best hypothesis instead of only confirming it
3. make the world model predict structured observation deltas and progress
   surrogates rather than only change magnitude
4. reward skills for opening new state families and branch escapes, not only
   for contiguous support inside one action family

The intended outcome is a modular system where DreamCoder, world-modeling, and
meta-harness guidance jointly improve `ls20` exploration without hidden-state
channels or hand-written solver shortcuts.

## 2. Empirical Problem Statement

Current traces show a consistent failure pattern:

1. the agent reliably rediscovers an opening corridor beginning with
   `ACTION3, ACTION3`
2. it often follows the known continuation `ACTION4, ACTION1, ACTION1, ACTION2`
3. later actions branch, but mostly remain within the same state family
4. level progress remains at 0

The current system is therefore no longer primarily stop-limited. It is
search-limited and representation-limited.

## 3. Main Hypotheses

### H1. World-model target mismatch

Intent:
make the world model useful for search, not just for describing whether the
grid changed.

Hypothesis:
the current world model underperforms because it predicts `diff_magnitude` and
`expected_diff_band` rather than:

- what observable structure should change
- whether the result represents progress toward a new checkpoint or state family
- which action best distinguishes the current hypothesis from alternatives

Verification:

1. a retained world draft must emit all three prediction heads:
   - `observation_prediction`
   - `progress_prediction`
   - `action_recommendation`
2. smoke tests must verify those fields are present in normalized predictions
3. live runs must log explicit recommended action rationales that mention either
   unseen observations, branch discrimination, or progress surrogates

### H1b. Progress-variable blind spot

Intent:
make the agent track whether it is entering meaningfully new territory, not
just whether the grid changed.

Hypothesis:
the current system cannot tell "large visible change inside the same opening
family" from "smaller but more important change that enters a new observation
family" because it lacks explicit progress variables derived from observation
history.

Verification:

1. each observed step must log progress-side fields such as:
   - `new_signature`
   - `new_family`
   - `branch_escape`
   - `repeated_family_depth`
2. the world model and DreamCoder scoring must read those fields directly
3. smoke tests must confirm lower-diff branch escape can beat higher-diff
   same-family continuation

### H2. Exploration objective mismatch

Intent:
replace corridor confirmation with targeted unseen-observation acquisition.

Hypothesis:
the current explorer/tester prompts underperform because they ask for
informative probing in general, but do not explicitly prioritize:

- visiting states with unseen observation signatures
- collecting observations that would falsify the current strongest hypothesis
- choosing actions that maximally separate competing hypotheses

Verification:

1. prompt overlay and orchestrator instructions must explicitly mention:
   - unseen observation probe
   - falsification probe
   - branch-discriminating action
2. traces must contain explicit probe summaries of the form:
   - current hypothesis
   - unseen observation sought
   - falsifier sought
   - next disambiguating action
3. smoke tests must confirm these instructions are present in the prompt stack

### H2b. Rival-hypothesis collapse

Intent:
keep multiple plausible local theories alive until one is falsified.

Hypothesis:
the current loop underperforms because once one opening family gets modest
support, the rest of the search collapses around it. The agent confirms the
most convenient theory instead of maintaining rivals and running
discriminative probes.

Verification:

1. live traces must expose:
   - current best hypothesis
   - at least one rival hypothesis
   - the next discriminative probe
2. prompts and overlays must ask for falsifying evidence, not only confirming
   evidence
3. smoke tests must verify the prompt surface requires a rival hypothesis when
   uncertainty remains

### H3. Skill-scoring objective mismatch

Intent:
grow a library that opens new regions of state space rather than rephrasing one
successful opening.

Hypothesis:
the current DreamCoder scoring over-rewards contiguous support within a single
opening family and under-rewards:

- novel observation family entry
- recovery from surprise into a new branch
- loop escape
- action policies that improve hypothesis discrimination

Verification:

1. skill score inputs must include novelty/progress-related counters in addition
   to support and reuse
2. top-ranked skills must diversify away from only `33/3341/334112` wrappers
3. smoke tests must confirm that a skill grounded only in repeated opening
   support is not automatically ranked above a branch-escape skill with real
   novelty support

### H3b. Skill-promotion leakage

Intent:
prevent prefix memorization from becoming the dominant library content.

Hypothesis:
the current skill library over-promotes opening wrappers because any repeated
exact suffix can gain support faster than a more abstract but causally useful
skill. This causes routing to prefer locally familiar spines over skills that
improve falsification or branch escape.

Verification:

1. promotion rules must require at least one of:
   - new family evidence
   - branch escape evidence
   - surprise-recovery evidence
   - explicit observation-conditioned controller
2. exact spines may remain as evidence leaves, but cannot dominate routing
   without a stronger wrapper
3. smoke tests must verify repeated-prefix support alone is insufficient for
   top-rank promotion

### H4. Harness objective mismatch

Intent:
make the outer loop optimize for scientific exploration quality, not just for
activity.

Hypothesis:
the current meta-harness remains too generic because it rewards action diversity
and non-zero diffs, but not:

- novelty of observation families
- hypothesis-falsification attempts
- sustained progress without returning to the same family

Verification:

1. run metrics must include explicit exploration-quality measures such as:
   - new signature count
   - approximate new family count
   - falsification probes attempted
   - repeated-family penalty
2. overlay selection must respond to these metrics
3. smoke tests must verify these metrics are populated and scored

### H4b. Local attractor trap

Intent:
force the control loop to abandon locally comfortable but globally sterile
regions.

Hypothesis:
the current system gets trapped in a local attractor because neither the world
prior, skill router, nor harness imposes a penalty for repeated return to the
same signature family with low falsification yield.

Verification:

1. run-time state must track repeated-family pressure or local-attractor depth
2. overlays and action priors must escalate branch-escape pressure when that
   pressure is high
3. smoke tests must verify repeated same-family runs trigger an escape policy

## 4. Distinct Research Program

This plan is intentionally distinct from naive stacking of
DreamCoder + world-model + meta-harness:

1. DreamCoder is not treated as generic library growth. It is tied to
   branch-opening and hypothesis-discriminating utility.
2. The world model is not treated as a one-step change predictor. It is a
   three-head local model:
   observation delta, progress surrogate, and action choice.
3. The harness is not tuned for more activity or more surprise alone. It is
   tuned for unseen observation acquisition and falsification pressure.

The combined program is:

- observe
- predict
- falsify
- revise abstraction
- re-score search policy

rather than:

- explore
- summarize
- store another opening skill

## 5. Scientist Control State

The control loop must maintain explicit agent-side state:

1. `best_hypothesis`
   - the currently strongest local explanation
2. `rival_hypotheses`
   - at least one alternative explanation that predicts a different outcome
3. `target_unseen_observation`
   - the next observation family or signature the run is trying to reach
4. `target_falsifier`
   - what observation would weaken the current best hypothesis
5. `progress_variables`
   - per-step indicators such as `new_signature`, `new_family`,
     `branch_escape`, and `repeated_family_depth`
6. `local_attractor_pressure`
   - accumulated evidence that the run is circling one family without opening
     new territory

The plan is incomplete if any module improves local quality but fails to
publish information that updates this shared control state.

## 6. Concrete Design Changes

### P1. World Model Upgrade

Required changes:

1. Extend normalized predictions to include:
   - `observation_prediction`
   - `progress_prediction`
   - `action_recommendation`
   - `rival_predictions`
2. Extend bridge observations so post-action evaluation can score:
   - observation prediction correctness
   - progress prediction correctness
   - whether the action reached a new signature family
   - whether it falsified or supported the current best hypothesis
3. Replace action-prior ranking based mainly on `avg_diff` with a combined
   value that prefers:
   - unseen or under-tested observation families
   - progress surrogate gain
   - hypothesis discrimination
   - local-attractor escape
4. The world-draft contract must allow iterative code-like refinement of a
   simulator or predictor body, but the scoring target is the richer
   observation/progress/action contract rather than diff magnitude alone.

Keep condition:
world-model recommendations mention new-observation or falsification targets in
trace, not only high-diff continuation.

### P2. Explore / Tester Upgrade

Required changes:

1. Prompt the explorer/tester to choose at least one of three intents before
   each short probe block:
   - confirm current hypothesis
   - falsify current hypothesis
   - seek unseen observation
2. Require probe reports to include:
   - current best hypothesis
   - rival hypothesis
   - target unseen observation
   - target falsifier
   - why the next action is discriminating
3. Teach the orchestrator to prefer follow-up on unresolved falsification or
   unseen-observation targets before repeating the strongest opening family
4. Add an explicit local-attractor escape rule: if repeated-family pressure is
   high, the next probe block must prioritize branch escape over confirmation.

Keep condition:
trace reports mention explicit unseen/falsification targets and later actions
match them.

### P3. DreamCoder Upgrade

Required changes:

1. Track novelty support separately from opening-family support
2. Track whether a skill produced a new signature family, new branch family, or
   surprise-resolving recovery
3. Penalize near-duplicate opening wrappers when they do not improve branch
   escape or progress surrogate quality
4. Add a promotion gate:
   - evidence leaf spines may be stored
   - reusable promoted skills must carry an observation-conditioned wrapper or
     concrete novelty / falsification evidence
   - routing priority must demote evidence-only spines unless the wrapper
     conditions match

Keep condition:
top-ranked skills include branch-opening and falsification-oriented entries, not
only opening-corridor variants.

### P4. Meta-Harness Upgrade

Required changes:

1. Add run metrics:
   - exact new signature count
   - approximate family novelty count
   - falsification probe count
   - repeated-family penalty
   - branch-escape count
   - unresolved-rival count
2. Add overlay candidates specialized for:
   - unseen observation pursuit
   - falsification before confirmation
   - abandoning an over-supported opening family
   - maintaining and testing rival hypotheses
3. Make `before_action()` non-passive when local-attractor pressure is high by
   publishing escape guidance or discriminative probe pressure through the
   bridge/prompt surface.

Keep condition:
selected overlays shift when novelty/falsification metrics are poor.

## 7. Discriminative Probe Policy

Every short probe block must answer:

1. what is the current best hypothesis
2. what is the best rival
3. what observation would separate them
4. which next action is most likely to separate them
5. what fallback action will be taken if the expected discriminator fails

If the run cannot name a discriminator, it must explicitly seek an unseen
observation family rather than replaying the strongest known opening.

## 8. Skill Promotion Gate

Promotion policy:

1. evidence leaves are allowed
   - exact action spines may be stored as evidence or fallback primitives
2. promoted reusable skills require one of:
   - observation-conditioned precondition
   - new family / branch escape evidence
   - surprise-recovery evidence
   - explicit falsification utility
3. routing priority favors:
   - structured wrappers with causal conditions
   - skills with new-family or branch-escape evidence
   - surprise-resolving or falsification utility
4. routing priority disfavors:
   - duplicate opening wrappers
   - exact prefixes with no branch-opening evidence
   - generic labels without scorable conditions

## 9. Local Attractor Escape Policy

When the run repeatedly returns to the same family without opening new
observations:

1. world priors must penalize same-family continuation
2. DreamCoder routing must demote exact-prefix continuation
3. meta-harness overlays must switch to branch-escape / falsification guidance
4. prompts must ask for a rival or unseen target before another confirmation
   attempt

## 10. Smoke / Critic Gates Before Any New Experiment

No new autoresearch experiment may run until these checks pass:

1. Unit tests pass for orchestrator and research modules
2. World-model normalization test confirms the three-head prediction structure
3. Prompt tests confirm unseen-observation and falsification instructions appear
4. Skill-scoring tests confirm novelty counters affect ranking
5. Meta-harness tests confirm novelty/falsification metrics are logged
6. Routing tests confirm same-family high-diff continuations can be demoted in
   favor of branch escape
7. Promotion-gate tests confirm repeated prefix evidence alone does not create
   top-rank promoted skills

## 11. Bounded Verification Metrics

The first bounded runs after implementation changes will be judged by:

1. non-reset actions before stop
2. best level reached
3. new exact signature count
4. approximate new family count
5. falsification probes attempted
6. number of top-ranked skills that are not opening-corridor rephrasings

Improvement can be accepted before level-up if it clearly improves metrics 3-6
without regressing 1-2.

## 12. Immediate Execution Order

1. revise this plan after codex critique
2. implement world-model prediction-head upgrade
3. implement explorer/tester falsification and unseen-observation upgrade
4. implement DreamCoder and meta-harness scoring upgrades
5. run smoke tests and critic loop until no critical issue remains
6. resume bounded autoresearch runs

## 13. Stop Conditions

Pause implementation changes and re-plan if:

1. smoke tests cannot be made stable without broad architecture drift
2. changes preserve action count but do not improve novelty/falsification metrics
3. the system still re-collapses into the same opening family after the new
   metrics are live

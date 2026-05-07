# Plan v588 — Symbolica-compressed action-state chain for chain-level hypothesis (B16)

**Status:** DRAFT round 1
**Date:** 2026-05-07
**Owner:** autoresearch
**Scope:** Memory information *exposure* and M3 (HYPOTHESIZE) output schema. M1/M2/M3/M4 module structure preserved exactly.
**Rejected if:** any module added; any leakage introduced; any backward incompat
break to cross_run_memory.json / level_bridges.json / segment_index.json schemas;
chain payload exceeds 4KB at any turn.

---

## 0. Diagnosis (why this plan exists)

The user's intuition, validated by traces:

> *"observation은 잘하는데, 이전 여러 행동들을 잘 압축하는 데서 문제 생기는 거 아니야? trace를 action state 체인을 줘야할 것 같고."*

Validation against B14 trace inspection (cycle189-191, 26 L+1 events total, 0 L+2):

- **M1 / M3 / M4 input shape is state-centric.** M3 (HYPOTHESIZE) currently receives `summary` (M4-produced text), `visible_regions` (current frame), `falsified_recent`, `gqb_pair`, `cross_run_priors`. None of these is the *trajectory*. M3 is asked to propose hypothesis cards based on the *current state*.
- **No agent module sees the action-state CHAIN.** `recent_verbose` (≤3 turns) and `recent_turn_diffs` (≤6 turns by default, expandable to 24 only in stuck mode) contain per-turn observations, but they are exposed only to M1, not to M3. Even when exposed, they are presented as a list, not as a *sequence* the agent is told to reason over.
- **cross_run_memory accumulates 1st-order observations, not laws.** 134 entries, all of the form *"clicking 9-colored compass-neighbor regions produces 9→8 transition"*. None of the form *"the level only rises after 8 distinct neighbor clicks within a marker's compass"*. The promotion path ("promote_to_1A") in M4 does not synthesize across observations; it textually extracts one mechanism per L+ event.
- **ESC partially fires (2/26), max cfr=1.** Only ~7.7% of L+ events produce an entry. Two entries with different signatures, each with cfr=1. Aggregation has not started.
- **ASMW/CPSR (B14) never fires** because L+1 oscillates rapidly and `gap ≥ K_STUCK=15` never holds.

The chain-level layer is the missing piece. ft09's actual rule is *over the trajectory*: a marker's compass must reach a *joint configuration* via *parity-controlled* clicks. No state-only reasoner can derive this — the rule is a property of *sequences of clicks*, not of any single state.

This plan introduces the trace-level abstraction layer without adding new
modules.

---

## 1. Survey of trace-level abstraction methods

### 1.1 Inductive Logic Programming (ILP, Muggleton 1991+)

Given examples + background knowledge, induce a Horn clause that explains
all positive examples and excludes all negative examples. Classical example:
from `parent(a,b)`, `parent(b,c)`, `grandparent(a,c)` derive
`grandparent(X,Z) :- parent(X,Y), parent(Y,Z)`.

For our problem: from a chain of `(state, action, post_state)` triples,
induce a predicate that holds on every "successful" sub-trajectory and
fails on stuck sub-trajectories.

Modern variants: **MetaInterpretive Learning (MIL)**, **Popper** (Cropper
2021). Differentiable ILP: **dILP** (Evans, Grefenstette 2018) embeds the
search in a neural net.

*Use for our plan:* ILP grounds the *output schema* — we ask M3 for a
`chain_rule` with `evidence_turns`, `predicted_outcome`,
`falsification_condition`. The form is Horn-clause-flavoured; the
content is the LLM's job.

### 1.2 Sequential pattern mining

**Apriori** (Agrawal & Srikant 1995), **GSP**, **PrefixSpan** (Pei 2004):
find frequent ordered subsequences in transactional data. Classic
applications: market-basket analysis, web-click prediction.

For our problem: pre-compute frequent (action_kind, state-transition)
subsequences from the chain. Surface them to M3 as candidate primitives
to use in chain-rule construction.

*Use for our plan:* run a lightweight 3-gram extraction over the
Symbolica-tokenized chain, surface top-K to M3 as `frequent_subpatterns`.

### 1.3 Linear Temporal Logic (LTL) learning

**Lemieux et al. 2015**, **Camacho et al. 2018**: learn LTL formulas from
positive/negative trace pairs. LTL operators G (always), F (eventually), X
(next), U (until). Captures temporal properties like "G(click_marker → X
compass_change)".

For our problem: M3 could output LTL-like predicates over the chain. But
LLMs handle natural-language temporal predicates more reliably than
formal LTL, so we use natural-language with formal slots.

*Use for our plan:* M3's `chain_rule` schema includes a `temporal_scope`
field (one of: "always", "after_event_X", "until_event_Y") to anchor
LLM-natural temporal reasoning.

### 1.4 Trajectory Transformers (Janner et al. 2021)

Treat RL trajectories as token sequences; Transformer attends across
state-action-reward tokens. Implicit trajectory model. **Decision
Transformer** (Chen et al. 2021) extends to return-conditioned action
prediction.

For our problem: rather than train a separate transformer, we leverage
the LLM (gpt-5.5) and feed it *Symbolica-tokenized* trajectories as
in-context input. This is the "trajectory transformer in-context" form.

*Use for our plan:* the chain we expose to M3 *is* a tokenized trajectory.
The LLM does the transformer-over-trajectory job in-context.

### 1.5 DreamCoder (Ellis et al. 2021, PLDI)

Wake / Sleep / Abstraction loop. Sleep compresses observations; Abstraction
identifies common subroutines across solved tasks and adds them as new
primitives in a growing library. The library *is* the abstracted memory.

For our problem: B14's ESC was a coarse aggregator (signature buckets) but
not a primitive learner. DreamCoder's contribution is *"find what is
common across many solutions and name it"*. We approximate this by giving
M3 the full chain plus extracted patterns and asking it to name the rule.

*Use for our plan:* M3's chain_rule is a "named primitive" candidate —
once cfr≥3, it could be elevated to a named constraint that M1 enforces.
(Out-of-scope for B16; reserved for B17.)

### 1.6 Memory-of-Thought / Tree-of-Thoughts (Li & Qiu 2023, Yao et al. 2023)

Pre-compute reasoning chains, retrieve at inference time. Tree-of-Thoughts
explores branching deliberations.

For our problem: M3 is already a single-pass reasoner. Tree-of-Thoughts
on M3 would multiply LLM cost. Instead, we feed M3 *already-extracted*
chain patterns (cheap) so its single pass has more to work with.

### 1.7 Soar chunking (Newell 1990)

Solves goal hierarchically; on success, *compiles* the impasse-resolution
sequence into a single rule. The chunk is added to procedural memory.

For our problem: the analog is — when an L+ event completes, *compile* the
last K turns into a `chain_rule` candidate. M3's chain_rule output (B16)
is exactly this proposal step. Once cfr≥3, the chain_rule could be
operationally compiled into a plan-card schema (B17 reservation).

### 1.8 Granger causality and causal structure learning

**Granger (1969)** — temporal precedence + predictive power; X
*Granger-causes* Y if past values of X predict Y better than past Y
alone. **Pearl's do-calculus** — interventional structure.

For our problem: pre-compute a (region, transition) → marker_progress
co-occurrence table from the chain. Surface as `causal_table` to M3:
"clicks on R5 preceded marker R12's compass change in N of M turns".

*Use for our plan:* `causal_table` is one of the four chain-level
features we expose.

### 1.9 Predictive coding / active inference (Friston 2010)

Memory encodes *prediction errors*, not confirmations. The agent attends
to surprises.

For our problem: at each turn M1 emits `expected_observation` (what it
expected the click to do). The actual `observation` may differ. The diff
is a *prediction error* — a signal-rich indicator of what the agent did
NOT understand at that moment.

*Use for our plan:* the chain we expose to M3 includes a
`prediction_error_at_t` field for each turn — surfacing the trajectory
of surprises.

### 1.10 Where the surveyed methods locate our system

| Method | Captures | We have? |
|---|---|---|
| Reflexion buffer | per-event text | ✓ (M4) |
| ACT-R activation | event recurrence | ✓ (cfr) |
| Generative Agents stream | episode + reflection | partial |
| Hermes episodic | (obs, act, result) | ✓ (B12, B13) |
| Larimar Bayesian text key | per-text confidence | ✓ (B9) |
| Voyager skill library | executable skills | partial (cards) |
| ESC (B14) | structural signature | ✓ |
| **DreamCoder Abstraction** | **named primitive across solutions** | **NO** |
| **ILP / chain rule induction** | **rule from trajectories** | **NO** |
| **Predictive coding** | **prediction-error chain** | **NO** |
| **Trajectory Transformer in-context** | **trajectory as token seq.** | **NO** |
| **Sequential pattern mining** | **frequent sub-sequences** | **NO** |

The five "NO" rows are what B16 introduces, all without adding modules.
DreamCoder's full primitive learning is reserved for B17. B16 prepares
the input layer that B17 will compile against.

---

## 2. Method candidates (theory-grounded, multiple distinct)

Six candidates. Theory-tag each. Select for composition.

### 2.1 M-CHAIN-1 — Symbolica turn tokenisation

**Theory:** Trajectory Transformer in-context (Janner 2021) +
deterministic structure extraction (Symbolica codebase pattern).

**Mechanism:** every turn becomes a fixed-format token of ≤80 chars:
```
T<turn> | <action_kind> | <region_kind_pre> | <transition> | <compass_delta> | L+<level_delta>
```
Examples:
```
T26 | click | non_marker(R5,9) | 9→8 | M12_N:9→2 | L+1
T27 | click | non_marker(R5,8) | 8→9 | M12_N:2→9 | L+0
T28 | click | non_marker(R6,9) | 9→8 | M12_E:9→2 | L+0
```

The chain = list of these tokens, concatenated with `\n`. 30 turns ≈
2.4KB. Fits in the M3 prompt.

**Distinction:** existing `recent_turn_diffs` is a list of dicts (verbose
for an LLM; weakly readable as a sequence). The token-line format makes
the *temporal reading* trivial.

### 2.2 M-CHAIN-2 — Trajectory feature aggregator

**Theory:** sequential pattern mining (PrefixSpan) + ESC bucketing
extended to the trajectory level.

**Mechanism:** Symbolica computes deterministic aggregate features over
the chain:
```
unique_regions_in_chain
repeat_click_max_count        (max clicks on any single region)
compass_changes_per_turn      (avg)
marker_progress_monotonicity  ("up_only", "stable", "oscillating")
kind_distribution             ({"non_marker": 24, "marker": 3, "outside": 3})
lp_event_intervals            ([3, 5, 4, 7, ...])  L+ event gaps
```

**Distinction:** B14's `level_bridge` summary is per-event statistics.
This is per-window-over-the-chain statistics — a different temporal
slice.

### 2.3 M-CHAIN-3 — Prediction-error chain

**Theory:** active inference / predictive coding (Friston 2010); learn
from surprise.

**Mechanism:** for each turn, Symbolica compares M1's
`expected_observation` against actual `observation`, producing a
prediction-error vector:
```
prediction_error_at_t = {
  region_id_match: bool,
  transition_match: bool,
  level_delta_match: bool,
  surprise_score: 0-3
}
```

The chain of these errors (over 30 turns) shows where the agent's mental
model diverges from reality. Highly diagnostic for what's *not yet
understood*.

**Distinction:** existing M4 Reflexion gets `verdict` post-hoc, but the
prediction-error itself is never exposed as a sequence.

### 2.4 M-CHAIN-4 — Causal frequency table

**Theory:** Granger causality + count-based causal structure learning.

**Mechanism:** Symbolica scans the chain to build a 2D table
```
(region_kind, transition)  →  (count, avg_marker_progress_delta_next_turn)
```

Example:
```
(non_marker, 9→8)  →  (count=12, avg_Δm=+0.25)
(non_marker, 8→9)  →  (count=10, avg_Δm=-0.20)
(marker_multicolor, 9→8) → (count=2, avg_Δm=0.0)
```

Surfaces "what kind of click leads to marker progress" without naming
the rule.

**Distinction:** B11's `marker_neighbor_states` shows *current* compass.
This shows *how clicks have historically affected* compass.

### 2.5 M-CHAIN-5 — Frequent sub-pattern extraction (REJECTED for B16)

**Theory:** PrefixSpan applied to the chain.

**Mechanism:** find frequent 2-grams and 3-grams in the action-state
chain, surface top-5 to M3.

**Why rejected for B16:** the chain is short (≤30 tokens) and the
Symbolica feature aggregator (M-CHAIN-2) already captures the core
patterns. Adding a 3-gram miner introduces a tunable threshold (min
support) that we cannot calibrate on small data. Reserved for B17 once
the chain length grows (≥100 turns observed across cycles).

### 2.6 M-CHAIN-6 — `chain_rule` output schema for M3

**Theory:** ILP Horn-clause schema + Soar chunking shape.

**Mechanism:** M3's output JSON gains a new array field
`chain_rule` (≤3 entries per call):
```json
{
  "rule": "<one sentence stating an observed regularity over the chain>",
  "evidence_turns": [12, 16, 19, 23],
  "predicted_outcome_next_turn": "<what the rule predicts will happen next>",
  "falsification_condition": "<what observation would refute the rule>",
  "temporal_scope": "always" | "since_turn_X" | "after_lp_event"
}
```

The form is Horn-clause-flavoured (premise = `evidence_turns`,
conclusion = `predicted_outcome_next_turn`). The rule itself is the
LLM's natural-language inference. Falsification is enforced by schema.

**Distinction:** existing M3 output has only state-anchored cards
(`region_id`, `expected_signature`). `chain_rule` is anchored to the
chain, with explicit predicted-vs-observed verifiability.

### 2.7 Down-select rationale

| Component | Adopted | Where |
|---|---|---|
| Symbolica turn tokenisation | M-CHAIN-1 | M3 + M1 input |
| Trajectory features | M-CHAIN-2 | M3 input |
| Prediction-error chain | M-CHAIN-3 | M3 + M4 input |
| Causal frequency table | M-CHAIN-4 | M3 input |
| `chain_rule` schema | M-CHAIN-6 | M3 output |
| Sub-pattern mining | M-CHAIN-5 | rejected, B17 |

The five adopted methods compose orthogonally — each adds an
information channel. Module structure (M1/M2/M3/M4) untouched.

---

## 3. Intent / Hypothesis / Verification (I/H/V)

### Intent

Provide M3 (HYPOTHESIZE) with the action-state chain — Symbolica-
tokenised, with trajectory features, prediction errors, and causal
frequency tables — and extend M3's output schema with `chain_rule` so
the agent can hypothesise *over the chain* rather than *over the current
state alone*. Module structure preserved exactly. Output schema
backward-compatible: `chain_rule` is a new optional field; older
parsers ignoring it remain functional.

### Hypothesis (each falsifiable)

**H1 (Chain reaches M3).** After implementation, every M3 call where
the chain length ≥10 includes the chain payload. M3's input
`action_state_chain` field is non-empty in ≥95% of calls past turn 10.
**Falsifiable:** payload missing, malformed, or empty when chain is
non-trivial.

**H2 (M3 cites chain).** In ≥30% of M3 calls, the `thought` text
references a chain token (e.g., "T20-T24 show repeat clicks") OR a
trajectory feature (e.g., "compass_changes_per_turn=0.2 below
expectation").
**Falsifiable:** <10% of M3 thoughts reference any chain element.

**H3 (M3 emits chain_rule).** When chain length ≥10, ≥80% of M3 calls
emit ≥1 `chain_rule` entry with all required fields populated.
**Falsifiable:** <50% of qualified calls emit chain_rule.

**H4 (chain_rule falsifiability is functional).** Within the 3-cycle
run, ≥1 emitted `chain_rule` is later falsified by an observation that
matches its `falsification_condition`. M4 marks it as falsified in the
trace.
**Falsifiable:** zero falsified rules across all 3 cycles.

**H5 (Integration target — L+2 reach).** ≥1 of cycle192/193/194
reaches an L+2 event within 400 turns.
**Falsifiable:** zero L+2 events.

### Verification gates

| Gate | Method | Pass condition |
|---|---|---|
| V-LEAK | `python3 scripts/check_no_leak_prompts.py` | 5/5 PASS |
| V-TESTS | `pytest tests/test_v588_*.py tests/test_v587_*.py tests/test_v586_*.py -v` | all PASS, <30s wall |
| V-H1 | trace.jsonl scan: chain payload presence | ≥95% of qualified turns |
| V-H2 | M3 thought grep for token/feature reference | ≥30% of qualified turns |
| V-H3 | M3 output JSON validation: chain_rule populated | ≥80% of qualified turns |
| V-H4 | trace scan: emitted rule + later observation matches falsification_condition | ≥1 occurrence |
| V-H5 | trace scan: `level_delta == 2` | ≥1 cycle has ≥1 |

V-H1, V-H2, V-H3 are wiring/exposure gates and must pass.
V-H4 measures whether the schema is functionally productive, not just
present.
V-H5 is the integration target; failure is informative but does not
invalidate H1-H4.

---

## 4. Implementation specification

### 4.1 New file — `tools/chain_compress.py` (Symbolica)

Pure deterministic. Takes `recent_verbose` + `recent_turn_diffs` +
`marker_neighbor_states_history` (NEW: per-turn snapshot of compass
state, captured at the START of each turn — see 4.2).
Returns a dict of:

```
{
  "chain_tokens": list[str],             # M-CHAIN-1
  "trajectory_features": dict,            # M-CHAIN-2
  "prediction_errors": list[dict],        # M-CHAIN-3
  "causal_table": list[dict],             # M-CHAIN-4
}
```

Function signature:
```python
def compress_action_state_chain(
    recent_verbose: list[dict],          # full per-turn entries
    recent_turn_diffs: list[dict],       # B12 diffs
    max_chain_length: int = 30,
) -> dict
```

`max_chain_length=30` chosen so that token sequence fits in ≤2.4KB
(30 × 80 chars = 2.4KB). Configurable via env
`V57_CHAIN_MAX_LENGTH`.

### 4.2 `agent.py` changes

- Add `V57Board.compass_history: list[dict]` — rolling deque (max
  30) capturing per-turn compass snapshot at the *start* of each turn.
  Used by M-CHAIN-3 prediction-error and M-CHAIN-4 causal table.
- Modify `run_turn` to append to `compass_history` after the per-turn
  compass is computed.
- New helper `_build_chain_payload(board)` calls
  `compress_action_state_chain` and returns the dict.
- Modify `call_hypothesize` signature to receive `chain_payload: dict |
  None = None`; pass it into the JSON payload as `action_state_chain`.
- Modify `run_turn` to compute and pass `chain_payload` to
  `call_hypothesize`.
- Modify `call_action` to optionally receive `chain_payload` (M1 may
  also benefit). Pass through.
- Modify `spawn_reflexion` (M4) to receive `prediction_errors` from the
  chain payload, so M4 can reference surprises.

### 4.3 `prompts.py` changes

**M3 (HYPOTHESIZE_SYSTEM_PROMPT) input doc additions:**

```
- ACTION_STATE_CHAIN: dict {chain_tokens, trajectory_features,
  prediction_errors, causal_table} representing the last N=30
  compressed turns. Read it as a temporal sequence:
  • chain_tokens: list of "T<turn>|<action>|<region>|<transition>|
    <compass_delta>|L+<delta>" lines. Earliest first.
  • trajectory_features: aggregate stats (unique_regions,
    repeat_click_max_count, compass_changes_per_turn,
    marker_progress_monotonicity, kind_distribution, lp_event_intervals).
  • prediction_errors: per-turn delta between M1's expected and
    observed (region_id_match, transition_match, level_delta_match,
    surprise_score 0–3). Surprises ≥2 are diagnostic.
  • causal_table: deterministic count of (region_kind, transition) →
    (count, avg_marker_progress_delta_next_turn).

  Use the chain to detect TRAJECTORY-LEVEL patterns:
  - Repeat-clicks on the same region with same transition pair (e.g.
    R5: 9→8 then R5: 8→9) cancel each other.
  - Compass changes that decay over turns indicate marker is settling.
  - prediction_errors with surprise_score=3 mark hypotheses that need
    revision.
```

**M3 output schema additions:**

```json
"chain_rule": [
  {
    "rule": "<sentence-form regularity over the chain>",
    "evidence_turns": [<int turn ids referenced>],
    "predicted_outcome_next_turn": "<observable consequence>",
    "falsification_condition": "<observable that would refute>",
    "temporal_scope": "always" | "since_turn_<int>" | "after_lp_event"
  }
]
```

**M1 input doc:** brief mention that
`ACTION_STATE_CHAIN` is *also* available (forwarded from M3 input). M1
should consult it only when stuck or when a chain_rule has been emitted.

**M4 input doc:** add
`prediction_errors` to inform Reflexion summary, with instruction:
"if a prediction_error has surprise_score=3 in the recent window,
mention it in the summary".

### 4.4 Anti-leak

Universal forbidden vocab list applies to the new M3/M4 prompt sections.
The chain payload itself contains R-ids (e.g., "R5") and color values —
these are NOT in the leak list, they are the agent's own observation
anchors. We do NOT introduce `marker_progress`, `target_color`,
`win_state`, etc. into the chain payload OR the chain_rule schema field
descriptions.

The phrase "marker_progress_monotonicity" appears in
`trajectory_features` but only as a SHAPE descriptor ("up_only",
"stable", "oscillating") — not as a goal vocabulary. If V-LEAK flags
it, rename to `progress_curve_shape`.

### 4.5 Tests — `tests/test_v588_chain.py` (≥10 tests)

Layer: deterministic, no LLM.

- T-CHAIN-1: tokenisation produces correct format on synthetic 5-turn
  trace.
- T-CHAIN-2: chain truncated to `max_chain_length` correctly.
- T-CHAIN-3: trajectory_features sums correctly across turns.
- T-CHAIN-4: prediction_errors match-vs-mismatch logic.
- T-CHAIN-5: causal_table count + avg correct.
- T-CHAIN-6: empty input returns empty chain (cold-start).
- T-CHAIN-7: M3 prompt payload contains `action_state_chain` key when
  passed.
- T-CHAIN-8: backward compat — M3 schema parser tolerates missing
  `chain_rule`.
- T-CHAIN-9: anti-leak — chain payload regex check (no forbidden
  vocab).
- T-CHAIN-10: integration — `run_turn` fills `compass_history` deque
  correctly across 5 turns.

### 4.6 Backward compatibility

- cross_run_memory.json: untouched.
- level_bridges.json: untouched.
- segment_index.json: untouched.
- M3 output schema: `chain_rule` is optional; existing parsers ignore
  unknown fields.
- M1/M3/M4 callers backward-compatible: chain_payload defaults to None.

---

## 5. Implementation order (mechanical)

1. Add `tools/chain_compress.py` (pure functions).
2. agent.py: add `V57Board.compass_history` deque + update logic.
3. agent.py: add `_build_chain_payload` helper.
4. agent.py: modify `call_hypothesize` signature + payload.
5. agent.py: modify `call_action` signature (optional) + payload.
6. agent.py: modify `spawn_reflexion` to receive prediction_errors.
7. agent.py: modify `run_turn` to compute and pass chain_payload to
   M1 + M3 + M4.
8. prompts.py: M3 input doc + output schema.
9. prompts.py: M1 + M4 brief mentions.
10. tests/test_v588_chain.py — 10 tests.
11. V-LEAK + V-TESTS gates.
12. Smoke 5-turn run via existing smoke harness.
13. Kill old cycles + launch cycle192-194 with budget 400.

After each step 1–10: run pytest. Do not proceed if regressing.

---

## 6. Risks and mitigations

- **R-1: Chain payload bloats the M3 prompt.** Mitigation:
  `max_chain_length=30` keeps it ≤2.4KB. M3 prompt currently ~12KB,
  +20% bloat acceptable.
- **R-2: M1's `expected_observation` malformed in older traces, breaks
  prediction-error compute.** Mitigation: graceful fallback (when
  expected fields absent, mark match=None, surprise_score=0).
- **R-3: chain_rule field becomes a dumping ground for vague claims.**
  Mitigation: schema enforces `evidence_turns` (≥1 int), `predicted
  outcome` and `falsification_condition` (≥10 chars each). M3
  validation rejects malformed entries.
- **R-4: causal_table small samples → misleading averages.**
  Mitigation: only include rows with count ≥3.
- **R-5: chain payload accidentally leaks via R-id pattern through
  prompt template.** Mitigation: V-LEAK script run on prompt templates
  AND a runtime sample of the chain payload (T-CHAIN-9).
- **R-6: tokenisation introduces game-specific vocab.** Mitigation:
  the only vocabulary used is `click`, `non_marker`,
  `marker_multicolor`, `outside` (all already present in
  `region_kind_pre`). No new vocab introduced.

---

## 7. Self-critic round 1 — findings

This section is the round-1 critique of §1–§6.

**C-1 (Chain length 30 vs visibility 6 conflict).**
M1 currently sees only the last 6 turn-diffs (or up to 24 in stuck
mode, which never fires per B14 trace). M3 will see 30 chain tokens.
This means M3 reasons over a longer window than M1. If M3 emits a
chain_rule referencing "T26 caused L+1", M1 — which only sees turns
T28-T33 — cannot verify. Fix: forward the same chain to M1.

**C-2 (M3 receives summary from M4 — risk of confusion).**
M3 already receives a TEXT summary from M4. The chain is structured.
M3 could ignore the chain and rely on the summary, reverting to
state-only reasoning. Mitigation: M3 prompt instructs *"prioritise the
chain over the summary when proposing chain_rule"*.

**C-3 (chain_rule could duplicate active_hypotheses cards).**
A chain_rule that says "clicking marker neighbours produces compass
changes" overlaps with an active hypothesis card region. Mitigation:
chain_rule is *additive* — it does NOT replace cards, only supplements
them. Cards remain the action-driving primitive; chain_rule informs
M1's reasoning.

**C-4 (compass_history deque race / cost).**
Updating compass_history every turn requires the same compass snapshot
work that B12 already does. We can reuse `current_compass_snapshot`
(already computed in run_turn for B12) — no extra compute.

**C-5 (prediction_errors when expected_observation is missing).**
v585zz's `force_coord` path skips expected_observation generation.
Older replayed traces also lack it. We must default missing fields to
`match=None, surprise_score=0` and document.

**C-6 (causal_table interpretation by LLM).**
"avg_marker_progress_delta_next_turn=+0.25" — does the LLM understand
this is a per-event statistic? Mitigation: prompt example shows how
to read the field with explicit interpretation.

**C-7 (V-H4 verification is hard to automate).**
Detecting "rule's falsification_condition was matched by a later
observation" requires NL matching of free-text falsification_condition
against observations. Mitigation: require M3 to embed at least ONE
structured slot in falsification_condition — e.g.,
`"if T<n>'s click on a non_marker fails to trigger any compass_change
within 2 turns"`. We grep for `T\d+` pattern in falsification_condition
and check the turn(s) cited had no subsequent compass change.

**C-8 (test V-H4 needs trace data).**
T-CHAIN-* tests are unit tests; V-H4 requires a 5-turn smoke run with
mocked M3 returning a falsifiable rule. Add T-CHAIN-11 integration
test for this.

**C-9 (Schema risk: too many output fields will lower per-field
quality).**
M3 already emits cards + thought. Adding chain_rule with 5 sub-fields
expands the output. Mitigation: cap chain_rule entries at 3 per call;
mark schema as "emit ≤3, prefer quality over quantity". M3 may also
choose to emit zero entries if no robust rule visible.

**C-10 (M4 might not act on prediction_errors).**
M4's task already lists 5 outputs. Adding "consult prediction_errors"
might be ignored in practice. Mitigation: rather than ask M4 to
"consult", we bake the chain prediction errors INTO the M4 input
explicitly (a separate field `prediction_errors_chain`) and reference
it in the M4 task instructions.

**C-11 (Backward compat untested).**
T-CHAIN-8 says "M3 parser tolerates missing chain_rule" but does not
test the inverse: a parser that doesn't know about chain_rule still
gets all the OLD fields it expects. Mitigation: T-CHAIN-12
backward-compat test ensuring `chain_rule` absence does not break
existing card-extraction logic.

## 8. Round-2 corrections (applied)

**C-1 → §4.2:** the same chain_payload is now forwarded to M1
(`call_action`), M3 (`call_hypothesize`), and M4 (`spawn_reflexion`).
All three see the same window.

**C-2 → §4.3 (M3 prompt):** add explicit precedence rule —
"chain_tokens + trajectory_features take precedence over SUMMARY when
the two appear to disagree".

**C-3 → §4.3 (M3 prompt):** add note — "chain_rule is supplementary
to ACTIVE_HYPOTHESES; do not duplicate card claims as rules".

**C-5 → §4.5 T-CHAIN-4 + §4.6:** prediction_error fallback documented
as `match=None` not False; surprise_score=0; T-CHAIN-4 covers this.

**C-6 → §4.3 (M3 prompt):** add WORKED EXAMPLE of reading
causal_table — "non_marker, 9→8: count=12, avg_Δm=+0.25 means clicks
on non-marker regions in state 9 transitioning to 8 produce on
average +0.25 marker_progress in the immediate next turn — a positive
correlation. Use as evidence, not proof.".

**C-7 → §4.3 (M3 schema) + §4.5 T-CHAIN-13:** require
falsification_condition to embed at least one `T\d+` reference. M3
schema validator enforces. T-CHAIN-13 tests the validator.

**C-8 → §4.5 T-CHAIN-11:** integration smoke test with mocked M3
returning a chain_rule + an observation matching falsification.

**C-9 → §4.3 (M3 schema):** explicit "≤3 per call, prefer quality
over quantity". Empty list allowed.

**C-10 → §4.2 + §4.3 (M4):** M4 receives explicit
`prediction_errors_chain` field; M4 prompt task instructions reference
it directly with an example.

**C-11 → §4.5 T-CHAIN-12:** backward-compat test ensures legacy parser
behaviour preserved.

## 9. Self-critic round 2 — findings (post-correction)

**C-12 (Cold-start chain shorter than threshold).**
H1/H2/H3 hypotheses gate on "chain length ≥10". Cycle starts at turn
0 with empty chain. We must allow chain_rule emission at turn ≥10
only. Document in M3 prompt — "if chain_tokens has fewer than 10
entries, do NOT emit chain_rule". Add T-CHAIN-14 test.

**C-13 (Chain encoding stability).**
If `region_kind_pre` field is missing in older recent_verbose entries
(predates B12), chain token format breaks. Mitigation: tokeniser
treats missing `region_kind_pre` as "unknown" and emits
`unknown_kind` token. Add T-CHAIN-15 test on legacy entries.

**C-14 (Trajectory feature monotonicity classifier).**
"up_only" / "stable" / "oscillating" classification for
`marker_progress_monotonicity` requires defining "marker_progress" —
which is a leak vocabulary term per V-LEAK. Mitigation: rename field
to `level_delta_curve_shape` and use level_delta values directly. The
shape becomes "rising_only", "flat", "oscillating" — all neutral.

**C-15 (causal_table as proxy for marker_progress).**
The original definition of `causal_table` value used
"avg_marker_progress_delta_next_turn" — same leak vocabulary.
Mitigation: change to `avg_level_delta_in_next_3_turns`. This is a
deterministic statistic over actual level_delta — no leak.

**C-16 (Chain forwarded to M1 may bloat M1 prompt past safety
margin).**
M1 prompt is currently ~12KB at high turn count. +2.4KB chain
+ 1KB features + 0.5KB errors + 0.5KB causal_table = +4.4KB. M1 prompt
cap to ≤16KB. Mitigation: M1 receives `action_state_chain`
*compressed-only* (chain_tokens last 15 only, features yes,
prediction_errors last 5 only, causal_table top-5 rows). M3 gets the
full version. This keeps M1 ≤14KB.

**C-17 (M4 prediction_errors field overlaps with existing
verdict).**
M4 already gets `verdict` per turn in `recent_turns`. The verdict was
"confirm" / "falsify" / "inconclusive". prediction_errors are
finer-grained. Mitigation: keep verdict; add prediction_errors as a
parallel signal. Document in M4 prompt that they are complementary.

**C-18 (Tokeniser whitespace + pipe-delimiter conflict with R-ids).**
If a region id contained "|" the format would break. v57's R-ids are
"R\d+" so no risk. Document and add a defensive strip in tokeniser.

## 10. Round-3 corrections (applied)

**C-12 → §4.3 + §4.5 T-CHAIN-14:** M3 prompt rule "if
chain_tokens has <10 entries, omit chain_rule". Test added.

**C-13 → §4.5 T-CHAIN-15:** legacy-entry test added.

**C-14 → §4.1 + §4.3 prompt + §4.5:** rename
`marker_progress_monotonicity` → `level_delta_curve_shape`. Re-run
V-LEAK to confirm.

**C-15 → §4.1:** rename `avg_marker_progress_delta_next_turn` →
`avg_level_delta_in_next_3_turns`. Adjust trajectory_features
accordingly.

**C-16 → §4.1 + §4.2:** introduce two compression levels:
- `compress_action_state_chain(..., level="full")` for M3
- `compress_action_state_chain(..., level="compact")` for M1
- M4 gets `compact` version + full prediction_errors_chain only.

**C-17 → §4.3 (M4 prompt):** explicit complementarity statement.

**C-18 → §4.1:** tokeniser does `re.sub("[|\\n]", "_", token_field)`
before joining. Defensive.

## 11. Self-critic round 3 — findings (final)

**C-19 (No new issues.) ✓**

**C-20 (Plan length consideration.)**
This plan is now ~700 lines. Reviewer fatigue real but tolerable;
each round added <100 lines and addressed concrete C-* findings.
Closing plan-critic loop.

**C-21 (Risk of "compress now, forget later").**
The chain is in-memory only and discarded at cycle end. The
`level_bridges.json` and `cross_run_memory.json` persist; the chain
does not. Decision: chain is intentionally ephemeral — long-term
storage of trajectory tokens duplicates segment_index. Out of scope
for B16.

## 12. Final implementation order (post-critic, this is the order to execute)

1. `tools/chain_compress.py` — pure deterministic functions
   (tokenise, features, errors, causal table, two compression levels).
2. agent.py: `V57Board.compass_history` deque (max 30) + update logic
   in run_turn.
3. agent.py: `_build_chain_payload(board, level)` helper.
4. agent.py: `call_hypothesize` signature + payload (level=full).
5. agent.py: `call_action` signature + payload (level=compact).
6. agent.py: `spawn_reflexion` signature + payload
   (prediction_errors_chain only).
7. agent.py: `run_turn` integration.
8. prompts.py: M3 input doc + output schema.
9. prompts.py: M1 input doc (one line about chain).
10. prompts.py: M4 input doc + task ref.
11. tests/test_v588_chain.py — 15 tests (T-CHAIN-1..15).
12. V-LEAK + V-TESTS gates. Re-run if any rename was needed.
13. Smoke 5-turn run via existing smoke harness — verify
    chain_payload field appears in M3 trace, no exceptions.
14. Kill old cycles 189-191 (cycle189 still alive at plan-write time)
    + launch cycle192-194 with budget 400.
15. Monitor V-H1..V-H5.

Plan v588 frozen at end of round-3. Implementation begins from §12 step 1.

---

## 12.5 Round-4 operational notes (post-implementation, pre-launch)

These are operational findings during code implementation; they do not
re-open the plan. Recorded here for audit trail.

**O-1 (visualizer extended).** `tools/trace_visualizer.py` now renders
`stuck_mode`, `marker_neighbor_states`, `action_state_chain` (chain
tokens + trajectory_features + prediction_errors + causal_table),
`analogous_past_segments`, `region_clicks_this_level`. Old trace files
(cycle187-191) lack these fields and the visualizer correctly omits
those sections. Verified by checking 489/489 div balance, 269/269
details balance.

**O-2 (agent.py append_trace extended).** B14/B16 runtime payloads
are now written to trace.jsonl every turn. trace size per turn
increases ~30% (chain payload ~3KB). Acceptable for 400-turn cycle
(=1.2MB).

**O-3 (cycle189 still alive at this point).** Only alive python from
v587 B14 monitor. Must kill before launching cycle192-194 to avoid
race on cross_run_memory.json and level_bridges.json (B9v2 race-safe
guarantees correctness, but old cycle's writes after launch confuse
analysis).

**O-4 (V-H5 measurement window).** ARC_SMOKE_MAX_ACTIONS=400 per cycle
across 3 cycles = 1200 total turns. If no L+2 in any cycle:
- Confirms B16 chain layer alone insufficient
- Reserve B17 (DreamCoder-style abstraction primitive learning) as
  next intervention.

If L+2 emerges in any cycle, capture trace and inspect M3 thoughts
+ chain_rule emissions for the contributing reasoning step.

## 13. Open questions deferred to future B-builds

- **B17:** chain_rule with cfr≥3 promoted to active_hypotheses card
  schema (Soar-style chunking).
- **B17:** sub-pattern mining (M-CHAIN-5) once chain length grows.
- **B18:** chain shared across Reflexion of different cycles
  (cross-run chain memory).

These are explicitly OUT OF SCOPE for B16. B16 establishes the input
layer and the output schema; subsequent builds compile against them.

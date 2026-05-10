# Plan v601 rev C — 3-Role Architecture + I-Saturation-Bundle (FROZEN)

**Date**: 2026-05-10
**Revision**: rev C (frozen after codex [CONVERGED] + 5 hardening edits)
**Predecessor**: v600 (predicate library + Beta-Bernoulli + LLM-on-stalemate, frozen tag `v600-confirmatory-protocol-d1`)
**Diagnosis driver**: `reports/why_gpt_cant_solve_ft09_2026_05_10.md` (saturation-gate)
**Survey input**: 24 papers 2025Q4-2026 (theoretical contrast §10)
**Codex review-counter**: rev A 2 rounds (architecture converged), rev B 3 self-loop iterations (23 gaps closed), rev C 1 codex final gate ([CONVERGED] + 5 hardening edits applied)

---

## 0. Top-level Intent / Hypothesis / Verification

**Intent (I)**. ft09 has been solved exactly once in 270+ cycles (cycle237, 0.37%). Diagnosis: the L+ trigger for at least one level is gated by an **aggregate completion predicate** ("all 8 compass neighbors of an active marker clicked ≥ 1 time"), and GPT-5.5 does not synthesize aggregate predicates from structured observation fields. v601's intent is to materialize the saturation gate end-to-end — in proposer prompt (Step-0 saturation computation), in posterior arm key (`saturation_status` dimension), and in long-term memory (paired-counterfactual entries on same-coord different-outcome) — atomically.

**Hypothesis (H)**. If the saturation gate is materialized in all three layers (prompt → posterior → memory), then over ≥10 episodes per Phase D1, the agent will produce L+2 in ≥1 episode (Wilson 95% LCL ≥ 0.0026 vs the v600 0.37% baseline) **and** the live trace will show R12-style markers transitioning saturation_status `near_complete → complete` immediately before each L+ event with probability ≥ 0.6. Failure of either prediction falsifies v601. Performance lift target across 30 pooled episodes: ≥5/30 L+2 (matches v600 plan §1.18 D3 stage gate).

**Verification (V)**. Two-tier:
- **Pre-launch (offline, fixture-only)**: 13 v600 fixtures + 5 v601 INT fixtures + 30 new v601 unit fixtures ALL pass at 100% on mocked LLM outputs and recorded traces. Wall-clock < 15 s for full suite. No code path exercised at < 99% (deterministic) or < 80%/75% train/val (LLM components).
- **Post-launch (live, statistical)**: Phase D1 (10 episodes, frame-seed-fixed) → D2 (≥20 episodes, ≥4 seeds, LOO) → D3 (≥30 pooled). Westfall-Young max-T FWER, 5-decoy spectrum, 7-arm RASI ablation (all carried from v600 §1.18-1.34). Additional v601-specific metric: **`P(L+ | saturation_just_became_complete)` ≥ 0.6**.

---

## 1. Self-loop iteration record (rev A → rev B)

Eighteen gaps surfaced in self-review of rev A; this section enumerates each with its resolution. Inclusion serves as durable trace for future reviewers.

| # | Gap (rev A) | Resolution (rev B) |
|---|---|---|
| G1 | INT01 fixture cited "R31 compass 7/8 saturated at T5" but trace at T5 had n=4 markers, not n=2 (R6/R12 only appear post-T6) | INT01 source rebased to **cycle237 T0-T5 4-marker pre-state**; pre-T6 active marker reconstructed as R31 (per chosen_hypothesis_id `P_crop_compass_sweep_R31`) |
| G2 | `region_hint` → `region_id` grounding mechanism unspecified | Spec §3.4: `region_hint` is the literal `region_id` string; if multiple matches in `visible_regions`, fallback to first by index |
| G3 | `saturation_status ∈ {none, near_complete, complete}` thresholds undefined | Spec §3.6: `complete` ≡ all compass.clicks ≥ 1; `near_complete` ≡ exactly one compass slot with clicks = 0; `none` ≡ ≥ 2 unclicked |
| G4 | Paired-cf "different outcome" trigger predicate undefined | Spec §3.7: trigger ≡ `(coord, primary_region_id) seen ≥ 2 times` AND (`max(level_delta) - min(level_delta) ≥ 1` OR `max(dt.count) > 3 × min(dt.count)`) |
| G5 | RASI prior × saturation interaction unspecified | Spec §3.8: RASI primes the base `(predicate, region)` arm; `saturation_status` splits the arm post-prime, sharing the prior's α via dirichlet-style allocation by uniform |
| G6 | Proposer fallback when no marker is `near_complete` | Spec §3.9: emit `predicate_id = P_saturation_progress`, target the marker with `max(saturation_M)`, region_hint = its most-clicked-but-not-yet-clicked compass slot; if all markers have `saturation_M = 0`, fall back to v600 default (highest UCB1 over uniform-prior arms) |
| G7 | `extract_pre_state_diff()` feature whitelist | Spec §3.10: ranked feature list — (1) per-marker compass saturation numerator, (2) per-marker compass last-click direction, (3) recent dominant_transition direction, (4) region click count delta. Diff selects top feature with non-zero variance across the two outcomes |
| G8 | INT03 stalemate trigger interacts with first-turn warm-up | Spec §4.3: warm-up Proposer call counted as turn 0 baseline; subsequent stalemate counter starts at turn 1; INT03 fixture asserts exactly 1 *stalemate-triggered* call across the 12-turn window |
| G9 | M4 conditional Reflector severity formula | Spec §3.11: `severity = max(|outcome_a.dt_count - outcome_b.dt_count| / max(outcome_a.dt_count, outcome_b.dt_count, 1), 0.5 if level_delta differs else 0)`; spawn iff severity > 0.5 AND last_reflector_call_turn < (current - 5) |
| G10 | Proposer LLM timeout / None fallback path | Spec §3.12: on timeout (45 s) or schema-validation failure, log `proposer_failure_reason ∈ {timeout, parse_error, schema_invalid, llm_no_client}`, return None to Policy; Policy uses default arm selection (no proposer hint) |
| G11 | I1 LLM-component "valid" definition | Spec §6: "valid" ≡ output passes JSON schema AND thought cites Step-0 saturation computation (regex `mean\(c\.clicks\s*>=\s*1\s+for\s+c\s+in\s+M\.compass\)` or paraphrase whitelist) |
| G12 | Survey theoretical contrast missing per alternative | New §10 enumerates 8 alternatives with theoretical differentiation against v601 |
| G13 | "Alternatives considered" section absent | Now §10 |
| G14 | Phase D1/D2/D3 protocol carryover from v600 | New §11 mirrors v600 §1.18-1.34 with v601-specific additions |
| G15 | Saturation gate vs cycle237 T6 L+1 trigger (R36 5→4 c1824 — different signature from L+2's 4→8 c564) | Spec §3.13: the gate generalizes — different marker (R31) per level; INT01 tests pre-T6 4-marker state + R31 saturation specifically |
| G16 | Confidence ∈ [0,1] usage in Policy | Spec §3.14: confidence is a tie-breaker only when UCB1 scores are within 0.05 of each other; if `confidence ≥ 0.5`, prefer arm matching `region_hint`; never updates posterior α/β directly (per [24] calibration gating) |
| G17 | Reflector output spec | Spec §3.15: returns `{discriminator_features: [str], reflexion_text: str ≤ 256 chars, suggested_exploration_boost: dict[arm_key → float ∈ [0, 0.3]]}`; boost applied to UCB1 exploration term, not posterior α/β |
| G18 | `tools/extract_v601_fixtures.py` contract | Spec §5.6: CLI `--trace path --turn-a N [--turn-b M] --fixture-type INT01-INT05 --out path`; reads trace.jsonl, computes pre_state_features per §3.10 whitelist, emits JSON matching INT-schema |

**Iteration 2 (additional 4 gaps surfaced + closed)**

| # | Gap (rev B iter-1) | Resolution (rev B iter-2) |
|---|---|---|
| G19' | Arm-key saturation_status convention had wrong default ("min" instead of "max" saturation_numerator) | §3.6 priority list rewritten: target = (proposer hint) > near_complete marker > argmax saturation > n/a sentinel |
| G20 | P_saturation_progress predicate not defined in library | §3.9 adds `_p12_saturation_progress` to STATIC_PREDICATES with explicit signature |
| G21 | RASI uniform allocation (α/3) could go below 1.0 → UCB1 explosion | §3.8 floor at 1.0 added |
| G22 | Proposer LLM client unspecified | §3.12 inherits TRAPI gpt-5.5 client from v600 `llm_extender.py` lines 60-110 |

**Iteration 3 (1 additional gap surfaced + closed)**

| # | Gap (rev B iter-2) | Resolution (rev B iter-3) |
|---|---|---|
| G23 | Paired-cf entry lifecycle for n≥3 occurrences (overwrite vs expand) | §3.7: list-of-outcomes per key, severity = max over all C(n,2) pairs |

Self-loop conclusion: 18 + 4 + 1 = 23/23 closed across three iterations. Iteration 4 self-pass found 0 new gaps. Proceeding to codex final gate.

---

## 2. Architecture (3 roles, codex-converged in rev A)

```
┌─────────────────── Per-turn loop (≤ 60 s wall-clock) ─────────────────┐
│                                                                          │
│   ┌────── Role 1: Proposer (LLM, conditional) ──────┐                   │
│   │  Triggered (any of):                              │                   │
│   │    • turn == 0 (warm-up)                          │                   │
│   │    • stalemate (turns_since_L+ > 8 AND            │                   │
│   │      max_posterior < 0.45)                        │                   │
│   │    • new paired-cf entry not yet consumed         │                   │
│   │  HARD: cannot call submit_action.                 │                   │
│   │  TIMEOUT: 45 s, fallback None.                    │                   │
│   │  OUTPUT (validated JSON):                         │                   │
│   │    candidate_predicate_id: str                    │                   │
│   │    region_hint: str (region_id)                   │                   │
│   │    expected_signature: dict                       │                   │
│   │    required_pre_state:                            │                   │
│   │       marker_id: str                              │                   │
│   │       saturation_threshold: int                   │                   │
│   │       saturation_denominator: int                 │                   │
│   │    confidence: float ∈ [0, 1]                     │                   │
│   └────────────────────────┬─────────────────────────┘                   │
│                            ▼ proposer_output (or None)                   │
│   ┌────── Role 2: Policy (deterministic, always) ─────┐                  │
│   │  arm_key = (predicate_id, region_id,               │                  │
│   │             saturation_status)                     │                  │
│   │  posterior.update(observation)                     │                  │
│   │  scores = {arm: ucb1_score(arm) for arm}           │                  │
│   │  if proposer_output and confidence ≥ 0.5:          │                  │
│   │     boost ties by region_hint match                │                  │
│   │  chosen_arm = argmax(scores)                       │                  │
│   │  chosen_coord = resolver(predicate, region, bbox)  │                  │
│   │  Beta-Bernoulli + 7-arm RASI (carry v600).         │                  │
│   └────────────────────────┬─────────────────────────┘                   │
│                            ▼ submit_action(coord) → observation          │
│   ┌────── Role 3: Memory writer (deterministic, always) ─┐               │
│   │  verdict = compare(expected, observed)                │                │
│   │  posterior.record_outcome(arm_key, verdict)            │                │
│   │  if same_coord_different_outcome detected (G4):        │                │
│   │     diff = extract_pre_state_diff() (G7)               │                │
│   │     paired_cf.append({coord, diff, outcome_a, _b})     │                │
│   │     if severity > 0.5 and last_reflector +5 turns ago: │                │
│   │        Reflector spawn (LLM, ≤ 30 s, returns G17)      │                │
│   └────────────────────────────────────────────────────────┘               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Specifications (gap closures in detail)

### §3.4 region_hint grounding (G2)
`proposer_output.region_hint` is the literal `region_id` string (e.g., `"R16"`). Policy looks it up in `visible_regions[*].region_id`. If absent (region not currently visible), fall through to UCB1 over visible arms ignoring the hint.

### §3.6 saturation_status thresholds (G3)
For each marker M with `is_primary_marker == true`:
```python
clicked = sum(1 for c in M.compass.values() if c.clicks >= 1)
N = len(M.compass)
if clicked == N:
    status = "complete"
elif clicked == N - 1:
    status = "near_complete"
else:
    status = "none"
```
**Arm-key saturation_status convention** (rev B iter-2 G19, corrected G19'): the arm key encodes the saturation_status of the **target marker** selected by this priority:
1. If `proposer_output.required_pre_state.marker_id` is present, that marker is the target.
2. Else, if any visible primary marker has status `near_complete`, target = that marker (if multiple, `argmax(saturation_numerator)`).
3. Else, if any visible primary marker has `saturation_numerator > 0`, target = `argmax(saturation_numerator)` (drives the most-saturated marker to completion).
4. Else (no primary markers visible, or all at zero saturation): arm key uses sentinel `saturation_status="n/a"`, Policy collapses to v600 default UCB1 over uniform prior.

Arm key uses decision-time status (state at the moment of the click). Once chosen, the same target_marker_id is logged with the turn record for post-hoc analysis.

### §3.7 paired-cf trigger predicate (G4)
```python
def trigger(history, new_obs):
    key = (new_obs.coord, new_obs.primary_region_id)
    prior = [h for h in history if (h.coord, h.primary_region_id) == key]
    if not prior:
        return False
    deltas = [h.observation.level_delta for h in prior] + [new_obs.level_delta]
    counts = [h.observation.dt.count if h.observation.dt else 0 for h in prior] + [new_obs.dt.count if new_obs.dt else 0]
    return (max(deltas) - min(deltas) >= 1) or (min(counts) > 0 and max(counts) > 3 * min(counts))
```
**Lifecycle for n≥3 occurrences (rev B iter-3 G23 + rev C C2)**: paired-cf is keyed by `(coord, primary_region_id)` but stores a list of all observed `outcome_features` for that key. On each new occurrence, the list grows (append, no overwrite). The discriminator extractor (§3.10) re-runs over the full list and writes the dominant discriminator (highest variance feature). Reflector spawn uses:
```python
best_pair_severity = max(severity(a, b) for a, b in combinations(outcomes, 2))
top_disc = argmax_feature(variance(feature) for feature in whitelist)
support_count = sum(1 for a, b in combinations(outcomes, 2)
                   if discriminator(a, b)[0] == top_disc)
spawn_reflector = (best_pair_severity > 0.5
                   AND support_count >= 2     # rev C C2: noise-resistance
                   AND last_reflector_turn <= current_turn - 5)
```
For exactly two outcomes (n=2), `support_count = 1` is acceptable (the single pair). For n≥3, requiring `support_count ≥ 2` prevents one noisy outlier pair from spawning Reflector. Cooldown §3.11 still caps at 1 spawn per 5 turns.

### §3.8 RASI × saturation interaction (G5)
v600's 7-arm RASI prior weights `(predicate, region)` arms. v601 adds `saturation_status` dimension with **uniform allocation** of the RASI α across the three sub-arms at the same `(predicate, region)`, with floor:
```python
α_base, β_base = rasi_lookup(predicate, region)  # from v600 cycle237 trace
for status in ["none", "near_complete", "complete"]:
    α_ext[(predicate, region, status)] = max(1.0, α_base / 3.0)
    β_ext[(predicate, region, status)] = max(1.0, β_base / 3.0)
```
**Floor at 1.0 (rev B iter-2 G21)** prevents zero-counts that would inflate UCB1 exploration term to infinity. Slight over-counting acceptable. Sentinel `saturation_status="n/a"` arms (when no primary marker visible) get vanilla `(α_base, β_base)` without splitting.

### §3.9 Proposer fallback when no near_complete marker (G6)
```python
if no marker has saturation_status == "near_complete":
    target_M = argmax_M(saturation_M_numerator)  # closest to completion
    if saturation_M_numerator(target_M) > 0:
        emit P_saturation_progress aimed at unclicked compass slot of target_M
    else:
        emit None  # Policy uses v600 default UCB1 selection
```
**P_saturation_progress predicate (rev B iter-2 G20)**: added to v601's static predicate library as `P12_saturation_progress`:
```python
def _p12_saturation_progress(state, t) -> list[RegionRef]:
    """Returns the unclicked compass slots (RegionRefs) of the target marker."""
    target_marker_id = state.get("target_marker_id")  # injected by Proposer or fallback
    if target_marker_id is None:
        return []
    for m in state.get("marker_neighbor_states", []):
        if m["marker_id"] == target_marker_id:
            return [_to_region_ref(r) for r in state.get("visible_regions", [])
                    if any(c["region_id"] == r.get("id") and c.get("clicks", 0) == 0
                           for c in m["compass"].values())]
    return []
```
Family: `saturation_progress`. Coord_policy: `centroid`. Registered alongside P00-P11 in `predicate_library.py` `STATIC_PREDICATES`.

### §3.10 extract_pre_state_diff feature whitelist (G7)
Ranked features (highest priority first):
1. `marker_X_compass_saturation_numerator` (per visible marker X)
2. `marker_X_compass_recent_click_direction` (last clicked compass dir, or null)
3. `recent_dominant_transition_direction` (e.g., `"9→12"`)
4. `region_X_click_count` (per visible region X)
5. `level_delta_since_last_paired_cf`

Diff selects the top feature where `outcome_a[feature] != outcome_b[feature]` AND has non-trivial magnitude (numeric: |diff| ≥ 1; categorical: any change). If all features tie, store all as `discriminating_features`.

### §3.11 severity formula (G9)
```python
def severity(outcome_a, outcome_b):
    s_count = abs(outcome_a.dt.count - outcome_b.dt.count) / max(outcome_a.dt.count, outcome_b.dt.count, 1)
    s_level = 0.5 if outcome_a.level_delta != outcome_b.level_delta else 0.0
    return max(s_count, s_level)
```
For T8 (c=36) vs T27 (c=564, ldelta+1): s_count = (564-36)/564 ≈ 0.94; s_level = 0.5; severity = 0.94 → spawn Reflector.

### §3.12 Proposer failure fallback (G10)
```python
proposer_output, failure_reason = try_proposer(timeout=45)
if failure_reason in {"timeout", "parse_error", "schema_invalid", "llm_no_client"}:
    log_event("proposer_failure", reason=failure_reason)
    proposer_output = None
# Policy proceeds with proposer_output=None → v600-style UCB1 default
```
**LLM client (rev B iter-2 G22)**: Proposer and Reflector use TRAPI gpt-5.5 via the same client pattern as v600's `llm_extender.py` (lines 60-110): `ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())` → `AsyncAzureOpenAI(azure_endpoint=_TRAPI_ENDPOINT, ...)`. Model preferences `["gpt-5.5_2025-08-07", "gpt-5.5", "gpt-5.4_2026-03-05"]`. `llm_no_client` failure when azure deps missing.

### §3.13 Saturation gate generalization across levels (G15)
Empirical pattern from cycle237:
- L+1 (T6): chosen_hyp `P_crop_compass_sweep_R31` → R36 5→4 c1824. R31 was the active marker (one of 4 pre-T6); saturation gate likely on R31's compass.
- L+2 (T27): chosen_hyp `P_R12_crop_sector_alignment` → R16 4→8 c564. R12 was the active marker (one of 2 post-T6).

Plan v601 treats saturation as a per-level reset feature: at each L+ event, marker_neighbor_states refresh (4→2 markers), click counters reset, saturation_status of new arms restarts at `none`. Phase D2 LOO will validate this generalizes.

### §3.14 Confidence usage (G16) + per-episode override cap (rev C C3)
```python
def select_arm(scores, proposer_output, episode_state):
    sorted_arms = sorted(scores.items(), key=lambda x: -x[1])
    top, second = sorted_arms[0], sorted_arms[1]
    if (top[1] - second[1] < 0.05 and proposer_output is not None
            and proposer_output.confidence >= 0.5
            and episode_state.confidence_override_count <
                min(3, math.ceil(0.15 * episode_state.policy_decisions))):
        # tie-break to region_hint match
        for arm, score in sorted_arms:
            if arm.region_id == proposer_output.region_hint:
                episode_state.confidence_override_count += 1
                return arm
    return top[0]
```
Confidence does NOT affect posterior α/β (per [24] calibration gating). Per-episode override cap = `min(3, ceil(0.15 × policy_decisions))` prevents the proposer's confidence from becoming a soft policy override channel. Each override is logged with `confidence_override_count` for post-hoc analysis. Once cap exceeded, fall back to pure UCB1.

**Marker hint validation (rev C C1)**: before accepting `proposer_output.required_pre_state.marker_id` as the target, verify `marker_id` exists in current `marker_neighbor_states` AND its observed `len(compass) == saturation_denominator`. If validation fails, demote target_marker to `near_complete` priority (G19' rule 2). Prevents stale or hallucinated marker hints from silently overriding grounded state.

### §3.15 Reflector output spec (G17)
```python
@dataclass
class ReflectorOutput:
    discriminator_features: list[str]
    reflexion_text: str  # ≤ 256 chars
    suggested_exploration_boost: dict[ArmKey, float]  # values ∈ [0, 0.3]
```
Boost is added to the UCB1 *exploration* term for the named arms during the next 5 turns:
```python
ucb1_score = exploit + sqrt(2*log(N)/n) + boost.get(arm, 0)
```

---

## 4. Five Integration Fixtures (deterministic, 100% pass required)

### INT01 — One-Turn Policy Pipeline (mocked Proposer, pre-T6 R31 state)
**Bottleneck pattern**: pre-L+1 R31 compass should be reaching `near_complete` before T6 trigger.
**Source**: `simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl` turn 5 (4-marker pre-state).
**Mock proposer output**: `{candidate_predicate_id: "P_saturation_progress", region_hint: "R36", required_pre_state: {marker_id: "R31", saturation_threshold: 7, saturation_denominator: 8}, confidence: 0.7}`
**Expected**: Policy chooses arm `(P_saturation_progress, R36, near_complete)`, coord within R36 bbox `[36,52,41,57]`, no Memory writer paired-cf trigger (no prior R36 click).
**LOC**: ≤ 60 pytest.

### INT02 — Paired-CF Discriminator (T8 vs T27 injection)
**Bottleneck pattern**: same coord [38,48]→R16, T8 `9→12 c36` vs T27 `4→8 c564`.
**Source**: same trace, turns 8 + 27.
**Injection**: feed Memory writer two turn records sequentially with computed `pre_state_features` per §3.10.
**Expected**: after second injection, paired_cf has 1 entry with `discriminating_features = ["marker_R12_compass_saturation_numerator"]`, severity = 0.94 → Reflector spawned.
**LOC**: ≤ 80 pytest.

### INT03 — Stalemate Cadence + Warm-up (G8)
**Bottleneck pattern**: latest cycle T32-T67 36-turn stall (no L+).
**Source**: `simple_logs/ft09-9ab2447a/v57_1778305600_1522780/trace.jsonl` turns 32-67 stalemate slice.
**Synthetic variants**: 3 variants (uniform stall / declining max_posterior / oscillating). Each is 12 turns.
**Expected**: warm-up Proposer at turn 0 (counted separately), then **exactly 1** stalemate-triggered Proposer call across the 12-turn window, then Reflector spawned exactly when paired-cf triggers (≥ 1 per variant).
**LOC**: ≤ 100 pytest.

### INT04 — Schema Violation Rejection
**Bottleneck pattern**: malformed Proposer output causes downstream crash.
**Source**: 5 synthetic malformed JSONs (missing field, illegal predicate, embedded `submit_action`, confidence > 1, region_hint not in visible_regions).
**Expected**: each is rejected with specific error code (`schema_missing_field`, `predicate_blacklisted`, `tool_call_blocked`, `confidence_out_of_range`, `region_unknown`); Policy proceeds with `proposer_output=None`; episode does not crash.
**LOC**: ≤ 100 pytest.

### INT05 — Game-Agnostic Prompt Lint (G13 scope + rev C C2 rendered-prompt variant)
**Bottleneck pattern**: prompts contain ft09-specific identifiers, breaking transferability.
**Scope (split into 2 sub-checks)**:
- **INT05a — Template lint**: `agents/templates/agentica_lite/proposer_prompt.py` + the prompt-construction sections of `agents/templates/agentica_lite/agent.py` (functions returning string templates).
- **INT05b — Rendered-prompt lint** (rev C C2): render the proposer prompt with synthetic state containing forbidden-looking IDs (e.g., `region_id: "R12"`); assert the FINAL rendered prompt does NOT include raw game labels at the prompt-construction layer. Generated names should be neutralized via `marker_0`, `region_primary`, `slot_N` aliases when emitted into prompt text — runtime `region_id` strings remain as data fields but do not appear in prompt prose.

**Forbidden tokens (both INT05a and INT05b output)**: `R6, R12, R16, R31, ft09, XOR, parity, saturation_R*`, predicate names like `marker_R12_compass`.
**Allowed tokens**: schema-level field names (`is_primary_marker`, `compass`, `clicks`, `region_id`) — these are field NAMES, not VALUES.
**Expected**: 0 forbidden matches in template source (INT05a) AND 0 forbidden matches in rendered prompt body (INT05b).
**LOC**: ≤ 80 pytest.

### INT06 — Severity + Cooldown + n≥3 Lifecycle (rev C C1 NEW)
**Bottleneck pattern**: noisy paired-cf evidence triggering Reflector spam, or repeated severity-high contrasts inside cooldown.
**Source**: 4 synthetic n=3 paired-cf scenarios.
**Sub-cases**:
1. n=3 outcomes, all 3 pairwise severities >0.5, all 3 select same discriminator → `support_count=3 ≥ 2` → Reflector spawned
2. n=3 outcomes, 1 pairwise severity >0.5, others <0.5 (one noisy outlier) → `best_pair_severity = high BUT support_count = 1` → Reflector NOT spawned
3. Two consecutive severity-high contrasts within 3 turns of each other → first spawns Reflector, second is suppressed by cooldown
4. n=4 outcomes, list-of-outcomes accumulates without overwriting (assert `len(outcomes) == 4`)

**Expected**: each sub-case asserts the precise spawn / no-spawn / accumulation behavior.
**LOC**: ≤ 100 pytest.

---

## 5. Implementation Sequence + tools

### §5.1 Day-by-day schedule
| Day | Deliverable | Verification |
|---|---|---|
| 1 | `tools/extract_v601_fixtures.py` (G18 spec) + INT01-INT05 fixture JSONs | All 5 fixtures load via `load_fixtures()`; pytest collects them |
| 2 | Memory writer + paired-cf entry + arm-key extension; pass INT02 | INT02 green; 13 v600 fixtures still green |
| 3 | Proposer prompt with §3 saturation step + JSON schema validator; pass INT04 | INT04 green; LLM mock returns valid JSON in fixtures |
| 4 | Stalemate trigger logic + Reflector cadence + first-turn warm-up; pass INT01, INT03 | INT01, INT03 green |
| 5 | Game-agnostic lint; iterative-code-review smoke-critic-fix | INT05 green; all 5 INT + 13 v600 fixtures green |
| 6 | Codex final gate on **CODE** (not plan). If pass → git tag `v601-3role-saturation-d1` | Tag pushed |
| 7 | Phase D1 launch (10 episodes) | n/a, autoresearch starts |

### §5.6 `tools/extract_v601_fixtures.py` contract (G18)
```
Usage: extract_v601_fixtures.py --trace TRACE.jsonl --fixture-type {INT01,INT02,INT03,INT04,INT05}
                                --turn-a N [--turn-b M] --out OUT.json [--mock-proposer J]
Behavior:
  - Reads trace.jsonl, optionally reads turn N+M for paired contrast
  - Computes pre_state_features per §3.10 whitelist
  - Emits JSON matching INT-schema in tests/v601/fixtures/{type}/
  - For INT04, generates from synthetic templates (no trace input)
  - For INT05, lints prompt files (no trace input)
LOC: ≤ 200
```

---

## 6. Acceptance Tiers (codex-converged, rev A)

| Tier | Components | Pass bar |
|---|---|---|
| Deterministic unit/property | resolver, verdict, arm-key encode, paired-cf diff, severity calculator, JSON schema validator | **≥99%** |
| LLM component fixtures | Proposer prompt sanity (output schema valid AND saturation Step-0 cited per §G11) | **rev C C3 tightened**: `valid_schema_rate ≥ 98%`, `grounded_step0_reference_rate ≥ 95%`, `overall_usable_proposal_rate ≥ 95% train / ≥ 90% val`, `train-val gap ≤ 8pt`. Old ≥80%/≥75% threshold retained only as a relaxed observability metric, not a gate. Justification: Proposer is constrained JSON production with grounded citation, not open-ended reasoning. |
| Hybrid integration (INT01-INT05) | mocked or frozen LLM responses | **100%** |

---

## 7. M4 Removal Confirmation

`agentica_simple/m4_prompt.py` deleted in v600 cleanup. v601 does NOT restore standalone M4. Useful M4 remnant (high-quality contrastive reflection) lives inside Memory writer as `Reflector` sub-call gated by §3.11 severity > 0.5 AND § 3.11 cooldown ≥ 5 turns. Expected ≤ 1 Reflector call per episode.

---

## 8. Code Layout

```
agents/templates/agentica_lite/
  agent.py              # Orchestrator: glue Proposer→Policy→Memory writer
  proposer.py (NEW)     # Role 1: LLM call, schema validator, fallback
  proposer_prompt.py (NEW)  # Step-0 saturation prompt, game-agnostic
  policy.py (NEW)       # Role 2: arm-key + posterior + UCB1 + resolver + verdict
  memory_writer.py (NEW)  # Role 3: paired-cf detector + diff extractor + Reflector
  reflector.py (NEW)    # conditional sub-LLM call
  predicate_library.py  # carry from v600
  predicate_posterior.py  # extended ArmKey from §3.6
  stalemate_trigger.py  # carry from v600 + first-turn warm-up flag
  memory_journal.py     # carry from v600
  fixtures.py           # extended fixture loader for v601
tests/v601/
  fixtures/
    INT01_one_turn_R31_pre_T6.json
    INT02_paired_cf_R16_T8_T27.json
    INT03_stalemate_*.json (3 variants)
    INT04_schema_violation_*.json (5 variants)
    INT05_prompt_lint.json (1 lint case)
  test_v601_fixtures.py
  test_v601_integration.py
tools/
  extract_v601_fixtures.py (NEW, §5.6 contract)
```

LOC budget: ≤ 1,400 (vs v600's 600). Increase justified by 3 explicit roles + paired-cf machinery.

---

## 9. Phase D Live Protocol (carry from v600 + v601 additions)

§ 9.1 Stage gates carried verbatim from v600 plan §1.18-1.34:
- D1: 10 episodes, target ≥ 8/10 L+1 (Wilson LCL ≥ 0.49)
- D2: ≥ 20 episodes across ≥ 4 frame seeds, target ≥ 1/20 L+2 + LOO robustness
- D3: ≥ 30 pooled, target ≥ 5/30 L+2 (Wilson LCL ≥ 0.072)
- 5-decoy spectrum (random-region, label-shuffled, impossible, cross-region, cross-episode), Bonferroni p<0.002
- 7-arm RASI ablation (null/targeted/dist-shuffle/phase-shift/variance-match/sign-invert/uniform-cal)
- Westfall-Young max-T or Holm step-down FWER

§ 9.2 v601 adds (rev C C4 Bayesian gate replaces raw-threshold gate):
- **Saturation invariant report (Bayesian gate, rev C C4)**: every L+ event logs the saturation_status of all visible markers immediately before AND after the trigger click. Compute posterior `Pr(p_v601 > p_v600 + δ | data)` with δ = 0.05 (operational) or 0.10 (conservative) using Beta-Binomial conjugate update. **Pass gate**: posterior mean ≥ 0.5 AND `Pr(p_v601 > p_v600) ≥ 0.8`. Also report 80% credible interval. The previous raw `P(L+ | sat just became complete) ≥ 0.6` is reported as an effect-size metric, not a gate (D2 only has 5-8 expected L+ events; raw threshold is statistically unstable).
- **Paired-cf log**: every paired-cf entry written during D1-D3 logged with discriminator_features. Across D3, expect ≥ 30% of entries to cite `marker_*_compass_saturation_numerator` as the discriminator (validates §3.10 whitelist).
- **Reflector frequency**: ≤ 2 Reflector spawns per episode (per §3.11 cooldown ≥ 5 turns).

---

## 10. Alternatives Considered + Rejected (theoretical differentiation)

User instruction: "이론에 기반해서 기존 방법들과 명확하게 구분되는 방법들을 여러개 고민해보고 논의해서 결정". Eight architectures evaluated. Each rejected for a specific reason; v601 = composition of the strongest parts.

### A1. Pure MCTS over (predicate, region, saturation_status) tree
*Cite*: AlphaZero family. *Theoretical fit*: tree search over 12 predicates × 50 regions × 3 saturation = 1,800 leaves is tractable in principle. *Rejection*: ft09 has a real environment but is action-budget-limited (~800 total across all levels per `agent.py:46 MAX_ACTIONS`). MCTS rollouts would deplete the budget on simulated branches without delivering learning signal beyond what the actual play produces. UCT-without-simulator collapses to UCB1 (which we already have). Adding tree depth without rollouts is empty machinery (per [16] VisualPredicator's finding that LLM perceptual queries fail OOD when used as rollout substitutes). **Verdict**: insufficient gain over UCB1 under the action budget; rejected.

### A2. Transformer-based world model + planning
*Cite*: PlaNet, Dreamer-V3 lineage. *Theoretical fit*: learn `p(s_{t+1} | s_t, a)` and plan in latent space. *Rejection*: requires per-task training data; ARC-AGI-3 is one-shot per puzzle (no replay buffer accumulates fast enough); world model would be undertrained at 270 episodes. ExoPredicator [17] partially addresses with Bayesian model selection but still requires environment-specific runs. **Verdict**: data-starved; rejected.

### A3. Classical STRIPS planner + auto-induced predicates
*Cite*: ExoPredicator [17] (closest), VisualPredicator [16]. *Theoretical fit*: induce `(precondition, action, postcondition)` triples; planner solves at symbolic level. *Rejection*: STRIPS requires accurate effect models; ft09's effect is the unknown (we are trying to LEARN it). ExoPredicator's "predicates depend on exogenous state" is closer but still expects environment-controllable observations, which ft09's stochastic-looking dt does not provide. **Verdict**: presupposes the model we are trying to learn; rejected.

### A4. Pure Voyager-style skill library (no posterior)
*Cite*: [1] Voyager, [2] AutoSkill, [4] SAGE. *Theoretical fit*: accumulate executable skills; LLM picks among them. *Rejection*: no Bayesian posterior = no principled exploration of when a skill works vs fails. Voyager succeeded in Minecraft because effects are largely deterministic; ft09's state-dependent triggers (saturation gate) defeat skill-library-only approaches. **Verdict**: missing posterior over skill applicability; rejected.

### A5. Reflexion-only loop (no formal predicate library)
*Cite*: Reflexion 2023, [12] MAR 2025, [13] CGI 2025. *Theoretical fit*: verbal reflection updates LLM beliefs across turns. *Rejection*: empirically tested in our v52 era — `active_hypotheses_after = []` for every turn, M4 reflexion did not actually feed back useful state. Verbal-only memory cannot represent aggregate predicates without a structured store. **Verdict**: empirically falsified in our codebase; rejected.

### A6. PPO/RL fine-tuning on the LLM
*Cite*: [4] SAGE (Skill-Augmented GRPO). *Theoretical fit*: optimize policy directly. *Rejection*: requires gradient access to gpt-5.5 (we don't have); requires labeled reward data (we have only sparse L+ signal). Cost-prohibitive for a 270-episode debugging cycle. **Verdict**: infrastructure-blocked; rejected.

### A7. Symbolica's full Orchestrator + Theorist/Tester/Solver dynamic dispatch
*Cite*: `/home/v-seungplee/ARC-AGI-3-Agents/agents/templates/agentica/`. *Theoretical fit*: capability-gated subagents reduce wasted actions. *Rejection*: their `Memories` is plain text — cannot represent paired counterfactuals or arm-key conditioning. Symbolica solves the *action-discipline* problem (no LLM calls submit_action prematurely) but does not solve the *aggregate-predicate-synthesis* problem. **Verdict**: useful pattern (we adopt the capability-gating subset); incomplete on the saturation gate; rejected as pure replacement.

### A8. v600 baseline (Beta-Bernoulli, no saturation)
*Cite*: our own v600 plan. *Theoretical fit*: 460 LOC minimal-fast. *Rejection*: arm key `(predicate, region)` cannot encode saturation_status; LLM-on-stalemate has no Step-0 saturation prompt. Per the diagnosed bottleneck (`why_gpt_cant_solve_ft09`), v600 will not progress past saturation-gated puzzles. **Verdict**: incomplete; rejected as final architecture.

### A9. v601 = composition of best parts
- Capability gating from Symbolica A7 → Proposer cannot submit_action
- Bayesian posterior from A4 → Beta-Bernoulli + UCB1
- Aggregate predicate from A3/A5 lessons → mandatory Step-0 saturation in prompt
- Paired-counterfactual memory from [14] CLEAR + [15] AutoGuide → Memory writer
- Calibration gating from [24] → confidence is tie-breaker, not α/β update
- 3-role decomposition from [21] Posterior Sampling LLM Exploration

**Theoretical contribution (rev C C5 — downgraded from "first" to system contribution)**: v601 contributes an **explicit saturation-gated predicate interface** that forces aggregation across structured observation sub-fields before policy selection, combining (a) LLM proposal of structured pre-state requirements, (b) deterministic posterior control with state-conditioned arm keys, and (c) memory-triggered counterfactual discrimination. Closest prior art: [8] MACLA (Bayesian utility over procedures, no field aggregation), [17] ExoPredicator (temporal predicates, not field-aggregation), [14] CLEAR / [15] AutoGuide (paired-trajectory contrast, no posterior arm-key materialization). v601 composes these primitives into a single closed loop targeting aggregate completion gates. Publishing target: NeurIPS 2026 / ICLR 2027 system track. Global firstness claim NOT made — broader survey would be needed.

---

## 11. Phase D protocol (full carry-over from v600 + v601 additions per §9)

(Detailed in §9, condensed here for top-level reference.)

D1 → D2 → D3 stage gates with Wilson LCL, 7-arm RASI, 5-decoy, FWER, LOO robustness — unchanged from v600 plan §1.18-1.34. v601 adds saturation-status invariant + paired-cf coverage report.

---

## 12. Convergence record

- **rev A (2026-05-10 morning)**: 2 codex rounds, converged on 3-role + I-saturation-bundle.
- **rev B iter-1 (2026-05-10 afternoon)**: closed 18 gaps (G1-G18).
- **rev B iter-2**: closed 4 gaps (G19'-G22).
- **rev B iter-3**: closed 1 gap (G23).
- **rev C (FROZEN, this doc)**: codex final gate returned `[CONVERGED]` with no killer concerns. 5 hardening edits applied:
  - C1: marker_id hint validation (§3.14) + INT06 fixture
  - C2: support_count requirement for n≥3 paired-cf (§3.7) + rendered-prompt lint INT05b
  - C3: per-episode confidence override cap (§3.14) + tightened LLM acceptance (§6: ≥98%/≥95% schema/grounding instead of ≥80%/≥75%)
  - C4: Bayesian gate replacing raw P≥0.6 threshold (§9.2)
  - C5: novelty claim downgraded from "first" to "system contribution" (§10 A9)

**Plan FROZEN at rev C 2026-05-10**. Proceeding to Day-1 implementation: `tools/extract_v601_fixtures.py` + INT01-INT06 fixture extraction via iterative-code-review skill smoke-critic-fix loop. Code freeze gate: codex re-review on implementation BEFORE autoresearch launch.

---

## Appendix A — Bottleneck-trace fixture extraction recipe (§5.6 + §G18)

Per user instruction "병목인 trace 일부를 추출해서 어떻게 확인해야 verifiable하고 reliable하게 전체 trace를 안 테스트 해보고도 확인할 수 있을지":

For each improvement Iₖ, the verification recipe is:
1. **Identify the failure trace span** that triggered the bottleneck (e.g., cycle237 T8/T27 paired contrast, latest cycle T32-T67 stall).
2. **Extract per-turn `pre_state_features`** from the trace using §3.10 whitelist via `tools/extract_v601_fixtures.py`.
3. **Compose a fixture JSON** matching INT01-INT05 schema; mock LLM outputs are deterministic strings frozen at fixture-author time.
4. **Write the assertion** in pytest as a single-function test that calls the v601 module under test with the fixture's input and compares the output to the fixture's expected.
5. **Run pytest in < 1 second per fixture, < 15 s for full suite**. No live game, no live LLM, no agent loop.

This satisfies "전체 trace 안 테스트해도 verifiable" because each fixture is a pure function `f(input_fixture) → output` that the v601 implementation must satisfy; trace replay is encapsulated as input data, not as runtime side-effect.

Reliability comes from:
- Source traces are git-tracked → fixtures are deterministic.
- Mocked LLM outputs are checked-in JSON → no API flakiness.
- Pytest parametrize + load_fixtures pattern reuses the v600 infrastructure (already passing 16/16 in 0.22s).
- Acceptance tiers (§6) prevent flaky fixtures from entering the suite.

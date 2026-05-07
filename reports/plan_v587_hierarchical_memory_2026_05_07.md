# Plan v587 — Hierarchical Memory Management for v57 Agent (B14)

**Status:** DRAFT round 1
**Date:** 2026-05-07
**Owner:** autoresearch
**Scope:** Memory information *management* only — module structure (M1 ACTION /
M3 HYPOTHESIZE / M4 REFLEXION) unchanged.
**Rejected if:** any leakage introduced; any module added; any backward-incompat
break to cross_run_memory.json schema.

---

## 0. Diagnosis (why this plan exists)

The v587 cycle186/187/188 traces show a ceiling phenomenon distinct from the
prior "L+1 unreachable" failure mode:

- cycle186: 69 turns, **L+1 events = 22**, L+2 events = **0**
- cycle187: 56 turns, L+1 = 17, L+2 = 0
- cycle188: 82 turns, L+1 = 26, L+2 = 0

Twenty-two L+1 jumps from one cycle means the agent reaches level 1, then drops
back, then re-reaches it — twenty-two times — without ever holding it long
enough to push to L+2. The reach-the-state mechanism works; the **stay-and-build
mechanism does not**.

Three structural deficits in the current memory layer make this near-inevitable:

1. **All `confirmed_runs == 1`.** cross_run_memory.json holds 111 mechanisms,
   every single one with confirmed_runs=1. The 26 L+1 events of cycle188 each
   produce a slightly differently-worded text observation, so dedup-by-text
   never fires. The agent never sees "this same thing happened 26 times" — only
   "here are 26 unrelated observations." Aggregation is broken.

2. **Fixed-size context regardless of progress.** `recent_turn_diffs` is
   `maxlen=6`, `recent_verbose` is `_VERBOSE_WINDOW=3`. After 22 stuck cycles of
   L+0 ↔ L+1 oscillation, the prompt looks identical in size and shape to the
   first 6 turns. There is no "I've been stuck — let me look back further" mode.

3. **Past-success priors carry no failure marker.** `LEVEL_BRIDGE_PRIORS`
   (B13) shows successful trajectories only. There is no representation of
   "this kind of trajectory looked like the past success but did NOT
   recover." The agent cannot reason counterfactually because no
   counterfactual is presented to it.

The current memory layer is, in cognitive-science terms, *purely additive* and
*purely descriptive* — it accumulates and labels, but it does not *summarise*,
*aggregate*, or *contrast*. The fix has to introduce all three operations
without adding new modules.

---

## 1. Survey of memory architectures relevant to this problem

### 1.1 Episodic / Semantic / Procedural tripartite (Tulving 1972)

Long-term memory in cognitive science is split into:
- **Episodic** — specific events with spatiotemporal context;
- **Semantic** — generalised concepts decoupled from origin;
- **Procedural** — skill-like state-action mappings.

Our current system has only crude episodic (`cross_run_memory` text entries)
and semi-procedural (`active_hypotheses` cards). It has *no semantic
consolidation* — the act of taking N episodic observations of the same kind
and producing one semantic prototype.

### 1.2 Memory in LLM agent literature

- **Reflexion (Shinn et al 2023, NeurIPS)** — verbal RL with a reflection
  buffer. The reflection buffer is *flat* and *append-only*; it has no
  aggregation. Our M4 already does Reflexion-style summarisation but every 5
  turns into a single 500-char string.
- **Generative Agents (Park et al 2023, UIST)** — memory stream + retrieval +
  reflection + planning. Importance scoring, recency, salience drive what gets
  surfaced. Their "reflection" produces tree-of-thought abstractions.
- **MemGPT (Packer et al 2023)** — virtual context with paged memory. Old
  context is paged out to disk; on-demand retrieval back. *Only* adapts to
  context limits, not to task-progress signal.
- **Voyager (Wang et al 2023)** — skill library with iterative refinement
  via self-verification. Skills are aggregated procedurally but episodes
  are not.
- **DreamCoder (Ellis et al 2021, PLDI)** — Wake/Sleep/Abstraction loop.
  *Wake* generates programs; *Sleep* compresses observations; *Abstraction*
  identifies reusable subroutines. The Wake/Sleep alternation is the
  prototype for "actor + consolidator."
- **Hermes-style (Nous Research, 2024)** — episodic buffer with structured
  triples (obs, act, result) and salience-based recall. Closer to our current
  Layer 1 + Layer 2 (B12+B13) but Hermes lacks the *hierarchical expansion
  on stuck* mechanism.
- **Larimar (Das et al 2024, ICML)** — episodic memory module bolted onto an
  LLM with Bayesian update of memory entries. Provides the formal model of
  "confirm vs refute" updates we already implement (`confirmed_runs` /
  `refuted_runs`) but limits aggregation to text-level.

### 1.3 Cognitive architectures (Soar, ACT-R)

- **ACT-R (Anderson & Lebiere 1998)** — *base-level activation* of a memory
  chunk decays over time but increases with each retrieval; chunks are also
  spread-activated by context. The chunk activation model is the formal source
  of `confirmed_runs` as a salience proxy.
- **Soar (Newell 1990)** — chunking compiles successful problem-solving
  sequences into single rules. Our `cross_run_memory` lacks any compile
  step — it is pure deposition.

### 1.4 Counterfactual reasoning

- **Roese (1997, Counterfactual Thinking)** — humans learn faster from
  paired (success, failure) cases than from either alone. The *contrast* is
  the lesson, not the cases.
- **Schank's Case-Based Reasoning (1982)** — index past cases by structural
  features, retrieve by similarity, adapt rather than copy. The closest
  cognitive analog of what we want for stuck-mode retrieval.

### 1.5 Adaptive context allocation

- **Sweller's Cognitive Load Theory (1988)** — when intrinsic load is high,
  external scaffolds must expand; when load is low, they should retract to
  avoid distraction.
- **Miller's chunking (1956, "magic number 7±2")** — short-term memory
  capacity is bounded but chunkable. The fact that we currently *show* 6
  turn-diffs and 3 verbose entries reflects this bound implicitly — it is
  not a bug.

### 1.6 What none of the above does

None of the surveyed systems combines all three of:
- **Structural-signature aggregation** (semantic compaction of episodes)
- **Adaptive context expansion on task-progress signal** (cognitive-load
  driven)
- **Counterfactual past-segment retrieval** (case-based + counterfactual
  cognition)

Within the same memory layer, with the same module set. This plan introduces
that combination.

---

## 2. Methods considered (theory-grounded, multiple candidates)

We list five candidate methods, theory-tag each, then justify the down-select
to three.

### 2.1 Method M-MEM-1 — Episodic Signature Clustering (ESC)

**Theory:** Prototype theory (Rosch 1973), ACT-R chunk activation.

**Mechanism:** Replace text-level dedup in `cross_run_memory.add_to_1A` with
structural-signature dedup. Each L+ event is compressed to a 4-tuple
signature `(kind_dist_hash, compass_change_traj_pattern, repeat_clicks_bucket,
click_count_bucket)`. Two events with the same signature collapse into the
same cluster, with `confirmed_runs += 1` and the textual observations
appended to a `cluster_observations` list (capped at 5).

**Effect:** A run that produces 22 L+1 events of the same shape produces *one*
cluster with confirmed_runs=22 instead of 22 separate entries. The agent
sees salience.

**Distinction from existing:**
- Current B9 dedup is `text.lower().strip()` exact match → never fires.
- Generative Agents importance scoring is *per-event* — does not aggregate.
- Larimar Bayesian update is per-text-key, not per-structural-key.

### 2.2 Method M-MEM-2 — Adaptive Stuck-Mode Window (ASMW)

**Theory:** Sweller cognitive-load theory (expand scaffold under load).

**Mechanism:** Detect stuck via `(turn_index - last_lp_event_turn_index) >
K_STUCK` (K_STUCK=10). On stuck, dynamically expand the
`serialised_recent_turn_diffs` payload from maxlen 6 to
`min(6 + 2*stuck_severity, 24)`. Set `stuck_mode=true` flag in M1 + M3 +
M4 input. Reflexion (M4) gains a forced "stuck-comparison" section when the
flag is set.

**Effect:** When progress halts, the agent sees a longer history and is
explicitly told to look at it for repetition patterns. When progress is
healthy, context stays small.

**Distinction from existing:**
- Current `recent_turn_diffs` is fixed maxlen=6, no stuck signal.
- MemGPT pages on context-limit, not on task-progress.
- Reflexion fires every 5 turns regardless of stuck — no special mode.

### 2.3 Method M-MEM-3 — Counterfactual Past-Segment Retrieval (CPSR)

**Theory:** Schank case-based reasoning + Roese counterfactual cognition.

**Mechanism:** A new offline tool `tools/index_traces.py` post-processes
existing `simple_logs/<game>/cycle*/trace.jsonl` files into a
`segment_index.json` with entries:
```
{
  segment_id, run_namespace, turn_range,
  kind: "pre_L+_5turns" | "post_L+_recovery" | "post_L+_stuck" | "oscillation",
  abstract_signature: <Symbolica-computed>,
  did_progress: bool,
  what_changed_in_next_5turns: <Symbolica abstraction>
}
```
At runtime, in **stuck-mode**, the agent retrieves the top-2 segments closest
to its current trajectory signature: one with `did_progress=true` and one
with `did_progress=false`. Both are injected as `ANALOGOUS_PAST_SEGMENTS`
into M1 input and into Reflexion (M4) input.

**Effect:** The agent reasons "what did the recovering past segment do that
the stuck past segment did not?" — counterfactual contrast surfaces.

**Distinction from existing:**
- LEVEL_BRIDGE_PRIORS (B13) is success-only, no failure marker.
- Generic RAG retrieval ignores Symbolica-computed structural signatures.
- Voyager retrieves skills, not segments.

### 2.4 Method M-MEM-4 (REJECTED) — Vector-store full-trace retrieval

**Theory:** Standard RAG (Lewis et al 2020).

**Why rejected:** Requires sentence-bert embedding, FAISS index, ~5MB extra
disk per game, and either an LLM summarisation pass or raw-text retrieval that
risks leaking R-ids and coords directly. Anti-leak audit is hard; the
Symbolica-signature route in M-MEM-3 is leak-safe by construction.

### 2.5 Method M-MEM-5 (REJECTED) — Constructive memory editing

**Theory:** Schacter constructive memory paradigm — the agent edits its own
past entries to refine.

**Why rejected:** Lossy under race conditions (two parallel cycles editing
same entry); user feedback memory `feedback_no_codex_review.md` style says
synthesis should stay in main agent, not delegated to model self-edit; the
audit trail is hard to reconstruct.

### 2.6 Down-select rationale

The three retained methods (ESC, ASMW, CPSR) each address a *distinct*
deficit identified in §0:

| Deficit | Method |
|---|---|
| All confirmed_runs=1 (no aggregation) | M-MEM-1 (ESC) |
| Fixed context regardless of progress | M-MEM-2 (ASMW) |
| Success-only priors (no contrast) | M-MEM-3 (CPSR) |

They compose orthogonally: ESC changes how memory is *stored*, ASMW changes
how memory is *exposed*, CPSR changes what memory is *retrieved*.

---

## 3. Intent / Hypothesis / Verification (I/H/V)

### Intent
Convert the memory layer from "additive flat lists" to "hierarchical
summarisable structure" so the agent can:
- See aggregated salience (this happened N times, not N unrelated events),
- Look further back when stuck without bloating prompts when not stuck,
- Reason counterfactually by seeing both what worked and what did not in
  structurally similar past trajectories.

The module structure (M1 / M3 / M4) is preserved exactly; only memory
storage, exposure, and retrieval behaviour change.

### Hypothesis (each falsifiable)

**H1 (ESC produces aggregation).** With structural-signature clustering,
after 3 cycles run on ft09, the cross_run_memory will contain at least one
cluster with `confirmed_runs >= 5`. Falsifiable: if all clusters end with
`confirmed_runs <= 2`, the signature is too fine-grained and ESC failed.

**H2 (ASMW produces longer reasoning under stuck).** When `stuck_mode=true`
fires (i.e. ≥10 turns since last L+ event), M1 thoughts will reference at
least 2 turn-diff entries from beyond the original 6-turn window. Falsifiable
via grep on M1 thought text for turn indices.

**H3 (CPSR produces counterfactual reasoning).** With ANALOGOUS_PAST_SEGMENTS
injected, Reflexion summaries will contain explicit comparison phrasing
("similar past segment did X, current segment did Y") in ≥50% of stuck-mode
turns. Falsifiable via grep on Reflexion output.

**H4 (Combined effect on L+2).** The combined memory hierarchy enables at
least 1 of 3 cycles in a fresh launch (cycle189/190/191) to reach an L+2
event within 200 turns. Falsifiable: zero L+2 events across all three cycles
within 200 turns each.

### Verification

| Gate | Method | Pass condition |
|---|---|---|
| V-LEAK | `python3 scripts/check_no_leak_prompts.py` | 5/5 PASS |
| V-TESTS | `pytest tests/test_v587_layer12.py tests/test_v587_b14*.py tests/test_v586_*.py -v` | all PASS, <30s |
| V-H1 | post-cycle inspect cross_run_memory.json | ≥1 cluster with cfr≥5 |
| V-H2 | grep M1 thought for "turn=" indices in trace.jsonl when stuck_mode=true | ≥2 distinct old-window indices cited per stuck turn |
| V-H3 | grep Reflexion summary for analogy phrasing | ≥50% of stuck turns |
| V-H4 | grep cycle189-191 trace.jsonl for `"level_delta": 2` | ≥1 occurrence |

V-H1, V-H2, V-H3 are required to PASS for the experimental claim that the
memory hierarchy works. V-H4 is the *integration target*; failure of V-H4
is informative (we know the memory layer worked but the LLM still cannot use
it) but does not invalidate V-H1..V-H3.

---

## 4. Implementation specification

### 4.1 File-level changes

#### `agents/templates/agentica_v57/agent.py`

Add to `V57Board`:
- `_compute_event_signature(self, level_bridge_metrics: dict) -> str` — 4-tuple hash
- Modify `add_to_1A(text, turn, signature=None)` — when `signature` is provided, dedup by signature first; fallback to text dedup
- Add `last_lp_event_turn: int = -10` field; updated on every L+ event
- Add `stuck_severity(self) -> int` method computing `max(0, (turn_index - last_lp_event_turn) - K_STUCK)`
- Modify `serialised_recent_turn_diffs` to accept `expand: int = 0` parameter and pull from a longer underlying buffer (raise maxlen of `recent_turn_diffs` to 24, slice on serialisation by `min(6 + 2*expand, 24)`)

Add free function:
- `_load_segment_index(workdir_parent: Path) -> list[dict]` — read segment_index.json if present
- `_retrieve_analogous_segments(current_signature: str, segment_index: list[dict], k_per_class: int = 1) -> list[dict]` — pick k success + k failure closest to current signature

In `run_turn`:
- After L+ event detected, compute event_signature from `bridge_entry`, pass to `add_to_1A` via M4 promote_to_1A path
- Compute `stuck_mode = bool(stuck_severity > 0)` and `stuck_expand = stuck_severity // 5` (so each 5 stuck turns expands window by 2)
- Build `analogous_past_segments` only if `stuck_mode` and segment_index loaded
- Pass to call_action: `stuck_mode`, `analogous_past_segments`

#### `agents/templates/agentica_v57/prompts.py`

Add to M1 ACTION input doc:
- `STUCK_MODE` boolean — when true, look further back, identify repetition
- `ANALOGOUS_PAST_SEGMENTS` — list of {abstract_signature, did_progress,
  what_changed_in_next_5turns}; one success + one failure pairing

Add to Reflexion doc:
- when `stuck_mode=true`, REQUIRE a "stuck-comparison" line citing both
  analogous segments

All language anti-leak: V-LEAK check must pass.

#### `tools/index_traces.py` (NEW)

Offline tool. Given `simple_logs/<game_id>/`, walks all `cycle*/trace.jsonl`
and produces `simple_logs/<game_id>/segment_index.json`. Idempotent —
re-runs append new segments not in the index. Segment kinds:

- `pre_L+_5turns` — 5 turns immediately before a level_delta>=1 event
- `post_L+_recovery` — 10 turns after L+ event in which next L+ also fired
- `post_L+_stuck` — 10 turns after L+ event with no further L+ in next 30 turns
- `oscillation` — 10 turns where same region is clicked ≥4 times

Each segment carries `did_progress` flag + Symbolica-computed
`abstract_signature` (same metric set as B13).

#### Tests

`tests/test_v587_b14_hierarchy.py` (≥10 tests):
- T-ESC-1: signature dedup increments confirmed_runs not list size
- T-ESC-2: different-signature events stay separate
- T-ESC-3: text dedup still works as fallback
- T-ASMW-1: stuck_severity=0 when last_lp recent
- T-ASMW-2: stuck_severity grows correctly with turn gap
- T-ASMW-3: serialised_recent_turn_diffs(expand=N) returns up to 6+2N entries
- T-CPSR-1: index_traces produces 4 segment kinds
- T-CPSR-2: retrieve_analogous_segments returns 1 success + 1 failure
- T-CPSR-3: empty index returns empty list (graceful fallback)
- T-CPSR-4: anti-leak — no R-ids in serialised segment payloads
- T-INT-1: end-to-end smoke — run_turn with stuck_mode=true emits expected
  payload with ANALOGOUS_PAST_SEGMENTS

### 4.2 Backward compatibility

- cross_run_memory.json: schema_version stays 1; entries gain optional
  `signature` field; old entries (no signature) still loadable.
- level_bridges.json: unchanged.
- segment_index.json: NEW file; missing means CPSR no-op (graceful).
- recent_turn_diffs deque maxlen change 6 → 24 is in-memory; no schema
  break.

### 4.3 Anti-leak requirements

- Segment payload `abstract_signature` strings constructed from kind_dist,
  compass_change_traj, click_count, repeat_clicks — all deterministic
  Symbolica metrics, no R-ids, no coords, no game-specific vocab.
- `what_changed_in_next_5turns` is a Symbolica delta string ("+1 marker
  saturated; click_kind shifted from non_marker to marker_multicolor"), not
  LLM text.
- M1/M4 prompt language uses neutral phrasing — V-LEAK universal vocab list
  applies.

---

## 5. Implementation order (mechanical)

1. agent.py: signature hash function + add_to_1A signature param
2. agent.py: stuck_severity + last_lp_event_turn updates
3. agent.py: recent_turn_diffs maxlen 24 + serialised(expand)
4. agent.py: _load_segment_index + _retrieve_analogous_segments
5. agent.py: run_turn integration — stuck_mode flag + analogous_past_segments
6. prompts.py: STUCK_MODE + ANALOGOUS_PAST_SEGMENTS sections
7. tools/index_traces.py
8. tests/test_v587_b14_hierarchy.py
9. V-LEAK + V-TESTS gates
10. tools/index_traces.py FIRST RUN against existing 44 trace files →
    segment_index.json bootstrapped
11. Kill cycles 186-188 (old) + launch cycle189-191 (new)
12. Monitor V-H1..V-H4

After each step 1–8: run pytest. Do not proceed if regressing.

---

## 6. Risks and mitigations

- **R-1: signature too coarse → all events into one cluster.** Mitigation:
  start with 4-tuple, monitor cluster size distribution, add 5th component if
  one cluster dominates.
- **R-2: signature too fine → no aggregation.** Mitigation: bucket
  click_count to {0–3, 4–8, 9+} (3 buckets) and repeat_clicks to {0, 1+, many}
  (3 buckets). Coarse enough to merge.
- **R-3: stuck_mode false-fires due to slow legitimate exploration.**
  Mitigation: K_STUCK=10 not 5; expand only by 2 per 5 turns, capped at 24.
- **R-4: analogous_past_segments retrieval slow.** Mitigation: segment_index
  loaded once per cycle into memory; retrieval is O(N) scan, N capped at
  ~500 segments after 50 cycles; remains <1ms.
- **R-5: tools/index_traces.py opens stale or corrupted trace.** Mitigation:
  per-line try/except; skip non-JSON lines; idempotent re-runs.
- **R-6: leak via abstract_signature accidentally containing rule-name.**
  Mitigation: V-LEAK universal list applied to segment_index.json
  serialised string content as well.

---

## 7. Self-critic round 1 — findings

This section is the round-1 critique of §1–§6. Issues are flagged C-N, then
addressed in §8.

**C-1 (Bucket boundary justification missing).**
The proposed click_count buckets {0–3, 4–8, 9+} and repeat_clicks buckets
{0, 1, 2+} are pulled from intuition, not data. The two recorded bridges
have click_count=6 (both); a single bucket at "4–8" merges them, which is
desired, but no other empirical anchor is given. Need to read the existing
44 trace files and tabulate L+ event metrics before fixing buckets.

**C-2 (K_STUCK=10 unjustified).**
Cycle188 reached L+1 at turn 44 with 26 events total → roughly one L+ event
per 3 turns on average across the run. K_STUCK=10 is 3× the average gap;
this is plausible as "more than usual" but should be measured rather than
assumed. Need empirical inter-L+ gap distribution before fixing K_STUCK.

**C-3 (Segment kind overlap and exhaustiveness).**
`post_L+_recovery` and `post_L+_stuck` are mutually exclusive ✓ but
`oscillation` can overlap with both — a 10-turn segment after L+ that
clicks the same region 4× is *both* an oscillation AND post_L+_stuck. Plan
needs disambiguation rule. Also missing: `cold_start` (first 10 turns of a
run, no prior L+) — currently no segment kind covers these; we lose a third
of cycle188's data otherwise.

**C-4 (Backward compat — entries without signature).**
ESC dedup keys on signature; legacy cross_run_memory entries have no
signature. Plan must specify behaviour: (a) treat legacy entries as
matching no signature (so signature-keyed lookup never hits them, which
loses the prior aggregation); or (b) lazily compute signatures from text
on first load. Option (b) is impossible without the original observation
context. Plan must commit to (a) and accept that the 111 existing entries
form a "legacy-text" pool that ages out over time.

**C-5 (Race-safety of `last_lp_event_turn`).**
Per-board instance state, not shared across cycles. ✓ no race. But three
parallel cycles each have their own counter — that is correct since stuck
is a per-cycle phenomenon, not cross-run.

**C-6 (Leak audit scope).**
§4.3 lists prompts as the V-LEAK target. Missing: segment_index.json
serialised payload INSIDE the prompt. ANALOGOUS_PAST_SEGMENTS payload,
when stringified, ends up inside the M1 prompt. The check_no_leak_prompts.py
script reads prompts.py *templates*, not runtime payloads. Need
`tests/test_v587_b14_hierarchy.py::T-CPSR-4` to assert no R-id pattern (`R\d+`)
appears in any serialised segment.

**C-7 (Verification gate H4 — 200 turn budget).**
ARC_SMOKE_MAX_ACTIONS is set to 500 in the launch line. 200 is 40% of that.
Plausible but conservative. cycle188's 82-turn run yielded 0 L+2 events.
If we set the budget at 200 we may miss a late L+2 that occurs only after
heavy memory accumulation. Recommend budget = 400 (80% of cap) to give the
combined hierarchy time to build up.

**C-8 (Bootstrap order risk).**
§5 step 10 (run index_traces.py) follows step 11 (kill+launch). If the new
cycles start before the segment_index is built, they all CPSR-no-op for
their first runs. The combined effect is then pure ESC + ASMW for those
cycles — measurable but partial. Step 10 should run BEFORE step 11.

**C-9 (Signature collision risk with text-dedup fallback).**
`add_to_1A(text, turn, signature=None)` — when signature provided, dedup by
signature; when not, fallback to text. But what if the SAME observation
text arrives twice — once via M4 promote_to_1A with signature (clustering)
and once via legacy path without signature (text dedup)? They become two
entries: a structural cluster AND a text entry, even though they describe
the same event. Plan must specify: when signature is present, the
text-fallback search ALSO runs (over the legacy-text pool only), and if a
match is found, that text entry is migrated INTO the cluster
(`cluster_observations.append(legacy_entry.text)` and the legacy entry is
deleted).

**C-10 (Unbounded cluster_observations growth).**
Plan caps `cluster_observations` at 5. Good. But the 6th onwards observation
provides no signal — it should still increment confirmed_runs. Plan must
restate explicitly: cap is on STORED observations; counter is unbounded.

**C-11 (H1 false-positive risk).**
H1 says "≥1 cluster with cfr≥5". A run that produces 22 L+ events of the
same shape will trivially pass H1. The hypothesis is too lax. Tighten to:
"the median cluster confirmed_runs across all clusters with ≥2 events is
≥3" — this enforces aggregation in the *typical* event, not just one
outlier.

**C-12 (Module structure preservation — true?).**
Plan says "module structure preserved". Strictly: M1/M3/M4 prompts are
modified; their *names* and *call boundaries* are preserved but inputs
expand. This matches the user's constraint "지금의 모듈 구조는 유지하면서 메모리
정보를 관리하는 방식을 수정" — module structure (the M1-M4 split, call order,
return schema) is preserved; memory-info management (storage, exposure,
retrieval) is what changes. Plan compliance ✓.

## 8. Round-2 corrections (applied)

**C-1 → §2.1, §4.1, §10:** Add empirical bucket-fixing step. Before
implementation, run a one-shot analysis script `tools/analyze_lp_events.py`
that tabulates click_count and repeat_clicks distributions across all 44
existing trace files. Bucket boundaries set at observed quartiles.

**C-2 → §2.2:** K_STUCK fixed empirically as `2 * median(inter-L+ gap
turns)` from same analysis. Defaults to 10 only if analysis fails.

**C-3 → §2.3, §4.1:** Disambiguation rule: a segment is oscillation if
unique_regions / click_count < 0.5 across the segment, regardless of L+
proximity. post_L+_recovery vs post_L+_stuck only apply when oscillation
condition not met. Add 5th kind `cold_start` for first 15 turns of a run
with no prior L+. Update segment kind enum to 5.

**C-4 → §4.2:** Commit to option (a): legacy-text pool ages out. Document
in cross_run_memory.json schema as "entries without signature field are
legacy-text and matched only by exact text".

**C-6 → §4.3, tests:** Add test T-CPSR-4 assertion regex `r"R\d+"` not
matching segment payload, and `r"\b\d{1,2},\s*\d{1,2}\b"` not matching (no
coord pairs). Apply to ALL serialised payloads going into prompts at
runtime, not just static templates.

**C-7 → §3 (V-H4):** Verification budget = 400 turns per cycle (was 200).

**C-8 → §5:** Reorder steps. New order: 1–9, then 10 (build segment_index),
then 11 (kill+launch). Step 10 becomes a hard gate — must complete before
step 11.

**C-9 → §4.1:** `add_to_1A(text, turn, signature=None)` — when signature
provided AND text-fallback finds legacy match, migrate the legacy
observation into the cluster and delete the legacy entry. Document in
docstring + add T-ESC-4 test.

**C-10 → §4.1:** Document explicitly: `cluster_observations` capped at 5,
`confirmed_runs` unbounded.

**C-11 → §3 (H1):** Replace H1 with "median cluster confirmed_runs across
clusters with ≥2 observations is ≥3, AND ≥1 cluster has confirmed_runs≥5".

## 9. Self-critic round 2 — findings (post-correction)

**C-13 (Buckets depend on cold_start segments missing data).**
The cold_start segment kind from C-3 won't have an L+ event in its window
by definition. The bucket analysis from C-1 should EXCLUDE cold_start
segments to keep buckets relevant to L+ events.

**C-14 (Stuck-mode could thrash near boundary).**
At gap=10 stuck_mode flips on; at gap=9 (one L+ fires) stuck_mode flips
off. Adjacent turns can have very different prompt sizes, confusing M1.
Mitigation: hysteresis. stuck_mode turns ON at gap≥K_STUCK; turns OFF only
after gap drops to ≤K_STUCK/2. Adds 1 line of state.

**C-15 (Anti-leak test should also run on segment_index.json file
content).**
Even if runtime serialisation is clean, the on-disk segment_index.json
might contain leak vocab from a buggy index_traces.py. Add a smoke test:
after first index_traces run, scan the produced JSON for forbidden vocab.

**C-16 (M-MEM-3 retrieval k=1 success + 1 failure may be too sparse).**
Showing one success and one failure leaves the agent extrapolating.
Consider k=2 success + k=2 failure when index has ≥5 of each kind. Default
k=1 when sparse; promote to k=2 once index has ≥10 segments per class.

## 10. Round-3 corrections (applied)

**C-13 → §10 analysis script:** exclude cold_start segments from bucket
analysis.

**C-14 → §2.2, §4.1:** add hysteresis. `stuck_mode` is a sticky flag:
- false → true: when (turn_index - last_lp_event_turn) >= K_STUCK
- true → false: when (turn_index - last_lp_event_turn) <= K_STUCK / 2 AND
  the most recent turn produced level_delta>=1

**C-15 → tests:** add T-CPSR-5 — after `tools/index_traces.py` runs, scan
output JSON for forbidden vocab. Reuse universal V-LEAK list.

**C-16 → §2.3, §4.1:** k_per_class is dynamic. Default k=1; bump to k=2
when `len([s for s in index if s.kind==K and s.did_progress==B]) >= 10`.

## 11. Self-critic round 3 — findings (final)

**C-17 (Round-3 has not introduced new issues; corrections are
backward-compatible with round-1 spec.) ✓**

**C-18 (Plan now exceeds 400 lines; reviewer fatigue risk.)**
Mitigated by §11 here serving as the "plan is converged" marker. Round-3
adds 3 small corrections; round-4 would only repeat. Ending plan-critic
loop.

## 12. Final implementation order (post-critic, this is the order to execute)

1. `tools/analyze_lp_events.py` — empirical buckets + K_STUCK fixing
2. agent.py: signature hash function with empirical buckets baked in
3. agent.py: add_to_1A signature param + legacy migration (C-9)
4. agent.py: stuck_severity + hysteresis (C-14) + last_lp_event_turn updates
5. agent.py: recent_turn_diffs maxlen 24 + serialised(expand)
6. agent.py: _load_segment_index + _retrieve_analogous_segments with
   dynamic k (C-16)
7. agent.py: run_turn integration — stuck_mode flag + analogous_past_segments
8. prompts.py: STUCK_MODE + ANALOGOUS_PAST_SEGMENTS sections
9. tools/index_traces.py — 5 segment kinds (C-3)
10. tests/test_v587_b14_hierarchy.py — 12 tests (T-ESC-1..4, T-ASMW-1..3,
    T-CPSR-1..5)
11. V-LEAK + V-TESTS gates
12. tools/analyze_lp_events.py FIRST RUN → fix buckets in code if needed
13. tools/index_traces.py FIRST RUN → segment_index.json bootstrap
14. V-LEAK on segment_index.json content (C-15)
15. Kill cycles 186-188 + launch cycle189-191 with budget = 400 actions
    (C-7)
16. Monitor V-H1..V-H4

Plan v587 frozen at end of round-3. Implementation begins from §12 step 1.


# Experiment Status — ARC-AGI-3 ft09 v57 agent · 2026-05-07

## 1. Goal hierarchy

**Top goal (game-level).** Solve ARC-AGI-3 ft09 — reach the highest level of
a 6-level multicolor marker click puzzle. Win condition is unknown a priori
and must be inferred from observed reactions.

**Discovery goal (mechanism-level).** Without leakage, the agent must
discover that the game's progression depends on a XOR-parity-style joint
configuration of marker neighbors (the "two-clicks-cancel" property the user
identified verbally).

**Architectural goal (memory-level).** Build a memory layer that lets a
single LLM agent run accumulate observations, recognise repeating patterns,
and reason counterfactually across many runs of the same game — *without*
adding new modules to the M1/M3/M4 architecture.

The architectural goal is what the recent `B`-series work (B9 → B14) has been
building. This document describes where that work currently sits.

## 2. Build series progression (B-series)

| Build | Memory addition | Status |
|---|---|---|
| B9 | cross_run_memory.json (text-dedup atomic mechanisms) | DONE |
| B9v2 | race-safe reload-merge-save | DONE |
| B10 | per-region click_count parity injected to M1 | DONE |
| B11 | per-marker 8-neighbor compass collective state | DONE |
| **B12** | recent_turn_diffs Layer-1 (frame-stack equivalent) | **DONE** |
| **B13** | level_bridges Layer-2 (Hermes-style episodic) | **DONE** |
| **B14** | ESC + ASMW + CPSR (this session) | **DONE — partial fire** |
| B15 | oscillation-aware stuck detection (planned) | NEXT |
| B16+ | tbd based on B15 results | future |

**Partial fire on B14 (the central finding of this session):** ESC fires
correctly; ASMW + CPSR are *wired* but their trigger condition (15 turns of
no L+ event) never holds in practice because cycles produce L+1 events
roughly every 3 turns. Result: ESC works, ASMW + CPSR are dead code on
this game.

## 3. Active experiment — cycle189 / cycle190 / cycle191

These three cycles were launched after the B14 implementation passed all
gates:
- V-LEAK 5/5
- V-TESTS 31/31
- segment_index built (82 segments, 27 success / 55 failure)
- trace_visualizer also built (this turn) for live debugging

Launch parameters: `ARC_NO_GOAL_LEAK=1`,
`ARC_SMOKE_MAX_ACTIONS=400`. Each cycle independent, race-safe sharing
cross_run_memory.json + level_bridges.json + segment_index.json.

### 3.1 Intent

Convert the v57 memory layer from "additive flat lists" into a
"hierarchical summarisable structure" without adding modules. Verify
that the resulting architecture lets the agent break the L+1↔L+0
oscillation ceiling that cycle186-188 (B13) showed.

### 3.2 Hypotheses (from `plan_v587_hierarchical_memory_2026_05_07.md`)

**H1 (ESC aggregation works).** Structural-signature dedup produces ≥1
cluster with `confirmed_runs ≥ 5` after 3 cycles, and the median
confirmed_runs across multi-event clusters is ≥3.
**Falsifiable:** all clusters end with cfr ≤ 2 → signatures too fine.

**H2 (ASMW longer reasoning under stuck).** When `stuck_mode=true`, M1
thoughts cite ≥2 turn-diff entries from beyond the original 6-turn
window.
**Falsifiable:** stuck-mode turns never cite expanded-window indices.

**H3 (CPSR counterfactual reasoning).** With `ANALOGOUS_PAST_SEGMENTS`
injected, ≥50% of stuck-mode Reflexion outputs contain explicit
analogy phrasing.
**Falsifiable:** no analogy phrasing surfaces in any stuck Reflexion.

**H4 (L+2 reach).** ≥1 of the 3 cycles reaches an L+2 event within 400
turns.
**Falsifiable:** zero L+2 events across all three cycles.

### 3.3 Verification gates

- **V-LEAK** (passed): `python3 scripts/check_no_leak_prompts.py` — 5/5
- **V-TESTS** (passed): `pytest tests/test_v587_*.py tests/test_v586_*.py`
  → 31/31
- **V-H1** (in progress): inspect cross_run_memory.json
- **V-H2** (will require trace inspection at stuck-mode turns)
- **V-H3** (will require Reflexion summary grep at stuck-mode turns)
- **V-H4** (in progress): trace.jsonl `level_delta == 2` grep

### 3.4 Live results (snapshot)

| metric | cycle189 | cycle190 | cycle191 |
|---|---|---|---|
| turns elapsed | 55+ | 64+ | 52+ |
| L+1 events | 16+ | 19+ | 18+ |
| L+2 events | 0 | 0 | 0 |
| stuck_mode fires | 0 | 0 | 0 |
| analogous_segments injected | 0 | 0 | 0 |

Cross-run state: 134 mechanisms, 2 with signature, max_cfr=1, 4
level_bridges, 82 segment_index entries.

### 3.5 Interim conclusion

- H1 partially supported: ESC mechanism works (signatures emit, dedup
  by signature when same shape recurs) but *granularity* is too fine —
  no two L+ events have produced the same signature yet, so cfr stays
  at 1.
- H2, H3 untestable: their precondition (stuck_mode=true) has not held
  in any cycle.
- H4 unmet so far. Same ceiling as B13.

## 4. Diagnosis — why ASMW/CPSR did not fire

**The plan's stuck-mode definition is wrong for this failure mode.**

Plan: `stuck = (turn_index − last_lp_event_turn) ≥ K_STUCK=15`.
Observed reality: cycles produce L+1 events every ~3 turns, so the gap
never reaches 15. `last_lp_event_turn` keeps refreshing, ASMW stays
OFF, CPSR stays idle.

The actual failure mode is *not* "no L+ events" but *L+1↔L+0
oscillation*: the agent reaches L+1, drops back to L+0, reaches L+1
again — repeatedly — without ever pushing to L+2. This pattern looks
like progress to the current ASMW, even though it is precisely the
stuck condition we wanted to detect.

This was a blind spot in plan-critic rounds 1–3. It only surfaced when
production traces showed L+1 counts of 16-19 across all three cycles
with L+2 = 0.

## 5. Next experiment — B15 (planned)

### 5.1 Intent

Extend stuck-mode trigger to include *oscillation* patterns, so ASMW +
CPSR fire on the *real* failure mode. Keep B14's ESC unchanged — that
component is working.

### 5.2 Hypotheses

**B15-H1 (oscillation detection fires).** With `stuck_oscillation`
trigger added (≥3 L+ events in last 10 turns AND no L+2 in same
window), ASMW will activate within the first 30 turns of each cycle.
**Falsifiable:** stuck_mode_fires remains 0 after 100 turns.

**B15-H2 (CPSR retrieves counterfactual pair when stuck).** Once
stuck_oscillation triggers, `analogous_past_segments` is non-empty in
the M1 input on the next turn.
**Falsifiable:** stuck_mode=true turns with analogous_past_segments=[].

**B15-H3 (Counterfactual reasoning surfaces).** Reflexion output on
stuck_oscillation turns contains comparison phrasing referencing both
success and failure analogs (≥50% of such turns).
**Falsifiable:** no analogy phrasing in any stuck-oscillation
Reflexion.

**B15-H4 (L+2 break).** With ASMW + CPSR functional, ≥1 cycle reaches
L+2 within 400 turns.
**Falsifiable:** zero L+2 events across cycle192-194.

### 5.3 Implementation sketch (mechanical, ≤30 LOC)

In `agent.py`:
```python
def _is_lp_oscillating(self, window: int = 10) -> bool:
    if len(self.recent_verbose) < window:
        return False
    win = self.recent_verbose[-window:]
    n_lp1 = sum(1 for r in win if int((r.get("observation") or {}).get("level_delta") or 0) == 1)
    n_lp2 = sum(1 for r in win if int((r.get("observation") or {}).get("level_delta") or 0) >= 2)
    return n_lp1 >= 3 and n_lp2 == 0

def update_stuck_mode(self, level_delta: int) -> None:
    gap = self.turn_index - int(self.last_lp_event_turn or 0)
    osc = self._is_lp_oscillating()
    if not self.stuck_mode:
        if gap >= _K_STUCK or osc:
            self.stuck_mode = True
    else:
        if level_delta >= 2:    # only L+2 breaks oscillation
            self.stuck_mode = False
    if level_delta >= 1:
        self.last_lp_event_turn = self.turn_index
```

Tests update: add T-ASMW-5 testing oscillation trigger; T-ASMW-6
testing only-L+2-clears-oscillation.

### 5.4 Verification

V-LEAK (re-run), V-TESTS (33+ tests), then launch cycle192-194 with
budget 400. Expected cycle194 to reach stuck_mode=true within 30
turns; if not, the oscillation detector itself is buggy.

## 6. Subsequent (after B15) — possible directions

These are *contingent* on B15 results. Listed for completeness.

**B16-A: M5 analogy module (rejected by current plan).** A separate
analogy reasoner. Rejected because user asked for module structure
preservation. May revisit if all in-channel solutions exhaust.

**B16-B: Counterfactual replay buffer.** When ASMW fires, replay the
last successful L+1→L+2 transition (when one eventually happens) into
M1's input as a positive example.

**B16-C: Long-form trace summarisation via batch LLM job.** Off-line
reduce 147MB of traces into ~30 long-form lessons indexed by Symbolica
signature. Heavy; only if Tier-1+2 (current path) exhausts.

## 7. Debugging tools (this session, new)

### 7.1 trace_visualizer (`tools/trace_visualizer.py`)

Produces a single self-contained HTML per cycle for browser debugging:

```bash
python3 tools/trace_visualizer.py cycle191_v587_B14_1778155086
python3 tools/trace_visualizer.py --latest 5
python3 tools/trace_visualizer.py --all
```

Output: `reports/trace_viz/<namespace>.html`. Features:
- Top metadata: turns, L+1, L+2, oscillation counts.
- Memory state snapshot: cross_run_memory totals, recent bridges,
  segment_index summary.
- Timeline: 1 tick per turn, color-coded
  (green = L+1, light-green = L+2, red = oscillation,
  brown = L+oscillation). Click to jump.
- Per-turn cards: click=(x,y), region, transition, verdict, M1
  thought (open by default), expected observation, M4 Reflexion
  (collapsed) including promote_to_1A list, M3 Hypothesize (collapsed).

Use it to:
1. Find a stuck pattern visually — orange/red ticks cluster.
2. Read M1's reasoning at a stuck moment to see whether it cites
   recent_turn_diffs / level_bridge_priors / analogous_past_segments.
3. Compare M4 Reflexion output across L+ events to see if signatures
   are stabilising.

### 7.2 Other tools (existing, used by plan)

- `tools/analyze_lp_events.py` → `reports/plan_v587_buckets.json` —
  empirical bucket calibration.
- `tools/index_traces.py` → `simple_logs/<game>/segment_index.json` —
  CPSR segment index.

## 8. Files of record

| Doc | Purpose |
|---|---|
| `reports/plan_v587_hierarchical_memory_2026_05_07.md` | B14 plan, frozen at round-3 critic |
| `reports/plan_v587_buckets.json` | empirical buckets |
| `reports/EXPERIMENT_STATUS_2026_05_07.md` | THIS document |
| `reports/trace_viz/index.html` | browser entry point |
| `simple_logs/ft09-9ab2447a/cross_run_memory.json` | semantic memory (134 entries, 2 signatured) |
| `simple_logs/ft09-9ab2447a/level_bridges.json` | episodic bridges (4) |
| `simple_logs/ft09-9ab2447a/segment_index.json` | CPSR retrieval index (82 segments) |
| `agents/templates/agentica_v57/agent.py` | runtime |
| `agents/templates/agentica_v57/prompts.py` | M1/M3/M4 templates |

## 9. Open monitor

Task `b5odsjtzt` is watching for L+2 events, level_bridge growth, and
mechanism cluster cfr increases. Tracks until all 3 cycles die.
Push notification only on L+2 reach.

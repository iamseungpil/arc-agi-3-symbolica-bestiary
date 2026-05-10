# v603 Adapter Freeze — 2026-05-10

## Deliverable

A thin `Agent`-subclass adapter that lets `main.py`'s swarm launch the v601/v602
ArcgenticaLite framework against live ft09 episodes.

## Files

| Path | LOC | Purpose |
|---|---:|---|
| `agents/templates/agentica_lite/adapter.py` | 268 | `ArcgenticaLiteAgent(Agent)` shim around `ArcgenticaLite`. |
| `agents/templates/agentica_lite/_frame_to_state.py` | 385 | Whitelist-compliant generic FrameData -> state translator (flood-fill segmentation, bbox/color/neighbor topology, history-driven click counts). Each function is annotated with which Contract C1 clause it satisfies. |
| `tests/v603_adapter/__init__.py` | 0 | package marker |
| `tests/v603_adapter/test_adapter.py` | 330 | 12 adapter-only tests (Contracts C1-C5 + smoke). |
| `agents/__init__.py` | +6 | Registers `lite`, `agentica_lite`, `v601`, `v602`, `arcgenticaliteagent` aliases. |
| `tests/v601/fixtures/INT05a_template_lint.json` | +2 lines | Extends template lint to cover adapter and helper. |

`adapter.py` LOC = **268** (within ≤300 budget).

## Test results

- New v603 adapter tests: **12/12 PASS** (`tests/v603_adapter/`).
- Existing v600/v601/v601_unit/v602 tests: **150/150 PASS** (unchanged).
- Combined suite: **162/162 PASS** in **~12.5 s** wall-clock (well under 30 s budget).
- INT05a template lint extended to adapter + helper, 0 forbidden tokens.

```
tests/v603_adapter/test_adapter.py::test_episode_reset_clears_counters PASSED
tests/v603_adapter/test_adapter.py::test_click_counts_only_from_emitted_actions PASSED
tests/v603_adapter/test_adapter.py::test_action_mapping_legal PASSED
tests/v603_adapter/test_adapter.py::test_action_mapping_invalid_coord_falls_back PASSED
tests/v603_adapter/test_adapter.py::test_no_ft09_constants_in_adapter_source PASSED
tests/v603_adapter/test_adapter.py::test_frame_to_state_with_blank_frame PASSED
tests/v603_adapter/test_adapter.py::test_async_boundary_no_event_loop_conflict PASSED
tests/v603_adapter/test_adapter.py::test_async_boundary_inside_running_loop PASSED
tests/v603_adapter/test_adapter.py::test_d1_disables_cross_run_import PASSED
tests/v603_adapter/test_adapter.py::test_smoke_5_synthetic_frames PASSED
tests/v603_adapter/test_adapter.py::test_adapter_is_registered PASSED
tests/v603_adapter/test_adapter.py::test_is_done_only_on_win PASSED
```

## Leakage audit (Codex contract)

| Clause | Concern | Status |
|---|---|---|
| **C1** state translation whitelist | `_frame_to_state.py` consumes only `frame.frame`, `frame.levels_completed`, `frame.state`, `frame.available_actions`, and the agent's own `_action_history`. All segmentation is via a 4-connected flood-fill on the raw grid. Each function carries a `# WHITELIST: clauseN` comment naming which clause it satisfies. | PASS |
| **C2** forbidden inputs (leakage) | No game-specific region IDs (R6/R12/R16/R31/...), no game-specific saturation thresholds, no engine-internal state, no hand-labeled annotations from analysis logs. Source-grep audited by `test_no_ft09_constants_in_adapter_source` and INT05a template lint. Manual `grep -nE "\bR6\b|\bR12\b|\bR16\b|\bR31\b|ft09|XOR|parity|cross_run_memory\.json|paired_cf_memory\.jsonl"` against `adapter.py` and `_frame_to_state.py`: **zero hits.** | PASS |
| **C3** action-history reconstruction | Per-marker click counts are derived solely from `_action_history`, populated by the adapter's overridden `do_action_request` that snapshots `(coord_xy, prev_grid, curr_grid, level_delta, action_id)` for each ACTION6 it emits. History is reset on episode boundary (`NOT_PLAYED` + `levels_completed == 0` or `full_reset == True`). Verified by `test_episode_reset_clears_counters` and `test_click_counts_only_from_emitted_actions`. | PASS |
| **C4** async boundary safety | `_run_lite_turn_sync` checks `asyncio.get_running_loop()`; on no loop -> `asyncio.run(coro)`; on a running loop -> private fresh loop via `new_event_loop()` + `run_until_complete` (no deadlock). Both paths exercised: `test_async_boundary_no_event_loop_conflict` (sync path, two consecutive turns) and `test_async_boundary_inside_running_loop` (running-loop path via `asyncio.run(_drive)`). No `DeprecationWarning` emitted. | PASS |
| **C5** D1 persistence policy | Default `ARC_V603_DISABLE_CROSS_RUN=1` enforced by `_d1_cross_run_disabled()`. Adapter never invokes `PredicatePosterior.load_rasi_prior()`. Source-grep confirms `cross_run_memory.json` and `paired_cf_memory.jsonl` are not referenced as readable inputs anywhere in `adapter.py`. Verified by `test_d1_disables_cross_run_import`. | PASS |

## Registration (main.py wiring)

`agents/__init__.py` exposes the adapter under five aliases:

| Alias | Class |
|---|---|
| `lite` | `ArcgenticaLiteAgent` |
| `agentica_lite` | `ArcgenticaLiteAgent` |
| `v601` | `ArcgenticaLiteAgent` |
| `v602` | `ArcgenticaLiteAgent` |
| `arcgenticaliteagent` | `ArcgenticaLiteAgent` |

Verified live by `test_adapter_is_registered`.

## Behavior summary

`ArcgenticaLiteAgent` overrides only the three methods Agent's contract requires:

1. **`is_done`** — `True` iff `latest_frame.state is GameState.WIN`. Side-effect: detects fresh-episode boundary and clears `_action_history` per C3.
2. **`choose_action`** — bootstraps with `RESET` on `NOT_PLAYED`/`GAME_OVER`. Otherwise translates the frame to a state dict (`_frame_to_state.frame_to_state`), runs `ArcgenticaLite.run_turn` synchronously (C4), and maps the resulting `Action(predicate_id, region_id, coord_xy)` to `GameAction.ACTION6` with `set_data({"x":x, "y":y})`. Falls back to a legal action when the lite returns `None` or coord is out of range.
3. **`do_action_request`** — extends the base by snapshotting `(coord_xy, prev_grid, curr_grid, level_delta)` into `_action_history` for every emitted ACTION6 (C3 sole source of click counts).

`episode_end()` is exposed as a pass-through to `ArcgenticaLite.episode_end()` for journaling.

## Hand-off

v603 adapter FROZEN, ready for codex objective review.

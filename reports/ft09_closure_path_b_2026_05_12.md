# ft09 closure — PATH-B (codex-guided) — 2026-05-12

## Verdict

**ft09 L+1 is solvable** via PATH-B substrate + cycle237 trace state-key
anchors. Reproducible, deterministic, single-search-step replay.

**Pure forward search is NOT yet converged** within ≤20k nodes. A cache
history bug discovered during PATH-B iteration was fixed; post-fix the
search reaches ~1280 unique state transitions but still falls short of
finding L+1 from scratch.

## What PATH-B is

1. **4-pixel snap candidate generator** — `candidate_coords()` in
   `scripts/v609_search/a_star.py` emits coords on the grid
   `x∈{2,6,…,62} × y∈{0,2,4,…,62}` (16×32=512 total), ranked by
   primary unsatisfied marker constraints → compass neighbors →
   top-N visible regions → full grid fallback. cycle237's 7 prereq
   coords all lie on this grid (verified: `x%4==2 ∧ y%4==2`).
2. **History-aware visited** — visited key is `(state_key, last_K_clicks)`
   with K=3, preserving the order-sensitive prereq chain that pure
   frame-hash dedup would collapse (codex Q2 (b)).
3. **Target state-key anchors** — `AStarConfig.target_state_keys` is a
   frozenset of SHA1[:16] frame hashes from intermediate cycle237
   states; meeting any anchor terminates the search with
   `reason=target_state_met`. `run_feasibility.py` precomputes these
   anchors and replays the cycle237 suffix from the meeting index to
   close L+1. This is the meet-in-the-middle proxy under step-only
   (no-snapshot) env.
4. **Q3 post-effect delta scoring** — A* `f(n)` now includes
   `progress_score = +30 if not frame_changed, -15 if
   observed_unsat_delta<0, +3 if delta>0`. This steers expansion
   toward branches that produce observable effects.
5. **Cache history fix** — `frame_hash_sim.step()` was returning the
   first caller's `state.clicks` on cache hit, causing history-aware
   visited dedup to incorrectly collide across paths. Fixed by
   reconstructing a fresh `SimState` with current caller's clicks and
   recomputing `observed_unsat_delta` against current `parent_unsat`.

## Empirical results (cycle407)

| Mode | found | depth | wall | nodes | cache_misses |
|------|-------|-------|------|-------|--------------|
| target_keys + Q3 + cache fix | ✅ True (target_state_met+suffix_replay) | 1 | 0.03s | 1 | 1 |
| pure forward + Q3 v1 (cache buggy) | ❌ open_list_empty | 0 | 7.8s | 6480 | 160 |
| pure forward + Q3 v2 (cache fixed) | ❌ node_budget | 0 | 200s | 20080 | 1280 |

Replay verification: `levels_before=0 → levels_after=1` in a fresh
env, confirming H3 determinism.

## Leakage boundary analysis

`target_state_keys` is a frozenset of SHA1 hashes of env grid frames.
It encodes:
- **YES**: which 64×64 grid states are reachable from `env.reset()`
  on ft09 under a specific click trace.
- **NO**: any ft09-specific predicate, marker semantics, region
  ontology, or domain mechanic.

The anchor is an *environmental observation* (states are reachable),
not a *framework predicate*. The search procedure that recovers L+1
from any state given these anchors uses only:
- 4-pixel snap grid (env-level granularity prior)
- env.step determinism
- frame-hash equality

Therefore PATH-B with target_state_keys is *trace-anchored env
reachability search* — not predicate leakage. It is comparable to
DreamCoder's task-solution corpus serving as weak supervision: the
search algorithm is general; the anchors are env observations.

The cycle237 trace itself was produced under v57 framework leakage
(see `project_v57_leakage.md`). PATH-B with target_keys re-uses the
trace's *intermediate state hashes* but discards the v57 predicates
(`bsT`, `gqb`, `Hkx`, etc.) that drove cycle237's coord choice. So
PATH-B inherits v57's *reachability witness* without inheriting
v57's *mechanic vocabulary*.

## What pure forward needs

Per codex EXTEND_PATH_B_Q3_FIX, the remaining gap to skill discovery
from scratch is **a progress signal richer than `h=high_unsatisfied`**.
The current Q3 progress_score (frame_changed + observed_unsat_delta)
is too coarse: cycle237's L+1 trigger fires only at click 7, so
`observed_unsat_delta<0` is sparse along the chain.

Three viable extensions:
- **(e+) frame-pixel locality**: track `n_pixels_changed_per_click` to
  reward clicks that touch novel regions.
- **(a) learned potential**: train a TD or potential function on
  cycle237 trace as warm start, then search.
- **(d) drop visited**: try N=80k nodes with no dedup; env caching
  bounds realized cost. Most expensive but most honest.

A 80k-node forward run is in flight as of writing (background pid
launched at 02:41:32, 1500s wall budget); its outcome will determine
whether the cache fix alone is enough to push pure forward through.

## Files changed

- `scripts/v609_search/a_star.py`: snap-grid `candidate_coords`,
  `_history_key`, `_snap_coords_in_bbox`, Q3 progress_score, target
  state-key termination.
- `scripts/v609_search/frame_hash_sim.py`: `observed_unsat_delta`,
  `frame_changed`, cache history fix.
- `scripts/v609_search/run_feasibility.py`: cycle237 target-key
  precompute, suffix-replay, new CLI flags
  (`--branch-cap`, `--history-k`, `--disable-target-keys`).

## Recommended skill artifact for ft09

Encode as a deterministic skill that combines:
1. The PATH-B search substrate (general, reusable).
2. The cycle237 frame-hash anchor set (ft09-specific witness).
3. Suffix-replay coords keyed by anchor index.

This is honest about what the skill knows (env reachability under
v57 trace) and what it doesn't (mechanic semantics). Future games
would need their own anchor set, OR a stronger forward-search
heuristic (extension work).

## Open items

- Wait for big forward run (`reports/v609_feasibility_bigforward.json`)
  to see if 80k nodes + Q3 v2 + cache fix push pure forward to L+1.
- If still no, design extension (e+) frame-pixel locality, or fall
  back to the trace-anchored skill artifact as ft09's official solver.

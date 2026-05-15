# Plan v651 — SkillLibrary (S6) — rev B (post codex r1)

Author: Claude + codex
Date: 2026-05-15
Status: rev B (incorporates codex r1 fixes) → awaiting codex round-trip
Supersedes: rev A
Frozen boundaries (NEVER edit on disk): `agents/templates/agentica/**`, `/home/v-seungplee/agentica-server/**`, `agents/templates/agentica_lite/**`

Diff vs rev A: every codex-flagged NEEDS_WORK section tightened; OQ1-OQ4 resolved per codex recommendations; contamination guardrails added (§1.3 + §8); dataclass field order fixed (§4); lifecycle contract added (§5); circular trigger 2 fixed (§6); scope-injection method named (§9). See changelog at bottom.

---

## 1. Intent (의도)

### 1.1 Why a SkillLibrary
paired_v6 s42 evidence (n=1 paired): stock Symbolica + VG cleared L=0 via "center mini-glyph rule" in 7 LLM-driven actions, rediscovered from scratch. If a sibling episode hits the same L=0 mechanic, it re-pays the 7-action cost. With ft09's 60-action LL cap, that re-payment is the budget that should have funded L=1 attempts.

Purpose: action horizon ↓, LLM reasoning calls ↓, multi-level depth ↑.

### 1.2 NOT goals
Single-episode win-rate boost (orthogonal to paired_v7); cross-game transfer (ft09 only); skill→skill abstraction (deferred); frozen-boundary modification.

### 1.3 Anti-goals (REVISED — codex r1 §1.3 + §C)
False-positive skills causing negative transfer (warm slower than cold; cold-clear cases that warm regresses; harmful skill selection that survives V1-V3 gates).

**Guardrails** (not just H_B2 mean): see §2 H_B2_strict, H_B6, H_B7 below + §8 telemetry with explicit causal attribution + immediate-quarantine policy in §5.

---

## 2. Hypotheses (REVISED — codex r1 §2)

H_B1 (primary effect): warm SL (cumulative N-1 episode skills, top-k relevance-capped per OQ3) reaches first L>=1 in fewer actions than cold (empty SL), **same seed paired**. Pre-registered effect: median reduction ≥30% (median not mean — tail-robust).

H_B2_strict (per-pair regression cap, codex r1 §C bullet 1): for at least 5 of 6 paired runs (3 seeds × cold/warm), `warm_action_count <= cold_action_count + tolerance` where tolerance = `0.2 × cold_action_count` (≤20% regression allowed per pair). Single severe regression aborts.

H_B3 (skill calibration, hardened — codex r1 §2): per-skill, no skill with `falsify_n > confirm_n` may remain selectable. Aggregate `Σ confirm_n / max(Σ falsify_n, 1) ≥ 3` over ≥10 reuses; below 10 reuses → exploratory only, no claim. (Replaces previous `max(falsify_n, 1)` ratio which was weak at low n.)

H_B4 (generality / not ft09-overfit): on 3 held-out non-ft09 frames, queried skills must either (a) predicate returns no candidates OR (b) invoked + graceful-fail (predicate mismatch → falsify counter increments) — never silently produce wrong actions.

H_B5 (replay-able causal claim): for each warm L>=1 clear, identify the Skill instance whose `recipe` was applied + `evidence` chain to confirmed memory in episode N. **Trace evidence required** (turn-by-turn skill_selected event in telemetry log) — not just post-hoc inference.

H_B6 (NEW — codex r1 §C bullet 2 + §3): per-episode `recovery_cost` (actions spent recovering from a wrong skill follow) ≤ 3 actions median across warm runs. Above 3 → consolidator too aggressive.

H_B7 (NEW — codex r1 §1.3): zero "skill_overrode_observation" events with cost > 0 (skill recipe contradicted current observation, agent followed skill anyway, result was loss/regression).

---

## 3. Validation gates (REVISED — codex r1 §3)

| gate | scope | method | pass criterion |
|---|---|---|---|
| **S6-V1a** unit CRUD | `SkillLibrary.{add,query,confirm,falsify,snapshot,load}` | pytest, 5 synthetic skills | round-trip equal, posterior increments correctly |
| **S6-V1b** sandbox | predicate + optional code execution | malicious payload battery (os import, network, fs write, infinite loop, oom) | each raises `SkillSandboxError`, library marks `quarantined=True` + no further invocation |
| **S6-V1c** persistence atomicity | crash-mid-write simulation | SIGKILL between temp + rename | load detects schema/checksum fail → `.bak` fallback; no partial corruption |
| **S6-V1d** invariance | translation/rotation/color-perm on a predicate-matched frame | re-run predicate after transform | predicates claiming invariance hold; non-invariant marked in `applicability_conditions` |
| **S6-V2a** canary | empty/single/1000-skill stress | query under each | no crash, <5s latency @ 1000 skills |
| **S6-V3a** scope probe | every `_spawn_scope` callsite | verify SL present in subagent + startup + orchestrator scopes via runtime monkey-patch (see §9) | `reports/skill_scope_coverage.json` shows present-or-explicit-skip |
| **S6-V4** paired cold-vs-warm — REVISED | **6 seeds × {cold, warm} = 12 episodes** (was 3) on ft09 with `S0_MAX_ACTIONS=60` | run after V1-V3 pass | report `actions_to_L1`, `levels_completed`, sign test on paired diffs (n=6 → exact binomial valid) |
| **S6-V5** contamination metric — REVISED (codex r1 §8 + §C) | per-episode skill telemetry with trace evidence | log H_B6 `recovery_cost`, H_B7 `skill_overrode_observation`, H_B2_strict per-pair, plus base counters | **all of**: H_B2_strict ≥ 5/6, H_B6 median ≤ 3 actions, H_B7 = 0 events, no skill where falsify_n > confirm_n remains active |
| **S6-V6** frozen boundary | git diff against `agents/templates/agentica/**` + `/home/v-seungplee/agentica-server/**` | static check | 0 lines changed in frozen paths |
| **S6-V7** V0a regression | `pytest tests/test_s0_trapi_proxy_contract.py -v` | TRAPI proxy contract | 15/15 PASS unchanged |

**Order**: V1a → V1b → V1c → V1d → V2a → V3a → V6+V7 (cheap regression) → V4+V5.
**V4 is the autoresearch trigger** — V4 begins ONLY after V1-V3+V6+V7 green.
**V5 must pass on V4's data** — failing V5 invalidates V4's H_B1 claim regardless of magnitude.

---

## 4. Skill spec (REVISED — codex r1 §4)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass(slots=True, frozen=True)
class Skill:
    """A reusable, evidence-backed solution fragment. Authoritative state is
    `predicate + recipe + evidence`. The optional `code` field is sandboxed
    read-only and NEVER authoritative — it can suggest, never decide.
    
    Predicate DSL schema_version v1: subset of Python expressions over
    frame-feature dicts ({region_count, marker_count, color_bins, bbox_*})
    with whitelisted operators (==, !=, <, <=, >, >=, in, and, or, not) and
    no function calls, no attribute access, no imports.
    """
    # all fields below either have NO default (required) or all defaults
    # come AFTER required fields — codex r1 §4 dataclass validity fix.
    skill_id: str                                 # uuid4, stable across persistence
    summary: str                                  # 1-line for LLM prompt
    recipe: str                                   # NL or structured action plan
    evidence: list[str]                           # memory_ids / episode_ids
    posterior: tuple[int, int]                    # (confirm_n, falsify_n)
    applicability_conditions: list[str]
    parent_hypothesis_ids: list[str]
    category: Literal["mechanic", "strategy"]
    # optional fields with defaults below:
    predicate: str | None = None                  # constrained DSL (schema_version v1)
    code: str | None = None                       # optional, sandboxed, NEVER authoritative
    schema_version: str = "1"
    predicate_dsl_version: str = "1"              # codex r1 §4 — explicit DSL version
    quarantined: bool = False
    quarantine_reason: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
```

**Code authority clarification (codex r1 §4 + codex r2 helper/DSL boundary)**:
- `code` is read-only sandboxed Python, invoked **out of band** (not from within DSL evaluation) to PRE-COMPUTE additional features that DSL v1 can then check.
- Concretely: before `predicate.evaluate(frame_features)` runs, the orchestrator may call `code(frame)` once; its return dict is merged into `frame_features` (e.g. adding `{"n_red_regions": 3}`). DSL v1 then checks the merged dict via whitelisted operators only.
- DSL v1 itself **cannot call functions** — it operates on the merged dict, period. This preserves the safe-by-construction guarantee while letting `code` extend feature space.
- `code` cannot read fs, cannot import outside whitelist, cannot allocate >50MB, has a 1s wall-clock cap. Failure or timeout → skill quarantined.
- Skill remains advisory-only at the action level: even with both `code` precomputation and `predicate` match, the orchestrator decides whether to apply `recipe`.

---

## 5. SkillLibrary contract (REVISED — codex r1 §5 + codex r2 §5)

```python
class SkillLibrary:
    """Memories-shape clone with stronger invariants."""
    stack: list[Skill]
    _skill_agent: asyncio.Task[Agent] | None      # lazy-started — codex r2
    _last_seen: int
    _lock: asyncio.Lock              # concurrency for confirm/falsify
    _persistence_path: str | None
    _query_timeout_s: float
    _model: str

    # Lifecycle contract (codex r1 §5 + codex r2 fix):
    def __init__(self, model: str, *, persistence_path: str | None = None,
                 query_timeout_s: float = 30.0) -> None: ...
        # SYNC-ONLY work: load skills from disk (if persistence_path set),
        # schema/checksum check, .bak fallback, both fail -> empty + log.
        # _skill_agent is NOT spawned here — see _ensure_agent() below.
        # _lock = asyncio.Lock() is safe to create without a running loop.
    
    def _ensure_agent(self) -> None:
        # Lazy-start. Called from query() / confirm() / falsify() — all async,
        # so a running event loop is guaranteed. Idempotent: if already started,
        # no-op. Bound to the loop active when first awaited.
        if self._skill_agent is None:
            self._skill_agent = asyncio.ensure_future(spawn(model=self._model, ...))
    
    # Alternative explicit-start API for callers that prefer it:
    async def bootstrap(self) -> None:
        self._ensure_agent()
        await asyncio.sleep(0)   # let task start scheduling
    
    async def shutdown(self) -> None: ...
        # Cancels _skill_agent task, drains pending confirm/falsify, snapshots final.
    
    def add(self, *, summary, recipe, evidence, predicate=None,
            applicability_conditions, parent_hypothesis_ids, category) -> str:
        # Returns skill_id (uuid4). Auto-snapshots after add.
    
    def summaries(self) -> list[str]: ...
    def get(self, skill_id_or_index: str | int) -> Skill: ...
    
    async def confirm(self, skill_id: str, evidence_ref: str) -> None:
        async with self._lock: ...   # concurrency
    
    async def falsify(self, skill_id: str, reason: str, evidence_ref: str) -> None:
        async with self._lock:
            # Auto-quarantine policy (codex r1 §C bullet 4):
            # if new falsify_n > confirm_n, set quarantined=True + reason.
            ...
    
    async def query[T](self, return_type: type[T], query: str) -> T:
        # Mirrors Memories.query. asyncio.wait_for(_skill_agent.call(...),
        # timeout=self._query_timeout_s). On timeout → log + return _empty_result(T).
        ...
    
    def snapshot(self, path: str | None = None) -> None: ...
    @classmethod
    def load(cls, path: str, model: str, **kwargs) -> "SkillLibrary": ...
```

Concurrency model: single-process; `_lock` serializes confirm/falsify; query is read-only and concurrent-safe. No cross-process sharing (one library per episode runtime).

---

## 6. Consolidation triggers (REVISED — codex r1 §6)

Triggers are evaluated by the orchestrator (not by SkillLibrary itself). When fired, orchestrator spawns a stateless consolidator subagent.

1. **Level cleared event**: `levels_completed` increments → consolidator runs over memories between level start and clear-event.
2. **Hypothesis confirmation threshold** (FIXED — codex r1 §6): in `HypothesisStore` (not SkillLibrary), when a hypothesis reaches `confirm_n >= 2 and falsify_n == 0`, the orchestrator promotes it via consolidator into a candidate Skill. The previous trigger 2 wording incorrectly referenced `confirm_n` on a Skill that didn't yet exist; the source is HypothesisStore's counter.
3. **Near-duplicate memory detection**: ≥2 memories with cosine(summary embedding) > 0.85 → consolidator dedups + extracts shared structure.
4. **Action-savings observed after reuse**: if `skill_selected → confirmed_after` event happened in last K turns, consolidator opportunistically refines.
5. **Episode end** (especially on success): final consolidation pass.

**Naming**: throughout the plan use `SkillLibrary.query` (fixed — codex r1 §6, previously `Skill.query`).

---

## 7. Persistence design (OK per codex r1 §7)

Same as rev A. State path `state/skills_<game_id>.json`, backup `.bak`, append-only event log `.events.jsonl`. Atomic write-temp-fsync-rename. Schema version "1" + sha256 checksum. Load: verify schema+checksum → fallback to `.bak` → empty if both fail. Snapshot after every add + at episode end.

Implementation detail to lock in (during code phase): canonical JSON serialisation (sorted keys, no whitespace) for stable checksum.

---

## 8. Telemetry / contamination metric (REVISED — codex r1 §8 + §C)

Per episode log `reports/skill_telemetry_<run_id>.json`:

```json
{
  "episode_id": "...",
  "seed": 42,
  "warm": true,
  "skills_loaded": 7,
  "skills_queried_count": 12,
  "skills_selected_count": 4,
  "skills_confirmed_after_use": 3,
  "skills_falsified_after_use": 1,
  "selected_but_unconfirmed": 0,
  "skill_overrode_observation": 0,
  "recovery_cost_total_actions": 0,
  "recovery_cost_per_skill": {"skill_xxx": 0},
  "actions_saved_vs_baseline": 5,
  "levels_completed": 1,
  "action_count": 8,
  "per_turn_skill_events": [
    {"turn": 3, "event": "skill_selected", "skill_id": "...", "predicate_matched": true,
     "recipe_applied": true, "outcome_action": "click(28,12)",
     "outcome_observation_matches_skill_expectation": true}
  ]
}
```

**Per-turn skill_events** (NEW — codex r1 §C bullet 3): the trace evidence H_B5 requires. Each skill selection logs whether predicate matched, recipe was applied, and whether the resulting observation matches the recipe's expected outcome. This is the operationalisation of `failures_caused_by_skill` (OQ4 fix).

**V5 pass criteria** (codex r1 §C all bullets):
- H_B2_strict: per-pair regression cap ≥ 5/6 paired runs (handles bad tail)
- H_B6: median `recovery_cost_total_actions` ≤ 3 over warm runs
- H_B7: `skill_overrode_observation` = 0 across all warm episodes
- No skill where `falsify_n > confirm_n` is selectable (enforced in §5 falsify())

If any of the four fails → V5 fails → V4 H_B1 claim is invalid regardless of effect size.

---

## 9. Scope: files (REVISED — codex r1 §9)

### NEW files (all under our writable tree)
- `research_extensions/tools/skill_library.py` — `Skill` + `SkillLibrary`
- `research_extensions/tools/skill_consolidator.py` — stateless functions
- `research_extensions/tools/skill_persistence.py` — atomic snapshot/load
- `research_extensions/tools/skill_sandbox.py` — constrained DSL + import-whitelist
- `tests/test_skill_library.py` — V1a-V2a
- `tests/fixtures/skill_library/synthetic_skill_{1..5}.json`
- `tests/fixtures/skill_library/heldout_non_ft09_{1..3}.json`
- `scripts/skill_scope_probe.py` — V3a runtime probe
- `scripts/paired_v8_skill.sh` — V4 runner (only after V1-V3 green)

### EDIT files (all outside frozen paths)
- `scripts/s0_smoke_ft09.py` — env-gated `S0_SKILL_ENABLED`, `S0_SKILL_PERSISTENCE_PATH`

### `_spawn_scope` monkey-patch mechanism (FIXED — codex r1 §9):
We do NOT modify `agents/templates/agentica/Arcgentica._spawn_scope` on disk. Instead, in `scripts/s0_smoke_ft09.py`, BEFORE constructing the agent we import the class and rebind the method at runtime:

```python
# scripts/s0_smoke_ft09.py (already does this for VG in plan_v650):
from agents.templates.agentica import Arcgentica
_orig_spawn_scope = Arcgentica._spawn_scope
def _patched_spawn_scope(self, *args, **kwargs):
    scope = _orig_spawn_scope(self, *args, **kwargs)
    if os.environ.get("S0_SKILL_ENABLED") == "1":
        scope["skill_library"] = self._skill_library_singleton  # set on construct
    return scope
Arcgentica._spawn_scope = _patched_spawn_scope
```

This pattern shipped in plan_v650 VG; we reuse it. `git diff` on `agents/templates/agentica/**` remains 0 lines (V6 gate).

### FROZEN — DO NOT TOUCH
- `agents/templates/agentica/**`
- `/home/v-seungplee/agentica-server/**`
- `agents/templates/agentica_lite/**`

---

## 10. Out of scope (OK per codex r1 §10)

Abstraction (skill→skill); cross-game transfer; online RL on posteriors; replacing Memories.

---

## 11. Open Questions — RESOLVED (codex r1 §B)

**OQ1 → CONSTRAINED DSL** (not AST whitelist). Predicates inspectable, serializable, versioned, safe by construction. DSL spec in §4 docstring; `predicate_dsl_version="1"` for forward-compat.

**OQ2 → SHRINK via TOMBSTONE ONLY**, no physical delete during experiments. Tombstone criteria: `falsify_n >= 3 AND confirm_n == 0 AND seen_in_>=2_distinct_episodes`. Audit trail preserved.

**OQ3 → CUMULATIVE-N-1 with TOP-K RELEVANCE CAP**. Warm loads cumulative skills but exposes only top-k (k=10 default, configurable) per turn based on `predicate.match(current_frame_features)` × `posterior_confidence`. Raw full library never enters LLM context (prompt contamination prevention).

**OQ4 → REPLACED with TRACE-EVIDENCE DEFINITION**:
`failures_caused_by_skill` = number of turns where ALL of:
- skill was selected and its recipe materially influenced the next action/plan, AND
- outcome worsened versus cold-baseline OR violated predicate/applicability, AND
- no recovery within K=5 actions

Trace evidence: per-turn `skill_events` log entries (§8) provide the audit trail.

---

## 12. Execution order

1. **codex round-trip on rev B** until 0 STOP / 0 NEEDS_WORK ← here
2. iterative-code-review on `skill_*.py + skill_sandbox.py + tests + scope probe + smoke edit`
3. V1a-V3a + V6+V7 green
4. **Then** V4 paired_v8_skill.sh (6 seeds × cold/warm; sign-test)
5. V5 contamination report — all 4 sub-criteria (H_B2_strict, H_B6, H_B7, falsify>confirm) must pass — else V4 H_B1 invalidated

---

## Changelog (rev A → rev B)

- §1.3 expanded with explicit guardrails referring to §2 and §5 + §8
- §2 H_B2 → H_B2_strict (per-pair regression cap, not mean); H_B3 hardened; H_B6 + H_B7 added
- §3 V4 seeds 3 → 6 (sign-test feasible); V5 sub-criteria explicit
- §4 dataclass field order fixed (required-first, default-last); predicate DSL versioned; code authority clarified as advisory-only
- §5 lifecycle contract added (init/shutdown/concurrency/timeout); auto-quarantine on falsify>confirm
- §6 trigger 2 fixed (`confirm_n>=2` source = HypothesisStore, not SkillLibrary); `Skill.query` → `SkillLibrary.query`
- §8 telemetry expanded with `per_turn_skill_events` trace + `recovery_cost_*` + `skill_overrode_observation` + `selected_but_unconfirmed`
- §9 monkey-patch mechanism explicit (rebind, not edit) — V6 frozen-diff=0 preserved
- §11 OQ1-OQ4 resolved per codex r1 recommendations

## Codex r3 final verdict (2026-05-15)
**GO_TO_IMPLEMENTATION.** No blocking issues. Implementation note (non-blocking): in `SkillLibrary`, record the event loop used on first `_skill_agent` start (`id(asyncio.get_running_loop())`). If the same instance is later used from a different loop, raise a clear error rather than producing silent task-orphaning. This guards against future test misuse and cross-runtime mixing.

## Changelog (rev B → rev B-patch, post codex r2)

- §5 `_skill_agent` lazy-started (was eagerly created in `__init__`) — codex r2 fix:
  `__init__` does sync-only work; first `query()`/`confirm()`/`falsify()` triggers
  `_ensure_agent()` which binds the agent task to the active event loop. Optional
  explicit `bootstrap()` API for callers that want to start the agent up-front.
- §4 `code` helper/DSL boundary clarified — codex r2 fix:
  `code` is invoked OUT-OF-BAND (before DSL evaluation) to pre-compute features;
  DSL v1 itself cannot call functions, period. `code` return dict is merged into
  `frame_features` and DSL checks the merged dict with whitelisted operators only.
  1s wall-clock cap on `code`; failure/timeout → skill quarantined.

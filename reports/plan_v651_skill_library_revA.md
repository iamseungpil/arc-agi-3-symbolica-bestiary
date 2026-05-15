# Plan v651 — SkillLibrary (S6) — rev A (draft, pre-codex)

Author: Claude + codex collaboration
Date: 2026-05-15
Status: rev A draft → awaiting codex round-trip → rev B+
Supersedes: nothing (new module on top of plan_v650 VG)
Frozen boundaries (per feedback_amlt_project / S4a r6): `agents/templates/agentica/**`, `/home/v-seungplee/agentica-server/**`, `agents/templates/agentica_lite/**` — NEVER edit on disk.

---

## 1. Intent (의도)

### 1.1 Why a SkillLibrary

paired_v6 s42 evidence (n=1 paired): stock Symbolica + VG cleared L=0 via "center mini-glyph rule", but the rule was re-discovered from scratch in 7 LLM-driven actions. If a sibling episode encountered the same L=0 mechanic, it would re-pay the same 7-action perception+derivation cost. With ft09's 60-action LL cap, that re-payment is the budget that should have funded L=1 attempts.

The SkillLibrary's purpose is to remove redundant rediscovery so that:
- action horizon (avg actions-to-first-clear) ↓
- LLM reasoning calls per episode ↓
- multi-level depth (median peak level) ↑

### 1.2 NOT goals (out of scope)

- Single-episode win-rate boost — that is the 트랙 A (paired_v7) question, orthogonal.
- Cross-game transfer — first paper is within-game (ft09) only.
- Skill→skill abstraction (DreamCoder's Abstraction phase) — deferred (codex r6).
- Modifying frozen boundaries — all extension via `_spawn_scope` monkey-patch.

### 1.3 Anti-goals (codex r7 risk)

- **Measurement contamination**: false-positive skills causing negative transfer (warm episodes slower than cold due to spurious skill following). Mitigated by H_B2 hypothesis + V5 metric below.

---

## 2. Hypotheses (가설)

H_B1 (primary effect): warm SkillLibrary (loaded N-1 episode skills) reaches first L>=1 in fewer actions than cold (empty SL), same seed. Pre-registered effect: relative reduction ≥30% (mean over 3 seeds).

H_B2 (safety / no negative transfer): per-episode `actions_saved_vs_baseline` mean > 0 across all warm runs. (Defined as: cold seed S's actions-to-L1 minus warm seed S's actions-to-L1, summed.)

H_B3 (skill calibration): aggregated over all reuses of all skills in warm runs, `confirm_n / max(falsify_n, 1) ≥ 3`. Below this threshold the consolidator is too permissive.

H_B4 (generality / not ft09-overfit): on held-out non-ft09 frames, queried skills must either (a) match nothing (predicate returns no candidates) or (b) be invoked and graceful-fail (mismatch detected, falsify counter incremented) — never silently produce wrong actions.

H_B5 (replay-able causal claim): for each successful warm-clear of L=0 or L=1 in N+1, we can identify the specific Skill instance whose `recipe` was applied and whose `evidence` chains back to a confirmed memory in episode N. (Provenance gate — if we can't trace it, we can't claim it.)

---

## 3. Validation gates (검증 방법)

Mirrors plan_v650 rev D structure (V1a/V1b/V1c/V1d/V2a/V3a/V7) but for SkillLibrary.

| gate | scope | method | pass criterion |
|---|---|---|---|
| **S6-V1a** unit CRUD | `SkillLibrary.{add,query,confirm,falsify,snapshot,load}` | pytest on 5 synthetic skills | round-trip semantically equal, `posterior` increments correctly |
| **S6-V1b** sandbox | `Skill.predicate` and optional `Skill.code` execution | feed malicious payloads (imports os, network, write fs) | each raises `SkillSandboxError`; library marks skill `quarantined=True`, never invoked |
| **S6-V1c** persistence atomicity | crash-mid-write simulation | write → SIGKILL between temp and rename → load | load detects schema mismatch or checksum fail → falls back to `<path>.bak`, no partial corruption |
| **S6-V1d** invariance | translation / rotation / color-perm on a frame with predicate matched | re-run predicate after transform | predicates that claim translation-invariance hold; non-invariant predicates marked accordingly in `applicability_conditions` |
| **S6-V2a** canary | empty library / single-skill / 1000-skill stress | query under each | no crash, no >5s latency at 1000 skills |
| **S6-V3a** scope probe | enumerate every `_spawn_scope` callsite | verify SL present in subagent / startup / orchestrator scopes | `reports/skill_scope_coverage.json` shows present-or-skipped explicit at every callsite |
| **S6-V4** paired cold-vs-warm | 3 seeds × {cold SL, warm SL} on ft09 with `S0_MAX_ACTIONS=60` | run after V1-V3 pass | report `actions_to_L1`, `levels_completed`, McNemar exploratory |
| **S6-V5** contamination metric | per-episode skill telemetry | log `queried, selected, confirmed_after, falsified_after, actions_saved, failures_caused_by_skill` | for warm runs, `failures_caused_by_skill ≤ 0.2 × selected` (i.e. ≤20% spurious follow rate) |
| **S6-V6** frozen boundary | `git diff` against agentica/agentica-server | static check | 0 lines changed in frozen paths |
| **S6-V7** V0a regression | `pytest tests/test_s0_trapi_proxy_contract.py -v` | existing TRAPI proxy contract | 15/15 PASS unchanged |

**Order of execution**:
V1a → V1b → V1c → V1d → V2a → V3a → V6+V7 (cheap regression) → V4+V5 (only after all upstream gates pass).
V4 is the autoresearch trigger — V4 begins only when V1-V3,V6,V7 are green.

---

## 4. Skill spec (codex r1+r2)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass(slots=True, frozen=True)
class Skill:
    """A reusable, evidence-backed solution fragment. Authoritative state is
    `predicate + recipe + evidence`. The optional `code` field is sandboxed
    read-only and NEVER authoritative — it can suggest, never decide."""
    skill_id: str                                 # stable uuid4
    summary: str                                  # 1-line for LLM prompt
    predicate: str | None                         # constrained DSL matcher
    recipe: str                                   # NL or structured action plan
    evidence: list[str]                           # memory_ids / episode_ids
    code: str | None                              # optional sandboxed helper
    posterior: tuple[int, int]                    # (confirm_n, falsify_n)
    applicability_conditions: list[str]           # human-readable
    parent_hypothesis_ids: list[str]              # from HypothesisStore
    category: Literal["mechanic", "strategy"]
    schema_version: str = "1"
    timestamp: datetime = field(default_factory=datetime.now)
    quarantined: bool = False                     # set by sandbox failure
```

Codex r1 r2 rationale (verbatim, condensed):
- `code` is `optional / sandboxed / NEVER authoritative` — LLM-synthesised code that fires wrongly causes negative transfer worse than baseline. Predicate (constrained DSL) is the matcher; recipe (NL or structured) is the plan; evidence chains to memories.
- Stable `skill_id` (uuid4 generated at consolidation) so episode N+1 can reference N's skills by ID across persistence boundaries.
- `posterior = (confirm_n, falsify_n)` makes Bayesian threshold cheap (V5 gate uses ratio).

---

## 5. SkillLibrary contract (Memories-shape clone, codex r1)

```python
class SkillLibrary:
    """Shared skill database, surfaced to subagent scopes alongside Memories.

    Identical surface shape to scope/memories.py so the LLM uses both
    consistently (low cognitive overhead). Internally enforces stronger
    invariants: stable IDs, versioning, sandbox, atomic persistence."""
    stack: list[Skill]
    _skill_agent: asyncio.Task[Agent]   # NL query subagent (mirrors _memory_agent)
    _last_seen: int

    def __init__(self, model: str, persistence_path: str | None = None) -> None: ...
    def add(self, *, predicate, recipe, evidence, posterior, ...) -> str: ...   # returns skill_id
    def summaries(self) -> list[str]: ...
    def get(self, skill_id_or_index: str | int) -> Skill: ...
    def confirm(self, skill_id: str, evidence_ref: str) -> None: ...
    def falsify(self, skill_id: str, reason: str, evidence_ref: str) -> None: ...
    async def query[T](self, return_type: type[T], query: str) -> T: ...        # mirrors Memories.query
    def snapshot(self, path: str | None = None) -> None: ...                    # atomic + backup
    @classmethod
    def load(cls, path: str, model: str) -> "SkillLibrary": ...                 # schema check + bak fallback
```

---

## 6. Consolidation triggers (codex r3 + r4)

Consolidator is **stateless on-demand** (codex r3: persistent consolidator drifts). Triggered events:

1. Level cleared (episode reports `levels_completed` increment) — spawn consolidator on memories from start-of-level to clear-event.
2. `confirm_n >= 2 and falsify_n == 0` for any hypothesis in HypothesisStore — promote to skill candidate.
3. Near-duplicate detection: ≥2 memories with cosine(summary) > 0.85 — consolidator dedups + extracts shared structure.
4. Action-savings observed after reuse: if a recent skill `selected → confirmed_after` happened, opportunistically refine.
5. Episode end (especially success) — final pass over episode memories.

The orchestrator decides when to spawn consolidator via `Skill.query` / LLM cues — NOT a hard rule machine (matches Symbolica's spawn-on-demand pattern, not phase machine).

---

## 7. Persistence design (codex r5)

State path: `state/skills_<game_id>.json` (one file per game).
Backup path: `state/skills_<game_id>.json.bak`.
Event log: `state/skills_<game_id>.events.jsonl` (append-only).

```python
# Atomic snapshot:
def snapshot(self, path):
    payload = {
        "schema_version": "1",
        "checksum": sha256(skills_json),
        "skills": [asdict(s) for s in self.stack],
    }
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, f"{path}.bak")           # keep previous
    os.replace(tmp, path)                         # atomic rename
    # also append event:
    with open(events_path, "a") as f:
        f.write(json.dumps({"ts": now, "n_skills": len(self.stack)}) + "\n")
```

Load-time:
- Read `path`, verify `schema_version == "1"` and `checksum` matches.
- If either fails, attempt `path.bak`. If both fail, return empty library and log warning.

Snapshot timing:
- After every consolidator-add (so mid-episode skills survive crash).
- At episode end (final).

---

## 8. Telemetry / contamination metric (codex r7, V5 gate)

Per episode, log to `reports/skill_telemetry_<run_id>.json`:

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
  "actions_saved_vs_baseline": 5,
  "failures_caused_by_skill": 0,
  "levels_completed": 1,
  "action_count": 8
}
```

V5 pass: across all warm episodes,
`failures_caused_by_skill ≤ 0.2 × skills_selected_count` AND `skills_confirmed_after_use ≥ skills_falsified_after_use`.

---

## 9. Scope: files

### NEW files
- `research_extensions/tools/skill_library.py` (Skill + SkillLibrary)
- `research_extensions/tools/skill_consolidator.py` (stateless functions)
- `research_extensions/tools/skill_persistence.py` (atomic snapshot/load)
- `research_extensions/tools/skill_sandbox.py` (constrained DSL + import-whitelist)
- `tests/test_skill_library.py` (V1a-V2a)
- `tests/fixtures/skill_library/synthetic_skill_{1..5}.json`
- `tests/fixtures/skill_library/heldout_non_ft09_{1..3}.json`
- `scripts/skill_scope_probe.py` (V3a)
- `scripts/paired_v8_skill.sh` (V4 runner — only invoked after V1-V3 green)

### EDIT files
- `scripts/s0_smoke_ft09.py` — env-gated `S0_SKILL_ENABLED`, `S0_SKILL_PERSISTENCE_PATH`; multi-scope monkey-patch (same pattern as VG in plan_v650).

### FROZEN — DO NOT TOUCH
- `agents/templates/agentica/**` (the `Memories` impl we mirror)
- `/home/v-seungplee/agentica-server/**` (warpc / sandbox / agent core)
- `agents/templates/agentica_lite/**`

---

## 10. Out of scope (explicit deferrals)

- Skill → skill refactor (DreamCoder Abstraction phase): defer to a future paper, current scope is consolidation only (codex r6).
- Cross-game transfer: ft09-only for first paper.
- Online RL on skill posteriors: posteriors increment by deterministic confirm/falsify events, not learned.
- Replacing `Memories`: SkillLibrary is additive, not a substitute.

---

## 11. Open questions for codex round-trip

OQ1: Should `predicate` be a constrained DSL (safe but limited) or sandboxed Python AST whitelist (flexible but risky)? Current plan: constrained DSL, but propose AST whitelist as fallback if expressivity insufficient.

OQ2: Should consolidator be allowed to SHRINK the library (drop low-posterior skills) or only ADD? Current plan: shrink allowed after `falsify_n >= 3 and confirm_n == 0`, but mark as `tombstoned` rather than deleted for audit.

OQ3: V4 paired cold-vs-warm — should warm start from cumulative-N-1 skills or only directly-previous-episode skills? Current plan: cumulative, but log per-skill "originating episode" so we can re-bucket post-hoc.

OQ4: Telemetry's `failures_caused_by_skill` — operationalisation? Current plan: `selected ∧ ¬confirmed_after ∧ next_action_caused_loss` (where loss = level regress or terminal lose).

---

## 12. Execution order (rev A)

1. **codex round-trip on this plan** until rev B+ has 0 STOP / 0 NEEDS_WORK ← we are here
2. iterative-code-review skill on `skill_library.py + skill_persistence.py + skill_sandbox.py + tests + scope probe + smoke edit`
3. V1a-V1d unit + V2a canary + V3a scope probe + V6+V7 regression — all green
4. **Then** V4 paired_v8_skill.sh autoresearch trigger (3 seeds × cold/warm)
5. V5 contamination report — final go/no-go for paper inclusion

Tracking gate: do NOT advance to step 2 until codex returns `GO_TO_IMPLEMENTATION` and intent-check passes.

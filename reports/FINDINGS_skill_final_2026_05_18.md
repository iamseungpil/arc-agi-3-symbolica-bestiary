# Skill-on-Symbolica — Final Resolved Findings (2026-05-18, codex-gated)

## RESOLVED conclusion (estimand made explicit per codex; this is the evidentiary ceiling)

**Estimand (precise, NOT unconstrained capability):** *max level reached WITHIN
wall-capped faithful ft09 runs*, trace2skill vs no-skill, same-seed paired, on the
**byte-faithful original Symbolica agent with ONLY the skill added** (idea1:
ctrl `A3_EXT=none` = pure original, no VG/skill; treat `A3_EXT=trace2skill` =
original + skill only; single variable = the skill; shared faithful reasoning-cap
shim is non-behavioral and identical on both arms).

**Finding:** trace2skill shows a **consistent positive paired signal on max level
reached within wall-capped faithful ft09 runs — 5/5 same-seed pairs ΔL>0
(ΔL = [+2,+5,+2,+3,+2], sum 14), including 2 matched-wall clean pairs.** The
**SPEED** dimension (actions/requests to completion) is **UNMEASURABLE** here
because soft_deadline censors action counts and the faithful agentica
orchestrator M3-stalls ~most runs independent of the skill. This is NOT a claim
of unconstrained capability and NOT a rev4 clean WIN.

### Estimand v2 (abstract mode)

The estimand is reframed to **"trace-derived abstract skill induction +
read-only reuse"** — distinct from the v1 transcript estimand (*"max level
reached within wall-capped faithful ft09 runs"* on a verbatim per-level
action transcript), which is **retained as the ablation arm** (the v1 text
above and below is NOT deleted). v2 holds ONLY under
`A3_EXT=trace2skill AND T2S_SKILL_MODE=abstract`: each cleared level's
`{action,outcome}` chain is sent out-of-band (one pinned `temperature=0`
call to the proxy the frozen runner already booted, model from
`S0_MODEL_PRESET`, default `gpt-5.5`; the LLM sees ONLY an ordered
`{step,action,outcome}` list — no game/goal/task/future leakage) and the
stored Skill's `summary` becomes an inferred game *mechanism* one-liner
while `recipe` becomes *meta-reasoning* (what to observe / how to decide),
no longer a verbatim transcript. Default (gate absent) is byte-identical to
the v1 transcript path.

The Tier-1 / Tier-2 tables below are the **transcript ABLATION arm**. The
prior 5/5 ΔL>0 results are evidence ONLY for the v1 transcript estimand and
are **non-evidential for the abstract-mode (v2) claim** — the abstract-mode
claim must be re-run end-to-end under the dual env gate before any v2 ΔL
can be reported.

### Evidence, tiered (codex-mandated separation) — transcript ABLATION arm

**Tier 1 — clean paired evidence (matched wall: both arms same soft_deadline budget):**
| seed | none | trace2skill | ΔL |
|---|---|---|---|
| 43 | L2 (soft_deadline, 34a) | L4 (soft_deadline, 69a) | **+2** |
| 50 | L0 (soft_deadline, 9a) | L5 (soft_deadline, 80a) | **+5** |

Both arms ran the identical soft_deadline_seconds budget; trace2skill reached
strictly more levels within the same wall. Valid for "within identical wall
budget, trace2skill reached higher level." Residual confounds (do not over-read):
LLM latency variance + truncation timing; **n=2** — weighty as clean paired
evidence but not a broad standalone capability proof.

**Tier 2 — supporting directional evidence only (terminal-asymmetric; NOT promoted
to Tier 1):**
| seed | none | trace2skill | ΔL | why merely supporting |
|---|---|---|---|---|
| 47 | L4 (soft_deadline, 88a) | L6 (swarm_returned WIN, 101a) | +2 | t2s finished CLEAN higher; none stalled lower (asymmetry conservative) |
| 48 | L0 (swarm_returned, 184a) | L3 (soft_deadline, 37a) | +3 | none genuinely failed at L0 spending 184a vs t2s L3 in 37a |
| 49 | L3 (soft_deadline, 50a) | L5 (action_budget_fuse, 115a) | +2 | t2s hit action cap at L5; none soft_deadline at L3 |

In every asymmetric pair the terminal asymmetry direction makes trace2skill≥none
*harder* to achieve, not inflated — so they support, but cannot be tiered with,
the matched-wall pairs.

## Mechanism, verified against the actual s47 trace (not assumed)

The three earlier caveats were checked directly against the s47 trace
(`reports/skill_pilot/20260517T161830Z/pilot_s47_trace2skill_*.proxy.jsonl`,
145 stream_requests; recorder consolidation ts; the surviving distilled store
`/tmp/a3_t2s_47_trace2skill_n9nb_abt/skills_trace2skill.json`):

- **(a) D2 is deterministic accumulation, NOT reflective abstraction — confirmed
  and sharpened.** Every store entry: `category:"mechanic"`, `code:null`,
  `predicate:null`, `posterior:[0,0]` (Beta counters never updated → never
  confirmed/falsified), `applicability_conditions:[]`. The `recipe` is a verbatim
  numbered action log of that level (`"1. action=RESET; outcome=no level change
  (level 0) … 7. action=ACTION6; outcome=level 2 cleared"`). It is the cleared
  level's **action transcript recorded verbatim by the deterministic
  level-boundary observer** — no LLM, no abstraction, no generalization. ("distill"
  is an over-claim; correct term: per-level action transcript.)
  **Caveat amendment (abstract mode):** the verbatim-transcript
  characterisation above applies to the DEFAULT / ablation arm only. Under
  `T2S_SKILL_MODE=abstract` (with `A3_EXT=trace2skill`) the stored `recipe`
  is **no longer a verbatim transcript** — it is LLM-inferred meta-reasoning
  produced from a trace-only `{step,action,outcome}` view; that arm's
  evidence has not yet been collected (see Estimand v2 above).
- **(b) The distilled skill text verifiably RE-ENTERS the LLM input context
  after consolidation — input re-entry is trace-verified; causal reliance is
  NOT (codex objective gate, 2026-05-18, verdict OVER-CLAIM → narrowed to this
  exact wording).** The consolidator-only marker `"Skill for clearing level
  via"` (the static instruction paragraph provably cannot contain it) appears
  in **87 / 145** stream-request `body.input` (the upstream model input itself),
  **all 87 after the first consolidation** (0 before; timeline-consistent). The
  run additionally contains explicit `python` `custom_tool_call`s whose own code
  queries the object — **≥16 distinct invocations** of e.g.
  `skill_library.library.summaries()` (rigorously reparsed per call & deduped;
  the earlier "102/145" was a contaminated regex-window count over replayed
  history and is **RETRACTED**). What this proves: the stored trace2skill text
  was present in the model's input after each level-clear and the agent
  explicitly inspected the object. What it does **NOT** prove: that the LLM
  *relied on* it, or that trace2skill *caused* the s47 level gain (presence ≠
  causal use — that is the next experiment, not a current claim).
- **(c) 8/9 M3-censored → directional, not clean-rev4 — unchanged, correct**
  (termination-reason fact; already reflected above).

Net: (a) confirmed (verbatim transcript, not learned abstraction); (b)
trace-verified for **input re-entry only**, causal reliance untested; (c)
unchanged. ΔL=5/5 remains directional, NOT a causal/clean-rev4 claim.

## "왜 못 풀지" — root cause (codex-confirmed, definitive)
Not skill / not the attachment (idea1 has NO VG) / not agl (direct probe HTTP 200).
The **faithful agentica orchestrator is intrinsically M3-fragile on ft09**: the
PURE ORIGINAL (`A3_EXT=none`, nothing added) itself rarely reaches a clean terminal
— it M3-stalls (post-completion / mid-run quiescence) in ~most runs (faithful
re-pin historically 1/5 L6; pilots 8/9 soft_deadline; 2026-05-18 idea1 rerun: pure
-original ctrl did 78 actions over 74 min → L0, 0 completed episodes before stop).

## Why a rev4 clean WIN is structurally unreachable & more wall is NOT justified
rev4 requires ≥5 BOTH-arms-clean {normal,max_episode_budget} same-seed pairs. With
~90% M3 incidence on the control arm (independent of treatment), clean-pair yield is
structurally too low; burning the 3.5h wall → m3_unmeasurable/INCONCLUSIVE, not a
stronger answer. codex (3 independent reviews) confirmed: STOP; do not chase the
clean verdict; more matched-wall pairs (would need ≥3–5 more at 2600–12600s each)
not worth it under a structurally censored harness. The only true unblock is fixing
the agentica orchestrator-quiescence (M3) at the agent level — out of scope (the
agent is FROZEN/byte-faithful) → a separate plan→codex→ICR agent-level build.

## Disposition: RESOLVED to the evidentiary ceiling
- Defensible reportable result: the tiered finding above with the explicit estimand
  ("max level reached within wall-capped faithful ft09 runs"). Honest, not
  over-claimed.
- Speed: explicitly UNMEASURABLE on faithful-ft09 (M3 censoring).
- No further wall (codex). Faithful-ft09 is not a clean rev4 venue until M3 is
  fixed at the agent level (separate effort).

## Process corrections recorded this session (memories; do not repeat)
1. `feedback_authoritative_progress_signal` — never infer progress from incidental
   logs (server-log 0B); use the harness's authoritative per-episode summary.json
   `result.levels_completed`. (A multi-hour wrong-diagnosis detour came from this.)
2. `feedback_pgrep_into_kill_self_match` — `pgrep -f <pat>`→kill self-matches the
   tool shell (exit 144); read-only inventory → exact numeric PID only.
3. Never lower `TRAPI_PROXY_STREAM_IDLE_TIMEOUT_S` below the 600s default for speed
   — it kills legitimate long reasoning=high streams (self-inflicted retry storm).

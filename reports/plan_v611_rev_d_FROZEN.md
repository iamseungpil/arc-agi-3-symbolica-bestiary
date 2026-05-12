# Plan v611 rev C — Δ7 multi-role separation added

Codex 8-round adversarial review (rev A→B → REJECT/MAJOR/MINOR/ACCEPT
→ Symbolica gap analysis → Δ7 review → ADD_DELTA7).

rev B의 Δ1-Δ6은 유지. **Δ7 multi-role separation 추가** —
Symbolica의 orchestrator + 4-subagent 구조의 minimum architectural
ablation.

## Δ7 — multi-role mechanical separation (NEW)

**Motivation**: Symbolica upstream solves ft09 6/6 (recording evt 345
WIN). Codex Q1 ranking: top cause is (d) region centroid bug (fixed
by Δ1) + (a) multi-agent orchestration. Our v52/v608d is single-LLM
M1-M4 in same context. Δ1-Δ6 fixes execution bug but not architecture.

**Δ7 minimum architectural separation** (codex round 8 spec):

Per turn, the framework runs **3 separate API calls** with
**3 distinct system prompts** and **3 isolated contexts**:

### Δ7a — M1 Proposer (text-only)
- **Input**: frame TEXT summary (NO raw pixel array) + SKILL.md context
- **Output**: `nl_strategy` + `suggested_click_region` (NL only,
  e.g. "bottom-right corner tile") + `expected_signature` +
  `rollback_trigger`
- **Forbidden**: `click_xy_hint` MUST NOT be present in M1 output
  (M1 proposes intent, not coords)
- Symbolica analog: **theorist** (text only, no submit_action)

### Δ7b — M2v Verifier (separate context)
- **Input**: M1's `nl_strategy` + `suggested_click_region` ONLY (no
  frame, no SKILL.md context)
- **Output**: `verdict` ∈ {`approve`, `reject_replan`, `reject_anchor`} +
  `reason_nl`
- **Forbidden**: see raw frame, see prior turn transcripts, see other
  module outputs
- On `reject_anchor`: orchestrator spawns FRESH M1 next turn with just
  the rejection reason (no transcript carryover) — Symbolica analog:
  "spawn clean agent with summary"
- Symbolica analog: theorist-style independent critic

### Δ7c — M2e Executor (visual grounding)
- **Input**: M1's APPROVED `nl_strategy` + `suggested_click_region` +
  frame PNG (visual grounding)
- **Output**: `click_xy_hint` [x, y] + `grounding_text` (NL explaining
  which visual feature x/y maps to)
- Symbolica analog: **tester / solver** (submit_action with confirmed
  strategy)

### Mechanical enforcement
- 3 distinct API calls per turn (NOT 1 call with role labels)
- 3 distinct system prompts (separate files/templates)
- No shared chat history across roles within a turn
- Only structured JSON handoff fields between roles
- M2v sees no frame; M1 sees no coords; M2e sees no SKILL.md mechanics
- Orchestrator implements `reject_anchor` as a fresh-spawn signal

## Δ7 cost model

- ~1500 actions per episode × 3 LLM calls/turn = 4500 calls
- @ gpt-5.4-mini ≈ $0.0001/call = ~$0.45/episode
- Latency: ~2-3 seconds per turn × 1500 turns = 50-75 min/episode
- Acceptable for ablation (3 episodes = ~$1.5, ~4h)

## Updated success criterion

For v611+Δ7 to claim progress against the architecture-gap hypothesis:
- ≥2 of 3 episodes reach L+1 on disjoint held-out seeds 5/6/7
- + (mechanism claim) ≥1 confirmed_skill NL encodes mechanic abstractly

For v611 (Δ1-Δ6 only, no Δ7) as ablation control:
- Run on seeds 5/6/7 with SAME prompts as Δ7 but unified context
- Compare L+1 reach rate to Δ7

If Δ7 ≥ 2/3 AND no-Δ7 ≤ 1/3: architecture gap claim supported.

## A_Δ7 ablation (REPURPOSED from rev B)

Codex round-2 mandated A_Δ7 was "semantic-null prompt content control".
That ablation remains (different from Δ7 multi-role).

Renaming for clarity:
- **A_PROMPT_NULL** = original A_Δ7 (semantic-null prompt content)
- **A_MULTIROLE** = new control comparing Δ7 multi-role vs unified-context

Both run on held-out seeds.

## Honesty disclaimers (codex round 8)

- **NOT a Symbolica baseline**: We did not run Symbolica upstream on
  ft09 (PIL bug + missing .env). The ft09 recording (1 historical
  run, 6/6 WIN) is *evidence* but not a controlled baseline.
- **NOT parity with Symbolica**: Δ7 has 3 roles (proposer/verifier/
  executor) vs Symbolica's 4 (explorer/theorist/tester/solver) +
  orchestrator + memories DB. We claim minimum architectural
  ablation, not parity.
- **Mechanical separation is the load-bearing claim**: if Δ7 collapses
  to "3 prompts in shared context", the ablation is invalid.

## Implementation order

### Step 2a — Δ7 schema + fixtures
- New validators: `validate_m1_proposer_output`,
  `validate_m2v_verifier_output`, `validate_m2e_executor_output`
- Fixture suite for each (≥4 fixtures each, train/val 60/40)
- Include role-boundary tests:
  - M1 with `click_xy_hint` field → REJECT
  - M2v with `frame` or `skillmd` reference in input → REJECT
  - M2e missing `grounding_text` → REJECT

### Step 2b — agent.py 3-role wiring
- Sequential API calls: M1 → M2v → (if approve) M2e → env.step
- On reject_replan: re-run M1 with rejection reason added (still same
  episode)
- On reject_anchor: spawn fresh M1 next turn with NO transcript

### Step 2c — 5-turn smoke
- 1 episode, 5 turns. Verify:
  - 3 API calls per turn measured (15 total)
  - No raw frame in M1 prompt
  - No coords in M1 output
  - No frame in M2v input
  - M2e receives PNG
  - SKILL.md hygiene gate passes

### Step 3-5 — same as rev B
- A_PROMPT_NULL ablation
- Canary leak test
- Live cycle ft09 (3 episodes, sequential SKILL.md inheritance)

## Status

- rev A: REJECT (round 1)
- rev B: codex 2-7 round → ACCEPT (Δ1-Δ6, frozen 2026-05-12)
- rev C: Δ7 added (round 8 ADD_DELTA7)
- rev D: rev C + matched-ablation protocol + matched reuse-vs-fresh
  policy + pilot/confirmatory phase split + MIN_DELTA=+0.4 effect
  size + 12-seed × 95% CI confirmatory + honest framing
- **Round 10: ACCEPT** (final)
- **STATUS: FROZEN as plan v611 rev D, 2026-05-12**

## rev D — matched-ablation protocol additions

### Matched conditions (Δ7 vs no-Δ7 unified-context control)
- IDENTICAL token budget: 1500 actions × ~3K tokens/turn
- IDENTICAL tool set: only env.step + skill_state read/write
- IDENTICAL stopping: 1500 actions OR L+6 OR explicit fail signal
- SAME JSON schema validators; failure → retry once, second fail
  → log + skip turn (no silent fallback)
- NO inference caching across episodes (fresh API session per episode)

### Matched reuse-vs-fresh policy
- Δ7 condition: M2v verifier `reject_anchor` → spawn fresh M1 with
  summary-only carry-over (no transcript)
- no-Δ7 control: when prediction failures ≥ 3 in rolling 30 turns →
  unified-context's prior reasoning summary replaces transcript
- Both conditions: anchor-threshold N=3, summary-only carryover

### Pilot vs confirmatory split (codex round 9-10 mandated)
- **Pilot phase (3 seeds 5/6/7) = EXPLORATORY ONLY**
- No hypothesis-support claim from pilot
- If pilot negative (Δ7 ≤ 1/3): conclude only 'pilot does not
  support Δ7 advantage'. Do NOT conclude absence of effect.
- **Confirmatory phase IF pilot positive (Δ7 ≥ 2/3 AND no-Δ7 = 0/3)**:
  - 12 seeds per condition (24 total runs)
  - Paired comparison
  - Pre-registered alpha = 0.05
  - **MIN_DELTA = +0.4 effect size** (40+ pp improvement needed for
    architecture-gap claim)
  - 95% CI on success-rate difference must exclude 0
  - If observed delta < 0.4 even with significant CI: claim 'small
    Δ7 effect' but NOT 'closes Symbolica gap'

### Honest scope of any claim
- Pilot alone: 'preliminary indication, not evidence'
- Confirmatory positive: 'evidence for Δ7 architectural separation
  effect on ft09'
- Cross-game (ls20, vc33): SEPARATE confirmatory needed
- Symbolica parity: NEVER claimed (we did not run baseline)

### Codex round 10 caveat (must include in writeup)
Because confirmatory testing is gated on positive pilot, the final
writeup must NOT present the full pipeline as an unconditional
estimate of effect size or success probability. Gating must be
disclosed.

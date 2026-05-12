# Plan v611 rev B — LLM-driven Skill Discovery (codex 6-round converged)

Codex adversarial reviewer 5 rounds 통과 후 작성된 rev B.
rev A의 모든 over-claim retract, 모든 must-fix 반영.

## 단일 진실 원천 (intent)

> Leakage 없이, 가설을 세우고, 그 다음 거기서 strategy를 자연어로
> 뽑아서 문제를 풀 수 있을 때까지 SKILL.md를 개선한다.

## Final framing (codex-approved)

> v611 is a context-conditioned candidate selection framework with
> **NL-grounded action emission**, **dual-scale memory with
> leak-audited episode inheritance**, and **pre-registered ablation
> protocol**.

NOT: "PSRL", "skill discovery proven", "minimal change", "guaranteed".
ONLY: "plausible intervention bundle that may enable ft09 L+1 under
the leak-audited protocol."

## Survey 종합 design principles (intent md §종합 시사)

1. NL-first reasoning (LLM speaks NL strategy before coord)
2. Context-conditioned candidate selection (NOT PSRL — no explicit
   belief variable defined)
3. Voyager-style skill library — **NL ONLY**, no executable code
4. Dual-scale SKILL.md (global confirmed + local trials)
5. Iterative self-verification (M4 3-step)
6. No master trace (leak-audited canary check)

## Δ — v608d → v611 변경 (codex-converged set)

### Δ1 — NL strategy first (M1)

M1 output now requires `nl_strategy` field FIRST, with `click_xy_hint`
GROUNDED in NL (not region centroid fallback):

```json
{
  "nl_strategy": "<visual description from PNG + intent + grounded click area>",
  "predicate_id": "P_NL_grounded",
  "click_xy_hint": [x, y],
  "expected_signature": {"frame_changed": bool, "unsat_delta": int},
  "rollback_trigger": "<NL condition>"
}
```

Region centroid auto-fallback REMOVED from `policy.resolve_coord`.

### Δ2 — Context-conditioned candidate selection (M2)

M2 receives M1's NL candidates + SKILL.md context, returns 1 choice.
NO claim of PSRL or posterior sampling. Pure description.

### Δ3 — NL-only skill compressor (M3, REVISED from rev A)

REVISED: NO executable Python code (rev A overclaim retracted).
M3 emits ONLY NL description + abstract precondition.

```json
{
  "skill_id": "S-NL-<hash>",
  "nl_description": "Clicking near a corner of the same color as
                     a marker's neighbor triggers a satisfaction change.",
  "abstract_precondition": "<NL condition>",
  "expected_observed_effect": "<NL effect>"
}
```

NO integer coords in `nl_description`. NO region IDs.
Leak-hygiene validator runs before save:
- Banned: `\\b\\d{1,2}\\b` (raw coord patterns), `C\\d+`, `R\\d+`,
  `L[1-6]`, ft09 vocab (bsT, gqb, Hkx, NTi, kCv, cwU, elp, Ycb)

### Δ4 — Dual-scale SKILL.md (GLoW pattern)

- **Global** `confirmed_skills`: NL only, no coords, no region IDs.
  Survives episode boundaries (with hygiene check).
- **Local** `recent_trials` (rolling 30): may contain coords for
  traceability, auto-purged on episode boundary.
- **Cross-run inheritance**: ONLY `confirmed_skills`, NEVER
  `recent_trials`. Validator gates.

### Δ5 — M4 3-step self-verification

M4 paragraph now structured:
1. **Predicted vs observed**: did `expected_signature` match env step?
2. **NL strategy validity**: was the NL hypothesis falsified?
3. **SKILL.md update**: explicit `{add: [...], promote: [...],
   falsify: [...]}` patch

### Δ6 — PNG visual grounding (M1+M2)

M1/M2 prompt forces:
- "Look at the PNG. Describe what you see in 2-3 sentences."
- "Identify connected regions of same color and their approximate
  pixel coordinates."
- "For each marker, estimate its 8 neighbors' pixel locations from
  the image."
- NL strategy MUST include 1+ visual reference before coord.

## Ablation protocol (codex-converged)

### A_Δ1 — NL-first vs centroid-only
Fixture: 10 ft09 root states + 5 turn-2 states.
Variants: (i) v608d region_hint-only, (ii) Δ1 nl_strategy + click_xy_hint.
Pass: click distribution shifts >40% away from region centroid AND
unique clicks >2x.

### A_Δ3 — NL-only compressor vs old NL hint
Fixture: 5-turn trace.
Pass: ≥1 leak-validated NL precondition that is cycle-runnable next turn.

### A_Δ5 — 3-step verify vs single-step
Fixture: turns where prediction failed.
Pass: Δ5 attribution accuracy >70%.

### A_Δ6 — PNG vs text-only
Fixture: 5 ft09 frames.
Pass: visual-concept mentions >2x.

### A_Δ7 — Semantic-null control (codex round 2 mandated)
Same JSON schema, same token budget, but `nl_strategy` replaced with
neutral paraphrase ("I am exploring the grid systematically. The
output should be a valid ACTION6 click."). Run on identical fixtures
as A_Δ1/A_Δ3 on held-out seeds 5/6/7.

**Pre-registered acceptance**: real Δ1 must beat A_Δ7 on
click-distribution shift / frame-effect rate / marker-state change
rate by margin >0.2 on held-out seeds.

If real ≤ A_Δ7: conclude prompt cosmetic, reject Δ1.

## Memory hygiene (codex round 2-3 mandated)

### SKILL.md no-leak rules
- `confirmed_skills`: NO `\\d` patterns, NO `[CR]\\d+`, NO `L[1-6]`,
  NO ft09 vocab
- Cross-episode: ONLY confirmed_skills inherited
- Validator runs BEFORE episode 2 begins; if FAIL, episode 2 starts
  with EMPTY SKILL.md

### Canary leak test (codex round 3-5 mandated)

**Setup**:
- Episode 1: inject synthetic canary card in SKILL.md:
  "Quadrant 4 always wraps to quadrant 0 after 2-step rotation."
  This is FALSE about ft09.
- Episode 2: run 100-turn evaluation with that SKILL.md inherited.

**Artifact sources scanned**:
- agent.log filtered: role=M1/M2/M3/M4 OR llm_response_text fields
  (excluding INFO/DEBUG headers + token-credential lines)
- skill_state.json all string fields (cards.claim, predicts,
  falsifiers, state_features, evidence[].notes)
- m1_emission_log.jsonl: `thought`, `nl_strategy`
- m4_history.jsonl: `paragraph`, `next_directive`

**Excluded** (boilerplate, not leak surface):
- 'ACTION6:', 'model=', 'PROPOSER_RAW', 'INFO', 'DEBUG'
- Azure/credential strings
- Validator's own canary-phrase list output

**Canonicalization pipeline** (shared canon() function applied to BOTH
artifact text AND canary phrase library):
```python
def canon(text: str) -> list[str]:
    s = text.lower()
    s = re.sub(r'[^a-z0-9\\s]', ' ', s)
    s = re.sub(r'\\s+', ' ', s).strip()
    return s.split()
```

**Phrase library** (compiled to n-gram sets via canon):
- unigrams: {'quadrant', 'rotate', 'rotation', 'wrap', 'sector', 'quad'}
- bigrams: {('2','step'), ('two','step'), ('wrap','around'),
            ('sector','rotate'), ('quadrant','cycle')}
- trigrams: {('q4','wraps','to'), ('two','step','rotation')}

**Per-source independent scan** (Round 7 final spec):
```python
for src in artifact_sources:
    tokens = canon(src.text)
    uni = set(tokens)
    bi = set(zip(tokens, tokens[1:]))
    tri = set(zip(tokens, tokens[1:], tokens[2:]))
    hits = ((uni & PHRASE_UNIGRAMS)
            | (bi & PHRASE_BIGRAMS)
            | (tri & PHRASE_TRIGRAMS))
    if hits: per_source_violations.append({src.name: hits})

scanner_hits_real = sum(len(v['hits']) for v in per_source_violations)
```

Cross-source n-grams (last token of agent.log + first token of
skill_state.json) are NEVER formed. Prevents accidental boundary
leakage.

**Match rule**: hit iff any unigram in artifact_unigrams OR any bigram
in artifact_bigrams OR any trigram in artifact_trigrams **within a
single source**.

**Baseline noise**: run scanner on 100-turn `no-canary` episode-2
(identical setup, SKILL.md neutral string). Measure background hit
count B.

**Pass**: scanner_hits_real ≤ B (in practice B expected 0 — canary
words not normal ARC-AGI vocab). Plan freezes B as measured value.

**Fail**: cross-episode contamination confirmed; episode inheritance
must be removed or revised.

## Success criterion (codex round 3-4 locked)

### L+1 progress (minimum acceptable)
- ≥2 of 3 episodes reach L+1 on DISJOINT held-out seeds 5/6/7
- DIFFERENT trajectories (not identical click sequences)
- Pre-registered noise baseline: random-baseline reaches L+1 in
  <1/10 episodes (already measured: cycle405 0/1 on 700+ actions)

### Mechanism claim (additional bar — only if reached)
- ≥1 confirmed_skill NL description encodes ft09 mechanism abstractly
- No coords in the NL
- Replay-verified on held-out seed by independent runner

## What v611 does NOT claim (codex retractions)

- ❌ "minimal change" — only "candidate intervention bundle"
- ❌ "PSRL" or "posterior sampling proof"
- ❌ "skill discovery proven" — only "may enable L+1 under protocol"
- ❌ "generalizes to ls20/vc33" — that's a separate experiment
- ❌ "zero leakage guaranteed" — only "leak-audited"

## Implementation plan (Voyager-style iterative)

### Step 1 — Module fixtures (Δ1-Δ6 unit tests)
4-6 fixtures per module. M1, M2, M3, M4 each test the new field /
schema / behavior. Train≥80%, val≥75%, gap≤15pt.

### Step 2 — Smoke 5-turn cycle
1 episode, 5 turns, verify:
- SKILL.md hygiene validator runs and passes (no leak in fresh state)
- M1 NL strategy field non-empty, mentions visual concept
- M4 3-step verification fields present
- M3 NL-only skill emitted (no code)

### Step 3 — A_Δ7 semantic-null ablation
Run on held-out seeds. Pre-register margins. Measure real vs control.

### Step 4 — Canary leak test
Synthetic canary in SKILL.md, episode 2 scan. Must pass before
proceeding.

### Step 5 — Live cycle ft09 (3 episodes, held-out seeds 5/6/7)
Sequential. Episode N+1 inherits ONLY confirmed_skills (hygiene
validated).

### Step 6 — Verdict
- ≥2/3 episodes L+1 different trajectories → progress claim
- ≥1 mechanism-encoding confirmed_skill → mechanism claim
- Canary 0 → leak claim
- All else → null result, document, iterate

## Open items (post-freeze monitoring)

- v605_arm7 multimodal arm 결과는 0 L+1 — vision grounding 한계
  의심. Δ6 PNG prompt가 이를 극복할 수 있을지 검증 필요.
- ls20/vc33 generalization은 별도 experiment.

## Codex review history

- Round 1: REJECT — coded substrate (Δ3 executable), SKILL.md leak risk
- Round 2: MAJOR_REVISIONS — ablations + memory hygiene + PSRL rename
- Round 3: MAJOR_REVISIONS — canary <5% too lenient, "minimal" overclaim
- Round 4: MINOR_REVISIONS — scanner spec not precise enough
- Round 5: MINOR_REVISIONS — phrase library canonicalization
- Round 6: MINOR_REVISIONS — independent vs concatenated scan
- **Round 7: ACCEPT** — per-source independent scan + B baseline gate

**STATUS: FROZEN as of round 7 verdict (2026-05-12)**

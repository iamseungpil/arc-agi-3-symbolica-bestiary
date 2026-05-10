# Plan v607 — Skill Discovery Revival (post-v606.x diagnosis)

Date: 2026-05-10 (late session)
Status: **APPROVED rev E (FROZEN) — codex r6 STOP verdict 2026-05-10**

Convergence path: r1 draft → r2 NEEDS_WORK → r3 NEEDS_WORK → r4 NEEDS_WORK → r5 NEEDS_WORK → r6 STOP (6 rounds)

## Codex review log
- Round 1 (assistant draft): rev A
- Round 2 (codex skeptic): NEEDS_WORK — 3 revisions applied to rev B
  - R1: structural alternative comparison required (Voyager/DreamCoder)
  - R2: D0 = utility gates not diversity
  - R3: Reflector stagnation-triggered + cooldown
- Round 3 (codex skeptic): NEEDS_WORK — 3 more revisions applied to rev C
  - R1: Arm-D hidden costs not addressed (sandbox AST validation, LLM code-gen latency, AST-failure fallback to B)
  - R2: U1/U2/U3 utility gates miss — (a) task success/reward impact, (b) novelty vs genuine utility, (c) cost-adjusted utility, (d) robustness across episodes, (e) attribution leakage from evaluator, (f) long-horizon degradation
  - R3: Reflector anti-thrashing — `best_advance_age >= k` OR minimum utility delta OR cap on total invocations per episode
- Round 4 (codex skeptic): NEEDS_WORK — 3 more revisions applied to rev D
  - R1: Cost normalization mixes per-skill numerator with episode-level denominator. Use marginal token cost attributed to fired episodes
  - R2: Jaccard 0.3 hard gate may false-reject useful overlapping skills. Make soft penalty; keep anti-leak hard
  - R3: Reflector cap=6 too rigid; should be adaptive (episode length, stagnation, budget)
- Round 5 (codex skeptic): NEEDS_WORK — 3 more revisions applied to rev E
  - R1: marginal token attribution not specific enough; need explicit shared-token rule + L_LLM > L_compute tie-break
  - R2: anti-leak HARD regex bypassable by paraphrase; unseen region_hint needs explicit allowed-set
  - R3: adaptive cap formula needs explicit units, rounding, stagnation_sev bounds
- Round 6 (codex skeptic): **STOP** — all three concerns explicit, no further iteration warranted. Plan FROZEN for implementation.
Predecessors:
- `plan_v600_minimal_fast_2026_05_09.md` (minimal-fast refactor, dropped TIER-B invention)
- `plan_v601_3role_saturation_bundle_2026_05_10.md` (Proposer/Policy/Memory, saturation framing)
- `plan_v602_skillmd_ssot_2026_05_10.md` (SKILL.md SSOT for hypotheses + skills)
Authors: assistant (this session) — codex review pending

User intent verbatim (from `project_skilldiscovery_intent.md`, restored after v606.x drift):
> "스킬을 반복적으로 만들고, 개선하고, 축적하면 LLM이 문제를 풀 수 있게 되는가?"

---

## 0. Empirical state at planning time (verified today, 20+ codex rounds)

**Falsified hypothesis**: "rolling history + TIER-B unlock + stable IDs alone → L+1"

| Version | Fix | Verified live? | L+1? |
|---|---|---|---|
| v606 | TIER-B branch split in Proposer prompt | 0/50 TIER-B (R-prefix mismatch) | ❌ |
| v606.1 | prefix-agnostic dynamic example injection | 31/31 TIER-B with C-prefix | ❌ |
| v606.2 | Sig-B cross-turn region-ID matching (IoU≥0.3) | stable IDs across turns | ❌ |
| v606.3 | rolling click history in prompt | LLM reads + reasons about history | ❌ |
| v606.4 | smallest-region click match + bg filter | meaningful region_id reporting | ❌ |

LLM at v606.4 turn 41 (verbatim from agent.log):
> "The recent history shows repeated attempts around the C9–C15 band with no level_delta advance, so I should try a different region this turn"

→ LLM **reasoning is healthy**. The bug is not prompt engineering.

cycle237 (1/270 reproducibility, ARC_NO_GOAL_LEAK=1, gpt-5.5):
- 30+ distinct chid types (P01-P12 TIER-A library + TIER-B P_R*_*, H_*_*)
- L+1 at T6 via TIER-B `P_crop_compass_sweep_R31` coord [38,54]
- L+2 at T27 via TIER-B `P_R12_crop_sector_alignment` coord [38,48]
- stuck_mode escalation False→sev5
- Reflexor active synthesizing lessons

v606.4 (60 turns):
- 1 chid TEMPLATE: `P_C{marker_id}_crop_sector_alignment` (marker substitution only)
- 4 distinct markers, 10 distinct region_hints
- NO Bayesian posterior tracking
- NO stuck escalation (advisory only)
- NO Reflexor active

**Empirical conclusion**: v606.x architecturally precludes skill DISCOVERY — single chid template, no posterior, no skill mutation. Misaligned with project core intent.

---

## 1. Intent (I/H/V format)

### Intent

Rebuild Proposer so the agent **discovers, accumulates, and improves multiple skill families** across episodes (per `project_skilldiscovery_intent.md`), instead of monomania on `P_C{m}_crop_sector_alignment`. The framework must produce skill TRAJECTORIES that show width (≥5 distinct chid families) and accumulation (cross-episode SKILL.md SSOT growth), without leaking ft09-specific knowledge.

Three-stage falsifiable goals (binomial LCL gates):
- **D0 (skill discovery)**: ≥5 distinct chid families emitted across 1 episode × 30 turns (LCL trivial — population threshold)
- **D1 (L+1 reach)**: ≥1 L+1 event in ≤30 turns × 5 episodes = ≥1/5 (LCL 0.005, beats cycle237's 1/270 baseline)
- **D2 (L+2 reach)**: ≥1 L+2 event in ≤50 turns × 5 episodes (LCL same as D1, harder)

**No D2 without D1.** No live autoresearch until D0 fixture suite passes.

---

## 2. Survey / theory grounding (literature search summary)

Three relevant lineages from skill-discovery + hypothesis-search literature:

### 2.1. DreamCoder (Ellis et al., 2020) — library learning loop
- **Wake**: tasks → programs
- **Sleep abstraction**: prune library, compress patterns into new primitives
- **Sleep dreaming**: solve hypothetical tasks with new primitives
- Key insight: **library width grows by compression of solved trajectories** (NOT by prompt engineering)

### 2.2. Voyager (Wang et al., 2023) — open-ended skill library
- Iterative code refinement based on env feedback
- Skill library = **GPT-written code with retrieval**
- Trajectory: env error → automatic curriculum → skill library augmentation

### 2.3. Reflexion (Shinn et al., 2023) — verbal reinforcement
- Verbal self-reflection synthesized into "lessons" stored in episodic memory
- LLM reads past lessons → policy update without weight modification
- v601-v605 has Reflector module but **INACTIVE** at runtime

### 2.4. Hypothesis search (HypGenic, Wang et al., 2024)
- Generate ≥k hypotheses per turn (not 1)
- Bayesian posterior over hypotheses based on observation evidence
- Top-k retained, bottom pruned, new spawned

### 2.5. ARC-specific prior art (within this project)
- `plan_v53_react_reflexion_lifecycle.md`: 4-module ReAct + Reflexion + SKILL.md
- `plan_v590_predicate_induction_2026_05_07.md`: predicate library with TIER-A/B
- `plan_v591_two_tier_invention_2026_05_08.md`: M1 invention restoration
- `plan_v600_minimal_fast_2026_05_09.md`: minimum framework after sprawl (dropped TIER-B — root cause of v601-v605 monomania)

**Theory-grounded diagnosis**: v600/v601 traded SKILL DIVERSITY for SPEED. The single P_saturation_progress predicate was supposed to be sufficient if Bayesian posterior found the right region — but posterior was never wired to active runtime (skill_state.json has it as inert spec). cycle237's success was a TIER-B invention not in the library; v600 explicitly removed this invention path for "minimal-fast" determinism.

---

## 3. Approach options (4 DISTINCT proposals, theory-tagged)

### Option A — **DreamCoder-style library learning** (compress trajectories into primitives)

**Mechanism**: After every 5 turns, a Sleep call extracts patterns from action_history. If ≥3 consecutive clicks on similar (region, coord_policy) tuples produced same observation, propose a new compound primitive `P_<rationale>_<region>`. Add to skill_state.json. Subsequent Proposer reads this expanded library.

**Theoretical basis**: DreamCoder library learning loop. Skill width grows from EVIDENCE, not from prompt engineering.

**vs v606.x**: v606.4 had ONE template + LLM marker substitution. Option A adds a Sleep-time COMPRESSION step that builds NEW chid families from observed patterns.

**Falsifiable**: D0 = after 1 Sleep cycle (5 turns), skill_state has ≥2 new primitives. D1 = after 3 Sleep cycles, Proposer emits ≥3 distinct families (verifiable in PROPOSER_RAW grep).

**Complexity**: Medium (Sleep module + library mutation API + SKILL.md re-render)

**Risk**: Sleep compression may extract spurious patterns. Mitigation: require ≥3 confirmations + Beta posterior gate.

---

### Option B — **Reflexion-style verbal skill mutation** (LLM proposes new chid based on lessons)

**Mechanism** (rev D, codex r2+r3+r4 R3 adaptive cap): Reflector activates on STUCK signal — ALL conditions:
1. `turns_since_advance > 3` (stagnation)
2. `cooldown_remaining == 0` (post-fire wait)
3. `reflector_fires_this_episode < ADAPTIVE_CAP` — **adaptive cap formula (rev E, codex r5 R3 explicit numerics)**:
   - `stagnation_severity = min(10, turns_since_advance - 3)`  (bounded int, 0-10)
   - `ADAPTIVE_CAP = int(min(0.4 * episode_length, max(3, stagnation_severity * 2)))`  (ceil-floor convention: floor)
   - For 30-turn ep: max cap 12. Stagnation_sev=5 → cap min 10. Stagnation_sev=10 → cap 12 (capped by 0.4×30=12).
   - For 50-turn ep: max cap 20. Stagnation_sev=10 → cap 20.
   - Units: integer turns. No fractional ticks.
4. `best_advance_age >= k` where k = 5 (genuinely stuck, not transient)
5. **OR** episode delivered ≥1 new dominant_transition since last fire (positive evidence trigger)

After fire, cooldown=5 turns AND fires_this_episode++. NOT periodic. Reads last 5 turn diffs + active hypotheses, calls LLM with prompt: "What new chid family should we ADD to test next? Output {new_chid_template, rationale}". Appended to skill_state.json with confirmed=0/refuted=0 priors. Proposer next turn can pick from expanded library.

**Theoretical basis**: Reflexion verbal RL. Skill creation by LLM verbal self-reflection.

**vs v606.x**: v606.4 LLM ONLY responds to current state; it does not REFLECT and CREATE new skill templates. Option B activates Reflector for skill emission, not just observation summary.

**Falsifiable**: D0 = Reflector produces ≥1 new chid_template after turn 5. D1 = after 4 Reflector calls (turns 5/10/15/20), skill_state has ≥4 distinct chid_templates. PROPOSER_RAW shows mix of templates.

**Complexity**: Small (Reflector already exists, just needs activation + new prompt + skill_state write)

**Risk**: LLM hallucinates non-useful chid templates. Mitigation: Beta posterior prunes failed ones after k attempts.

---

### Option C — **Hypothesis-search posterior + top-k sampling** (Bayesian skill ranking)

**Mechanism**: Maintain Beta(α=1, β=1) per chid in skill_state. Each turn, Policy samples top-3 chids by posterior expected value. LLM is given the TOP-K (not single best), proposes which to TRY. After observation, update posterior. New chids spawned from Reflector (B) get prior Beta(0.5, 0.5).

**Theoretical basis**: HypGenic + Thompson sampling. Multi-armed bandit over skills.

**vs v606.x**: v606.4 has NO posterior; LLM picks chid freely per prompt. Option C constrains LLM to a posterior-ranked top-k, breaking monomania.

**Falsifiable**: D0 = after 10 turns with ≥3 skills in library, sampling spreads emissions across ≥3 distinct chids. D1 = posterior of highest chid > 0.3 after 20 turns.

**Complexity**: Medium (Beta posterior store + sampling + Policy.run_turn integration)

**Risk**: Posterior priors too weak to differentiate. Mitigation: warmup with uniform sampling.

---

### Option D — **Voyager-style code-skill primitives** (LLM writes skill as code)

**Mechanism**: Reflector emits a small Python lambda for each new chid: `def skill_X(state) -> coord_policy_hint`. Predicate library installs via sandbox (existing predicate_library.py:install_skill). LLM's predict-effect logic uses the code, not the chid name.

**Theoretical basis**: Voyager skill-as-code library.

**vs v606.x**: v606.4 chids are STRINGS only; click resolution always falls to centroid. Option D makes chids EXECUTABLE Python that compute non-centroid coords.

**Falsifiable**: D0 = 1 successful code-skill install after Reflector r1. D1 = code-skill emits coord ≠ centroid in ≥1 turn.

**Complexity**: Large (sandbox already exists, but full skill-as-code requires Reflector emission + AST validation + integration tests)

**Risk**: Skill code may fail safety (sandbox already gates). Cost of LLM calls 2x (skill writing + skill picking).

---

## 4. Recommended path — rev B (codex r2 revisions applied)

**Decision after codex r2 NEEDS_WORK**: replace B+C as monolith. Run **B+D as a 2-arm structural comparison**:

### Arm 1: **B+C (verbal mutation + Bayesian)**
- B: Reflector emits chid_template via LLM verbal reflection (no code)
- C: Beta posterior per chid; top-3 sampling
- Pros: small/medium complexity. Cons: surface-form mutation only (codex r2 concern)

### Arm 2: **D+C (skill-as-code + Bayesian)**
- D: Reflector emits Python lambda for new chid (executable, persistent, transferable)
- C: same Beta posterior over emitted code-skills
- Pros: structural skill representation (codex r2 demand); skills are reusable artifacts
- Cons: large complexity; sandbox AST validation cost

### Comparison protocol (rev D, codex r3+r4 R1 marginal cost attribution)

Cost-normalized comparison at **per-skill granularity** (codex r4 R1 fix):

For each generated skill `s_i` (rev E, codex r5 R1 explicit attribution):
- **L_LLM(s_i)** = sum of (single Reflector emit-call tokens for s_i = ~800) + (per-Proposer-call tokens × number_of_emissions_of_s_i = ~600 × N_emit(s_i))
- **L_compute(s_i)** = AST validation + sandbox time (Arm D only, capped 5s = ~5000ms units)
- Shared-token rule: Proposer prompt tokens are amortized across ALL emitted skills that turn (divide equally); Reflector emit tokens attributed solely to s_i.
- Tie-break when L_LLM >> L_compute: efficiency uses L_LLM-only; L_compute reported as auxiliary.
- **utility(s_i)** = (Beta posterior EV at episode end) − 0.5 (Jeffreys prior baseline). Window = full episode.
- **efficiency(s_i)** = utility(s_i) / L_LLM(s_i)

Arm ranking: median efficiency over emitted skills (not aggregate per-episode rate). This avoids the per-skill vs episode-level granularity mismatch codex r4 flagged.

D0 fixture: BOTH arms must pass utility gates (see §5.1 revised).
D1 live: BOTH arms run 3 episodes × 30 turns; advance the arm with higher **median efficiency** + ≥1 L+1.
This is STRUCTURAL ALTERNATIVE + MARGINAL-COST-NORMALIZED comparison per codex r2-r4.

**Anti-recommendation**: Do NOT continue v606.x patches (history injection is sufficient; the bottleneck is skill REPRESENTATION + UTILITY).

---

## 5. Verification protocol (D0/D1/D2 gates)

### 5.1. D0 (skill UTILITY) fixture suite — rev B per codex r2

Codex r2 R2 + r3 R2: D0 must measure UTILITY not diversity, AND must avoid the gap categories codex r3 enumerated.

**Gate U1 — held-out reuse**: Take a discovered skill from training episode T_train, freeze it. In separate test episode T_test (different scorecard, same game), check the SAME skill is selected by top-k posterior AND its emission cite the same family. PASS if ≥1 of ≥3 discovered skills reuse.

**Gate U2 — cross-episode improvement**: 3 episodes of N=20 turns each. Episode 2 cumulative_distinct_chids should ≥ episode 1's + ≥1 new entry. Episode 3 cumulative same. PASS if growth monotonic. (Tests that mutations persist via SKILL.md SSOT.)

**Gate U3 — downstream tie**: For each emitted skill, the empirical success rate (Beta posterior expected value at episode end) must correlate with skill usage frequency — high-EV skills used ≥2× as often as low-EV skills. PASS if Spearman ρ > 0.4 over ≥5 skills.

**Gate U4 (rev C, codex r3 R2 (a))** — task success / reward impact: discovered skills MUST measurably improve a downstream metric. Specifically: average `dominant_transition.count` per emission for skills with EV > 0.4 must exceed baseline `P_C{m}_crop_sector_alignment` count by ≥1.5×. (Tests that skill emission produces more state change, not just different names.)

**Gate U5 (rev E, codex r5 R2 explicit sets)** — novelty + anti-leak:
- **HARD (gating)** — anti-leak with explicit allowed-set:
  - V-LEAK regex (forbidden token list, from `scripts/check_no_leak_prompts.py`): R28, R31, R12, gqb, bsT, Hkx, NTi, kCv, cwU, elp, Ycb (ft09-specific identifiers extracted by existing leak checker)
  - cycle237 literal block: full ban on any 4-char substring from cycle237 trace's chid strings (`P_crop_compass_sweep_R31`, `P_R12_crop_sector_alignment`, etc.)
  - **region_hint allowed-set**: must be member of CURRENT episode's `state.visible_region_ids` (concretely enforced in proposer schema validator at runtime, NOT at Reflector emit time — Reflector emits chid TEMPLATES that get instantiated with current region_id at proposer time)
  - Paraphrase bypass mitigation: combine regex with embedding similarity check vs forbidden corpus (cosine > 0.8 → reject). Implementation: TF-IDF on chid_template strings vs forbidden_corpus
- **SOFT (penalty)** — novelty Jaccard < 0.3 → utility score -0.1. Allows useful overlapping skills.

**Gate U6 (rev C, codex r3 R2 (c) + (d))** — cost-adjusted + robustness: utility (L+1 rate) per Reflector LLM token must beat v606.4 baseline (which spent ZERO Reflector tokens for 0 L+1 → infinite cost ratio). Specifically: arm must hit ≥1 L+1 in ≤500 total Reflector tokens × 3 episodes.

Module-level fixtures (per `feedback_module_fixture_first.md`):

- **F-Reflector-stuck**: triggered ONLY when turns_since_advance > 3, NOT every 5 turns (codex r2 R3)
- **F-Reflector-cooldown**: after fire, cooldown of 5 turns prevents re-fire
- **F-Reflector-new-chid**: emits chid_template absent from current skill_state
- **F-Posterior-rank**: top-k samples correctly per Beta(α,β) ordering
- **F-Posterior-warmup**: uniform sampling during turns 1-5 (priors weak)
- **F-Integration-stuck-cycle**: 8-turn slice — turns 1-3 no advance, turn 4 stuck triggers Reflector, turn 5+ Proposer emits new skill from posterior

Train ≥80% pass / Val ≥75% pass / gap ≤15pt = D0 gate per existing methodology.

### 5.2. D1 (L+1 reach) live gate

5 episodes × 30 turns, ARC_NO_GOAL_LEAK=1. Success = ≥1 L+1 event total. Each episode logs:
- PROPOSER_RAW count + distinct chid templates
- Skill mutations per episode
- Beta posterior trajectories
- L+ events with chid attribution

If 0/5 → diagnose via codex r1 (NOT immediately patch — falsifies B+C jointly).

### 5.3. D2 (L+2 reach) gate

After D1 hits, repeat at 50 turns × 5 episodes. Success = ≥1 L+2 event.

---

## 6. Risks + abandonment criteria

- **R1**: Reflector LLM hallucinates dummy chids (low signal). Mitigation: require chid_template format `P_<verb>_<noun>_<region_id>`.
- **R2**: Posterior priors converge too slowly with N=30 turns. Mitigation: Beta(0.5, 0.5) Jeffreys prior + warmup with uniform.
- **R3**: D1 0/5 → falsifies the entire "skill discovery → L+1" thesis. Pivot to Option A or D.
- **R4**: TRAPI codex API instability. Mitigation: continue with self-loop, save final plan for codex review when API recovers.

**Abandonment criterion**: If 5 fully-tested D1 runs across A/B/C combinations show 0/25 L+1, the lite framework family is fundamentally inadequate for ft09 → pivot to (1) restoring full v57 sprite-aware extraction, or (2) ls20 validation.

---

## 7. Codex review-counter loop status

**Round 1 (assistant draft, this document)**: rev A
**Round 2 (codex objective reviewer)**: PENDING — API instability blocking, will issue when recovered

**Plan-critic checklist (to verify after codex round 2)**:
- [ ] All hypotheses in 3.A-D have ≥1 falsifiable predictor
- [ ] No game-specific leakage (no ft09 string in fixture mock data)
- [ ] Module fixtures all train/val/gap measurable
- [ ] D0 gate condition explicit (≥5 distinct chid families)
- [ ] D1 binomial LCL > cycle237 baseline (0.005)

---

## 8. Out of scope (deferred)

- multi-step action sequences (Plan ≥2 clicks ahead) — v608 if D1 passes
- vision/multimodal grid input — orthogonal
- ls20 generalization — after ft09 D2 hit
- BFS coord search — anti-discovery, defers to deterministic baseline path
- model swap to gpt-5.4-pro / gpt-5.5 — orthogonal, can be combined with B+C

# Plan v611 rev A — LLM-driven Skill Discovery (survey-grounded)

본 plan은 `reports/skill_discovery_intent_v611.md`의 의도를 구현하기 위한
v611 framework 설계 초안. **codex review-counter loop를 통과해야 freeze**.

## 단일 진실 원천 (intent)

> Leakage 없이, 가설을 세우고, 그 다음 거기서 strategy를 자연어로
> 뽑아서 문제를 풀 수 있을 때까지 SKILL.md를 개선한다.

## Survey 종합 design principles (intent md §종합 시사)

1. **NL-first reasoning**: M1/M2가 coord 발언 전에 자연어 strategy 발화
2. **Posterior Sampling exploration**: 무작위/brute-force 금지
3. **Voyager-style skill library**: executable + NL description
4. **Dual-scale SKILL.md**: global confirmed + local trials
5. **Iterative self-verification**: M4가 verify-evidence 강하게
6. **No master trace**: leak-free 자력 발견

## v608d → v611 변경 사항

### Δ1 — NL strategy 단계 추가 (M1)

**현재 (v608d)**:
```json
{"candidate_predicate_id": "P_C4_probe_edge",
 "region_hint": "C5",
 "expected_signature": {"level_delta": 1},
 "thought": "..."}
```
M1이 predicate + region 한 번에 출력.

**v611**:
```json
{"nl_strategy": "Looking at the PNG, the marker at C4 has its East
neighbor C5 unsatisfied. To repair I should TRY clicking near
coordinates (40-50, 40-50) which is C5's bounding region. If frame
changes and C4 satisfaction count drops, the strategy works.",
 "predicate_id": "P_NL_grounded",
 "click_xy_hint": [44, 44],
 "expected_signature": {"frame_changed": true,
                          "unsat_delta": "<0"},
 "rollback_trigger": "frame unchanged after 3 attempts"}
```
M1이 자연어 strategy를 먼저 발화 → 거기서 click_xy_hint를 grounding.
**중요**: click_xy는 LLM의 NL reasoning에서 grounded (region centroid
fallback 없음).

### Δ2 — Posterior Sampling M2

**현재 (v608d)**: deterministic UCB1 over predicate-region pairs.

**v611**: M2가 M1의 후보들을 보고 *posterior belief* update + sample.
- 각 후보 (NL strategy)는 prior posterior (cards에서 success_rate 추정)
- M2 LLM이 belief × information-gain weighted sampling: "어느 후보가
  가장 정보를 줄까?"
- 결과 click은 단일 선택 (PSRL과 유사)

### Δ3 — Voyager-style executable skill (M3)

**현재**: M3 5턴마다 NL hint만 emit.

**v611**: M3이 두 layer 출력:
- **NL description**: "Tile (44,44) toggles marker C4-E"
- **Executable code** (검증된 skill): Python snippet that env.step에
  대한 effect 측정 후 condition 부합 시 click 추천
- Voyager 패턴: skill 검증 시 (success ≥ 2/3 turns), library에 추가

### Δ4 — Dual-scale SKILL.md (GLoW)

**현재 v608d**: cards / active_hypotheses / falsifications (flat).

**v611**: 두 층 명시
- **Global** `confirmed_skills`: ≥2 reps에서 효과 입증된 NL+code skill
- **Local** `recent_trials` (rolling 30): 최근 turn trial의 NL strategy
  + observed effect

### Δ5 — Self-verification 강화 (M4)

**현재**: M4 paragraph + verdict + next_directive.

**v611**: M4가 매 turn 3 검증 step:
1. **Predicted vs observed effect** 비교 (expected_signature vs env result)
2. **NL strategy validity** 평가 ("Did the strategy work as predicted?")
3. **SKILL.md update commitment** (어떤 카드를 어떻게 update할지 명시)

### Δ6 — visual grounding 강화 (M1/M2)

**현재**: M1이 PNG를 input으로 받지만 사용 명확하지 않음.

**v611**: M1 prompt에 explicit instruction:
- "Look at the PNG. Identify connected regions of same color."
- "For each marker (small symbol), estimate its EAST/SOUTH neighbor
  approximate coord from the image."
- "Output NL strategy describing what you SEE before choosing click."

GPT-5.4-mini multimodal에 PNG 시각 grounding 강제.

## 평가 프로토콜

### 단위 테스트
- M1: NL strategy 출력 형식 검증 (≥30 chars, click_xy_hint 형식)
- M2: posterior sampling 동작 검증 (belief update 후 distribution shift)
- M3: skill compressor가 NL + code 동시 출력 검증
- M4: 3 verification step 모두 발화하는지

### 통합 테스트 (cycle smoke)
- 5-turn smoke on ft09 → SKILL.md에 NL strategy 1개 이상 등록 확인
- ≥1 `recent_trial`이 PNG visual reference 포함

### Live cycle
- 1500-action ft09 episode
- 목표: L+1 풀이 + SKILL.md에 confirmed skill 1+ 등록
- 풀리지 않더라도 NL strategy quality가 cycle405 대비 향상

### Iterative 측정 (Voyager 정신)
- Episode 1, 2, 3 sequential. Episode N+1은 Episode N의 SKILL.md
  로드해서 시작
- 목표: turn-to-L+1이 episode 따라 감소 (또는 episode 3에서 L+2 진입)

## 금지 사항 재확인 (메모리)

- ❌ PATH-D 같은 4096-click scan을 코드로 hardcode (v611 Hybrid REJECT)
- ❌ cycle237 trace의 어떤 정보든 inject
- ❌ 사람이 NL strategy 작성해서 SKILL.md에 seed (단 빈 SKILL.md + 메타
  스킬 템플릿만 허용)
- ❌ marker_constraint_summary 같은 framework predicate가 *유일한*
  reasoning source (PNG visual + NL strategy가 1차)
- ❌ 1 episode 결과로 결론

## 미해결 / codex 협의 필요

1. **PNG vision grounding이 GPT-5.4-mini로 충분한가?** v605_arm7 이미
   multimodal arm 시도했고 0 L+1. Vision LLM 한계인가 prompt 문제인가?
2. **NL strategy → click coord grounding** 메커니즘은? 단순 LLM 신뢰?
   또는 coord-confidence check?
3. **Posterior Sampling**을 LLM으로 정확히 어떻게 구현? Naive sampling
   vs explicit belief table?
4. **Episode-level skill 보존**이 leak risk? (이전 episode의 confirmed
   skill을 inject = 사실상 master trace 시뮬레이션?)

## 다음 단계

1. ⏳ Codex adversarial review (이 plan rev A)
2. ⏳ Codex 답 반영 rev B / rev C 작성
3. ⏳ 수렴 시 plan freeze
4. ⏳ 모듈별 fixture suite 작성 (Δ1-Δ6 각각)
5. ⏳ iterative-code-review 구현
6. ⏳ smoke test → live cycle

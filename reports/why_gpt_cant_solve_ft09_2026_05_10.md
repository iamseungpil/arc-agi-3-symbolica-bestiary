# 왜 GPT는 ft09를 못 풀까 — 처음 보는 사람을 위한 프레임워크 해부

작성일: 2026-05-10
근거 자료: cycle237 (gold L+2), v57_1778305600 (latest L+1 stall), codex 2-round review-counter loop

---

## 0. 한 문장 요약

> GPT는 데이터가 부족해서 못 푸는 게 아니다. **관찰값에 들어 있는 8개 카운터를 모두 더해서 "다 채워졌나?"를 묻는 한 줄짜리 집계 추론**을 못 해서 못 푼다.

이 글은 그 한 문장이 어떻게 270번의 시도 중에 단 1번만 답을 찾게 만들었는지를, 코드와 trace를 보지 못한 사람도 이해할 수 있게 풀어 쓴다.

---

## 1. ft09가 어떤 게임인지부터 (1분 이내)

ft09는 ARC-AGI-3 챌린지의 한 퍼즐이다. 화면 위 색칠된 셀들을 클릭하면 색이 바뀐다. 정답을 맞히면 "레벨"이 올라간다.

핵심 특징 두 가지:

1. **보상이 매우 희소하다.** 보통 50번 클릭 중 1번 정도만 의미 있는 변화를 일으킨다. 나머지는 작은 토글(예: 색 9 ↔ 12로 36개 셀이 깜빡)에 그친다.
2. **상태 의존적이다.** 같은 좌표를 클릭해도 그 이전에 무엇을 했느냐에 따라 결과가 달라진다.

목표는 L+0 → L+1 → L+2 → ... 까지 레벨을 올리는 것. 우리는 270번 넘게 시도했고, **L+2에 도달한 건 cycle237 한 번뿐**이다.

---

## 2. 우리가 만든 프레임워크 — 한눈에

LLM(여기서는 gpt-5.5 via TRAPI)을 매 턴 세 단계로 돌린다.

```
        ┌─────── M1 (가설 생성) ───────┐
        │  visible_regions, marker     │
관찰값  │  states, recent_turns 보고   │   chosen_hypothesis_id
──────►│  "어디를 왜 클릭할까?"        ├──► chosen coord
        │  카드(card) 형태로 가설 제출 │   expected_observation
        └──────────────┬───────────────┘
                       ▼
                 환경에 클릭 전송
                       ▼
        ┌─────── M2 (verdict 채점) ────┐
        │  관찰된 dt(dominant_         │
        │  transition), level_delta로  ├──► verdict ∈ {confirmed,
        │  M1의 expected_obs와 비교    │     refuted, inconclusive}
        └──────────────┬───────────────┘
                       ▼
        ┌─────── M3 (반성+카드정리) ───┐
        │  성공/실패 카드를 1A(추상),  │
        │  1B(특정), F(falsified)에    ├──► SKILL.md 업데이트
        │  분류. M4가 추가 reflexion.  │
        └──────────────────────────────┘
```

이 위에 v600이라는 새 시스템이 얹혀 있다. v600은 LLM 호출을 **stalemate 시에만** 쓰고, 평소엔 12개의 사전 정의된 predicate(예: P03_sector_alignment) × 여러 region을 **Beta-Bernoulli 밴딧**으로 골라 클릭한다. 즉 "어떤 (predicate, region) 조합이 가장 자주 보상을 주는가"를 베이지안 사후분포로 추적한다. RASI prior라는 이름으로 cycle237의 성공 trace가 사전 가중치로 들어간다.

여기까지가 "프레임워크"다. 정상 작동한다. 16/16 fixture+smoke 테스트 통과. 그런데 풀지 못한다.

---

## 3. 왜 못 푸는지 — 가능한 원인 5가지부터

처음 봤을 때 후보로 떠오르는 원인들:

| # | 가설 | 의미 |
|---|---|---|
| H1 | **상태의존 트리거** | 같은 클릭이 상태에 따라 다른 결과 |
| H2 | 가설 누적 망가짐 | active_hypotheses_after가 항상 비어있음 |
| H3 | M1 prompt가 약함 | LLM이 적절한 reasoning을 못 함 |
| H4 | Coord miss | LLM이 region 밖을 자꾸 클릭 |
| H5 | API/timeout | gpt-5.5 응답 실패 |
| H6 | 메모리 cross-run 부재 | 이전 사이클의 교훈을 못 가져옴 |
| H7 | 프레임워크 구조 자체가 부적합 | v600의 밴딧이 본질적으로 표현 불가 |

이 중 어느 게 진짜 원인인지 확인하려면 **trace**를 봐야 한다.

---

## 4. 결정적 증거 — R16 클릭 3번의 비교표

cycle237 (성공)과 latest (실패) 두 trace에서 **같은 좌표 [38, 48]을 클릭한 순간**들만 골라낸다.

| 시점 | 좌표 | 클릭한 region | 결과 dt | level_delta | 의미 |
|---|---|---|---|---|---|
| **cycle237 T8** | [38,48] | R16 | **9→12 count=36** | 0 | 잡음 |
| **cycle237 T27** | [38,48] | R16 | **4→8 count=564** | **+1** | **L+2 트리거!** |
| **latest T39** | [38,48] | R16 | **9→12 count=36** | 0 | 잡음 |

**같은 좌표, 같은 region, 다른 결과.** 클릭하는 행위 자체로는 설명할 수 없다. 그렇다면 환경이 클릭 직전에 어떤 "다른 상태"였는지 봐야 한다.

trace에는 `marker_neighbor_states`라는 필드가 있다. 화면 위 두 marker(R6, R12) 각각의 8방향(N/NE/E/SE/S/SW/W/NW) 이웃 region이 **이번 레벨에서 몇 번 클릭됐는지** 기록한다.

각 시점의 R12 marker compass:

```
cycle237 T8 (실패):
  N=R9:0  NE=R10:0  E=R13:1  SE=R16:0
  S=R15:0 SW=R14:0  W=R11:0  NW=R8:0
  → 8칸 중 1칸만 클릭됨 (1/8)

cycle237 T27 (성공, L+2):
  N=R9:2  NE=R10:1  E=R13:2  SE=R16:1
  S=R15:1 SW=R14:1  W=R11:2  NW=R8:1
  → 8칸 모두 클릭됨 (8/8) ✓

latest T39 (실패):
  N=R9:1  NE=R10:0  E=R13:1  SE=R16:0
  S=R15:0 SW=R14:0  W=R11:1  NW=R8:0
  → 8칸 중 3칸만 클릭됨 (3/8)
```

3개 사례 모두 한 가지 패턴으로 정리된다:

> **R16 클릭이 L+2를 트리거하려면, R12 marker의 8개 compass 이웃이 모두 적어도 1번씩 클릭되어 있어야 한다.**

이걸 **포화 게이트(saturation gate)** 라고 부르자. cycle237 T27에서 우연히 8/8을 만족했기 때문에 L+2가 나왔다. T8과 T39는 각각 1/8과 3/8이라서 같은 클릭이 잡음으로 끝났다.

---

## 5. 그럼 GPT는 왜 이걸 못 봤는가

여기가 핵심이다. **포화 정보가 trace에 이미 있었다.** marker_neighbor_states는 매 턴 LLM에게 전달된다. 8개 카운터가 그대로 들어 있다.

그런데 cycle237 T27에서 LLM이 R16을 고른 이유는 "포화가 됐으니 SE 슬롯을 채우자"가 아니라:

> *"ABSTRACT: rule cites colors=[none], rids=[R31], relations=[crop_sector_class, geometric_compass_position, neighbor_color_parity]; transferable=NO. rule is level-spe..."*

GPT는 `crop_sector_class`, `geometric_compass_position`, `neighbor_color_parity` 같은 **개별 패턴 라벨**을 나열했다. 그러나 어디에도 "8개 카운터 합산"은 없다. **GPT는 8개 sub-field를 `all(clicks ≥ 1)`로 집계하는 술어를 만들지 않았다.**

cycle237이 성공한 이유는 GPT가 R12 포화 게이트를 이해해서가 아니라, **20여 턴 동안 9↔12 토글을 수행하다 보니 의도치 않게 R12의 8개 이웃이 모두 클릭됐기 때문**이다. 운이다.

latest T39를 보면 더 분명하다. LLM은 chosen_hypothesis로 `P_crop_nonzero_sequence_R6_R12`라는 카드를 골랐다. 이름은 "R6_R12 crop nonzero sequence"인데, 실제로 R12 compass의 5개 슬롯이 비어 있었다. 이름은 R12를 가리키는데 R12 포화 점검은 안 한다.

---

## 6. 7가지 후보 가설을 다시 정리

이제 증거를 바탕으로 7개 후보를 4단계로 재배치할 수 있다.

```
L1 (근본 원인): GPT-5.5는 구조화된 관찰 필드를 집계해서
                 완성도 술어(completion predicate)를 만들지 못한다.
                 특히 8개 compass cell을 all(...)로 묶는 추론을 안 한다.

L2 (구조적 결과): v600의 밴딧 키 (predicate × region)는
                   `R12 포화 여부`를 표현 못 한다.
                   → 같은 arm이 성공/실패 evidence를 섞어 받음

L3 (학습 결과): 가설 store가 비어 있어서 "T8 실패 vs T27 성공"이라는
                  paired counterfactual이 메모리에 안 남는다.
                  보상이 트리거 클릭(T27)에만 부여되고
                  set-up sequence(T16-T26의 9↔12 토글들)에는 안 부여됨.

L4 (운영 잡음): coord miss(전체의 ~10%), API timeout, null card —
                  비용은 있지만 게임을 못 푸는 진짜 이유는 아님.
```

| 원인 | 우선순위 | 증거 | 처방 |
|---|---|---|---|
| L1 reasoning | **dominant** | T8/T27/T39 같은 클릭, marker counter는 보였지만 LLM이 집계 안 함 | M1 prompt에 "for each marker, compute saturation = sum(clicks≥1)" 강제 추론 단계 추가 |
| L2 framework | **high** | v600 arm key가 saturation 무시 → 같은 (P, R16) arm이 fail/success 섞어 받음 | arm key 확장: `(predicate, region, marker_saturation_context)` |
| L3 learning | medium | active_hypotheses_after=[] 매 턴, 신용 할당이 트리거 클릭에만 | hypothesis store 디버깅, set-up sequence에도 reward 분배 |
| L4 ops | low | T0,T2 _outside_ 클릭 등 | centroid resolver 강화 (이미 v600에서 일부 처리됨) |

처음 후보였던 H5 (API 오류) 와 H6 (cross-run 메모리 부재) 는 거의 무관함이 밝혀졌다. trace에는 LLM이 일관된 thought를 생성하고 있고, cross_run_memory는 매 사이클 로드되고 있다.

H3 (M1 prompt가 약함) 은 L1과 같은 말이다 — prompt가 약하다 ≈ LLM이 올바른 reasoning을 안 한다.

---

## 7. 그래서 GPT의 어떤 능력이 부족한 건가 — 4가지 진단

LLM 실패의 종류를 좀 더 잘게 나누면:

### (A) Reasoning 실패 — *집계 술어 합성 불가*
**구체적인 모습**: 8개 sub-field를 보고 "다 채워졌나?"를 묻지 않는다. 대신 "이 패턴 라벨 vs 저 패턴 라벨" 같은 표면적 비교를 한다.
**왜 critical한가**: ft09는 신호가 8칸짜리 집계에 숨어 있다. 이걸 못 만들면 어떤 prompt도 안 통한다.

### (B) Prompting 실패 — *집계를 강제하는 step이 prompt에 없음*
**구체적인 모습**: M1 prompt는 "ACTIVE_HYPOTHESES, RECENT_TURNS, IMAGE를 보고 information_value를 계산해 카드를 골라라"라고 한다. **"각 marker의 8개 compass를 더해서 saturation을 먼저 계산해라"는 step은 없다.**
**왜 (A)와 다른가**: 만약 prompt가 명시적으로 saturation 계산을 강제하면, GPT는 잘 한다 (단순 sum이니까). 그러나 자발적으로는 안 한다. → prompt 결함이지 모델 결함이 아닐 수 있다. 둘 다 가능성 있음.

### (C) Framework 실패 — *밴딧 표현력 부족*
**구체적인 모습**: v600이 (P03_sector_alignment, R16) arm을 추적할 때, 이 arm이 이전 cycle에서 "saturated 상태에서 클릭됐을 때 성공" + "unsaturated 상태에서 클릭됐을 때 실패"를 모두 기록한다. 평균을 내면 "가끔 성공"이라는 misleading 사후분포가 나온다.
**왜 critical한가**: prompt를 고쳐도 framework가 학습한 것을 다음 cycle에 못 가져온다. 270 cycle에서 1번만 성공한 이유 중 하나.

### (D) Memory 실패 — *paired counterfactual 부재*
**구체적인 모습**: cycle237에서 T8(실패)과 T27(성공)이 같은 클릭이라는 사실이 어디에도 저장 안 됨. cross_run_memory.json은 1A 추상 mechanism만 보관 ("R31 around crop sector activation works"). 그런데 진짜 가치 있는 정보는 "같은 클릭, 다른 outcome → 두 시점의 marker counter가 달랐다" 라는 contrast 자체.
**왜 critical한가**: 이게 있으면 다음 cycle에서 LLM에게 "이전엔 unsaturated에서 R16 클릭이 실패했다"를 보여줄 수 있다. 없으면 매번 처음부터 더듬는다.

---

## 8. 구체적인 시나리오 워크스루 — latest cycle T31~T67을 무대 위에 올려본다

**T31 (L+1 트리거)**: agent가 [38,46]을 클릭, R30이 5→4 c1824 transition 발생, level_delta=+1. 좋다, 여기까지는 cycle237과 똑같다. dominant_transition signature(5→4 c1824)도 동일.

**T32~T38**: agent가 9개 region(R9, R3, R4, R11, R13, R8, R7, R10, R5)을 클릭. 모두 9↔12 c36 토글. 이 동안 R12 compass에 N=R9, E=R13, W=R11 세 칸이 1씩 채워짐. NE, SE, S, SW, NW 다섯 칸은 여전히 0.

**T39**: agent가 [38,48] 클릭 → R16 (R12의 SE 이웃). dominant_transition은 9→12 c36. **L+2 안 발생.** 왜? R12 compass가 3/8밖에 안 됐기 때문. SE를 채웠어도 NE, S, SW, NW가 비어 있어 saturation 게이트가 안 열림.

**T40~T67**: agent가 27턴 더 다양한 region을 클릭. 그러나 R10, R14, R15, R8 같은 R12의 unsaturated compass slot으로 전혀 안 가고 R3, R4, R5, R7, R9, R11, R13 위주로 반복. 16턴 동안 R10:0, R14:0, R15:0 안 변함. 결국 timeout.

**왜 agent가 R10/R14/R15에 안 갔나?** chosen_hypothesis_id를 보면 `P_lower_crop_R9_completion`, `P_unvisited_lower_zero_sector_R15` 같은 이름이 나오긴 한다. 그런데 좌표 변환 후 결국 R3, R5 같은 already-clicked region에 떨어진다. **LLM은 "R12의 compass 중 어떤 region이 아직 0인가"를 묻지 않고, 자기가 만든 추상 이름을 기준으로 region을 고른다.**

이게 codex와 합의한 결론이다:

> GPT did not fail because the information was missing. It failed because the useful rule was an aggregate structural predicate: "have all eight compass neighbors of this marker been clicked?" The trace exposed the counters, but the model reasoned in local pattern labels instead of computing the completion condition.

---

## 9. 그래서 v600이 이걸 풀까? — 솔직한 진단

v600의 design은 "predicate library + Beta-Bernoulli + UCB1 + LLM-on-stalemate-only" 다. 이 design 자체로는 **위 saturation 게이트를 못 푼다**. 이유:

- 12개 정적 predicate 중 어느 것도 `marker_compass_saturation`을 포함하지 않음.
- LLM extender가 stalemate에 호출되긴 하지만, 그 prompt에도 "saturation을 계산하라"는 강제가 없음.
- arm key (predicate, region)에 saturation context가 없어 학습이 섞임.

v600이 풀려면 **최소 3가지 추가**가 필요하다:

1. **predicate에 P_marker_saturation_complete 추가** (`fn(state) → list of regions where clicks_in_compass_of_visible_marker[*] == 0`)
2. **arm key에 saturation context 추가** (`(predicate, region, marker_saturation_status)`)
3. **M1 prompt에 mandatory step 추가** ("Step 0: for each marker, compute compass[*].clicks; flag any with all ≥1 as saturated; flag any with exactly 1 unclicked as 'one click from saturation'")

이 셋 중 (3)이 가장 cheap이고 가장 강력하다. 다른 둘은 v600 코드를 건드려야 한다.

**Phase D1 autoresearch는 어떻게 해야 하나**: codex의 제안을 받아들여, "v600이 ft09를 푸는가"를 묻는 대신 "v600이 saturation paired counterfactual을 충분히 생성하는가"를 묻는 confirmatory 실험으로 frame해야 한다. 즉:
- 10 episode 돌리면서 R16 클릭이 발생할 때마다 직전 marker compass 상태를 기록
- saturated/unsaturated 비율과 outcome 매핑
- 이 데이터가 (1)~(3) 패치를 정당화하는 통계적 근거를 만드는지 검증

---

## 10. 결론 — 한 페이지로

> ft09는 marker 8-compass saturation 게이트를 풀어야 진행되는 퍼즐이다. 정보는 매 턴 LLM에게 주어진다. 하지만 GPT는 8개 sub-field를 합쳐 `all(clicks ≥ 1)`을 계산하는 자발적 추론을 하지 않는다. 대신 표면적 패턴 라벨로 reasoning한다. v600 framework는 그 raw counter를 arm key에 통합하지 않아서, 한 cycle에서 운으로 saturation을 만나도 다음 cycle로 학습이 이어지지 않는다.

해결 방향은 분명하다. M1 prompt에 saturation 계산을 강제하고, arm key에 saturation context를 넣고, paired counterfactual을 cross-run memory에 보관한다. 그러면 270 cycle 중 1번 성공이 아니라, 매 cycle 안정적으로 L+2를 찍는 쪽으로 분포가 옮겨질 것으로 예측한다.

Phase D1을 confirmatory contrastive dataset 수집 실험으로 돌리고, 결과를 보고 patch 우선순위를 정하면 된다.

---

## 부록 A — 자주 받는 반론들

**"trace에 진짜 없는 거 아냐? marker_neighbor_states가 LLM에게 안 보였을 수도?"**
보였다. agent.py의 visible_regions/marker_neighbor_states 빌드 코드는 prompt에 직접 넣는다. cycle237 T0의 첫 thought도 marker 4개를 언급한다. 데이터는 거기 있었다.

**"cycle237이 우연이라면, 왜 latest는 36턴 동안 saturation을 못 채웠나?"**
T31~T67을 보면 동일 region(R3, R4, R5, R9, R11, R13)을 반복 클릭한다. R10, R14, R15 같은 unvisited compass slot으로의 push가 부족했다. 이는 M1이 "어떤 region이 unvisited인가"가 아니라 "어떤 카드가 information_value 높은가"로 reasoning하기 때문. 둘은 다르다.

**"saturation 가설이 다른 marker에도 적용되나?"**
검증 안 됨. 현재까지 R12 compass × R16 SE-slot 트리거 한 케이스만 paired counterfactual로 확보. R6 compass 트리거나 다른 level의 게이트는 모름. Phase D1에서 추가 데이터 수집 필요.

**"왜 290 cycle 동안 이걸 못 알아챘나?"**
이전 디버깅은 trace를 줄 단위로 보고 "이 카드가 왜 안 emit됐나" 같은 module 단위 문제로 frame했다. T8 vs T27 vs T39 contrast를 같은 표에 올린 건 이번이 처음. **대조 실험을 하기 전엔 신호가 안 보였다.**

---

## 부록 B — codex review-counter loop 요약

라운드 1 (60s, gpt-5.5 medium reasoning):
- Codex 반박: "parity"라고 단정하지 마라. 가능 alternative는 ① directionality ② phase gate ③ saturation. v600은 fatal일 수 있다. paired counterfactual을 먼저 추출해라.

라운드 2:
- 우리: marker_neighbor_states clicks 카운터 추출 → T8 1/8, T27 8/8, T39 3/8. 명확히 ③ saturation 지지.
- Codex 수렴: directionality와 phase gate를 secondary로 demote. v600 incomplete (not fatal). arm key 확장이 cheap한 patch. 설명문은 saturation 예시로 lead.

수렴 taxonomy: L1 reasoning 실패(집계 술어 미합성) → L2 framework 한계(featureization 부족) → L3 learning 부재(credit assignment + paired memory) → L4 ops 잡음.

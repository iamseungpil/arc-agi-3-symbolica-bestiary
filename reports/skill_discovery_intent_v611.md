# Skill Discovery 의도 — v611 plan 기반 문서 (2026-05-12)

본 문서는 ft09 (및 ls20, vc33) 풀이 framework의 **재설계 의도**를 고정한다.
v52 PDF (`reports/v5_simple_runtime_talk.pdf`)와 memory
`project_skilldiscovery_intent.md`의 원칙을 그대로 따른다.

## 핵심 의도 (한국어, 원문 보존)

> **"Leakage 없이, 가설을 세우고, 그 다음 거기서 strategy를 자연어로
> 뽑아서 문제를 풀 수 있을 때까지 SKILL.md를 개선한다."**

이 한 문장이 framework의 단일 진실 원천이다. 모든 설계 결정은 이 문장에
부합해야 한다.

## 의도 분해 (4 요건)

| 요건 | 의미 |
|------|------|
| **Leakage 없이** | (a) cycle237 같은 leaked trace 사용 금지 (b) ft09-specific predicate hardcode 금지 (c) 사람이 affordance 또는 search procedure를 코드로 미리 작성하지 않음 |
| **가설을 세우고** | LLM이 매 turn frame (text + PNG) 보고 mechanic 가설 생성. 가설은 falsifiable해야 함 (expected effect 명시) |
| **Strategy를 자연어로 뽑아서** | 검증된 가설로부터 자연어 strategy (Condition / Policy / Termination / Interface = CπTR) 추출. 자연어이지 코드 아님 |
| **SKILL.md를 개선** | 매 iteration (episode 또는 turn) SKILL.md에 active_hypotheses / confirmed_skills / strategies / falsified 축적. **이 축적이 다음 iteration 풀이 능력을 높여야 함** |

## 금지 사항 (이전 churn 패턴)

| 금지 | 사례 |
|------|------|
| ❌ 사람이 코드로 affordance / search substrate 작성 | PATH-B/C/D, v611 Hybrid Phase A scan |
| ❌ 고정 프롬프트 1회 실행 | 단발성 evaluation, prompt engineering |
| ❌ ft09-specific token / predicate hardcode | `bsT`, `gqb`, `Hkx`, `cycle237_path` |
| ❌ leak trace inject (replay scaffold) | PATH-B target_state_keys |
| ❌ 코드 BFS / MCTS를 LLM "도구"로 위장 | "scan tool"이 사실 hardcoded enumeration |
| ❌ 1 episode 결과로 결론 | 반복적 개선이 본질 |

## 성공 기준

### 단기 (ft09)
- L+1 풀이를 **LLM이 자율적으로** 달성 (코드 substrate 없이)
- 풀이 trajectory가 SKILL.md에 confirmed_skill로 자동 등록
- 다음 episode에서 그 skill 활용 (또는 변형) 시도

### 중기 (ft09)
- L+1 → L+2 → ... → L+6 점진적 진전
- 각 level 풀이가 이전 SKILL.md 축적의 결과
- skill 수 vs level 곡선이 monotonic 증가

### 장기 (cross-game)
- ls20, vc33도 같은 framework로 풀림
- ft09 SKILL.md에서 학습된 *abstract* skill (mechanic 추상화)이 다른 게임에
  부분적으로 transfer
- Iteration 별 진전 측정 (skill 0개 baseline vs 10개 후 성능)

## 평가 메트릭

1. **레벨 달성**: `levels_completed` per episode
2. **Skill 수**: SKILL.md의 confirmed_skill 카드 개수
3. **Skill 사용률**: 각 skill의 mention count + usage in M2 decision
4. **Iteration 진전**: episode N+1의 평균 turn-to-L+1이 episode N보다 짧아지는가?
5. **Cross-level transfer**: L+1 풀이 skill이 L+2 풀이에 활용되는가?
6. **Zero leakage**: 모든 결과는 `ARC_NO_GOAL_LEAK=1` + anti-leak gate 통과

## 진단 (2026-05-12 현재)

- **v52 framework (agentica_lite)**: M1-M4 + SKILL.md 구조 구축됨
- **ft09 결과**: 0/6 levels (10 configs × 7000 actions)
- **핵심 문제**: LLM은 marker_constraint 가설은 만들지만 (cycle405 700+
  actions), region bbox centroid 클릭이 ft09 affordance (8 tiles in
  bottom-right region) 밖에 떨어져 evidence 못 모음
- **PATH-D 발견 (leak 인정)**: ft09 affordance space는 9개 6×6 tile (4096
  click 중 effective 288개). 이건 ground-truth observation이지만 *LLM이
  자율 발견해야 할 정보*임

## 아직 결정 안 된 것 (survey + codex 협의 필요)

- LLM이 affordance space를 **자율 발견**할 방법 (vision-LLM이 PNG 보고?
  LLM-driven random exploration? Bayesian active learning?)
- SKILL.md의 **schema** — natural language CπTR vs structured cards
- Hypothesis **falsification** 메커니즘 — turn count? 통계적 test?
- **Cross-level skill 보존** vs invalidation 기준

## 관련 연구 (2026-05-12 survey 완료)

### 가장 직접 관련 — 우리 framework의 부모/형제

- **Voyager (arXiv 2305.16291)** — 본 framework의 *직접 부모*. 3 요소:
  (1) automatic curriculum (exploration 최대화), (2) ever-growing skill
  library (executable code 저장+검색), (3) iterative prompting
  (env feedback + execution errors + self-verification). v52 M1-M4 +
  SKILL.md 구조와 거의 동일. 우리가 못 푸는 게 신호 — 구현 디테일 문제.

- **C1 Chess (arXiv 2603.20510, master distillation)** — Expert system이
  NL chain-of-thought 설명을 만들고 LLM이 4B 모델로 그걸 distill하여
  near-zero → 48.1% accuracy. **시사**: master trace가 있으면 NL
  reasoning이 학습된다. ft09에서는 cycle237이 그 master trace였으나
  leak 판정 — leak-free에서는 *LLM이 자력으로 NL reasoning을 발화하고
  검증*해야 함.

- **Product of Experts ARC (arXiv 2505.07859, 71.6%)** — 단일 LLM을
  generator + scorer 2-role로 사용. 두 번째 LLM 호출이 *score by
  re-conditioning on solution candidate*. 가설-검증 loop을 generator/
  scorer 분리로 구현. 우리 M1(propose)/M4(verdict) 분리와 동형.

- **ARC code evolution + NL (Imbue, ARC-AGI-2 95%)** — LLM이
  *natural language로 transformation rule을 먼저 설명* → 그 다음 Python
  code 출력. "Reasoning in NL aligns model priors with human visual
  language". **우리 현재 결함**: M2가 NL strategy 없이 region_hint →
  coord direct mapping. NL-first 단계가 빠짐.

### 가장 직접 관련 — Exploration policy

- **PSRL with LLMs (ICLR 26 `dilipa.github.io/papers/iclr26_psrl_llms.pdf`)**
  — LLM = atomic functions, Posterior Sampling RL 구조. Wordle MDP example.
  **시사**: LLM이 *random exploration* 또는 *brute-force*가 아닌
  posterior 기반 sampling으로 다음 click을 선택. 우리 PSRL-style M2
  는 belief × information gain 가중치.

- **LLM-Explorer (OpenReview VA5P0rUZPx)** — adaptive task-specific
  exploration strategy via LLM. 37% RL improvement. 우리 M1 prompt에서
  exploration prior를 동적으로 갱신 가능.

- **GLoW dual-scale memory (OpenReview bH5uHIVtTe)** — global high-value
  discoveries + local trial-and-error. Jericho text-game SOTA.
  **시사**: SKILL.md를 두 층으로 — `confirmed_skills` (global) +
  `recent_turn_trials` (local).

### 가장 직접 관련 — Reward / Strategy 자율 발견

- **EUREKA** — LLM이 evolutionary search로 reward function 자율 설계,
  human-level. "Practice-and-refine loop". 우리 M3 skill compressor에
  자연어 strategy를 *evolutionary mutate*하는 메커니즘 추가 가능.

### 비교적 약한 관련 (참고만)

- **ChessArena (arXiv 2509.24239)** — 13 LLMs 평가, 대부분 amateur 미달
- **LATCH (medRxiv)** — NL hypothesis → executable analysis (clinical)
- **TLA+ + LLM** — formal verification, 보조 역할만
- **Pi-Autoresearch** — continuous edit-measure-keep/revert

## 종합 시사 — v611 설계 원칙 (survey-grounded)

1. **NL-first reasoning** (ARC code evolution / C1 chess) — M1/M2 모두
   coord 결정 전에 자연어 strategy를 먼저 발화. 그 자연어가 click
   coord로 grounding됨.

2. **Posterior Sampling exploration** (PSRL with LLMs) — M2 click 선택은
   belief × information gain. 무작위 brute-force도 아니고 코드
   substrate scan도 아님.

3. **Voyager-style skill library** — confirmed code skill (executable)
   + NL description. 우리의 SKILL.md `confirmed_skills`가 이 역할.

4. **Dual-scale SKILL.md** (GLoW) — global high-value (confirmed) +
   local turn trials (recent_verbose). 이미 v608 구조에 있음.

5. **Iterative prompting with self-verification** (Voyager) — env feedback
   + error + self-verification. 우리 M4 paragraph + verdict가 이 역할.
   M4가 self-verify를 강하게 하면 다음 turn 가설 quality가 올라감.

6. **No master trace** — leak-free 의도상 cycle237 같은 expert trace
   금지. LLM이 자력으로 환경에서 hypothesis 발화 + 검증해야.

## 진단된 결함 (cycle405 분석 + survey 종합)

| 결함 | 진단 |
|------|------|
| LLM이 NL strategy를 발화 안 함 | `proposer_prompt.py`가 region_hint → coord direct mapping. NL "Click bottom-right tile to repair C4 E-slot" 단계 없음 |
| Exploration이 PSRL이 아니라 stuck-trigger fallback | LLM 가설이 같은 region 반복 시도. belief update 안 됨 |
| Skill compressor가 code skill 안 만듦 | M3이 NL hint만 emit. Voyager 식 executable program 없음 |
| Multi-modal grounding 약함 | M2가 PNG 보지만 region centroid만 클릭. visual affordance 추론 없음 |


## 다음 단계

1. ✅ 이 의도 문서 freeze
2. ⏳ 관련 연구 서베이 (web search)
3. ⏳ Codex와 review-counter loop (수렴까지)
4. ⏳ 최종 plan freeze
5. ⏳ 구현 + ft09 cycle launch

# v57 → v609 redundancy audit (2026-05-12)

## TL;DR

Codex 메타 평가 *"Better instrumentation, worse direct-policy efficacy"*가
정량적으로 입증됨. **agents/templates/의 ~90% + research_extensions/의
100%가 v609 ft09 hot path 외부.**

| 카테고리 | kloc | 비율 | 처리 |
|----------|------|------|------|
| ACTIVE (v609 hot path) | 6.8 | ~10% | 유지 |
| DEAD (research_extensions) | 29.2 | ~45% | 삭제 후보 |
| DORMANT (legacy agents) | ~10+ | ~45% | 삭제 또는 archive |

## Phase 1 — 코드 인벤토리 (완료)

### Hot path (유지)

- `agents/templates/agentica_lite/` (21 files, 6.8 kloc, 100% ACTIVE)
  - Core orchestration: agent.py / adapter.py / policy.py
  - LLM roles: proposer.py / proposer_prompt.py / reflector.py / llm_extender.py
  - Prediction & learning: predicate_posterior.py / predicate_library.py
    / seed_cards.py / region_canonical.py / anti_leak.py
  - State & rendering: skill_state.py / skill_md_renderer.py /
    _frame_to_state.py / _render_frame.py / memory_writer.py
  - Support: memory_journal.py / stalemate_trigger.py / fixtures.py
- `scripts/v609_search/` (3 files, PATH-B 솔버)
  - a_star.py / frame_hash_sim.py / run_feasibility.py
- 3 active launchers (v607/v608b/v608d) — 모두 `-a lite`만 호출

### Dead code (삭제 후보)

- `research_extensions/` (29.2 kloc, ZERO active imports)
  - phase_controller, hypothesis_store, wake_planner, sleep_handler
  - abstraction, divergence_monitor, generator_synth/backtest
  - modules/world_model (2.7k), modules/dreamcoder (2.3k),
    modules/planner (954), meta_harness, seed_library
  - Imported only by:
    - agents/templates/agentica/agent.py (legacy v55, test-only)
    - agents/templates/arcgentica_research/agent.py (245k monolith,
      test-only)
    - scripts/run_research_smoke.py, scripts/run_bounded_experiment.py
      (v585-era, not v609)
- `agents/templates/arcgentica_research/` (488KB, 1 file)
  - 7 unit tests에서만 import (rev_m, rev_s, w1, p13, p4, p1)
  - No active launcher
- `agents/templates/agentica/` (460KB, legacy v55)
  - test-only
- `agents/templates/agentica_simple/` (~300KB, pre-v600)
  - launch_simple_detached.sh referenced, but v608d 이후 deprecated
- `agents/templates/agentica_v57/` (forensic archive)
  - test_v57_modules / test_v586_splice / scripts/smoke_v585x_forensic.py
- 2 dormant launchers: launch_v591_detached.sh (agentica_v57),
  launch_simple_detached.sh

## Phase 2 — 74 commit 카테고리화

### v57-v593 (B14~B20 era — research_extensions/arcgentica_research path)

전부 DEAD. 작업 내용:
- B14 v587: hierarchical memory (ESC + ASMW + CPSR) → research_extensions
- B16 v588: action-state chain compression + chain-level hypothesis →
  research_extensions/modules/world_model
- B17 v589: typed candidate tests → arcgentica_research
- B18 v590: predicate-induction → 이후 round-3 REVERT (regression)
- B19 v591: two-tier invention (TIER-A library + TIER-B invented) →
  arcgentica_research
- B20 v592: TIER balance + bidirectional reflection →
  arcgentica_research
- v593: reseed on L+ + early TIER-A T8-T15 force → arcgentica_research

**Net contribution to v609 ft09 capability**: ZERO. 이 작업들의 핵심
insight (anti-leak / two-tier invention / Trace2Skill distill) 일부가
v607 anti_leak.py와 v607 predicate_posterior.py에 흡수됐지만, 인프라
자체는 폐기.

### v600+ (agentica_lite path — ACTIVE)

- v600: minimal-fast — agentica_lite reset (B-series 인프라 폐기)
- v601: 3-role saturation bundle (Proposer/Policy/Memory)
- v602: SKILL.md SSOT + 81 strict per-module unit tests
- v603: ArcgenticaLite Agent adapter
- v604: TRAPI robustness (circuit breaker, JSON extractor, model swap)
- v605: 7 arms (multimodal, marker filter, ...)
- v606: TIER-B region-anchored predicate invention unlock
- v607: anti_leak.py + Beta posterior + chid_template (codex 6-round)
- v608b-f: constraint card skill (atomic citation, transition log,
  active override)
- v609: A* search + 4-pixel snap grid + cycle237 anchors (PATH-B)

**Net contribution to v609 ft09 capability**: ALL OF IT. 단, ft09 L+1
실제 풀이는 v609 PATH-B + cycle237 anchor에서만 발생.

### 통계

- 74 commits 중 **agentica_lite 영향 commits**: ~50개
- 74 commits 중 **research_extensions/arcgentica_research 영향 commits**:
  ~24개
- **ft09 L+1 capability 실제 증대 commits**: 2개 (v57 base + v609 PATH-B)

## Phase 3 — 삭제 안전성

### 1순위 삭제 (안전도 100%)

`research_extensions/` 전체 (29.2 kloc) — agentica_lite/*.py에서
zero imports. test 하위에서는 일부 import 있지만 모두 legacy agent
호출 path.

### 2순위 삭제 (안전도 95%)

`agents/templates/arcgentica_research/` (488KB) — research_extensions
삭제 후 imports orphan. 유지 시 legacy test harness만 보존.

### 3순위 삭제 (안전도 98%)

`agents/templates/agentica/` + `agentica_simple/` + `agentica_v57/`
(~1.2 MB combined) — test-only. agentica/scope/frame.py만 Frame type
dependency 확인 필요.

### 부수 cleanup

- `scripts/launch_v591_detached.sh`, `scripts/launch_simple_detached.sh`
- `scripts/extract_fixtures_v585y.py`, `scripts/extract_splice_v586.py`,
  `scripts/run_splice_v586.py`, `scripts/measure_v591_invention.py`,
  `scripts/run_research_smoke.py`, `scripts/run_bounded_experiment.py`
- `tests/test_v57_modules.py`, `tests/test_v586_*`, `tests/test_w1_*`,
  `tests/test_rev_m_*`, `tests/test_p13_*`, `tests/test_p4_*`,
  `tests/test_p1_*` (legacy)

## Phase 4 — v610 minimal manifest (제안)

```
agents/templates/agentica_lite/      # 6.8 kloc, 21 files (KEEP)
scripts/v609_search/                  # PATH-B substrate (KEEP)
scripts/launch_v608d_detached.sh      # active launcher (KEEP)
scripts/check_no_leak_prompts.py      # anti-leak gate (KEEP)
scripts/score_*.py                    # instrumentation (KEEP)
scripts/build_*.py                    # viz instrumentation (KEEP, optional)
tests/v608/                           # active test suite (KEEP)
tests/test_v53_modules.py             # M1-M4 prompt tests (KEEP)
tests/test_v609_*.py                  # PATH-B tests (KEEP)
reports/                              # closure notes (KEEP)
```

**삭제**: ~31 kloc + 1.2MB legacy agents + ~15 legacy scripts +
~10 legacy tests.

## 권고

이 audit 자체는 destructive action 없음 (read-only). 다음 단계:
1. user 승인 후 `research_extensions/` + legacy agents archive
   (git branch `archive/pre-v610` 만들고 main에서 제거)
2. v610 manifest를 README 또는 ARCHITECTURE.md로 freeze
3. PATH-B + agentica_lite만 유지한 minimal substrate에서 다음 게임
   일반화 검증 (ls20 등) 진행

이렇게 하면 "더 정직해졌지만 안 강해진" 시스템의 churn 구조가 제거되고,
실제 capability에 직접 기여하는 코드만 남게 됨.

# ARC-AGI-3 ls20: LLM World-Model Cheating Bestiary

Research artifacts from autoresearch experiments teaching an LLM
(`gpt-5.3-codex` via TRAPI) to act as a Code World Model (CWM, Lehrach et al.
2025) + DreamCoder (Ellis et al. 2021) + MCTS planner on ARC-AGI-3 ls20.

**Headline**: Level 0 still unsolved after 15+ iterations, but we documented
four concrete ways the LLM **games the world-model scorer** and a
source-grounded fix for each.

## Outputs

- **`reports/v3_mcts_wake_sleep_talk.pdf`** — 23-slide Beamer talk
  (bestiary narrative)
- **`presentation/bestiary.html`** — browsable cheat index + live prompt
- **`presentation/trace_viewer_<ns>.html`** — per-experiment step viewer
- **`experiment_logs/autoresearch_results.tsv`** — every bounded run

## Bestiary of cheating modes

| # | version | cheat | fix |
|---|---|---|---|
| 1 | v3.6 | Wildcard (`'unknown'` on every field → pass trivially) | Strict matcher |
| 2 | v3.11 | 1-cell delta (precision 100%, recall ≈ 0) | Require full `expected_next_grid` |
| 3 | v3.14 | Identity (`predicted = input` unchanged) | Identity guard in matcher |
| 4 | v3.15.1 | Terminal-always-true (both branches return `expect_change=False`) | **pending** |
| 5 | v3.14 (delivery) | exec builtins missing `str`/`enumerate` → silent NameError | Expanded builtins |

## Honest progression

| run | reported `transition_accuracy` | honest (strict + identity guard) |
|---|---|---|
| v3-full | 0.922 | 0.00 |
| v3.6 | 1.000 (gamed) | 0.00 |
| v3.11 | 0.00 | 0.00 |
| v3.14 | 0.00 | 0.00 |
| **v3.15.1** | **0.016 (1/64)** | **0.016** first honest pass |

## Architecture

```
agent (TRAPI gpt-5.3-codex, Symbolica-style tools)
    │  predict / world_update / propose_skill / env_note
    ▼
ResearchBridge (SharedBridge)
    │
    ├── WorldModelModule         code simulator (predict_effect)
    │       ↑ wake on surprise   unit_tests + transition_accuracy
    │
    ├── DreamCoderModule         skill library
    │       ↑ sleep refactor     abstract_primitive + observed_law + mcts + free + wrapper
    │
    ├── MCTSPlanner              imagination on predict_effect (no real env)
    │       ↑ propose_from_mcts
    │
    └── MetaHarnessModule        overlay selection by exploration metrics
```

## License

Code is MIT. ls20 game © LatticeFlow.

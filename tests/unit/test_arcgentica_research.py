"""Tests for the modular research track.

Design goals being enforced here:
- Modules are independently toggleable (baseline stays clean when all off).
- No module references hidden progress markers like ``levels_completed``.
- Skill names / trigger shapes / overlay texts are NEVER hardcoded per run.
- DreamCoder only enters its sleep loop in response to surprise events on
  the shared bridge, not in response to level-up.
- World-model surprise detection is based on grid diff, not on a hidden
  progress counter.
- Meta-harness scores runs using agent-side metrics only (diversity,
  diff rate, surprises, proposals), not ``levels_completed``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from arcengine import GameAction, GameState

from agents.templates.agentica.agent import Arcgentica
from agents.templates.arcgentica_research.agent import (
    ArcgenticaResearch,
    _resolve_shared_namespace,
)
from research_extensions import ResearchConfig, load_research_config
from research_extensions.config import ModuleConfig
from research_extensions.grid_utils import current_grid
from research_extensions.modules.world_model import WorldModelModule
from research_extensions.registry import ModuleRegistry, ResearchRuntimeContext
from research_extensions.verification import (
    analyze_dreamcoder_state,
    analyze_meta_harness_state,
    analyze_world_model_state,
    verify_prompt_contract,
)


def _make_frame(grid: list[list[int]]) -> SimpleNamespace:
    """Build a minimal frame object that mimics FrameData."""
    return SimpleNamespace(
        frame=[grid],
        state=SimpleNamespace(name="NOT_FINISHED"),
        available_actions=[1, 2, 3, 4],
        levels_completed=0,
        win_levels=7,
        game_id="ls20-test",
    )


def _make_context(tmp_path: Path, tag: str = "run") -> ResearchRuntimeContext:
    run_dir = tmp_path / tag
    shared = tmp_path / "shared"
    run_dir.mkdir(parents=True, exist_ok=True)
    shared.mkdir(parents=True, exist_ok=True)
    return ResearchRuntimeContext(
        game_id="ls20-test",
        workdir=run_dir,
        shared_dir=shared,
    )


# -- baseline preservation -------------------------------------------------
def test_registry_loads_no_modules_by_default(tmp_path: Path):
    cfg = ResearchConfig()
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    assert registry.active() == {}
    assert registry.prompt_overlay() == ""


def test_agent_name_includes_baseline_suffix():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    agent.game_id = "ls20"
    agent.research_config = ResearchConfig()
    assert agent.name == "ls20.arcgentica_research.baseline"


def test_default_runtime_config_is_baseline_off():
    cfg = load_research_config()
    assert cfg.active_modules() == []


def test_agent_system_prompt_describes_truthful_local_surface():
    prompt = ArcgenticaResearch._default_system_prompt()
    assert "GAME_REFERENCE" not in prompt
    assert "`history`" in prompt
    assert "`memories_summaries`, `memories_get`, `memories_add`, and `memories_query`" in prompt
    assert "`frame_render`, `frame_diff`, `frame_render_diff`" in prompt
    assert "`submit_action` is this host's action commit tool" in prompt
    assert "EXPECT_CHANGE:" in prompt
    assert "agent-side research overlays" in prompt
    assert "`world_update`" in prompt


def test_direct_startup_env_helpers(monkeypatch):
    monkeypatch.delenv("ARC_AGENTICA_DIRECT_STARTUP", raising=False)
    monkeypatch.delenv("ARC_AGENTICA_DIRECT_STARTUP_BUDGET", raising=False)
    assert Arcgentica._direct_startup_enabled() is False
    assert Arcgentica._direct_startup_budget(24) == 8

    monkeypatch.setenv("ARC_AGENTICA_DIRECT_STARTUP", "1")
    monkeypatch.setenv("ARC_AGENTICA_DIRECT_STARTUP_BUDGET", "5")
    assert Arcgentica._direct_startup_enabled() is True
    assert Arcgentica._direct_startup_budget(24) == 5
    assert Arcgentica._direct_startup_budget(3) == 3


def test_main_bootstrap_resets_before_first_llm_turn():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    agent.frames = [SimpleNamespace(state=GameState.NOT_PLAYED)]
    bootstrapped = SimpleNamespace(
        state=GameState.NOT_FINISHED,
        frame=[[[0]]],
        available_actions=[1, 2],
        levels_completed=0,
        win_levels=7,
        game_id="ls20-test",
    )
    seen: list[str] = []

    def fake_take_action(action):
        seen.append(action.name)
        return bootstrapped

    agent.take_action = fake_take_action
    agent.append_frame = lambda frame: agent.frames.append(frame)

    agent._ensure_bootstrap_frame()

    assert seen == ["RESET"]
    assert agent.frames[-1] is bootstrapped


def test_default_shared_namespace_is_ephemeral(monkeypatch):
    monkeypatch.delenv("ARC_RESEARCH_SHARED_NAMESPACE", raising=False)
    timestamp = 1776207000
    monkeypatch.setattr("agents.templates.arcgentica_research.agent.time.time", lambda: timestamp)
    monkeypatch.setattr("agents.templates.arcgentica_research.agent.os.getpid", lambda: 4321)
    assert _resolve_shared_namespace() == "ephemeral_1776207000_4321"


def test_explicit_shared_namespace_is_preserved(monkeypatch):
    monkeypatch.setenv("ARC_RESEARCH_SHARED_NAMESPACE", "experiment_ls20_iter1")
    assert _resolve_shared_namespace() == "experiment_ls20_iter1"


def test_frame_to_text_renders_full_grid_without_default_truncation():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    grid = [[(r + c) % 10 for c in range(8)] for r in range(20)]
    frame = SimpleNamespace(frame=[grid], state=SimpleNamespace(name="NOT_FINISHED"))
    text = agent._frame_to_text(frame)

    # Symbolica-unified Frame.render uses y=NN prefix.
    assert "y=19" in text
    assert "... (" not in text


def test_main_respects_max_actions_exactly():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    agent.MAX_ACTIONS = 2
    agent.frames = [_make_frame([[0]])]
    agent.frames[-1].state = GameState.NOT_FINISHED
    agent.action_counter = 0
    agent.action_history = []
    agent.best_level = 0
    agent.reset_count = 0
    agent.game_id = "ls20-test"
    agent.research_config = ResearchConfig()
    agent.research_workdir = Path("/tmp")
    agent.shared_namespace = "test_namespace"
    agent._registry = None
    agent.cleanup = lambda: None
    agent.is_done = lambda frames, latest: False
    agent.choose_action = lambda frames, latest: GameAction.ACTION1

    calls: list[str] = []

    def fake_take_action(action):
        calls.append(action.name)
        return SimpleNamespace(
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            game_id="ls20-test",
            available_actions=[1, 2, 3, 4],
            frame=[[[0]]],
        )

    agent.take_action = fake_take_action
    agent.append_frame = lambda frame: agent.frames.append(frame)

    agent.main()

    assert calls == ["ACTION1", "ACTION1"]
    assert agent.action_counter == 2


def test_tool_definitions_expose_symbolica_like_helpers():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    tools = agent._tool_definitions()
    names = {tool["name"] for tool in tools}
    assert "history" in names
    assert "memories_add" in names
    assert "memories_get" in names
    assert "memories_query" in names
    assert "frame_render" in names
    assert "frame_diff" in names
    assert "frame_render_diff" in names
    assert "submit_action" in names


# -- module independence ---------------------------------------------------
def test_each_module_can_run_alone(tmp_path: Path):
    for module_name in ("dreamcoder", "world_model", "meta_harness"):
        cfg = ResearchConfig()
        setattr(cfg, module_name, ModuleConfig(enabled=True))
        registry = ModuleRegistry(cfg, _make_context(tmp_path, tag=module_name))
        registry.load()
        assert list(registry.active().keys()) == [module_name]
        # The active-only overlay should not be empty (each module returns
        # at least a small scaffold string).
        assert registry.prompt_overlay() != ""


def test_registry_before_action_prefers_dreamcoder_continuation_over_world_prior(tmp_path: Path):
    cfg = ResearchConfig(
        dreamcoder=ModuleConfig(enabled=True),
        world_model=ModuleConfig(enabled=True),
    )
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]
    wm = registry.active()["world_model"]

    registry.record_agent_skill_proposal(
        {
            "name": "Exact winning spine",
            "body": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
            "action_spine": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
            "description": "Validated successful run.",
        }
    )
    dc.after_action(None, "ACTION2", None)
    skill = dc._find_by_name("Exact winning spine")
    assert skill is not None
    skill.observed_support = 5
    skill.informative_hits = 5

    registry.bridge.recent_actions = ["ACTION3", "ACTION3", "ACTION4"]

    frame = _make_frame([[0, 0], [0, 0]])
    sig = wm._signature(frame)
    wm.transitions[sig] = {
        "ACTION4": {"count": 3.0, "avg_diff": 5.0, "zero_count": 0.0},
        "ACTION1": {"count": 3.0, "avg_diff": 1.0, "zero_count": 0.0},
    }

    chosen = registry.before_action(frame, "ACTION4", ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert chosen == "ACTION1"


# -- DreamCoder: surprise triggers sleep, no level-up reference -----------
def test_dreamcoder_sleep_is_surprise_driven(tmp_path: Path):
    # DC alone cannot turn observations into surprises — it relies on the
    # bridge being fed. Here we feed the bridge directly to test the DC
    # sleep reaction in isolation.
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    from research_extensions.bridge import (
        Observation,
        Prediction,
        SurpriseEvent,
    )

    bridge = registry.bridge
    bridge.surprises.append(
        SurpriseEvent(
            action="ACTION1",
            predicted=Prediction(
                action="ACTION1", expected_change_summary="nothing changes"
            ),
            observation=Observation(
                action="ACTION1",
                before_signature="a",
                after_signature="b",
                diff_magnitude=4,
            ),
            recent_trajectory=["ACTION1", "ACTION2", "ACTION1"],
        )
    )

    # A normal after_action call should drain the new surprise and trigger
    # the DC sleep cue machinery.
    before = _make_frame([[1, 2], [3, 4]])
    after = _make_frame([[1, 2], [3, 5]])
    registry.after_action(before, "ACTION1", after)

    assert dc._pending_sleep_cues
    assert any("Surprise" in cue or "surprise" in cue for cue in dc._pending_sleep_cues)


def test_dreamcoder_explicit_supersedes_revises_existing_skill(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "probe-map",
            "steps": ["ACTION1", "ACTION2"],
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[1]]))

    registry.record_agent_skill_proposal(
        {
            "name": "probe-map-v2",
            "supersedes": "probe-map",
            "steps": ["ACTION1", "ACTION2", "ACTION3"],
        }
    )
    registry.after_action(_make_frame([[1]]), "ACTION2", _make_frame([[2]]))

    assert len(dc.skills) == 1
    assert dc.skills[0].revision_count == 1
    assert dc.skills[0].payload["name"] == "probe-map-v2"


def test_dreamcoder_accepts_agent_named_skill_verbatim(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "whatever-agent-wants",
            "description": "agent's own words",
            "trigger": {"kind": "grid-pattern", "rows": [0, 1]},
            "body": ["ACTION1", "ACTION2"],
            "expected_outcome": "agent's own words",
        }
    )
    # after_action is the place where DC drains new proposals from the bridge.
    registry.after_action(
        _make_frame([[0]]), "ACTION1", _make_frame([[0]])
    )
    names = [r.payload["name"] for r in dc.skills]
    assert names == ["whatever-agent-wants"]
    assert dc.skills[0].depth == 0


def test_dreamcoder_overlay_prefers_observable_reusable_skills(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    overlay = registry.prompt_overlay()

    assert "observable behavior" in overlay
    assert "Avoid naming a skill after a level number" in overlay
    assert "revise or supersede" in overlay
    assert "local mechanic" not in overlay
    assert "include explicit action names" in overlay


def test_dreamcoder_preserves_longer_body_preview_and_reference_counts(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "seed-skill",
            "description": "base skill",
            "body": ["ACTION1", "ACTION2"],
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[0]]))

    long_body = [f"ACTION{i % 4 + 1}" for i in range(12)] + ["skill:seed-skill"]
    registry.record_agent_skill_proposal(
        {
            "name": "composed-skill",
            "description": "long skill",
            "body": long_body,
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION2", _make_frame([[0]]))

    overlay = dc.prompt_overlay()
    assert "ACTION4" in overlay
    assert "skill:seed-skill" in overlay
    seed = next(record for record in dc.skills if record.payload["name"] == "seed-skill")
    assert seed.times_referenced == 1


def test_dreamcoder_scores_skill_when_body_matches_observation_trajectory(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "two-step-pattern",
            "description": "captures two actions",
            "body": ["ACTION1", "ACTION2"],
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[1]]))
    registry.after_action(_make_frame([[1]]), "ACTION2", _make_frame([[2]]))

    skill = next(record for record in dc.skills if record.payload["name"] == "two-step-pattern")
    assert skill.observed_support >= 1
    assert skill.informative_hits >= 1
    assert skill.score() > 0


def test_dreamcoder_scores_string_body_with_action_tokens(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "string-body",
            "description": "body is free-form text",
            "body": "Try ACTION1 then ACTION2 if the pattern advances.",
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[1]]))
    registry.after_action(_make_frame([[1]]), "ACTION2", _make_frame([[2]]))

    skill = next(record for record in dc.skills if record.payload["name"] == "string-body")
    assert skill.observed_support >= 1


def test_dreamcoder_revises_similar_skill_instead_of_adding_duplicate(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {"name": "first", "body": ["ACTION1", "ACTION2"], "description": "seed"}
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[1]]))

    registry.record_agent_skill_proposal(
        {
            "name": "second",
            "rule": "ACTION1 then ACTION2 again with clearer wording",
            "description": "revision candidate",
            "revises": "first",
        }
    )
    registry.after_action(_make_frame([[1]]), "ACTION2", _make_frame([[2]]))

    assert len(dc.skills) == 1
    assert dc.skills[0].revision_count >= 1


def test_dreamcoder_keeps_distinct_skills_that_share_actions(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    registry.record_agent_skill_proposal(
        {
            "name": "edge lift",
            "body": "ACTION1 then ACTION2 to raise the left edge marker",
            "description": "left edge tactic",
        }
    )
    registry.after_action(_make_frame([[0]]), "ACTION1", _make_frame([[1]]))

    registry.record_agent_skill_proposal(
        {
            "name": "chamber shift",
            "body": "ACTION1 then ACTION2 to shift the upper chamber gate",
            "description": "upper chamber tactic",
        }
    )
    registry.after_action(_make_frame([[1]]), "ACTION2", _make_frame([[2]]))

    assert len(dc.skills) == 2


# -- World-model: no levels_completed read anywhere ----------------------
def test_world_model_records_transitions_without_level(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    before = _make_frame([[1, 1], [1, 1]])
    after = _make_frame([[1, 2], [1, 1]])
    registry.after_action(before, "ACTION1", after)

    state_sig = next(iter(wm.transitions.keys()))
    action_stats = wm.transitions[state_sig]["ACTION1"]
    assert action_stats["count"] == 1.0
    assert action_stats["avg_diff"] > 0


def test_current_grid_uses_last_layer_for_current_observation():
    frame = SimpleNamespace(
        frame=[
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
        ]
    )
    assert current_grid(frame) == [[2, 2], [2, 2]]


def test_current_grid_prefers_frame_wrapper_grid_attribute():
    frame = SimpleNamespace(
        grid=((3, 4), (5, 6)),
        frame=[
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]],
        ],
    )
    assert current_grid(frame) == [[3, 4], [5, 6]]


def test_world_model_signature_changes_for_lower_grid_change(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    top = [[0] * 16 for _ in range(16)]
    lower_a = [[0] * 16 for _ in range(16)]
    lower_b = [[0] * 16 for _ in range(16)]
    lower_b[-1][-1] = 7
    before = _make_frame(top + lower_a)
    after = _make_frame(top + lower_b)

    assert wm._signature(before) != wm._signature(after)


def test_world_model_emits_surprise_on_mismatch(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()

    before = _make_frame([[0, 0], [0, 0]])
    after_unchanged = _make_frame([[0, 0], [0, 0]])

    registry.record_agent_prediction("ACTION1", "EXPECT_CHANGE: cells will flip")
    registry.after_action(before, "ACTION1", after_unchanged)

    assert len(registry.bridge.surprises) == 1


def test_world_model_ignores_unstructured_prediction_for_surprise(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()

    before = _make_frame([[0, 0], [0, 0]])
    after_diff = _make_frame([[1, 0], [0, 0]])

    registry.record_agent_prediction("ACTION1", "maybe something moves")
    registry.after_action(before, "ACTION1", after_diff)

    assert len(registry.bridge.surprises) == 0


def test_world_model_scores_structured_prediction_dict(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()

    before = _make_frame([[0, 0], [0, 0]])
    after_diff = _make_frame([[1, 0], [0, 0]])

    registry.record_agent_prediction(
        "ACTION1",
        {"expect_change": True, "focus": "left edge", "note": "something should move"},
    )
    registry.after_action(before, "ACTION1", after_diff)

    assert len(registry.bridge.surprises) == 0


def test_world_model_normalizes_richer_prediction_heads(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()

    registry.record_agent_prediction(
        "ACTION1",
        {
            "expect_change": True,
            "focus": "left edge",
            "observation_prediction": "new_family",
            "progress_prediction": "branch_escape",
            "action_recommendation": "ACTION4",
            "recommendation_rationale": "best falsifier",
            "rival_predictions": ["same_family tail", "new_family detour"],
        },
    )

    prediction = registry.bridge.current_prediction
    assert prediction is not None
    assert prediction.observation_prediction == "new_family"
    assert prediction.progress_prediction == "branch_escape"
    assert prediction.action_recommendation == "ACTION4"
    assert prediction.rival_predictions == ["same_family tail", "new_family detour"]


def test_world_model_structured_prediction_mismatch_emits_surprise(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()

    before = _make_frame([[0, 0], [0, 0]])
    after_same = _make_frame([[0, 0], [0, 0]])

    registry.record_agent_prediction(
        "ACTION1",
        {"expect_change": True, "focus": "top row"},
    )
    registry.after_action(before, "ACTION1", after_same)

    assert len(registry.bridge.surprises) == 1


def test_world_model_accepts_markdown_world_update_and_scores_it(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    registry.record_agent_world_update(
        {
            "draft": "```python\nif action == 'ACTION1': expect change\n```",
            "focus": "ACTION1 local rule",
            "self_score": 3,
        }
    )
    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[1, 0], [0, 0]])
    registry.record_agent_prediction("ACTION1", {"expect_change": True, "focus": "left edge"})
    registry.after_action(before, "ACTION1", after)

    assert wm.world_drafts
    best = wm._best_draft()
    assert best is not None
    assert "```python" in best.body
    assert best.empirical_matches == 1
    assert best.simulator_matches >= 1
    assert best.score() > 3


def test_world_model_parses_markdown_score_and_focus():
    body, focus, score = WorldModelModule._parse_markdown_world_update(
        "score: 2.5\nfocus: left edge\n```python\nrule = 1\n```"
    )
    assert score == 2.5
    assert focus == "left edge"
    assert "```python" in body


def test_world_model_wraps_prose_update_with_code_scaffold():
    body, focus, score = WorldModelModule._normalize_world_update(
        "score: 1.5\nfocus: local sweep\nrule:\nprobe each action once\ntrack the largest diff"
    )
    assert score == 1.5
    assert focus == "local sweep"
    assert "probe each action once" in body
    assert "```python" in body
    assert "def predict_effect" in body


def test_world_model_can_override_with_best_known_action(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    frame = _make_frame([[0, 0], [0, 0]])
    sig = wm._signature(frame)
    wm.transitions[sig] = {
        "ACTION1": {"count": 2.0, "avg_diff": 20.0, "zero_count": 0.0},
        "ACTION2": {"count": 2.0, "avg_diff": 1.0, "zero_count": 1.0},
    }
    chosen = wm.before_action(frame, "ACTION2", ["ACTION1", "ACTION2"])
    assert chosen == "ACTION1"
    assert wm._latest_action_hint is not None
    assert "ACTION1" in wm._latest_action_hint
    assert wm.prior_suggestions == 1
    assert wm.prior_overrides == 1
    assert wm.recent_prior_decisions[-1]["chosen"] == "ACTION1"


def test_world_model_prefers_branch_escape_over_high_diff_same_family(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    frame = _make_frame([[0, 0], [0, 0]])
    sig = wm._signature(frame)
    wm.transitions[sig] = {
        "ACTION1": {
            "count": 5.0,
            "avg_diff": 50.0,
            "zero_count": 0.0,
            "new_signature_count": 0.0,
            "new_family_count": 0.0,
            "branch_escape_count": 0.0,
            "falsify_count": 0.0,
            "support_count": 5.0,
            "repeat_depth_sum": 15.0,
        },
        "ACTION2": {
            "count": 3.0,
            "avg_diff": 4.0,
            "zero_count": 0.0,
            "new_signature_count": 2.0,
            "new_family_count": 2.0,
            "branch_escape_count": 2.0,
            "falsify_count": 1.0,
            "support_count": 0.0,
            "repeat_depth_sum": 0.0,
        },
    }

    chosen = wm.before_action(frame, "ACTION1", ["ACTION1", "ACTION2"])
    assert chosen == "ACTION2"
    assert "branch_escape" in (wm._latest_action_hint or "")


def test_world_model_avoids_reset_when_probe_remains(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    frame = _make_frame([[0, 0], [0, 0]])
    sig = wm._signature(frame)
    wm.transitions[sig] = {
        "ACTION1": {"count": 2.0, "avg_diff": 0.0, "zero_count": 2.0},
    }

    chosen = wm.before_action(frame, "RESET", ["ACTION1", "ACTION2"])
    assert chosen == "ACTION2"
    assert wm.prior_overrides == 1
    assert "Avoid RESET" in (wm._latest_action_hint or "")


def test_world_model_can_use_nearest_bucket_for_action_prior(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True, params={"generalize_max_distance": 1.0}))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    seen_frame = _make_frame([[0, 0], [0, 0]])
    target_frame = _make_frame([[0, 0], [0, 1]])
    seen_sig = wm._signature(seen_frame)
    wm.signature_features[seen_sig] = wm._features(seen_frame)
    wm.transitions[seen_sig] = {
        "ACTION1": {
            "count": 3.0,
            "avg_diff": 8.0,
            "zero_count": 0.0,
            "new_signature_count": 2.0,
            "new_family_count": 1.0,
            "branch_escape_count": 1.0,
            "falsify_count": 0.0,
            "support_count": 2.0,
            "repeat_depth_sum": 0.0,
        }
    }

    chosen = wm.before_action(target_frame, "ACTION2", ["ACTION1", "ACTION2"])

    assert chosen == "ACTION1"
    assert wm.prior_suggestions == 1
    assert wm.prior_overrides == 1
    assert "approximate" in (wm._latest_action_hint or "")


def test_world_model_force_novelty_probe_on_same_family_attractor(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    frame = _make_frame([[0, 0], [0, 0]])
    sig = wm._signature(frame)
    wm.transitions[sig] = {
        "ACTION1": {
            "count": 3.0,
            "avg_diff": 9.0,
            "zero_count": 0.0,
            "new_signature_count": 1.0,
            "new_family_count": 0.0,
            "branch_escape_count": 0.0,
            "falsify_count": 0.0,
            "support_count": 3.0,
            "repeat_depth_sum": 12.0,
        }
    }
    wm.recent_outcomes = [("ACTION1", 2.0)] * 4
    wm.bridge.recent_families = ["f:same"] * 4

    chosen = wm.before_action(frame, "ACTION1", ["ACTION1", "ACTION2", "ACTION3"])

    assert chosen in {"ACTION2", "ACTION3"}
    assert chosen != "ACTION1"
    assert "force a novelty probe" in (wm._latest_action_hint or "")


def test_world_model_persists_verification_counters(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    registry.record_agent_world_update(
        "score: 3\nfocus: edge shift\n```python\nif action == 'ACTION1': return 'change'\n```"
    )
    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[1, 0], [0, 0]])
    registry.record_agent_prediction("ACTION1", {"expect_change": True, "focus": "left"})
    registry.after_action(before, "ACTION1", after)
    wm.before_action(before, "ACTION1", ["ACTION1", "ACTION2"])
    registry.on_run_end(history=[], finish_status=None)

    saved = json.loads((registry.context.shared_dir / "world_model.json").read_text())
    assert saved["structured_predictions"] == 1
    assert saved["scored_prediction_matches"] == 1
    assert "recent_prior_decisions" in saved


def test_world_model_applies_posthoc_score_to_late_draft(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[1, 0], [0, 0]])
    registry.record_agent_prediction("ACTION1", {"expect_change": True, "focus": "left"})
    registry.after_action(before, "ACTION1", after)
    registry.record_agent_world_update("focus: late draft\nrule:\nexpect change after ACTION1")

    best = wm._best_draft()
    assert best is not None
    assert best.empirical_matches == 1
    assert "```python" in best.body


def test_world_model_auto_seeds_draft_after_observations(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[0, 0], [0, 0]])
    for action in ("ACTION1", "ACTION2", "ACTION3"):
        registry.record_agent_prediction(action, {"expect_change": True})
        registry.after_action(before, action, after)

    best = wm._best_draft()
    assert best is not None
    assert "```python" in best.body
    assert "def predict_effect" in best.body
    assert "expected_diff_band" in best.body


def test_world_model_simulator_contract_scores_without_agent_prediction(tmp_path: Path):
    cfg = ResearchConfig(world_model=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    wm = registry.active()["world_model"]

    registry.record_agent_world_update(
        {
            "draft": (
                "```python\n"
                "def predict_effect(action, observation):\n"
                "    return {'expect_change': True, 'expected_diff_band': 'small'}\n"
                "```"
            ),
            "focus": "test simulator",
            "self_score": 1.0,
        }
    )
    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[1, 0], [0, 0]])
    registry.after_action(before, "ACTION1", after)

    best = wm._best_draft()
    assert best is not None
    assert best.simulator_matches == 1


def test_world_model_prompt_overlay_includes_top_dreamcoder_skills(tmp_path: Path):
    cfg = ResearchConfig(
        dreamcoder=ModuleConfig(enabled=True),
        world_model=ModuleConfig(enabled=True),
    )
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]
    wm = registry.active()["world_model"]

    dc.record_agent_proposal(
        {
            "name": "edge confirm then commit",
            "description": "Short grounded opening with recovery.",
            "precondition": "fresh board",
            "controller": ["ACTION3", "ACTION4"],
            "expected_effect": "checkpoint",
            "body": "ACTION3 ACTION4 ACTION1",
        }
    )
    dc.after_action(None, "ACTION3", None)

    overlay = wm.prompt_overlay()
    assert "Top reusable skills currently in library:" in overlay
    assert "edge confirm then commit" in overlay


def test_dreamcoder_prompt_and_routing_use_world_model_hints(tmp_path: Path):
    cfg = ResearchConfig(
        dreamcoder=ModuleConfig(enabled=True),
        world_model=ModuleConfig(enabled=True),
    )
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]
    wm = registry.active()["world_model"]

    wm.bridge.update_hint(
        "world_model",
        {
            "suggested_action": "ACTION2",
            "hint": "ACTION2 is locally strongest here.",
            "best_focus": "left corridor transition",
            "best_score": 2.0,
            "recent_surprise": True,
        },
    )
    overlay = dc.prompt_overlay()
    assert "[World-model cues]" in overlay
    assert "ACTION2" in overlay
    assert "left corridor transition" in overlay

    dc.record_agent_proposal(
        {
            "name": "world-aligned continuation",
            "description": "Follows the world prior when the same corridor is active.",
            "body": "ACTION1 ACTION2 ACTION4",
        }
    )
    dc.record_agent_proposal(
        {
            "name": "misaligned continuation",
            "description": "Alternative branch with the same prefix length.",
            "body": "ACTION1 ACTION3 ACTION4",
        }
    )
    dc.after_action(None, "ACTION1", None)
    registry.bridge.recent_actions = ["ACTION1"]

    chosen = dc.before_action(None, "RESET", ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert chosen == "ACTION2"


def test_dreamcoder_routes_next_action_from_skill_sequence(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Probe pair then continue",
            "description": "Use a short two-step prefix before continuing.",
            "body": "ACTION1 ACTION2 ACTION3",
        }
    )
    dc.after_action(None, "ACTION1", None)
    registry.bridge.recent_actions = ["ACTION1", "ACTION2"]

    chosen = dc.before_action(
        None,
        "ACTION4",
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
    )
    assert chosen == "ACTION3"
    assert "Probe pair then continue" in (dc._latest_route_hint or "")


def test_dreamcoder_intercepts_reset_when_skill_can_continue(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Reset-free continuation",
            "description": "Continue with an available action instead of resetting.",
            "body": "ACTION2 ACTION4",
        }
    )
    dc.after_action(None, "ACTION2", None)
    registry.bridge.recent_actions = ["ACTION2"]

    chosen = dc.before_action(None, "RESET", ["ACTION2", "ACTION4"])
    assert chosen == "ACTION4"
    assert "Avoid RESET" in (dc._latest_route_hint or "")


def test_dreamcoder_demotes_exact_spine_without_frontier_evidence(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Exact corridor spine",
            "description": "Repeated exact opening.",
            "body": ["ACTION1", "ACTION2", "ACTION3"],
            "action_spine": ["ACTION1", "ACTION2", "ACTION3"],
        }
    )
    dc.record_agent_proposal(
        {
            "name": "Branch escape wrapper",
            "description": "Leave the repeated family after the shared prefix.",
            "precondition": "same opening but repeated-family pressure is high",
            "controller": ["ACTION1", "ACTION4"],
            "expected_effect": "branch escape into a new family",
            "body": ["ACTION1", "ACTION4"],
        }
    )
    dc.after_action(None, "ACTION1", None)
    registry.bridge.recent_actions = ["ACTION1"]

    exact = dc._find_by_name("Exact corridor spine")
    branch = dc._find_by_name("Branch escape wrapper")
    assert exact is not None
    assert branch is not None
    exact.observed_support = 10
    exact.informative_hits = 10
    branch.novel_family_hits = 2
    branch.branch_escape_hits = 2
    branch.falsification_utility_hits = 1

    chosen = dc.before_action(None, "ACTION2", ["ACTION2", "ACTION3", "ACTION4"])
    assert chosen == "ACTION4"


def test_dreamcoder_reports_skill_usage_metadata(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Usage tracked sequence",
            "description": "Track when this skill is selected.",
            "body": "ACTION1 ACTION2",
        }
    )
    dc.after_action(None, "ACTION1", None)
    registry.bridge.recent_actions = ["ACTION1"]
    chosen = dc.before_action(None, "RESET", ["ACTION1", "ACTION2"])

    assert chosen == "ACTION2"
    listed = dc.list_skills()
    assert listed[0]["selected"] >= 1
    assert listed[0]["reset_saves"] >= 1
    detail = dc.read_skill("Usage tracked sequence")
    assert detail is not None
    assert detail["times_selected"] >= 1
    assert detail["reset_intercepts"] >= 1


def test_dreamcoder_rejects_ungrounded_skill_proposals(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "abstract sweep",
            "description": "Try available actions carefully.",
            "body": "check status, try each action, compare history",
        }
    )
    dc.after_action(None, "ACTION1", None)

    assert dc.list_skills() == []


def test_dreamcoder_attaches_recent_action_spine_to_abstract_skill(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]
    registry.bridge.recent_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

    dc.record_agent_proposal(
        {
            "name": "available sweep",
            "description": "Sweep available actions and compare outcomes.",
            "body": "try all available actions and repeat the best one",
        }
    )
    dc.after_action(None, "ACTION4", None)

    listed = dc.list_skills()
    assert len(listed) == 1
    assert "action_spine" in listed[0]["body_preview"]
    assert "ACTION1" in listed[0]["body_preview"]
    detail = dc.read_skill("available sweep")
    assert detail is not None
    assert detail["payload"]["action_spine"] == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]


def test_dreamcoder_accepts_embedded_action_refs_inside_list_body(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Natural language chain",
            "description": "A reusable chain described in natural language.",
            "body": [
                "Press ACTION1 once to probe the corridor.",
                "Then use ACTION2 if the first probe moved the marker.",
            ],
        }
    )
    dc.after_action(None, "ACTION1", None)
    registry.bridge.recent_actions = ["ACTION1"]

    listed = dc.list_skills()
    assert len(listed) == 1
    assert "ACTION1" in listed[0]["body_preview"]
    chosen = dc.before_action(None, "RESET", ["ACTION1", "ACTION2"])
    assert chosen == "ACTION2"


def test_dreamcoder_prefers_action_spine_over_generic_body_for_routing(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Concrete spine beats prose",
            "description": "Generic prose should not override a proven short sequence.",
            "trigger": "Need informative probe from uncertain state with ACTION2 and ACTION3 available",
            "body": "ACTION2, ACTION3, ACTION1, ACTION4",
            "action_spine": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
        }
    )
    dc.after_action(None, "ACTION1", None)

    registry.bridge.recent_actions = ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1"]
    chosen = dc.before_action(None, "RESET", ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    assert chosen == "ACTION2"


def test_dreamcoder_does_not_immediately_restart_completed_exact_spine(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    spine = ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"]
    dc.record_agent_proposal(
        {
            "name": "Validated exact spine ACTION3-ACTION3-ACTION4-ACTION1-ACTION1-ACTION2",
            "description": "Exact action spine preserved from a validated successful run.",
            "body": spine,
            "action_spine": spine,
        }
    )
    dc.after_action(None, "ACTION2", None)
    registry.bridge.recent_actions = list(spine)

    chosen = dc.before_action(None, "ACTION4", ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert chosen != "ACTION3"


def test_dreamcoder_support_matching_ignores_trigger_only_action_mentions(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Trigger mentions actions but body is concrete elsewhere",
            "trigger": "When ACTION2 then ACTION3 seem promising",
            "description": "A note about ACTION2 and ACTION3 in prose.",
            "body": ["ACTION4", "ACTION1"],
        }
    )
    before = _make_frame([[0, 0], [0, 0]])
    after = _make_frame([[1, 0], [0, 0]])
    dc.after_action(before, "ACTION2", after)
    dc.after_action(before, "ACTION3", after)

    detail = dc.read_skill("Trigger mentions actions but body is concrete elsewhere")
    assert detail is not None
    assert detail["observed_support"] == 0


def test_dreamcoder_progressive_spine_outranks_higher_score_generic_probe(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "Generic probe",
            "body": "ACTION2, ACTION3, ACTION1, ACTION4",
        }
    )
    dc.after_action(None, "ACTION1", None)
    dc.skills[0].observed_support = 20
    dc.skills[0].informative_hits = 20

    dc.record_agent_proposal(
        {
            "name": "Exact winning spine",
            "body": "ACTION2, ACTION3, ACTION1, ACTION4",
            "action_spine": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
        }
    )
    dc.after_action(None, "ACTION2", None)

    registry.bridge.recent_actions = ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1"]
    chosen = dc.before_action(None, "ACTION2", ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])

    assert chosen == "ACTION2"
    detail = dc.read_skill("Exact winning spine")
    assert detail is not None
    assert detail["times_selected"] >= 1


def test_dreamcoder_consolidation_dedupes_identical_execution_signatures(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    dc.record_agent_proposal(
        {
            "name": "First exact sequence",
            "body": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
        }
    )
    dc.after_action(None, "ACTION2", None)
    dc.skills[0].observed_support = 5

    dc.record_agent_proposal(
        {
            "name": "Second paraphrase same sequence",
            "description": "Same behavior, different wording.",
            "body": "ACTION3 ACTION3 ACTION4 ACTION1 ACTION1 ACTION2",
            "action_spine": ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"],
        }
    )
    dc.after_action(None, "ACTION2", None)
    dc.skills[-1].observed_support = 2

    dc._consolidate_and_prune()

    names = [skill["name"] for skill in dc.list_skills()]
    assert len(names) == 1
    assert names == ["Second paraphrase same sequence"]


def test_dreamcoder_promotes_validated_exact_spine_on_win(tmp_path: Path):
    cfg = ResearchConfig(dreamcoder=ModuleConfig(enabled=True))
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    dc = registry.active()["dreamcoder"]

    history = [
        ("RESET", _make_frame([[0]])),
        ("ACTION3", _make_frame([[0]])),
        ("ACTION3", _make_frame([[0]])),
        ("ACTION4", _make_frame([[0]])),
        ("ACTION1", _make_frame([[0]])),
        ("ACTION1", _make_frame([[0]])),
        ("ACTION2", _make_frame([[0]])),
    ]
    finish_status = SimpleNamespace(status="win", reason="validated", levels_completed=(1, 7))

    dc.on_run_end(history=history, finish_status=finish_status, workdir=tmp_path)

    skills = dc.list_skills()
    assert len(skills) == 2
    assert skills[0]["name"].startswith("Validated exact spine")
    detail = dc.read_skill(skills[0]["name"])
    assert detail is not None
    assert detail["payload"]["action_spine"] == ["ACTION3", "ACTION3", "ACTION4", "ACTION1", "ACTION1", "ACTION2"]
    wrapper = next(skill for skill in skills if skill["name"].startswith("Opening wrapper for"))
    wrapper_detail = dc.read_skill(wrapper["name"])
    assert wrapper_detail is not None
    assert wrapper_detail["payload"]["subskills"] == [f"skill:{skills[0]['name']}"]
    assert wrapper_detail["payload"]["controller"]


def test_parse_action_extracts_world_update_side_channel():
    agent = ArcgenticaResearch.__new__(ArcgenticaResearch)
    response = (
        '{"action":"ACTION2","world_update":"```python\\nrule = \\"test\\"\\n```",'
        '"propose_skill":{"name":"x"}}'
    )
    action, side = agent._parse_action(response, [1, 2, 3, 4])
    assert action == "ACTION2"
    assert "world_update" in side
    assert "```python" in side["world_update"]


def test_meta_harness_scores_run_without_levels(tmp_path: Path):
    cfg = ResearchConfig(
        world_model=ModuleConfig(enabled=True),
        meta_harness=ModuleConfig(enabled=True),
    )
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    mh = registry.active()["meta_harness"]

    before = _make_frame([[0, 0]])
    after_same = _make_frame([[0, 0]])
    after_diff = _make_frame([[1, 0]])

    registry.record_agent_prediction(
        "ACTION1",
        {"expect_change": True, "progress_prediction": "falsify"},
    )
    registry.after_action(before, "ACTION1", after_same)
    for action, nxt in [
        ("ACTION2", after_diff),
        ("ACTION3", after_diff),
        ("RESET", after_same),
    ]:
        registry.after_action(before, action, nxt)

    registry.on_run_end(history=[], finish_status=None)

    assert mh.run_history
    metrics = mh.run_history[-1]["metrics"]
    assert "unique_actions" in metrics
    assert metrics["unique_actions"] == 3
    assert metrics["resets"] == 1
    assert "new_signature_count" in metrics
    assert "new_family_count" in metrics
    assert "falsification_probe_count" in metrics
    assert "repeated_family_penalty" in metrics
    assert not any("level" in k for k in metrics)


def test_all_three_modules_together_share_bridge(tmp_path: Path):
    cfg = ResearchConfig(
        dreamcoder=ModuleConfig(enabled=True),
        world_model=ModuleConfig(enabled=True),
        meta_harness=ModuleConfig(enabled=True),
    )
    registry = ModuleRegistry(cfg, _make_context(tmp_path))
    registry.load()
    assert set(registry.active().keys()) == {
        "dreamcoder",
        "world_model",
        "meta_harness",
    }

    assert registry.bridge is registry.active()["dreamcoder"].bridge
    assert registry.bridge is registry.active()["world_model"].bridge
    assert registry.bridge is registry.active()["meta_harness"].bridge


def test_verification_helpers_flag_expected_artifacts(tmp_path: Path):
    shared = tmp_path / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    (shared / "dreamcoder_library.json").write_text(
        json.dumps(
            [
                {
                    "payload": {
                        "name": "edge-nudge",
                        "policy": "Try ACTION1, then ACTION2 if the block does not move.",
                        "precondition": "edge contact is visible",
                        "expected_effect": "either the block shifts or the fallback probe fires",
                        "description": "Reusable local shift probe.",
                    },
                    "revision_count": 1,
                    "times_linked_to_surprise": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    (shared / "world_model.json").write_text(
        json.dumps(
            {
                "transitions": {
                    "sig": {
                        "ACTION1": {
                            "count": 2,
                            "avg_diff": 4,
                            "zero_count": 0,
                        }
                    }
                },
                "world_drafts": [
                    {
                        "body": "score: 4\nfocus: edge\n```python\nif action == 'ACTION1': return 'change'\n```",
                        "revisions": 2,
                        "empirical_matches": 1,
                        "empirical_mismatches": 0,
                        "simulator_matches": 1,
                        "simulator_mismatches": 0,
                    }
                ],
                "prior_suggestions": 1,
                "prior_overrides": 1,
                "structured_predictions": 1,
                "scored_prediction_matches": 1,
                "scored_prediction_mismatches": 0,
            }
        ),
        encoding="utf-8",
    )

    (shared / "meta_harness.json").write_text(
        json.dumps(
            {
                "active_keys": ["stay_curious"],
                "run_history": [
                    {
                        "metrics": {
                            "unique_actions": 3,
                            "resets": 0,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dc = analyze_dreamcoder_state(shared / "dreamcoder_library.json")
    wm = analyze_world_model_state(shared / "world_model.json")
    mh = analyze_meta_harness_state(shared / "meta_harness.json")
    prompt = verify_prompt_contract()

    assert dc["ok"] is True
    assert wm["ok"] is True
    assert mh["ok"] is True
    assert prompt["ok"] is True

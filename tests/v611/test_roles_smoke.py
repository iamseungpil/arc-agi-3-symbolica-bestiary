"""v611_roles.py import + structure smoke (no real LLM call)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_v611_roles_imports_without_error():
    from agents.templates.agentica_lite import v611_roles
    assert hasattr(v611_roles, "run_m1_proposer")
    assert hasattr(v611_roles, "run_m2v_verifier")
    assert hasattr(v611_roles, "run_m2e_executor")
    assert hasattr(v611_roles, "_extract_json_block")
    assert callable(v611_roles.run_m1_proposer)


def test_prompts_loaded_at_import():
    """All 3 system prompts must be loaded into module globals."""
    from agents.templates.agentica_lite import v611_roles
    assert len(v611_roles._M1_SYSTEM) > 500
    assert "Proposer" in v611_roles._M1_SYSTEM
    assert len(v611_roles._M2V_SYSTEM) > 500
    assert "Verifier" in v611_roles._M2V_SYSTEM or \
        "Consistency Critic" in v611_roles._M2V_SYSTEM
    assert len(v611_roles._M2E_SYSTEM) > 500
    assert "Executor" in v611_roles._M2E_SYSTEM or \
        "Visual Grounder" in v611_roles._M2E_SYSTEM


def test_extract_json_block_handles_clean():
    from agents.templates.agentica_lite import v611_roles
    text = '{"verdict": "approve", "reason_nl": "ok"}'
    assert v611_roles._extract_json_block(text) == text


def test_extract_json_block_strips_markdown_fence():
    from agents.templates.agentica_lite import v611_roles
    text = '```json\n{"verdict": "approve"}\n```'
    out = v611_roles._extract_json_block(text)
    assert out.strip().startswith("{")
    assert out.strip().endswith("}")


def test_extract_json_block_handles_wrapped_text():
    from agents.templates.agentica_lite import v611_roles
    text = ("Here is the response:\n"
            '{"verdict": "approve", "reason_nl": "ok"}\n'
            "End of response.")
    out = v611_roles._extract_json_block(text)
    import json
    parsed = json.loads(out)
    assert parsed["verdict"] == "approve"


def test_extract_json_block_empty_returns_empty_object():
    from agents.templates.agentica_lite import v611_roles
    assert v611_roles._extract_json_block("") == "{}"
    assert v611_roles._extract_json_block(None) == "{}"


def test_extract_json_block_handles_nested():
    from agents.templates.agentica_lite import v611_roles
    text = '{"a": {"b": {"c": 1}}, "d": [1, 2]}'
    out = v611_roles._extract_json_block(text)
    import json
    parsed = json.loads(out)
    assert parsed["a"]["b"]["c"] == 1
    assert parsed["d"] == [1, 2]


def test_role_runner_signatures_match_orchestrator_protocols():
    """The 3 role runners must match the v611_orchestrator Protocol
    callable signatures (matched via mock-substitutability)."""
    from agents.templates.agentica_lite import v611_roles
    from agents.templates.agentica_lite.v611_orchestrator import (
        AnchorCounter, run_v611_turn,
    )

    # We do NOT actually call the LLM — we substitute mocks but verify
    # that the protocol signatures are compatible.
    # Specifically, the real runners should accept the kwargs the
    # orchestrator passes.

    import inspect
    sig_m1 = inspect.signature(v611_roles.run_m1_proposer)
    assert "state_text" in sig_m1.parameters
    assert "skill_md_summary" in sig_m1.parameters
    assert "anchor_summary" in sig_m1.parameters
    assert "rejection_reason" in sig_m1.parameters

    sig_m2v = inspect.signature(v611_roles.run_m2v_verifier)
    assert "proposer_out" in sig_m2v.parameters
    assert "state_text_summary" in sig_m2v.parameters

    sig_m2e = inspect.signature(v611_roles.run_m2e_executor)
    assert "approved_out" in sig_m2e.parameters
    assert "png_bytes" in sig_m2e.parameters

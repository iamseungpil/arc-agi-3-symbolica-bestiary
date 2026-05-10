"""Per-module unit tests for proposer.py.

Plan v602 §11 addendum: 10 critical branch tests.

Two-layer mocking strategy (codex final-gate refinement):
  - Layer 1 (validator-isolation): test parse_and_validate(raw_dict, visible_rids)
    directly. 6 schema-violation branches + 1 happy path.
  - Layer 2 (mocked async LLM-call): patch openai/azure imports to simulate
    timeout / parse-error / no-client failure modes. 3 tests.

Branches under test:
  1. schema valid full (Layer 1 happy path)
  2. schema missing-field
  3. schema extra-field tolerated (raw kept, no rejection)
  4. confidence-OOR (out of range)
  5. predicate-blacklisted
  6. region-unknown
  7. tool-call-blocked
  8. parse-error (Layer 2: malformed JSON)
  9. timeout (Layer 2: asyncio.TimeoutError)
  10. llm-no-client (Layer 2: ImportError on azure deps)
"""

from __future__ import annotations

import asyncio
import builtins
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.proposer import (  # noqa: E402
    Proposer, ProposerOutput, ProposerResult, parse_and_validate,
)


# ============================================================================
# Layer 1: validator-isolation tests (no LLM mock)
# ============================================================================

def _full_valid_payload(region_id: str = "R36") -> dict:
    return {
        "candidate_predicate_id": "P_saturation_progress",
        "region_hint": region_id,
        "expected_signature": {"level_delta": 1},
        "required_pre_state": {
            "marker_id": "M0",
            "saturation_threshold": 7,
            "saturation_denominator": 8,
        },
        "confidence": 0.7,
        "thought": "saturation(M) approaches 1.0",
    }


# ---------- 1. schema valid-full ---------------------------------------------

def test_schema_valid_full_happy_path():
    """Full valid payload validates and returns ProposerOutput."""
    raw = _full_valid_payload()
    res = parse_and_validate(raw, visible_region_ids=["R36", "R12"])
    assert res.failure_reason is None
    assert res.output is not None
    assert isinstance(res.output, ProposerOutput)
    assert res.output.candidate_predicate_id == "P_saturation_progress"
    assert res.output.region_hint == "R36"
    assert res.output.confidence == 0.7
    assert res.output.required_pre_state["marker_id"] == "M0"


# ---------- 2. schema missing-field -------------------------------------------

def test_schema_missing_field_rejected():
    """Missing required field -> schema_invalid + schema_missing_field."""
    raw = _full_valid_payload()
    del raw["region_hint"]
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "schema_missing_field"
    assert res.output is None


def test_schema_missing_pre_state_subfield_rejected():
    """required_pre_state missing marker_id -> schema_missing_field."""
    raw = _full_valid_payload()
    del raw["required_pre_state"]["marker_id"]
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "schema_missing_field"


# ---------- 3. extra-field tolerated -----------------------------------------

def test_schema_extra_field_tolerated():
    """Extra (unknown) fields don't reject; passed through in raw dict."""
    raw = _full_valid_payload()
    raw["extra_field_unknown"] = "bonus"
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason is None
    assert res.output is not None
    # raw payload retains the extra field
    assert res.output.raw.get("extra_field_unknown") == "bonus"


# ---------- 4. confidence out-of-range ---------------------------------------

def test_schema_confidence_out_of_range_rejected():
    """confidence > 1.0 (or non-numeric) -> confidence_out_of_range."""
    raw = _full_valid_payload()
    raw["confidence"] = 1.7
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "confidence_out_of_range"
    # also non-numeric
    raw["confidence"] = "high"
    res2 = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res2.schema_error_code == "confidence_out_of_range"


# ---------- 5. predicate blacklisted -----------------------------------------

def test_schema_predicate_blacklisted():
    """candidate_predicate_id in blacklist (submit_action / click / ...) rejected."""
    raw = _full_valid_payload()
    raw["candidate_predicate_id"] = "submit_action"
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "predicate_blacklisted"


# ---------- 6. region unknown -------------------------------------------------

def test_schema_region_unknown_rejected():
    """region_hint absent from visible_region_ids list rejected."""
    raw = _full_valid_payload(region_id="R_NONEXISTENT")
    res = parse_and_validate(raw, visible_region_ids=["R36", "R12"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "region_unknown"


# ---------- 7. tool-call blocked ----------------------------------------------

def test_schema_tool_call_blocked():
    """Embedded tool_calls field is rejected as tool_call_blocked."""
    raw = _full_valid_payload()
    raw["tool_calls"] = [{"name": "submit_action"}]
    res = parse_and_validate(raw, visible_region_ids=["R36"])
    assert res.failure_reason == "schema_invalid"
    assert res.schema_error_code == "tool_call_blocked"


# ============================================================================
# Layer 2: mocked async LLM-call path tests
# ============================================================================

# ---------- 8. parse-error on inner exception --------------------------------

def test_proposer_parse_error_when_inner_raises():
    """If _inner raises an exception, propose() returns failure_reason='parse_error'."""
    p = Proposer(llm_timeout_s=5.0)

    async def _bad_inner(state, ids):
        raise RuntimeError("simulated upstream failure")

    p._inner = _bad_inner  # type: ignore[assignment]
    res = asyncio.run(p.propose({}, ["R0"]))
    assert res.failure_reason == "parse_error"
    assert res.output is None


# ---------- 9. timeout --------------------------------------------------------

def test_proposer_timeout_returns_timeout_reason():
    """If _inner exceeds llm_timeout_s, propose() returns failure_reason='timeout'."""
    p = Proposer(llm_timeout_s=0.05)

    async def _slow_inner(state, ids):
        await asyncio.sleep(1.0)
        return ProposerResult(output=None, failure_reason=None)

    p._inner = _slow_inner  # type: ignore[assignment]
    res = asyncio.run(p.propose({}, ["R0"]))
    assert res.failure_reason == "timeout"


# ---------- 10. llm-no-client (azure deps unavailable) -----------------------

def test_proposer_llm_no_client_when_azure_deps_missing(monkeypatch):
    """If azure.identity / openai imports fail, _inner returns llm_no_client."""
    real_import = builtins.__import__

    def _block_azure(name, *args, **kwargs):
        if name.startswith("azure") or name == "openai":
            raise ImportError(f"simulated missing dep: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_azure)
    p = Proposer(llm_timeout_s=5.0)
    res = asyncio.run(p.propose({}, ["R0"]))
    assert res.failure_reason == "llm_no_client"
    assert res.output is None


# ---------- 11. propose() success path with mocked _inner --------------------

def test_proposer_propose_returns_validated_output_via_mocked_inner():
    """Layer-2 happy path: propose() forwards _inner's ProposerResult untouched
    on success."""
    p = Proposer(llm_timeout_s=5.0)
    expected_out = ProposerOutput(
        candidate_predicate_id="P_saturation_progress",
        region_hint="R0",
        expected_signature={"level_delta": 1},
        required_pre_state={"marker_id": "M0", "saturation_threshold": 0,
                            "saturation_denominator": 4},
        confidence=0.85,
        thought="cited Step-0",
    )

    async def _success_inner(state, ids):
        return ProposerResult(output=expected_out, failure_reason=None)

    p._inner = _success_inner  # type: ignore[assignment]
    res = asyncio.run(p.propose({}, ["R0"]))
    assert res.failure_reason is None
    assert res.output is expected_out

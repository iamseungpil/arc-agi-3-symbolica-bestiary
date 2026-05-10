"""Per-module unit tests for reflector.py.

Plan v602 §11 addendum: 6 critical branch tests.

Branches under test:
  1. structured-patch-emit (parse_response builds ArmKey-keyed boost dict)
  2. invalid-arm-dropped (malformed key strings filtered out silently)
  3. text-truncation-256 (reflexion_text truncated to 256 chars)
  4. boost-clamped-0-0.3 (numeric boost values clamped into [0, 0.3])
  5. timeout (asyncio.wait_for raises TimeoutError -> reflect returns None)
  6. parse-error (inner raises -> reflect returns None)

We exercise the module-level _parse_response helper directly (Layer-1
isolation) for branches 1-4, and patch the Reflector._inner async hook for
branches 5-6 to test the timeout/exception paths without touching the live
TRAPI client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.predicate_posterior import ArmKey  # noqa: E402
from agents.templates.agentica_lite.reflector import (  # noqa: E402
    Reflector, ReflectorOutput, _parse_response,
)


# ---------- 1. structured-patch-emit -----------------------------------------

def test_structured_patch_emits_arm_key_dict():
    """Valid Reflector response builds ArmKey-keyed exploration_boost dict."""
    raw = {
        "discriminator_features": ["marker_M0_compass_saturation_numerator"],
        "reflexion_text": "Saturation drift suggests near_complete arm.",
        "suggested_exploration_boost": {
            "P12_saturation_progress|R36|near_complete": 0.2,
            "P03_sector_alignment|R36": 0.05,
        },
    }
    out = _parse_response(raw)
    assert out.discriminator_features == ["marker_M0_compass_saturation_numerator"]
    assert "Saturation drift" in out.reflexion_text
    # 3-part key parses with saturation_status
    assert out.suggested_exploration_boost[ArmKey("P12_saturation_progress", "R36", "near_complete")] == 0.2
    # 2-part key defaults to "n/a"
    assert out.suggested_exploration_boost[ArmKey("P03_sector_alignment", "R36")] == 0.05


# ---------- 2. invalid-arm-dropped -------------------------------------------

def test_invalid_arm_keys_dropped():
    """Malformed boost keys (wrong number of pipe-separated parts) silently dropped."""
    raw = {
        "discriminator_features": [],
        "reflexion_text": "",
        "suggested_exploration_boost": {
            "single_part_no_pipes": 0.1,        # 1 part, invalid
            "p|r|s|extra|tail": 0.1,             # 5 parts, invalid
            "P03|R36": 0.15,                      # valid
            "P12|R36|near_complete": 0.25,        # valid
        },
    }
    out = _parse_response(raw)
    keys = list(out.suggested_exploration_boost.keys())
    assert len(keys) == 2  # only the 2 valid ones
    assert ArmKey("P03", "R36") in out.suggested_exploration_boost
    assert ArmKey("P12", "R36", "near_complete") in out.suggested_exploration_boost


# ---------- 3. text-truncation-256 ------------------------------------------

def test_reflexion_text_truncated_to_256():
    """reflexion_text exceeding 256 chars is truncated to first 256."""
    long_text = "A" * 1024
    raw = {"discriminator_features": [], "reflexion_text": long_text,
           "suggested_exploration_boost": {}}
    out = _parse_response(raw)
    assert len(out.reflexion_text) == 256
    assert out.reflexion_text == "A" * 256


# ---------- 4. boost-clamped-0-0.3 -------------------------------------------

def test_boost_values_clamped_to_zero_zero_three():
    """Numeric boost values clamped into [0, 0.3]; non-numeric dropped."""
    raw = {
        "discriminator_features": [],
        "reflexion_text": "",
        "suggested_exploration_boost": {
            "P01|R01": -0.5,         # below floor -> 0.0
            "P02|R02": 0.7,          # above ceil -> 0.3
            "P03|R03": 0.15,         # in-range
            "P04|R04": "not_a_num",  # non-numeric -> dropped
            "P05|R05": None,         # non-numeric -> dropped
        },
    }
    out = _parse_response(raw)
    assert out.suggested_exploration_boost[ArmKey("P01", "R01")] == 0.0
    assert out.suggested_exploration_boost[ArmKey("P02", "R02")] == 0.3
    assert out.suggested_exploration_boost[ArmKey("P03", "R03")] == 0.15
    # invalid types dropped
    assert ArmKey("P04", "R04") not in out.suggested_exploration_boost
    assert ArmKey("P05", "R05") not in out.suggested_exploration_boost


# ---------- 5. timeout --------------------------------------------------------

def test_reflect_returns_none_on_timeout():
    """When _inner exceeds llm_timeout_s, reflect() returns None (fallback)."""
    r = Reflector(llm_timeout_s=0.05)

    async def _slow_inner(payload):
        await asyncio.sleep(1.0)
        return ReflectorOutput([], "")

    r._inner = _slow_inner  # type: ignore[assignment]
    result = asyncio.run(r.reflect({"coord": [0, 0]}))
    assert result is None


# ---------- 6. parse-error ----------------------------------------------------

def test_reflect_returns_none_on_inner_exception():
    """When _inner raises, reflect() catches and returns None (graceful fallback)."""
    r = Reflector(llm_timeout_s=5.0)

    async def _bad_inner(payload):
        raise RuntimeError("simulated network failure")

    r._inner = _bad_inner  # type: ignore[assignment]
    result = asyncio.run(r.reflect({"coord": [0, 0]}))
    assert result is None

"""v611 role-runners (Plan rev D Step 2c, codex round 11-18).

Three role-specific TRAPI gpt-5.4-mini callers, one per Δ7 role:
  - run_m1_proposer:    text-only, NO frame, outputs NL strategy
  - run_m2v_verifier:   text-only, NO frame, NO skill_md
  - run_m2e_executor:   multimodal PNG + approved NL

Each function opens a FRESH chat session (no transcript carryover).
JSON parsing is tolerant of wrapper text via `_extract_json_block`.
On any LLM/parse failure, returns `{}` — validators then fail and
the orchestrator handles the skip path.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TRAPI_ENDPOINT = os.environ.get(
    "ARC_LITE_TRAPI_ENDPOINT",
    "https://trapi.research.microsoft.com/gcr/shared",
)
_API_VERSION = os.environ.get("ARC_LITE_API_VERSION", "2025-04-01-preview")
_MODEL_DEFAULT = os.environ.get("ARC_LITE_MODEL", "gpt-5.4-mini_2026-03-17")
_VISION_MODEL = os.environ.get(
    "ARC_LITE_V611_VISION_MODEL", _MODEL_DEFAULT,
)


# Load system prompts once at module init.
_PROMPTS_DIR = Path(__file__).parent / "v611_prompts"
_M1_SYSTEM = (_PROMPTS_DIR / "m1_proposer_system.md").read_text(
    encoding="utf-8"
)
_M2V_SYSTEM = (_PROMPTS_DIR / "m2v_verifier_system.md").read_text(
    encoding="utf-8"
)
_M2E_SYSTEM = (_PROMPTS_DIR / "m2e_executor_system.md").read_text(
    encoding="utf-8"
)
_M3_SYSTEM = (_PROMPTS_DIR / "m3_skill_compressor_system.md").read_text(
    encoding="utf-8"
)
_M4_SYSTEM = (_PROMPTS_DIR / "m4_reflector_system.md").read_text(
    encoding="utf-8"
)


# ─────────────────────────────────────────────────────────────────
# Shared client + JSON parsing
# ─────────────────────────────────────────────────────────────────


def _extract_json_block(text: str) -> str:
    """Lift the JSON object out of a wrapped LLM response."""
    if not text:
        return "{}"
    text = text.strip()
    # Strip markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # Locate the first '{' and matching '}'.
    start = text.find("{")
    if start < 0:
        return "{}"
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return text[start:]
    return text[start:end]


def _build_client():
    """Build a SYNC AzureOpenAI client with TRAPI credentials.

    Each role runner builds its own client → fresh chat session, no
    shared transcript.
    """
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        DefaultAzureCredential,
    )
    from openai import AzureOpenAI

    cred = ChainedTokenCredential(
        AzureCliCredential(), DefaultAzureCredential(),
    )

    def _token_provider() -> str:
        return cred.get_token("api://trapi/.default").token

    return AzureOpenAI(
        azure_endpoint=_TRAPI_ENDPOINT,
        api_version=_API_VERSION,
        azure_ad_token_provider=_token_provider,
    )


def _call_llm(
    *,
    system_prompt: str,
    user_text: str,
    model: str = _MODEL_DEFAULT,
    max_tokens: int = 500,
    png_bytes: bytes | None = None,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Single LLM call. Returns parsed JSON dict or {} on any failure."""
    try:
        client = _build_client()
    except Exception as e:  # noqa: BLE001
        logger.warning("v611 client build failed: %s", e)
        return {}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if png_bytes:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]})
    else:
        messages.append({"role": "user", "content": user_text})
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    # gpt-5.4-mini / gpt-5.5 require default temperature=1.
    # Only set explicit temperature for older models that support it.
    if model.endswith("-pro") or "gpt-5.3" in model.lower() or \
       "gpt-4" in model.lower():
        kwargs["temperature"] = 0.0
    if model.endswith("-pro") or "json" in model.lower():
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning("v611 LLM call failed (model=%s): %s", model, e)
        return {}
    raw = resp.choices[0].message.content or "{}"
    try:
        obj = json.loads(_extract_json_block(raw))
    except json.JSONDecodeError as e:
        logger.warning("v611 JSON parse failed: %s | raw[:200]=%r",
                       e, raw[:200])
        return {}
    return obj if isinstance(obj, dict) else {}


# ─────────────────────────────────────────────────────────────────
# Role runners
# ─────────────────────────────────────────────────────────────────


def run_m1_proposer(
    state_text: str,
    skill_md_summary: str,
    anchor_summary: str | None = None,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    """Δ7a Proposer — text-only, NO coords in output."""
    parts = [
        "STATE TEXT:",
        state_text,
        "",
        "SKILL.md SUMMARY:",
        skill_md_summary,
    ]
    if anchor_summary:
        parts += ["", "ANCHOR SUMMARY (start fresh from this only):",
                  anchor_summary]
    if rejection_reason:
        parts += ["", "REJECTION HINT (constraint):", rejection_reason]
    user_text = "\n".join(parts)
    return _call_llm(
        system_prompt=_M1_SYSTEM,
        user_text=user_text,
        max_tokens=600,
    )


def run_m2v_verifier(
    proposer_out: dict[str, Any],
    state_text_summary: str,
) -> dict[str, Any]:
    """Δ7b Verifier — separate session, sees only NL + summary."""
    nl = proposer_out.get("nl_strategy", "")
    region = proposer_out.get("suggested_click_region", "")
    user_text = (
        "PROPOSER NL STRATEGY:\n"
        f"{nl}\n\n"
        "SUGGESTED REGION:\n"
        f"{region}\n\n"
        "STATE SUMMARY:\n"
        f"{state_text_summary}\n"
    )
    return _call_llm(
        system_prompt=_M2V_SYSTEM,
        user_text=user_text,
        max_tokens=200,
    )


def run_m3_compressor(
    recent_trials: list[dict[str, Any]],
    existing_skills: list[dict[str, Any]],
) -> dict[str, Any]:
    """Δ3 Skill Compressor — runs every 5 turns, emits NL-only skill.

    Input has NO raw coords; trial text is bounded by upstream M1 NL.
    """
    trial_lines = []
    for i, t in enumerate(recent_trials[-5:], 1):
        trial_lines.append(
            f"  {i}. nl_strategy={t.get('nl_strategy', '')[:80]!r} "
            f"suggested_region={t.get('suggested_region', '')[:40]!r} "
            f"verdict={t.get('verdict', '?')} "
            f"frame_changed={t.get('frame_changed', '?')} "
            f"unsat_delta={t.get('unsat_delta', '?')}"
        )
    skill_lines = []
    for s in existing_skills[:8]:
        skill_lines.append(
            f"  - {s.get('skill_id', '?')}: "
            f"{s.get('nl_description', '')[:80]}"
        )
    user_text = (
        "RECENT TRIALS:\n"
        + ("\n".join(trial_lines) if trial_lines else "  (none)")
        + "\n\nEXISTING SKILLS:\n"
        + ("\n".join(skill_lines) if skill_lines else "  (none)")
    )
    return _call_llm(
        system_prompt=_M3_SYSTEM,
        user_text=user_text,
        max_tokens=400,
    )


def run_m4_reflector(
    proposer_out: dict[str, Any],
    verifier_out: dict[str, Any],
    executor_out: dict[str, Any],
    env_observation: dict[str, Any],
    prior_skills: list[dict[str, Any]],
) -> dict[str, Any]:
    """Δ5 M4 Reflector — privileged role, sees ALL turn artifacts,
    emits 3-step verification + SKILL.md patch."""
    skill_lines = []
    for s in prior_skills[:8]:
        skill_lines.append(
            f"  - {s.get('skill_id', '?')}: "
            f"{s.get('nl_description', '')[:80]}"
        )
    user_text = (
        "PROPOSER OUT:\n"
        f"  nl_strategy: {proposer_out.get('nl_strategy', '')[:200]}\n"
        f"  expected_signature: {proposer_out.get('expected_signature')}\n"
        f"  rollback_trigger: {proposer_out.get('rollback_trigger', '')[:100]}\n"
        "\nVERIFIER OUT:\n"
        f"  verdict: {verifier_out.get('verdict')}\n"
        f"  reason_nl: {verifier_out.get('reason_nl', '')[:150]}\n"
        "\nEXECUTOR OUT:\n"
        f"  click_xy_hint: {executor_out.get('click_xy_hint')}\n"
        f"  grounding_text: {executor_out.get('grounding_text', '')[:150]}\n"
        "\nENV OBSERVATION:\n"
        f"  frame_changed: {env_observation.get('frame_changed')}\n"
        f"  unsat_delta: {env_observation.get('unsat_delta')}\n"
        f"  level_delta: {env_observation.get('level_delta')}\n"
        "\nPRIOR SKILLS:\n"
        + ("\n".join(skill_lines) if skill_lines else "  (none)")
    )
    return _call_llm(
        system_prompt=_M4_SYSTEM,
        user_text=user_text,
        max_tokens=500,
    )


def run_m2e_executor(
    approved_out: dict[str, Any],
    png_bytes: bytes,
) -> dict[str, Any]:
    """Δ7c Executor — PNG visual + approved NL, emits click_xy."""
    nl = approved_out.get("nl_strategy", "")
    region = approved_out.get("suggested_click_region", "")
    user_text = (
        "APPROVED NL STRATEGY:\n"
        f"{nl}\n\n"
        "SUGGESTED REGION:\n"
        f"{region}\n"
    )
    return _call_llm(
        system_prompt=_M2E_SYSTEM,
        user_text=user_text,
        model=_VISION_MODEL,
        max_tokens=300,
        png_bytes=png_bytes,
    )

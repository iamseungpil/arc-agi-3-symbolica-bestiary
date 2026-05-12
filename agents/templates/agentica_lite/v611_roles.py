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

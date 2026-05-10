"""v601 Role 1: Proposer (LLM, conditional). Plan rev C §3.12.

Triggers: warm-up (turn 0), stalemate, or new paired-cf entry.
HARD: cannot emit submit_action.
TIMEOUT: 45 s; fallback to None on timeout / parse_error / schema_invalid /
llm_no_client. The proposer's structured output is validated against the
JSON schema before being passed to Policy.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .proposer_prompt import build_messages

logger = logging.getLogger(__name__)

_TRAPI_ENDPOINT = "https://trapi.research.microsoft.com/gcr/shared"
_API_VERSION = "2025-04-01-preview"
_MODEL_PREFERENCES = ["gpt-5.5_2025-08-07", "gpt-5.5", "gpt-5.4_2026-03-05"]

# Plan §4 INT04 + §3.12 fallback codes.
_FAILURE_CODES = {"timeout", "parse_error", "schema_invalid", "llm_no_client"}

# Plan §4 INT04: blacklist of forbidden predicate ids that hint at tool calls.
_PREDICATE_BLACKLIST = {"submit_action", "click", "tool_call", "execute"}


@dataclass
class ProposerOutput:
    candidate_predicate_id: str
    region_hint: str
    expected_signature: dict
    required_pre_state: dict
    confidence: float
    thought: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class ProposerResult:
    output: ProposerOutput | None
    failure_reason: str | None  # one of _FAILURE_CODES, or None on success
    schema_error_code: str | None = None  # populated on schema_invalid


def _validate_schema(raw: Any, visible_region_ids: list[str]) -> tuple[ProposerOutput | None, str | None]:
    """Validate raw JSON against plan §3 + INT04 schema.

    Returns (output, error_code). On success error_code is None.
    Error codes (plan §4 INT04):
      schema_missing_field, predicate_blacklisted, tool_call_blocked,
      confidence_out_of_range, region_unknown.
    """
    if not isinstance(raw, dict):
        return None, "schema_missing_field"
    # Tool-call block: any embedded `tool_calls` field is rejected.
    if "tool_calls" in raw:
        return None, "tool_call_blocked"
    required = (
        "candidate_predicate_id", "region_hint",
        "expected_signature", "required_pre_state", "confidence",
    )
    for field_name in required:
        if field_name not in raw:
            return None, "schema_missing_field"
    pid = raw["candidate_predicate_id"]
    if not isinstance(pid, str) or pid.strip() == "":
        return None, "schema_missing_field"
    if pid in _PREDICATE_BLACKLIST:
        return None, "predicate_blacklisted"
    region_hint = raw["region_hint"]
    if not isinstance(region_hint, str) or region_hint.strip() == "":
        return None, "schema_missing_field"
    if visible_region_ids and region_hint not in visible_region_ids:
        return None, "region_unknown"
    confidence = raw["confidence"]
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None, "confidence_out_of_range"
    if not (0.0 <= confidence <= 1.0):
        return None, "confidence_out_of_range"
    pre_state = raw["required_pre_state"]
    if not isinstance(pre_state, dict):
        return None, "schema_missing_field"
    for f in ("marker_id", "saturation_threshold", "saturation_denominator"):
        if f not in pre_state:
            return None, "schema_missing_field"
    return ProposerOutput(
        candidate_predicate_id=pid,
        region_hint=region_hint,
        expected_signature=raw.get("expected_signature") or {},
        required_pre_state=pre_state,
        confidence=confidence,
        thought=str(raw.get("thought") or "")[:1024],
        raw=raw,
    ), None


def parse_and_validate(raw_json: Any, visible_region_ids: list[str]) -> ProposerResult:
    """Public helper: validate a raw proposer JSON dict (used by INT04 fixtures)."""
    out, err = _validate_schema(raw_json, visible_region_ids)
    if err is not None:
        return ProposerResult(output=None, failure_reason="schema_invalid", schema_error_code=err)
    return ProposerResult(output=out, failure_reason=None)


class Proposer:
    def __init__(self, llm_timeout_s: float = 45.0) -> None:
        self.llm_timeout_s = llm_timeout_s

    async def propose(
        self,
        state: dict[str, Any],
        visible_region_ids: list[str],
    ) -> ProposerResult:
        try:
            return await asyncio.wait_for(
                self._inner(state, visible_region_ids),
                timeout=self.llm_timeout_s,
            )
        except asyncio.TimeoutError:
            return ProposerResult(output=None, failure_reason="timeout")
        except Exception as e:  # noqa: BLE001
            logger.warning("Proposer unexpected error: %s", e)
            return ProposerResult(output=None, failure_reason="parse_error")

    async def _inner(
        self,
        state: dict[str, Any],
        visible_region_ids: list[str],
    ) -> ProposerResult:
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
            )
            from openai import AsyncAzureOpenAI
        except Exception as e:  # noqa: BLE001
            logger.info("Proposer deps missing — fallback (%s)", e)
            return ProposerResult(output=None, failure_reason="llm_no_client")

        cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

        def _token_provider() -> str:
            return cred.get_token("api://trapi/.default").token

        client = AsyncAzureOpenAI(
            azure_endpoint=_TRAPI_ENDPOINT,
            api_version=_API_VERSION,
            azure_ad_token_provider=_token_provider,
        )
        messages = build_messages(state)
        last_err: Exception | None = None
        for model in _MODEL_PREFERENCES:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                obj = json.loads(resp.choices[0].message.content or "{}")
                out, err = _validate_schema(obj, visible_region_ids)
                if err is not None:
                    return ProposerResult(
                        output=None, failure_reason="schema_invalid",
                        schema_error_code=err,
                    )
                return ProposerResult(output=out, failure_reason=None)
            except json.JSONDecodeError as e:
                logger.info("Proposer json decode failed (%s): %s", model, e)
                last_err = e
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.info("Proposer model %s failed: %s", model, e)
        logger.warning("Proposer exhausted models; last_err=%s", last_err)
        return ProposerResult(output=None, failure_reason="parse_error")

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
# v605 arm7: prefer multimodal model first when image-mode is enabled.
# v607 Phase 10 (P1 codex pivot): ARC_LITE_MODEL env var overrides default.
import os as _os
_model_override = _os.environ.get("ARC_LITE_MODEL", "").strip()
if _model_override:
    _MODEL_PREFERENCES = [_model_override]
elif _os.environ.get("ARC_LITE_MULTIMODAL", "0") == "1":
    _MODEL_PREFERENCES = ["gpt-4o-mini_2024-07-18", "gpt-4o_2024-11-20"]
else:
    _MODEL_PREFERENCES = ["gpt-5.4-mini_2026-03-17", "gpt-5.4_2026-03-05"]

# Plan §4 INT04 + §3.12 fallback codes.
_FAILURE_CODES = {"timeout", "parse_error", "schema_invalid", "llm_no_client"}

def _extract_json_block(text: str) -> str:
    """Extract first {...}-balanced JSON block from arbitrary LLM output.

    Many TRAPI deployments do NOT support response_format=json_object and
    instead return Markdown-wrapped JSON or prose-prefixed JSON. We use a
    forgiving extractor that finds the first balanced object."""
    if not text:
        return "{}"
    s = text.strip()
    # Strip ```json ... ``` fences if present.
    if s.startswith("```"):
        end = s.find("```", 3)
        if end > 3:
            inner = s[3:end].lstrip()
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            s = inner.strip()
    # Find first balanced {...}.
    start = s.find("{")
    if start < 0:
        return s
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


# Plan §4 INT04: blacklist of forbidden predicate ids that hint at tool calls.
_PREDICATE_BLACKLIST = {"submit_action", "click", "tool_call", "execute"}


# v604.1: Process-global TRAPI circuit breaker. After a 403/429 rate-limit
# response, suspend ALL TRAPI calls (Proposer + Reflector) for N seconds
# rather than hammering the API and worsening the block.
import time as _time

_TRAPI_BREAKER_UNTIL = 0.0
_TRAPI_BREAKER_COOLDOWN_S = 600.0  # 10 minutes after a rate-limit hit


def _breaker_open() -> bool:
    return _time.time() < _TRAPI_BREAKER_UNTIL


def _trip_breaker(error_text: str) -> None:
    global _TRAPI_BREAKER_UNTIL
    if "403" in error_text or "429" in error_text or "blocked" in error_text.lower() \
            or "rate" in error_text.lower():
        _TRAPI_BREAKER_UNTIL = _time.time() + _TRAPI_BREAKER_COOLDOWN_S
        logger.warning(
            "TRAPI circuit breaker tripped for %.0fs (reason: %s)",
            _TRAPI_BREAKER_COOLDOWN_S, error_text[:120],
        )


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
    # v605 arm2b diagnostic: LLM was setting region_hint == marker_id, causing
    # Policy to click the marker itself instead of its compass slot. Reject.
    if pre_state.get("marker_id") and region_hint == pre_state["marker_id"]:
        return None, "region_hint_equals_marker_id"
    # v607 Phase 5: anti-leak gate on candidate_predicate_id (plan rev E §5.1 U5).
    # Defends against paraphrased cycle237 chid templates being smuggled in by
    # the Reflector or Proposer LLM. Returns schema_error_code = "anti_leak:<reason>".
    from .anti_leak import validate_chid_template
    leak_ok, leak_reason = validate_chid_template(pid)
    if not leak_ok:
        return None, f"anti_leak:{leak_reason}"
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
        # v604.1 circuit breaker: skip TRAPI when rate-limit cooldown active.
        if _breaker_open():
            return ProposerResult(output=None, failure_reason="llm_no_client")
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
        # v605 arm7: append a vision content block to the user message
        # when a raw grid is available + multimodal env var is set.
        if _os.environ.get("ARC_LITE_MULTIMODAL", "0") == "1":
            try:
                from ._render_frame import render_grid_data_url
                grid = state.get("_latest_grid") if isinstance(state, dict) else None
                if grid:
                    data_url = render_grid_data_url(grid)
                    # convert last user message to multipart content
                    last_idx = max(
                        (i for i, m in enumerate(messages) if m.get("role") == "user"),
                        default=-1,
                    )
                    if last_idx >= 0:
                        text = messages[last_idx]["content"]
                        messages[last_idx] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "image_url",
                                 "image_url": {"url": data_url}},
                            ],
                        }
            except Exception as e:  # noqa: BLE001
                logger.warning("multimodal render failed: %s", e)
        last_err: Exception | None = None
        for model in _MODEL_PREFERENCES:
            try:
                # NB: response_format json_object is unsupported by some
                # TRAPI deployments (e.g. gpt-5.3-codex returns 400). We
                # rely on the system prompt to demand JSON output and parse
                # any wrapper text via _extract_json_block before json.loads.
                # gpt-5.5 only accepts default temperature=1; gpt-5.4 supports 0.0.
                kwargs: dict = {"model": model, "messages": messages}
                if "5.5" not in model.lower():
                    kwargs["temperature"] = 0.0
                if "json" in model.lower() or model.endswith("-pro"):
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content or "{}"
                # v605 F4 codex round 3: log every successful proposer turn
                logger.info("PROPOSER_RAW model=%s len=%d raw=%r",
                            model, len(raw), raw[:600])
                obj = json.loads(_extract_json_block(raw))
                # v605 arm4 codex C5 fix: auto-rewrite region_hint when LLM
                # confuses marker_id with compass slot. Use unclicked compass
                # from state to redirect the click target to a neighbor.
                if isinstance(obj, dict):
                    pre = obj.get("required_pre_state") or {}
                    mid = pre.get("marker_id") if isinstance(pre, dict) else None
                    if mid and obj.get("region_hint") == mid:
                        # find unclicked compass slot for this marker in state
                        markers = state.get("marker_neighbor_states") or []
                        target = next((m for m in markers if m.get("marker_id") == mid), None)
                        if target:
                            compass = target.get("compass") or {}
                            unclicked = [
                                slot.get("region_id")
                                for slot in compass.values()
                                if (slot or {}).get("clicks", 0) == 0
                                and (slot or {}).get("region_id")
                            ]
                            if unclicked:
                                logger.info(
                                    "Proposer auto-rewrite region_hint=%s -> %s (marker_id collision)",
                                    mid, unclicked[0],
                                )
                                obj["region_hint"] = unclicked[0]
                out, err = _validate_schema(obj, visible_region_ids)
                if err is not None:
                    logger.warning(
                        "Proposer schema_invalid err=%s model=%s raw=%r",
                        err, model, raw[:600],
                    )
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
                _trip_breaker(str(e))
                if _breaker_open():
                    # don't try further models in this call once breaker tripped
                    break
        logger.warning("Proposer exhausted models; last_err=%s", last_err)
        return ProposerResult(output=None, failure_reason="parse_error")

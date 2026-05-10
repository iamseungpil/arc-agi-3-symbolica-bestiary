"""One-shot LLM call: emit ONE candidate predicate (plan §6.2 + §1.14).

Hard 45 s timeout, deterministic fallback to None on timeout / parse error.
LLM is gpt-5.5 via TRAPI / agl-dev. Output is a JSON envelope:
  {name, family, coord_policy, natural_language, lambda_body}
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TRAPI_ENDPOINT = "https://trapi.research.microsoft.com/gcr/shared"
_API_VERSION = "2025-04-01-preview"
_MODEL_PREFERENCES = ["gpt-5.5_2025-08-07", "gpt-5.5", "gpt-5.4_2026-03-05"]
_SYSTEM_PROMPT = (
    "You are a predicate generator for ARC-AGI-3. Output a JSON envelope with "
    "fields: name, family, coord_policy, natural_language, lambda_body. "
    "lambda_body is Python source defining `def fn(chain_state, t) -> list[RegionRef]`. "
    "Use only whitelisted helpers: int, float, str, bool, list, dict, tuple, set, "
    "range, len, abs, min, max, sum, sorted, reversed, enumerate, zip. "
    "No imports, no eval/exec, no dunder access."
)


@dataclass
class ExtenderInput:
    visible_regions: list
    posterior_top10: list[tuple[str, str, float]]
    recent_failures: list[dict]


@dataclass
class ExtenderOutput:
    accepted: bool
    name: str | None
    family: str | None
    coord_policy: str | None
    natural_language: str | None
    lambda_body: str | None
    reason: str  # "ok" | "timeout" | "parse_error" | "sandbox_reject" | "no_client"


class LLMExtender:
    def __init__(self, llm_timeout_s: float = 45.0) -> None:
        self.llm_timeout_s = llm_timeout_s

    async def propose(self, inp: ExtenderInput) -> ExtenderOutput:
        try:
            return await asyncio.wait_for(self._inner(inp), timeout=self.llm_timeout_s)
        except asyncio.TimeoutError:
            return ExtenderOutput(False, None, None, None, None, None, "timeout")
        except Exception as e:  # noqa: BLE001
            logger.warning("LLMExtender unexpected error: %s", e)
            return ExtenderOutput(False, None, None, None, None, None, "parse_error")

    async def _inner(self, inp: ExtenderInput) -> ExtenderOutput:
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
            )
            from openai import AsyncAzureOpenAI
        except Exception as e:  # noqa: BLE001
            logger.info("LLMExtender deps missing — fallback (%s)", e)
            return ExtenderOutput(False, None, None, None, None, None, "no_client")

        cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

        def _token_provider() -> str:
            return cred.get_token("api://trapi/.default").token

        client = AsyncAzureOpenAI(
            azure_endpoint=_TRAPI_ENDPOINT,
            api_version=_API_VERSION,
            azure_ad_token_provider=_token_provider,
        )
        user_prompt = json.dumps({
            "visible_regions": inp.visible_regions[:8],
            "posterior_top10": inp.posterior_top10,
            "recent_failures": inp.recent_failures[:5],
        }, default=str)[:6000]

        last_err: Exception | None = None
        for model in _MODEL_PREFERENCES:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                obj = json.loads(resp.choices[0].message.content or "{}")
                return ExtenderOutput(
                    True, obj.get("name"), obj.get("family"),
                    obj.get("coord_policy"), obj.get("natural_language"),
                    obj.get("lambda_body"), "ok",
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.info("LLMExtender model %s failed: %s", model, e)
        logger.warning("LLMExtender exhausted models; last_err=%s", last_err)
        return ExtenderOutput(False, None, None, None, None, None, "parse_error")

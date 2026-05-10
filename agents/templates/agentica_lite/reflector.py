"""v601 conditional Reflector sub-LLM. Plan rev C §3.15.

Returns ReflectorOutput with discriminator features, brief reflexion text,
and an UCB1 exploration boost (values in [0, 0.3], lifetime 5 turns) that
Policy applies to the named arms.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .predicate_posterior import ArmKey

logger = logging.getLogger(__name__)

_TRAPI_ENDPOINT = "https://trapi.research.microsoft.com/gcr/shared"
_API_VERSION = "2025-04-01-preview"
_MODEL_PREFERENCES = ["gpt-5.3-codex_2026-02-24", "gpt-5.4_2026-03-05", "gpt-5.4-mini_2026-03-17"]

REFLECTOR_SYSTEM_PROMPT = (
    "You are a contrastive analyzer for a structured-observation puzzle agent.\n"
    "Two outcomes occurred at the same coordinate but with different signatures.\n"
    "Your task: identify which discriminating feature(s) most likely caused the\n"
    "difference, and suggest a small UCB1 exploration boost (in [0, 0.3]) for\n"
    "arms that should be tried next.\n\n"
    "Output strict JSON:\n"
    "  discriminator_features: list[str]\n"
    "  reflexion_text: str (<= 256 chars)\n"
    "  suggested_exploration_boost: dict[str, float]  # arm key as string -> boost\n"
)


@dataclass
class ReflectorOutput:
    discriminator_features: list[str]
    reflexion_text: str
    suggested_exploration_boost: dict[ArmKey, float] = field(default_factory=dict)


def _parse_arm_key(s: str) -> ArmKey | None:
    """Parse 'pid|rid|status' or 'pid|rid' to ArmKey."""
    parts = s.split("|")
    if len(parts) == 2:
        return ArmKey(parts[0], parts[1])
    if len(parts) == 3:
        return ArmKey(parts[0], parts[1], parts[2])
    return None


def _parse_response(obj: Any) -> ReflectorOutput:
    if not isinstance(obj, dict):
        return ReflectorOutput([], "")
    feats = obj.get("discriminator_features") or []
    if not isinstance(feats, list):
        feats = []
    text = str(obj.get("reflexion_text") or "")[:256]
    raw_boost = obj.get("suggested_exploration_boost") or {}
    boost: dict[ArmKey, float] = {}
    if isinstance(raw_boost, dict):
        for k, v in raw_boost.items():
            arm = _parse_arm_key(str(k))
            if arm is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            f = max(0.0, min(0.3, f))
            boost[arm] = f
    return ReflectorOutput(
        discriminator_features=[str(f) for f in feats],
        reflexion_text=text,
        suggested_exploration_boost=boost,
    )


class Reflector:
    def __init__(self, llm_timeout_s: float = 30.0) -> None:
        self.llm_timeout_s = llm_timeout_s

    async def reflect(self, contrast_payload: dict[str, Any]) -> ReflectorOutput | None:
        """Return ReflectorOutput, or None on failure (fallback)."""
        try:
            return await asyncio.wait_for(
                self._inner(contrast_payload), timeout=self.llm_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.info("Reflector timeout")
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("Reflector error: %s", e)
            return None

    async def _inner(self, contrast_payload: dict[str, Any]) -> ReflectorOutput | None:
        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                DefaultAzureCredential,
            )
            from openai import AsyncAzureOpenAI
        except Exception as e:  # noqa: BLE001
            logger.info("Reflector deps missing: %s", e)
            return None

        cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

        def _token_provider() -> str:
            return cred.get_token("api://trapi/.default").token

        client = AsyncAzureOpenAI(
            azure_endpoint=_TRAPI_ENDPOINT,
            api_version=_API_VERSION,
            azure_ad_token_provider=_token_provider,
        )
        user_payload = json.dumps(contrast_payload, default=str)[:6000]
        for model in _MODEL_PREFERENCES:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                obj = json.loads(resp.choices[0].message.content or "{}")
                return _parse_response(obj)
            except Exception as e:  # noqa: BLE001
                logger.info("Reflector model %s failed: %s", model, e)
        return None

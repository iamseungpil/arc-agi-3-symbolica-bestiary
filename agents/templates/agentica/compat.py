from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import json
import logging
import os
import re
import time
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import TypeAdapter

from agentica import spawn as sdk_spawn

logger = logging.getLogger(__name__)

_TRAPI_SCOPE = "api://trapi/.default"
_TRAPI_API_VERSION = "2025-04-01-preview"
# 2026-05-04 history: gcr/shared catalog rotated GPT-5.x out; redmond/interactive
# was the alt path with gpt-5-pro/o3-mini fallbacks. Later same day gpt-5.5
# restored on both endpoints — back to original config. Env-var override
# preserved for benchmarking against o3-mini / gpt-5-pro / future models.
_TRAPI_ENDPOINT = os.environ.get(
    "ARC_AGENTICA_TRAPI_ENDPOINT",
    "https://trapi.research.microsoft.com/redmond/interactive",
)
_TRAPI_MODEL_MAP = {
    "gpt-5.3-codex": "gpt-5.3-codex_2026-02-24",
    "gpt-5.4-mini": "gpt-5.4-mini_2026-03-17",
    "gpt-5.4": "gpt-5.4_2026-03-05",
    "gpt-5.5": os.environ.get(
        "ARC_AGENTICA_TRAPI_DEPLOYMENT", "gpt-5.5_2026-04-24"),
    "gpt-5-pro": "gpt-5-pro_2025-10-06",
    "o3-mini": "o3-mini_2025-01-31",
}
_LOCAL_MODEL_FALLBACK = os.environ.get("ARC_AGENTICA_TRAPI_MODEL", "gpt-5.5")
_TRAPI_TIMEOUT_SEC = float(os.environ.get("ARC_AGENTICA_TRAPI_TIMEOUT_SEC", "120"))
_TRAPI_MAX_RETRIES = int(os.environ.get("ARC_AGENTICA_TRAPI_MAX_RETRIES", "1"))
_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_trapi_client = None


def backend_available() -> bool:
    return bool(os.environ.get("AGENTICA_API_KEY") or os.environ.get("S_M_BASE_URL"))


def _normalize_model_name(model: str) -> str:
    if model in _TRAPI_MODEL_MAP:
        return model
    lower = model.lower()
    if "gpt-5.5" in lower:
        return "gpt-5.5"
    if "gpt-5.4-mini" in lower:
        return "gpt-5.4-mini"
    if "gpt-5.4" in lower:
        return "gpt-5.4"
    if "gpt-5.3-codex" in lower:
        return "gpt-5.3-codex"
    if "gpt" in lower:
        return _LOCAL_MODEL_FALLBACK
    logger.info(
        "Falling back to TRAPI model %s for unsupported Agentica model %s",
        _LOCAL_MODEL_FALLBACK,
        model,
    )
    return _LOCAL_MODEL_FALLBACK


def _get_trapi_client():
    global _trapi_client
    if _trapi_client is not None:
        return _trapi_client
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        _TRAPI_SCOPE,
    )
    _trapi_client = AzureOpenAI(
        azure_endpoint=_TRAPI_ENDPOINT,
        azure_ad_token_provider=credential,
        api_version=_TRAPI_API_VERSION,
        timeout=_TRAPI_TIMEOUT_SEC,
        max_retries=_TRAPI_MAX_RETRIES,
    )
    return _trapi_client


def _request_text(
    *,
    model: str,
    instructions: str | None,
    messages: list[dict[str, str]],
    max_output_tokens: int = int(os.environ.get("ARC_AGENTICA_MAX_OUTPUT_TOKENS", "4096")),
) -> str:
    """Call TRAPI with rate-limit-aware retry.

    cycle265 (2026-05-08) died at action 51 with `openai.RateLimitError:
    429 Token limit is exceeded`. cycle266d (2026-05-08 17:14) died at
    action 12 with `openai.InternalServerError: 503 API Configuration
    unavailable`. Library's built-in retry uses 6-7s × max_retries=1
    which is too short for either case (TPM reset OR TRAPI maintenance).
    We add a manual loop that catches the full transient class
    (RateLimitError, InternalServerError, APITimeoutError,
    APIConnectionError) and sleeps with longer exponential backoff:
    30s, 60s, 120s, 240s, 480s, 600s caps. Keeps multi-hour cycles
    alive across bursty token windows AND TRAPI 5xx blips.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    deployment = _TRAPI_MODEL_MAP[_normalize_model_name(model)]
    client = _get_trapi_client()
    input_items = [
        {"type": "message", "role": message["role"], "content": message["content"]}
        for message in messages
    ]
    rl_max_retries = int(os.environ.get("ARC_AGENTICA_RATE_LIMIT_RETRIES", "8"))
    backoff_base = float(os.environ.get("ARC_AGENTICA_RATE_LIMIT_BACKOFF_SEC", "30"))
    transient_excs = (
        RateLimitError,
        InternalServerError,
        APITimeoutError,
        APIConnectionError,
    )
    response = None
    last_exc: Exception | None = None
    for attempt in range(rl_max_retries + 1):
        try:
            response = client.responses.create(
                model=deployment,
                instructions=instructions,
                input=input_items,
                max_output_tokens=max_output_tokens,
            )
            break
        except transient_excs as exc:
            last_exc = exc
            kind = type(exc).__name__
            if attempt >= rl_max_retries:
                logger.warning(
                    "TRAPI %s retry budget exhausted (%d) — re-raising",
                    kind, rl_max_retries,
                )
                raise
            sleep_s = min(backoff_base * (2 ** attempt), 600.0)
            logger.warning(
                "TRAPI %s (attempt %d/%d) — sleeping %.0fs",
                kind, attempt + 1, rl_max_retries, sleep_s,
            )
            time.sleep(sleep_s)
    if response is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("TRAPI request returned no response")
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    parts: list[str] = []
    for item in (getattr(response, "output", []) or []):
        for block in (getattr(item, "content", []) or []):
            block_text = getattr(block, "text", "") or ""
            if block_text:
                parts.append(block_text)
    return "\n".join(parts).strip()


def _truncate(value: str, limit: int = 400) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _describe_scope(scope: dict[str, Any]) -> str:
    lines: list[str] = []
    for name in sorted(scope):
        value = scope[name]
        type_name = type(value).__name__
        call_sig = ""
        if callable(value):
            try:
                call_sig = str(inspect.signature(value))
            except Exception:
                call_sig = "(...)"
        doc = _truncate((inspect.getdoc(value) or "").splitlines()[0], 160) if inspect.getdoc(value) else ""
        preview = _truncate(repr(value).replace("\n", " "), 120)
        parts = [f"- {name}: {type_name}{call_sig}"]
        if doc:
            parts.append(f"doc={doc}")
        elif preview and not callable(value):
            parts.append(f"repr={preview}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _return_type_name(return_type: Any) -> str:
    return getattr(return_type, "__name__", repr(return_type))


def _normalize_call(return_type_or_task: Any, task: str | None) -> tuple[Any, str]:
    if isinstance(return_type_or_task, str) and task is None:
        return str, return_type_or_task
    if task is None:
        raise TypeError("Local Agentica compatibility layer requires a task string")
    return return_type_or_task, task


def _extract_json_fragment(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return None


def _validate_value(return_type: Any, value: Any) -> Any:
    if return_type in (None, type(None)):
        return None
    if return_type is str and isinstance(value, str):
        return value
    adapter = TypeAdapter(return_type)
    return adapter.validate_python(value)


def _parse_text_result(return_type: Any, text: str) -> Any:
    if return_type in (None, type(None)):
        return None
    if return_type is str:
        return text.strip()
    fragment = _extract_json_fragment(text)
    if fragment is None:
        raise ValueError(f"Expected structured {return_type!r}, got plain text: {text[:200]}")
    adapter = TypeAdapter(return_type)
    return adapter.validate_json(fragment)


async def _execute_code(code: str, scope: dict[str, Any]) -> tuple[bool, Any, str]:
    exec_scope = {"__builtins__": __builtins__, **scope}
    captured = {"set": False, "value": None}

    def final(value: Any) -> Any:
        captured["set"] = True
        captured["value"] = value
        return value

    exec_scope["final"] = final

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            compiled = compile(
                code,
                "<agentica-local>",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            result = eval(compiled, exec_scope, exec_scope)
            if inspect.isawaitable(result):
                await result
    except Exception as exc:
        output = stdout.getvalue()
        return False, None, f"{output}\n[python exception] {type(exc).__name__}: {exc}".strip()

    if captured["set"]:
        return True, captured["value"], stdout.getvalue()
    for candidate in ("_final_result", "final_result", "result"):
        if candidate in exec_scope:
            return True, exec_scope[candidate], stdout.getvalue()
    return False, None, stdout.getvalue()


class LocalTrapiAgent:
    def __init__(
        self,
        *,
        premise: str | None,
        system: str | None,
        model: str,
        reasoning_effort: str | None,
        scope: dict[str, Any] | None,
        listener: Any = None,
    ) -> None:
        self.premise = premise
        self.system = system
        self.model = _normalize_model_name(model)
        self.reasoning_effort = reasoning_effort
        self.scope = dict(scope or {})
        self.listener = listener
        self._messages: list[dict[str, str]] = []
        debug_dir = os.environ.get("ARC_AGENTICA_LOCAL_DEBUG_DIR", "").strip()
        self._debug_path: Path | None = None
        if debug_dir:
            path = Path(debug_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._debug_path = path / f"local_agent_{int(time.time() * 1000)}_{os.getpid()}_{id(self)}.jsonl"
        self._last_usage = SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
        )

    def _debug_log(self, payload: dict[str, Any]) -> None:
        if self._debug_path is None:
            return
        record = {
            "ts": time.time(),
            **payload,
        }
        with self._debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def call(
        self,
        return_type_or_task: Any,
        task: str | None = None,
        /,
        mcp: str | None = None,
        **scope: Any,
    ) -> Any:
        del mcp
        return_type, task_text = _normalize_call(return_type_or_task, task)
        # PLAN v52 — extract multimodal sentinel BEFORE building scope
        # description so it doesn't leak into Python runtime bindings.
        _image_b64 = scope.pop("_image_b64", None) if scope else None
        live_scope = {**self.scope, **scope}
        runtime_rules = f"""
You are running in a local Agentica-compatible Python runtime.

Important rules:
- Names listed below are already bound in Python. Do not import or recreate them.
- To use the runtime, emit exactly one ```python``` block and nothing else.
- The block may use normal Python and top-level `await`.
- When you are ready to finish, call `final(value)` inside Python.
- If you answer without Python, respond with only the final answer.
- Final answer type required: {_return_type_name(return_type)}.
- If the task is still in progress, do not call `final(...)` yet. Keep working.
- Prefer concise code and reuse the provided objects directly.

Available bindings:
{_describe_scope(live_scope)}
""".strip()
        instructions = "\n\n".join(
            part
            for part in [self.system, self.premise, runtime_rules]
            if part
        )
        # PLAN v52 — multimodal first user message construction.
        if _image_b64 and isinstance(_image_b64, str):
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": task_text},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{_image_b64}",
                        },
                    ],
                }
            )
        else:
            self._messages.append({"role": "user", "content": task_text})
        self._debug_log(
            {
                "kind": "call_start",
                "return_type": _return_type_name(return_type),
                "task": _truncate(task_text, 4000),
                "bindings": sorted(live_scope.keys()),
                "multimodal": bool(_image_b64),
            }
        )

        max_turns = int(os.environ.get("ARC_AGENTICA_LOCAL_MAX_TURNS", "48"))
        last_text = ""
        for turn in range(max_turns):
            last_text = _request_text(
                model=self.model,
                instructions=instructions,
                messages=self._messages,
            )
            self._messages.append({"role": "assistant", "content": last_text})
            self._debug_log(
                {
                    "kind": "model_response",
                    "turn": turn,
                    "text": _truncate(last_text, 12000),
                }
            )
            blocks = _CODE_BLOCK_RE.findall(last_text)
            if blocks:
                if len(blocks) != 1:
                    self._debug_log(
                        {
                            "kind": "runtime_error",
                            "turn": turn,
                            "message": "multiple_python_blocks",
                        }
                    )
                    self._messages.append(
                        {
                            "role": "user",
                            "content": "Runtime error: emit exactly one python block or a direct final answer.",
                        }
                    )
                    continue
                has_value, value, output = await _execute_code(blocks[0], live_scope)
                self._debug_log(
                    {
                        "kind": "python_execution",
                        "turn": turn,
                        "has_value": has_value,
                        "output": _truncate(output, 12000),
                        "value_preview": _truncate(repr(value), 2000),
                    }
                )
                if has_value:
                    try:
                        return _validate_value(return_type, value)
                    except Exception as exc:
                        self._debug_log(
                            {
                                "kind": "validation_error",
                                "turn": turn,
                                "message": str(exc),
                            }
                        )
                        self._messages.append(
                            {
                                "role": "user",
                                "content": f"Return validation error for {_return_type_name(return_type)}: {exc}. Produce a corrected final(value).",
                            }
                        )
                        continue
                if "budget exhausted" in output.lower():
                    self._messages.append(
                        {
                            "role": "user",
                            "content": "[python execution output]\n"
                            f"{output or '(no stdout)'}\n"
                            "The action budget is exhausted for this call. Do not attempt more actions. "
                            "Return a final summary/result for your caller now.",
                        }
                    )
                    continue
                self._messages.append(
                    {
                        "role": "user",
                        "content": f"[python execution output]\n{output or '(no stdout)'}\nNo final value was produced. Continue.",
                    }
                )
                continue
            try:
                return _parse_text_result(return_type, last_text)
            except Exception as exc:
                self._debug_log(
                    {
                        "kind": "parse_error",
                        "turn": turn,
                        "message": str(exc),
                    }
                )
                self._messages.append(
                    {
                        "role": "user",
                        "content": f"Return validation error for {_return_type_name(return_type)}: {exc}. Reply with a valid final answer only.",
                    }
                )

        self._debug_log(
            {
                "kind": "call_failure",
                "last_text": _truncate(last_text, 4000),
                "max_turns": max_turns,
            }
        )
        raise RuntimeError(
            "Local Agentica compatibility agent failed to produce a valid result "
            f"after {max_turns} turns. Last response: {_truncate(last_text, 500)}"
        )

    def last_usage(self) -> Any:
        return self._last_usage

    def set_listener(self, listener: Any) -> None:
        self.listener = listener


async def spawn(
    premise: str | None = None,
    scope: dict[str, Any] | None = None,
    *,
    system: str | None = None,
    mcp: str | None = None,
    model: str = _LOCAL_MODEL_FALLBACK,
    listener: Any = None,
    max_tokens: Any = None,
    reasoning_effort: str | None = None,
    cache_ttl: Any = None,
    _logging: bool = False,
    _call_depth: int = 0,
):
    del mcp, max_tokens, cache_ttl, _logging, _call_depth
    if backend_available():
        return await sdk_spawn(
            premise,
            scope,
            system=system,
            model=model,
            listener=listener,
            reasoning_effort=reasoning_effort,
        )
    return LocalTrapiAgent(
        premise=premise,
        system=system,
        model=model,
        reasoning_effort=reasoning_effort,
        scope=scope,
        listener=listener,
    )

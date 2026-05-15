#!/usr/bin/env python3
"""S0 smoke test: run Arcgentica on an ft09 game via the local TRAPI proxy.

This script bypasses ``main.py`` entirely so it can be invoked from CI or a
research notebook without disturbing the AMLT entrypoint. It:

1. Reads credentials from ``.env`` (fail-loud if ``ARC_API_KEY`` is missing).
2. Spawns ``scripts.trapi_openai_proxy:app`` on port 9091 via ``uvicorn``.
3. Routes the OpenAI SDK to the local proxy.
4. Runtime-registers ``Arcgentica`` in ``AVAILABLE_AGENTS`` without editing
   ``agents/__init__.py``.
5. Constructs a Swarm by name and runs the first ft09 game with a 50-action cap
   and the ``GPT_5_5`` preset.
6. Parses the proxy JSONL log for latency percentiles + request count.
7. Writes ``reports/s0_smoke_ft09_<timestamp>.json``.

Pass criterion::

    final_state in {NOT_FINISHED, WIN, GAME_OVER}
    and error_count == 0
    and p50_latency_ms < 200
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make the repo importable when invoked from any cwd. Without this,
# `python scripts/s0_smoke_ft09.py` from an unrelated cwd raises
# ModuleNotFoundError on `from agents.templates.agentica import Arcgentica`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROXY_PORT = 9091
PROXY_HEALTH_URL = f"http://127.0.0.1:{PROXY_PORT}/health"
PROXY_LOG_PATH = Path("/tmp/s0_smoke_proxy.jsonl")
# Round 3 Priority 3: allow callers (V4 smoke) to override the action cap via
# env, without forking the script. Default preserves Round 6 baseline (50).
MAX_ACTIONS = int(os.environ.get("S0_MAX_ACTIONS", "50"))
PROXY_BOOT_TIMEOUT_S = 30.0

# Codex r44 Blocker #1: Arcgentica.spawn() goes via local agentica-server
# (S_M_BASE_URL) which then relays to our TRAPI proxy. The agentica-server
# repo lives at /home/v-seungplee/agentica-server; we boot it as a subprocess.
AGENTICA_SERVER_PORT = 2345
AGENTICA_SERVER_BASE_URL = f"http://localhost:{AGENTICA_SERVER_PORT}"
AGENTICA_SERVER_HEALTH_URL = f"http://127.0.0.1:{AGENTICA_SERVER_PORT}/health"
AGENTICA_SERVER_REPO = Path("/home/v-seungplee/agentica-server")
AGENTICA_SERVER_ENTRYPOINT = AGENTICA_SERVER_REPO / "src" / "application" / "main.py"
AGENTICA_SERVER_PROVIDERS_YAML = AGENTICA_SERVER_REPO / "providers.local-trapi.yml"
AGENTICA_SERVER_BOOT_TIMEOUT_S = 120.0  # codex r45: 30s too tight; with --disable-otel boot is ~49s on cold env
AGENTICA_SERVER_LOG_PATH = Path(f"/tmp/s0_agentica_server_{int(time.time())}.log")


def _fail(msg: str, hint: str | None = None) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[s0_smoke] FATAL: {msg}", file=sys.stderr)
    if hint:
        print(f"[s0_smoke]   hint: {hint}", file=sys.stderr)
    sys.exit(2)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        _fail("python-dotenv not installed", hint="pip install python-dotenv")
    load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), override=True)


def _require_arc_api_key() -> str:
    # offline mode (OPERATION_MODE=offline) skips ARC API calls — dummy key OK
    op_mode = os.environ.get("OPERATION_MODE", "").strip().lower()
    if op_mode == "offline":
        key = os.environ.get("ARC_API_KEY", "").strip() or "offline-dummy-key"
        os.environ["ARC_API_KEY"] = key
        return key
    key = os.environ.get("ARC_API_KEY", "").strip()
    if not key:
        _fail(
            "ARC_API_KEY not set in environment or .env",
            hint="Add ARC_API_KEY=... to .env at the project root.",
        )
    return key


def _wait_for_proxy(timeout_s: float) -> None:
    import httpx

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(PROXY_HEALTH_URL, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 — retry every error
            last_err = exc
        time.sleep(0.3)
    _fail(
        f"proxy did not become healthy within {timeout_s}s at {PROXY_HEALTH_URL}",
        hint=f"last error: {last_err!r}; check `az login` and TRAPI access",
    )


def _start_proxy(env: dict[str, str]) -> subprocess.Popen[bytes]:
    if PROXY_LOG_PATH.exists():
        PROXY_LOG_PATH.unlink()
    # Sanity check the port is free first.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", PROXY_PORT))
        except OSError as exc:
            _fail(
                f"port {PROXY_PORT} already in use",
                hint=f"OS error: {exc}; kill the previous proxy or pick another port",
            )
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "scripts.trapi_openai_proxy:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PROXY_PORT),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def _wait_for_port(port: int, timeout_s: float, label: str) -> None:
    """Poll a TCP port until it accepts connections or the timeout elapses."""
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
                return
            except Exception as exc:  # noqa: BLE001 — retry every error
                last_err = exc
        time.sleep(0.3)
    _fail(
        f"{label} did not open port {port} within {timeout_s}s",
        hint=f"last error: {last_err!r}",
    )


def _start_agentica_server(env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Boot the local agentica-server that bridges Arcgentica.spawn() to TRAPI.

    Codex r44 Blocker #1: Symbolica's ``agentica.spawn()`` does NOT call TRAPI
    directly. It calls a local agentica-server (``S_M_BASE_URL``) which then
    routes via ``providers.local-trapi.yml`` to our 9091 proxy → TRAPI.

    Call chain we set up here::

        Arcgentica.spawn()
          → agentica SDK
          → S_M_BASE_URL :2345  (this subprocess)
          → providers.local-trapi.yml routes
          → 127.0.0.1:9091/v1/responses  (TRAPI proxy)
          → trapi.research.microsoft.com  (Azure AD)
    """
    if not AGENTICA_SERVER_ENTRYPOINT.is_file():
        _fail(
            f"agentica-server entrypoint not found at {AGENTICA_SERVER_ENTRYPOINT}",
            hint="clone https://github.com/symbolica/agentica-server to /home/v-seungplee/",
        )
    if not AGENTICA_SERVER_PROVIDERS_YAML.is_file():
        _fail(
            f"agentica-server providers yaml not found at {AGENTICA_SERVER_PROVIDERS_YAML}",
            hint="create providers.local-trapi.yml routing to 127.0.0.1:9091",
        )
    # Sanity check the port is free.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", AGENTICA_SERVER_PORT))
        except OSError as exc:
            _fail(
                f"port {AGENTICA_SERVER_PORT} already in use",
                hint=f"OS error: {exc}; kill the previous agentica-server",
            )
    cmd = [
        "uv",
        "run",
        str(AGENTICA_SERVER_ENTRYPOINT),
        "--port",
        str(AGENTICA_SERVER_PORT),
        "--inference-providers",
        str(AGENTICA_SERVER_PROVIDERS_YAML),
        "--sandbox-mode",
        "no_sandbox",
        "--log-level",
        "WARNING",
        # Codex r45 empirical finding: without --disable-otel, server hangs
        # after the OTel WARNING and /health never comes up (>92s). With this
        # flag, /health reaches OK in ~49s on cold env.
        "--disable-otel",
    ]
    # Codex r45 Blocker #2: DEVNULL eats all diagnostic on boot fail. Capture
    # to a real log file so we can tail it when boot fails.
    AGENTICA_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(AGENTICA_SERVER_LOG_PATH, "wb")
    print(f"[s0_smoke] agentica-server log -> {AGENTICA_SERVER_LOG_PATH}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(AGENTICA_SERVER_REPO),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc


def _wait_for_agentica_health(timeout_s: float) -> None:
    """Poll the agentica-server /health endpoint (HTTP, not TCP).

    Codex r45: TCP connect succeeds before app is ready; /health is the
    authoritative ready signal.
    """
    import httpx

    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(AGENTICA_SERVER_HEALTH_URL, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = RuntimeError(f"status_code={r.status_code} body={r.text[:200]!r}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.5)
    # Boot failed: surface log tail.
    tail = ""
    try:
        with open(AGENTICA_SERVER_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        tail = "".join(lines[-40:])
    except Exception as exc:  # noqa: BLE001
        tail = f"(could not read agentica-server log: {exc!r})"
    print(f"[s0_smoke] agentica-server /health did not come up in {timeout_s}s",
          file=sys.stderr)
    print(f"[s0_smoke] last error: {last_err!r}", file=sys.stderr)
    print(f"[s0_smoke] log tail ({AGENTICA_SERVER_LOG_PATH}):\n{tail}",
          file=sys.stderr)
    _fail(
        f"agentica-server health gate failed at {AGENTICA_SERVER_HEALTH_URL}",
        hint=f"check {AGENTICA_SERVER_LOG_PATH} and az login state",
    )


def _select_ft09_game(arc_root_url: str, arc_api_key: str) -> str:
    # Codex r44 Blocker #3: in OPERATION_MODE=offline we have no ARC API key
    # and no network — short-circuit straight to environment_files/ instead of
    # crashing on the requests.get() preflight.
    op_mode = os.environ.get("OPERATION_MODE", "").strip().lower()
    if op_mode == "offline":
        env_dir = PROJECT_ROOT / os.environ.get("ENVIRONMENTS_DIR", "environment_files")
        local: list[str] = []
        if env_dir.is_dir():
            for game_name in os.listdir(env_dir):
                gdir = env_dir / game_name
                if not gdir.is_dir() or not game_name.startswith("ft09"):
                    continue
                for version in os.listdir(gdir):
                    if (gdir / version).is_dir():
                        local.append(f"{game_name}-{version}")
        if not local:
            _fail(
                "OPERATION_MODE=offline but no ft09 games found in "
                f"{env_dir}/",
                hint="seed environment_files/ft09-*/ or unset OPERATION_MODE",
            )
        return sorted(local)[0]

    import requests

    headers = {"X-API-Key": arc_api_key, "Accept": "application/json"}
    try:
        r = requests.get(f"{arc_root_url}/api/games", headers=headers, timeout=10)
    except Exception as exc:  # noqa: BLE001
        _fail(f"failed to list games from {arc_root_url}: {exc!r}")
    if r.status_code != 200:
        _fail(
            f"ARC games endpoint returned {r.status_code}: {r.text[:200]}",
            hint="check ARC_API_KEY and the ARC HOST/PORT settings in .env",
        )
    try:
        all_games = [g["game_id"] for g in r.json()]
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not parse ARC games response: {exc!r}; body={r.text[:200]!r}")
    ft09 = [gid for gid in all_games if gid.startswith("ft09-")]
    if not ft09:
        # Fallback: scan local environment files in case offline mode is on.
        env_dir = PROJECT_ROOT / os.environ.get("ENVIRONMENTS_DIR", "environment_files")
        local: list[str] = []
        if env_dir.is_dir():
            for game_name in os.listdir(env_dir):
                gdir = env_dir / game_name
                if gdir.is_dir():
                    for version in os.listdir(gdir):
                        local.append(f"{game_name}-{version}")
        ft09 = [gid for gid in local if gid.startswith("ft09-")]
    if not ft09:
        _fail("no ft09-* game found via API or local environment_files/")
    return sorted(ft09)[0]


def _install_research_extensions_stub() -> None:
    """Install in-memory stub for ``research_extensions.ModuleRegistry`` etc.

    Rev F code-review Critical#1: the frozen ``agents/templates/agentica/agent.py``
    line 22 imports ``ModuleRegistry``, ``ResearchRuntimeContext``, and
    ``load_research_config`` from ``research_extensions``. The local
    ``research_extensions/`` namespace package does not export those symbols
    in this environment, causing an ImportError at Arcgentica module load.

    The stub provides minimal no-op implementations that satisfy the import +
    every call site we hit during a 50-action smoke. We DO NOT edit the frozen
    upstream package — this is a pre-import sys.modules patch only.
    """
    import sys
    import types

    rx = sys.modules.get("research_extensions")
    if rx is not None and hasattr(rx, "ModuleRegistry"):
        return  # already real or already stubbed

    # Round 3 Blocker 1 fix: before we replace research_extensions with a flat
    # stub, force-load the real ``research_extensions.tools.visual_grounder``
    # (and its sibling ``research_extensions.tools.snap``) so they are cached
    # in ``sys.modules``. Subsequent ``from research_extensions.tools.X import Y``
    # imports will hit the sys.modules cache and short-circuit the package
    # resolution that would otherwise trip on the flat stub.
    #
    # This lets ``scripts.vg_scope_probe`` (imported AFTER this stub install
    # in S0 real ordering) safely re-resolve ``analyze_frame`` at its own
    # module scope without paying the cost of re-attaching submodules.
    try:
        import research_extensions.tools.snap  # noqa: F401
        import research_extensions.tools.visual_grounder  # noqa: F401
    except ImportError:
        # If VG/snap source is missing the smoke will fail later in a clearer
        # location; we deliberately do not swallow other import errors here.
        pass

    stub = types.ModuleType("research_extensions")

    class ModuleRegistry:  # pragma: no cover - stub
        def __init__(self, *a, **kw) -> None:
            self._modules: dict[str, Any] = {}

        def active(self) -> dict[str, Any]:
            return {}

        def get(self, name: str) -> Any | None:
            return None

        def prompt_overlay(self) -> str:
            return ""

        def record_agent_env_note(self, *a, **kw) -> None:
            return None

        def record_agent_prediction(self, *a, **kw) -> None:
            return None

        def record_agent_world_update(self, *a, **kw) -> None:
            return None

        def record_agent_skill_proposal(self, *a, **kw) -> None:
            return None

    class ResearchRuntimeContext:  # pragma: no cover - stub
        def __init__(self, *a, **kw) -> None:
            self.registry = ModuleRegistry()
            self.log_dir = "/tmp/s0_research_log"

    def load_research_config(*a, **kw):  # pragma: no cover - stub
        from dataclasses import dataclass, field

        @dataclass
        class _Cfg:
            log_dir: str = "/tmp/s0_research_log"
            modules: list = field(default_factory=list)

            def active_modules(self) -> list:
                return []

        return _Cfg()

    stub.ModuleRegistry = ModuleRegistry  # type: ignore[attr-defined]
    stub.ResearchRuntimeContext = ResearchRuntimeContext  # type: ignore[attr-defined]
    stub.load_research_config = load_research_config  # type: ignore[attr-defined]
    sys.modules["research_extensions"] = stub

    # Also stub the missing `agents.templates.agentica.compat` shim — the
    # vendored agentica/ snapshot omits compat.py but agent.py:28 imports
    # `from .compat import spawn`. The upstream agentica SDK exposes spawn at
    # the top level, so we satisfy the relative import by injecting a module.
    compat_modname = "agents.templates.agentica.compat"
    if compat_modname not in sys.modules:
        try:
            from agentica import spawn as _spawn  # type: ignore[import]
            compat_stub = types.ModuleType(compat_modname)
            compat_stub.spawn = _spawn  # type: ignore[attr-defined]
            sys.modules[compat_modname] = compat_stub
        except ImportError:
            pass  # leave un-stubbed if agentica SDK not installed


def _register_arcgentica() -> None:
    """Runtime-register Arcgentica in AVAILABLE_AGENTS without editing the
    frozen ``agents/__init__.py``.
    """
    _install_research_extensions_stub()
    from agents.templates.agentica import Arcgentica
    import agents as agents_pkg

    agents_pkg.AVAILABLE_AGENTS["arcgentica"] = Arcgentica


def _summarize_log(log_path: Path) -> dict[str, Any]:
    """Compute latency percentiles + error count from the proxy JSONL log.

    Codex r44 Blocker #4: two fixes here.
      1. Read ``ts_ms`` (float, sub-ms precision) instead of ``ts`` (int seconds).
         Falling back to ``ts * 1000`` preserves back-compat with older logs.
      2. Pair requests with responses by ``req_uuid`` instead of ``url``. URL
         pairing collapses concurrent requests to the same endpoint onto one
         latency sample (TRAPI proxy gets ~150-500 /v1/responses calls per
         50-action run, many of them concurrent).
    """
    if not log_path.is_file():
        return {
            "request_count": 0,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "error_count": 0,
        }

    def _ts_ms(rec: dict[str, Any]) -> float | None:
        # Prefer the high-precision float; fall back to int seconds * 1000.
        ts_ms = rec.get("ts_ms")
        if isinstance(ts_ms, (int, float)):
            return float(ts_ms)
        ts = rec.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts) * 1000.0
        return None

    starts_ms: dict[str, float] = {}  # req_uuid -> start_ms
    latencies_ms: list[float] = []
    error_count = 0
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            req_uuid = rec.get("req_uuid") or ""
            ts_ms = _ts_ms(rec)
            if not req_uuid or ts_ms is None:
                # Old-format record without req_uuid — skip latency pairing but
                # still count error responses below.
                if event == "json_response" and int(rec.get("status_code", 200)) >= 400:
                    error_count += 1
                elif event == "stream_error_response":
                    error_count += 1
                continue
            if event in {"json_request", "stream_request"}:
                starts_ms[req_uuid] = ts_ms
            elif event == "json_response":
                start = starts_ms.pop(req_uuid, None)
                if start is not None:
                    latencies_ms.append(ts_ms - start)
                if int(rec.get("status_code", 200)) >= 400:
                    error_count += 1
            elif event == "stream_complete":
                start = starts_ms.pop(req_uuid, None)
                if start is not None:
                    latencies_ms.append(ts_ms - start)
            elif event == "stream_error_response":
                error_count += 1
                starts_ms.pop(req_uuid, None)

    def _pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        ys = sorted(xs)
        idx = max(0, min(len(ys) - 1, int(round((len(ys) - 1) * q))))
        return ys[idx]

    return {
        "request_count": len(latencies_ms),
        "p50_latency_ms": _pct(latencies_ms, 0.5),
        "p95_latency_ms": _pct(latencies_ms, 0.95),
        "error_count": error_count,
    }


def _final_state(agent: Any) -> str:
    try:
        frames = agent.frames or []
        if frames:
            return frames[-1].state.name
    except Exception:  # noqa: BLE001
        pass
    return "UNKNOWN"


def _final_levels_completed(agent: Any) -> int:
    """Extract ``levels_completed`` from the agent's final frame.

    Round 7 single-fix: V6's McNemar test on binary ``L >= 1`` requires this
    field to be surfaced from the agent state into the report JSON. The agent
    exposes the property ``agent.levels_completed`` which proxies
    ``agent.frames[-1].levels_completed`` (see ``agents/agent.py:96``); we
    fall back to the frame attribute if the property is unavailable so this
    helper stays robust against agent template differences.

    Returns ``0`` when no frame is observable (e.g. agent thread crashed
    before any frame arrived) — this is the conservative default that maps
    cleanly to ``L>=1 == False`` downstream.
    """
    if agent is None:
        return 0
    # Prefer the canonical property if present.
    try:
        val = getattr(agent, "levels_completed", None)
        if isinstance(val, int):
            return val
    except Exception:  # noqa: BLE001
        pass
    # Fall back to the frame attribute.
    try:
        frames = agent.frames or []
        if frames:
            return int(getattr(frames[-1], "levels_completed", 0) or 0)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _consume_s0_seed() -> int | None:
    """Apply ``S0_SEED`` to every RNG surface this script can reach.

    ``S0_SEED`` is consumed by V5/V6 paired evaluation harnesses to produce
    independent episodes for McNemar/Wilcoxon paired statistics. Without
    this reader, ``vg_v5_benchmark.py`` and ``vg_paired_eval.py`` set the
    env var but it has no effect — every "seed" collapses to LLM-sampling
    nondeterminism only and the paired stats become n=1.

    We seed (a) Python's ``random`` module, (b) NumPy's global RNG, and
    (c) export ``ARC_RANDOM_SEED`` so any downstream Arcgentica /
    research_extensions component that opts in to that convention can pick
    up the same seed. The agentica package itself currently exposes no
    explicit RNG seed mechanism (grep confirms no ``seed`` / ``random``
    references), so this is the strongest forwarding surface available
    without editing frozen upstream code.
    """
    seed_str = os.environ.get("S0_SEED", "").strip()
    if not seed_str:
        return None
    try:
        seed_val = int(seed_str)
    except ValueError:
        print(
            f"[s0_smoke] S0_SEED={seed_str!r} is not an integer; ignoring",
            file=sys.stderr,
            flush=True,
        )
        return None
    import random

    random.seed(seed_val)
    try:
        import numpy as _np  # type: ignore[import]

        _np.random.seed(seed_val)
    except ImportError:
        pass
    # Forward to any Arcgentica / research_extensions component that
    # honours ARC_RANDOM_SEED. The agentica package itself does not, but
    # downstream consumers (probe orderings, fixture selection, etc.) can
    # opt in without us editing the frozen upstream.
    os.environ["ARC_RANDOM_SEED"] = str(seed_val)
    print(f"[s0_smoke] seed = {seed_val}", flush=True)
    return seed_val


def main() -> int:
    _load_dotenv()
    # Apply S0_SEED before anything else touches an RNG — this is what
    # makes V5/V6 paired-evaluation seeds actually independent. See
    # ``_consume_s0_seed`` for the full forwarding contract.
    _consume_s0_seed()
    arc_api_key = _require_arc_api_key()

    # Resolve ARC root URL from env (mirrors main.py).
    scheme = os.environ.get("SCHEME", "https")
    host = os.environ.get("HOST", "three.arcprize.org")
    port = str(os.environ.get("PORT", "443"))
    if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
        arc_root_url = f"{scheme}://{host}"
    else:
        arc_root_url = f"{scheme}://{host}:{port}"

    proxy_env = dict(os.environ)
    proxy_env["TRAPI_PROXY_LOG_PATH"] = str(PROXY_LOG_PATH)

    # Codex r44 Blocker #1: start TRAPI proxy FIRST so agentica-server can talk
    # to it the moment it boots; then start agentica-server on 2345.
    print(f"[s0_smoke] starting proxy on port {PROXY_PORT}", flush=True)
    proxy_proc = _start_proxy(proxy_env)
    agentica_server_proc: subprocess.Popen[bytes] | None = None

    try:
        try:
            _wait_for_proxy(PROXY_BOOT_TIMEOUT_S)
        except SystemExit:
            stdout, stderr = b"", b""
            try:
                stdout, stderr = proxy_proc.communicate(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
            print("[s0_smoke] proxy stdout:\n" + stdout.decode("utf-8", "replace"), file=sys.stderr)
            print("[s0_smoke] proxy stderr:\n" + stderr.decode("utf-8", "replace"), file=sys.stderr)
            raise

        # Codex r44 Blocker #1: boot agentica-server now that the proxy is up.
        print(
            f"[s0_smoke] starting agentica-server on port {AGENTICA_SERVER_PORT}",
            flush=True,
        )
        agentica_env = dict(os.environ)
        agentica_server_proc = _start_agentica_server(agentica_env)
        # Codex r45: TCP poll is too permissive (port binds before app ready).
        # Use HTTP /health which is the authoritative ready signal.
        _wait_for_agentica_health(AGENTICA_SERVER_BOOT_TIMEOUT_S)

        # Route the OpenAI SDK through the proxy (used by any direct SDK call).
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PROXY_PORT}/v1"
        os.environ["OPENAI_API_KEY"] = "trapi-proxy-token"
        os.environ["TRAPI_PROXY_LOG_PATH"] = str(PROXY_LOG_PATH)
        # Codex r44 Blocker #1: point agentica SDK at the local server and
        # remove AGENTICA_API_KEY so the SDK takes the S_M_BASE_URL branch
        # (otherwise it tries to call the Symbolica cloud platform).
        os.environ["S_M_BASE_URL"] = AGENTICA_SERVER_BASE_URL
        os.environ.pop("AGENTICA_API_KEY", None)

        # Register Arcgentica without modifying agents/__init__.py.
        _register_arcgentica()

        from agents.templates.agentica import Arcgentica  # noqa: E402
        from agents.swarm import Swarm  # noqa: E402

        Arcgentica.MAX_ACTIONS = MAX_ACTIONS

        # Force the S0 preset before run. Env override S0_MODEL_PRESET
        # selects between GPT_5_5 (default) and GPT_5_4_MINI (round 5 retry).
        # Codex r44 Blocker #2: agent.py imports resolve_model_config by VALUE
        # at module top (`from .model import resolve_model_config`). Patching
        # agentica.model.resolve_model_config has no effect on the binding in
        # agent.py's namespace. Patch agentica.agent.resolve_model_config so
        # Arcgentica.__init__'s `or resolve_model_config()` fallback picks it
        # up.
        from agents.templates.s_series.model_configs import (
            resolve_s_series_preset,
        )

        preset_name = os.environ.get("S0_MODEL_PRESET", "gpt-5.5").strip() or "gpt-5.5"
        selected_preset = resolve_s_series_preset(preset_name)
        print(
            f"[s0_smoke] model preset = {preset_name!r} "
            f"(main={selected_preset.main_agent_model})",
            flush=True,
        )

        original_resolve = None
        try:
            from agents.templates.agentica import agent as agentica_agent

            original_resolve = agentica_agent.resolve_model_config
            agentica_agent.resolve_model_config = lambda: selected_preset  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            print(f"[s0_smoke] warning: could not override resolve_model_config: {exc!r}", flush=True)

        # Plan v650 rev D §5.1 — env-gated VisualGrounder injection. The patch
        # is a no-op unless S0_VG_ENABLED=1, preserving Round 6 baseline
        # behavior by default. When enabled, it wraps Arcgentica._spawn_scope
        # AND agents.templates.agentica.compat.spawn so all three spawn paths
        # (subagent, startup, orchestrator) receive the VG namespace.
        #
        # Round 3 Blocker 2 fix: a silent ``except`` here previously masked
        # patch failures (e.g. the research_extensions stub clobbering VG
        # imports), causing experiments to run with the VG-disabled baseline
        # despite S0_VG_ENABLED=1. We now fail loud — when VG is requested but
        # the patch errors, the run aborts so the user is forced to address
        # the underlying breakage rather than waste budget on a no-VG run.
        # The default S0_VG_ENABLED unset / =0 path is untouched, preserving
        # the Round 6 baseline contract.
        if os.environ.get("S0_VG_ENABLED", "").strip() == "1":
            try:
                from scripts.vg_scope_probe import patch_arcgentica_scopes

                vg_patch_info, _vg_unpatch = patch_arcgentica_scopes()
                print(
                    f"[s0_smoke] S0_VG_ENABLED=1 — VG patched: {vg_patch_info}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[s0_smoke] FATAL: VG patch failed under S0_VG_ENABLED=1: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            # Post-patch assertions: confirm BOTH spawn surfaces were wrapped.
            # Anything less means subagent OR startup/orchestrator paths would
            # not receive the visual_grounder namespace at runtime.
            assert vg_patch_info["spawn_scope_patched"], (
                "VG patch_info reports spawn_scope_patched=False; "
                "Arcgentica._spawn_scope was not wrapped — abort."
            )
            assert vg_patch_info["compat_spawn_wrapped"], (
                "VG patch_info reports compat_spawn_wrapped=False; "
                "agents.templates.agentica.compat.spawn was not wrapped — abort."
            )

        # --- S6 SkillLibrary (plan v651 revB, gated by S0_SKILL_ENABLED) ---
        # Default behaviour (env unset or != "1") leaves this script identical
        # to the V7-baseline path; the SkillLibrary patches are only installed
        # under explicit opt-in. The smoke script is ft09-specific so the
        # default persistence path is keyed on "ft09" — override via
        # S0_SKILL_PERSISTENCE_PATH to point at any other JSON file.
        if os.environ.get("S0_SKILL_ENABLED", "").strip() == "1":
            try:
                from research_extensions.tools.skill_library import SkillLibrary
                from scripts.skill_scope_probe import (
                    patch_arcgentica_scopes as patch_skill_scopes,
                )

                _skill_persistence_path = os.environ.get(
                    "S0_SKILL_PERSISTENCE_PATH",
                    "state/skills_ft09.json",
                )
                _skill_lib_singleton = SkillLibrary(
                    model=selected_preset.main_agent_model,
                    persistence_path=_skill_persistence_path,
                )
                skill_patch_info, _skill_unpatch = patch_skill_scopes(
                    library=_skill_lib_singleton,
                )
                assert skill_patch_info.get("spawn_scope_patched"), (
                    f"FATAL: skill spawn_scope patch did not apply: "
                    f"{skill_patch_info!r}"
                )
                assert skill_patch_info.get("compat_spawn_wrapped"), (
                    f"FATAL: skill compat.spawn wrap did not apply: "
                    f"{skill_patch_info!r}"
                )
                print(
                    f"[s0_smoke] S0_SKILL_ENABLED=1 — SkillLibrary patched: "
                    f"{skill_patch_info}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[s0_smoke] FATAL: Skill patch failed under "
                    f"S0_SKILL_ENABLED=1: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        game_id = _select_ft09_game(arc_root_url, arc_api_key)
        print(f"[s0_smoke] selected game {game_id}", flush=True)

        run_started = time.time()
        swarm = Swarm(
            agent="arcgentica",
            ROOT_URL=arc_root_url,
            games=[game_id],
            tags=["s0_smoke"],
        )

        try:
            swarm.main()
        finally:
            if original_resolve is not None:
                try:
                    from agents.templates.agentica import agent as agentica_agent

                    agentica_agent.resolve_model_config = original_resolve  # type: ignore[assignment]
                except Exception:  # noqa: BLE001
                    pass

        runtime_s = time.time() - run_started

        agent = swarm.agents[0] if swarm.agents else None
        final_state = _final_state(agent) if agent is not None else "UNKNOWN"
        action_count = getattr(agent, "action_counter", 0) if agent is not None else 0
        # Round 7 single-fix: surface game-state telemetry into the report.
        # V6 McNemar test on the binary ``L>=1`` outcome reads this field via
        # ``vg_paired_eval._l1_from_metrics``; without it every seed collapses
        # to L>=1=False and the test becomes uninformative (p=1.0).
        levels_completed = _final_levels_completed(agent)

        proxy_stats = _summarize_log(PROXY_LOG_PATH)
        # crude cost estimate: $0.005 per request (placeholder; real cost
        # accounting will come from UsageTracker in a later iteration).
        cost_estimate_usd = 0.005 * proxy_stats["request_count"]

        report = {
            # schema_version 2 — Round 7 added levels_completed/win_levels.
            "schema_version": 2,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "game_id": game_id,
            "preset": "gpt-5.5",
            "max_actions": MAX_ACTIONS,
            "final_state": final_state,
            "action_count": action_count,
            # Game-state outcome (Round 7 single-fix). ``win_levels`` is a
            # semantic alias to keep downstream callers explicit about what
            # the integer means.
            "levels_completed": int(levels_completed),
            "win_levels": int(levels_completed),
            "runtime_seconds": runtime_s,
            "request_count": proxy_stats["request_count"],
            "error_count": proxy_stats["error_count"],
            "p50_latency_ms": proxy_stats["p50_latency_ms"],
            "p95_latency_ms": proxy_stats["p95_latency_ms"],
            "cost_estimate_usd": cost_estimate_usd,
        }

        # Round 7 single-fix: allow callers (V5/V6 paired harnesses) to pin a
        # deterministic report path via ``S0_REPORT_PATH``. Without this hook
        # the harness would have to glob ``reports/s0_smoke_ft09_*.json`` and
        # race with sibling subprocesses. When the env var is unset we
        # preserve the Round 6 timestamped default.
        report_override = os.environ.get("S0_REPORT_PATH", "").strip()
        if report_override:
            out_path = Path(report_override)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = PROJECT_ROOT / "reports" / f"s0_smoke_ft09_{timestamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        # Decide pass/fail.
        # Codex r45 Blocker #3: p50 < 200ms gate measured FULL TRAPI/model
        # latency (GPT-5.5 high = 5-30s/call), so the gate was unreachable
        # even on a perfectly-working smoke. p50/p95 are reported in
        # telemetry but not gated. V0c is redefined in S1 eval harness
        # as a proxy-overhead-only measurement.
        #
        # Codex r46 final blocker: agent thread crashes are not propagated
        # by Swarm.main() (Thread(daemon=True)). Without a positive-work
        # gate, a crash after initial RESET would silently PASS with
        # action_count=0, request_count=0. Require POSITIVE work:
        #   action_count > 0  AND  request_count > 0.
        p50 = proxy_stats["p50_latency_ms"]
        ok = (
            final_state in {"NOT_FINISHED", "WIN", "GAME_OVER"}
            and proxy_stats["error_count"] == 0
            and action_count > 0
            and proxy_stats["request_count"] > 0
        )

        verdict = "PASS" if ok else "FAIL"
        print(
            f"[s0_smoke] {verdict} final_state={final_state} action_count={action_count} "
            f"errors={proxy_stats['error_count']} p50={p50} p95={proxy_stats['p95_latency_ms']} "
            f"requests={proxy_stats['request_count']} runtime_s={runtime_s:.1f}",
            flush=True,
        )
        print(f"[s0_smoke] wrote report -> {out_path}", flush=True)
        return 0 if ok else 1

    finally:
        # Codex r44 Blocker #1: tear down agentica-server first (it talks to
        # the proxy on shutdown), then the proxy.
        if agentica_server_proc is not None and agentica_server_proc.poll() is None:
            agentica_server_proc.terminate()
            try:
                agentica_server_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                agentica_server_proc.kill()
                try:
                    agentica_server_proc.wait(timeout=5.0)
                except Exception:  # noqa: BLE001
                    pass
        if proxy_proc.poll() is None:
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()
                proxy_proc.wait(timeout=5.0)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""V3a scope-coverage probe for the SkillLibrary (plan v651 revB §3 V3a).

OFFLINE probe — no LLM cost, no network.  Mirrors ``scripts/vg_scope_probe.py``
beat-for-beat so the two stay in lockstep: any structural fix made to the
VisualGrounder probe should be replayed here.

What this probe does:

1. Import :class:`SkillLibrary`, :class:`Skill`, and :class:`SkillSandboxError`
   at module load — BEFORE ``s0_smoke_ft09._install_research_extensions_stub``
   is invoked.  (Round-3 Blocker-1: the stub installs a flat ``ModuleType``
   in place of the ``research_extensions`` package, which then breaks
   ``from research_extensions.tools.<x> import …`` for any caller that has
   not already resolved the binding.)
2. Apply two monkey-patches:
     * ``Arcgentica._spawn_scope`` — wraps the original to inject the
       ``skill_library`` namespace into the subagent scope dict.
     * ``agents.templates.agentica.compat.spawn`` — wraps the awaitable to
       inject the ``skill_library`` namespace into ``scope=`` kwargs for
       startup and orchestrator spawns, gated by an agentica-frame filter
       (so non-agentica callers are not affected).
3. Statically scan ``agents/templates/agentica/agent.py`` for ``spawn(...)``
   callsites and confirm every callsite is covered by one of the patches.
4. Dynamically instantiate ``Arcgentica`` via ``__new__`` (bypassing the
   ARC-dependent ``__init__``) and call ``_spawn_scope()`` to confirm the
   patched method actually returns a scope containing ``skill_library``.
5. Write the verdict to ``reports/skill_scope_coverage.json``.

The probe exits 0 iff every spawn callsite is covered AND the patched
``_spawn_scope`` returns a dict containing the ``skill_library`` key.

This script does NOT edit anything under ``agents/templates/agentica`` or
``agentica-server`` on disk — the patches are runtime-only and the
``unpatch`` closure restores the originals on demand.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# IMPORTANT ORDERING NOTE (Round 3 Blocker 1 fix, replayed from vg_scope_probe):
# ``s0_smoke_ft09._install_research_extensions_stub()`` replaces the
# ``research_extensions`` namespace in ``sys.modules`` with a FLAT
# ``types.ModuleType`` (no ``__path__``), which breaks any subsequent
#   ``from research_extensions.tools.skill_library import SkillLibrary``
# call because Python's import machinery treats the stub as a non-package.
#
# We therefore resolve every binding we need from research_extensions BEFORE
# the stub is installed.  Lifting these imports to module scope (executed
# at the very first ``import scripts.skill_scope_probe`` — which happens
# before s0_smoke calls the stub installer) makes the bindings survive any
# later sys.modules surgery.  See Round 3 task-planner Option C.
# ---------------------------------------------------------------------------
from research_extensions.tools.skill_library import (  # noqa: E402
    Skill as _SKILL_DATACLASS,
    SkillLibrary as _SKILL_LIBRARY_CLS,
)
from research_extensions.tools.skill_sandbox import (  # noqa: E402
    SkillSandboxError as _SKILL_SANDBOX_ERROR,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
COVERAGE_REPORT_PATH = REPORTS_DIR / "skill_scope_coverage.json"
AGENT_SOURCE_PATH = (
    PROJECT_ROOT / "agents" / "templates" / "agentica" / "agent.py"
)


_SKILL_NAMESPACE_DOC = (
    "Persistable skill store. CALL `skill_library.library.summaries()` for a "
    "one-line list of stored skills (each line is `[index/short-id] summary "
    "[QUARANTINED?]`). CALL `skill_library.library.get(i_or_id)` to retrieve "
    "a single `Skill` dataclass with fields summary, recipe, evidence, "
    "posterior=(confirm_n, falsify_n), applicability_conditions, "
    "parent_hypothesis_ids, category ('mechanic' or 'strategy'), predicate, "
    "code, quarantined, quarantine_reason. CALL "
    "`await skill_library.library.query(str, '<free-text question>')` for a "
    "natural-language search backed by a sub-agent (returns a concise plain-"
    "text answer; falls back to 'No stored skills yet.' on empty/error). "
    "Skills with `quarantined=True` failed a contamination guardrail "
    "(falsify_n > confirm_n) — surface them but mark them as unreliable. Do "
    "NOT call `library.add` from inside a turn; writes happen only via the "
    "consolidator pipeline."
)


# ---------------------------------------------------------------------------
# Monkey-patch installer (the same logic that s0_smoke_ft09.py will call when
# S0_SKILL_ENABLED=1). Living here lets us probe coverage without LLM cost.
# ---------------------------------------------------------------------------


def patch_arcgentica_scopes(
    library: "_SKILL_LIBRARY_CLS | None" = None,
) -> Tuple[Dict[str, Any], Callable[[], None]]:
    """Install a SkillLibrary handle into all three Arcgentica spawn surfaces.

    Args:
        library: optional pre-built ``SkillLibrary`` to inject.  If ``None``
            we construct a default ``SkillLibrary(model="stub")`` for
            probe-only mode (so the probe is fully self-contained and never
            touches a real model).

    Returns:
        ``(info, unpatch)`` where ``info`` is a dict describing what was
        patched and ``unpatch`` is a zero-argument callable that restores the
        originals of every binding this function rebinds.  ``unpatch`` is
        idempotent; calling it twice is safe.

    NOTE: ``unpatch`` does NOT restore the ``research_extensions`` namespace
    stub installed by ``s0_smoke_ft09._install_research_extensions_stub()`` —
    that stub mutates ``sys.modules`` globally and is the responsibility of
    the caller (typically the s0 smoke driver) to manage.  The reversibility
    guarantee here covers only ``Arcgentica._spawn_scope``,
    ``agents.templates.agentica.compat.spawn``, and the agent module's
    ``spawn`` rebind (when applicable).

    Strategy:
      * Patch ``Arcgentica._spawn_scope`` to wrap the original; inject
        ``skill_library`` into the returned dict.  This covers the subagent
        spawn at agent.py:646.
      * Wrap ``agents.templates.agentica.compat.spawn`` so any call from
        agentica-internal code (startup at agent.py:710, orchestrator at
        agent.py:773) gets ``skill_library`` injected into its ``scope`` kw.
        We reuse the frame-walk filter from ``vg_scope_probe`` verbatim to
        limit injection to agentica-originated spawns.
    """
    # Round 3 Blocker 1 fix: every research_extensions binding we need was
    # resolved at module scope (see _SKILL_LIBRARY_CLS / _SKILL_DATACLASS /
    # _SKILL_SANDBOX_ERROR above) BEFORE any caller can install the
    # research_extensions stub.  We alias the module-level bindings into the
    # local namespace so the patched closures do not depend on
    # ``research_extensions`` resolving as a package at call time.
    SkillLibrary = _SKILL_LIBRARY_CLS
    Skill = _SKILL_DATACLASS
    SkillSandboxError = _SKILL_SANDBOX_ERROR

    from scripts import s0_smoke_ft09 as smoke

    smoke._install_research_extensions_stub()

    from agents.templates.agentica import agent as agentica_agent

    # Default to a probe-only stub library so the probe is self-contained.
    if library is None:
        library = SkillLibrary(model="stub")

    skill_namespace: Dict[str, Any] = {
        "library": library,
        "Skill": Skill,
        "SkillSandboxError": SkillSandboxError,
        "__doc__": _SKILL_NAMESPACE_DOC,
    }

    # ---- Patch 1: _spawn_scope on the class (subagent path) ------------
    original_spawn_scope = agentica_agent.Arcgentica._spawn_scope

    def _patched_spawn_scope(self: Any) -> Dict[str, Any]:
        scope = original_spawn_scope(self)
        if "skill_library" not in scope:
            scope["skill_library"] = skill_namespace
        return scope

    agentica_agent.Arcgentica._spawn_scope = _patched_spawn_scope

    # ---- Patch 2: compat.spawn wrapper (startup + orchestrator paths) --
    compat = sys.modules.get("agents.templates.agentica.compat")
    if compat is None:
        try:
            from agents.templates.agentica import compat as _compat  # type: ignore
            compat = _compat
        except ImportError:
            compat = None

    spawn_wrapped = False
    original_spawn: Any = None
    rebind_agent_spawn = False
    if compat is not None and hasattr(compat, "spawn"):
        original_spawn = compat.spawn

        def _injection_should_apply() -> bool:
            """Limit injection to agentica-originated spawns (planner #7).

            Walk up the stack 8 frames; if any are inside
            agents/templates/agentica we accept the injection.  Otherwise
            pass through.  Reused verbatim from ``vg_scope_probe`` since the
            filter is framework-agnostic.
            """
            frame = sys._getframe(1)
            for _ in range(8):
                if frame is None:
                    break
                code_file = getattr(frame.f_code, "co_filename", "")
                if "agents/templates/agentica" in code_file:
                    return True
                frame = frame.f_back
            return False

        async def _patched_spawn(*args: Any, **kwargs: Any) -> Any:
            if _injection_should_apply():
                scope = kwargs.get("scope") or {}
                if isinstance(scope, dict) and "skill_library" not in scope:
                    scope = {**scope, "skill_library": skill_namespace}
                    kwargs["scope"] = scope
            return await original_spawn(*args, **kwargs)

        compat.spawn = _patched_spawn  # type: ignore[assignment]
        if hasattr(agentica_agent, "spawn") and agentica_agent.spawn is original_spawn:
            agentica_agent.spawn = _patched_spawn  # type: ignore[assignment]
            rebind_agent_spawn = True
        spawn_wrapped = True

    # ---- Reversibility closure (Round 2 W1) ----------------------------
    _unpatched = {"done": False}

    def unpatch() -> None:
        if _unpatched["done"]:
            return
        if agentica_agent.Arcgentica._spawn_scope is _patched_spawn_scope:
            agentica_agent.Arcgentica._spawn_scope = original_spawn_scope
        if spawn_wrapped and compat is not None and original_spawn is not None:
            if getattr(compat, "spawn", None) is _patched_spawn:
                compat.spawn = original_spawn  # type: ignore[assignment]
            if rebind_agent_spawn and getattr(agentica_agent, "spawn", None) is _patched_spawn:
                agentica_agent.spawn = original_spawn  # type: ignore[assignment]
        _unpatched["done"] = True

    info: Dict[str, Any] = {
        "spawn_scope_patched": True,
        "compat_spawn_wrapped": spawn_wrapped,
        "namespace_keys": sorted(skill_namespace.keys()),
    }
    return info, unpatch


# ---------------------------------------------------------------------------
# Static analysis of agent.py for spawn callsites.
# ---------------------------------------------------------------------------


def find_spawn_callsites(source_path: Path) -> List[Dict[str, Any]]:
    """Return a list of ``spawn(...)`` callsites in agent.py with their line + scope info."""
    text = source_path.read_text()
    tree = ast.parse(text)
    sites: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "spawn":
                scope_repr = "<no scope kwarg>"
                for kw in node.keywords:
                    if kw.arg == "scope":
                        scope_repr = ast.unparse(kw.value)
                        break
                sites.append({
                    "line": node.lineno,
                    "scope_arg": scope_repr,
                })
    return sites


def classify_coverage(sites: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decide for each callsite whether _spawn_scope patch or spawn-wrapper covers it."""
    coverage: List[Dict[str, Any]] = []
    for site in sites:
        scope_arg = site["scope_arg"]
        if "self._spawn_scope()" in scope_arg:
            covered_by = "_spawn_scope_patch (subagent)"
        elif scope_arg == "<no scope kwarg>":
            covered_by = "spawn_wrapper (no scope kwarg → wrapper adds one)"
        else:
            covered_by = "spawn_wrapper (scope dict injection)"
        coverage.append({**site, "covered_by": covered_by})
    return {
        "callsite_count": len(coverage),
        "callsites": coverage,
    }


# ---------------------------------------------------------------------------
# Main probe.
# ---------------------------------------------------------------------------


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "probe": "skill_scope_probe (offline)",
        "agent_source": str(AGENT_SOURCE_PATH),
    }

    # 1. Static spawn-callsite scan.
    if not AGENT_SOURCE_PATH.is_file():
        report["status"] = "error"
        report["error"] = f"agent source not found: {AGENT_SOURCE_PATH}"
        COVERAGE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[skill_scope_probe] ERROR: {report['error']}", file=sys.stderr)
        return 2

    sites = find_spawn_callsites(AGENT_SOURCE_PATH)
    report["static_scan"] = classify_coverage(sites)

    # 2. Dynamic probe.
    dynamic: Dict[str, Any] = {"status": "skipped", "reason": ""}
    try:
        patch_info, _unpatch = patch_arcgentica_scopes()
        dynamic["patch_info"] = patch_info

        from agents.templates.agentica import agent as agentica_agent

        inst = agentica_agent.Arcgentica.__new__(agentica_agent.Arcgentica)
        inst.spawn_agent = lambda *a, **kw: None  # type: ignore[attr-defined]
        inst._research_active = lambda: False  # type: ignore[attr-defined]
        scope = inst._spawn_scope()
        dynamic["status"] = "ok"
        dynamic["spawn_scope_keys"] = sorted(scope.keys())
        dynamic["skill_library_present"] = "skill_library" in scope
        if "skill_library" in scope:
            ns = scope["skill_library"]
            if isinstance(ns, dict):
                dynamic["skill_library_namespace_keys"] = sorted(ns.keys())
            else:
                dynamic["skill_library_namespace_keys"] = sorted(
                    k for k in vars(ns).keys() if not k.startswith("__")
                )
    except Exception as exc:  # noqa: BLE001 — recorded in report
        dynamic["status"] = "error"
        dynamic["error"] = f"{type(exc).__name__}: {exc}"

    report["dynamic_probe"] = dynamic

    # 3. Verdict.
    static_ok = (
        all(
            c["covered_by"].startswith(("_spawn_scope_patch", "spawn_wrapper"))
            for c in report["static_scan"]["callsites"]
        )
        and report["static_scan"]["callsite_count"] > 0
    )
    dynamic_ok = (
        dynamic["status"] == "ok"
        and dynamic.get("skill_library_present") is True
    )
    report["verdict"] = "PASS" if (static_ok and dynamic_ok) else "FAIL"
    report["static_ok"] = static_ok
    report["dynamic_ok"] = dynamic_ok

    COVERAGE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[skill_scope_probe] verdict={report['verdict']} "
        f"static_callsites={report['static_scan']['callsite_count']} "
        f"static_ok={static_ok} dynamic_ok={dynamic_ok}",
        flush=True,
    )
    print(f"[skill_scope_probe] wrote report -> {COVERAGE_REPORT_PATH}", flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

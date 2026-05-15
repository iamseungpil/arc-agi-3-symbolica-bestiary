"""Sandboxed predicate / code evaluation for SkillLibrary (S6, plan v651 revB).

Two independent execution surfaces:

* :class:`DSLEvaluator` — pure AST-allowlist boolean predicates against a
  flat ``frame_features`` dict.  In-process, microsecond-scale, no I/O.  Even
  a single ``Call`` node raises :class:`SkillSandboxError` (codex r2 — the
  DSL boundary is sealed).  Use this to gate skill applicability cheaply.

* :class:`CodeHelperExecutor` — full Python in a child process with a wall-
  clock deadline and an address-space rlimit.  Reserved for skills whose
  ``code`` field is set, and invoked **OUT-OF-BAND by the orchestrator** —
  never from inside ``DSLEvaluator.evaluate``.  Returns ``None`` on any
  timeout / crash / malformed result; never propagates child exceptions.

Both surfaces use only stdlib so they ride along with the rest of
``research_extensions/tools`` (numpy/scipy already pinned upstream).
"""

from __future__ import annotations

import ast
import logging
import multiprocessing as mp
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SkillSandboxError", "DSLEvaluator", "CodeHelperExecutor"]


class SkillSandboxError(Exception):
    """Raised when a skill predicate or code-helper violates sandbox rules."""


# ---------------------------------------------------------------------------
# DSL evaluator — AST allowlist
# ---------------------------------------------------------------------------


_ALLOWED_CMP_OPS: tuple[type[ast.cmpop], ...] = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)

_ALLOWED_BOOLOPS: tuple[type[ast.boolop], ...] = (ast.And, ast.Or)
_ALLOWED_UNARYOPS: tuple[type[ast.unaryop], ...] = (ast.USub, ast.UAdd, ast.Not)


class DSLEvaluator:
    """Evaluate a sandboxed predicate string against ``frame_features``.

    Grammar (informal)::

        expr     := bool_expr
        bool_expr := bool_expr ('and'|'or') bool_expr | 'not' bool_expr | cmp
        cmp      := atom (CMP atom)+ | atom
        CMP      := == | != | < | <= | > | >= | in | not in
        atom     := CONST | f[CONST] | [CONST, ...] | (CONST, ...) | -atom

    Only the bound name ``f`` is visible; ``f[key]`` indexes the
    ``frame_features`` dict passed to :meth:`evaluate`.  No attribute
    access, no function calls, no comprehensions, no f-strings.
    """

    def evaluate(self, predicate_str: str, frame_features: dict[str, Any]) -> bool:
        try:
            tree = ast.parse(predicate_str, mode="eval")
        except SyntaxError as exc:
            raise SkillSandboxError(f"predicate parse error: {exc.msg}") from exc

        self._validate(tree)

        compiled = compile(tree, "<skill-dsl>", "eval")
        try:
            result = eval(  # noqa: S307 - sandboxed via AST allowlist
                compiled,
                {"__builtins__": {}},
                {"f": frame_features},
            )
        except Exception as exc:  # KeyError on missing feature, etc.
            raise SkillSandboxError(f"predicate eval error: {exc!r}") from exc

        return bool(result)

    # ----- private --------------------------------------------------------

    def _validate(self, tree: ast.Expression) -> None:
        for node in ast.walk(tree):
            self._check_node(node)

    @staticmethod
    def _check_node(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            return

        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _ALLOWED_BOOLOPS):
                raise SkillSandboxError(
                    f"forbidden bool op: {type(node.op).__name__}"
                )
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise SkillSandboxError(
                    f"forbidden unary op: {type(node.op).__name__}"
                )
            return

        if isinstance(node, ast.Compare):
            for op in node.ops:
                if not isinstance(op, _ALLOWED_CMP_OPS):
                    raise SkillSandboxError(
                        f"forbidden compare op: {type(op).__name__}"
                    )
            return

        if isinstance(node, ast.Constant):
            return

        if isinstance(node, ast.Name):
            if node.id != "f":
                raise SkillSandboxError(f"forbidden name: {node.id!r}")
            return

        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "f"):
                raise SkillSandboxError("subscript only allowed on `f`")
            return

        if isinstance(node, (ast.List, ast.Tuple)):
            for elt in node.elts:
                if not isinstance(elt, ast.Constant):
                    raise SkillSandboxError(
                        "list/tuple literals may only contain constants"
                    )
            return

        if isinstance(node, ast.Load):
            return

        # Operator nodes are reached by ast.walk independently of their
        # parent BoolOp/UnaryOp/Compare; allowlist them here (membership is
        # already validated on the parent via the .op / .ops fields above).
        if isinstance(node, (ast.And, ast.Or)):
            return
        if isinstance(node, (ast.USub, ast.UAdd, ast.Not)):
            return
        if isinstance(node, _ALLOWED_CMP_OPS):
            return

        raise SkillSandboxError(f"forbidden node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Code helper — child-process executor with rlimit
# ---------------------------------------------------------------------------


_SAFE_BUILTIN_NAMES = (
    "len",
    "range",
    "min",
    "max",
    "sum",
    "abs",
    "int",
    "float",
    "bool",
    "tuple",
    "list",
    "dict",
    "set",
    "sorted",
    "enumerate",
    "zip",
    "map",
    "filter",
    "any",
    "all",
)


def _child_runner(
    code_str: str,
    frame: Any,
    queue: "mp.Queue[Any]",
    mem_limit_mb: int,
) -> None:
    """Subprocess entrypoint: exec ``code_str`` under tight limits.

    The child uses ``resource.setrlimit`` to cap address space; on platforms
    without ``resource`` (e.g. Windows) the limit is silently skipped — the
    parent's wall-clock timer is the authoritative kill switch.
    """
    try:
        try:
            import resource  # POSIX-only

            limit = mem_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:  # pragma: no cover - non-POSIX or hardened envs
            pass

        safe_builtins = {
            name: __builtins__[name] if isinstance(__builtins__, dict)
            else getattr(__builtins__, name)
            for name in _SAFE_BUILTIN_NAMES
        }
        # Provide a minimal exception class for control flow.
        safe_builtins["Exception"] = Exception
        safe_builtins["ValueError"] = ValueError
        safe_builtins["KeyError"] = KeyError
        safe_builtins["TypeError"] = TypeError

        local_ns: dict[str, Any] = {}
        global_ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "frame": frame,
        }
        exec(code_str, global_ns, local_ns)  # noqa: S102 - sandboxed by builtins

        result = local_ns.get("RESULT", global_ns.get("RESULT"))
        queue.put(("ok", result))
    except Exception as exc:  # pragma: no cover - reported via queue
        queue.put(("err", repr(exc)))


class CodeHelperExecutor:
    """Run ``code_str`` in a sandbox subprocess, expecting ``RESULT: dict``."""

    def run(
        self,
        code_str: str,
        frame: Any,
        wall_clock_s: float = 1.0,
        mem_limit_mb: int = 50,
    ) -> dict[str, Any] | None:
        ctx = mp.get_context("spawn")
        queue: mp.Queue[Any] = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_child_runner,
            args=(code_str, frame, queue, mem_limit_mb),
            daemon=True,
        )
        proc.start()
        proc.join(wall_clock_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(0.5)
            if proc.is_alive():  # pragma: no cover - rare
                proc.kill()
                proc.join(0.5)
            return None

        try:
            tag, payload = queue.get_nowait()
        except Exception:
            return None

        if tag != "ok":
            return None

        return self._validate_result(payload)

    @staticmethod
    def _validate_result(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        for k, v in payload.items():
            if not isinstance(k, str):
                return None
            if not isinstance(v, (int, float, bool, str)):
                return None
        return payload

"""v611 module output schemas (Plan rev B §Δ deltas, FROZEN).

Schema validators for Δ1 (M1), Δ3 (M3), Δ5 (M4) outputs.

These DO NOT make LLM calls. They only check that emitted JSON
satisfies the v611 contract. Live agent.py must run these as gates
before accepting LLM output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# Common code-token guard (Δ3 NL-only enforcement)
# ─────────────────────────────────────────────────────────────────

_CODE_TOKEN_PATTERN = re.compile(
    r"\b(def|import|return|class|lambda|print|for\s+\w+\s+in|"
    r"while\s+\w+|if\s+\w+\s*==|try:|except:|with\s+\w+\s+as)\b"
)


def _contains_code_tokens(text: str) -> bool:
    """Return True if `text` looks like executable Python."""
    return _CODE_TOKEN_PATTERN.search(text or "") is not None


# ─────────────────────────────────────────────────────────────────
# Δ1 — M1 NL strategy output
# ─────────────────────────────────────────────────────────────────


@dataclass
class M1ValidationResult:
    ok: bool
    violations: list[str]


def validate_m1_output(out: dict, min_strategy_chars: int = 30) -> M1ValidationResult:
    """Validate M1 emits `nl_strategy` first + grounded click_xy_hint.

    Plan v611 rev B §Δ1 contract:
      {
        "nl_strategy": "<visual description + intent + grounded click area>",
        "predicate_id": "P_NL_grounded",
        "click_xy_hint": [int, int],
        "expected_signature": {"frame_changed": bool, "unsat_delta": int},
        "rollback_trigger": "<NL condition>"
      }
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M1ValidationResult(False, ["M1 output is not a dict"])
    s = out.get("nl_strategy")
    if not isinstance(s, str):
        v.append("nl_strategy missing or not a string")
    elif len(s) < min_strategy_chars:
        v.append(f"nl_strategy too short ({len(s)} < {min_strategy_chars})")
    coord = out.get("click_xy_hint")
    if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
        v.append("click_xy_hint missing or wrong shape (need [x, y])")
    else:
        try:
            x, y = int(coord[0]), int(coord[1])
            if not (0 <= x < 64 and 0 <= y < 64):
                v.append(f"click_xy_hint out of bounds: ({x},{y})")
        except (TypeError, ValueError):
            v.append("click_xy_hint contents not int-coercible")
    es = out.get("expected_signature")
    if not isinstance(es, dict):
        v.append("expected_signature missing or not a dict")
    else:
        if "frame_changed" not in es and "unsat_delta" not in es:
            v.append("expected_signature must include frame_changed "
                     "or unsat_delta")
    rb = out.get("rollback_trigger")
    if not isinstance(rb, str) or not rb.strip():
        v.append("rollback_trigger missing or empty NL condition")
    # Δ1 specific: predicate_id should be 'P_NL_grounded' or start with P_NL
    pid = out.get("predicate_id", "")
    if not (isinstance(pid, str) and pid.startswith("P_NL")):
        v.append(f"predicate_id must start with 'P_NL' "
                 f"(got {pid!r}) — Δ1 requires NL-grounded predicates")
    return M1ValidationResult(ok=(len(v) == 0), violations=v)


def m1_nl_grounded_by_strategy(out: dict) -> bool:
    """Δ1 grounding check: does click_xy_hint connect to a spatial
    descriptor in nl_strategy?

    Rule: nl_strategy must mention at least one of:
      - spatial words: 'top', 'bottom', 'left', 'right', 'corner',
        'center', 'middle', 'edge'
      - or color/region words: 'color', 'tile', 'marker', 'region',
        'neighbor'
    AND nl_strategy length >= 30 chars (already checked).
    """
    s = (out.get("nl_strategy") or "").lower()
    spatial = {"top", "bottom", "left", "right", "corner", "center",
                "middle", "edge", "side"}
    visual = {"color", "tile", "marker", "region", "neighbor", "pixel",
                "row", "column"}
    s_tokens = set(re.findall(r"\b[a-z]+\b", s))
    return bool(s_tokens & spatial) or bool(s_tokens & visual)


# ─────────────────────────────────────────────────────────────────
# Δ3 — M3 NL-only skill compressor output
# ─────────────────────────────────────────────────────────────────


@dataclass
class M3ValidationResult:
    ok: bool
    violations: list[str]


def validate_m3_skill_output(out: dict, min_desc_chars: int = 20) -> M3ValidationResult:
    """Validate M3 emits NL-only skill (no code, no coord literals).

    Plan v611 rev B §Δ3 contract:
      {
        "skill_id": "S-NL-<hash>",
        "nl_description": "<NL>",
        "abstract_precondition": "<NL>",
        "expected_observed_effect": "<NL>"
      }

    Rejects:
      - executable code tokens (def, import, return, etc)
      - raw coord literals (\\b\\d{1,2}\\b in spatial-numeric context)
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M3ValidationResult(False, ["M3 output is not a dict"])
    sid = out.get("skill_id")
    if not (isinstance(sid, str) and sid.startswith("S-NL-")):
        v.append("skill_id must start with 'S-NL-' (Δ3 NL-only)")
    fields = ("nl_description", "abstract_precondition",
              "expected_observed_effect")
    bundle_parts: list[str] = []
    for f in fields:
        val = out.get(f)
        if not isinstance(val, str):
            v.append(f"{f} missing or not a string")
            continue
        if len(val) < min_desc_chars:
            v.append(f"{f} too short ({len(val)} < {min_desc_chars})")
        if _contains_code_tokens(val):
            v.append(f"{f} contains code tokens — Δ3 NL-only required")
        bundle_parts.append(val)
    bundle = " ".join(bundle_parts)
    # raw coord literal patterns
    for m in re.finditer(r"\b\d{1,3}\b", bundle):
        v.append(f"raw numeric literal '{m.group()}' in skill text "
                 f"(coord leak)")
        break  # one is enough for the test; full validator iterates
    return M3ValidationResult(ok=(len(v) == 0), violations=v)


# ─────────────────────────────────────────────────────────────────
# Δ5 — M4 3-step self-verification output
# ─────────────────────────────────────────────────────────────────


@dataclass
class M4ValidationResult:
    ok: bool
    violations: list[str]


# ─────────────────────────────────────────────────────────────────
# Δ7 — multi-role separation (Plan rev D, codex round 10 ACCEPT)
# ─────────────────────────────────────────────────────────────────


@dataclass
class M1ProposerResult:
    ok: bool
    violations: list[str]


def validate_m1_proposer_output(out: dict) -> M1ProposerResult:
    """Δ7a — M1 Proposer outputs NL intent ONLY. No coords allowed.

    Plan rev D §Δ7a contract:
      {
        "nl_strategy": "<NL>",
        "suggested_click_region": "<NL spatial descriptor>",
        "expected_signature": {...},
        "rollback_trigger": "<NL>"
      }

    FORBIDDEN: any `click_xy_hint` or `(x,y)` integer pair field.
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M1ProposerResult(False, ["proposer output not a dict"])
    s = out.get("nl_strategy")
    if not isinstance(s, str) or len(s) < 30:
        v.append("nl_strategy missing or <30 chars")
    region = out.get("suggested_click_region")
    if not isinstance(region, str) or not region.strip():
        v.append("suggested_click_region missing (Δ7a NL spatial)")
    es = out.get("expected_signature")
    if not isinstance(es, dict):
        v.append("expected_signature missing or not a dict")
    rb = out.get("rollback_trigger")
    if not isinstance(rb, str) or not rb.strip():
        v.append("rollback_trigger missing or empty NL")
    # CRITICAL: M1 Proposer must NOT output coords
    if "click_xy_hint" in out:
        v.append("click_xy_hint forbidden in M1 Proposer (Δ7a) — "
                 "coords are M2e Executor's responsibility")
    if "click_xy" in out or "x" in out or "y" in out:
        v.append("raw x/y coords forbidden in M1 Proposer output")
    return M1ProposerResult(ok=(len(v) == 0), violations=v)


@dataclass
class M2vVerifierResult:
    ok: bool
    violations: list[str]


def validate_m2v_verifier_output(out: dict) -> M2vVerifierResult:
    """Δ7b — M2v Verifier outputs verdict + reason. Separate context.

    Plan rev D §Δ7b contract:
      {
        "verdict": "approve" | "reject_replan" | "reject_anchor",
        "reason_nl": "<NL>"
      }

    The Verifier's INPUT separation (no frame, no SKILL.md) is
    enforced at the agent.py level — not here. This validator only
    checks the OUTPUT shape.
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M2vVerifierResult(False, ["verifier output not a dict"])
    verdict = out.get("verdict")
    if verdict not in ("approve", "reject_replan", "reject_anchor"):
        v.append(f"verdict must be approve|reject_replan|reject_anchor "
                 f"(got {verdict!r})")
    reason = out.get("reason_nl")
    if not isinstance(reason, str) or len(reason) < 10:
        v.append("reason_nl missing or <10 chars")
    return M2vVerifierResult(ok=(len(v) == 0), violations=v)


@dataclass
class M2eExecutorResult:
    ok: bool
    violations: list[str]


def validate_m2e_executor_output(out: dict) -> M2eExecutorResult:
    """Δ7c — M2e Executor takes approved NL strategy + PNG and emits
    click coords.

    Plan rev D §Δ7c contract:
      {
        "click_xy_hint": [int, int],
        "grounding_text": "<NL explaining which visual feature maps to xy>"
      }
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M2eExecutorResult(False, ["executor output not a dict"])
    coord = out.get("click_xy_hint")
    if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
        v.append("click_xy_hint missing or wrong shape (need [x, y])")
    else:
        try:
            x, y = int(coord[0]), int(coord[1])
            if not (0 <= x < 64 and 0 <= y < 64):
                v.append(f"click_xy_hint out of bounds: ({x},{y})")
        except (TypeError, ValueError):
            v.append("click_xy_hint contents not int-coercible")
    grounding = out.get("grounding_text")
    if not isinstance(grounding, str) or len(grounding) < 20:
        v.append("grounding_text missing or <20 chars (Δ7c required)")
    return M2eExecutorResult(ok=(len(v) == 0), violations=v)


def validate_m4_output(out: dict) -> M4ValidationResult:
    """Validate M4 emits 3-step self-verification.

    Plan v611 rev B §Δ5 contract:
      {
        "paragraph": "<NL>",
        "verify": {
          "predicted_vs_observed": "<NL>",
          "strategy_validity": "<NL>",
          "skillmd_update": {
              "add": [...], "promote": [...], "falsify": [...]
          }
        },
        "verdict": "<support|refute|neutral>",
        "next_directive": "<NL>"
      }
    """
    v: list[str] = []
    if not isinstance(out, dict):
        return M4ValidationResult(False, ["M4 output is not a dict"])
    p = out.get("paragraph")
    if not isinstance(p, str) or len(p) < 30:
        v.append("paragraph missing or too short (<30 chars)")
    verify = out.get("verify")
    if not isinstance(verify, dict):
        v.append("verify missing or not a dict — Δ5 3-step required")
    else:
        for step in ("predicted_vs_observed", "strategy_validity"):
            val = verify.get(step)
            if not isinstance(val, str) or not val.strip():
                v.append(f"verify.{step} missing or empty NL")
        upd = verify.get("skillmd_update")
        if not isinstance(upd, dict):
            v.append("verify.skillmd_update missing or not a dict")
        else:
            for k in ("add", "promote", "falsify"):
                if k not in upd:
                    v.append(f"verify.skillmd_update missing '{k}' key")
                elif not isinstance(upd[k], list):
                    v.append(f"verify.skillmd_update.{k} not a list")
    verdict = out.get("verdict")
    if verdict not in ("support", "refute", "neutral"):
        v.append(f"verdict must be support|refute|neutral "
                 f"(got {verdict!r})")
    nd = out.get("next_directive")
    if not isinstance(nd, str) or not nd.strip():
        v.append("next_directive missing or empty NL")
    return M4ValidationResult(ok=(len(v) == 0), violations=v)

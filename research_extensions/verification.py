from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.templates.arcgentica_research.agent import ArcgenticaResearch

from .config import load_research_config
from .modules.dreamcoder import DreamCoderModule


FORBIDDEN_PROMPT_MARKERS = (
    "play.py",
    "multi_step",
    "restore_snapshot",
    "adapter.snapshot",
)


@dataclass(slots=True)
class CheckResult:
    ok: bool
    detail: str


def verify_prompt_contract() -> dict[str, Any]:
    prompt = ArcgenticaResearch._default_system_prompt()
    required_checks = {
        "mentions_helper_tools": "`history`" in prompt and "`memories_summaries`, `memories_get`, `memories_add`, and `memories_query`" in prompt,
        "mentions_overlay_truthfully": "agent-side research overlays" in prompt,
        "mentions_world_update": "`world_update`" in prompt,
        "mentions_open_skill_schema": "You choose the shape of that object and the terms " in prompt,
        "mentions_submit_action": "`submit_action` is this host's action commit tool" in prompt,
        "mentions_winning_source": "source='winning'" in prompt,
    }
    forbidden_hits = [marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in prompt]
    return {
        "ok": all(required_checks.values()) and not forbidden_hits,
        "required_checks": required_checks,
        "forbidden_hits": forbidden_hits,
    }


def find_latest_summary_for_namespace(
    log_dir: Path, namespace: str, game_prefix: str | None = None
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in log_dir.rglob("summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("shared_namespace") != namespace:
            continue
        if game_prefix and not any(part.startswith(game_prefix) for part in path.parts):
            continue
        candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None, None
    _, best_path, best_data = max(candidates, key=lambda item: item[0])
    return best_path, best_data


def analyze_dreamcoder_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "ok": False, "reason": "missing dreamcoder_library.json"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": True, "ok": False, "reason": f"invalid json: {exc}"}
    if not isinstance(raw, list):
        return {"present": True, "ok": False, "reason": "library is not a list"}

    forbidden = 0
    action_grounded = 0
    structured = 0
    hierarchical = 0
    exact_spines = 0
    revised = 0
    surprise_linked = 0
    duplicate_pairs = 0
    names: list[str] = []
    signatures: list[set[str]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload", {})
        name = str(payload.get("name", "")).strip()
        names.append(name)
        if re.search(r"(^|[-_\\s])(level|solve)([-_\\s]|$)", name, flags=re.IGNORECASE):
            forbidden += 1
        items = DreamCoderModule._payload_as_items(payload)
        grounded = False
        for entry in items:
            text = str(entry)
            if re.search(r"\bACTION[1-7]\b|\bRESET\b|skill:[A-Za-z0-9_./\\\\-]+", text):
                grounded = True
                break
        if grounded:
            action_grounded += 1
        if DreamCoderModule._structural_field_count(payload) > 0:
            structured += 1
        if payload.get("subskills"):
            hierarchical += 1
        if payload.get("action_spine"):
            exact_spines += 1
        if int(item.get("revision_count", 0)) > 0:
            revised += 1
        if int(item.get("times_linked_to_surprise", 0)) > 0:
            surprise_linked += 1
        signatures.append(DreamCoderModule._signature_tokens(payload))

    for idx, left in enumerate(signatures):
        if not left:
            continue
        for right in signatures[idx + 1 :]:
            if not right:
                continue
            overlap = len(left & right) / max(1, len(left | right))
            if overlap >= 0.8:
                duplicate_pairs += 1

    total = len(raw)
    grounded_ratio = (action_grounded / total) if total else 0.0
    structured_ratio = (structured / total) if total else 0.0
    return {
        "present": True,
        "ok": total > 0 and forbidden == 0 and grounded_ratio >= 0.5 and structured > 0,
        "skill_count": total,
        "forbidden_name_count": forbidden,
        "action_grounded_count": action_grounded,
        "action_grounded_ratio": grounded_ratio,
        "structured_skill_count": structured,
        "structured_skill_ratio": structured_ratio,
        "hierarchical_skill_count": hierarchical,
        "exact_spine_skill_count": exact_spines,
        "revised_skill_count": revised,
        "surprise_linked_skill_count": surprise_linked,
        "near_duplicate_pairs": duplicate_pairs,
        "skill_names": names,
    }


def analyze_world_model_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "ok": False, "reason": "missing world_model.json"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": True, "ok": False, "reason": f"invalid json: {exc}"}

    drafts = raw.get("world_drafts", [])
    transitions = raw.get("transitions", {})
    code_like_drafts = 0
    revised_drafts = 0
    empirically_scored = 0
    simulator_scored = 0
    for draft in drafts:
        if not isinstance(draft, dict):
            continue
        body = str(draft.get("body", ""))
        if "```" in body or re.search(r"\b(if|elif|else|return|def|for|while)\b", body):
            code_like_drafts += 1
        if int(draft.get("revisions", 0)) > 1:
            revised_drafts += 1
        if int(draft.get("empirical_matches", 0)) + int(draft.get("empirical_mismatches", 0)) > 0:
            empirically_scored += 1
        if int(draft.get("simulator_matches", 0)) + int(draft.get("simulator_mismatches", 0)) > 0:
            simulator_scored += 1

    transition_action_count = sum(
        len(actions) for actions in transitions.values() if isinstance(actions, dict)
    )
    prior_suggestions = int(raw.get("prior_suggestions", 0))
    prior_overrides = int(raw.get("prior_overrides", 0))
    structured_predictions = int(raw.get("structured_predictions", 0))
    return {
        "present": True,
        "ok": bool(drafts)
        and code_like_drafts > 0
        and empirically_scored > 0
        and simulator_scored > 0,
        "draft_count": len(drafts),
        "code_like_draft_count": code_like_drafts,
        "revised_draft_count": revised_drafts,
        "empirically_scored_draft_count": empirically_scored,
        "simulator_scored_draft_count": simulator_scored,
        "transition_state_count": len(transitions),
        "transition_action_count": transition_action_count,
        "prior_suggestions": prior_suggestions,
        "prior_overrides": prior_overrides,
        "structured_predictions": structured_predictions,
        "scored_prediction_matches": int(raw.get("scored_prediction_matches", 0)),
        "scored_prediction_mismatches": int(raw.get("scored_prediction_mismatches", 0)),
    }


def analyze_meta_harness_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "ok": False, "reason": "missing meta_harness.json"}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"present": True, "ok": False, "reason": f"invalid json: {exc}"}
    run_history = raw.get("run_history", [])
    active_keys = raw.get("active_keys", [])
    last_metrics = run_history[-1].get("metrics", {}) if run_history else {}
    metrics_have_level_leak = any("level" in str(key).lower() for key in last_metrics)
    return {
        "present": True,
        "ok": bool(run_history) and bool(active_keys) and not metrics_have_level_leak,
        "run_count": len(run_history),
        "active_overlay_count": len(active_keys),
        "last_metrics": last_metrics,
        "metrics_have_level_leak": metrics_have_level_leak,
    }


def verify_run_intent(
    *,
    repo_root: Path,
    namespace: str,
    game_prefix: str = "ls20",
    config_path: str | Path | None = None,
    expected_actions: int | None = None,
    verify_prompt_contract_flag: bool = True,
) -> dict[str, Any]:
    cfg = load_research_config(config_path)
    log_dir = repo_root / cfg.log_dir
    summary_path, summary = find_latest_summary_for_namespace(log_dir, namespace, game_prefix)
    shared_dir = log_dir / "shared" / f"{game_prefix}-cb3b57cc" / namespace
    if not shared_dir.exists():
        alt_dirs = list((log_dir / "shared").glob(f"{game_prefix}*"))
        for candidate in alt_dirs:
            probe = candidate / namespace
            if probe.exists():
                shared_dir = probe
                break

    prompt_contract = (
        verify_prompt_contract()
        if verify_prompt_contract_flag
        else {"ok": True, "required_checks": {}, "forbidden_hits": []}
    )
    checks: dict[str, Any] = {"prompt_contract": prompt_contract}

    contract_ok = summary is not None
    summary_checks: dict[str, CheckResult] = {}
    if summary is None:
        summary_checks["summary_present"] = CheckResult(False, "no summary found")
    else:
        expected_modules = cfg.active_modules()
        summary_checks["active_modules_match"] = CheckResult(
            summary.get("active_modules") == expected_modules,
            f"expected {expected_modules}, got {summary.get('active_modules')}",
        )
        summary_checks["namespace_match"] = CheckResult(
            summary.get("shared_namespace") == namespace,
            f"expected {namespace}, got {summary.get('shared_namespace')}",
        )
        if expected_actions is not None:
            summary_checks["action_budget_match"] = CheckResult(
                0 <= int(summary.get("actions", -1)) <= expected_actions,
                f"expected <= {expected_actions}, got {summary.get('actions')}",
            )
        contract_ok = contract_ok and all(item.ok for item in summary_checks.values())
    checks["summary"] = {
        "path": str(summary_path) if summary_path else None,
        "data": summary,
        "checks": {key: asdict(value) for key, value in summary_checks.items()},
        "ok": contract_ok,
    }

    active_modules = set(cfg.active_modules())
    if "dreamcoder" in active_modules:
        checks["dreamcoder"] = analyze_dreamcoder_state(shared_dir / "dreamcoder_library.json")
    if "world_model" in active_modules:
        checks["world_model"] = analyze_world_model_state(shared_dir / "world_model.json")
    if "meta_harness" in active_modules:
        checks["meta_harness"] = analyze_meta_harness_state(shared_dir / "meta_harness.json")

    module_checks = [
        value.get("ok", True)
        for key, value in checks.items()
        if key not in {"prompt_contract", "summary"}
    ]
    overall_ok = prompt_contract["ok"] and contract_ok and all(module_checks)
    return {
        "ok": overall_ok,
        "repo_root": str(repo_root),
        "namespace": namespace,
        "game_prefix": game_prefix,
        "config_path": str(config_path) if config_path is not None else None,
        "checks": checks,
    }

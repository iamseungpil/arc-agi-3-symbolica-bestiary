"""v602 typed `skill_state.json` substrate (plan v602 §2).

Canonical write target for skill discovery: confirmed mechanics, active
hypotheses, falsifications, and skill lifecycle records. Persists via JSON
serializer with `schema_version=1`. Mutated only via structured patches
(see `apply_patch`); never hand-edited.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2  # v607: Bayesian Beta posterior fields + Reflector chid_template emit


# =============================================================================
# Dataclasses (per plan v602 §2 SkillState schema)
# =============================================================================

@dataclass
class SkillStateMetadata:
    schema_version: int = SCHEMA_VERSION
    game_id: str = ""
    last_updated: str = ""
    cross_run_imported: bool = False  # one-time migration flag (plan §7)
    # v608f-fix: telemetry dict (repeats_fired / repeat_regions / etc).
    # Persisted via `asdict` so post-cycle analysis scripts can read it
    # from skill_state.json without instrumenting the live process.
    v608f_counters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmedMechanic:
    id: str  # A001, A002, ...
    natural_language: str
    evidence_runs: list[str] = field(default_factory=list)
    first_seen_run: str = ""
    first_seen_turn: int = -1
    confirmation_count: int = 0
    paired_cf_offsets: list[int] = field(default_factory=list)


@dataclass
class ActiveHypothesis:
    id: str  # H_<run_id>_<turn>
    predicate_id: str
    region_hint: str
    required_pre_state: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    status: Literal["under_test", "confirmed", "refuted", "inconclusive"] = "under_test"
    proposed_at_turn: int = -1


@dataclass
class Falsification:
    id: str  # F_<run_id>_<turn>
    natural_language: str
    failure_mode: Literal[
        "coord_miss", "mechanic_absence", "wrong_state", "schema_violation"
    ] = "wrong_state"
    evidence: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    turn: int = -1


@dataclass
class SkillRecord:
    id: str  # S00-S12 static, Sext-<sha> dynamic, Sref-<sha> Reflector-emitted (v607)
    family: str
    description: str
    is_static: bool = True
    install_run: str | None = None
    confirmed_count: int = 0
    falsified_count: int = 0
    last_used_run: str | None = None
    used_in_successful_L_plus: int = 0
    status: Literal["active", "pending_eviction", "retired"] = "active"
    # v607 Arm 1 (B+C): Reflector verbal chid_template fields
    chid_template: str = ""  # e.g. "P_<verb>_<noun>_R{region_id}"; empty for static/legacy
    beta_alpha: float = 0.5  # Jeffreys prior; success count + 0.5
    beta_beta: float = 0.5   # Jeffreys prior; failure count + 0.5
    cooldown_remaining: int = 0  # turns until Reflector may re-emit related family
    emit_count: int = 0  # how many times Proposer instantiated this template
    emit_tokens: int = 0  # cumulative L_LLM cost attributed to this skill
    last_emit_turn: int = -1


@dataclass
class SkillCard:
    """v608 card-ledger record rendered into SKILL.md.

    Cards are operational: they must name observable state features,
    predictions, falsifiers, and policy hooks so Reflector updates remain
    testable instead of prose-only.
    """
    id: str
    card_type: Literal["mechanic", "strategy", "hypothesis"]
    status: Literal["draft", "active", "narrowed", "falsified", "retired"] = "draft"
    claim: str = ""
    state_features: list[str] = field(default_factory=list)
    predicts: list[str] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    policy_hooks: list[str] = field(default_factory=list)
    support_count: int = 0
    refute_count: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # v608d: optional fields. `auto_draft` marks cards created by the
    # atomic-citation auto-draft path so the sleep compressor can prefer
    # human-curated seeds. `last_predicted` carries the most recent
    # next-state prediction payload for the prediction-vs-observation
    # comparator in the next turn.
    auto_draft: bool = False
    last_predicted: dict[str, Any] | None = None


@dataclass
class SkillMetrics:
    skills_proposed: int = 0
    skills_confirmed: int = 0
    skills_retired: int = 0
    proposal_to_confirmation_rate: float = 0.0
    median_episodes_to_confirmation: float | None = None
    skill_reuse_rate: float = 0.0
    successful_L_plus_with_discovered_skill_rate: float = 0.0
    skill_promotion_to_1A_rate: float = 0.0


@dataclass
class SkillState:
    """v602 canonical skill substrate."""
    metadata: SkillStateMetadata = field(default_factory=SkillStateMetadata)
    confirmed_mechanics: list[ConfirmedMechanic] = field(default_factory=list)
    active_hypotheses: list[ActiveHypothesis] = field(default_factory=list)
    falsifications: list[Falsification] = field(default_factory=list)
    skill_lifecycle: list[SkillRecord] = field(default_factory=list)
    cards: list[SkillCard] = field(default_factory=list)
    metrics: SkillMetrics = field(default_factory=SkillMetrics)

    # ----------------------------------------------------------------- I/O

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys, indent=2)."""
        return json.dumps(asdict(self), sort_keys=True, indent=2,
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "SkillState":
        d = json.loads(payload)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "SkillState":
        meta = SkillStateMetadata(**d.get("metadata", {}))
        cm = [ConfirmedMechanic(**c) for c in d.get("confirmed_mechanics", [])]
        ah = [ActiveHypothesis(**a) for a in d.get("active_hypotheses", [])]
        fals = [Falsification(**f) for f in d.get("falsifications", [])]
        # v607: explicit field-by-field SkillRecord construction with v1 backfill
        # so legacy v1 records (without beta_alpha/chid_template/etc.) still load.
        sl: list[SkillRecord] = []
        _sr_defaults = {
            "chid_template": "", "beta_alpha": 0.5, "beta_beta": 0.5,
            "cooldown_remaining": 0, "emit_count": 0,
            "emit_tokens": 0, "last_emit_turn": -1,
        }
        for s in d.get("skill_lifecycle", []):
            row = dict(s)
            for k, v in _sr_defaults.items():
                row.setdefault(k, v)
            sl.append(SkillRecord(**row))
        cards = [SkillCard(**c) for c in d.get("cards", [])]
        m = SkillMetrics(**d.get("metrics", {}))
        return cls(metadata=meta, confirmed_mechanics=cm,
                   active_hypotheses=ah, falsifications=fals,
                   skill_lifecycle=sl, cards=cards, metrics=m)

    def touch(self) -> None:
        """Update last_updated timestamp."""
        self.metadata.last_updated = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Structured patches (plan v602 §4 Reflector)
# =============================================================================

@dataclass
class PromoteA1Patch:
    """Promote a paired_cf-supported observation to a confirmed_mechanic (1A)."""
    op: Literal["promote_1A"] = "promote_1A"
    natural_language: str = ""
    pcf_offsets: list[int] = field(default_factory=list)
    evidence_runs: list[str] = field(default_factory=list)
    first_seen_run: str = ""
    first_seen_turn: int = -1


@dataclass
class ProposeDynamicSkillPatch:
    """Install a dynamic skill (S.dynamic family)."""
    op: Literal["propose_dynamic_skill"] = "propose_dynamic_skill"
    family: str = "extended"
    description: str = ""
    lambda_body: str = ""
    coord_policy: Literal["centroid", "sprite_center", "corner_top_left"] = "centroid"
    install_run: str = ""


@dataclass
class RetireSkillPatch:
    op: Literal["retire_skill"] = "retire_skill"
    skill_id: str = ""
    reason: str = ""
    evidence: list[int] = field(default_factory=list)


@dataclass
class FalsifyPatch:
    op: Literal["falsify"] = "falsify"
    hypothesis_id: str = ""
    natural_language: str = ""
    failure_mode: str = "wrong_state"


@dataclass
class MergeHypothesisPatch:
    """Merge two active hypotheses into one (one becomes the canonical)."""
    op: Literal["merge_hypothesis"] = "merge_hypothesis"
    keep_id: str = ""
    drop_id: str = ""
    rationale: str = ""


@dataclass
class EmitChidTemplatePatch:
    """v607 Arm 1 (B+C): Reflector emits a VERBAL chid_template (no code).

    Distinct from ProposeDynamicSkillPatch (Arm 2 = code-skill). This patch
    appends a new SkillRecord with chid_template populated, Beta(0.5, 0.5)
    Jeffreys prior, and Reflector emit cost telemetry attribution.
    """
    op: Literal["emit_chid_template"] = "emit_chid_template"
    chid_template: str = ""  # e.g. "P_<verb>_<noun>_R{region_id}"
    family: str = "reflector"
    description: str = ""
    emit_run: str = ""
    emit_turn: int = -1
    emit_tokens: int = 0  # tokens consumed by Reflector LLM call that produced this
    cooldown: int = 5  # initial cooldown_remaining


# Union type for documentation / type hints.
Patch = (
    PromoteA1Patch | ProposeDynamicSkillPatch | RetireSkillPatch
    | FalsifyPatch | MergeHypothesisPatch | EmitChidTemplatePatch
)


@dataclass
class PatchResult:
    accepted: bool
    reason: str  # "ok", "promote_insufficient_evidence", "skill_id_unknown", ...


# =============================================================================
# Applier — validates + mutates SkillState
# =============================================================================


def _next_a_id(state: SkillState) -> str:
    n = len(state.confirmed_mechanics) + 1
    return f"A{n:03d}"


def apply_patch(state: SkillState, patch: Patch) -> PatchResult:
    """Validate + apply a structured patch to skill_state.

    Returns PatchResult.accepted = True iff patch was applied.

    Validation rules (codex round 1 §10 killer concerns):
      - PromoteA1Patch: requires len(pcf_offsets) >= 2 (avoid weak-evidence promotion)
      - RetireSkillPatch: requires skill_id present in lifecycle
      - FalsifyPatch: requires hypothesis_id present in active_hypotheses
      - MergeHypothesisPatch: requires both ids present in active_hypotheses
      - ProposeDynamicSkillPatch: requires non-empty lambda_body + family
    """
    if isinstance(patch, PromoteA1Patch):
        if len(patch.pcf_offsets) < 2:
            return PatchResult(False, "promote_insufficient_evidence")
        # codex final-gate Q5: require DISTINCT paired_cf evidence; duplicate
        # offsets pointing to same entry are weak evidence even if count >= 2
        if len(set(patch.pcf_offsets)) < 2:
            return PatchResult(False, "promote_insufficient_distinct_evidence")
        new_id = _next_a_id(state)
        state.confirmed_mechanics.append(ConfirmedMechanic(
            id=new_id, natural_language=patch.natural_language,
            evidence_runs=list(patch.evidence_runs),
            first_seen_run=patch.first_seen_run,
            first_seen_turn=patch.first_seen_turn,
            confirmation_count=len(patch.pcf_offsets),
            paired_cf_offsets=list(patch.pcf_offsets),
        ))
        state.metrics.skill_promotion_to_1A_rate = (
            (state.metrics.skill_promotion_to_1A_rate
             * max(0, state.metrics.skills_confirmed - 1)
             + 1.0)
            / max(1, state.metrics.skills_confirmed)
        ) if state.metrics.skills_confirmed > 0 else 0.0
        state.touch()
        return PatchResult(True, "ok")

    if isinstance(patch, ProposeDynamicSkillPatch):
        if not patch.lambda_body.strip():
            return PatchResult(False, "missing_lambda_body")
        if not patch.family.strip():
            return PatchResult(False, "missing_family")
        # build deterministic id from lambda body fingerprint
        import hashlib
        sha = hashlib.sha256(patch.lambda_body.encode("utf-8")).hexdigest()[:8]
        sid = f"Sext-{sha}"
        # de-duplicate by id
        if any(s.id == sid for s in state.skill_lifecycle):
            return PatchResult(False, "duplicate_skill_id")
        state.skill_lifecycle.append(SkillRecord(
            id=sid, family=patch.family, description=patch.description,
            is_static=False, install_run=patch.install_run,
            confirmed_count=0, falsified_count=0,
            status="active",
        ))
        state.metrics.skills_proposed += 1
        state.touch()
        return PatchResult(True, "ok")

    if isinstance(patch, RetireSkillPatch):
        for s in state.skill_lifecycle:
            if s.id == patch.skill_id:
                s.status = "retired"
                state.metrics.skills_retired += 1
                state.touch()
                return PatchResult(True, "ok")
        return PatchResult(False, "skill_id_unknown")

    if isinstance(patch, FalsifyPatch):
        for h in state.active_hypotheses:
            if h.id == patch.hypothesis_id:
                h.status = "refuted"
                state.falsifications.append(Falsification(
                    id=f"F_{patch.hypothesis_id}",
                    natural_language=patch.natural_language,
                    failure_mode=patch.failure_mode,  # type: ignore[arg-type]
                    evidence={},
                    run_id="",
                    turn=h.proposed_at_turn,
                ))
                state.touch()
                return PatchResult(True, "ok")
        return PatchResult(False, "hypothesis_id_unknown")

    if isinstance(patch, MergeHypothesisPatch):
        keep = next((h for h in state.active_hypotheses if h.id == patch.keep_id), None)
        drop = next((h for h in state.active_hypotheses if h.id == patch.drop_id), None)
        if keep is None or drop is None:
            return PatchResult(False, "merge_id_unknown")
        # remove drop; keep stays
        state.active_hypotheses = [
            h for h in state.active_hypotheses if h.id != patch.drop_id
        ]
        state.touch()
        return PatchResult(True, "ok")

    if isinstance(patch, EmitChidTemplatePatch):
        # v607 Arm 1 (B+C): verbal chid_template from Reflector.
        if not patch.chid_template.strip():
            return PatchResult(False, "missing_chid_template")
        import hashlib
        sha = hashlib.sha256(
            (patch.chid_template + str(patch.emit_turn)).encode("utf-8")
        ).hexdigest()[:8]
        sid = f"Sref-{sha}"
        if any(s.id == sid for s in state.skill_lifecycle):
            return PatchResult(False, "duplicate_skill_id")
        state.skill_lifecycle.append(SkillRecord(
            id=sid, family=patch.family,
            description=patch.description or patch.chid_template,
            is_static=False, install_run=patch.emit_run,
            confirmed_count=0, falsified_count=0,
            status="active",
            chid_template=patch.chid_template,
            beta_alpha=0.5, beta_beta=0.5,  # Jeffreys prior
            cooldown_remaining=patch.cooldown,
            emit_count=0,
            emit_tokens=patch.emit_tokens,
            last_emit_turn=patch.emit_turn,
        ))
        state.metrics.skills_proposed += 1
        state.touch()
        return PatchResult(True, "ok")

    return PatchResult(False, "unknown_patch_type")


# =============================================================================
# Atomic file write (plan v602 §11 codex final-gate refinement #8)
# =============================================================================


def atomic_write_text(path: str | os.PathLike, content: str) -> None:
    """Atomic write: temp file + fsync + rename.

    Ensures readers never see a partial file even under concurrent writes.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync may not be supported on all FS (e.g., tmpfs); soft-fail
                pass
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def save_state(state: SkillState, path: str | os.PathLike) -> None:
    """Write skill_state.json atomically."""
    state.touch()
    atomic_write_text(path, state.to_json())


def load_state(path: str | os.PathLike) -> SkillState | None:
    p = Path(path)
    if not p.exists():
        return None
    return SkillState.from_json(p.read_text(encoding="utf-8"))

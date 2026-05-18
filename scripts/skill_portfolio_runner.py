#!/usr/bin/env python3
"""Consolidated skill-variation portfolio runner.

Implements ``reports/PLAN_skill_portfolio_2026_05_17.md`` (rev4, r6 Option-A).
All path / JSON / classification logic lives in THIS module; the shell
launcher (``scripts/skill_portfolio.sh``) is a thin ≤40-line preflight + exec.

================================================================================
rev4-r6 M3 watchdog + bounded retry (Option-A)
================================================================================
The post-completion orchestrator-quiescence wedge (the UNFIXED M3 bug) can
hold a faithful arm open for the full wall+grace window even though the
agent reached the max level and is doing zero further positive work. r6 adds
a cooperative, stdlib-only watchdog that OBSERVES our own launched process
tree and the append-only proxy / recorder logs and, once a ``stream_complete``
event has been seen (so the quiescence signature is valid only
post-stream-complete), declares ``orchestrator_stall`` when ALL of
(no new proxy stream_request, server-log size frozen, recorder row-count +
max-level frozen, proc-tree CPU jiffies delta == 0) hold continuously for a
predeclared quiescence window. The watchdog is a faithful-failure
ACCELERATOR, not a measurability claim: it only converts a wedge that WOULD
have ended as ``orchestrator_stall`` at the wall into the same status sooner;
it never invents a verdict and it NEVER fires before stream_complete or on a
genuinely progressing long run (any CPU / log / recorder progress resets the
window). A healthy (non-wedged) episode is byte-identical to the pre-r6
behavior because the watchdog stays passive until ALL of those conditions
hold for the full window post-stream-complete.

C2 (codex r1) — frozen-faithful enforcement is the runner's OWN preflight
G2 HARD-ABORT over the consumed copy ``/tmp/upstream_arcagi3`` plus the G3
``PINNED_RUNNER_SHA`` / ``PINNED_PROXY_SHA`` sha256 gate; the in-repo
``agents/templates/agentica*`` G1 probe is RECORDED-only by design, so a
dirty in-repo ``agentica_lite/`` (separate v611 work, not in the consumed
faithful path) does NOT affect faithful-arm integrity.

``m3_unmeasurable`` (summary.json) is definitive ONLY when True: it asserts
that across every attempted seed and every bounded retry at least one arm
terminated in ``orchestrator_stall`` / ``m3_nonfaithful`` so NOT A SINGLE
valid pair was obtainable — the M3 wedge made this idea unmeasurable on the
faithful harness, full stop. A False value is merely "at least one valid
pair was obtained" and carries no M3 claim.

The bounded same-seed retry is a PREDECLARED harness-reliability policy
applied IDENTICALLY to both arms (arm-agnostic at the run_idea level): when
either arm of a pair terminates outside {normal, max_episode_budget} the
WHOLE pair (both arms, same seed, same faithful / S0 path) is re-run up to
``SKILL_M3_RETRY_N`` times. It is not an M3 remediation and does not fold
into ``valid_pair_count``; the per-arm M3 / stall rate is accumulated over
ALL attempts and reported SEPARATELY (``m3_rate``) so the retry can never
launder an M3 wedge into a valid datum.

================================================================================
rev4 §4 contract binding decision (resolves the field-name divergence)
================================================================================
The plan §4 names four required output families per idea:

    manifest.jsonl      {idea_id, arm, seed, order_index, game_id, model,
                          store_sha, code_sha, ports, start_time, end_time}
    metrics.jsonl       {levels_completed, env_actions, proxy_requests,
                          wall_seconds, terminal_status}
    faithfulness.jsonl  {m3_flag, m3_reason, frozen_config_sha, prompt_sha,
                          skill_store_sha}
    summary.json        {valid_pair_count, per-arm terminal counts,
                          M3 incidence delta, classifier_result}

The plan §3 paired-delta classifier, however, reads per-episode fields named
``levels`` / ``env_actions`` / ``proxy_requests`` and a ``terminal_status``,
keyed by paired seed. These two views are the SAME data under two names. To
remove that divergence WITHOUT mutating the rev4 §4 quartet, this runner ALSO
emits a single denormalized ``episodes.jsonl`` that is the deterministic
INNER JOIN of manifest.jsonl ⋈ metrics.jsonl ⋈ faithfulness.jsonl keyed by
``(idea_id, seed, arm, order_index)``. ``episodes.jsonl`` is the ONLY input
the paired-delta + terminal classifiers consume. The §4 quartet remains the
authoritative code-reviewable contract; ``episodes.jsonl`` is a derived,
reproducible projection (pure function of the quartet — re-running the join on
the same quartet yields a byte-stable file modulo dict ordering).

Per-episode field source-of-truth decisions (faithful vs S0):

  levels_completed  faithful: summary.result.levels_completed
                    S0:       report.levels_completed
  env_actions       faithful: summary.result.recorder_action_count
                              (authoritative recorded env-action count; the
                               recorder is the positive-work signal per the
                               runner docstring, not the proxy jsonl)
                    S0:       report.action_count
  proxy_requests    faithful: summary.result.request_count
                    S0:       report.request_count
  wall_seconds      faithful: summary.result.runtime_seconds
                    S0:       report.runtime_seconds
  terminal_status   computed by the terminal classifier (this module) from
                    summary.result.termination_reason + fuse + soft_deadline +
                    the D5 watchdog stderr signal (faithful) / rc + report +
                    stderr traceback (S0).

DECISION_RULE_VERSION is pinned and stamped into summary.json.

================================================================================
Coexistence with in-flight ad-hoc experiments (read-only — never killed)
================================================================================
The faithful arm uses a DISJOINT locked port pool (base 9500 proxy / 2800
server, stride 4 per slot) — never the 9400-9426 / 2700-2726 range held by the
in-flight pilots. The single S0 {9091,2345} slot is taken ONLY via a global
flock and ONLY while no other holder has it. S0 cleanup is SCOPED
(``fuser -k`` on the two ports + tracked child PIDs only); there is NO
broad process-name kill (no global uvicorn / agentica-server / s0_smoke
sweep) anywhere in this module (the recurring bug — serialization via the
S0 flock makes any such broad kill unnecessary).
"""
from __future__ import annotations

import argparse
import dataclasses
import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants / pinned invariants (rev4 §0, §3)
# --------------------------------------------------------------------------- #

DECISION_RULE_VERSION = "rev4-r6-M3watchdog-OptionA"

REPO_ROOT = Path("/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research")

# Frozen-pinned faithful SHAs (rev4 §0; also asserted verbatim in
# /tmp/skill_pilot_paired.sh:17-18 and scripts/repin_supervisor.sh:G3).
PINNED_RUNNER_SHA = "56d7eb143d9494c350004b3c647854a284b43405ea2167ca96114477b00d74ab"
# Recomputed after the sanctioned M3 refusal-reroll proxy change
# (scripts/trapi_openai_proxy.py now carries _REFUSAL_REROLL_MAX / _REFUSAL_RE
# / quarantine + reroll). old=0d82e505f0c98ac8094be82cb867a97eea3647cb7bc569c5256f9f73c6009111
PINNED_PROXY_SHA = "32220fdc9e3d3864146c7b71b0a95afe988155d7474bb905db73ddb7f3e0433b"

FAITHFUL_RUNNER = "scripts/run_clean_upstream_ft09.py"
FAITHFUL_PROXY = "scripts/trapi_openai_proxy.py"
S0_SMOKE = "scripts/s0_smoke_ft09.py"

PY312_PYTHON = "/home/v-seungplee/miniconda3/envs/arcagi3-runner-py312/bin/python"
UPSTREAM_ROOT = "/tmp/upstream_arcagi3"

# Disjoint faithful port pool. Base 9500/2800, stride 4. NEVER 9400-9426 /
# 2700-2726 (in-flight ad-hoc pilots) and NEVER 9091/2345 (the S0 slot).
FAITHFUL_PROXY_PORT_BASE = 9500
FAITHFUL_SERVER_PORT_BASE = 2800
FAITHFUL_PORT_STRIDE = 4

# The single global S0 slot (serialized via flock — only one S0 arm at a time).
S0_PROXY_PORT = 9091
S0_SERVER_PORT = 2345

LOCK_DIR = REPO_ROOT / "reports" / "skill_portfolio" / ".locks"
OUT_ROOT = REPO_ROOT / "reports" / "skill_portfolio"

# Terminal-status enum (rev4 §3 taxonomy).
TS_NORMAL = "normal"
TS_MAX_EPISODE_BUDGET = "max_episode_budget"
TS_MAX_WALL_BUDGET = "max_wall_budget"
TS_M3_NONFAITHFUL = "m3_nonfaithful"
TS_ORCHESTRATOR_STALL = "orchestrator_stall"
TS_HARNESS_CRASH = "harness_crash"
TS_SHA_MISMATCH = "sha_mismatch"
TS_PORT_CONFLICT = "port_conflict"
TS_MISSING_METRICS = "missing_metrics"

ALL_TERMINAL_STATUSES = (
    TS_NORMAL,
    TS_MAX_EPISODE_BUDGET,
    TS_MAX_WALL_BUDGET,
    TS_M3_NONFAITHFUL,
    TS_ORCHESTRATOR_STALL,
    TS_HARNESS_CRASH,
    TS_SHA_MISMATCH,
    TS_PORT_CONFLICT,
    TS_MISSING_METRICS,
)

# Status precedence (rev4 §3 r3#2): highest wins when multiple signals fire.
# infra_error{...} > orchestrator_stall > max_wall_budget > m3_nonfaithful
# > max_episode_budget > normal.
INFRA_ERROR_STATUSES = (
    TS_HARNESS_CRASH,
    TS_SHA_MISMATCH,
    TS_PORT_CONFLICT,
    TS_MISSING_METRICS,
)
_PRECEDENCE = [
    # Infra terminals first; harness_crash beats the others when several
    # infra signals coincide (e.g. a traceback that ALSO trips a sha check).
    TS_HARNESS_CRASH,
    TS_SHA_MISMATCH,
    TS_PORT_CONFLICT,
    TS_MISSING_METRICS,
    TS_ORCHESTRATOR_STALL,
    TS_MAX_WALL_BUDGET,
    TS_M3_NONFAITHFUL,
    TS_MAX_EPISODE_BUDGET,
    TS_NORMAL,
]
_PRECEDENCE_RANK = {s: i for i, s in enumerate(_PRECEDENCE)}

# VALID-pair rule (rev4 §3 r5 Option-A): a paired seed is VALID iff BOTH arms
# have terminal_status ∈ this set AND both emit numeric frozen metrics AND
# both pass faithfulness/SHA. max_wall_budget / orchestrator_stall are NEVER
# valid (censored env-action counts would contaminate the speed metric).
VALID_TERMINAL_STATUSES = frozenset({TS_NORMAL, TS_MAX_EPISODE_BUDGET})

# Paired-delta classifier thresholds (rev4 §3).
ACTION_DELTA_THRESHOLD = 0.10  # median(ΔA) ≥ 10%
REQUEST_DELTA_WIN_TO_MIXED = 0.25  # median(ΔR) > 25% downgrades WIN→MIXED
MIN_VALID_PAIRS_DECIDE = 5  # 5 = PILOT WIN/KILL
SCOUT_VALID_PAIRS = 3  # 3 = SCOUT/TRIAGE only (never WIN/KILL)
PROMOTION_VALID_PAIRS = 7  # 7+ = PROMOTION
M3_INCIDENCE_EPISODE_DELTA = 2  # ≥2 episode incidence delta ⇒ INCONCLUSIVE
M3_INCIDENCE_PP_DELTA = 0.25  # ≥25pp incidence delta ⇒ INCONCLUSIVE

# Breadth cap (rev4 §4 r2#4).
BREADTH_MAX_FILES = 2
BREADTH_MAX_WALL_S = 3600.0

# rev4-r6 M3 watchdog + bounded same-seed retry. The watchdog quiescence
# window (K) is intentionally ≪ the faithful wall+grace (wall_s+1800) so a
# true post-completion wedge is converted to orchestrator_stall WELL before
# the wall; a genuinely progressing long run never trips it (every CPU /
# log / recorder advance resets the window). All three are env-overridable
# for tests / tuning but default to the predeclared policy values.
M3_WATCHDOG_QUIESCENCE_S = float(os.environ.get("SKILL_M3_WATCHDOG_S", "600"))
M3_WATCHDOG_POLL_S = float(os.environ.get("SKILL_M3_WATCHDOG_POLL_S", "15"))
M3_SAME_SEED_RETRY_N = int(os.environ.get("SKILL_M3_RETRY_N", "3"))


# --------------------------------------------------------------------------- #
# Contract dataclasses + IdeaSpec schema (step 1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmSpec:
    """One arm of a same-seed controlled comparison.

    role is ``ctrl`` or ``treat`` (the paired-delta classifier subtracts
    ctrl from treat). env_overlay is merged ON TOP of the harness env recipe
    (recipe stays VERBATIM; the overlay only sets per-arm mechanism toggles
    such as ``A3_EXT`` / ``S0_VG_ENABLED``).
    """

    name: str
    role: str  # "ctrl" | "treat"
    env_overlay: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ("ctrl", "treat"):
            raise ValueError(f"ArmSpec.role must be ctrl|treat, got {self.role!r}")
        if not self.name:
            raise ValueError("ArmSpec.name must be non-empty")


@dataclass(frozen=True)
class IdeaSpec:
    """A single portfolio idea (one row of the rev4 §2 table).

    seeds is an ORDERED reserve of ≥10 paired seeds, consumed until 5 valid
    pairs are obtained (rev4 §3 seed reserve & power).
    """

    idea_id: str
    harness: str  # "faithful" | "s0"
    arms: List[ArmSpec]
    seeds: List[int]
    game_id: str = "ft09"
    model: str = "gpt-5.5"
    max_actions: int = 400
    wall_s: int = 12600
    s0_max_actions: int = 60

    def __post_init__(self) -> None:
        if self.harness not in ("faithful", "s0"):
            raise ValueError(f"harness must be faithful|s0, got {self.harness!r}")
        roles = {a.role for a in self.arms}
        if "ctrl" not in roles or "treat" not in roles:
            raise ValueError("IdeaSpec.arms must contain at least one ctrl and one treat")
        if len(self.seeds) < 10:
            raise ValueError(
                f"IdeaSpec.seeds must be an ordered ≥10-seed reserve, got {len(self.seeds)}"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("IdeaSpec.seeds must be unique")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IdeaSpec":
        arms = [
            ArmSpec(
                name=a["name"],
                role=a["role"],
                env_overlay=dict(a.get("env_overlay", {})),
            )
            for a in d["arms"]
        ]
        return IdeaSpec(
            idea_id=d["idea_id"],
            harness=d["harness"],
            arms=arms,
            seeds=list(d["seeds"]),
            game_id=d.get("game_id", "ft09"),
            model=d.get("model", "gpt-5.5"),
            max_actions=int(d.get("max_actions", 400)),
            wall_s=int(d.get("wall_s", 12600)),
            s0_max_actions=int(d.get("s0_max_actions", 60)),
        )

    def ctrl_arm(self) -> ArmSpec:
        return next(a for a in self.arms if a.role == "ctrl")

    def treat_arm(self) -> ArmSpec:
        return next(a for a in self.arms if a.role == "treat")


@dataclass
class EpisodeResult:
    """One arm-run of one seed (a single subprocess invocation).

    This is the in-memory shape; it is projected into the rev4 §4 quartet
    (manifest/metrics/faithfulness jsonl) and the denormalized
    episodes.jsonl by :func:`emit_contract_files`.
    """

    idea_id: str
    seed: int
    arm: str
    order_index: int
    game_id: str
    model: str
    code_sha: str
    store_sha: str
    ports: Dict[str, int]
    start_time: str
    end_time: str
    # metrics
    levels_completed: Optional[int]
    env_actions: Optional[int]
    proxy_requests: Optional[int]
    wall_seconds: Optional[float]
    terminal_status: str
    # faithfulness
    m3_flag: bool
    m3_reason: str
    frozen_config_sha: str
    prompt_sha: str
    skill_store_sha: str
    # internal (not part of the contract files; carried for the classifier)
    sha_ok: bool = True
    faithfulness_ok: bool = True
    raw_summary_path: str = ""


# --------------------------------------------------------------------------- #
# Step 2: Python preflight (lift repin_supervisor.sh:224-281 — G1/G2/G3 split)
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_porcelain_count(cwd: str, paths: Sequence[str]) -> Optional[int]:
    """``git -C cwd status --porcelain -- <paths>`` line count.

    Returns ``None`` if the directory is not a git work-tree (caller treats
    that as a HARD abort for G2 — we can't verify the pin).
    """
    try:
        rev = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if rev.returncode != 0:
            return None
        out = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain", "--", *paths],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0:
            return None
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        return len(lines)
    except (OSError, subprocess.SubprocessError):
        return None


def preflight() -> Dict[str, Any]:
    """Return ``{"result": pass|sha_mismatch|abort, ...}``.

    Lifts the G1/G2/G3 + Py3.12 guards from
    ``scripts/repin_supervisor.sh:224-281`` VERBATIM in structure. Does NOT
    use the broken out-of-repo ``git diff -- … | wc -l`` false-pass form.
    """
    rec: Dict[str, Any] = {"checks": {}}

    # ---- Py3.12 runner env precondition -----------------------------------
    py = PY312_PYTHON
    if not (os.path.isfile(py) and os.access(py, os.X_OK)):
        rec["result"] = "abort"
        rec["reason"] = f"py3.12 runner python not found/executable at {py}"
        return rec
    try:
        ver = subprocess.run(
            [py, "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip() or subprocess.run(
            [py, "--version"], capture_output=True, text=True, timeout=30
        ).stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        rec["result"] = "abort"
        rec["reason"] = f"py3.12 version probe failed: {exc!r}"
        return rec
    if not ver.startswith("Python 3.12."):
        rec["result"] = "abort"
        rec["reason"] = f"runner env python is {ver!r} -- expected Python 3.12.x"
        return rec
    rec["checks"]["py312"] = ver

    # ---- G1: in-repo frozen-scope working-tree diff (RECORDED only) -------
    # Fork template; NOT consumed by clean-upstream. Record-only, not a gate.
    g1 = _git_porcelain_count(
        str(REPO_ROOT),
        ["agents/templates/agentica/", "agents/templates/agentica_lite/"],
    )
    rec["checks"]["G1_inrepo_frozen_porcelain"] = g1  # informational only

    # ---- G2: /tmp/upstream_arcagi3 frozen scope CLEAN (HARD ABORT) -------
    g2 = _git_porcelain_count(
        UPSTREAM_ROOT,
        ["agents/templates/agentica/", "agents/templates/agentica_lite/"],
    )
    if g2 is None:
        rec["result"] = "abort"
        rec["reason"] = f"{UPSTREAM_ROOT} is not a git work-tree -- cannot verify pin"
        return rec
    if g2 != 0:
        rec["result"] = "abort"
        rec["reason"] = (
            f"{UPSTREAM_ROOT} frozen-scope working tree DIRTY "
            f"(status --porcelain lines={g2})"
        )
        return rec
    rec["checks"]["G2_upstream_frozen_porcelain"] = 0

    # ---- G3: live sha256 of consumed faithful files == pinned ------------
    runner_sha = _sha256_file(REPO_ROOT / FAITHFUL_RUNNER)
    proxy_sha = _sha256_file(REPO_ROOT / FAITHFUL_PROXY)
    rec["checks"]["G3_runner_sha"] = runner_sha
    rec["checks"]["G3_proxy_sha"] = proxy_sha
    g3_fail = []
    if runner_sha != PINNED_RUNNER_SHA:
        g3_fail.append(f"runner_sha({runner_sha})")
    if proxy_sha != PINNED_PROXY_SHA:
        g3_fail.append(f"proxy_sha({proxy_sha})")
    if g3_fail:
        rec["result"] = "sha_mismatch"
        rec["reason"] = "G3 frozen-sha mismatch -->" + " ".join(g3_fail)
        return rec

    rec["result"] = "pass"
    return rec


# --------------------------------------------------------------------------- #
# Step 3: flock port / lock manager
# --------------------------------------------------------------------------- #


class PortConflict(RuntimeError):
    """Raised when a port tuple cannot be locked / is bound."""


def _port_bound(port: int) -> bool:
    """True if ``ss -tan`` shows ``port`` bound on the LOCAL address.

    Parses the Local-Address:Port column by token (NOT a fragile ``:NNNN``
    substring — that would false-match ``:28130`` for port ``2813``). The
    ``ss -tan`` row layout is::

        State Recv-Q Send-Q  Local-Address:Port  Peer-Address:Port  ...

    so we take column index 3, strip an IPv6 ``[...]`` wrapper, split off
    the port after the LAST ``:`` and compare it as an integer.
    """
    try:
        out = subprocess.run(
            ["ss", "-tan"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # Fail closed: if we cannot introspect, treat as bound (refuse).
        return True
    lines = out.splitlines()
    for ln in lines[1:]:  # skip the header row
        cols = ln.split()
        if len(cols) < 4:
            continue
        local = cols[3]
        # IPv6 form is [::1]:PORT — drop the bracketed host before splitting.
        if local.startswith("["):
            rb = local.rfind("]")
            local = local[rb + 1 :] if rb != -1 else local
        if ":" not in local:
            continue
        port_tok = local.rsplit(":", 1)[1]
        try:
            if int(port_tok) == int(port):
                return True
        except ValueError:
            continue
    return False


class LockHandle:
    """A held flock + (for faithful) a reserved port tuple.

    Use as a context manager. ``release`` is idempotent and is the ONLY
    teardown path (no broad process-name kill anywhere).
    """

    def __init__(self, lock_path: Path, ports: Dict[str, int]) -> None:
        self.lock_path = lock_path
        self.ports = ports
        self._fd: Optional[int] = None

    def acquire(self) -> "LockHandle":
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise PortConflict(
                    f"flock not acquirable: {self.lock_path} (held by another run)"
                ) from exc
            raise
        # flock held — now refuse if any of the ports is already bound.
        for label, port in self.ports.items():
            if _port_bound(port):
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                raise PortConflict(f"port {port} ({label}) already bound (ss -tan)")
        os.write(fd, f"pid={os.getpid()} ts={_utcnow()}\n".encode())
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def __enter__(self) -> "LockHandle":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def faithful_ports_for_slot(slot: int) -> Dict[str, int]:
    """Disjoint locked tuple from the faithful pool (base 9500/2800, stride 4).

    NEVER returns 9400-9426 / 2700-2726 (in-flight pilots) or 9091/2345 (S0).
    """
    proxy = FAITHFUL_PROXY_PORT_BASE + slot * FAITHFUL_PORT_STRIDE
    server = FAITHFUL_SERVER_PORT_BASE + slot * FAITHFUL_PORT_STRIDE
    # Hard guard against ever colliding with the frozen-forbidden ranges.
    if 9400 <= proxy <= 9426 or 2700 <= server <= 2726:
        raise PortConflict(f"slot {slot} maps into the in-flight ad-hoc range")
    if proxy in (S0_PROXY_PORT, S0_SERVER_PORT) or server in (
        S0_PROXY_PORT,
        S0_SERVER_PORT,
    ):
        raise PortConflict(f"slot {slot} maps onto the S0 slot")
    return {"proxy": proxy, "server": server}


def faithful_lock(slot: int) -> LockHandle:
    ports = faithful_ports_for_slot(slot)
    return LockHandle(LOCK_DIR / f"faithful_slot_{slot}.lock", ports)


def s0_lock() -> LockHandle:
    """The ONE global S0 slot lock — serializes ALL S0 arms over {9091,2345}."""
    return LockHandle(
        LOCK_DIR / "s0_slot.lock",
        {"proxy": S0_PROXY_PORT, "server": S0_SERVER_PORT},
    )


# --------------------------------------------------------------------------- #
# Step 6: Terminal classifier (HIGHEST RISK — code-reviewer focus)
# --------------------------------------------------------------------------- #
#
# code-reviewer focus note:
#   The soft_deadline 3-way split is the densest part of the taxonomy. It maps
#   the faithful runner's single ``soft_deadline`` termination_reason onto
#   THREE distinct rev4 statuses by inspecting the per-action calibration
#   recorder progression + the D5 watchdog stderr dump:
#
#     max_wall_budget    progressed, then hit the wall (last recorded level
#                        < max AND recorder kept advancing — a genuine
#                        wall-time truncation; env-action count is censored ⇒
#                        NEVER a valid pair).
#     m3_nonfaithful     progressed, reached the max level, THEN wedged with
#                        no further advance for the silence window before the
#                        deadline AND the D5 watchdog stderr dump is present
#                        (post-completion orchestrator-quiescence wedge — the
#                        UNFIXED M3 bug; flagged non-faithful).
#     orchestrator_stall never progressed: recorder plateaued early / zero
#                        recorder growth across the whole window (the agent
#                        never did positive work — NOT a censored speed datum,
#                        also NEVER valid).
#
#   Precedence (r3#2) is applied AFTER the per-signal decision so a traceback
#   that coincides with a soft_deadline still classifies as harness_crash.


@dataclass
class FaithfulSignals:
    """Inputs the terminal classifier needs for a faithful arm-run."""

    summary: Optional[Dict[str, Any]]  # parsed <tag>.summary.json (or None)
    per_action_records: List[Dict[str, Any]]
    stderr_has_traceback: bool
    stderr_has_d5_watchdog_dump: bool
    subprocess_rc: int
    sha_ok: bool
    port_conflict: bool
    silence_window_s: float = 1200.0  # D5 default 20 min (A3_FAULT_SILENCE_MIN)


@dataclass
class S0Signals:
    """Inputs the terminal classifier needs for an S0 arm-run."""

    report: Optional[Dict[str, Any]]  # parsed S0 report json (or None)
    subprocess_rc: int
    stderr_has_traceback: bool
    sha_ok: bool
    port_conflict: bool


def _resolve_precedence(candidates: Sequence[str]) -> str:
    """Pick the highest-precedence terminal status among ``candidates``."""
    best = TS_NORMAL
    best_rank = _PRECEDENCE_RANK[TS_NORMAL]
    for c in candidates:
        if c not in _PRECEDENCE_RANK:
            raise ValueError(f"unknown terminal status {c!r}")
        if _PRECEDENCE_RANK[c] < best_rank:
            best, best_rank = c, _PRECEDENCE_RANK[c]
    return best


def _faithful_soft_deadline_split(sig: FaithfulSignals) -> str:
    """The rev4 soft_deadline 3-way split (max_wall / m3 / orchestrator_stall).

    Decided purely from the calibration recorder progression + D5 dump.
    """
    recs = sig.per_action_records
    if not recs:
        # No recorded env action whatsoever in the whole window: the agent
        # never did positive work — orchestrator never progressed.
        return TS_ORCHESTRATOR_STALL

    def _lvl(r: Dict[str, Any]) -> int:
        # level_after_action is authoritative; fall back to levels_completed.
        v = r.get("level_after_action")
        if v is None:
            v = r.get("levels_completed", 0)
        return int(v or 0)

    levels = [_lvl(r) for r in recs]
    last_level = levels[-1]
    summary = sig.summary or {}
    result = summary.get("result", {}) if isinstance(summary, dict) else {}
    # The run's own max-level claim (capability_claim_validity / result).
    # Explicit None-checks (NOT an or-chain): a PRESENT levels_completed==0
    # must be honored as 0 (not fall through to last_level as a falsy zero),
    # and an inflated levels_completed must NOT be silently demoted — so a
    # true post-completion M3 wedge is not hidden as max_wall_budget.
    lc = result.get("levels_completed")
    cap = summary.get("capability_claim_validity", {}).get("max_level_reached")
    claimed_max = int(lc if lc is not None else (cap if cap is not None else last_level))

    # "never progressed" = recorder plateaued at the starting level for the
    # whole window (zero recorder level growth) → orchestrator_stall.
    if last_level <= 0 and max(levels) <= 0:
        return TS_ORCHESTRATOR_STALL

    reached_max = last_level >= claimed_max and claimed_max > 0

    # post-completion wedge: reached the max level, then NO further advance
    # for the silence window before the deadline, AND the D5 watchdog stderr
    # dump fired (the UNFIXED M3 quiescence wedge) → m3_nonfaithful.
    if reached_max and sig.stderr_has_d5_watchdog_dump:
        return TS_M3_NONFAITHFUL

    # Progressed and hit the wall while still working below the max level
    # (or no M3 dump): genuine wall truncation → max_wall_budget.
    return TS_MAX_WALL_BUDGET


def classify_faithful_terminal(sig: FaithfulSignals) -> str:
    """Map faithful signals → rev4 terminal_status (precedence applied)."""
    candidates: List[str] = []

    if not sig.sha_ok:
        candidates.append(TS_SHA_MISMATCH)
    if sig.port_conflict:
        candidates.append(TS_PORT_CONFLICT)
    if sig.stderr_has_traceback:
        candidates.append(TS_HARNESS_CRASH)

    summary = sig.summary
    if summary is None or not isinstance(summary, dict) or "result" not in summary:
        candidates.append(TS_MISSING_METRICS)
        return _resolve_precedence(candidates or [TS_MISSING_METRICS])

    result = summary.get("result", {})
    for key in ("levels_completed", "recorder_action_count", "request_count"):
        if result.get(key) is None:
            candidates.append(TS_MISSING_METRICS)
            break

    reason = str(result.get("termination_reason", "unknown"))
    fuse = summary.get("action_budget_fuse", {})
    fuse_fired = bool(result.get("fuse_fired")) or bool(fuse.get("fired"))

    if reason == "swarm_returned":
        candidates.append(TS_NORMAL)
    elif reason == "action_budget_fuse" or fuse_fired:
        candidates.append(TS_MAX_EPISODE_BUDGET)
    elif reason == "soft_deadline" or result.get("soft_deadline_hit"):
        candidates.append(_faithful_soft_deadline_split(sig))
    else:
        # Unknown / SIGTERM emergency summary with no clean reason.
        if sig.subprocess_rc != 0:
            candidates.append(TS_HARNESS_CRASH)
        else:
            candidates.append(TS_NORMAL)

    return _resolve_precedence(candidates)


def classify_s0_terminal(sig: S0Signals) -> str:
    """Map S0 signals → rev4 terminal_status (precedence applied)."""
    candidates: List[str] = []

    if not sig.sha_ok:
        candidates.append(TS_SHA_MISMATCH)
    if sig.port_conflict:
        candidates.append(TS_PORT_CONFLICT)
    if sig.stderr_has_traceback:
        candidates.append(TS_HARNESS_CRASH)

    report = sig.report
    if report is None or not isinstance(report, dict):
        # rc==124 from `timeout` with no report ⇒ wall budget; else missing.
        if sig.subprocess_rc == 124:
            candidates.append(TS_MAX_WALL_BUDGET)
        else:
            candidates.append(TS_MISSING_METRICS)
        return _resolve_precedence(candidates or [TS_MISSING_METRICS])

    for key in ("levels_completed", "action_count", "request_count"):
        if report.get(key) is None:
            candidates.append(TS_MISSING_METRICS)
            break

    if sig.subprocess_rc == 124:
        # `timeout` killed it — wall budget (env-action count censored).
        candidates.append(TS_MAX_WALL_BUDGET)
    else:
        final_state = str(report.get("final_state", ""))
        action_count = int(report.get("action_count", 0) or 0)
        max_actions = int(report.get("max_actions", 0) or 0)
        if final_state == "WIN":
            candidates.append(TS_NORMAL)
        elif max_actions and action_count >= max_actions:
            candidates.append(TS_MAX_EPISODE_BUDGET)
        elif sig.subprocess_rc != 0:
            candidates.append(TS_HARNESS_CRASH)
        else:
            # Completed the smoke without WIN and without exhausting the
            # action budget — a normal terminal episode (L6 not required
            # for VALIDITY per rev4 §3).
            candidates.append(TS_NORMAL)

    return _resolve_precedence(candidates)


# --------------------------------------------------------------------------- #
# Step 7: Paired-delta classifier (rev4 §3 verbatim — pure reducer)
# --------------------------------------------------------------------------- #


def _is_valid_episode(ep: Dict[str, Any]) -> bool:
    """rev4 §3 r5 Option-A per-arm validity precondition."""
    if ep.get("terminal_status") not in VALID_TERMINAL_STATUSES:
        return False
    for k in ("levels_completed", "env_actions", "proxy_requests"):
        v = ep.get(k)
        if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
    if not ep.get("sha_ok", True):
        return False
    if not ep.get("faithfulness_ok", True):
        return False
    return True


def classify_paired_delta(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic paired-delta classifier (rev4 §3, DECISION_RULE_VERSION).

    Pure reducer over ``episodes.jsonl`` rows. Pairs by ``seed`` within a
    single-idea episodes.jsonl (``emit_contract_files`` is per-idea, so
    episodes.jsonl carries exactly one ``idea_id`` — seed alone is the
    pairing key; the grouping logic is unchanged).
    ``Delta_L = treat.levels − ctrl.levels``;
    ``Delta_A = (ctrl.env_actions − treat.env_actions)/ctrl.env_actions``;
    ``Delta_R = (treat.proxy_requests − ctrl.proxy_requests)/ctrl.proxy_requests``.

    Decision order (rev4 §3): INCONCLUSIVE (M3/validity) is evaluated BEFORE
    WIN/KILL. Replaces the McNemar/Wilcoxon of vg_paired_eval.py:_summarize
    with this deterministic rule.
    """
    out: Dict[str, Any] = {
        "decision_rule_version": DECISION_RULE_VERSION,
        "result": None,
        "reason": None,
        "valid_pair_count": 0,
        "pairs": [],
        "deltas": {"L": [], "A": [], "R": []},
        "m3": {},
    }

    # --- group episodes into (seed -> {role: ep}) -------------------------
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for ep in episodes:
        seed = ep["seed"]
        by_seed.setdefault(seed, {})[ep["arm_role"]] = ep

    # --- M3 incidence per arm (over ALL paired-complete seeds) ------------
    m3_counts = {"ctrl": 0, "treat": 0}
    m3_totals = {"ctrl": 0, "treat": 0}
    valid_pairs: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    # Stable seed order = the order seeds first appear in episodes.jsonl
    # (the runner writes them in the pre-registered reserve order).
    seed_order: List[int] = []
    seen = set()
    for ep in episodes:
        if ep["seed"] not in seen:
            seen.add(ep["seed"])
            seed_order.append(ep["seed"])

    for seed in seed_order:
        arms = by_seed.get(seed, {})
        ctrl = arms.get("ctrl")
        treat = arms.get("treat")
        if ctrl is None or treat is None:
            continue
        for role, ep in (("ctrl", ctrl), ("treat", treat)):
            m3_totals[role] += 1
            if ep.get("m3_flag"):
                m3_counts[role] += 1
        if _is_valid_episode(ctrl) and _is_valid_episode(treat):
            valid_pairs.append((seed, ctrl, treat))

    out["valid_pair_count"] = len(valid_pairs)
    out["pairs"] = [
        {"seed": s, "ctrl_terminal": c["terminal_status"], "treat_terminal": t["terminal_status"]}
        for s, c, t in valid_pairs
    ]

    # --- M3 differential ⇒ INCONCLUSIVE (evaluated BEFORE WIN/KILL) -------
    m3_episode_delta = abs(m3_counts["treat"] - m3_counts["ctrl"])
    ctrl_rate = (m3_counts["ctrl"] / m3_totals["ctrl"]) if m3_totals["ctrl"] else 0.0
    treat_rate = (m3_counts["treat"] / m3_totals["treat"]) if m3_totals["treat"] else 0.0
    m3_pp_delta = abs(treat_rate - ctrl_rate)
    out["m3"] = {
        "ctrl_count": m3_counts["ctrl"],
        "treat_count": m3_counts["treat"],
        "ctrl_rate": ctrl_rate,
        "treat_rate": treat_rate,
        "episode_delta": m3_episode_delta,
        "pp_delta": m3_pp_delta,
    }
    if (
        m3_episode_delta >= M3_INCIDENCE_EPISODE_DELTA
        or m3_pp_delta >= M3_INCIDENCE_PP_DELTA
    ):
        out["result"] = "INCONCLUSIVE"
        out["reason"] = (
            f"M3 incidence differential (episodes Δ={m3_episode_delta}, "
            f"pp Δ={m3_pp_delta:.3f})"
        )
        return out

    n = len(valid_pairs)

    # --- power gate (rev4 §3): <5 valid ⇒ PARK/INFRA_BLOCK ---------------
    if n < MIN_VALID_PAIRS_DECIDE:
        # 3 ≤ n < 5 ⇒ SCOUT is the deliberate reading of rev4 §3
        # ("3=SCOUT, 5=decide"): only n<3 is INFRA_BLOCK, the rest SCOUT.
        out["result"] = "INFRA_BLOCK" if n < SCOUT_VALID_PAIRS else "SCOUT"
        out["reason"] = (
            f"only {n} valid non-M3 pairs (<{MIN_VALID_PAIRS_DECIDE}); "
            "SCOUT/TRIAGE only — never WIN/KILL"
        )
        return out

    # --- deltas over valid pairs (rev4 §3) -------------------------------
    dL: List[float] = []
    dA: List[float] = []
    dR: List[float] = []
    for _seed, ctrl, treat in valid_pairs:
        dL.append(float(treat["levels_completed"]) - float(ctrl["levels_completed"]))
        ca = float(ctrl["env_actions"])
        if ca != 0:
            dA.append((ca - float(treat["env_actions"])) / ca)
        cr = float(ctrl["proxy_requests"])
        if cr != 0:
            dR.append((float(treat["proxy_requests"]) - cr) / cr)
    out["deltas"] = {"L": dL, "A": dA, "R": dR}

    sum_L = sum(dL)
    neg_L = sum(1 for x in dL if x < 0)
    med_A = statistics.median(dA) if dA else 0.0
    med_R = statistics.median(dR) if dR else 0.0
    out["stats"] = {
        "sum_L": sum_L,
        "neg_L_count": neg_L,
        "median_A": med_A,
        "median_R": med_R,
    }

    # --- WIN / MIXED / KILL (rev4 §3 verbatim) ---------------------------
    action_ok = med_A >= ACTION_DELTA_THRESHOLD
    if sum_L < 0 or (sum_L == 0 and not action_ok):
        result = "KILL"
        reason = (
            f"sum(ΔL)={sum_L} <0"
            if sum_L < 0
            else f"sum(ΔL)==0 and median(ΔA)={med_A:.3f}<{ACTION_DELTA_THRESHOLD}"
        )
    elif sum_L >= 0 and neg_L <= 1 and action_ok:
        result = "WIN"
        reason = (
            f"sum(ΔL)={sum_L}≥0, neg_L={neg_L}≤1, median(ΔA)={med_A:.3f}"
            f"≥{ACTION_DELTA_THRESHOLD}"
        )
        if med_R > REQUEST_DELTA_WIN_TO_MIXED:
            result = "MIXED"
            reason += (
                f"; downgraded WIN→MIXED (median(ΔR)={med_R:.3f}"
                f">{REQUEST_DELTA_WIN_TO_MIXED})"
            )
    else:
        # sum(ΔL)>0 but action threshold fails; OR median(ΔA)≥10% but ≥2 ΔL<0.
        result = "MIXED"
        reason = (
            f"mixed: sum(ΔL)={sum_L}, neg_L={neg_L}, median(ΔA)={med_A:.3f}"
        )

    out["result"] = result
    out["reason"] = reason
    if n >= PROMOTION_VALID_PAIRS:
        out["promotion_eligible"] = True
    return out


# --------------------------------------------------------------------------- #
# rev4 §4 contract emission: quartet + denormalized episodes.jsonl
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def emit_contract_files(
    out_dir: Path,
    episodes: List[EpisodeResult],
    arm_role_by_name: Dict[str, str],
    retry_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit the rev4 §4 quartet AND the denormalized episodes.jsonl.

    ``retry_stats`` (rev4-r6, optional — default ``None`` so the existing
    3-arg callers / tests stay byte-identical) carries the SEPARATE per-arm
    ``m3_rate`` + top-level ``m3_unmeasurable`` / ``m3_unmeasurable_reason``
    + ``retry_policy``. None ⇒ those keys are simply absent (a pre-r6
    summary.json is unchanged); the M3 rate is NEVER folded into
    ``valid_pair_count``.

    Returns the classifier result dict (also written to summary.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    faith_rows: List[Dict[str, Any]] = []
    join_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        key = {
            "idea_id": ep.idea_id,
            "seed": ep.seed,
            "arm": ep.arm,
            "order_index": ep.order_index,
        }
        manifest_rows.append(
            {
                **key,
                "game_id": ep.game_id,
                "model": ep.model,
                "store_sha": ep.store_sha,
                "code_sha": ep.code_sha,
                "ports": ep.ports,
                "start_time": ep.start_time,
                "end_time": ep.end_time,
            }
        )
        metrics_rows.append(
            {
                **key,
                "levels_completed": ep.levels_completed,
                "env_actions": ep.env_actions,
                "proxy_requests": ep.proxy_requests,
                "wall_seconds": ep.wall_seconds,
                "terminal_status": ep.terminal_status,
            }
        )
        faith_rows.append(
            {
                **key,
                "m3_flag": ep.m3_flag,
                "m3_reason": ep.m3_reason,
                "frozen_config_sha": ep.frozen_config_sha,
                "prompt_sha": ep.prompt_sha,
                "skill_store_sha": ep.skill_store_sha,
            }
        )
        # Denormalized inner-join row (the classifier input). Names follow
        # the rev4 §3 classifier vocabulary; this row is a pure projection.
        join_rows.append(
            {
                **key,
                "arm_role": arm_role_by_name.get(ep.arm, ""),
                "game_id": ep.game_id,
                "model": ep.model,
                "levels_completed": ep.levels_completed,
                "env_actions": ep.env_actions,
                "proxy_requests": ep.proxy_requests,
                "wall_seconds": ep.wall_seconds,
                "terminal_status": ep.terminal_status,
                "m3_flag": ep.m3_flag,
                "m3_reason": ep.m3_reason,
                "sha_ok": ep.sha_ok,
                "faithfulness_ok": ep.faithfulness_ok,
                "ports": ep.ports,
            }
        )

    _write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    _write_jsonl(out_dir / "metrics.jsonl", metrics_rows)
    _write_jsonl(out_dir / "faithfulness.jsonl", faith_rows)
    _write_jsonl(out_dir / "episodes.jsonl", join_rows)

    classifier = classify_paired_delta(join_rows)

    # Per-arm terminal counts (the §4 summary.json shape).
    per_arm_terminal: Dict[str, Dict[str, int]] = {}
    for ep in episodes:
        d = per_arm_terminal.setdefault(ep.arm, {})
        d[ep.terminal_status] = d.get(ep.terminal_status, 0) + 1

    summary = {
        "decision_rule_version": DECISION_RULE_VERSION,
        "valid_pair_count": classifier["valid_pair_count"],
        "per_arm_terminal_counts": per_arm_terminal,
        "m3_incidence_delta": classifier.get("m3", {}),
        "classifier_result": classifier["result"],
        "classifier_reason": classifier["reason"],
        "classifier": classifier,
    }
    if retry_stats is not None:
        # rev4-r6 SEPARATE M3-rate block — NEVER folded into
        # valid_pair_count. m3_unmeasurable is definitive ONLY when True.
        summary["m3_rate"] = retry_stats.get("m3_rate", {})
        summary["m3_unmeasurable"] = bool(retry_stats.get("m3_unmeasurable", False))
        summary["m3_unmeasurable_reason"] = str(
            retry_stats.get("m3_unmeasurable_reason", "")
        )
        summary["retry_policy"] = retry_stats.get(
            "retry_policy",
            {
                "n": M3_SAME_SEED_RETRY_N,
                "K_s": M3_WATCHDOG_QUIESCENCE_S,
                "applied_to": "both_arms_identically",
            },
        )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


# --------------------------------------------------------------------------- #
# Steps 4 & 5: arm drivers (faithful / S0)
# --------------------------------------------------------------------------- #


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stderr_signals(text: str) -> Tuple[bool, bool]:
    """(has_python_traceback, has_d5_watchdog_dump) from a runner stderr blob."""
    has_tb = "Traceback (most recent call last)" in text
    has_d5 = "D5 watchdog: no proxy activity" in text
    return has_tb, has_d5


def _trapi_env() -> Dict[str, str]:
    """The TRAPI_* block — VERBATIM from /tmp/skill_pilot_paired.sh:34-39."""
    return {
        "TRAPI_API_VERSION": "2025-04-01-preview",
        "TRAPI_BASE_URL": "https://agl-dev.cognitiveservices.azure.com/openai",
        "TRAPI_PROXY_AAD_SCOPE": "https://cognitiveservices.azure.com/.default",
        "TRAPI_PROXY_DEFAULT_MODEL": "gpt-5.5",
        "TRAPI_PROXY_MAX_RETRIES": "10",
        "TRAPI_PROXY_MAX_TRANSPARENT_RETRIES": "10",
        "TRAPI_PROXY_STRIP_DATE_SUFFIX": "1",
        "TRAPI_PROXY_TRANSPARENT_RETRY_SLEEP_S": "180",
    }


def run_faithful_arm(
    idea: IdeaSpec,
    arm: ArmSpec,
    seed: int,
    order_index: int,
    slot: int,
    out_dir: Path,
) -> EpisodeResult:
    """Subprocess the pinned faithful runner with the VERBATIM env recipe.

    Recipe lifted from /tmp/skill_pilot_paired.sh:31-40. The arm's
    ``env_overlay`` (e.g. ``A3_EXT``) is merged on top — the recipe itself is
    never edited; the faithful runner / proxy are subprocess-invoked only.
    """
    ts = _utcnow()
    tag = f"{idea.idea_id}_s{seed}_{arm.name}_{ts}"
    plog = out_dir / f"{tag}.proxy.jsonl"
    rout = out_dir / f"{tag}.runner.out"
    summary_path = REPO_ROOT / "reports" / "a3_calibration" / f"{tag}.summary.json"

    with faithful_lock(slot) as lock:
        ports = lock.ports
        env = dict(os.environ)
        # VERBATIM recipe (skill_pilot_paired.sh:31-39); A3_EXT defaults to
        # none and is the ONLY mechanism toggle the overlay may flip.
        env.update(
            {
                "A3_TAG": tag,
                "S0_MAX_ACTIONS": str(idea.max_actions),
                "A3_WALL_S": str(idea.wall_s),
                "S0_MODEL_PRESET": idea.model,
                "S0_SEED": str(seed),
                "S0_SKILL_ENABLED": "0",
                "S0_VG_ENABLED": "0",
                "A3_EXT": "none",
                "A3_PROXY_PORT": str(ports["proxy"]),
                "A3_SERVER_PORT": str(ports["server"]),
                "A3_PROXY_LOG_PATH": str(plog),
            }
        )
        env.update(_trapi_env())
        env.update(arm.env_overlay)  # per-arm mechanism overlay (e.g. A3_EXT)

        # Faithful recorder (the positive-work artifact) + per-episode proxy
        # jsonl. K (M3_WATCHDOG_QUIESCENCE_S, default 600s) ≪ the faithful
        # wall+grace (wall_s+1800) so a true post-completion wedge is
        # converted to orchestrator_stall WELL before the wall; a genuinely
        # progressing long run never trips (every CPU / log / recorder
        # advance resets the window). _faithful_soft_deadline_split is left
        # UNTOUCHED (a disjoint, summary-driven trigger).
        recorder_jsonl = REPO_ROOT / "reports" / "a3_calibration" / f"{tag}.jsonl"
        start = _utcnow()
        rc = -1
        proc: Optional[subprocess.Popen] = None
        wd_thread: Optional[threading.Thread] = None
        wd_stop = threading.Event()
        wd_stall: List[bool] = [False]
        try:
            # subprocess.Popen with the SAME stdout=fh + stderr=STDOUT
            # writes the child's stream byte-for-byte to rout exactly as the
            # prior subprocess.run did; proc.wait(timeout=…)→TimeoutExpired
            # ⇒ rc=124 preserves the prior healthy-path semantics verbatim.
            with open(rout, "w", encoding="utf-8") as fh:
                proc = subprocess.Popen(
                    [PY312_PYTHON, FAITHFUL_RUNNER],
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
                # Byte-offset captured RIGHT AFTER launch (per-episode plog).
                plog_after = _path_size(plog)
                wd_thread = threading.Thread(
                    target=_episode_watchdog,
                    args=(
                        proc.pid,
                        plog,            # per-episode proxy jsonl
                        rout,            # runner stdout/stderr log (size-progress)
                        recorder_jsonl,  # calibration recorder (rows + level)
                        plog_after,
                        wd_stop,
                        wd_stall,
                    ),
                    daemon=True,
                )
                wd_thread.start()
                try:
                    rc = proc.wait(timeout=idea.wall_s + 1800)
                except subprocess.TimeoutExpired:
                    rc = 124
                # Watchdog confirmed a post-stream-complete wedge: kill the
                # EXACT proc-tree (numeric PIDs only — NEVER a broad
                # process-name kill; _proc_tree_pids(proc.pid) covers its
                # proxy + agentica-server descendants).
                if wd_stall[0] and proc.poll() is None:
                    _kill_proc_tree(proc.pid)
                    rc = -1
        finally:
            # Stop + JOIN the watchdog BEFORE leaving the locked region (the
            # _kill_proc_tree above is proc.poll()-guarded ⇒ idempotent, no
            # double-kill / no race).
            wd_stop.set()
            if wd_thread is not None:
                wd_thread.join(timeout=M3_WATCHDOG_POLL_S + 5.0)
            if proc is not None and proc.poll() is None:
                _kill_proc_tree(proc.pid)
        end = _utcnow()
        watchdog_stalled = wd_stall[0]

    # ---- parse summary + recorder jsonl ---------------------------------
    # ``recorder_jsonl`` was bound above (watchdog target); re-use it.
    summary = _load_json(summary_path)
    per_action = _load_jsonl(recorder_jsonl)
    stderr_text = _safe_read(rout)
    has_tb, has_d5 = _stderr_signals(stderr_text)
    cur_runner_sha = _sha256_file(REPO_ROOT / FAITHFUL_RUNNER)
    cur_proxy_sha = _sha256_file(REPO_ROOT / FAITHFUL_PROXY)
    sha_ok = (
        cur_runner_sha == PINNED_RUNNER_SHA and cur_proxy_sha == PINNED_PROXY_SHA
    )
    prov = (summary or {}).get("upstream_provenance", {})
    is_clean = bool(prov.get("is_clean_upstream"))

    sig = FaithfulSignals(
        summary=summary,
        per_action_records=per_action,
        stderr_has_traceback=has_tb,
        stderr_has_d5_watchdog_dump=has_d5,
        subprocess_rc=rc,
        sha_ok=sha_ok,
        port_conflict=False,
    )
    terminal = classify_faithful_terminal(sig)
    if watchdog_stalled:
        # The watchdog confirmed a post-stream-complete quiescence wedge.
        # Compose through the EXISTING precedence resolver so an infra
        # signal (traceback / sha) still outranks it (rev4 §3 r3#2). The
        # disjoint, summary-driven _faithful_soft_deadline_split is left
        # untouched. Healthy path ⇒ watchdog_stalled False ⇒ this branch is
        # never entered ⇒ terminal byte-identical to the pre-r6 result.
        terminal = _resolve_precedence([terminal, TS_ORCHESTRATOR_STALL])
    m3_flag = terminal == TS_M3_NONFAITHFUL
    result = (summary or {}).get("result", {})

    return EpisodeResult(
        idea_id=idea.idea_id,
        seed=seed,
        arm=arm.name,
        order_index=order_index,
        game_id=idea.game_id,
        model=idea.model,
        code_sha=cur_runner_sha or "",
        store_sha=str(prov.get("agent_py_sha256", "")),
        ports=lock.ports,
        start_time=start,
        end_time=end,
        levels_completed=_as_num(result.get("levels_completed")),
        env_actions=_as_num(result.get("recorder_action_count")),
        proxy_requests=_as_num(result.get("request_count")),
        wall_seconds=_as_num(result.get("runtime_seconds")),
        terminal_status=terminal,
        m3_flag=m3_flag,
        m3_reason="d5_watchdog_postcompletion_wedge" if m3_flag else "",
        frozen_config_sha=str(prov.get("agent_py_sha256", "")),
        prompt_sha=str(prov.get("prompts_py_sha256", "")),
        skill_store_sha=str(
            (summary or {}).get("trace2skill", {}).get("prov_tag", "")
        ),
        sha_ok=sha_ok,
        faithfulness_ok=is_clean and sha_ok,
        raw_summary_path=str(summary_path),
    )


def run_s0_arm(
    idea: IdeaSpec,
    arm: ArmSpec,
    seed: int,
    order_index: int,
    out_dir: Path,
) -> EpisodeResult:
    """Subprocess s0_smoke_ft09.py serialized through the ONE global S0 lock.

    S0 env recipe lifted from scripts/paired_v9_safe60.sh:98-113. A
    deterministic per-episode S0_REPORT_PATH eliminates the
    ``ls -t … | head -1`` race. Cleanup = SCOPED fuser on {9091,2345} +
    tracked child PID kill + ports-closed wait + flock release. There is NO
    broad process-name kill (no global uvicorn / agentica-server / s0_smoke
    sweep).
    """
    ts = _utcnow()
    tag = f"{idea.idea_id}_s{seed}_{arm.name}_{ts}"
    rout = out_dir / f"{tag}.runner.out"
    report_path = out_dir / f"{tag}.s0report.json"

    with s0_lock():  # serializes ALL S0 arms over {9091,2345}
        env = dict(os.environ)
        # VERBATIM S0 recipe (paired_v9_safe60.sh:98-113).
        env.update(
            {
                "S0_VG_ENABLED": "0",
                "S0_SKILL_ENABLED": "0",
                "S0_MAX_ACTIONS": str(idea.s0_max_actions),
                "S0_MODEL_PRESET": idea.model,
                "S0_SEED": str(seed),
                "TRAPI_PROXY_AAD_SCOPE": "https://cognitiveservices.azure.com/.default",
                "TRAPI_BASE_URL": "https://agl-dev.cognitiveservices.azure.com/openai",
                "TRAPI_PROXY_STRIP_DATE_SUFFIX": "1",
                "TRAPI_PROXY_DEFAULT_MODEL": "gpt-5.5",
                "TRAPI_API_VERSION": "2025-04-01-preview",
                "TRAPI_PROXY_MAX_RETRIES": "10",
                "TRAPI_PROXY_TRANSPARENT_RETRY_SLEEP_S": "180",
                "TRAPI_PROXY_MAX_TRANSPARENT_RETRIES": "10",
                # deterministic per-episode report path (no ls -t race).
                "S0_REPORT_PATH": str(report_path),
            }
        )
        env.update(arm.env_overlay)  # per-arm mechanism overlay (VG/skill)

        start = _utcnow()
        rc = -1
        child: Optional[subprocess.Popen] = None
        # S0 proxy jsonl: the single shared slot path, flock-serialized by
        # the S0 lock held above — a fixed byte cursor is an EXACT scoping.
        s0_plog = Path("/tmp/s0_smoke_proxy.jsonl")
        wd_thread: Optional[threading.Thread] = None
        wd_stop = threading.Event()
        wd_stall: List[bool] = [False]
        try:
            with open(rout, "w", encoding="utf-8") as fh:
                child = subprocess.Popen(
                    [sys.executable, S0_SMOKE],
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
                # Byte-offset captured RIGHT AFTER launch (flock-serialized
                # shared slot ⇒ exact, race-free scoping).
                plog_after = _path_size(s0_plog)
                wd_thread = threading.Thread(
                    target=_episode_watchdog,
                    args=(
                        child.pid,
                        s0_plog,
                        rout,            # server/stdout log (size-progress)
                        report_path,     # S0 positive-work artifact
                        plog_after,
                        wd_stop,
                        wd_stall,
                    ),
                    daemon=True,
                )
                wd_thread.start()
                try:
                    rc = child.wait(timeout=idea.wall_s + 600)
                except subprocess.TimeoutExpired:
                    rc = 124
                # Watchdog confirmed a post-stream-complete wedge: kill the
                # EXACT child proc-tree (numeric PIDs only — never a broad
                # process-name kill).
                if wd_stall[0] and child.poll() is None:
                    _kill_proc_tree(child.pid)
                    rc = -1
        finally:
            # Stop + JOIN the watchdog BEFORE the idempotent scoped cleanup
            # (child.poll()-guarded ⇒ no double-kill / no race).
            wd_stop.set()
            if wd_thread is not None:
                wd_thread.join(timeout=M3_WATCHDOG_POLL_S + 5.0)
            # ---- SCOPED cleanup (NO broad process-name kill anywhere) ---
            _s0_scoped_cleanup(child)
        end = _utcnow()
        watchdog_stalled = wd_stall[0]

    report = _load_json(report_path)
    stderr_text = _safe_read(rout)
    has_tb, _ = _stderr_signals(stderr_text)
    cur_s0_sha = _sha256_file(REPO_ROOT / S0_SMOKE)

    sig = S0Signals(
        report=report,
        subprocess_rc=rc,
        stderr_has_traceback=has_tb,
        sha_ok=True,  # S0 smoke is not on the frozen-pin list
        port_conflict=False,
    )
    terminal = classify_s0_terminal(sig)
    if watchdog_stalled:
        # The watchdog confirmed a post-stream-complete quiescence wedge.
        # Compose it through the EXISTING precedence resolver so an infra
        # signal (traceback / sha) still outranks it (rev4 §3 r3#2). On the
        # healthy path watchdog_stalled is False ⇒ this branch is never
        # entered ⇒ byte-identical to the pre-r6 result.
        terminal = _resolve_precedence([terminal, TS_ORCHESTRATOR_STALL])
    r = report or {}

    return EpisodeResult(
        idea_id=idea.idea_id,
        seed=seed,
        arm=arm.name,
        order_index=order_index,
        game_id=idea.game_id,
        model=idea.model,
        code_sha=cur_s0_sha or "",
        store_sha="",
        ports={"proxy": S0_PROXY_PORT, "server": S0_SERVER_PORT},
        start_time=start,
        end_time=end,
        levels_completed=_as_num(r.get("levels_completed")),
        env_actions=_as_num(r.get("action_count")),
        proxy_requests=_as_num(r.get("request_count")),
        wall_seconds=_as_num(r.get("runtime_seconds")),
        terminal_status=terminal,
        m3_flag=False,  # M3 is a faithful-only post-completion wedge
        m3_reason="",
        frozen_config_sha=cur_s0_sha or "",
        prompt_sha="",
        skill_store_sha="",
        sha_ok=True,
        faithfulness_ok=True,
        raw_summary_path=str(report_path),
    )


# --------------------------------------------------------------------------- #
# rev4-r6 M3 watchdog primitives (stdlib-only — NO psutil, NO broad kill)
# --------------------------------------------------------------------------- #
#
# code-reviewer focus note:
#   Every primitive here is a PURE OBSERVER over /proc and our own
#   append-only logs. The only side effect anywhere in the watchdog path is
#   ``os.kill(pid, sig)`` on EXACT numeric PIDs that are members of the
#   process tree rooted at the child we ourselves launched (root_pid +
#   transitively-PPID-linked descendants) — never a name pattern, never a
#   broad process-name sweep. The current runner PID is never in that tree
#   (its PPID chain does not pass through the launched child), so the kill
#   cannot self-match. Every /proc read is race-tolerant: a pid that exits
#   mid-scan is silently skipped (treated as 0 work / not present), never
#   raised.


def _path_size(p: Path) -> int:
    """``os.path.getsize`` or 0 (missing / unreadable → 0, never raises)."""
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _proc_tree_pids(root_pid: int) -> List[int]:
    """BFS the /proc PPID forest → [root_pid] + all transitive descendants.

    Reads field 4 (PPID) of every ``/proc/<pid>/stat``. The comm field
    (field 2) can contain spaces / parentheses, so we split on the LAST
    ``')'`` and index the remaining whitespace-tokenized tail (PPID is the
    2nd token after the ``)``: state, ppid, ...). Race-tolerant: a pid that
    vanishes mid-scan is skipped. The returned list is exactly the PIDs we
    are entitled to signal (all rooted at our own launched child).
    """
    children: Dict[int, List[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return [root_pid]
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
                data = fh.read()
        except OSError:
            continue  # pid exited mid-scan — skip (race-tolerant)
        rp = data.rfind(")")
        if rp == -1:
            continue
        tail = data[rp + 1 :].split()
        # tail = [state, ppid, pgrp, ...]
        if len(tail) < 2:
            continue
        try:
            ppid = int(tail[1])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
    out: List[int] = [root_pid]
    seen = {root_pid}
    queue = [root_pid]
    while queue:
        cur = queue.pop(0)
        for ch in children.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
                queue.append(ch)
    return out


def _proc_starttime(pid: int) -> Optional[int]:
    """``/proc/<pid>/stat`` field-22 (``starttime``) or ``None`` if gone.

    W3 (codex r1): PID-reuse guard. Field 22 (1-indexed) is the process
    start time in clock ticks since boot — unique per (pid, incarnation).
    On the comm-aware post-``)`` tail (0-indexed) it is ``tail[19]``:
    [state(0), ppid(1), pgrp(2), session(3), tty_nr(4), tpgid(5), flags(6),
    minflt(7), cminflt(8), majflt(9), cmajflt(10), utime(11), stime(12),
    cutime(13), cstime(14), priority(15), nice(16), num_threads(17),
    itrealvalue(18), starttime(19), ...]. A vanished / unreadable / malformed
    proc returns ``None`` (never raises). Comparing the snapshot starttime to
    a fresh read before EACH ``os.kill`` proves the integer pid still names
    the SAME process incarnation we are entitled to signal; any mismatch /
    None means the pid was recycled (or exited) and MUST NOT be killed.
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
            data = fh.read()
    except OSError:
        return None
    rp = data.rfind(")")
    if rp == -1:
        return None
    tail = data[rp + 1 :].split()
    if len(tail) < 20:
        return None
    try:
        return int(tail[19])
    except ValueError:
        return None


def _proc_tree_cpu_jiffies(pids: Sequence[int]) -> int:
    """Sum utime(field14)+stime(field15) over ``pids``.

    A missing / exited pid contributes 0 (never raises). Same comm-aware
    split as :func:`_proc_tree_pids`: after the last ``')'`` the tokens are
    [state(0), ppid(1), pgrp(2), session(3), tty_nr(4), tpgid(5), flags(6),
    minflt(7), cminflt(8), majflt(9), cmajflt(10), utime(11), stime(12), ...]
    (0-indexed on the post-``)`` tail; utime/stime are tail[11]/tail[12]).
    """
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as fh:
                data = fh.read()
        except OSError:
            continue  # missing pid → 0
        rp = data.rfind(")")
        if rp == -1:
            continue
        tail = data[rp + 1 :].split()
        if len(tail) < 13:
            continue
        try:
            total += int(tail[11]) + int(tail[12])
        except ValueError:
            continue
    return total


def _proxy_new_stream_request(plog: Path, after_byte: int) -> Tuple[int, bool]:
    """Scan the append-only proxy jsonl from ``after_byte``.

    Returns ``(new_eof, saw_new_stream_request)`` where ``new_eof`` is the
    byte offset to record for the next poll and the bool is True iff any
    ``event == "stream_request"`` row appeared past ``after_byte``. The
    proxy jsonl is append-only and (for the shared S0 slot) flock-serialized,
    so a fixed byte cursor is an exact, race-free scoping. Unreadable /
    missing file → ``(after_byte, False)``.
    """
    try:
        size = os.path.getsize(plog)
    except OSError:
        return after_byte, False
    if size <= after_byte:
        return size, False
    saw = False
    try:
        with open(plog, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(after_byte)
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") == "stream_request":
                    saw = True
    except OSError:
        return after_byte, False
    return size, saw


def _proxy_saw_stream_complete(plog: Path, after_byte: int) -> bool:
    """True iff a ``stream_complete`` event exists AT/AFTER ``after_byte``.

    The watchdog quiescence window only STARTS counting after a
    stream_complete has been observed (codex #2: the post-completion
    quiescence signature is valid ONLY post-stream-complete — before it,
    silence is normal in-flight latency, never a wedge).

    W2 (codex r1): the scan MUST be anchored at ``after_byte`` (the episode's
    recorded plog start offset, ``plog_after``). The S0 proxy jsonl
    ``/tmp/s0_smoke_proxy.jsonl`` is a SINGLE shared slot that is append-only
    across every flock-serialized S0 episode, so a PRIOR episode's
    ``stream_complete`` (bytes strictly before this episode's ``after_byte``)
    MUST NOT be mistaken for this episode's completion — doing so would
    prematurely arm the quiescence window for a fresh S0 episode whose own
    LLM request is still in flight and let the watchdog false-kill a
    legitimately alive run (the core faithfulness guarantee). A
    ``stream_complete`` strictly before ``after_byte`` is therefore ignored.
    The faithful arm's proxy jsonl is per-episode (``{tag}.proxy.jsonl``) so
    ``after_byte`` is typically 0, but it is honored on BOTH arms for
    uniformity / correctness. Missing / unreadable file → ``False``.
    """
    try:
        size = os.path.getsize(plog)
    except OSError:
        return False
    if size <= after_byte:
        return False
    try:
        with open(plog, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(after_byte)
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("event") == "stream_complete":
                    return True
    except OSError:
        return False
    return False


def _recorder_progress(recorder_jsonl: Path) -> Tuple[int, int]:
    """``(row_count, max_level)`` of the calibration recorder jsonl.

    ``max_level`` reads ``level_after_action`` (authoritative) falling back
    to ``levels_completed``. Both monotonic non-decreasing for a progressing
    run; either advancing resets the watchdog window. Missing → ``(0, 0)``.
    """
    rows = 0
    max_lvl = 0
    try:
        with open(recorder_jsonl, "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except ValueError:
                    continue
                rows += 1
                if isinstance(row, dict):
                    v = row.get("level_after_action")
                    if v is None:
                        v = row.get("levels_completed", 0)
                    try:
                        lvl = int(v or 0)
                    except (TypeError, ValueError):
                        lvl = 0
                    if lvl > max_lvl:
                        max_lvl = lvl
    except OSError:
        return 0, 0
    return rows, max_lvl


def _episode_watchdog(
    root_pid: int,
    plog: Path,
    server_log: Path,
    recorder_jsonl: Path,
    plog_after_byte: int,
    stop_event: "threading.Event",
    stall_flag: List[bool],
    *,
    quiescence_s: float = M3_WATCHDOG_QUIESCENCE_S,
    poll_s: float = M3_WATCHDOG_POLL_S,
) -> Optional[str]:
    """Cooperative post-stream-complete quiescence monitor.

    Runs in a daemon thread concurrent with ``child.wait``. The window only
    STARTS counting after a ``stream_complete`` event has appeared in the
    proxy jsonl (the quiescence signature is valid ONLY post-stream-complete;
    pre-stream-complete silence is normal in-flight latency, never a wedge).

    Once post-stream-complete, on every ``poll_s`` sample it requires ALL of:
      (a) NO new proxy ``stream_request`` past the recorded byte offset,
      (b) the agentica server log size UNCHANGED,
      (c) the recorder jsonl row-count AND max-level UNCHANGED,
      (d) the proc-tree CPU jiffies delta == 0 (the disambiguator: quiet
          logs + still-burning CPU is a busy run, NOT a wedge).
    If ALL hold continuously for ``quiescence_s`` → returns
    ``TS_ORCHESTRATOR_STALL`` and sets ``stall_flag[0]=True``. ANY sample
    failing ANY of a–d RESETS the window start. The monitor exits early
    (returning ``None``, no stall) the instant ``stop_event`` is set (the
    child finished on its own — the healthy path; the watchdog stayed
    passive and never touched the kill path). It NEVER fires pre-
    stream-complete.
    """
    after_byte = plog_after_byte
    pids = _proc_tree_pids(root_pid)
    last_cpu = _proc_tree_cpu_jiffies(pids)
    last_server = _path_size(server_log)
    last_rec = _recorder_progress(recorder_jsonl)
    window_start: Optional[float] = None
    stream_complete_seen = False

    while not stop_event.is_set():
        if stop_event.wait(poll_s):
            return None  # child exited / joined — healthy path, no stall

        if not stream_complete_seen:
            # W2: anchored at the CURRENT cursor (initially the episode's
            # recorded ``plog_after`` start offset). Across polls the
            # stream_complete scan windows are contiguous
            # ([start,EOF1)∪[EOF1,EOF2)…) so no completion is missed, while a
            # PRIOR S0 episode's stream_complete strictly before the episode
            # start offset is NEVER scanned ⇒ the window cannot arm on a stale
            # completion for a fresh in-flight S0 episode.
            stream_complete_seen = _proxy_saw_stream_complete(plog, after_byte)
            # Keep advancing the proxy cursor even pre-stream-complete so the
            # post-completion stream_request check is correctly anchored.
            after_byte, _ = _proxy_new_stream_request(plog, after_byte)
            # Refresh progress baselines so the FIRST post-complete sample
            # measures deltas from stream-complete, not from launch.
            last_cpu = _proc_tree_cpu_jiffies(_proc_tree_pids(root_pid))
            last_server = _path_size(server_log)
            last_rec = _recorder_progress(recorder_jsonl)
            window_start = None
            continue

        after_byte, saw_req = _proxy_new_stream_request(plog, after_byte)
        cur_server = _path_size(server_log)
        cur_rec = _recorder_progress(recorder_jsonl)
        cur_pids = _proc_tree_pids(root_pid)
        cur_cpu = _proc_tree_cpu_jiffies(cur_pids)

        quiet = (
            (not saw_req)                       # (a) no new stream_request
            and cur_server == last_server       # (b) server log frozen
            and cur_rec == last_rec             # (c) recorder rows+level frozen
            and (cur_cpu - last_cpu) == 0       # (d) zero CPU progress
        )
        # CPU/server/recorder baselines roll forward every sample so a
        # SLOW-but-real progression (one tick every few polls) still resets.
        last_cpu = cur_cpu
        last_server = cur_server
        last_rec = cur_rec

        if not quiet:
            window_start = None
            continue
        now = time.time()
        if window_start is None:
            window_start = now
        elif (now - window_start) >= quiescence_s:
            stall_flag[0] = True
            return TS_ORCHESTRATOR_STALL
    return None


def _starttime_gated_kill(pid: int, sig: int, snap_start: Optional[int]) -> None:
    """``os.kill(pid, sig)`` ONLY if ``pid``'s starttime still == ``snap_start``.

    W3 (codex r1): re-read ``/proc/<pid>/stat`` field-22 immediately before
    the signal. If the process is gone, or its starttime differs from the
    value captured when the pid was first discovered, the integer pid now
    names a DIFFERENT (recycled / unrelated) incarnation — SKIP the kill, do
    NOT signal it. ESRCH from a race between the re-read and ``os.kill`` is
    swallowed. ``snap_start`` of ``None`` (pid already gone at snapshot time)
    is likewise never signalled.
    """
    if snap_start is None:
        return
    if _proc_starttime(pid) != snap_start:
        return  # recycled / exited — not the incarnation we snapshotted
    try:
        os.kill(pid, sig)
    except (OSError, ProcessLookupError):
        pass


def _kill_proc_tree(root_pid: int, grace_s: float = 15.0) -> None:
    """SIGTERM then (after ``grace_s``) SIGKILL EXACT pids of OUR tree.

    ONLY numeric PIDs from the PPID-rooted forest are signalled — there is NO
    name match and NO broad process-name sweep anywhere. ESRCH (already gone)
    is swallowed; the current runner PID is never in this tree so it can
    never self-signal.

    W3 (codex r1) — PID-reuse hardening: the tree is snapshotted ONCE (pid →
    field-22 starttime) before the SIGTERM pass. Every ``os.kill`` (both the
    SIGTERM pass and the post-grace SIGKILL re-scan) is starttime-gated: the
    pid is re-checked against its snapshot starttime immediately before the
    signal and SKIPPED if it changed or the proc is gone (the integer pid was
    recycled to an unrelated process during the grace window). Re-root guard:
    the SIGKILL pass does NOT re-resolve a fresh tree from ``root_pid`` (a
    recycled root could graft an unrelated subtree); it only re-signals the
    still-alive subset of the ORIGINAL snapshot whose starttime still matches.
    """
    # Snapshot the exact tree + each pid's starttime ONCE (the kill-set).
    snap: Dict[int, Optional[int]] = {}
    for pid in _proc_tree_pids(root_pid):
        snap[pid] = _proc_starttime(pid)
    root_start = snap.get(root_pid)

    for pid, st in snap.items():
        _starttime_gated_kill(pid, signal.SIGTERM, st)

    deadline = time.time() + grace_s
    while time.time() < deadline:
        # Re-root guard: probe the ORIGINAL root incarnation only.
        if root_start is None or _proc_starttime(root_pid) != root_start:
            break  # original root gone (or pid recycled) — stop waiting
        try:
            os.kill(root_pid, 0)
        except OSError:
            break
        time.sleep(0.5)

    # SIGKILL pass over the SAME snapshot (no fresh tree re-resolution from a
    # possibly-recycled root) — only the still-matching incarnations.
    for pid, st in snap.items():
        _starttime_gated_kill(pid, signal.SIGKILL, st)


def _s0_scoped_cleanup(child: Optional[subprocess.Popen]) -> None:
    """SCOPED S0 teardown. NEVER a broad process-name / global kill.

    1. kill ONLY the tracked child PID (+ its process group),
    2. ``fuser -k -KILL`` the two S0 ports ONLY (9091/tcp, 2345/tcp),
    3. wait for the ports to close.
    The S0 flock (held by the caller) makes a broad kill unnecessary because
    no other S0 arm can be running concurrently.
    """
    if child is not None and child.poll() is None:
        try:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        except OSError:
            pass
    # Scoped port reclamation — the two S0 ports ONLY.
    for port in (S0_PROXY_PORT, S0_SERVER_PORT):
        if shutil.which("fuser"):
            try:
                subprocess.run(
                    ["fuser", "-k", "-KILL", f"{port}/tcp"],
                    capture_output=True,
                    timeout=20,
                )
            except (OSError, subprocess.SubprocessError):
                pass
    # ports-closed wait.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if not _port_bound(S0_PROXY_PORT) and not _port_bound(S0_SERVER_PORT):
            break
        time.sleep(1.0)


# --------------------------------------------------------------------------- #
# small json/io helpers
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    continue
    except OSError:
        return rows
    return rows


def _safe_read(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _as_num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Step 8: Orchestration (breadth cap, arm-order alternation, reserve, retry)
# --------------------------------------------------------------------------- #


def run_idea(
    idea: IdeaSpec,
    out_dir: Path,
    breadth_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one idea end-to-end: consume the ordered reserve until 5 valid
    pairs (or the reserve is exhausted), alternating arm order per seed,
    retrying a one-arm-only-infra-error pair ONCE, then emit the contract.

    rev4-r6: each seed is additionally wrapped in a PREDECLARED bounded
    same-seed retry. When EITHER arm of the pair terminates outside
    {normal, max_episode_budget} the WHOLE pair (both arms, same seed, same
    faithful / S0 path) is re-run up to ``M3_SAME_SEED_RETRY_N`` times. The
    retry is arm-agnostic at this level (applied identically to both arms),
    is NOT folded into ``valid_pair_count``, and the per-arm M3 / stall
    incidence is accumulated over ALL attempts (including retried) and
    reported SEPARATELY in summary.json (``m3_rate``). The inner
    ``_run_pair`` one-shot infra retry is preserved unchanged.

    breadth_state (optional) carries {touched_files:int, wall_start:float};
    when the per-idea breadth cap (≤2 files OR ≤1h, 1 harness + 1 smoke pass)
    is exhausted while still failing, the idea PARKs with reason
    ``breadth_cap_exhausted``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl = idea.ctrl_arm()
    treat = idea.treat_arm()
    arm_role_by_name = {ctrl.name: "ctrl", treat.name: "treat"}

    episodes: List[EpisodeResult] = []
    valid = 0
    wall_start = (breadth_state or {}).get("wall_start", time.time())
    touched = (breadth_state or {}).get("touched_files", 0)

    # rev4-r6 per-arm attempt / M3 / stall accumulators over ALL attempts
    # (incl. bounded retries). These never fold into valid_pair_count; they
    # drive the SEPARATE m3_rate / m3_unmeasurable summary fields.
    attempt_counts: Dict[str, int] = {ctrl.name: 0, treat.name: 0}
    m3_counts: Dict[str, int] = {ctrl.name: 0, treat.name: 0}
    # Per-seed: did EVERY attempt of this seed leave ≥1 arm M3-wedged?
    seed_all_attempts_wedged: List[bool] = []

    def _arm_wedged(ep: Optional[EpisodeResult]) -> bool:
        return ep is not None and ep.terminal_status in (
            TS_ORCHESTRATOR_STALL,
            TS_M3_NONFAITHFUL,
        )

    slot = 0
    for i, seed in enumerate(idea.seeds):
        if valid >= MIN_VALID_PAIRS_DECIDE:
            break
        # breadth cap (rev4 §4 r2#4): ≤2 files OR ≤1h wall.
        if touched > BREADTH_MAX_FILES or (time.time() - wall_start) > BREADTH_MAX_WALL_S:
            return _park(
                idea, out_dir, episodes, arm_role_by_name,
                "breadth_cap_exhausted",
                _retry_stats(
                    attempt_counts, m3_counts, seed_all_attempts_wedged, valid
                ),
            )

        # per-seed arm-order alternation (order_index even → ctrl-first).
        # ctrl_first is keyed on the SEED INDEX i (NOT the attempt), so the
        # i%2 alternation is preserved IDENTICALLY across all bounded
        # retries of this seed.
        ctrl_first = (i % 2) == 0

        seed_pair_eps: List[EpisodeResult] = []
        seed_valid = False
        # every attempt of THIS seed had a wedged arm? (assume True until an
        # attempt is observed clean for ≥1 of its arms-of-interest).
        seed_every_attempt_wedged = True
        # +1 = the initial attempt; up to M3_SAME_SEED_RETRY_N RE-runs.
        for attempt in range(M3_SAME_SEED_RETRY_N + 1):
            # A transient port collision / ss fail-closed surfaces as a
            # PortConflict out of the lock acquire inside _run_pair. It must
            # NOT crash the whole idea. Catch it, synthesize a
            # TS_PORT_CONFLICT EpisodeResult for the affected arm, and route
            # through the EXISTING _run_pair retry-once path (no new path).
            try:
                pair_eps, _retried = _run_pair(
                    idea, ctrl, treat, seed, i, slot, out_dir, ctrl_first
                )
            except PortConflict:
                try:
                    pair_eps, _retried = _run_pair(
                        idea, ctrl, treat, seed, i, slot, out_dir, ctrl_first
                    )
                except PortConflict:
                    pair_eps = _port_conflict_pair(
                        idea, ctrl, treat, seed, i, slot
                    )
            slot += 1

            c = next((e for e in pair_eps if e.arm == ctrl.name), None)
            t = next((e for e in pair_eps if e.arm == treat.name), None)
            # Accumulate per-arm attempt + M3/stall counts over EVERY
            # attempt (incl. retried) — arm-identical, never folded into
            # valid_pair_count.
            for arm_obj, ep in ((ctrl, c), (treat, t)):
                if ep is not None:
                    attempt_counts[arm_obj.name] += 1
                    if _arm_wedged(ep):
                        m3_counts[arm_obj.name] += 1
            this_attempt_wedged = _arm_wedged(c) or _arm_wedged(t)
            if not this_attempt_wedged:
                seed_every_attempt_wedged = False

            # Keep the LATEST attempt's pair as the contract episodes for
            # this seed (one row-pair per seed in episodes.jsonl, same as
            # the pre-r6 one-shot behavior — the retry replaces, never
            # appends, so valid_pair_count is unaffected by the retry).
            seed_pair_eps = pair_eps

            # Seed counts valid ONLY if BOTH arms ∈ {normal,
            # max_episode_budget} within N. A clean pair stops the retry.
            both_terminal_ok = (
                c is not None
                and t is not None
                and c.terminal_status in VALID_TERMINAL_STATUSES
                and t.terminal_status in VALID_TERMINAL_STATUSES
            )
            if both_terminal_ok:
                seed_valid = c is not None and t is not None and _ep_valid(c) and _ep_valid(t)
                break
            # Either arm outside {normal,max_episode_budget} ⇒ re-run the
            # WHOLE pair (both arms, same seed) up to N times.

        episodes.extend(seed_pair_eps)
        seed_all_attempts_wedged.append(seed_every_attempt_wedged)
        if seed_valid:
            valid += 1

    return _finalize(
        idea, out_dir, episodes, arm_role_by_name,
        _retry_stats(attempt_counts, m3_counts, seed_all_attempts_wedged, valid),
    )


def _retry_stats(
    attempt_counts: Dict[str, int],
    m3_counts: Dict[str, int],
    seed_all_attempts_wedged: List[bool],
    valid_pair_count: int,
) -> Dict[str, Any]:
    """Assemble the rev4-r6 SEPARATE m3_rate / m3_unmeasurable / retry_policy
    block. ``m3_rate`` is per-arm stalls/attempts (NOT folded into
    valid_pair_count). ``m3_unmeasurable`` is True ONLY when not a single
    valid pair was obtained AND every attempted seed hit
    orchestrator_stall / m3_nonfaithful on ≥1 arm across ALL N retries."""
    m3_rate: Dict[str, float] = {}
    for arm_name, attempts in attempt_counts.items():
        m3_rate[arm_name] = (
            (m3_counts.get(arm_name, 0) / attempts) if attempts else 0.0
        )
    any_seed_attempted = len(seed_all_attempts_wedged) > 0
    all_seeds_wedged = any_seed_attempted and all(seed_all_attempts_wedged)
    unmeasurable = bool(
        valid_pair_count == 0 and any_seed_attempted and all_seeds_wedged
    )
    reason = ""
    if unmeasurable:
        reason = (
            "0 valid pairs AND every attempted seed hit "
            "orchestrator_stall/m3_nonfaithful on ≥1 arm across all "
            f"{M3_SAME_SEED_RETRY_N} bounded retries — the M3 wedge made "
            "this idea unmeasurable on the faithful harness (definitive "
            "ONLY because True)"
        )
    return {
        "m3_rate": m3_rate,
        "m3_unmeasurable": unmeasurable,
        "m3_unmeasurable_reason": reason,
        "retry_policy": {
            "n": M3_SAME_SEED_RETRY_N,
            "K_s": M3_WATCHDOG_QUIESCENCE_S,
            "applied_to": "both_arms_identically",
        },
    }


def _port_conflict_episode(
    idea: IdeaSpec,
    arm: ArmSpec,
    seed: int,
    order_index: int,
    ports: Dict[str, int],
) -> EpisodeResult:
    """A terminal EpisodeResult for an arm whose port tuple never locked.

    No subprocess ran (the PortConflict was raised by the lock acquire), so
    every metric is censored (None) and the terminal is the EXISTING
    TS_PORT_CONFLICT (an INFRA_ERROR_STATUSES member ⇒ never a valid pair,
    excluded by _ep_valid / _is_valid_episode). No fabricated metrics.
    """
    now = _utcnow()
    return EpisodeResult(
        idea_id=idea.idea_id,
        seed=seed,
        arm=arm.name,
        order_index=order_index,
        game_id=idea.game_id,
        model=idea.model,
        code_sha="",
        store_sha="",
        ports=ports,
        start_time=now,
        end_time=now,
        levels_completed=None,
        env_actions=None,
        proxy_requests=None,
        wall_seconds=None,
        terminal_status=TS_PORT_CONFLICT,
        m3_flag=False,
        m3_reason="",
        frozen_config_sha="",
        prompt_sha="",
        skill_store_sha="",
        sha_ok=True,
        faithfulness_ok=True,
        raw_summary_path="",
    )


def _port_conflict_pair(
    idea: IdeaSpec,
    ctrl: ArmSpec,
    treat: ArmSpec,
    seed: int,
    order_index: int,
    slot: int,
) -> List[EpisodeResult]:
    """Both arms as TS_PORT_CONFLICT when the slot's lock cannot be acquired.

    A slot-lock PortConflict blocks the whole pair (neither subprocess ran),
    so both arms are the affected arm. Port labels come from the existing
    slot→port mapping so the manifest still records the intended tuple.
    """
    try:
        ports = (
            faithful_ports_for_slot(slot)
            if idea.harness == "faithful"
            else {"proxy": S0_PROXY_PORT, "server": S0_SERVER_PORT}
        )
    except PortConflict:
        ports = {"proxy": -1, "server": -1}
    return [
        _port_conflict_episode(idea, ctrl, seed, order_index, ports),
        _port_conflict_episode(idea, treat, seed, order_index, ports),
    ]


def _run_pair(
    idea: IdeaSpec,
    ctrl: ArmSpec,
    treat: ArmSpec,
    seed: int,
    order_index: int,
    slot: int,
    out_dir: Path,
    ctrl_first: bool,
) -> Tuple[List[EpisodeResult], bool]:
    """Run BOTH arms for one seed (alternated order). One-arm-only infra error
    ⇒ retry the WHOLE pair ONCE (same epoch), then exclude (never substitute
    a different-epoch arm)."""
    eps = _run_both(idea, ctrl, treat, seed, order_index, slot, out_dir, ctrl_first)
    c = next((e for e in eps if e.arm == ctrl.name), None)
    t = next((e for e in eps if e.arm == treat.name), None)
    c_infra = c is not None and c.terminal_status in INFRA_ERROR_STATUSES
    t_infra = t is not None and t.terminal_status in INFRA_ERROR_STATUSES
    # one-arm-only infra error ⇒ retry both arms once (same epoch).
    if c_infra ^ t_infra:
        eps2 = _run_both(
            idea, ctrl, treat, seed, order_index, slot, out_dir, ctrl_first
        )
        return eps2, True
    return eps, False


def _run_both(
    idea: IdeaSpec,
    ctrl: ArmSpec,
    treat: ArmSpec,
    seed: int,
    order_index: int,
    slot: int,
    out_dir: Path,
    ctrl_first: bool,
) -> List[EpisodeResult]:
    first, second = (ctrl, treat) if ctrl_first else (treat, ctrl)
    eps: List[EpisodeResult] = []
    for arm in (first, second):
        if idea.harness == "faithful":
            eps.append(
                run_faithful_arm(idea, arm, seed, order_index, slot, out_dir)
            )
        else:
            eps.append(run_s0_arm(idea, arm, seed, order_index, out_dir))
    return eps


def _ep_valid(ep: EpisodeResult) -> bool:
    return (
        ep.terminal_status in VALID_TERMINAL_STATUSES
        and ep.levels_completed is not None
        and ep.env_actions is not None
        and ep.proxy_requests is not None
        and ep.sha_ok
        and ep.faithfulness_ok
    )


def _finalize(
    idea: IdeaSpec,
    out_dir: Path,
    episodes: List[EpisodeResult],
    arm_role_by_name: Dict[str, str],
    retry_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return emit_contract_files(out_dir, episodes, arm_role_by_name, retry_stats)


def _park(
    idea: IdeaSpec,
    out_dir: Path,
    episodes: List[EpisodeResult],
    arm_role_by_name: Dict[str, str],
    reason: str,
    retry_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = emit_contract_files(out_dir, episodes, arm_role_by_name, retry_stats)
    summary["classifier_result"] = "PARK"
    summary["classifier_reason"] = reason
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _builtin_idea(idea_id: str) -> IdeaSpec:
    """The rev4 §2 wave-1 ideas (idea_id selects one)."""
    seeds = [43, 47, 48, 49, 51, 53, 55, 57, 59, 61, 63]
    if idea_id == "idea1_faithful_threshold":
        return IdeaSpec(
            idea_id=idea_id,
            harness="faithful",
            arms=[
                ArmSpec("none", "ctrl", {"A3_EXT": "none"}),
                ArmSpec("trace2skill", "treat", {"A3_EXT": "trace2skill"}),
            ],
            seeds=seeds,
        )
    if idea_id == "idea2_s0_vg_attribution":
        return IdeaSpec(
            idea_id=idea_id,
            harness="s0",
            arms=[
                ArmSpec("pure", "ctrl", {"S0_VG_ENABLED": "0", "S0_SKILL_ENABLED": "0"}),
                ArmSpec("vg", "treat", {"S0_VG_ENABLED": "1", "S0_SKILL_ENABLED": "0"}),
            ],
            seeds=seeds,
        )
    raise SystemExit(f"unknown builtin idea_id {idea_id!r}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Skill-variation portfolio runner")
    ap.add_argument("idea_id", help="builtin idea id, or path to an IdeaSpec JSON")
    ap.add_argument(
        "--spec-json", help="path to an IdeaSpec JSON (overrides builtin)"
    )
    ap.add_argument("--out-dir", help="output dir (default reports/skill_portfolio/<idea>/<ts>)")
    ap.add_argument(
        "--preflight-only", action="store_true", help="run preflight then exit"
    )
    args = ap.parse_args(argv)

    pf = preflight()
    print(json.dumps({"preflight": pf}))
    if pf["result"] != "pass":
        # sha_mismatch / abort ⇒ no subprocess is ever launched.
        return 13 if pf["result"] == "sha_mismatch" else 12
    if args.preflight_only:
        return 0

    if args.spec_json:
        idea = IdeaSpec.from_dict(_load_json(Path(args.spec_json)) or {})
    elif os.path.isfile(args.idea_id):
        idea = IdeaSpec.from_dict(_load_json(Path(args.idea_id)) or {})
    else:
        idea = _builtin_idea(args.idea_id)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else OUT_ROOT / idea.idea_id / _utcnow()
    )
    summary = run_idea(idea, out_dir)
    print(json.dumps({"idea_id": idea.idea_id, "out_dir": str(out_dir),
                       "classifier_result": summary.get("classifier_result")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

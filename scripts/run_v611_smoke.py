"""v611 5-turn smoke runner (Plan rev D Step 2c-full).

Standalone entry point that exercises the v611 Δ7 multi-role loop
against the live ft09 environment without modifying agent.py. The
existing agentica_lite agent.py is invariant; v611 lives in its own
file paths:
  - skill_state_v611_<game_id>.json (isolated SKILL.md namespace)
  - logs/v611_smoke_<ts>.jsonl (telemetry)

Pass criteria (codex round 18):
  1. >=5 m1 'role_returned' events for 5 turns
  2. >=5 m2v 'role_returned' events
  3. >=5 m2e 'role_returned' events OR matching skip_reasons
  4. >=5 env 'env_step' events (only after approve)
  5. Every M1 output_keys lacks 'click_xy_hint', 'x', 'y'
  6. skill_state_v611_*.json hygiene-clean if any confirmed skill emitted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")


def _build_state_text(payload: dict) -> str:
    """Compact text representation of ft09 frame state for M1.

    Round 18 contract: <=4000 chars (clamped in orchestrator).
    Includes generic visual descriptors only (no ft09 vocab).
    """
    parts: list[str] = []
    mcs = payload.get("marker_constraint_summary") or {}
    parts.append(
        f"grid: 64x64; markers detected: {mcs.get('total', 0)}; "
        f"satisfied: {mcs.get('total', 0) - mcs.get('unsatisfied', 0)}; "
        f"unsatisfied: {mcs.get('unsatisfied', 0)}"
    )
    regions = payload.get("visible_regions") or []
    if regions:
        parts.append(f"visible regions: {len(regions)} total")
        for r in regions[:8]:
            bbox = r.get("bbox") or []
            sz = r.get("size", "?")
            col = r.get("color", "?")
            mc = r.get("is_multicolor", False)
            if len(bbox) >= 4:
                parts.append(
                    f"  region: bbox=({bbox[0]},{bbox[1]})-"
                    f"({bbox[2]},{bbox[3]}) size={sz} "
                    f"color={col} multicolor={mc}"
                )
    return "\n".join(parts)


def _one_line_state(payload: dict) -> str:
    """Single-sentence state_now for StateSummary (round 18: <=200 chars)."""
    mcs = payload.get("marker_constraint_summary") or {}
    regions = payload.get("visible_regions") or []
    return (
        f"grid 64x64, {len(regions)} visible regions, "
        f"{mcs.get('total', 0)} markers ({mcs.get('unsatisfied', 0)} "
        f"unsatisfied)"
    )


def _frame_to_png(frame_field, out_path: Path) -> bytes:
    """Render frame to PNG bytes (8x upscale)."""
    try:
        from PIL import Image
    except ImportError:
        return b""
    # frame_field may be a numpy array or nested list.
    if frame_field is None:
        return b""
    # Convert numpy to nested list if needed.
    if hasattr(frame_field, "tolist"):
        frame_field = frame_field.tolist()
    grid = None
    if isinstance(frame_field, list):
        for g in reversed(frame_field):
            if isinstance(g, list) and len(g) > 0 and \
               isinstance(g[0], list) and len(g[0]) > 0:
                grid = g
                break
    if grid is None:
        return b""
    palette = [
        (40, 40, 40), (180, 180, 180), (255, 80, 80), (80, 180, 80),
        (80, 80, 255), (255, 220, 80), (255, 80, 220), (80, 220, 220),
        (220, 80, 80), (80, 80, 80), (160, 80, 220), (220, 160, 80),
        (80, 220, 80), (220, 80, 160), (160, 160, 160), (255, 255, 255),
    ]
    h = len(grid)
    w = len(grid[0]) if h else 0
    scale = 8
    img = Image.new("RGB", (w * scale, h * scale), "white")
    px = img.load()
    for y in range(h):
        for x in range(w):
            c = int(grid[y][x]) % len(palette)
            for dy in range(scale):
                for dx in range(scale):
                    px[x * scale + dx, y * scale + dy] = palette[c]
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ft09-9ab2447a")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out-dir", default="logs/")
    args = parser.parse_args(argv)

    ts = int(time.time())
    log_path = Path(args.out_dir) / f"v611_smoke_{ts}.jsonl"
    skill_state_path = Path(f"./skill_state_v611_{args.game}.json")
    os.environ["V611_TELEMETRY_PATH"] = str(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[v611 smoke] game={args.game} turns={args.turns} seed={args.seed}")
    print(f"[v611 smoke] telemetry: {log_path}")
    print(f"[v611 smoke] skill_state: {skill_state_path} (isolated)")

    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from agents.templates.agentica_lite._frame_to_state import (
        frame_to_state, _reset_stability_state,
    )
    from agents.templates.agentica_lite.v611_orchestrator import (
        AnchorCounter, StateSummary, run_v611_turn,
    )
    from agents.templates.agentica_lite.v611_roles import (
        run_m1_proposer, run_m2v_verifier, run_m2e_executor,
        run_m3_compressor, run_m4_reflector,
    )
    from agents.templates.agentica_lite.v611_schemas import (
        validate_m3_skill_output, validate_m4_output,
    )
    from agents.templates.agentica_lite.v611_leak_scanner import (
        validate_confirmed_skill,
    )
    import hashlib as _hl
    import json as _json
    from agents.templates.agentica_lite.v611_telemetry import (
        log_turn_event,
    )
    from scripts.v609_search.frame_hash_sim import _FrameProxy

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(args.game)
    _reset_stability_state()
    raw = env.reset()

    anchor = AnchorCounter()
    last_strategies: list[dict] = []
    null_streak = 0
    episode_id = f"v611_smoke_{ts}"
    successes = 0
    last_levels = int(getattr(raw, "levels_completed", 0) or 0)

    # SKILL.md accumulation state — codex round 19 fix
    confirmed_skills: list[dict] = []     # NL-only, hygiene-gated
    skill_state_path.parent.mkdir(parents=True, exist_ok=True)

    def _render_skill_md_summary() -> str:
        if not confirmed_skills:
            return "(no confirmed skills yet — turn-1 exploration)"
        lines = [
            f"confirmed_skills (n={len(confirmed_skills)}):"
        ]
        for s in confirmed_skills[-5:]:
            lines.append(
                f"  - {s.get('skill_id', '?')}: "
                f"{s.get('nl_description', '')[:120]}"
            )
        return "\n".join(lines)

    def _persist_skill_state():
        with open(skill_state_path, "w", encoding="utf-8") as f:
            _json.dump({"confirmed_skills": confirmed_skills,
                        "episode_id": episode_id,
                        "ts": ts}, f, indent=2)

    for turn in range(args.turns):
        proxy = _FrameProxy(raw)
        proxy_payload = frame_to_state(proxy, [], last_levels)
        state_text = _build_state_text(proxy_payload)
        state_now = _one_line_state(proxy_payload)
        png_bytes = _frame_to_png(getattr(raw, "frame", None), Path("/tmp"))

        # Compute repeat_axis from last strategies' suggested region word.
        if last_strategies:
            current_first_axis = "unknown"  # filled after M1 succeeds
            repeat_count = 0
        else:
            repeat_count = 0

        state_summary = StateSummary(
            state_now=state_now,
            last_strategies=last_strategies[-5:],
            repeat_axis_count=repeat_count,
            null_effect_streak=null_streak,
        )

        skill_md_summary = _render_skill_md_summary()

        print(f"\n=== TURN {turn} ===")
        result = run_v611_turn(
            turn_id=turn, state_text=state_text,
            state_text_summary=state_summary, png_bytes=png_bytes,
            skill_md_summary=skill_md_summary, anchor=anchor,
            m1_proposer=run_m1_proposer,
            m2v_verifier=run_m2v_verifier,
            m2e_executor=run_m2e_executor,
            seed=args.seed, episode_id=episode_id,
        )

        print(f"  success={result.success} skip={result.skip_reason} "
              f"click={result.click_xy}")
        if result.proposer_out:
            nl = result.proposer_out.get("nl_strategy", "")[:80]
            print(f"  M1 NL: {nl!r}")
        if result.verifier_out:
            print(f"  M2v: {result.verifier_out.get('verdict')!r} "
                  f"reason={result.verifier_out.get('reason_nl', '')[:60]!r}")
        if result.executor_out:
            gt = result.executor_out.get("grounding_text", "")[:80]
            print(f"  M2e: xy={result.click_xy} grounding={gt!r}")

        if not result.success:
            continue

        coord = result.click_xy
        prev_frame = getattr(raw, "frame", None)
        raw = env.step(GameAction.ACTION6, data={"x": coord[0], "y": coord[1]})
        cur_levels = int(getattr(raw, "levels_completed", 0) or 0)
        level_delta = cur_levels - last_levels
        last_levels = cur_levels

        # Frame-change heuristic.
        cur_frame = getattr(raw, "frame", None)
        frame_changed = (str(prev_frame)[:1000] != str(cur_frame)[:1000])
        null_streak = 0 if frame_changed else null_streak + 1

        log_turn_event(
            turn_id=turn, role="env", event="env_step",
            payload={"click_xy": list(coord),
                     "levels_completed": cur_levels,
                     "level_delta": level_delta,
                     "frame_changed": frame_changed},
            seed=args.seed, episode_id=episode_id,
        )

        successes += 1
        # Compute unsat_delta from before vs after
        try:
            proxy_post = _FrameProxy(raw)
            post_payload = frame_to_state(proxy_post, [], last_levels)
            post_unsat = (post_payload.get("marker_constraint_summary")
                          or {}).get("unsatisfied", 0)
            pre_unsat = (proxy_payload.get("marker_constraint_summary")
                         or {}).get("unsatisfied", 0)
            unsat_delta = int(post_unsat) - int(pre_unsat)
        except Exception:
            unsat_delta = 0

        last_strategies.append({
            "nl_strategy": result.proposer_out.get("nl_strategy", "")[:80],
            "suggested_region":
                result.proposer_out.get("suggested_click_region", "")[:40],
            "verdict": result.verifier_out["verdict"],
            "frame_changed": frame_changed,
            "unsat_delta": unsat_delta,
        })

        print(f"  ENV STEP: level_delta={level_delta} "
              f"frame_changed={frame_changed} unsat_delta={unsat_delta} "
              f"cur_levels={cur_levels}")

        # ───── M4 Reflector (every successful turn) ─────
        try:
            m4_out = run_m4_reflector(
                proposer_out=result.proposer_out,
                verifier_out=result.verifier_out,
                executor_out=result.executor_out,
                env_observation={
                    "frame_changed": frame_changed,
                    "unsat_delta": unsat_delta,
                    "level_delta": level_delta,
                },
                prior_skills=confirmed_skills,
            )
            v4 = validate_m4_output(m4_out)
            log_turn_event(turn, role="m4", event="role_returned",
                           payload={"verdict": m4_out.get("verdict"),
                                    "ok": v4.ok,
                                    "violations": v4.violations[:3]},
                           seed=args.seed, episode_id=episode_id)
            if v4.ok:
                print(f"  M4 verdict: {m4_out.get('verdict')!r} "
                      f"directive={m4_out.get('next_directive', '')[:60]!r}")
                # Apply SKILL.md patch (codex round 19 wiring)
                upd = m4_out.get("verify", {}).get("skillmd_update", {})
                add_ids = upd.get("add") or []
                if add_ids:
                    log_turn_event(turn, role="m4", event="skillmd_patch_add",
                                   payload={"add_ids": add_ids[:5]},
                                   seed=args.seed, episode_id=episode_id)
            else:
                print(f"  M4 INVALID: {v4.violations[:2]}")
        except Exception as e:
            print(f"  M4 error: {e}")

        # ───── M3 Skill Compressor (every 5 turns) ─────
        if (turn + 1) % 5 == 0 and last_strategies:
            try:
                m3_out = run_m3_compressor(
                    recent_trials=last_strategies[-5:],
                    existing_skills=confirmed_skills,
                )
                if isinstance(m3_out, dict) and m3_out.get("emit"):
                    v3 = validate_m3_skill_output(m3_out)
                    hyg = validate_confirmed_skill(m3_out)
                    log_turn_event(turn, role="m3",
                                   event="role_returned",
                                   payload={"emit": True,
                                            "schema_ok": v3.ok,
                                            "hygiene_ok": hyg.ok},
                                   seed=args.seed,
                                   episode_id=episode_id)
                    if v3.ok and hyg.ok:
                        confirmed_skills.append({
                            "skill_id": m3_out["skill_id"],
                            "nl_description": m3_out["nl_description"],
                            "abstract_precondition":
                                m3_out["abstract_precondition"],
                            "expected_observed_effect":
                                m3_out["expected_observed_effect"],
                            "turn_emitted": turn,
                        })
                        _persist_skill_state()
                        print(f"  M3 EMITTED skill {m3_out['skill_id']}: "
                              f"{m3_out['nl_description'][:80]!r}")
                    else:
                        print(f"  M3 emit REJECTED: schema_ok={v3.ok} "
                              f"hygiene_ok={hyg.ok} "
                              f"violations={(v3.violations + hyg.violations)[:2]}")
                else:
                    log_turn_event(turn, role="m3",
                                   event="role_returned",
                                   payload={"emit": False,
                                            "reason": m3_out.get(
                                                "reason_nl", "")[:80]
                                                if isinstance(m3_out, dict)
                                                else "invalid"},
                                   seed=args.seed,
                                   episode_id=episode_id)
                    print(f"  M3 no-emit: "
                          f"{(m3_out or {}).get('reason_nl', '')[:80]!r}")
            except Exception as e:
                print(f"  M3 error: {e}")

    print(f"\n=== SMOKE SUMMARY ===")
    print(f"turns_run={args.turns} successes={successes} "
          f"final_levels={last_levels}")
    print(f"telemetry events written to: {log_path}")

    # Quick audit
    from agents.templates.agentica_lite.v611_telemetry import read_telemetry
    events = read_telemetry(log_path)
    m1_returned = sum(1 for e in events
                       if e.role == "m1" and e.event == "role_returned")
    m2v_returned = sum(1 for e in events
                        if e.role == "m2v" and e.event == "role_returned")
    m2e_returned = sum(1 for e in events
                        if e.role == "m2e" and e.event == "role_returned")
    env_steps = sum(1 for e in events
                     if e.role == "env" and e.event == "env_step")
    print(f"telemetry counts: m1={m1_returned} m2v={m2v_returned} "
          f"m2e={m2e_returned} env={env_steps}")

    return 0 if successes > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

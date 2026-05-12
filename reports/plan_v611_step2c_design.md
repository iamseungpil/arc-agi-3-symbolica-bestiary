# v611 Step 2c — production wiring design (pre-codex)

Plan rev D §Step 2b-2 / 2b-3. This document is the design proposal
before implementation. Goes through codex adversarial review.

## Goal

Wire the v611 orchestrator (Step 2b, 96/96 tests) into the real
agentica_lite agent.py loop with real LLM (gpt-5.4-mini via TRAPI) and
run a 5-turn ft09 smoke test.

## New module: `v611_roles.py`

Three role-runner functions injected into `run_v611_turn()`:

### `run_m1_proposer(state_text, skill_md_summary, anchor_summary, rejection_reason)`

1. Load `v611_prompts/m1_proposer_system.md` once at module init
2. Build user message:
   ```
   STATE TEXT:
   {state_text}

   SKILL.md SUMMARY:
   {skill_md_summary}

   {if anchor_summary: "ANCHOR SUMMARY (start fresh from this only):"
                       + anchor_summary}
   {if rejection_reason: "REJECTION HINT (avoid):" + rejection_reason}
   ```
3. Call `client.chat.completions.create(model='gpt-5.4-mini',
   messages=[{system: M1_PROPOSER_SYSTEM}, {user: ...}],
   response_format={'type': 'json_object'}, max_tokens=600)`
4. Parse JSON; on JSONDecodeError or non-dict result → return `{}`
5. Return parsed dict

### `run_m2v_verifier(proposer_out, state_text_summary)`

1. Load `v611_prompts/m2v_verifier_system.md` once
2. Build user message:
   ```
   PROPOSER NL STRATEGY:
   {proposer_out['nl_strategy']}

   SUGGESTED REGION:
   {proposer_out['suggested_click_region']}

   STATE SUMMARY:
   {state_text_summary}
   ```
3. Call `client.chat.completions.create(model='gpt-5.4-mini',
   messages=[{system: M2V_VERIFIER_SYSTEM}, {user: ...}],
   response_format={'type': 'json_object'}, max_tokens=200)`
4. Parse JSON; return dict
5. NO frame, NO PNG, NO skill_md in messages list

### `run_m2e_executor(approved_out, png_bytes)`

1. Load `v611_prompts/m2e_executor_system.md` once
2. Build multimodal user message:
   ```
   APPROVED NL STRATEGY:
   {approved_out['nl_strategy']}

   SUGGESTED REGION:
   {approved_out['suggested_click_region']}
   ```
   + image attachment (base64 PNG)
3. Call `client.chat.completions.create(model='gpt-5.4-mini' OR
   'gpt-5.4-vision' if available, messages=[{system: M2E_EXECUTOR_SYSTEM},
   {user: text + image_url(base64)}], response_format={'type':
   'json_object'}, max_tokens=300)`
4. Parse JSON; return dict
5. NO skill_md, NO state_text in messages list

## agent.py integration

Add to existing `ArcgenticaLiteAgent` class (NOT new file):

```python
def _maybe_v611_path(self, env, max_actions: int):
    """If ARC_LITE_V611=1, run the multi-role loop."""
    if os.environ.get("ARC_LITE_V611") != "1":
        return None  # falls back to legacy v608d path
    from .v611_orchestrator import AnchorCounter, run_v611_turn, StateSummary
    from .v611_roles import (run_m1_proposer, run_m2v_verifier,
                              run_m2e_executor)
    from .v611_telemetry import log_turn_event

    anchor = AnchorCounter()
    skill_md_summary = ""  # populated each turn
    last_strategies_buffer: list[dict] = []
    null_streak = 0
    episode_id = self.scorecard_id or "ep_v611"

    for turn in range(max_actions):
        # 1. Build frame artifacts
        frame = env.frame
        state_text = self._build_state_text(frame)
        png_bytes = self._render_png(frame)
        skill_md_summary = self._build_skill_md_summary()

        # 2. Build StateSummary for M2v
        state_summary = StateSummary(
            state_now=self._one_line_state(frame),
            last_strategies=last_strategies_buffer[-5:],
            repeat_axis_count=self._count_repeated_axis(
                last_strategies_buffer),
            null_effect_streak=null_streak,
        )

        # 3. Run 3-role turn
        result = run_v611_turn(
            turn_id=turn, state_text=state_text,
            state_text_summary=state_summary, png_bytes=png_bytes,
            skill_md_summary=skill_md_summary, anchor=anchor,
            m1_proposer=run_m1_proposer,
            m2v_verifier=run_m2v_verifier,
            m2e_executor=run_m2e_executor,
            seed=int(os.environ.get("ARC_LITE_SEED", 5)),
            episode_id=episode_id,
        )

        if not result.success:
            # skip / reject_anchor / etc
            continue

        # 4. Execute env step
        coord = result.click_xy
        env_result = env.step(GameAction.ACTION6,
                               data={"x": coord[0], "y": coord[1]})
        log_turn_event(turn, role="env", event="env_step",
                       payload={"click_xy": list(coord),
                                "levels_completed":
                                  int(getattr(env_result, "levels_completed",
                                              0) or 0),
                                "unsat_delta":
                                  self._compute_unsat_delta(frame,
                                                              env_result)},
                       seed=..., episode_id=episode_id)

        # 5. Update last_strategies_buffer + null_streak
        frame_changed = self._frame_hash(frame) != self._frame_hash(env_result)
        last_strategies_buffer.append({
            "text": result.proposer_out.get(
                "suggested_click_region", "")[:60],
            "verdict": result.verifier_out["verdict"],
            "frame_changed": frame_changed,
        })
        null_streak = 0 if frame_changed else (null_streak + 1)

        # 6. M4 reflection + SKILL.md update — uses v608d M4 path
        # (unchanged from rev B Δ5)
        self._run_m4_reflector(...)

    return "v611_done"
```

Add a helper to wire into existing `Agent.run()`:

```python
def run(self, ...):
    v611 = self._maybe_v611_path(self.env, self.max_actions)
    if v611 is None:
        return self._legacy_run(...)  # existing path unchanged
    return v611
```

## Smoke test plan

`scripts/run_v611_smoke.sh`:
```bash
#!/usr/bin/env bash
set -e
export ARC_LITE_V611=1
export ARC_LITE_SEED=5
export V611_TELEMETRY_PATH="logs/v611_smoke_$(date +%s).jsonl"
mkdir -p logs

python main.py -a lite -g ft09-9ab2447a --max-actions 5 \
    --no-record 2>&1 | tee logs/v611_smoke_$(date +%s)_stdout.log
```

Pass criteria (test_v611_smoke_invariants.py):
1. Telemetry file exists and contains ≥3 events per turn (m1/m2v/m2e
   role_returned + validator_ok) × 5 turns ≥ 15 events min
2. Every M1 role_returned event has `output_keys` NOT containing
   'click_xy_hint', 'x', 'y'
3. ≥1 env_step event with click_xy in [0,63]²
4. Substitute drift events counted and logged
5. SKILL.md after smoke is hygiene-clean (no coords, no level IDs)

## Open questions for codex review

Q1: Is GPT-5.4-mini multimodal endpoint sufficient for M2e PNG
grounding? Or do we need GPT-4o-vision specifically? (cycle405 used
gpt-5.4-mini text-only.)

Q2: Token budget per role realistic? M1 600 + M2v 200 + M2e 300 =
1100 output tokens per turn. Input + output ≈ 13.5K. Per-turn cost
estimate ~$2/episode.

Q3: How should `_build_state_text(frame)` differ from
`_one_line_state(frame)`? state_text goes to M1 (full); state_now
field in StateSummary goes to M2v (concise). Need explicit length
contracts.

Q4: M4 reflection runs in existing agent.py code path (v608d's M4).
Should v611 use its own Δ5 3-step M4 prompt? Or piggyback on existing?
(Plan rev B Δ5 says new 3-step verify; existing M4 is single-step.)

Q5: Sequential 5-turn smoke vs parallel episode comparison — for
smoke alone, sequential is fine. But the eventual A_MULTIROLE
ablation needs matched runs.

Q6: Should v611 share SKILL.md state with v608d? Or separate
namespace? (If shared, v608d artifacts contaminate v611 hygiene
audit.)

## Test plan checklist

Before merge:
- [ ] v611_roles.py syntax + import only test (no LLM)
- [ ] agent.py path branch test (ARC_LITE_V611=0 → legacy unchanged;
       =1 → new path entered)
- [ ] 5-turn smoke on ft09 produces telemetry with all invariants
- [ ] Hygiene audit on smoke SKILL.md output
- [ ] Run twice with same seed to verify deterministic env (LLM is
       not deterministic but env is)

## Next-step verdict

ROUND_17_VERDICT: pending codex review

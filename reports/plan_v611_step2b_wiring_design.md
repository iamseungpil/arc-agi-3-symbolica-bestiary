# v611 Step 2b — agent.py 3-role wiring design

Per Plan rev D §Δ7. Codex adversarial review before implementation.

## Per-turn flow

```
[turn N starts]
  ↓
state_payload, png = build_state(env_frame, skill_md)
  ↓
[ROLE 1: M1 Proposer] (separate API call)
  system_prompt = M1_PROPOSER_SYSTEM
  messages = [{system}, {user: state_text_summary + skill_md_summary}]
  → nl_strategy + suggested_click_region + expected_signature + rollback_trigger
  ↓
validate_m1_proposer_output(proposer_out)
  - if FAIL: retry once with parse-error feedback → if 2nd FAIL: log + skip turn
  ↓
[ROLE 2: M2v Verifier] (separate API call, fresh context)
  system_prompt = M2V_VERIFIER_SYSTEM
  messages = [{system}, {user: proposer_out.nl_strategy + suggested_region}]
  ☆ NOT PASSED: state_payload, png, skill_md, prior turn transcripts
  → verdict + reason_nl
  ↓
validate_m2v_verifier_output(verifier_out)
  - if FAIL: retry once → if 2nd FAIL: log + skip turn
  ↓
[verdict branching]
  verdict == "approve" → continue to M2e
  verdict == "reject_replan" → re-run M1 with rejection reason; retry once;
                                 second reject_replan → log + skip turn
  verdict == "reject_anchor" → log; next turn starts with FRESH M1 (no
                                 transcript carry, only summary)
  ↓
[ROLE 3: M2e Executor] (separate API call)
  system_prompt = M2E_EXECUTOR_SYSTEM
  messages = [{system},
              {user: proposer_out.nl_strategy + suggested_region + PNG image}]
  ☆ NOT PASSED: skill_md, prior reasoning, state_payload mechanics
  → click_xy_hint [x, y] + grounding_text
  ↓
validate_m2e_executor_output(executor_out)
  - if FAIL: retry once → if 2nd FAIL: log + skip turn
  ↓
env.step(ACTION6, data={"x": click_xy_hint[0], "y": click_xy_hint[1]})
  ↓
[ROLE 4: M4 Reflector] (single call, same as v608d but Δ5 3-step verify)
  Receives: full turn context (proposer, verifier, executor, env response)
  Returns: paragraph + verify{predicted_vs_observed, strategy_validity,
           skillmd_update} + verdict + next_directive
  ↓
SKILL.md update via hygiene-validated patch
  ↓
[every 5 turns: M3 NL-only skill compressor]
  ↓
[turn N+1]
```

## Mechanical separation guarantees

| Guarantee | Enforcement |
|-----------|-------------|
| Separate API calls | 3 distinct `llm_client.create()` invocations |
| Separate system prompts | 3 distinct markdown files in `v611_prompts/` |
| No shared transcript | each call has fresh `messages` list, only the immediate handoff JSON as user content |
| M1 outputs no coords | `validate_m1_proposer_output` reject any `click_xy_hint` |
| M2v sees no frame | input includes `proposer_out.nl_strategy + suggested_region` only |
| M2e sees no SKILL.md mechanics | input includes only approved `nl_strategy + suggested_region + PNG` |

## anchor-streak management

```python
class AnchorCounter:
    streak: int = 0
    last_summary: str | None = None

    def on_reject_anchor(self, m1_summary: str):
        self.streak += 1
        self.last_summary = m1_summary

    def consume_for_fresh_spawn(self) -> str | None:
        """Called at next turn start. Returns summary to use as the
        ONLY context passed to fresh M1; clears streak."""
        s = self.last_summary
        self.last_summary = None
        return s

    def on_other_verdict(self):
        self.streak = 0  # any non-anchor verdict resets
```

## File layout (new)

```
agents/templates/agentica_lite/
  v611_prompts/
    m1_proposer_system.md
    m2v_verifier_system.md
    m2e_executor_system.md
  v611_roles.py
    run_m1_proposer(state, skill_md, anchor_summary)
    run_m2v_verifier(proposer_out)
    run_m2e_executor(approved_out, png_bytes)
  v611_orchestrator.py
    AnchorCounter
    run_v611_turn(state, png, skill_md) -> TurnResult
  agent.py  [MODIFY]
    Add v611 main loop path enabled by env var ARC_LITE_V611=1
```

## v611 main loop (pseudocode)

```python
def _v611_main_loop(self, env, max_actions):
    anchor = AnchorCounter()
    for turn in range(max_actions):
        frame = env.frame
        state_text = build_state_text(frame, skill_md=self.state)
        png = render_png(frame)

        # Role 1: Proposer (with anchor-fresh-spawn handling)
        anchor_summary = anchor.consume_for_fresh_spawn()
        m1 = run_m1_proposer(state_text, self.state, anchor_summary)
        if not validate_m1_proposer_output(m1).ok:
            m1 = run_m1_proposer(state_text, self.state, anchor_summary,
                                 retry=True)
            if not validate_m1_proposer_output(m1).ok:
                log_skip("m1_invalid"); continue

        # Role 2: Verifier (separate context, no frame)
        m2v = run_m2v_verifier(m1)
        if not validate_m2v_verifier_output(m2v).ok:
            m2v = run_m2v_verifier(m1, retry=True)
            if not validate_m2v_verifier_output(m2v).ok:
                log_skip("m2v_invalid"); continue

        if m2v["verdict"] == "reject_replan":
            m1_replan = run_m1_proposer(state_text, self.state,
                                         rejection_reason=m2v["reason_nl"])
            if not validate_m1_proposer_output(m1_replan).ok:
                log_skip("m1_replan_invalid"); continue
            m1 = m1_replan
            m2v = run_m2v_verifier(m1)
            if m2v["verdict"] != "approve":
                log_skip("second_reject"); continue

        if m2v["verdict"] == "reject_anchor":
            anchor.on_reject_anchor(summarize(m1))
            log("anchor_reject_for_next_turn"); continue

        assert m2v["verdict"] == "approve"
        anchor.on_other_verdict()

        # Role 3: Executor
        m2e = run_m2e_executor(m1, png)
        if not validate_m2e_executor_output(m2e).ok:
            m2e = run_m2e_executor(m1, png, retry=True)
            if not validate_m2e_executor_output(m2e).ok:
                log_skip("m2e_invalid"); continue

        # Execute
        coord = m2e["click_xy_hint"]
        env_result = env.step(ACTION6, data={"x": coord[0], "y": coord[1]})

        # Role 4: M4 reflector (single call, sees everything)
        m4 = run_m4_reflector(m1, m2v, m2e, env_result, self.state)
        if validate_m4_output(m4).ok:
            apply_skill_md_patch(self.state, m4["verify"]["skillmd_update"])

        # M3 every 5 turns
        if turn % 5 == 4:
            m3 = run_m3_compressor(self.state)
            if validate_m3_skill_output(m3).ok and \
               validate_confirmed_skill(m3).ok:
                self.state.add_confirmed_skill(m3)
```

## Token accounting per turn

- M1 Proposer: input ~3K, output ~500 (frame text + skill_md summary in, NL out)
- M2v Verifier: input ~500, output ~100 (NL only in/out)
- M2e Executor: input ~5K (PNG ~4K + NL ~1K), output ~200
- M4 Reflector: input ~5K, output ~500
- M3 (every 5 turns): input ~2K, output ~300

Per turn ≈ 13.5K input + 1.4K output. 1500 turns ≈ 20M tokens. At
gpt-5.4-mini rates (~$0.15/1M input, $0.60/1M output) ≈ $3.84/episode.

(Codex Q3 estimate was $0.45 assuming ~$0.0001/call which was too
low. Real cost ~$4/episode. Still acceptable for pilot.)

## Open questions for codex review

Q1: Is M2v's input scope (only `nl_strategy + suggested_region`)
sufficient for it to verify without seeing frame? Or does the
verifier need at least a state-text summary too?

Q2: When verdict = `reject_replan`, the retry M1 sees the rejection
reason — is this still 'separate context' or has the boundary
weakened?

Q3: Should M2e receive ANY information beyond PNG + approved
proposer output? In particular: does it need to see prior
rejected attempts to avoid re-clicking the same area?

Q4: M4 reflector sees ALL turn outputs (M1+M2v+M2e+env). Is this
'cross-role context contamination'? Or is M4 inherently a privileged
role (it's after env step, the role boundary is no longer needed)?

Q5: Token cost ~$4/episode (not $0.45 as initially estimated). Still
acceptable for 3-seed pilot? For 12-seed confirmatory (~$50)?

Q6: Telemetry to log per turn for ablation analysis: list essentials.

## Test plan before live cycle

### Step 2b-1: unit tests for each role-runner
- `test_run_m1_proposer_calls_llm_with_correct_system_prompt`
- `test_run_m2v_verifier_input_excludes_frame_and_skill_md`
- `test_run_m2e_executor_input_includes_png_excludes_mechanics`
- `test_anchor_counter_streak_logic`

### Step 2b-2: integration smoke (3-role flow with mock LLM)
- Mock LLM returns canned valid outputs for each role
- Verify 3 distinct API calls happen
- Verify validators run after each
- Verify env.step called only after approve verdict

### Step 2b-3: 5-turn live smoke (real LLM)
- 1 episode × 5 turns on ft09
- Real gpt-5.4-mini calls
- Verify per-turn telemetry shows 3 calls
- Verify schema gates pass
- Verify no coords leak through M1 output
- Verify SKILL.md updates via hygiene gate

## Next-step verdict required from codex

ROUND_11_VERDICT: ACCEPT or revisions

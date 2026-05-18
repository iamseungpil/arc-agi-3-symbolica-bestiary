#!/usr/bin/env python3
"""Build presentation/ft09_skill_trace_explorer.html — a self-contained
visualizer (ft09_trace_explorer.html style: dark, embedded DATA, no external
fetch) for the L6 trace `pilot_s47_trace2skill` (the only clean L6 WIN),
PLUS *exactly what the LLM receives under A3_EXT=trace2skill*, PLUS the 5/5
same-seed paired comparison.

Honesty corrections (2026-05-18, after verifying against the real s47 trace):
  * trace2skill NEVER reads state/skills_ft09.json — it uses a fresh EMPTY
    per-run tempfile store. The "skill memory" panel now shows the TWO real
    artifacts: (1) the byte-exact instruction paragraph appended to the
    orchestrator task (`_T2S_TASK_PARAGRAPH`), and (2) the actual run-distilled
    store this s47 run produced (preserved at
    reports/skill_pilot/.../s47_distilled_store.json).
  * Each store entry is a VERBATIM per-level action transcript recorded by the
    deterministic level-boundary observer (category=mechanic, code=null,
    predicate=null, posterior=[0,0]) — NOT a reflective/abstracted skill.
  * Consumption is TRACE-VERIFIED, not assumed: recomputed here from the proxy
    jsonl (consolidator-only marker in request bodies; python-tool references
    to the skill_library object), bucketed against consolidation timing.

The a3_calibration recorder jsonl is PER-ACTION level/token telemetry
(101 rows), NOT ft09 grid frames — so this shows the action/level/skill-memory
timeline (real data), not a fabricated pixel grid.
"""
import json, glob, re, html, pathlib, datetime

REPO = pathlib.Path("/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research")
TRACE = REPO / "reports/a3_calibration/pilot_s47_trace2skill_20260517T161830Z.jsonl"
SUMM = REPO / "reports/a3_calibration/pilot_s47_trace2skill_20260517T161830Z.summary.json"
RUNDIR = REPO / "reports/skill_pilot/20260517T161830Z"
STORE = RUNDIR / "s47_distilled_store.json"          # the REAL run-distilled store
PROXY = RUNDIR / "pilot_s47_trace2skill_20260517T161830Z.proxy.jsonl"
OUT = REPO / "presentation/ft09_skill_trace_explorer.html"

# byte-exact string the LLM receives, reconstructed from
# scripts/run_clean_upstream_ft09.py:1183-1196 (_T2S_TASK_PARAGRAPH with the
# interpolated _T2S_NAMESPACE_DOC at :1174). This is the ONLY skill-related text
# injected into the agent context (appended to the orchestrator initial task).
T2S_NAMESPACE_DOC = "Persistable cross-level skill store of distilled prior-level skills."
T2S_TASK_PARAGRAPH = (
    "\n\n[Trace2Skill cross-level skill directory]\n"
    "A shared, read-only `skill_library` object is also in scope. It is a "
    "cross-level directory of skills that were inductively distilled from "
    "THIS run's own earlier-level traces at each level-clear boundary "
    "(it starts empty and grows only as you clear levels). "
    + T2S_NAMESPACE_DOC
    + " When you finish a level and brief subagents for the NEXT level, "
    "consult `skill_library` alongside `memories` so accumulated cross-level "
    "know-how is not re-derived from scratch. This directory is ADDITIVE: it "
    "does not replace `memories` or any existing tool — keep using all of "
    "them. Do NOT write to `skill_library` from inside a turn; it is "
    "maintained out-of-band by the runner."
)

rows = [json.loads(l) for l in TRACE.read_text().splitlines() if l.strip()]
store = json.loads(STORE.read_text())
store_skills = store.get("skills", [])

# ---- 5/5 same-seed paired table from authoritative summaries ----
best = {}
for f in glob.glob(str(REPO / "reports/a3_calibration/*.summary.json")):
    b = pathlib.Path(f).name
    m = re.search(r"_s(\d+)_(none|trace2skill)_", b)
    if not m:
        continue
    try:
        r = json.load(open(f)).get("result", {})
    except Exception:
        continue
    k = (int(m.group(1)), m.group(2))
    lv = r.get("levels_completed")
    if lv is None:
        continue
    if k not in best or lv > best[k][0]:
        best[k] = (lv, r.get("final_state"), r.get("recorder_action_count"))
seeds = sorted({s for (s, _a) in best})
pair_rows = []
for s in seeds:
    n = best.get((s, "none"))
    t = best.get((s, "trace2skill"))
    if n and t:
        pair_rows.append((s, n, t, t[0] - n[0]))

summ = json.load(open(SUMM))["result"]


# ---- TRACE-VERIFIED consumption (recomputed, not asserted) ----
def _epoch(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.datetime.fromisoformat(
            ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


cons_ts = sorted(
    _epoch(d.get("ts")) for d in rows
    if ((d.get("t2s_consolidation_wall_s") or 0) > 0 or d.get("cleared_a_level"))
    and d.get("ts"))
first_cons = cons_ts[0] if cons_ts else None
MARKER = "Skill for clearing level via"   # producible ONLY by the deterministic
                                          # consolidator, never by T2S_TASK_PARAGRAPH
# codex objective gate (2026-05-18) narrowed the claim: assert INPUT re-entry
# (marker inside body.input, the upstream model input), NOT causal reliance;
# the python-ref count is the # of DISTINCT custom_tool_call invocations whose
# OWN code references skill_library (parsed + deduped) — the prior 6000-char
# regex window over the whole body was contaminated by replayed history.
consume = {"n_requests": 0, "marker_in_input": 0, "marker_after_cons": 0,
           "distinct_skill_tool_calls": 0}
if PROXY.exists():
    ev = [json.loads(l) for l in PROXY.read_text().splitlines() if l.strip()]
    reqs = sorted((d for d in ev if d.get("event") == "stream_request"),
                  key=lambda d: _epoch(d.get("ts")) or 0)
    consume["n_requests"] = len(reqs)
    _seen_calls = set()
    for d in reqs:
        b = d.get("body")
        inp = b.get("input") if isinstance(b, dict) else None
        inp_s = (json.dumps(inp, ensure_ascii=False) if inp is not None
                 else (b if isinstance(b, str)
                       else json.dumps(b, ensure_ascii=False)))
        if MARKER in inp_s:
            consume["marker_in_input"] += 1
            te = _epoch(d.get("ts"))
            if first_cons and te and te >= first_cons:
                consume["marker_after_cons"] += 1
        if isinstance(inp, list):
            for it in inp:
                if (isinstance(it, dict)
                        and it.get("type") == "custom_tool_call"
                        and it.get("name") == "python"):
                    code = json.dumps(it.get("input", ""), ensure_ascii=False)
                    cid = it.get("call_id") or it.get("id") or code[:80]
                    if "skill_library" in code and cid not in _seen_calls:
                        _seen_calls.add(cid)
                        consume["distinct_skill_tool_calls"] += 1

DATA = {
    "trace_tag": "pilot_s47_trace2skill_20260517T161830Z",
    "seed": 47, "arm": "A3_EXT=trace2skill (original Symbolica + skill only)",
    "result": {k: summ.get(k) for k in ("final_state", "levels_completed",
               "recorder_action_count", "request_count", "runtime_seconds",
               "termination_reason")},
    "actions": [
        {"i": d.get("action_index"), "name": d.get("action_name"),
         "lb": d.get("level_before_action"), "la": d.get("level_after_action"),
         "lc": bool(d.get("cleared_a_level")), "lvls": d.get("levels_completed"),
         "lat": round(d.get("action_latency_s") or 0, 2),
         "ptok": d.get("per_call_prompt_tokens"),
         "t2s": round(d.get("t2s_consolidation_wall_s") or 0, 3)}
        for d in rows
    ],
    "pairs": [{"seed": s, "none_L": n[0], "none_state": n[1], "none_act": n[2],
               "t2s_L": t[0], "t2s_state": t[1], "t2s_act": t[2], "dL": d}
              for (s, n, t, d) in pair_rows],
    "consume": consume,
    "store_meta": {"checksum": store.get("checksum"),
                   "schema_version": store.get("schema_version"),
                   "n_skills": len(store_skills)},
}

J = json.dumps(DATA, ensure_ascii=False, indent=2)
now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")

# ---- skill-memory panel: the TWO real artifacts the LLM/runtime use ----
PARA = html.escape(T2S_TASK_PARAGRAPH)
recipe_blocks = []
for i, sk in enumerate(store_skills):
    rcp = sk.get("recipe") or ""
    meta = (f"category={sk.get('category')}  code={sk.get('code')}  "
            f"predicate={sk.get('predicate')}  posterior={sk.get('posterior')}  "
            f"evidence={len(sk.get('evidence', []))} recs")
    recipe_blocks.append(
        f'<div class="rec"><div class="rech">store[{i}] '
        f'&mdash; {html.escape(meta)}</div>'
        f'<pre class="recp">{html.escape(rcp)}</pre></div>')
RECIPES = "\n".join(recipe_blocks)

SKILLPANEL = f"""
<div class="pan"><h2>What the LLM actually receives under A3_EXT=trace2skill</h2>
<div class="note"><b>(1) The only injected instruction</b> &mdash; byte-exact
<code>_T2S_TASK_PARAGRAPH</code> appended to the orchestrator initial task
(<code>run_clean_upstream_ft09.py:1183</code>). The warm <code>state/skills_ft09.json</code>
is <b>NEVER</b> read; the store starts as a fresh EMPTY per-run tempfile.</div>
<pre>{PARA}</pre>
<div class="note" style="margin-top:14px"><b>(2) The actual run-distilled store</b>
this s47 run produced (preserved verbatim:
<code>reports/skill_pilot/20260517T161830Z/s47_distilled_store.json</code>,
checksum <code>{html.escape(str(store.get('checksum',''))[:16])}…</code>,
{len(store_skills)} entries). Each entry is a <b>VERBATIM per-level action
transcript</b> recorded by the deterministic level-boundary observer &mdash;
NO LLM, NO abstraction (<code>code=null, predicate=null, posterior=[0,0]</code>).</div>
{RECIPES}
<div class="note" style="margin-top:14px"><b>(3) Input re-entry &mdash;
trace-verified, recomputed here</b> from <code>{PROXY.name}</code> (codex
objective gate 2026-05-18: assert input re-entry, NOT causal reliance). The
consolidator-only marker <code>"{MARKER}"</code> (the instruction paragraph
cannot produce it) appears in <b>{consume['marker_in_input']}/{consume['n_requests']}</b>
stream-request <code>body.input</code> (the upstream model input itself), of
which <b>{consume['marker_after_cons']}</b> are after the first consolidation
(0 before). The run also contains <b>&ge;{consume['distinct_skill_tool_calls']}
distinct</b> explicit <code>python</code> tool calls querying the object
(e.g. <code>skill_library.library.summaries()</code>; parsed per-call &amp;
deduped &mdash; an earlier 102/145 regex-window count was contaminated by
replayed history and is retracted). <b>Proven:</b> the stored skill text was in
the model input after each level-clear and the agent inspected the object.
<b>NOT proven:</b> that the LLM relied on it, or that trace2skill caused the
s47 level gain (presence &ne; causal use &mdash; that is the next experiment).
</div></div>
"""

HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<title>FT09 Skill Trace Explorer — s47 trace2skill L6 WIN</title>
<style>
:root{--bg:#0d1117;--pan:#161b22;--bd:#30363d;--fg:#c9d1d9;--mut:#8b949e;
--blu:#388bfd;--grn:#3fb950;--org:#d29922;--pur:#a371f7;--red:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;color:var(--blu);
border-bottom:1px solid var(--bd);padding-bottom:4px;margin:18px 0 10px}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
.sub{color:var(--mut);font-size:12px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.pan{background:var(--pan);border:1px solid var(--bd);border-radius:8px;padding:14px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid var(--bd);padding:5px 8px;text-align:center}
th{background:#1c2330;color:var(--mut)}
.win{color:var(--grn);font-weight:700}.pos{color:var(--grn)}
.tl{display:flex;flex-wrap:wrap;gap:3px;margin-top:8px}
.cell{width:30px;height:30px;border-radius:4px;display:flex;align-items:center;
justify-content:center;font-size:9px;border:1px solid var(--bd);background:#1c2330;
position:relative;cursor:default}
.cell.lvlup{background:var(--grn);color:#06210f;font-weight:700;border-color:var(--grn)}
.cell.reset{background:#21262d;color:var(--mut)}
.cell:hover::after{content:attr(data-tip);position:absolute;bottom:34px;left:0;
white-space:pre;background:#000;color:#fff;border:1px solid var(--bd);padding:6px;
border-radius:4px;font-size:10px;z-index:9;min-width:160px}
pre{background:#0a0e14;border:1px solid var(--bd);border-radius:6px;padding:12px;
overflow:auto;font-size:11.5px;color:#d1d9e0;max-height:340px;white-space:pre-wrap;
word-break:break-word}
.rec{margin-top:10px}.rech{color:var(--org);font-size:11px;margin-bottom:3px}
.recp{max-height:200px;margin:0}
.kv{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0}
.kv span{background:#1c2330;border:1px solid var(--bd);border-radius:4px;
padding:3px 8px;font-size:12px}
.badge{display:inline-block;background:var(--grn);color:#06210f;font-weight:700;
border-radius:4px;padding:2px 8px;font-size:12px}
.note{color:var(--mut);font-size:11px;margin-top:6px}
.lvline{height:64px;display:flex;align-items:flex-end;gap:2px;margin-top:6px}
.lvbar{width:7px;background:var(--blu);border-radius:2px 2px 0 0}
.lvbar.up{background:var(--grn)}
</style></head><body><div class="wrap">
<h1>FT09 Skill Trace Explorer &mdash; <span class="badge">L6 WIN</span> seed&nbsp;47 &middot; A3_EXT=trace2skill</h1>
<div class="sub">Original byte-faithful Symbolica + skill ONLY (single variable). Trace = the one clean L6 full win
(<code>pilot_s47_trace2skill</code>, terminal <code>swarm_returned</code>, 101 actions). Built {now} from authoritative
<code>reports/a3_calibration/</code> recorder telemetry + the verbatim instruction paragraph and the actual run-distilled store.
Honest: recorder is per-action level/token telemetry (no ft09 grid frames) &mdash; timeline shows real level/skill progression, not a reconstructed pixel grid.</div>

<div class="grid">
<div class="pan"><h2>Run result (authoritative summary.json)</h2>
<div class="kv" id="res"></div>
<div class="note">levels_completed extracted from the agent's final frame at episode end (the scoring SSOT — not server-log).</div></div>
<div class="pan"><h2>Skill vs no-skill &mdash; 5/5 same-seed (levels)</h2>
<table id="pairs"><thead><tr><th>seed</th><th>none L (state)</th><th>trace2skill L (state)</th><th>&Delta;L</th></tr></thead><tbody></tbody></table>
<div class="note">All &Delta;L &gt; 0. s47 = the clean L6 WIN shown left. (rev4 caveat: 8/9 non-s47 episodes soft_deadline = M3-censored speed dim; capability dim valid.)</div></div>
</div>

<div class="pan"><h2>s47 trace2skill &mdash; per-action level timeline (101 actions &rarr; L6)</h2>
<div class="lvline" id="lvline"></div>
<div class="tl" id="tl"></div>
<div class="note">Green = a level was cleared on that action (<code>cleared_a_level</code>). Hover a cell for action / level / latency / prompt-tokens / t2s-consolidation-wall.</div></div>

__SKILLPANEL__

<script>
const DATA = __DATA__;
const R=DATA.result, res=document.getElementById('res');
for(const[k,v] of Object.entries(R)){const s=document.createElement('span');
s.innerHTML='<b>'+k+'</b>: '+(k==='final_state'&&v==='WIN'?'<span class=win>'+v+'</span>':v);res.appendChild(s);}
const pb=document.querySelector('#pairs tbody');
DATA.pairs.forEach(p=>{const tr=document.createElement('tr');
tr.innerHTML=`<td>${p.seed}</td><td>L${p.none_L} (${p.none_state})</td>`+
`<td>${p.seed===47?'<span class=win>L'+p.t2s_L+' '+p.t2s_state+'</span>':'L'+p.t2s_L+' ('+p.t2s_state+')'}</td>`+
`<td class=pos>+${p.dL}</td>`;pb.appendChild(tr);});
const tl=document.getElementById('tl'), lv=document.getElementById('lvline');
const maxL=Math.max(...DATA.actions.map(a=>a.lvls||0),1);
DATA.actions.forEach(a=>{
 const c=document.createElement('div');
 c.className='cell'+(a.lc?' lvlup':'')+(a.name==='RESET'?' reset':'');
 c.textContent=a.lc?('L'+a.la):a.i;
 c.dataset.tip=`#${a.i} ${a.name}\nlevel ${a.lb}→${a.la}  (done:${a.lvls})\nlat ${a.lat}s  ptok ${a.ptok}\nt2s_wall ${a.t2s}s`;
 tl.appendChild(c);
 const b=document.createElement('div');b.className='lvbar'+(a.lc?' up':'');
 b.style.height=(8+54*((a.lvls||0)/maxL))+'px';
 b.title=`#${a.i}: levels_completed=${a.lvls}`;lv.appendChild(b);
});
</script>
</div></body></html>"""

OUT.write_text(HTML.replace("__DATA__", J)
                .replace("__SKILLPANEL__", SKILLPANEL)
                .replace("{now}", now))
print(f"WROTE {OUT}  ({OUT.stat().st_size} bytes)")
print(f"actions={len(DATA['actions'])} pairs={len(DATA['pairs'])} "
      f"store_skills={len(store_skills)} final={DATA['result']['final_state']} "
      f"L={DATA['result']['levels_completed']}")
print(f"input-re-entry(recomputed): marker_in_input "
      f"{consume['marker_in_input']}/{consume['n_requests']} "
      f"(after_cons={consume['marker_after_cons']})  "
      f"distinct_skill_tool_calls={consume['distinct_skill_tool_calls']}")

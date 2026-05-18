#!/usr/bin/env python3
"""Polished multi-panel grid trace-explorer from a REAL arcgentica
recording.jsonl (ft09_trace_explorer.html-style layout, honest content).

The byte-faithful agentica recorder dumps, per turn, the actual 64x64
ft09 grid the agent saw (recordings/<game>.arcgentica.<guid>.recording
.jsonl). This renders it as a polished explorer: run-overview hero,
grid triptych (prev -> THIS action -> result with changed-cell
highlight), level-coloured timeline strip, facts panel, palette legend,
playback bar.

HONEST by construction: the byte-faithful agent is a SINGLE orchestrator
+ dynamic subagents (NO M1/M2/M3/M4 modules) and its per-turn
chain-of-thought is PROVIDER-ENCRYPTED in the proxy capture. So there is
NO per-module / reasoning panel to fill -- that panel states this
explicitly instead of fabricating module rationales (which would
misrepresent the agent). Action names are best-effort ordinal pairing
with the run's a3_calibration recorder.

Usage:
  build_ft09_recording_grid_explorer.py RECORDING.jsonl OUT.html [RECORDER.jsonl] [TITLE]
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
    "#F012BE", "#FF851B", "#7FDBFF", "#870C25", "#555555", "#B10DC9",
    "#001F3F", "#7D4E4E", "#4E7D4E", "#4E4E7D",
]


def _grid(frame):
    if not isinstance(frame, list):
        return None
    for g in reversed(frame):
        if isinstance(g, list) and g and isinstance(g[0], list):
            return [[int(c) for c in row] for row in g]
    if frame and isinstance(frame[0], list):
        return [[int(c) for c in row] for row in frame]
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    rec_p, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    jargs = [a for a in sys.argv[3:] if a.endswith(".jsonl")]
    proxy_p = next((Path(a) for a in jargs if ".proxy.jsonl" in a), None)
    recorder_p = next(
        (Path(a) for a in jargs if ".proxy.jsonl" not in a), None
    )
    title = (
        sys.argv[-1]
        if len(sys.argv) > 3 and not sys.argv[-1].endswith(".jsonl")
        else "ft09 real recording — step-by-step grids"
    )

    lines = [json.loads(x) for x in rec_p.read_text().splitlines() if x.strip()]
    actions = []
    if recorder_p and recorder_p.exists():
        rr = [
            json.loads(x)
            for x in recorder_p.read_text().splitlines()
            if x.strip()
        ]
        actions = [str(r.get("action_name")) for r in rr]

    # --- recover the agent's OWN natural-language reasoning: deduped
    #     memories.add(summary, details) from the proxy tool-call code
    #     (NOT encrypted; this is what the subagents actually concluded) ---
    import re as _re
    mems: list[tuple[str, str]] = []
    if proxy_p and proxy_p.exists():
        _seen: set[str] = set()
        _pat = _re.compile(
            r"memories\.add\(\s*(['\"])(.+?)\1\s*,\s*(['\"])(.+?)\3\s*\)",
            _re.S,
        )
        for _ln in proxy_p.read_text().splitlines():
            if not _ln.strip() or "memories.add" not in _ln:
                continue
            try:
                _b = json.loads(_ln).get("body")
            except Exception:
                continue
            if not isinstance(_b, dict):
                continue
            for _it in (_b.get("input") or []):
                if (
                    isinstance(_it, dict)
                    and _it.get("type") == "custom_tool_call"
                ):
                    _c = _it.get("input")
                    _c = _c if isinstance(_c, str) else json.dumps(_c)
                    for _m in _pat.finditer(_c):
                        s, d = _m.group(2).strip(), _m.group(4).strip()
                        k = s[:60] + d[:80]
                        if k in _seen:
                            continue
                        _seen.add(k)
                        mems.append((s, d))

    steps = []
    prev = None
    for i, ln in enumerate(lines):
        d = ln.get("data", {})
        g = _grid(d.get("frame"))
        lv = d.get("levels_completed")
        lv = int(lv) if lv is not None else (prev if prev is not None else 0)
        act = actions[i] if i < len(actions) else "(action n/a)"
        ai = d.get("action_input") or {}
        steps.append(
            {
                "n": i,
                "action": act,
                "level": lv,
                "state": str(d.get("state", "")),
                "grid": g,
                "levelup": prev is not None and lv > prev,
                "ts": ln.get("timestamp", ""),
                # --- recording's own fields, faithfully surfaced ---
                "ai_id": ai.get("id"),
                "ai_data": json.dumps(ai.get("data", {}), separators=(",", ":")),
                "ai_reasoning": ai.get("reasoning"),
                "avail": d.get("available_actions"),
                "win": d.get("win_levels"),
                "freset": d.get("full_reset"),
                "guid": str(d.get("guid", ""))[:8],
            }
        )
        prev = lv

    final_level = steps[-1]["level"] if steps else 0
    final_state = steps[-1]["state"] if steps else ""
    n_lu = sum(1 for s in steps if s["levelup"])
    data = json.dumps({"steps": steps, "palette": PALETTE}, separators=(",", ":"))
    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{k}</span>'
        for k, c in enumerate(PALETTE)
    )
    if mems:
        mem_html = (
            '<div class="na" style="margin-bottom:8px">The agent\'s '
            "<b>own natural-language reasoning</b>, recovered verbatim from "
            "its <code>memories.add(summary, details)</code> writes in the "
            "(un-encrypted) tool-call code &mdash; what each subagent "
            "actually concluded. (Raw model chain-of-thought stays "
            "provider-encrypted; this is the readable functional "
            "reasoning.)</div>"
            + "".join(
                '<div class="memo"><div class="memh">'
                f"{html.escape(s)}</div><div class=\"memd\">"
                f"{html.escape(d)}</div></div>"
                for s, d in mems
            )
        )
    else:
        mem_html = (
            '<div class="na"><b>No proxy supplied / no memories writes '
            "found.</b> Raw model chain-of-thought is provider-encrypted; "
            "pass the run’s <code>*.proxy.jsonl</code> to surface the "
            "agent’s own <code>memories.add</code> reasoning.</div>"
        )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
:root{{--bg:#0d1117;--pan:#161b22;--bd:#30363d;--fg:#c9d1d9;--mut:#8b949e;
--blu:#388bfd;--grn:#3fb950;--org:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}}
.w{{max-width:1200px;margin:0 auto;padding:18px}}
h1{{font-size:17px;margin:0 0 2px}}.sub{{color:var(--mut);font-size:12px}}
.hero{{background:var(--pan);border:1px solid var(--bd);border-radius:8px;
padding:12px 14px;margin:10px 0;display:flex;gap:22px;flex-wrap:wrap}}
.hero b{{color:var(--blu)}}.hero .k{{color:var(--mut)}}
.ban{{background:#1c1606;border:1px solid var(--org);border-radius:6px;
padding:9px 12px;color:#e3b341;font-size:11.5px;margin:8px 0}}
.grid2{{display:grid;grid-template-columns:2fr 1fr;gap:14px}}
.card{{background:var(--pan);border:1px solid var(--bd);border-radius:8px;
padding:12px}}.card h2{{font-size:12px;color:var(--blu);margin:0 0 8px;
border-bottom:1px solid var(--bd);padding-bottom:5px}}
.trip{{display:flex;gap:12px;align-items:flex-end;justify-content:center}}
.trip figure{{margin:0;text-align:center}}.trip figcaption{{color:var(--mut);
font-size:11px;margin-top:4px}}
canvas{{image-rendering:pixelated;border:1px solid var(--bd);background:#000}}
#cur{{width:384px;height:384px}}.sm{{width:150px;height:150px}}
.bar{{display:flex;gap:8px;align-items:center;margin:12px 0}}
button{{background:#21262d;color:var(--fg);border:1px solid var(--bd);
border-radius:5px;padding:6px 12px;cursor:pointer;font:inherit}}
button:hover{{background:#30363d}}#sl{{flex:1}}
#tt{{font-size:14px;color:var(--grn);font-weight:700;margin:6px 0}}
.tl{{display:flex;flex-wrap:wrap;gap:2px;margin-top:6px}}
.cell{{width:13px;height:13px;border-radius:2px;border:1px solid #0008;
cursor:pointer}}
.fct{{display:flex;justify-content:space-between;border-bottom:1px solid
var(--bd);padding:4px 0;font-size:12px}}.fct b{{color:var(--fg)}}
.fct span{{color:var(--mut)}}
.lg{{display:inline-flex;align-items:center;gap:3px;margin:2px 6px 2px 0;
font-size:10px;color:var(--mut)}}.lg i{{width:11px;height:11px;
display:inline-block;border:1px solid #0006}}
.na{{color:var(--mut);font-size:11.5px;line-height:1.55}}
.memwrap{{max-height:420px;overflow:auto}}
.memo{{border:1px solid var(--bd);border-left:3px solid var(--grn);
border-radius:5px;padding:7px 9px;margin:7px 0;background:#0a0e14}}
.memh{{color:var(--grn);font-weight:700;font-size:11.5px;margin-bottom:3px}}
.memd{{color:#9db4d0;font-size:11px;line-height:1.5;white-space:pre-wrap}}
.lv{{color:var(--org);font-weight:700}}.win{{color:var(--grn);font-weight:700}}
</style></head><body><div class="w">
<h1>{html.escape(title)}</h1>
<div class="sub">Real recorded ft09 frames &mdash; the actual grid the
byte-faithful agent saw each turn (<code>{html.escape(rec_p.name)}</code>).
No replay, no synthetic data.</div>
<div class="hero" id="hero"></div>
<div class="ban"><b>Why no per-module / reasoning panel:</b> this is the
byte-faithful original Symbolica agent &mdash; a SINGLE orchestrator with
dynamically-spawned subagents, with <b>no M1/M2/M3/M4 modules</b>, and its
per-turn chain-of-thought is <b>provider-encrypted</b> in the proxy
capture. There is therefore no per-module response to show; fabricating
one would misrepresent the agent. This explorer shows what IS real: the
exact grids, actions, level transitions, and changed cells.</div>
<div class="bar">
<button id="pv">&#9664; Prev</button>
<button id="pl">&#9654; Play</button>
<button id="nx">Next &#9654;</button>
<input id="sl" type="range" min="0" value="0"/><span id="ix"></span></div>
<div id="tt"></div>
<div class="grid2">
 <div class="card"><h2>Grid triptych &mdash; prev &rarr; this turn &rarr;
 result (changed cells ringed magenta)</h2>
 <div class="trip">
  <figure><canvas id="pre" class="sm" width="256" height="256"></canvas>
   <figcaption>previous turn</figcaption></figure>
  <figure><canvas id="cur" width="512" height="512"></canvas>
   <figcaption id="curcap">this turn</figcaption></figure>
  <figure><canvas id="nxt" class="sm" width="256" height="256"></canvas>
   <figcaption>next turn</figcaption></figure>
 </div>
 <div style="margin-top:10px">{legend}</div>
 <h2 style="margin-top:14px">Timeline (colour = level; click to jump)</h2>
 <div class="tl" id="tl"></div></div>
 <div>
  <div class="card"><h2>This turn</h2><div id="facts"></div></div>
  <div class="card" style="margin-top:14px">
   <h2>Subagent reasoning (agent's own memories writes)</h2>
   <div class="memwrap">{mem_html}</div></div>
 </div>
</div>
<script id="d" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('d').textContent);
const S=D.steps,P=D.palette;let i=0,timer=null;
const $=id=>document.getElementById(id);
$('sl').max=S.length-1;
const lvColor=l=>['#30363d','#0e3b66','#7a2a16','#1d5a2b','#7a6a12',
 '#444','#5a1450'][l]||'#3fb950';
$('hero').innerHTML=
 `<div><span class=k>turns</span> <b>${{S.length-1}}</b></div>`+
 `<div><span class=k>level-ups</span> <b>{n_lu}</b></div>`+
 `<div><span class=k>final</span> <b class=win>L{final_level} `+
 `{html.escape(final_state)}</b></div>`+
 `<div><span class=k>source</span> <b>recorded frames</b></div>`;
function paint(cvid,g,ring){{const cv=$(cvid),cx=cv.getContext('2d');
 if(!g){{cx.clearRect(0,0,cv.width,cv.height);return;}}
 const H=g.length,W=g[0].length,px=Math.floor(cv.width/Math.max(H,W));
 cx.clearRect(0,0,cv.width,cv.height);
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){{
  cx.fillStyle=P[((g[y][x]%16)+16)%16]||'#000';
  cx.fillRect(x*px,y*px,px,px);}}
 if(ring){{cx.strokeStyle='#ff00ff';cx.lineWidth=1;
  for(const[y,x]of ring){{cx.strokeRect(x*px,y*px,px,px);}}}}}}
function diff(a,b){{const r=[];if(!a||!b)return r;
 for(let y=0;y<b.length;y++)for(let x=0;x<b[0].length;x++)
  if(!a[y]||a[y][x]!==b[y][x])r.push([y,x]);return r;}}
function build_tl(){{const tl=$('tl');tl.innerHTML='';
 S.forEach((s,k)=>{{const c=document.createElement('div');
  c.className='cell';c.style.background=s.levelup?'#3fb950':lvColor(s.level);
  c.title=`#${{s.n}} ${{s.action}} L${{s.level}}`;
  c.onclick=()=>go(k);tl.appendChild(c);}});}}
function draw(){{const s=S[i],pg=i>0?S[i-1].grid:null,
 ng=i<S.length-1?S[i+1].grid:null,dd=diff(pg,s.grid);
 paint('cur',s.grid,dd);paint('pre',pg,null);paint('nxt',ng,null);
 const lu=s.levelup?` <span class=lv>&#8593; LEVEL UP</span>`:'';
 const wn=(s.state||'').includes('WIN')?` <span class=win>WIN</span>`:'';
 $('tt').innerHTML=`Turn ${{s.n}} / ${{S.length-1}} &mdash; `+
  `<b>${{s.action}}</b> &rarr; L${{s.level}}${{lu}}${{wn}}`;
 $('curcap').textContent=`this turn (#${{s.n}}) — ${{dd.length}} cells changed`;
 $('ix').textContent=`${{i}}/${{S.length-1}}`;$('sl').value=i;
 const rj=v=>v===undefined||v===null?'null':(typeof v==='object'?
  JSON.stringify(v):String(v));
 $('facts').innerHTML=
  `<div class=fct><span>turn</span><b>${{s.n}}</b></div>`+
  `<div class=fct><span>action (paired)</span><b>${{s.action}}</b></div>`+
  `<div class=fct><span>levels_completed</span><b>${{s.level}}</b></div>`+
  `<div class=fct><span>win_levels</span><b>${{rj(s.win)}}</b></div>`+
  `<div class=fct><span>state</span><b>${{s.state}}</b></div>`+
  `<div class=fct><span>cells changed</span><b>${{dd.length}}</b></div>`+
  `<div class=fct><span>action_input.id</span><b>${{rj(s.ai_id)}}</b></div>`+
  `<div class=fct><span>action_input.data</span><b>${{rj(s.ai_data)}}</b></div>`+
  `<div class=fct><span>action_input.reasoning</span><b>${{rj(s.ai_reasoning)}}</b></div>`+
  `<div class=fct><span>available_actions</span><b>${{rj(s.avail)}}</b></div>`+
  `<div class=fct><span>full_reset</span><b>${{rj(s.freset)}}</b></div>`+
  `<div class=fct><span>guid</span><b>${{s.guid}}</b></div>`+
  `<div class=fct><span>ts</span><b>${{s.ts}}</b></div>`;
}}
function go(j){{i=Math.max(0,Math.min(S.length-1,j));draw();}}
$('nx').onclick=()=>go(i+1);$('pv').onclick=()=>go(i-1);
$('sl').oninput=e=>go(+e.target.value);
$('pl').onclick=function(){{
 if(timer){{clearInterval(timer);timer=null;this.innerHTML='&#9654; Play';}}
 else{{this.innerHTML='&#10073;&#10073; Pause';
  timer=setInterval(()=>{{if(i>=S.length-1){{clearInterval(timer);
   timer=null;$('pl').innerHTML='&#9654; Play';}}else go(i+1);}},240);}}}};
build_tl();draw();
</script></div></body></html>"""
    out_p.write_text(doc)
    print(
        f"WROTE {out_p} ({out_p.stat().st_size}B) turns={len(steps)} "
        f"levelups={n_lu} final=L{final_level} {final_state}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

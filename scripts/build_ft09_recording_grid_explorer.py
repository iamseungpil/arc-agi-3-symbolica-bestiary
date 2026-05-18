#!/usr/bin/env python3
"""Grid step-by-step explorer from a REAL arcgentica recording.jsonl.

The byte-faithful agentica recorder dumps, per turn, the actual 64x64
ft09 grid the agent saw (recordings/<game>.arcgentica.<guid>.recording.jsonl,
each line {timestamp, data:{frame, levels_completed, state, ...}}). This
is the ground-truth visual trace — no replay, no coord mining, no
fabrication. Action NAMES are attached by ordinal pairing with the
matching a3_calibration recorder jsonl (same run window, also 101 rows);
labelled as best-effort ordinal alignment.

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
    recorder_p = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].endswith(".jsonl") else None
    title = sys.argv[-1] if len(sys.argv) > 3 and not sys.argv[-1].endswith(".jsonl") else "ft09 real recording — step-by-step grids"

    lines = [json.loads(x) for x in rec_p.read_text().splitlines() if x.strip()]
    actions = []
    if recorder_p and recorder_p.exists():
        rr = [json.loads(x) for x in recorder_p.read_text().splitlines() if x.strip()]
        actions = [str(r.get("action_name")) for r in rr]

    steps = []
    prev = None
    for i, ln in enumerate(lines):
        d = ln.get("data", {})
        g = _grid(d.get("frame"))
        lv = d.get("levels_completed")
        lv = int(lv) if lv is not None else (prev if prev is not None else 0)
        act = actions[i] if i < len(actions) else "(action n/a)"
        steps.append(
            {
                "n": i,
                "action": act,
                "level": lv,
                "state": str(d.get("state", "")),
                "grid": g,
                "levelup": prev is not None and lv > prev,
                "ts": ln.get("timestamp", ""),
            }
        )
        prev = lv

    final_level = steps[-1]["level"] if steps else 0
    final_state = steps[-1]["state"] if steps else ""
    n_lu = sum(1 for s in steps if s["levelup"])
    data = json.dumps({"steps": steps, "palette": PALETTE}, separators=(",", ":"))

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{background:#0d1117;color:#c9d1d9;font:13px ui-monospace,Menlo,monospace;margin:0}}
.w{{max-width:1000px;margin:0 auto;padding:18px}}h1{{font-size:17px;margin:0 0 4px}}
.ban{{background:#161b22;border:1px solid #2d7d46;border-radius:6px;padding:9px;
color:#7fdbb6;font-size:12px;margin:10px 0}}.bar{{display:flex;gap:8px;
align-items:center;margin:10px 0}}button{{background:#21262d;color:#c9d1d9;
border:1px solid #30363d;border-radius:5px;padding:6px 12px;cursor:pointer;
font:inherit}}button:hover{{background:#30363d}}#tt{{font-size:14px;
color:#3fb950;font-weight:700}}#cv{{image-rendering:pixelated;border:1px solid
#30363d;background:#000;display:block;margin:8px 0}}#sl{{flex:1}}
.lv{{color:#d29922}}.win{{color:#3fb950;font-weight:700}}
</style></head><body><div class="w">
<h1>{html.escape(title)}</h1>
<div class="ban">REAL recorded ft09 frames (byte-faithful agentica recorder,
<code>{html.escape(rec_p.name)}</code>) &mdash; the actual 64&times;64 grid the
agent saw each turn. No replay, no synthetic data. {len(steps)} turns &middot;
{n_lu} level-ups &middot; final <b>L{final_level} {html.escape(final_state)}</b>.
Action names are best-effort ordinal pairing with the run's a3_calibration
recorder.</div>
<div class="bar"><button id="pv">&#9664; Prev</button>
<button id="pl">&#9654; Play</button><button id="nx">Next &#9654;</button>
<input id="sl" type="range" min="0" value="0"/><span id="ix"></span></div>
<div id="tt"></div><canvas id="cv" width="512" height="512"></canvas>
<div id="meta"></div>
<script id="d" type="application/json">{data}</script><script>
const D=JSON.parse(document.getElementById('d').textContent);
const S=D.steps,P=D.palette,cv=document.getElementById('cv'),
cx=cv.getContext('2d');let i=0,timer=null;
document.getElementById('sl').max=S.length-1;
function draw(){{const s=S[i],g=s.grid;
 if(!g){{cx.clearRect(0,0,512,512);}}else{{
 const H=g.length,W=g[0].length,px=Math.floor(512/Math.max(H,W));
 cx.clearRect(0,0,512,512);
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){{
  cx.fillStyle=P[((g[y][x]%16)+16)%16]||'#000';
  cx.fillRect(x*px,y*px,px,px);}}}}
 const lu=s.levelup?` <span class=lv>&#8593; LEVEL UP</span>`:'';
 const wn=(s.state||'').includes('WIN')?` <span class=win>WIN</span>`:'';
 document.getElementById('tt').innerHTML=
  `Turn ${{s.n}} / ${{S.length-1}} &mdash; <b>${{s.action}}</b> &rarr; `+
  `L${{s.level}}${{lu}}${{wn}}`;
 document.getElementById('ix').textContent=`${{i}}/${{S.length-1}}`;
 document.getElementById('sl').value=i;
 document.getElementById('meta').textContent=`state=${{s.state}} ts=${{s.ts}}`;}}
function go(j){{i=Math.max(0,Math.min(S.length-1,j));draw();}}
document.getElementById('nx').onclick=()=>go(i+1);
document.getElementById('pv').onclick=()=>go(i-1);
document.getElementById('sl').oninput=e=>go(+e.target.value);
document.getElementById('pl').onclick=function(){{
 if(timer){{clearInterval(timer);timer=null;this.innerHTML='&#9654; Play';}}
 else{{this.innerHTML='&#10073;&#10073; Pause';
  timer=setInterval(()=>{{if(i>=S.length-1){{clearInterval(timer);timer=null;
   document.getElementById('pl').innerHTML='&#9654; Play';}}else go(i+1);}},
   240);}}}};
draw();
</script></div></body></html>"""
    out_p.write_text(doc)
    print(
        f"WROTE {out_p} ({out_p.stat().st_size}B) turns={len(steps)} "
        f"levelups={n_lu} final=L{final_level} {final_state}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

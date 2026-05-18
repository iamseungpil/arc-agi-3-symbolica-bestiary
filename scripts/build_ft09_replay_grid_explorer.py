#!/usr/bin/env python3
"""Deterministic ft09 REPLAY -> grid step-by-step explorer.

ft09 is deterministic: given the exact ordered (action, x, y) sequence the
agent submitted (recovered from actions_log/<game>/level_*.jsonl, sliced by
the run's ts window), replaying it on the real ft09 env regenerates EXACTLY
the grids the agent saw. This is NOT a fabrication — it is the real game
state reconstructed by re-executing the real recorded action sequence.

Usage:
  build_ft09_replay_grid_explorer.py SEQ.json OUT.html [TITLE]
    SEQ.json = ordered list of {action,x?,y?,level,count,state,ts}
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


def _grid(raw):
    frame = getattr(raw, "frame", None)
    layers = list(frame) if frame is not None else []
    for g in reversed(layers):
        gl = g.tolist() if hasattr(g, "tolist") else list(g)
        if gl and isinstance(gl[0], (list, tuple)) and len(gl[0]) > 0:
            return [[int(c) for c in row] for row in gl]
    return None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    seq_p, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else "ft09 deterministic replay"
    seq = json.loads(seq_p.read_text())

    from arc_agi import Arcade, OperationMode  # noqa: E402
    from arcengine import GameAction  # noqa: E402

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ft09-9ab2447a")
    raw = env.reset()
    steps = []
    g0 = _grid(raw)
    steps.append(
        {
            "n": 0, "action": "RESET(initial)", "x": None, "y": None,
            "level": int(getattr(raw, "levels_completed", 0) or 0),
            "state": str(getattr(raw, "state", "")), "grid": g0,
        }
    )
    prev_level = steps[0]["level"]
    for i, d in enumerate(seq, start=1):
        name = str(d.get("action"))
        ga = getattr(GameAction, name, None)
        if ga is None:
            continue
        x, y = d.get("x"), d.get("y")
        try:
            if x is not None and y is not None:
                raw = env.step(ga, data={"x": int(x), "y": int(y)})
            else:
                raw = env.step(ga)
        except Exception as exc:  # noqa: BLE001
            steps.append(
                {
                    "n": i, "action": name, "x": x, "y": y,
                    "level": prev_level, "state": f"REPLAY_ERR:{exc}",
                    "grid": steps[-1]["grid"], "err": True,
                }
            )
            continue
        g = _grid(raw) or steps[-1]["grid"]
        lvl = int(getattr(raw, "levels_completed", prev_level) or prev_level)
        steps.append(
            {
                "n": i, "action": name, "x": x, "y": y, "level": lvl,
                "state": str(getattr(raw, "state", "")), "grid": g,
                "levelup": lvl > prev_level,
            }
        )
        prev_level = lvl

    final_level = steps[-1]["level"]
    final_state = steps[-1]["state"]
    n_levelup = sum(1 for s in steps if s.get("levelup"))
    data = json.dumps(
        {"steps": steps, "palette": PALETTE}, separators=(",", ":")
    )

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{background:#0d1117;color:#c9d1d9;font:13px ui-monospace,Menlo,monospace;
margin:0}}.w{{max-width:1000px;margin:0 auto;padding:18px}}
h1{{font-size:17px;margin:0 0 4px}}.ban{{background:#161b22;border:1px solid
#2d7d46;border-radius:6px;padding:9px;color:#7fdbb6;font-size:12px;
margin:10px 0}}.bar{{display:flex;gap:8px;align-items:center;margin:10px 0}}
button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
border-radius:5px;padding:6px 12px;cursor:pointer;font:inherit}}
button:hover{{background:#30363d}}#tt{{font-size:14px;color:#3fb950;
font-weight:700}}#cv{{image-rendering:pixelated;border:1px solid #30363d;
background:#000;display:block;margin:8px 0}}#sl{{flex:1}}
.lv{{color:#d29922}}.win{{color:#3fb950;font-weight:700}}
</style></head><body><div class="w">
<h1>{html.escape(title)}</h1>
<div class="ban">Deterministic ft09 REPLAY of the real recorded action+coord
sequence (actions_log ts-slice). Grids are the actual game state regenerated
by re-executing the real moves on the deterministic ft09 env &mdash; not a
fabrication. {len(steps)} steps &middot; {n_levelup} level-ups &middot;
final L{final_level} {html.escape(final_state)}.</div>
<div class="bar">
<button id="pv">&#9664; Prev</button>
<button id="pl">&#9654; Play</button>
<button id="nx">Next &#9654;</button>
<input id="sl" type="range" min="0" value="0"/>
<span id="ix"></span></div>
<div id="tt"></div>
<canvas id="cv" width="512" height="512"></canvas>
<div id="meta"></div>
<script id="d" type="application/json">{data}</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const S=D.steps,P=D.palette,cv=document.getElementById('cv'),
cx=cv.getContext('2d');let i=0,timer=null;
document.getElementById('sl').max=S.length-1;
function draw(){{
 const s=S[i],g=s.grid;if(!g){{cx.clearRect(0,0,512,512);return;}}
 const H=g.length,W=g[0].length,px=Math.floor(512/Math.max(H,W));
 cx.clearRect(0,0,512,512);
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){{
  cx.fillStyle=P[((g[y][x]%16)+16)%16]||'#000';
  cx.fillRect(x*px,y*px,px,px);}}
 if(s.x!=null&&s.y!=null){{cx.strokeStyle='#ff00ff';cx.lineWidth=2;
  cx.strokeRect(s.x*px-2,s.y*px-2,px+4,px+4);}}
 const coord=(s.x!=null)?` (${{s.x}},${{s.y}})`:'';
 const lu=s.levelup?` <span class=lv>&#8593; LEVEL UP</span>`:'';
 const wn=(s.state||'').includes('WIN')?` <span class=win>WIN</span>`:'';
 document.getElementById('tt').innerHTML=
  `Step ${{s.n}} / ${{S.length-1}} &mdash; <b>${{s.action}}</b>${{coord}}`+
  ` &rarr; L${{s.level}}${{lu}}${{wn}}`;
 document.getElementById('ix').textContent=`${{i}}/${{S.length-1}}`;
 document.getElementById('sl').value=i;
 document.getElementById('meta').textContent=
  `state=${{s.state}}  count=${{s.n}}`;
}}
function go(j){{i=Math.max(0,Math.min(S.length-1,j));draw();}}
document.getElementById('nx').onclick=()=>go(i+1);
document.getElementById('pv').onclick=()=>go(i-1);
document.getElementById('sl').oninput=e=>go(+e.target.value);
document.getElementById('pl').onclick=function(){{
 if(timer){{clearInterval(timer);timer=null;this.innerHTML='&#9654; Play';}}
 else{{this.innerHTML='&#10073;&#10073; Pause';
  timer=setInterval(()=>{{if(i>=S.length-1){{clearInterval(timer);
   timer=null;document.getElementById('pl').innerHTML='&#9654; Play';}}
   else go(i+1);}},220);}}}};
draw();
</script></div></body></html>"""
    out_p.write_text(doc)
    print(
        f"WROTE {out_p} ({out_p.stat().st_size}B) steps={len(steps)} "
        f"levelups={n_levelup} final=L{final_level} {final_state}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

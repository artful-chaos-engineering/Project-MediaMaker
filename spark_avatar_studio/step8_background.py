#!/usr/bin/env python3
"""Step 8 - Remove or replace the background of a rendered avatar video (no new lip-sync).

Takes out/<name>/<name>_raw.mp4 (or any video) and writes out/<name>/<name>_<mode>.mp4. Choose:
  keep         leave the room as filmed
  transparent  alpha channel: ProRes 4444 .mov (editors) or VP9 .webm (browsers); drop it over slides in your editor
  color        one flat colour, e.g. brand navy
  gradient     two-colour vertical blend
  image        a still: slide, office, brand backdrop (cover-fitted)
  video        a looping clip behind you
  blur         the original room, blurred and darkened
Engines: birefnet (AI matte, works on any room) or chroma (green/blue screen key: cleaner edges if you filmed on a screen).
Edge tuning: --choke pulls the edge inward against halos of the old room; --alpha-smooth steadies flicker; --feather softens.

    python step8_background.py smoke --mode color --color "#0B1F3A"
    python step8_background.py lesson --mode image --image backdrops/office.jpg
    python step8_background.py lesson --mode transparent --webm
    python step8_background.py lesson --mode video --video loops/particles.mp4 --engine chroma --chroma-key "#00B140"
"""
import argparse, subprocess, sys
from pathlib import Path
from studio.common import ROOT, Spark, cfg, say, set_state


def main():
    c = cfg(); b = c["background"]; ap = argparse.ArgumentParser(); ap.add_argument("name")
    ap.add_argument("--mode", default=b["mode"]); ap.add_argument("--engine", default=b["engine"]); ap.add_argument("--color", default=b["color"]); ap.add_argument("--color2", default=b["color2"])
    ap.add_argument("--image"); ap.add_argument("--video"); ap.add_argument("--chroma-key", default=b["chroma_key"]); ap.add_argument("--choke", type=float, default=b["choke"])
    ap.add_argument("--alpha-smooth", type=float, default=b["alpha_smooth"]); ap.add_argument("--feather", type=float, default=b["feather"]); ap.add_argument("--webm", action="store_true")
    ap.add_argument("--src", help="a different input video"); args = ap.parse_args()
    d = ROOT / "out" / args.name; src = Path(args.src) if args.src else d / f"{args.name}_raw.mp4"
    if not src.exists(): sys.exit(f"{src} not found; render with step 7 first")
    ext = "webm" if (args.mode == "transparent" and args.webm) else "mov" if args.mode == "transparent" else "mp4"
    dst = d / f"{args.name}_{args.mode}.{ext}"; sp = Spark(); a = c["avatar"]["name"]
    sp.sync_code(); sp.put(src, f"out/{args.name}/{src.name}")
    extra = ""
    for k in ("image", "video"):
        v = getattr(args, k)
        if v: sp.put(Path(v), f"backdrops/{Path(v).name}"); extra += f" --{k} backdrops/{Path(v).name}"
    say(f"background stage ({args.mode}, engine {args.engine}) on the Spark")
    sp.run(f"PYTHONPATH=. python -m studio.background out/{args.name}/{src.name} out/{args.name}/{dst.name} --mode {args.mode} --engine {args.engine} --model {c['models']['matting']} "
           f"--size {b['matte_size']} --color '{args.color}' --color2 '{args.color2}' --chroma-key '{args.chroma_key}' --alpha-smooth {args.alpha_smooth} --choke {args.choke} --feather {args.feather}{extra}")
    sp.get(f"out/{args.name}/{dst.name}", dst); set_state(last_background=str(dst)); say(f"wrote {dst}")


if __name__ == "__main__":
    main()

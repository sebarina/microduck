"""OnDeskGen command line.

  ondeskgen generate desk --text "浅色橡木书桌，右侧带两个抽屉"         text -> concept image -> 3D -> asset
  ondeskgen generate sofa --text "深灰色布艺三人沙发" --no-concept       text -> 3D directly
  ondeskgen generate chair --image ./chair.jpg --material                photo -> 3D -> material transfer
  ondeskgen generate desk --image ./desk.jpg --four-view                 one image -> 4 views -> 3D
  ondeskgen generate desk --image a.jpg --image b.jpg --image c.jpg      multi-view photos -> 3D
  ondeskgen generate sofa --image ./sofa.jpg --restyle "换成墨绿色丝绒"   same shape, new material
  ondeskgen generate desk --text "..." --size 1.4 0.7 0.75 --final       given size (W D H, m), Lux3D G1
  ondeskgen generate desk --text "..." --dry-run                         plan only, no credits
  ondeskgen pack my_pack --all                                           MuJoCo asset pack (+ zip)
  ondeskgen train-export --all                                           static MJCF (box / mesh collision)

Environment: AHOLO_API_KEY (Lux3D, region in AHOLO_REGION, default cn), ONDESKGEN_UPLOAD_URL (file
service for handing local files to Lux3D by URL), ONDESKGEN_HOME (data root), ONDESKGEN_CONFIG.
Every Lux3D task is archived and reused for an identical request, so re-running costs no credits.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import engine_config
from .engine import MARKER, RUNS, Engine, Request, build_prompt
from .events import Events
from .paths import ASSETS_PROCESSED


def _all_assets() -> list[str]:
    return sorted(p.parent.name for p in ASSETS_PROCESSED.glob(f"*/{MARKER}"))


def cmd_generate(a) -> int:
    if not (a.text or a.prompt or a.image):
        print("give --text/--prompt or --image", file=sys.stderr)
        return 2
    req = Request(category=a.category, text=a.text, images=a.image,
                  size=dict(zip(("width", "depth", "height"), a.size)) if a.size else None,
                  concept_image=not a.no_concept, four_view=a.four_view, material_transfer=a.material,
                  part_split=not a.no_parts,
                  articulation=False if a.no_articulation else (True if a.articulation_prompt else None),
                  export=not a.no_export, version="G1" if a.final else None, name=a.name, prompt=a.prompt,
                  articulation_prompt=a.articulation_prompt, material_prompt=a.restyle)
    if req.text or req.prompt or not req.images:
        print("prompt:", build_prompt(req))
    name = Engine.asset_name(req)
    ev = Events(RUNS / name / "events.jsonl")
    try:
        rec = Engine(dry_run=a.dry_run, events=ev, log=lambda m: (print(m), ev("log", "info", msg=str(m)))).generate(req)
    except Exception as e:
        ev("run", "error", error=f"{type(e).__name__}: {e}")
        raise
    print(json.dumps({k: rec.get(k) for k in ("asset", "status", "processed", "credits_spent", "seconds")},
                     ensure_ascii=False, indent=1))
    return 0 if rec["status"] in ("done", "dry_run") else 1


def cmd_pack(a) -> int:
    from .pack import build_pack
    names = _all_assets() if a.all else a.assets
    if not names:
        print("give asset names or --all", file=sys.stderr)
        return 2
    p = build_pack(names, a.name, zip_it=not a.no_zip)
    print(json.dumps({"assets": [x["id"] for x in p["assets"]], "zip": p.get("zip")}, ensure_ascii=False, indent=1))
    return 0


def cmd_train_export(a) -> int:
    from .train_export import EXPORTS, export_all
    names = _all_assets() if a.all else a.assets
    if not names:
        print("give asset names or --all", file=sys.stderr)
        return 2
    man = export_all(names, a.name, zip_it=not a.no_zip)
    print(f"[done] {EXPORTS / a.name}" + (f"  zip {man['zip']['bytes'] / 1e6:.1f} MB" if "zip" in man else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ondeskgen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="text / image(s) -> simulation-ready MuJoCo asset")
    g.add_argument("category", choices=sorted(engine_config()["categories"]))
    g.add_argument("--text", default=None)
    g.add_argument("--prompt", default=None, help="full Lux3D prompt (bypasses the category template)")
    g.add_argument("--image", action="append", default=[])
    g.add_argument("--size", nargs=3, type=float, metavar=("W", "D", "H"))
    g.add_argument("--no-concept", action="store_true", help="text -> 3D directly (no concept image)")
    g.add_argument("--four-view", action="store_true", help="one image -> 4 views -> multi-view 3D")
    g.add_argument("--material", action="store_true", help="re-texture the mesh from the (first) photo")
    g.add_argument("--restyle", default=None, help="new material: Lux3D image+text -> reference -> material transfer")
    g.add_argument("--no-parts", action="store_true")
    g.add_argument("--no-articulation", action="store_true")
    g.add_argument("--articulation-prompt", default=None)
    g.add_argument("--no-export", action="store_true", help="skip Lux3D USDZ / OBJ / STL export")
    g.add_argument("--final", action="store_true", help="Lux3D G1 instead of G1-Turbo")
    g.add_argument("--name", default=None)
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(fn=cmd_generate)
    p = sub.add_parser("pack", help="asset pack (MJCF fragments, meshes, provenance, Lux3D exports)")
    p.add_argument("name")
    p.add_argument("assets", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.set_defaults(fn=cmd_pack)
    t = sub.add_parser("train-export", help="static MJCF for policy training (box and mesh collision)")
    t.add_argument("assets", nargs="*")
    t.add_argument("--all", action="store_true")
    t.add_argument("--name", default="ondeskgen_static_v1")
    t.add_argument("--no-zip", action="store_true")
    t.set_defaults(fn=cmd_train_export)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

"""Raw Lux3D GLB -> processed asset directory.

assets/processed/<name>/
  <name>.xml        MJCF fragment (see mjcf.py)
  visual/part_*.obj decimated visual mesh, one file per color cluster
  collision/hull_*.obj  convex collision hulls (CoACD, furniture default)
  asset.json        provenance, normalization report, proxies, physics
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ondeskgen.paths import REPO_ROOT, AssetSpec
from .mjcf import ChildBody, ContactParams, DynamicParams, FragmentSpec, write_fragment
from .normalize import load_mesh_with_colors, normalize
from .hulls import CATEGORY_MASS, FURNITURE_FLOOR_FRICTION, HullParams, box_solids, fit_convex_hulls, hull_meshes, mass_properties
from .proxies import CATEGORY_PARAMS, ProxyParams, VoxelParams, ball_proxy, fit_boxes, fit_voxel_boxes
from .visual import build_visual

PIPELINE_VERSION = "d28.0"     # m1.0: static furniture, layered boxes

# Official kick ball (third_party/microduck_rl/.../robot/microduck/ball.xml):
# 70 mm / 15 g thin hollow sphere, I = 2/3 m r^2, friction (0.5, 0.005, ...).
# DEVIATION: condim 6 + rolling friction 0.002 instead of the
# official condim 3 (where rolling friction is ignored and a 1 m/s ball rolls
# ~30 m). Measured: 1 m/s -> ~2.0 m, stops in ~6.8 s.
BALL = {
    "radius": 0.035,
    "mass": 0.015,
    "diaginertia": (1.225e-5, 1.225e-5, 1.225e-5),
    "contact": ContactParams(friction=(0.5, 0.005, 0.002), condim=6),
}
# Furniture geoms keep mu = 1.0 against the robot and the ball (MuJoCo takes
# the max of the two geoms); furniture-floor contacts are explicit <pair>s with
# hulls.FURNITURE_FLOOR_FRICTION, added by the scene builder. Values: engine config.
from ondeskgen.config import engine_config as _engine_config  # noqa: E402

_PHYS = _engine_config()["physics"]
FURNITURE_CONTACT = ContactParams(friction=tuple(_PHYS["furniture_friction"]), condim=int(_PHYS["condim"]),
                                  solref=tuple(_PHYS["solref"]), solimp=tuple(_PHYS["solimp"]))


def default_hull_params() -> HullParams:
    return HullParams(**_engine_config()["collision"]["coacd"])
COLLISION_METHODS = ("coacd", "layers", "voxel")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 (no commit yet)
        return None


def find_raw_source(raw_root: Path, name: str, raw_run: str | None = None) -> tuple[Path, dict | None]:
    """Pinned run (``raw_run`` = "drafts/<task_id>" or "final/<task_id>") if given;
    else prefer the final G1 run; fall back to the newest G1-Turbo draft."""
    if raw_run:
        run = raw_root / name / raw_run
        return run / "model.glb", json.loads((run / "meta.json").read_text())
    for kind in ("final", "drafts"):
        d = raw_root / name / kind
        if not d.exists():
            continue
        runs = sorted((p for p in d.iterdir() if (p / "meta.json").exists()),
                      key=lambda p: json.loads((p / "meta.json").read_text())["finished_at"])
        if runs:
            run = runs[-1]
            return run / "model.glb", json.loads((run / "meta.json").read_text())
    raise FileNotFoundError(f"no Lux3D run archived for {name} under {raw_root / name}")


def process_asset(spec: AssetSpec, raw_glb: Path, out_root: Path, *,
                  lux3d_meta: dict | None = None, up_axis: str = "y",
                  target_faces: int = 4000, n_colors: int = 6,
                  proxy_params: ProxyParams | None = None,
                  voxel_params: VoxelParams | None = None,
                  collision: str | None = None,
                  hull_params: HullParams | None = None,
                  dynamic: bool = True,
                  parts_glb: Path | None = None,
                  articulation_zip: Path | None = None) -> Path:
    """Collision method for furniture (``collision``):
      "coacd"  convex decomposition (default, hulls.py)
      "layers" layered boxes (fit_boxes, the layered-box method)
      "voxel"  voxel boxes (fit_voxel_boxes experiment); also selected by ``voxel_params``
    ``dynamic``: furniture gets a freejoint and a reference mass (hulls.CATEGORY_MASS);
    False writes a static fragment (no joint, no mass).
    OnDeskGen, CoACD only:
      ``parts_glb``        Lux3D part_split output (same frame as ``raw_glb``): convex decomposition
                           per part (cleaner hulls than on the fused mesh)
      ``articulation_zip`` Lux3D articulation output (URDF): moving parts become child bodies with
                           slide/hinge joints; their pull direction also fixes the front (+X)."""
    if collision is None:
        collision = "voxel" if voxel_params is not None else "coacd"
    if collision not in COLLISION_METHODS:
        raise ValueError(f"collision must be one of {COLLISION_METHODS}, got {collision!r}")
    t0 = time.time()
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh, colors = load_mesh_with_colors(raw_glb)
    artic = None
    if articulation_zip is not None and spec.category != "ball":
        from ondeskgen.articulation import load_articulation
        artic = load_articulation(Path(articulation_zip), mesh)
        fv = artic.front_vector()
        if fv is not None:
            spec.extra = {**(spec.extra or {}), "front_vector_raw": fv.tolist()}
    norm = normalize(mesh, colors, spec, up_axis=up_axis)
    for old in (out_dir / "visual").rglob("*.obj") if (out_dir / "visual").exists() else []:
        old.unlink()
    children: list[ChildBody] = []
    artic_info = None
    if artic is not None:
        T = norm.transform
        R = T[:3, :3] / norm.scale
        link_mesh = {}
        for lk in artic.links:
            m = lk.mesh.copy()
            m.apply_transform(T)
            link_mesh[lk.name] = (m, lk.face_colors)
        root = artic.root
        parts = build_visual(*link_mesh[root], out_dir, target_faces, n_colors, subdir=f"visual/{root}")
    else:
        parts = build_visual(norm.mesh, norm.face_colors, out_dir, target_faces, n_colors)

    if spec.category == "ball":
        # Physics layer is the official ball; the Lux3D mesh is visual only and
        # was already scaled to the same diameter by normalize().
        proxies = ball_proxy(BALL["radius"])
        contact = BALL["contact"]
        dynamic = DynamicParams(mass=BALL["mass"], diaginertia=BALL["diaginertia"],
                                com=(0.0, 0.0, BALL["radius"]))
        params_used = None
        fit_stats = None
        mass_info = None
    else:
        fit_stats: dict = {}
        t_fit = time.time()
        for old in (out_dir / "collision").glob("*.obj"):
            old.unlink()
        if collision == "coacd" and artic is not None:
            params_used = hull_params or default_hull_params()
            proxies = fit_convex_hulls(link_mesh[root][0], out_dir, params_used, stats=fit_stats)
            solids = hull_meshes(proxies, out_dir)
            child_parts = []
            for j in artic.joints:
                if j.type not in ("prismatic", "revolute"):
                    continue
                cm, cc = link_mesh[j.child]
                cst: dict = {}
                cpx = fit_convex_hulls(cm, out_dir, params_used, prefix=f"{j.child}_hull", stats=cst, clear=False)
                cvis = build_visual(cm, cc, out_dir, max(800, target_faces // 3), n_colors,
                                    subdir=f"visual/{j.child}")
                o = (T @ np.append(j.origin, 1.0))[:3]
                ax = R @ j.axis
                ax = ax / np.linalg.norm(ax)
                rng = (j.lower * norm.scale, j.upper * norm.scale) if j.type == "prismatic" else (j.lower, j.upper)
                child_parts.append((j, cpx, cvis, o, ax, rng, hull_meshes(cpx, out_dir), cst))
            fit_stats["links"] = {root: {"hulls": len(proxies)},
                                  **{c[0].child: {"hulls": len(c[1])} for c in child_parts}}
        elif collision == "coacd" and parts_glb is not None:
            params_used = hull_params or default_hull_params()
            import trimesh as _tm
            scene = _tm.load(parts_glb, force="scene")
            proxies, first = [], True
            per_part = []
            for k, node in enumerate(scene.graph.nodes_geometry):
                Tn, g = scene.graph[node]
                geom = scene.geometry[g]
                if not isinstance(geom, _tm.Trimesh) or not len(geom.faces):
                    continue
                pm = geom.copy()
                pm.apply_transform(Tn)
                pm.apply_transform(norm.transform)
                pst: dict = {}
                px = fit_convex_hulls(pm, out_dir, params_used, prefix=f"part{k}_hull", stats=pst, clear=first)
                first = False
                proxies += px
                per_part.append({"part": k, "faces": int(len(pm.faces)), "hulls": len(px),
                                 "bounds": [np.round(b, 4).tolist() for b in pm.bounds]})
            fit_stats.update({"method": "coacd", "per_part": per_part, "params": params_used.__dict__,
                              "hulls": len(proxies), "source": "lux3d part_split"})
            solids = hull_meshes(proxies, out_dir)
        elif collision == "coacd":
            params_used = hull_params or default_hull_params()
            proxies = fit_convex_hulls(norm.mesh, out_dir, params_used, stats=fit_stats)
            solids = hull_meshes(proxies, out_dir)
        elif collision == "voxel":
            params_used = voxel_params or VoxelParams()
            fit_stats["method"] = "voxel"
            proxies = fit_voxel_boxes(norm.mesh, params_used, stats=fit_stats)
            solids = box_solids(proxies)
        else:
            params_used = proxy_params or CATEGORY_PARAMS[spec.category]
            fit_stats["method"] = "layers"
            proxies = fit_boxes(norm.mesh, params_used, stats=fit_stats)
            solids = box_solids(proxies)
        fit_stats["seconds"] = round(time.time() - t_fit, 3)
        contact = FURNITURE_CONTACT
        mass_info = None
        if dynamic:
            total = CATEGORY_MASS[spec.category]
            if artic is not None and child_parts:
                # split the reference mass by solid volume between the root and the moving parts
                vols = [sum(s.volume for s in solids)] + [sum(s.volume for s in c[6]) for c in child_parts]
                masses = [total * v / sum(vols) for v in vols]
                mp = mass_properties(solids, masses[0])
                for c, mc in zip(child_parts, masses[1:]):
                    j, cpx, cvis, o, ax, rng, csol, cst = c
                    cmp_ = mass_properties(csol, mc)
                    children.append(ChildBody(
                        name=j.child, pos=tuple(o), joint_type="slide" if j.type == "prismatic" else "hinge",
                        axis=tuple(ax), range=tuple(rng), visual_parts=cvis, proxies=cpx,
                        dynamic=DynamicParams(mass=cmp_.mass, com=tuple(cmp_.com), fullinertia=tuple(cmp_.fullinertia))))
                artic_info = {
                    "source": artic.source, "corrections": artic.corrections, "root": root,
                    "joints": [{"name": f"{spec.name}_{c.name}_joint", "child": c.name, "type": c.joint_type,
                                "axis": [round(x, 5) for x in c.axis], "range": [round(x, 4) for x in c.range],
                                "origin": [round(x, 4) for x in c.pos], "mass_kg": round(c.dynamic.mass, 3),
                                "hulls": len(c.proxies), "damping": c.damping, "frictionloss": c.frictionloss}
                               for c in children]}
            else:
                mp = mass_properties(solids, total)
            mass_info = mp.__dict__
            dynamic = DynamicParams(mass=mp.mass, com=tuple(mp.com), fullinertia=tuple(mp.fullinertia))
        else:
            dynamic = None

    frag = FragmentSpec(spec.name, parts, proxies, contact, dynamic, children)
    all_proxies = proxies + [p for c in children for p in c.proxies]
    xml_path = write_fragment(frag, out_dir / f"{spec.name}.xml")

    lo = np.min([p.bounds()[0] for p in all_proxies], axis=0)
    hi = np.max([p.bounds()[1] for p in all_proxies], axis=0)
    info = {
        "name": spec.name,
        "category": spec.category,
        "pipeline": {"version": PIPELINE_VERSION, "git_commit": _git_commit(),
                     "processed_at": time.time(), "seconds": round(time.time() - t0, 2)},
        "source": {
            "raw_file": str(Path(raw_glb).resolve().relative_to(REPO_ROOT))
            if Path(raw_glb).resolve().is_relative_to(REPO_ROOT) else str(raw_glb),
            "raw_sha256": _sha256(raw_glb),
            "prompt": (lux3d_meta or {}).get("prompt", spec.prompt),
            "lux3d_task_id": (lux3d_meta or {}).get("task_id"),
            "lux3d_version": (lux3d_meta or {}).get("version"),
            "lux3d_region": (lux3d_meta or {}).get("region"),
            "is_draft": (lux3d_meta or {}).get("version") == "G1-Turbo",
            "synthetic": lux3d_meta is None,
            # text + image generation only (absent for text-only assets, so older records are unchanged)
            **({"mode": "text+image", "image": lux3d_meta["image"]}
               if (lux3d_meta or {}).get("image") else {}),
        },
        "frame": {"units": "meter", "up_axis": "z", "origin": "bottom_center", "front": "+x"},
        "normalization": norm.report,
        "size_m": {"depth_x": norm.report["extents_xyz"][0], "width_y": norm.report["extents_xyz"][1],
                   "height_z": norm.report["extents_xyz"][2]},
        "visual": {"faces_raw": int(len(mesh.faces)), "faces": int(sum(p.faces for p in parts)),
                   "parts": [p.__dict__ for p in parts]},
        "articulation": artic_info,
        "collision": {"proxies": [p.to_dict() for p in all_proxies],
                      "bounds": [lo.tolist(), hi.tolist()],
                      "params": params_used.__dict__ if params_used else None,
                      "fit_stats": fit_stats},
        "physics": ({"static": True, "mass": None, **{k: v for k, v in contact.__dict__.items()}}
                    if dynamic is None else
                    {"static": False, "mass": dynamic.mass,
                     "diaginertia": None if dynamic.diaginertia is None else list(dynamic.diaginertia),
                     "fullinertia": None if dynamic.fullinertia is None else list(dynamic.fullinertia),
                     "com": list(dynamic.com),
                     **({"mass_source": mass_info["source"], "solid_volume_m3": round(mass_info["volume_m3"], 6),
                         "floor_friction": FURNITURE_FLOOR_FRICTION, "estimate": True}
                        if spec.category != "ball" else {}),
                     **{k: v for k, v in contact.__dict__.items()}}),
        "mjcf": xml_path.name,
    }
    (out_dir / "asset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=list))
    return out_dir

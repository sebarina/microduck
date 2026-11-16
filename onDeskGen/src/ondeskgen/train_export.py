"""Static MJCF export of OnDeskGen assets for policy training (e.g. mjlab on Hugging Face).

Differences from the asset pack's dynamic fragments:
  - one static body per asset: no freejoint, no joints (drawers welded shut), no inertial, no
    actuators; body origin at the bottom center (the dynamic fragments put it at the center of mass)
  - every mesh rewritten in that frame (geoms at the identity pose), .obj next to the XML, relative paths
  - two collision variants:
      box   one oriented box per CoACD hull (z-aligned when that costs < 15 % volume): primitives
            only, fastest on GPU simulators, matches the official apartment.xml
      mesh  the CoACD hulls, each simplified to <= MAX_VERTS vertices (MuJoCo collides meshes as
            their convex hull; keeping many convex pieces keeps the concave shape)
  - visual geoms: group 2, contype/conaffinity 0; collision geoms: group 3, contype/conaffinity 1
  - every name starts with the asset id; no MuJoCo <default> classes (merges cleanly)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from ondeskgen.paths import ASSETS_PROCESSED, OUTPUTS, REPO_ROOT

from .config import engine_config

MAX_VERTS = 64
YAW_BOX_SLACK = 1.15               # z-aligned box if its volume <= 1.15 x the free oriented box
EXPORTS = OUTPUTS / "train_export"
_P = engine_config()["physics"]


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.ravel(v))


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# OnDeskGen static export (meters, Z up, origin at the asset's bottom center)\n")
        fh.writelines(f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in v)
        fh.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f)


def _world_geoms(asset_dir: Path, info: dict):
    """Compiled dynamic fragment at rest (drawers closed): every mesh geom's vertices in the asset frame
    (origin bottom center). Returns (visual [(verts, faces, rgba)], hulls [verts])."""
    m = mujoco.MjModel.from_xml_path(str(asset_dir / info["mjcf"]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    vis, col = [], []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        if m.geom_group[g] == 2:
            mat = m.geom_matid[g]
            rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
            vis.append((v, m.mesh_face[fa:fa + fn].copy(), rgba.copy()))
        elif m.geom_contype[g] or m.geom_conaffinity[g]:
            col.append(v)
    return vis, col


def simplify_hull(v: np.ndarray, n: int = MAX_VERTS) -> np.ndarray:
    """At most n hull vertices: keep the 6 axis extremes, then farthest-point sampling over the rest
    (an inner approximation)."""
    hv = v[ConvexHull(v).vertices]
    if len(hv) <= n:
        return hv
    keep = sorted({int(i) for i in np.r_[hv.argmin(0), hv.argmax(0)]})
    dist = np.min(np.linalg.norm(hv[:, None] - hv[keep][None], axis=2), axis=1)
    while len(keep) < n:
        i = int(dist.argmax())
        keep.append(i)
        dist = np.minimum(dist, np.linalg.norm(hv - hv[i], axis=1))
    return hv[keep]


def fit_box(v: np.ndarray) -> dict:
    """Oriented box around a hull: z-aligned (yaw only) unless a free orientation is much tighter.
    Hulls standing on the floor always get a z-aligned box (flat bottom at z = 0, never below it)."""
    T2, ext2 = trimesh.bounds.oriented_bounds_2D(v[:, :2])
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    vol_yaw = ext2[0] * ext2[1] * (z1 - z0)
    T3, ext3 = trimesh.bounds.oriented_bounds(v)
    Tinv = np.linalg.inv(T3)
    corners = (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * np.asarray(ext3) / 2) \
        @ Tinv[:3, :3].T + Tinv[:3, 3]
    if z0 < 0.002 or corners[:, 2].min() < 0.0 or vol_yaw <= YAW_BOX_SLACK * float(np.prod(ext3)):
        R2 = T2[:2, :2].T                               # T2 maps points to the box frame
        c2 = R2 @ (-T2[:2, 2])
        R = np.eye(3)
        R[:2, :2] = R2
        return {"pos": np.r_[c2, (z0 + z1) / 2], "R": R, "half": np.r_[np.asarray(ext2) / 2, (z1 - z0) / 2],
                "mode": "yaw"}
    return {"pos": Tinv[:3, 3], "R": Tinv[:3, :3], "half": np.asarray(ext3) / 2, "mode": "free"}


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


# ------------------------------------------------------------------ MJCF
def _mjcf(name: str, vis, col_kind: str, shapes) -> str:
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")
    body = ET.SubElement(wb, "body", name=name, pos="0 0 0")
    for i, (_, _, rgba) in enumerate(vis):
        ET.SubElement(asset, "mesh", name=f"{name}_vis_{i}", file=f"visual/vis_{i}.obj", inertia="shell")
        ET.SubElement(asset, "material", name=f"{name}_mat_{i}", rgba=_f(rgba), specular="0.2", shininess="0.2")
        ET.SubElement(body, "geom", name=f"{name}_vis_{i}", type="mesh", mesh=f"{name}_vis_{i}",
                      material=f"{name}_mat_{i}", contype="0", conaffinity="0", group="2", density="0")
    common = {"contype": "1", "conaffinity": "1", "group": "3", "rgba": "0.0 0.9 1.0 0.35", "density": "0",
              "friction": _f(_P["furniture_friction"]), "condim": str(_P["condim"]),
              "solref": _f(_P["solref"]), "solimp": _f(_P["solimp"])}
    for i, s in enumerate(shapes):
        if col_kind == "box":
            ET.SubElement(body, "geom", name=f"{name}_col_{i}", type="box", size=_f(s["half"]), pos=_f(s["pos"]),
                          quat=_f(_quat(s["R"])), **common)
        else:
            ET.SubElement(asset, "mesh", name=f"{name}_col_{i}", file=f"collision/col_{i}.obj")
            ET.SubElement(body, "geom", name=f"{name}_col_{i}", type="mesh", mesh=f"{name}_col_{i}", **common)
    ET.indent(root, "  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def export_asset(asset_dir: Path, out_dir: Path, kind: str) -> dict:
    asset_dir, out_dir = Path(asset_dir), Path(out_dir)
    info = json.loads((asset_dir / "asset.json").read_text())
    name = info["name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    vis, hulls = _world_geoms(asset_dir, info)
    for i, (v, f, _) in enumerate(vis):
        _write_obj(out_dir / "visual" / f"vis_{i}.obj", v, f)
    if kind == "box":
        shapes = [fit_box(v) for v in hulls]
        extra = {"yaw_boxes": sum(s["mode"] == "yaw" for s in shapes), "free_boxes": sum(s["mode"] == "free" for s in shapes)}
    else:
        shapes = []
        for i, v in enumerate(hulls):
            sv = simplify_hull(v)
            h = trimesh.convex.convex_hull(sv)          # consistent outward winding
            _write_obj(out_dir / "collision" / f"col_{i}.obj", np.asarray(h.vertices), np.asarray(h.faces))
            shapes.append(np.asarray(h.vertices))
        extra = {"max_vertices_before": int(max(len(ConvexHull(v).vertices) for v in hulls)),
                 "max_vertices": int(max(len(s) for s in shapes))}
    xml = out_dir / f"{name}.xml"
    xml.write_text(_mjcf(name, vis, kind, shapes))
    lo = np.vstack(hulls).min(0)
    hi = np.vstack(hulls).max(0)
    rec = {"asset": name, "category": info["category"], "variant": kind, "mjcf": f"{name}.xml",
           "collision_geoms": len(shapes), "visual_geoms": len(vis), **extra,
           "collision_bounds_m": [np.round(lo, 4).tolist(), np.round(hi, 4).tolist()],
           "size_m": info["size_m"], "frame": "meters, Z up, origin at bottom center, front faces +X",
           "articulation_note": ("drawers welded shut (Lux3D articulation joints removed)"
                                 if info.get("articulation") else None)}
    rec["files"] = {str(p.relative_to(out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                    for p in sorted(out_dir.rglob("*")) if p.is_file()}
    return rec


def _geom_points(m, d, g) -> np.ndarray:
    """World-frame points of a collision geom: mesh vertices, or box corners."""
    R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        local = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
        local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
    return local @ R.T + p


# ------------------------------------------------------------------ render + package
def render(xml: Path, out_png: Path, size=(480, 400)) -> Path:
    from PIL import Image
    xml, out_png = Path(xml), Path(out_png)
    s = mujoco.MjSpec()
    s.worldbody.add_light(pos=[1, 1, 3], dir=[-0.3, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], rgba=[0.22, 0.26, 0.31, 1])
    s.attach(mujoco.MjSpec.from_file(str(xml)), frame=s.worldbody.add_frame(), prefix="")
    m = s.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pts = np.vstack([_geom_points(m, d, g) for g in range(m.ngeom) if m.geom_group[g] == 3])   # not the floor
    lo, hi = pts.min(0), pts.max(0)
    w, h = size
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (lo + hi) / 2
    cam.distance, cam.azimuth, cam.elevation = 2.0 * float(np.linalg.norm(hi - lo)) + 0.3, 210, -25
    tiles = []
    for vis, col in ((1, 0), (1, 1), (0, 1)):
        opt = mujoco.MjvOption()
        opt.geomgroup[2], opt.geomgroup[3] = vis, col
        r.update_scene(d, cam, opt)
        tiles.append(Image.fromarray(r.render()))
    r.close()
    sheet = Image.new("RGB", (w * 3, h))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * w, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png


def export_all(names: list[str], out_name: str = "ondeskgen_static_v1", processed: Path = ASSETS_PROCESSED,
               zip_it: bool = True, log=print) -> dict:
    root = EXPORTS / out_name
    if root.exists():
        shutil.rmtree(root)
    manifest = {"format": "ondeskgen-static/1", "config_hash": None, "variants": {}, "assets": names,
                "notes": ["static: no freejoint/joints/actuators/inertial; drawers welded shut",
                          "meters, Z up, origin at each asset's bottom center, front faces +X",
                          "visual: group 2, contype/conaffinity 0; collision: group 3, contype/conaffinity 1",
                          "all names start with the asset id; no <default> classes"]}
    from .config import config_hash
    manifest["config_hash"] = config_hash()
    for kind in ("box", "mesh"):
        recs = []
        for n in names:
            rec = export_asset(processed / n, root / kind / n, kind)
            render(root / kind / n / f"{n}.xml", root / kind / n / "preview.png")
            recs.append(rec)
            log(f"[export] {kind:4s} {n}: {rec['collision_geoms']} collision geoms")
        manifest["variants"][kind] = recs
        (root / kind / "manifest.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False, default=float))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float))
    (root / "README.md").write_text(_readme(manifest))
    if zip_it:
        z = shutil.make_archive(str(root), "zip", root.parent, root.name)
        manifest["zip"] = {"path": str(Path(z).relative_to(REPO_ROOT)), "bytes": Path(z).stat().st_size}
    return manifest


def _readme(man: dict) -> str:
    L = ["# OnDeskGen 静态训练资产（MJCF）", "",
         "从 OnDeskGen 资产转出的**静态**家具，专门给策略训练用（例如 mjlab）。两个变体，碰撞层不同，视觉层相同：", "",
         "- `box/`：每个碰撞块是一个 box 图元（每个 CoACD 凸包换成它的有向包围盒）。GPU 仿真最快，推荐默认用这个。",
         f"- `mesh/`：每个碰撞块是一个凸网格（CoACD 凸包，简化到最多 {MAX_VERTS} 个顶点）。形状更准。", "",
         "## 约定", "",
         "- 单位米，Z 朝上；每个资产的原点在**底面中心**，正面朝 **+X**。放进场景时直接设 body 的 pos / quat 即可，z = 0 就是贴地。",
         "- **静态刚体**：没有 freejoint、关节、执行器、inertial；书桌抽屉固定在关闭位置。",
         "- 视觉几何：group 2，contype/conaffinity 0（不参与碰撞）；碰撞几何：group 3，contype/conaffinity 1。",
         "- 所有 body / geom / mesh / material 名字都以资产 id 开头（如 `odg_sofa_concept_a_col_3`），没有 `<default>` 类，"
         "合并进其他场景时不容易重名；也可以在 attach 时再统一加前缀。",
         "- 网格都是 `.obj`，路径相对于各自的 XML（`visual/`、`collision/`），整个目录一起拷走即可。",
         f"- 碰撞参数：friction {_P['furniture_friction']}，condim {_P['condim']}，solref {_P['solref']}，solimp {_P['solimp']}。",
         "- 碰撞块是按几何形状切的（CoACD 凸分解），**不是**按坐垫、扶手这样的语义部件切的；凹处（扶手之间、桌下）保留。",
         "", "## 资产", "",
         "| 资产 | 类别 | 尺寸 深×宽×高 (m) | box 个数 | mesh 个数 |",
         "|---|---|---|---|---|"]
    by = {k: {r["asset"]: r for r in v} for k, v in man["variants"].items()}
    for n in man["assets"]:
        b, mm = by["box"][n], by["mesh"][n]
        s = b["size_m"]
        L.append(f"| `{n}` | {b['category']} | {s['depth_x']:.2f}×{s['width_y']:.2f}×{s['height_z']:.2f} | "
                 f"{b['collision_geoms']} | {mm['collision_geoms']} |")
    L += ["", "## 合并示例", "", "```python", "import mujoco",
          "scene = mujoco.MjSpec.from_file('your_scene.xml')",
          "sofa = mujoco.MjSpec.from_file('box/odg_sofa_concept_a/odg_sofa_concept_a.xml')",
          "scene.attach(sofa, frame=scene.worldbody.add_frame(pos=[1.5, 0, 0], euler=[0, 0, 3.1416]), prefix='sofa_')",
          "model = scene.compile()", "```", "",
          "直接拼 XML 也可以：把 `<asset>` 里的 mesh / material 和 `<worldbody>` 里的那个 body 拷进你的场景，"
          "mesh 的 `file` 路径相对你的场景文件改一下（或者在 `<compiler meshdir=...>` 里指过去）。", "",
          "`preview.png`：视觉 / 视觉 + 碰撞 / 只看碰撞。", ""]
    return "\n".join(L)

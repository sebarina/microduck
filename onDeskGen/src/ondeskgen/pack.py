"""OnDeskGen asset pack (format ``ondeskgen-pack/1``): MuJoCo-ready furniture + provenance.

  <pack>/
    pack.json                 index: format, engine version + config hash, assets, checksums
    README.md, LICENSE_NOTES.md, engine_config.json
    assets/<id>/
      <id>.xml                MJCF fragment: free body (+ joints for Lux3D-articulated parts)
      visual/**.obj           decimated, colour-clustered visual meshes (no collision)
      collision/*.obj         CoACD convex hulls (one MuJoCo mesh geom each)
      asset.json              normalization, collision, physics, articulation, provenance
      physics.json            masses, inertia, friction, contact params, domain-randomization ranges
      preview.glb             visual + collision layers (web viewer)
      source/                 lux3d_tasks.json (every Lux3D task in the chain), concept image, views
      exports/                Lux3D multi-format export: model.usdz, model_obj.zip, model.stl
    scenes/<id>.xml           the asset on a floor (paths relative to the pack root)
    scenes/showroom.xml       every asset side by side

Load in MuJoCo: ``python -m mujoco.viewer --mjcf=<pack>/scenes/showroom.xml``. No robot model is included.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import mujoco

from ondeskgen.paths import ASSETS_PROCESSED, ASSETS_RAW, OUTPUTS, REPO_ROOT
from .config import ENGINE_CONFIG, category, config_hash, engine_config

PACKS = OUTPUTS / "packs"
FORMAT = "ondeskgen-pack/1"
EXPORT_NAMES = {"usdz": "model.usdz", "zip": "model_obj.zip", "stl": "model.stl", "3mf": "model.3mf", "fbx": "model.fbx"}


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lux3d_chain(asset: str, raw_root: Path, runs: Path | None = None) -> list[dict]:
    """Lux3D tasks the asset was built from: the engine run's task list when there is one
    (cached tasks may live under another asset), else every archived task of the asset."""
    out = []
    run_json = (runs / asset / "run.json") if runs else None
    if run_json is not None and run_json.exists():
        metas = [REPO_ROOT / t["dir"] / "meta.json" for t in json.loads(run_json.read_text()).get("tasks", [])
                 if t.get("dir")]
    else:
        metas = sorted((raw_root / asset).glob("*/*/meta.json"))
    for meta_p in metas:
        m = json.loads(meta_p.read_text())
        feature = m.get("feature") or ("text_to_3d" if meta_p.parent.parent.name in ("drafts", "final") else
                                       meta_p.parent.parent.name)
        out.append({"feature": feature, "task_id": m.get("task_id"), "seconds": m.get("seconds"),
                    "request": m.get("request") or {"prompt": m.get("prompt"), "version": m.get("version")},
                    "inputs": {k: v for k, v in (m.get("inputs") or {}).items() if k != "key"},
                    "outputs": [{k: f.get(k) for k in ("file", "type", "sha256", "bytes")} for f in m.get("files", [])],
                    "dir": _rel(meta_p.parent)})
    return out


def _physics(info: dict) -> dict:
    c = category(info["category"])
    ph = info["physics"]
    cfg = engine_config()["physics"]
    links = [{"link": "root", "mass_kg": ph.get("mass"), "com": ph.get("com"), "fullinertia": ph.get("fullinertia")}]
    for j in (info.get("articulation") or {}).get("joints", []):
        links.append({"link": j["child"], "mass_kg": j["mass_kg"], "joint": {k: j[k] for k in
                                                                              ("type", "axis", "range", "damping", "frictionloss")}})
    total = sum(x["mass_kg"] or 0 for x in links)
    return {"total_mass_kg": round(total, 3), "links": links, "mass_source": ph.get("mass_source"),
            "estimate": ph.get("estimate", True),
            "contact": {k: ph.get(k) for k in ("friction", "condim", "solref", "solimp")},
            "floor_friction": ph.get("floor_friction"),
            "domain_randomization": {"total_mass_kg": c.get("mass_range"), "floor_friction": cfg.get("floor_friction_range"),
                                     "note": "domain-randomization ranges for training; the MJCF uses the nominal values"}}


def _scene_spec(entries: list[tuple[str, Path, list[float]]]) -> mujoco.MjSpec:
    """Floor + light + the given fragments (id, fragment xml in the pack, xy position)."""
    from ondeskgen.asset.mjcf import load_fragment_spec
    s = mujoco.MjSpec()
    s.option.timestep = 0.005
    s.visual.headlight.ambient = [0.35, 0.35, 0.35]
    tex = s.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                        rgb1=[0.72, 0.58, 0.42], rgb2=[0.66, 0.52, 0.37], width=300, height=300)
    mat = s.add_material(name="floor", texrepeat=[8, 8], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    s.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], material="floor",
                         friction=[1.0, 0.005, 0.0001])
    for aid, frag, xy in entries:
        child = load_fragment_spec(frag)
        frame = s.worldbody.add_frame(pos=[xy[0], xy[1], 0.0])
        s.attach(child, frame=frame, prefix=f"{aid}/")
    return s


def _write_scene(spec: mujoco.MjSpec, path: Path, scene_dir: Path) -> None:
    """Save with mesh paths relative to the scene file (portable pack)."""
    import os
    xml = spec.to_xml()                       # needs resolvable (absolute) paths
    for f in sorted({m.file for m in spec.meshes}, key=len, reverse=True):
        rel = Path(os.path.relpath(Path(f).resolve(), scene_dir.resolve())).as_posix()
        xml = xml.replace(f'file="{f}"', f'file="{rel}"')
    path.write_text(xml)


def build_pack(assets: list[str], name: str, out_root: Path = PACKS, processed: Path = ASSETS_PROCESSED,
               runs: Path = OUTPUTS / "ondeskgen" / "runs", raw_root: Path = ASSETS_RAW, zip_it: bool = True,
               log=print) -> dict:
    root = out_root / name
    if root.exists():
        shutil.rmtree(root)
    (root / "assets").mkdir(parents=True)
    (root / "scenes").mkdir()
    shutil.copyfile(ENGINE_CONFIG, root / "engine_config.json")
    index = []
    for aid in assets:
        src = processed / aid
        info = json.loads((src / "asset.json").read_text())
        dst = root / "assets" / aid
        shutil.copytree(src, dst)
        (dst / "physics.json").write_text(json.dumps(_physics(info), ensure_ascii=False, indent=2))
        # source chain + images + exports
        chain = _lux3d_chain(aid, raw_root, runs)
        srcd = dst / "source"
        srcd.mkdir()
        (srcd / "lux3d_tasks.json").write_text(json.dumps(chain, ensure_ascii=False, indent=2))
        exports = dst / "exports"
        for t in chain:
            d = REPO_ROOT / t["dir"]
            for f in t["outputs"]:
                p = d / f["file"]
                if not p.exists():
                    continue
                if t["feature"] == "concept_image":
                    shutil.copyfile(p, srcd / f"concept{p.suffix}")
                elif t["feature"] == "four_view":
                    (srcd / "views").mkdir(exist_ok=True)
                    shutil.copyfile(p, srcd / "views" / p.name)
                elif t["feature"] == "export" and f["type"] in EXPORT_NAMES:
                    exports.mkdir(exist_ok=True)
                    shutil.copyfile(p, exports / EXPORT_NAMES[f["type"]])
        # preview for the web viewer + single-asset scene
        s = _scene_spec([(aid, dst / info["mjcf"], [0.0, 0.0])])
        model = s.compile()
        from ondeskgen.preview import export_scene_glb
        export_scene_glb(model, dst / "preview.glb")
        _write_scene(s, root / "scenes" / f"{aid}.xml", root / "scenes")
        ph = json.loads((dst / "physics.json").read_text())
        index.append({
            "id": aid, "category": info["category"], "label": category(info["category"])["label"],
            "size_m": info["size_m"], "total_mass_kg": ph["total_mass_kg"],
            "collision": {"method": (info["collision"].get("fit_stats") or {}).get("method"),
                          "hulls": len(info["collision"]["proxies"])},
            "articulated": bool(info.get("articulation")),
            "joints": [j["name"] for j in (info.get("articulation") or {}).get("joints", [])],
            "lux3d_features": sorted({t["feature"] for t in chain}), "lux3d_tasks": [t["task_id"] for t in chain],
            "mjcf": f"assets/{aid}/{info['mjcf']}", "scene": f"scenes/{aid}.xml",
            "exports": sorted(p.name for p in exports.glob("*")) if exports.exists() else []})
        log(f"[pack] {aid}: {len(chain)} Lux3D tasks")
    # showroom: assets in a row along Y
    entries, y = [], 0.0
    for e in index:
        w = e["size_m"]["width_y"]
        entries.append((e["id"], root / "assets" / e["id"] / Path(e["mjcf"]).name, [0.0, y + w / 2]))
        y += w + 0.6
    s = _scene_spec(entries)
    s.compile()
    _write_scene(s, root / "scenes" / "showroom.xml", root / "scenes")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    pack = {"format": FORMAT, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine": {"version": engine_config()["version"], "config_hash": config_hash()},
            "simulator": {"mujoco": mujoco.__version__, "timestep": 0.005},
            "frame": {"units": "meter", "up_axis": "z", "origin": "bottom_center", "front": "+x"},
            "assets": index,
            "files": {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in files}}
    (root / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=2))
    (root / "README.md").write_text(_readme(pack))
    (root / "LICENSE_NOTES.md").write_text(LICENSE_NOTES)
    if zip_it:
        zp = out_root / f"{name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, f"{name}/{p.relative_to(root)}")
        pack["zip"] = {"path": _rel(zp), "bytes": zp.stat().st_size}
    log(f"[pack] {root} ({sum(p.stat().st_size for p in files) / 1e6:.0f} MB)")
    return pack


LICENSE_NOTES = """# 许可说明

- 家具 3D 模型由群核 Lux3D 生成（任务号见每个资产的 `source/lux3d_tasks.json`）。使用前请确认 Lux3D 的服务条款对生成内容的许可。
- 碰撞凸包、MJCF 和物理参数由 OnDeskGen 生成。
- 本资产包不包含任何机器人模型。
- 物理参数（质量、惯量、摩擦）是按类别估计的，见 `physics.json` 的 `estimate` 字段。
"""


def _readme(p: dict) -> str:
    L = [f"# {p['name']}（{p['format']}）", "",
         f"OnDeskGen 生成的 MuJoCo 仿真就绪家具资产包。引擎 {p['engine']['version']}，配置哈希 `{p['engine']['config_hash']}`，"
         f"MuJoCo {p['simulator']['mujoco']}，生成于 {p['created']}。", "",
         "坐标系：米制、Z 朝上、原点在底面中心、正面朝 +X。每个家具是一个自由刚体（freejoint），由 Lux3D 识别出的抽屉等部件"
         "通过滑动/转动关节挂在主体上。", "",
         "## 资产", "", "| id | 类别 | 尺寸 深×宽×高 (m) | 质量 (kg) | 碰撞凸包 | 可动部件 | 用到的 Lux3D 功能 |",
         "|---|---|---|---|---|---|---|"]
    for a in p["assets"]:
        s = a["size_m"]
        L.append(f"| `{a['id']}` | {a['label']} | {s['depth_x']:.2f}×{s['width_y']:.2f}×{s['height_z']:.2f} | "
                 f"{a['total_mass_kg']} | {a['collision']['hulls']} | {len(a['joints']) or '—'} | {'、'.join(a['lux3d_features'])} |")
    L += ["", "## 使用", "", "```bash", "python -m mujoco.viewer --mjcf=scenes/showroom.xml", "```", "",
          "在自己的场景里使用：用 `mujoco.MjSpec.from_file('assets/<id>/<id>.xml')` 读入片段（网格路径相对片段文件），"
          "再 `attach` 到场景里任意位置；家具和地板之间的摩擦建议用 `physics.json` 里的 `floor_friction`（显式 contact pair）。", "",
          "## 每个资产目录", "",
          "- `<id>.xml` MJCF 片段；`visual/` 视觉网格；`collision/` CoACD 凸包；`asset.json` 全部处理记录",
          "- `physics.json` 质量、惯量、摩擦和训练用的随机化范围（估计值）",
          "- `source/lux3d_tasks.json` Lux3D 任务链（每一步的任务号、请求、输入输出哈希）；概念图、四视图",
          "- `exports/` Lux3D 多格式导出（USDZ / OBJ / STL）", "- `preview.glb` 网页预览（视觉层 + 碰撞层）", ""]
    return "\n".join(L)

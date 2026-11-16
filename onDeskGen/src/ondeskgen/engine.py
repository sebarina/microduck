"""OnDeskGen engine: text or image(s) -> simulation-ready MuJoCo asset.

Lux3D chain (each step optional, all recorded):
  text        --concept_image--> image   (or --text_to_3d--> mesh directly)
  image       --four_view-------> 4 views (multi-view reconstruction)
  image(s)    --img_to_3d-------> textured metric mesh (ai_predict_size, PBR)
  mesh+image  --material_transfer> mesh with the photo's material
  mesh        --part_split------> parts      --articulation(prompt)--> URDF: moving parts + joints
  mesh        --multi_format_export--> USDZ / OBJ / STL
then: normalize -> visual -> CoACD collision (per link / per part) -> mass, inertia, friction
-> MJCF fragment (free body, joints) -> web preview; then asset pack (ondeskgen.pack) and static
MJCF for policy training (ondeskgen.train_export).

Every Lux3D task result is archived under assets/raw/<asset>/<feature>/<task>/ and reused
for an identical request (same feature + inputs), so re-running costs nothing. Before a
new task is created the account's server-side task count is checked against the credit cap
(configs/ondeskgen.json: lux3d.credit_cap).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ondeskgen.paths import ASSETS_PROCESSED, ASSETS_RAW, OUTPUTS
from ondeskgen.lux3d.features import (TaskRun, aholo_upload, ensure_model_url, image_data_url, make_client,
                                                run_feature, server_task_count)
from .config import category, config_hash, engine_config, spec_for

CREDIT_CAP = int(engine_config()["lux3d"].get("credit_cap", 1000))

RUNS = OUTPUTS / "ondeskgen" / "runs"


class CreditCapReached(RuntimeError):
    pass


@dataclass
class Request:
    category: str                                  # sofa | desk | (table, chair, cabinet, goal)
    text: str | None = None                        # description (text mode)
    images: list[str] = field(default_factory=list)  # 1 = single image, >1 = multi-view
    size: dict | None = None                       # optional real dimensions {width, depth, height} (m)
    concept_image: bool = True                     # text -> concept image -> image-to-3D
    four_view: bool = False                        # one image -> 4 views -> multi-view image-to-3D
    material_transfer: bool = False                # re-texture the mesh from the (first) image
    part_split: bool = True
    per_part_collision: bool = False               # CoACD per Lux3D part (285 vs 26 hulls on a sofa)
    articulation: bool | None = None               # None: when the category defines a prompt
    export: bool = True
    version: str | None = None                     # G1-Turbo (default from config) | G1
    name: str | None = None
    prompt: str | None = None                      # full prompt, bypasses the category template
    articulation_prompt: str | None = None         # overrides the category's articulation prompt
    material_prompt: str | None = None             # new material: Lux3D image+text -> reference image -> material transfer

    def key(self) -> str:
        img_h = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.images]
        d = {k: v for k, v in asdict(self).items() if k not in ("images", "name")}
        d["images"] = img_h
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def build_prompt(req: Request) -> str:
    if req.prompt:
        return req.prompt
    c = category(req.category)
    s = {**c["size"], **(req.size or {})}
    tpl = c.get("text_template") or "一个{A}" + c["label"] + "，宽约{W}厘米、深约{D}厘米、高约{H}厘米，单个物体，纯色背景"
    a = (req.text or "").strip().rstrip("。.")
    p = tpl.format(A="", W=round(s["width"] * 100), D=round(s["depth"] * 100), H=round(s["height"] * 100), EXTRA="")
    return f"{a}。{p}" if a else p


class Engine:
    def __init__(self, client=None, raw_root: Path = ASSETS_RAW, processed: Path = ASSETS_PROCESSED,
                 runs: Path = RUNS, log=print, events=None, dry_run: bool = False, fetch=None):
        self._client = client
        self.fetch = fetch
        self.raw_root, self.processed, self.runs = raw_root, processed, runs
        self.log = log
        self.ev = events or (lambda *a, **k: None)
        self.dry_run = dry_run
        self.cfg = engine_config()
        self.tasks: list[dict] = []

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.cfg["lux3d"]["region"])
        return self._client

    # ------------------------------------------------------------------ Lux3D step with cache + budget
    def step(self, asset: str, feature: str, args: dict, key: str, inputs: dict) -> TaskRun | None:
        # cache: an archived task of this feature with the same request key, under any asset
        own = sorted((self.raw_root / asset / feature).glob("*/meta.json"))
        for meta_p in own + sorted(set(self.raw_root.glob(f"*/{feature}/*/meta.json")) - set(own)):
            meta = json.loads(meta_p.read_text())
            if meta.get("inputs", {}).get("key") == key and meta.get("files"):
                run = TaskRun(feature, meta["task_id"], meta_p.parent, meta["files"], meta)
                self.tasks.append({"feature": feature, "task_id": run.task_id, "cached": True, "seconds": 0,
                                   "dir": str(meta_p.parent.relative_to(self.raw_root.parent.parent))})
                self.log(f"[odg] {feature}: cached task {run.task_id} ({meta_p.parent.parent.parent.name})")
                self.ev(feature, "done", task_id=run.task_id, cached=True, files=run.files)
                return run
        if self.dry_run:
            self.tasks.append({"feature": feature, "task_id": None, "would_create": True})
            self.ev(feature, "skipped", why="dry_run")
            return None
        used = server_task_count(self.client)
        if used >= CREDIT_CAP:
            raise CreditCapReached(f"Lux3D credit cap reached ({used}/{CREDIT_CAP})")
        self.ev(feature, "start", credits_used=used)
        t = time.time()
        run = run_feature(feature, asset, args, inputs={**inputs, "key": key}, client=self.client,
                          raw_root=self.raw_root, region=self.cfg["lux3d"]["region"], log=self.log,
                          **({"fetch": self.fetch} if self.fetch else {}))
        self.tasks.append({"feature": feature, "task_id": run.task_id, "cached": False,
                           "seconds": round(time.time() - t, 1),
                           "dir": str(run.run_dir.relative_to(self.raw_root.parent.parent))})
        self.ev(feature, "done", task_id=run.task_id, cached=False, seconds=round(time.time() - t, 1), files=run.files)
        return run

    # ------------------------------------------------------------------ main
    @staticmethod
    def asset_name(req: Request) -> str:
        return req.name or f"odg_{req.category}_{req.key()}"

    def generate(self, req: Request) -> dict:
        from ondeskgen.asset.build import process_asset

        t0 = time.time()
        L = self.cfg["lux3d"]
        cat = category(req.category)
        version = req.version or L["version"]
        name = self.asset_name(req)
        out = self.runs / name
        out.mkdir(parents=True, exist_ok=True)
        rec: dict = {"engine": self.cfg["version"], "config_hash": config_hash(), "asset": name,
                     "request": {**asdict(req), "images": [Path(p).name for p in req.images]},
                     "steps": {}, "status": "running"}
        opts = {"output_format": ["glb"], "ai_predict_size": L["ai_predict_size"], "enable_pbr": L["enable_pbr"]}
        if L.get("face_count"):
            opts["face_count"] = L["face_count"]
        prompt = build_prompt(req) if (req.text or req.prompt or not req.images) else None
        rec["prompt"] = prompt
        self.ev("request", "done", asset=name, prompt=prompt, category=req.category)

        def save():
            (out / "run.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str))

        # 1. images to reconstruct from
        images: list[tuple[str, dict]] = [image_data_url(Path(p)) for p in req.images]
        if not images and req.concept_image:
            r = self.step(name, "concept_image", {"prompt": prompt}, _h("concept", prompt), {"prompt": prompt})
            rec["steps"]["concept_image"] = _rs(r)
            if r is not None:
                images = [image_data_url(r.first(".png", ".jpg", ".webp"))]
        if len(images) == 1 and req.four_view:
            r = self.step(name, "four_view", {"img": images[0][0]}, _h("fourview", images[0][1]["sha256"]),
                          {"image": images[0][1]})
            rec["steps"]["four_view"] = _rs(r)
            if r is not None:
                views = [r.run_dir / f["file"] for f in r.files if f["type"] in ("png", "jpg", "webp")]
                images = [image_data_url(v) for v in views] or images
        save()
        # 2. geometry
        if images:
            args = {**({"img": images[0][0]} if len(images) == 1 else {"imgs": [u for u, _ in images]}),
                    "version": version, **opts}
            r = self.step(name, "img_to_3d", args, _h("img3d", version, opts, *[i["sha256"] for _, i in images]),
                          {"images": [i for _, i in images]})
            rec["steps"]["img_to_3d"] = _rs(r)
        else:
            r = self.step(name, "text_to_3d", {"prompt": prompt, "version": version, **opts},
                          _h("text3d", prompt, version, opts), {"prompt": prompt})
            rec["steps"]["text_to_3d"] = _rs(r)
        if r is None:
            rec["status"] = "dry_run"
            save()
            return rec
        glb = r.first(".glb")
        save()
        # 3. material: from the user's photo, or from a reference image Lux3D re-styles from the source image
        ref = None                                      # (path, sha256) of the material reference image
        if req.material_prompt and images:
            try:
                surl, srec = aholo_upload(Path(images[0][1]["path"]), region=L["region"], log=self.log)
                rr = self.step(name, "concept_image", {"img": surl, "prompt": req.material_prompt},
                               _h("restyle", images[0][1]["sha256"], req.material_prompt),
                               {"image": srec, "prompt": req.material_prompt, "role": "material_reference"})
                rec["steps"]["material_reference"] = _rs(rr)
                if rr is not None:
                    rp = rr.first(".png", ".jpg", ".webp")
                    ref = (rp, hashlib.sha256(Path(rp).read_bytes()).hexdigest())
            except Exception as e:  # noqa: BLE001
                rec["steps"]["material_reference"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.ev("concept_image", "error", error=str(e))
        elif req.material_transfer and req.images:      # the user's own photo (not a generated concept image)
            ref = (Path(images[0][1]["path"]), images[0][1]["sha256"])
        if ref is not None:
            murl, mrec = ensure_model_url(glb, log=self.log)
            try:
                # the reference image must live on Aholo's storage: official Asset upload
                iurl, irec = aholo_upload(Path(ref[0]), region=L["region"], log=self.log)
                mr = self.step(name, "material_transfer",
                               {"img": iurl, "mesh_url": murl, "version": L["material_transfer_version"],
                                "output_format": ["glb"], "ai_predict_size": True},
                               _h("material2", mrec["sha256"], ref[1]), {"mesh": mrec, "image": irec})
                rec["steps"]["material_transfer"] = _rs(mr)
                if mr is not None and mr.first(".glb"):
                    glb = mr.first(".glb")
            except Exception as e:  # noqa: BLE001 -- keep the untextured-by-photo mesh
                rec["steps"]["material_transfer"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                self.ev("material_transfer", "error", error=str(e))
        # 4. parts, articulation, export (from the geometry mesh)
        want_artic = req.articulation if req.articulation is not None else bool(cat.get("articulation_prompt"))
        parts_glb = artic_zip = None
        if req.part_split or want_artic or req.export:
            url, mrec = ensure_model_url(glb, log=self.log)
        if req.part_split or want_artic:
            pr = self.step(name, "part_split", {"glb_url": url}, _h("parts", mrec["sha256"]), {"model": mrec})
            rec["steps"]["part_split"] = _rs(pr)
            parts_glb = pr.first(".glb") if pr else None
            if want_artic and pr is not None:
                purl = next((f["url_path"] for f in pr.files if f["type"] == "glb"), None)
                ap = req.articulation_prompt or cat.get("articulation_prompt") or "可动部件可以打开和关闭"
                try:
                    ar = self.step(name, "articulation", {"glb_url": purl, "prompt": ap},
                                   _h("artic", purl, ap), {"model_url": purl, "prompt": ap,
                                                           "from_part_split": pr.task_id})
                    rec["steps"]["articulation"] = _rs(ar)
                    artic_zip = ar.first(".zip") if ar else None
                except Exception as e:  # noqa: BLE001 -- a static asset is still useful
                    rec["steps"]["articulation"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    self.ev("articulation", "error", error=str(e))
        if req.export:
            try:
                er = self.step(name, "export", {"model_url": url, "output_format": L["export_formats"]},
                               _h("export", mrec["sha256"], *L["export_formats"]), {"model": mrec})
                rec["steps"]["export"] = _rs(er)
            except Exception as e:  # noqa: BLE001
                rec["steps"]["export"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        save()
        # 5. simulation-ready asset
        self.ev("process", "start")
        t = time.time()
        spec = spec_for(name, req.category, prompt or f"{cat['label']}（图片生成）", req.size)
        d = process_asset(spec, glb, self.processed, lux3d_meta={"task_id": r.task_id, "version": version,
                                                                   "prompt": prompt, "region": L["region"]},
                          parts_glb=parts_glb if (req.per_part_collision and not artic_zip) else None,
                          articulation_zip=artic_zip)
        info = json.loads((d / "asset.json").read_text())
        try:                                    # web preview: visual + collision layers, joints closed
            from ondeskgen.preview import export_scene_glb
            from .pack import _scene_spec
            export_scene_glb(_scene_spec([(name, d / info["mjcf"], [0.0, 0.0])]).compile(), out / "preview.glb")
        except Exception as e:  # noqa: BLE001
            self.log(f"[odg] preview export failed: {e}")
        rec["processed"] = {"dir": str(d.relative_to(self.processed.parent.parent)), "seconds": round(time.time() - t, 1),
                            "hulls": len(info["collision"]["proxies"]), "mass_kg": info["physics"]["mass"],
                            "size_m": info["size_m"], "front": info["normalization"]["front"],
                            "articulated": bool(info.get("articulation"))}
        self.ev("process", "done", **rec["processed"])
        save()
        rec["tasks"] = self.tasks
        rec["credits_spent"] = sum(1 for t in self.tasks if not t.get("cached") and t.get("task_id"))
        rec["seconds"] = round(time.time() - t0, 1)
        rec["status"] = "done"
        save()
        write_marker(rec, d)
        self.ev("run", "done", outcome="done", asset=name, credits_spent=rec["credits_spent"])
        return rec


MARKER = "ondeskgen.json"


def write_marker(rec: dict, asset_dir: Path) -> dict:
    """Small summary next to the processed asset so other tools (the web UI, packs) can find OnDeskGen
    assets without reading the run record."""
    req = rec.get("request") or {}
    m = {"engine": rec.get("engine"), "config_hash": rec.get("config_hash"), "asset": rec["asset"],
         "category": req.get("category"), "text": req.get("text"), "prompt": rec.get("prompt"),
         "image_input": bool(req.get("images")),
         "articulated": bool((rec.get("processed") or {}).get("articulated")),
         "lux3d_features": [k for k, v in (rec.get("steps") or {}).items() if v and v.get("ok", True)],
         "run": f"outputs/ondeskgen/runs/{rec['asset']}/run.json"}
    (Path(asset_dir) / MARKER).write_text(json.dumps(m, ensure_ascii=False, indent=2))
    return m


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]


def _rs(run: TaskRun | None) -> dict | None:
    if run is None:
        return None
    return {"ok": True, "task_id": run.task_id, "dir": str(run.run_dir), "files": run.files,
            "seconds": run.meta.get("seconds")}

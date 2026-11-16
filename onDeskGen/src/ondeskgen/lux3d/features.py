"""Every Lux3D feature behind one task runner (OnDeskGen).

  feature             SDK resource                     input                         typical output
  text_to_3d          client.text_to_3d                prompt (+ img URL)            glb
  img_to_3d           client.img_to_3d                 one image or up to 32 views   glb
  four_view           client.image_to_four_view        image and/or prompt           4 images
  concept_image       client.multimodal_to_image       prompt and/or image           image
  material_transfer   client.material_transfer         image + mesh URL              glb
  part_split          client.part_split                glb URL                       per-part output
  articulation        client.articulation_animation    glb URL + prompt              articulated output
  export              client.multi_format_export       model URL + formats           usdz/obj/fbx/stl/3mf

Each run is archived immediately (download links expire) under
``assets/raw/<asset>/<feature>/<task_id>/`` with ``meta.json``: feature, inputs
(prompts, image hashes, input-model sha256 and the URL handed to Lux3D),
options, timestamps and sha256/size/detected type of every output. Images given
as local files go to Lux3D as data URLs (no upload); models that a feature needs
by URL are PUT to your file service first (ONDESKGEN_UPLOAD_URL; cached by content hash).

Credits: every task created counts once. ``server_task_count`` asks the Lux3D
task list (authoritative), ``credits_used`` in pipeline.assets counts archives.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ondeskgen.paths import ASSETS_RAW, OUTPUTS
import urllib.request

import importlib.metadata


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sdk_version() -> str:
    try:
        return importlib.metadata.version("manycore-aholo-sdk-lux3d")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

from .upload import USER_AGENT, PutFn, _http_put, sniff_image, upload_bytes

def http_get(url: str, timeout_s: float = 300.0) -> bytes:
    """Download with the project's User-Agent (some CDNs reject Python's default)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (https URLs from the API)
        return resp.read()


def expand_outputs(result: dict) -> list[str]:
    """Output URLs of a finished task. Some features return a JSON list of URLs in one output
    (four_view: 4 images); placeholders ("null", NOT_REQUESTED) are dropped."""
    urls = []
    for out in result.get("outputs") or []:
        c = (out.get("content") or "").strip()
        if not c or c in ("null", "NOT_REQUESTED"):
            continue
        if c.startswith("["):
            try:
                urls += [u for u in json.loads(c) if isinstance(u, str) and u.startswith("http")]
                continue
            except ValueError:
                pass
        if c.startswith(("http://", "https://")):
            urls.append(c)
    return urls


FEATURES = ("text_to_3d", "img_to_3d", "four_view", "concept_image", "material_transfer",
            "part_split", "articulation", "export")
MODEL_UPLOAD_CACHE = OUTPUTS / "uploads" / "models.json"
AHOLO_UPLOAD_CACHE = OUTPUTS / "uploads" / "aholo.json"
_LOCK = threading.Lock()


def sniff_output(data: bytes, url_path: str = "") -> str:
    """File type of a Lux3D output: URL extension when it has one, else magic bytes."""
    ext = Path(urllib.parse.urlsplit(url_path).path).suffix.lower()
    if ext in (".glb", ".gltf", ".zip", ".ply", ".usdz", ".obj", ".fbx", ".stl", ".3mf", ".png", ".jpg",
               ".jpeg", ".webp", ".json", ".mp4", ".bvh"):
        return ".jpg" if ext == ".jpeg" else ext
    if data[:4] == b"glTF":
        return ".glb"
    img = sniff_image(data)
    if img:
        return img
    if data[:2] == b"PK":
        return ".zip"
    if data[:3] == b"ply":
        return ".ply"
    if data[:18] == b"Kaydara FBX Binary":
        return ".fbx"
    head = data[:64].lstrip()
    if head[:1] in (b"{", b"["):
        return ".json"
    if head[:5] == b"solid":
        return ".stl"
    return ".bin"


@dataclass
class TaskRun:
    feature: str
    task_id: int
    run_dir: Path
    files: list[dict]
    meta: dict = field(default_factory=dict)

    def first(self, *exts: str) -> Path | None:
        for f in self.files:
            if Path(f["file"]).suffix in exts:
                return self.run_dir / f["file"]
        return None


def image_data_url(path: Path) -> tuple[str, dict]:
    data = Path(path).read_bytes()
    ext = sniff_image(data)
    if ext is None:
        raise ValueError(f"{path}: not a JPEG/PNG/WebP image")
    mime = mimetypes.types_map.get(ext, "image/jpeg")
    rec = {"source": Path(path).name, "sha256": sha256_bytes(data), "bytes": len(data), "content_type": mime,
           "path": str(Path(path))}
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", rec


def aholo_upload(path: Path, *, region: str = "cn", client=None, log=print) -> tuple[str, dict]:
    """URL on Aholo's own storage (official Asset upload, OUS), cached by sha256.

    Lux3D material transfer could not fetch reference images from a generic file service in our
    tests (the same official example image failed from there and worked from Aholo's CDN), so images for
    that feature go through the official upload the API documentation prescribes. The remote
    file name is content-addressed (the local name is not sent)."""
    data = Path(path).read_bytes()
    h = sha256_bytes(data)
    rec = {"source": Path(path).name, "sha256": h, "bytes": len(data)}
    with _LOCK:
        known = json.loads(AHOLO_UPLOAD_CACHE.read_text()) if AHOLO_UPLOAD_CACHE.exists() else {}
    if h in known:
        log(f"[aholo-upload] {Path(path).name}: already uploaded ({h[:8]})")
        return known[h]["url"], {**rec, "url": known[h]["url"], "host": "aholo"}
    if client is None:
        from manycore.aholo_sdk_asset import create_asset_client
        from manycore.aholo_sdk_core import AholoClientConfig
        client = create_asset_client(AholoClientConfig(region=region))
    ext = sniff_image(data) or Path(path).suffix.lower() or ".bin"
    res = client.upload_bytes(data, filename=f"ondeskgen_{h[:16]}{ext}")
    log(f"[aholo-upload] {Path(path).name} ({len(data) / 1e6:.2f} MB) -> {res.url}")
    with _LOCK:
        known = json.loads(AHOLO_UPLOAD_CACHE.read_text()) if AHOLO_UPLOAD_CACHE.exists() else {}
        known[h] = {"url": res.url, "md5": res.md5, "file_name": f"ondeskgen_{h[:16]}{ext}", "region": region}
        AHOLO_UPLOAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        AHOLO_UPLOAD_CACHE.write_text(json.dumps(known, indent=2) + "\n")
    return res.url, {**rec, "url": res.url, "host": "aholo"}


def ensure_model_url(path: Path, *, put: PutFn | None = None, log=print) -> tuple[str, dict]:
    """URL for a local model file (GLB/zip) on your file service (ONDESKGEN_UPLOAD_URL), cached by sha256."""
    data = Path(path).read_bytes()
    h = sha256_bytes(data)
    ext = Path(path).suffix.lower() or ".glb"
    rec = {"source": str(path), "sha256": h, "bytes": len(data)}
    with _LOCK:
        known = json.loads(MODEL_UPLOAD_CACHE.read_text()) if MODEL_UPLOAD_CACHE.exists() else {}
    if h in known:
        log(f"[upload] {Path(path).name}: already uploaded ({h[:8]})")
        return known[h]["url"], {**rec, "url": known[h]["url"]}
    ctype = {".glb": "model/gltf-binary", ".zip": "application/zip"}.get(ext, "application/octet-stream")
    url = upload_bytes(data, f"ondeskgen_{h[:16]}{ext}", ctype, put=put or _http_put)
    log(f"[upload] {Path(path).name} ({len(data) / 1e6:.1f} MB) -> {url}")
    with _LOCK:
        known = json.loads(MODEL_UPLOAD_CACHE.read_text()) if MODEL_UPLOAD_CACHE.exists() else {}
        known[h] = {"url": url, "file_name": f"ondeskgen_{h[:16]}{ext}"}
        MODEL_UPLOAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MODEL_UPLOAD_CACHE.write_text(json.dumps(known, indent=2) + "\n")
    return url, {**rec, "url": url}


def make_client(region: str = "cn"):
    from manycore.aholo_sdk_core import AholoClientConfig
    from manycore.aholo_sdk_lux3d import create_lux3d_client
    return create_lux3d_client(AholoClientConfig(region=region))


def server_task_count(client: Any = None, region: str = "cn") -> int:
    """Tasks the Lux3D account has created (authoritative credit count)."""
    client = client or make_client(region)
    return int(client.tasks.list(page=1, page_size=1).get("total") or 0)


_CREATE = {
    "text_to_3d": lambda c, a: c.text_to_3d.create(**a),
    "img_to_3d": lambda c, a: c.img_to_3d.create(**a),
    "four_view": lambda c, a: c.image_to_four_view.create(**a),
    "concept_image": lambda c, a: c.multimodal_to_image.create(**a),
    "material_transfer": lambda c, a: c.material_transfer.create(**a),
    "part_split": lambda c, a: c.part_split.create(**a),
    "articulation": lambda c, a: c.articulation_animation.create(**a),
    "export": lambda c, a: c.multi_format_export.create(**a),
}


def _redact(args: dict) -> dict:
    """Request args for meta.json: data URLs replaced by their hash (they are megabytes)."""
    def one(v):
        if isinstance(v, str) and v.startswith("data:"):
            return f"<data url sha256={hashlib.sha256(v.encode()).hexdigest()[:16]}>"
        if isinstance(v, (list, tuple)):
            return [one(x) for x in v]
        return v
    return {k: one(v) for k, v in args.items()}


def run_feature(feature: str, asset: str, args: dict, *, inputs: dict | None = None, client: Any = None,
                raw_root: Path = ASSETS_RAW, region: str = "cn", fetch: Callable[[str], bytes] = http_get,
                timeout_ms: int = 900_000, log=print, task_id: int | None = None) -> TaskRun:
    """Create one Lux3D task of ``feature`` (costs one task), wait, archive every output.
    ``task_id`` resumes an existing task instead (free: only polls and downloads)."""
    if feature not in _CREATE:
        raise ValueError(f"unknown feature {feature!r}; one of {FEATURES}")
    client = client or make_client(region)
    t0 = time.time()
    if task_id is None:
        task_id = int(_CREATE[feature](client, args))
        log(f"[lux3d] {asset}/{feature}: task {task_id} created, waiting...")
    else:
        log(f"[lux3d] {asset}/{feature}: resuming task {task_id}")
    result = client.tasks.wait_for(task_id, timeout_ms=timeout_ms)
    t1 = time.time()
    run_dir = raw_root / asset / feature / str(task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    files = []
    echoed = {v for v in args.values() if isinstance(v, str) and v.startswith("http")}
    for i, url in enumerate(expand_outputs(result)):
        if url in echoed:                     # some features list their input model again
            continue
        data = fetch(url)
        parsed = urllib.parse.urlsplit(url)
        path_only = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        ext = sniff_output(data, parsed.path)
        name = f"output_{i}{ext}"
        (run_dir / name).write_bytes(data)
        files.append({"file": name, "sha256": sha256_bytes(data), "bytes": len(data), "type": ext.lstrip("."),
                      "url_path": path_only})
    meta = {"asset": asset, "feature": feature, "task_id": task_id, "status": result.get("status"),
            "biz_id": result.get("bizId"), "region": region, "request": _redact(args), "inputs": inputs or {},
            "sdk": {"package": "manycore-aholo-sdk-lux3d", "version": sdk_version()},
            "created_at": t0, "finished_at": t1, "seconds": round(t1 - t0, 1), "files": files}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"[lux3d] {asset}/{feature}: task {task_id} done in {t1 - t0:.0f} s, {len(files)} output(s): "
        + ", ".join(f"{f['file']} ({f['bytes'] / 1e6:.2f} MB)" for f in files))
    if not files:
        raise RuntimeError(f"{feature} task {task_id} finished without outputs")
    return TaskRun(feature, task_id, run_dir, files, meta)

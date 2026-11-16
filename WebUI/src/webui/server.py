"""OnDeskGen web UI: three tabs — demo video, asset engine, sim training.

  ondeskgen-webui                      # http://127.0.0.1:8765  (or: uv run python -m webui.server)

Standard library only; binds to 127.0.0.1. The asset engine runs ``ondeskgen generate`` as a
subprocess (one at a time) that writes <ONDESKGEN_HOME>/outputs/ondeskgen/runs/<asset>/events.jsonl;
the page polls it. Files are served only from OnDeskGen's outputs/ and assets/processed/, this
project's media/ and the static directory, with path-traversal checks. The Lux3D API key is never
read into the page: the subprocess inherits the environment, the page only learns whether a key is set.

  media/demo.mp4 + media/demo_chapters.json      the demo video tab ({"duration_s", "chapters": [{t, title, kicker, lines}]})
  configs/sim_train.json                          the sim-training tab's text (pipeline, cloud stages, runs)
  media/sim_train/checks/<name>.txt               optional: training-side check outputs (last line "exit 0" = passed)
  media/sim_train/slots/<label>/slot.json         optional: policy results on the assets, with videos
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ondeskgen.paths import ASSETS_PROCESSED, ASSETS_RAW, OUTPUTS, REPO_ROOT as DATA_ROOT

STATIC = Path(__file__).parent / "static"
WEBUI_ROOT = Path(os.environ.get("WEBUI_HOME") or Path(__file__).resolve().parents[2]).resolve()
MEDIA = WEBUI_ROOT / "media"
SIM_TRAIN = MEDIA / "sim_train"
ODG_RUNS = OUTPUTS / "ondeskgen" / "runs"
TRAIN_EXPORT = OUTPUTS / "train_export" / "ondeskgen_static_v1"
ODG_FLAGS = {"concept_image": ("--no-concept", False), "four_view": ("--four-view", True),
             "material_transfer": ("--material", True), "part_split": ("--no-parts", False),
             "articulation": ("--no-articulation", False), "export": ("--no-export", False)}
UPLOADS = OUTPUTS / "ui_uploads"
DATA_ROOTS = (OUTPUTS.resolve(), ASSETS_PROCESSED.resolve())
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_IMAGE = 20 * 1024 * 1024
ASSET_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
FEATURE_CN = {"concept_image": "文生图（概念图）", "four_view": "单图生四视图", "img_to_3d": "图生 3D",
              "text_to_3d": "文生 3D", "material_transfer": "材质迁移", "part_split": "部件拆分",
              "articulation": "铰接（URDF）", "export": "多格式导出"}
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("font/woff2", ".woff2")


def rel(p: Path | str | None) -> str | None:
    """URL for a file we may serve: /files/... (OnDeskGen data) or /media/... (this project), else None."""
    if not p:
        return None
    p = Path(p)
    p = (DATA_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.exists():
        return None
    if any(p.is_relative_to(r) for r in DATA_ROOTS):
        return "/files/" + str(p.relative_to(DATA_ROOT))
    if p.is_relative_to(MEDIA.resolve()):
        return "/media/" + str(p.relative_to(MEDIA.resolve()))
    return None


# --------------------------------------------------------------------------- asset engine runner
class OdgRunner:
    """At most one ``ondeskgen generate`` subprocess."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.run_id: str | None = None

    def running(self) -> str | None:
        with self.lock:
            return self.run_id if self.proc is not None and self.proc.poll() is None else None

    def start(self, req: dict) -> str:
        from ondeskgen.config import engine_config
        from ondeskgen.engine import Engine, Request
        cats = engine_config()["categories"]
        cat = req.get("category")
        if cat not in cats:
            raise ValueError(f"类别只能是 {sorted(cats)}")
        mode = req.get("mode", "text")
        text = str(req.get("text") or "").strip()
        if mode == "text" and not (1 <= len(text) <= 200):
            raise ValueError("文本模式请输入 1–200 字的描述")
        imgs = req.get("images") or []
        if mode == "image" and not (1 <= len(imgs) <= 8):
            raise ValueError("图片模式请给 1–8 张图片")
        if not req.get("confirm_credits"):
            raise ValueError("会调用 Lux3D（可能花积分），请先勾选确认")
        opts = req.get("options") or {}
        size = req.get("size") or None
        if size:
            size = {k: float(v) for k, v in size.items() if k in ("width", "depth", "height") and v not in (None, "")}
            if any(not 0.2 <= v <= 3.0 for v in size.values()):
                raise ValueError("尺寸要在 0.2–3.0 m 之间")
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError(f"已有资产在生成：{self.run_id}")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            paths = [_save_image(f"odg_{stamp}", f"img{i}", im) for i, im in enumerate(imgs)] if mode == "image" else []
            r = Request(category=cat, text=text or None, images=paths, size=size or None,
                        concept_image=bool(opts.get("concept_image", True)), four_view=bool(opts.get("four_view", False)),
                        material_transfer=bool(opts.get("material_transfer", False)),
                        part_split=bool(opts.get("part_split", True)),
                        articulation=None if opts.get("articulation", True) else False,
                        export=bool(opts.get("export", True)))
            name = Engine.asset_name(r)
            args = [sys.executable, "-m", "ondeskgen.cli", "generate", cat, "--name", name]
            if text:
                args += ["--text", text]
            for p in paths:
                args += ["--image", p]
            if size and len(size) == 3:
                args += ["--size", str(size["width"]), str(size["depth"]), str(size["height"])]
            for k, (flag, when_true) in ODG_FLAGS.items():
                on = bool(opts.get(k, k in ("concept_image", "part_split", "articulation", "export")))
                if on == when_true:
                    args.append(flag)
            out = ODG_RUNS / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "events.jsonl").write_text("")
            log = open(out / "stdout.log", "w")
            self.proc = subprocess.Popen(args, cwd=WEBUI_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
            self.run_id = name
            return name

    def stop(self, run_id: str) -> bool:
        with self.lock:
            if self.run_id == run_id and self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                return True
            return False


def _save_image(run_id: str, stem: str, img: dict) -> str:
    if isinstance(img, str):
        raise ValueError("请直接上传图片文件")
    ctype = img.get("type", "")
    if ctype not in IMAGE_TYPES:
        raise ValueError("图片只支持 JPEG / PNG / WebP")
    try:
        data = base64.b64decode(img.get("data", ""), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("图片数据损坏") from e
    if not data or len(data) > MAX_IMAGE:
        raise ValueError("图片要在 20 MB 以内")
    d = UPLOADS / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stem}{IMAGE_TYPES[ctype]}"
    p.write_bytes(data)
    return str(p)


ODG = OdgRunner()
_CREDITS: dict = {"t": 0.0, "n": None}


def credits_now() -> tuple[int, str]:
    """Lux3D tasks created by this account: the server's count (cached 60 s) when a key is set,
    else the number of archived tasks."""
    if os.environ.get("AHOLO_API_KEY"):
        if time.time() - _CREDITS["t"] > 60:
            try:
                from ondeskgen.lux3d.features import server_task_count
                _CREDITS.update(t=time.time(), n=server_task_count())
            except Exception:  # noqa: BLE001
                _CREDITS.update(t=time.time(), n=None)
        if _CREDITS["n"] is not None:
            return int(_CREDITS["n"]), "server"
    return sum(1 for _ in ASSETS_RAW.glob("*/*/*/meta.json")), "local"


# --------------------------------------------------------------------------- data
def _odg_files(run: dict) -> dict:
    """Images and exports produced by the Lux3D steps of an engine run (served under /files)."""
    out: dict = {"concept": None, "views": [], "exports": [], "inputs": []}
    for t in run.get("tasks", []):
        d = DATA_ROOT / t["dir"] if t.get("dir") else None
        if d is None or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        for f in meta.get("files", []):
            u = rel(d / f["file"])
            if t["feature"] == "concept_image" and out["concept"] is None:
                out["concept"] = u
            elif t["feature"] == "four_view":
                out["views"].append(u)
            elif t["feature"] == "export":
                out["exports"].append({"type": f["type"], "url": u, "bytes": f["bytes"]})
    return out


def odg_summary(name: str) -> dict | None:
    p = ODG_RUNS / name / "run.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    pr = r.get("processed") or {}
    return {"asset": name, "category": r.get("request", {}).get("category"), "status": r.get("status"),
            "text": r.get("request", {}).get("text") or r.get("prompt"),
            "mode": "image" if r.get("request", {}).get("images") else "text",
            "features": sorted({t["feature"] for t in r.get("tasks", [])}),
            "articulated": pr.get("articulated"), "hulls": pr.get("hulls"), "size_m": pr.get("size_m"),
            "credits_spent": r.get("credits_spent"), "seconds": r.get("seconds"), "mtime": p.stat().st_mtime,
            "concept": _odg_files(r)["concept"]}


def odg_detail(name: str) -> dict | None:
    p = ODG_RUNS / name / "run.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    out = {"run": r, "files": _odg_files(r), "summary": odg_summary(name)}
    a = ASSETS_PROCESSED / name / "asset.json"
    if a.exists():
        info = json.loads(a.read_text())
        out["asset"] = {k: info.get(k) for k in ("category", "size_m", "normalization", "physics", "articulation")}
        out["asset"]["collision"] = {"hulls": len(info["collision"]["proxies"]),
                                     "fit_stats": info["collision"].get("fit_stats"), "bounds": info["collision"]["bounds"]}
        out["asset"]["visual"] = {k: info["visual"].get(k) for k in ("faces_raw", "faces")}
    out["preview_glb"] = rel(ODG_RUNS / name / "preview.glb")
    out["feature_cn"] = FEATURE_CN
    return out


def odg_events(name: str, start: int) -> dict:
    p = ODG_RUNS / name / "events.jsonl"
    lines = p.read_text().splitlines() if p.exists() else []
    so = ODG_RUNS / name / "stdout.log"
    tail = "\n".join(so.read_text(errors="replace").splitlines()[-30:]) if so.exists() else ""
    alive = ODG.running() == name
    done = any('"stage": "run"' in ln and ('"done"' in ln or '"error"' in ln) for ln in lines)
    return {"events": [json.loads(x) for x in lines[start:] if x.strip()], "next": len(lines),
            "running": alive, "finished": done or not alive, "stdout_tail": tail}


def demo_info() -> dict:
    """Demo video tab: media/demo.mp4 and its chapter marks (media/demo_chapters.json)."""
    ch = MEDIA / "demo_chapters.json"
    d = json.loads(ch.read_text()) if ch.exists() else {"chapters": []}
    v = MEDIA / "demo.mp4"
    return {"video": rel(v) if v.exists() else None, "duration_s": d.get("duration_s"), "chapters": d.get("chapters", []),
            "generated": time.strftime("%Y-%m-%d %H:%M", time.localtime(v.stat().st_mtime)) if v.exists() else None}


def train_info() -> dict:
    """Sim-training tab: configs/sim_train.json + OnDeskGen's static training export + optional media."""
    cfg = json.loads((WEBUI_ROOT / "configs" / "sim_train.json").read_text())
    out = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "hidden_slots"}
    man = TRAIN_EXPORT / "box" / "manifest.json"
    out["assets"] = []
    if man.exists():
        for r in json.loads(man.read_text()):
            if r["category"] not in ("sofa", "desk"):
                continue
            d = TRAIN_EXPORT / "box" / r["asset"]
            out["assets"].append({"asset": r["asset"], "category": r["category"], "size_m": r["size_m"],
                                  "boxes": r["collision_geoms"],
                                  "preview": rel(d / "preview.png") if (d / "preview.png").exists() else None})
    out["checks"] = {}
    for f in sorted((SIM_TRAIN / "checks").glob("*.txt")):
        txt = f.read_text()
        out["checks"][f.stem] = {"text": txt[-3500:], "passed": txt.rstrip().endswith("exit 0"),
                                 "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))}
    slots = []
    hidden = set(cfg.get("hidden_slots", []))
    for f in sorted((SIM_TRAIN / "slots").glob("*/slot.json")):
        if f.parent.name in hidden:
            continue
        sl = json.loads(f.read_text())
        for x in sl.get("floor", []):
            x["video_url"] = rel(f.parent / x["video"]) if x.get("video") else None
        for x in sl.get("surface", []):
            x["video_url"] = rel(f.parent / "surface" / x["video"]) if x.get("video") else None
        slots.append(sl)
    out["slots"] = sorted(slots, key=lambda s: (s.get("kind") != "trained", s.get("label", "")))
    return out


def status() -> dict:
    from ondeskgen.engine import CREDIT_CAP
    n, src = credits_now()
    return {"key_set": bool(os.environ.get("AHOLO_API_KEY")), "region": os.environ.get("AHOLO_REGION", "cn"),
            "credits_used": n, "credits_source": src, "credit_cap": CREDIT_CAP, "odg_running": ODG.running()}


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "ondeskgen-webui/1.0"

    def log_message(self, fmt, *args):  # quieter console
        if os.environ.get("WEBUI_VERBOSE"):
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _file(self, path: Path):
        if not path.is_file():
            return self._err(404, "not found")
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and (m := re.match(r"bytes=(\d*)-(\d*)$", rng.strip())):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):
                start = max(0, size - int(m.group(2)))
            if start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        path = unquote(u.path)
        q = parse_qs(u.query)
        pages = {"/": "demo.html", "/demo": "demo.html", "/assets": "assets.html", "/train": "train.html"}
        if path in pages:
            return self._file(STATIC / pages[path])
        if path == "/api/status":
            return self._json(status())
        if path == "/api/demo":
            return self._json(demo_info())
        if path == "/api/train":
            return self._json(train_info())
        if path == "/api/odg/assets":
            rows = [s for d in (ODG_RUNS.iterdir() if ODG_RUNS.exists() else []) if (s := odg_summary(d.name))]
            return self._json(sorted(rows, key=lambda r: -r["mtime"]))
        if m := re.match(r"^/api/odg/asset/([^/]+)(/events)?$", path):
            rid = m.group(1)
            if not ASSET_ID.match(rid):
                return self._err(400, "bad asset id")
            if m.group(2):
                return self._json(odg_events(rid, int(q.get("from", ["0"])[0])))
            d = odg_detail(rid)
            return self._json(d) if d else self._err(404, "no such asset")
        if path.startswith("/static/"):
            p = (STATIC / path[len("/static/"):]).resolve()
            return self._file(p) if p.is_relative_to(STATIC.resolve()) else self._err(403, "forbidden")
        if path.startswith("/files/"):
            p = (DATA_ROOT / path[len("/files/"):]).resolve()
            return self._file(p) if any(p.is_relative_to(r) for r in DATA_ROOTS) else self._err(403, "forbidden")
        if path.startswith("/media/"):
            p = (MEDIA / path[len("/media/"):]).resolve()
            return self._file(p) if p.is_relative_to(MEDIA.resolve()) else self._err(403, "forbidden")
        return self._err(404, "not found")

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        if n > 3 * MAX_IMAGE:
            return self._err(413, "请求太大")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._err(400, "bad json")
        if path == "/api/odg/run":
            try:
                return self._json({"asset": ODG.start(body)})
            except ValueError as e:
                return self._err(400, str(e))
            except RuntimeError as e:
                return self._err(409, str(e))
        if m := re.match(r"^/api/odg/asset/([^/]+)/stop$", path):
            return self._json({"stopped": ODG.stop(m.group(1))})
        if path == "/api/odg/pack":
            names = [n for n in (body.get("assets") or []) if isinstance(n, str) and ASSET_ID.match(n)]
            if not names or any(not (ASSETS_PROCESSED / n / "asset.json").exists() for n in names):
                return self._err(400, "请选择已生成完成的资产")
            pname = f"ondeskgen_pack_{time.strftime('%Y%m%d_%H%M%S')}"
            from ondeskgen.pack import build_pack
            p = build_pack(names, pname, log=lambda *_: None)
            return self._json({"name": pname, "zip": rel(DATA_ROOT / p["zip"]["path"]), "bytes": p["zip"]["bytes"],
                               "readme": rel(OUTPUTS / "packs" / pname / "README.md"), "assets": len(p["assets"])})
        return self._err(404, "not found")


def main() -> None:
    ap = argparse.ArgumentParser(description="OnDeskGen web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"OnDeskGen web UI: http://{a.host}:{a.port}  (Ctrl+C 退出)")
    print(f"OnDeskGen 数据目录: {DATA_ROOT}")
    print(f"Lux3D key: {'已设置' if os.environ.get('AHOLO_API_KEY') else '未设置（只能浏览已有资产）'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

"""Reference images for Lux3D "text + image" generation.

Lux3D's text-to-3D accepts an optional reference image, but only as a URL.
Local files are uploaded first to a file service you provide (env ONDESKGEN_UPLOAD_URL): an HTTP PUT
endpoint taking ``?fileName=...`` that answers ``{"code": 200, "data": {"fileUrl": "https://..."}}``
(e.g. a small Cloudflare Worker in front of an R2 bucket). Lux3D's part split, articulation and export
also need the model by URL, so meshes go through the same service.

  - an http(s) URL is passed through unchanged (never downloaded here)
  - a local file is type-checked by its magic bytes, hashed, and uploaded once:
    the sha256 -> fileUrl mapping is cached, and the uploaded file name is
    derived from the hash (the local file name never leaves the machine)
Uploading is lazy (``ensure_uploaded``) so dry runs and cache hits send nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ondeskgen.paths import OUTPUTS

def upload_url() -> str:
    url = os.environ.get("ONDESKGEN_UPLOAD_URL")
    if not url:
        raise UploadError("set ONDESKGEN_UPLOAD_URL: an HTTP PUT file endpoint (?fileName=...) answering "
                          '{"code": 200, "data": {"fileUrl": "https://..."}} (needed to hand files to Lux3D by URL)')
    return url
UPLOAD_CACHE = OUTPUTS / "uploads" / "uploads.json"
USER_AGENT = "ondeskgen/1.0 (asset pipeline)"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
CONTENT_TYPES = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_LOCK = threading.Lock()

# put(url, data, content_type) -> response body
PutFn = Callable[[str, bytes, str], bytes]


class ImageError(ValueError):
    """Unusable reference image (missing, not an image, too large)."""


class UploadError(RuntimeError):
    """The upload service refused the file or answered unexpectedly."""


def sniff_image(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


@dataclass
class ImageInput:
    source: str                 # what the user gave: a URL, or the local file's base name
    kind: str                   # "url" | "file"
    url: str | None = None      # URL handed to Lux3D (None until a local file is uploaded)
    sha256: str | None = None   # local files only
    ext: str | None = None
    data: bytes | None = None   # local files only (archived next to the generated model)

    @property
    def key(self) -> str:
        """Identity used for cache names: content hash for files, the URL itself for URLs."""
        return self.sha256 if self.sha256 else hashlib.sha256(self.source.encode()).hexdigest()

    def record(self) -> dict:
        return {"kind": self.kind, "source": self.source, "url": self.url, "sha256": self.sha256,
                "content_type": CONTENT_TYPES.get(self.ext or "")}


def load_image(image: str | Path) -> ImageInput:
    s = str(image)
    if s.startswith(("http://", "https://")):
        return ImageInput(source=s, kind="url", url=s)
    p = Path(s).expanduser()
    if not p.is_file():
        raise ImageError(f"reference image not found: {p}")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"{p.name}: {len(data) / 1e6:.1f} MB, limit {MAX_IMAGE_BYTES / 1e6:.0f} MB")
    ext = sniff_image(data)
    if ext is None:
        raise ImageError(f"{p.name}: not a JPEG/PNG/WebP image")
    return ImageInput(source=p.name, kind="file", sha256=hashlib.sha256(data).hexdigest(), ext=ext, data=data)


def _http_put(url: str, data: bytes, content_type: str, timeout_s: float = 60.0) -> bytes:
    # Cloudflare in front of the service rejects Python's default User-Agent (error 1010).
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": content_type, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (configured https endpoint)
        return resp.read()


def upload_bytes(data: bytes, file_name: str, content_type: str, *, endpoint: str | None = None,
                 put: PutFn = _http_put) -> str:
    endpoint = endpoint or upload_url()
    url = f"{endpoint}?{urllib.parse.urlencode({'fileName': file_name})}"
    body = put(url, data, content_type)
    try:
        resp = json.loads(body)
    except (ValueError, TypeError) as e:
        raise UploadError(f"upload service returned non-JSON: {body[:200]!r}") from e
    file_url = (resp.get("data") or {}).get("fileUrl") if isinstance(resp, dict) else None
    if not isinstance(resp, dict) or resp.get("code") != 200 or not file_url:
        raise UploadError(f"upload refused: {str(resp)[:200]}")
    if not str(file_url).startswith(("http://", "https://")):
        raise UploadError(f"upload returned a non-URL fileUrl: {str(file_url)[:200]}")
    return str(file_url)


def ensure_uploaded(img: ImageInput, *, endpoint: str | None = None, put: PutFn | None = None,
                    cache: Path | None = None, log=print) -> ImageInput:
    """Give a local image a URL (cached by content hash). URLs are returned as they are."""
    if img.url:
        return img
    endpoint = endpoint or upload_url()
    cache = cache or UPLOAD_CACHE
    with _LOCK:
        known = json.loads(cache.read_text()) if cache.exists() else {}
    hit = known.get(img.sha256)
    if hit and hit.get("endpoint") == endpoint:
        img.url = hit["url"]
        log(f"[upload] {img.source}: already uploaded ({img.sha256[:8]})")
        return img
    name = f"ondeskgen_{img.sha256[:16]}{img.ext}"
    img.url = upload_bytes(img.data, name, CONTENT_TYPES[img.ext], endpoint=endpoint, put=put or _http_put)
    log(f"[upload] {img.source} -> {img.url}")
    with _LOCK:
        known = json.loads(cache.read_text()) if cache.exists() else {}
        known[img.sha256] = {"url": img.url, "endpoint": endpoint, "file_name": name}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=2) + "\n")
    return img

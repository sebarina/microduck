"""Engine configuration (configs/ondeskgen.json): load, hash, per-category AssetSpec."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from ondeskgen.paths import PROJECT_ROOT, AssetSpec

ENGINE_CONFIG = Path(os.environ.get("ONDESKGEN_CONFIG") or PROJECT_ROOT / "configs" / "ondeskgen.json")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def engine_config(path: Path = ENGINE_CONFIG) -> dict:
    return _load(str(path))


def config_hash(path: Path = ENGINE_CONFIG) -> str:
    """sha256 of the canonical JSON (key order / whitespace independent), first 16 hex chars."""
    canon = json.dumps(engine_config(path), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def category(cat: str, path: Path = ENGINE_CONFIG) -> dict:
    cats = engine_config(path)["categories"]
    if cat not in cats:
        raise ValueError(f"unknown category {cat!r}; one of {sorted(cats)}")
    return cats[cat]


def spec_for(name: str, cat: str, prompt: str, size: dict | None = None, path: Path = ENGINE_CONFIG) -> AssetSpec:
    """AssetSpec for a generated asset of ``cat`` (optional user size overrides the default target)."""
    c = category(cat, path)
    s = {**c["size"], **(size or {})}
    extra = {"size_range": c.get("size_range"), "fallback_scale_by": c.get("fallback_scale_by", "width")}
    if size:                       # the user gave real dimensions: scale to them, do not trust prediction
        scale_by = c.get("fallback_scale_by", "width") if c["scale_by"] == "lux3d" else c["scale_by"]
        scale_by = next((k for k in ("width", "depth", "height") if k in size), scale_by)
    else:
        scale_by = c["scale_by"]
    return AssetSpec(name=name, category=cat, prompt=prompt, width=s["width"], depth=s["depth"], height=s["height"],
                     scale_by=scale_by, front_rule=c.get("front_rule", "none"), extra=extra)

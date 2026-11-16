"""Where OnDeskGen keeps its data.

  ONDESKGEN_HOME    data root (default: this project's directory):
                      assets/raw/<asset>/<feature>/<task>/   every Lux3D task, archived (inputs, outputs, hashes)
                      assets/processed/<asset>/              simulation-ready asset (MJCF fragment, meshes, hulls)
                      outputs/ondeskgen/runs/<asset>/        run record, events, preview
                      outputs/packs/, outputs/train_export/  asset packs, static MJCF for training
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("ONDESKGEN_HOME") or PROJECT_ROOT).resolve()
ASSETS_RAW = REPO_ROOT / "assets" / "raw"
ASSETS_PROCESSED = REPO_ROOT / "assets" / "processed"
OUTPUTS = REPO_ROOT / "outputs"

CATEGORIES = ("sofa", "desk", "table", "chair", "cabinet", "goal")


@dataclass
class AssetSpec:
    name: str
    category: str
    prompt: str
    width: float   # along Y (lateral)
    depth: float   # along X (front-back)
    height: float  # along Z
    scale_by: str = "width"
    front_rule: str = "none"
    yaw_deg: float | None = None
    raw_run: str | None = None
    extra: dict = field(default_factory=dict)

    def target(self, axis: str) -> float:
        return {"width": self.width, "depth": self.depth, "height": self.height}[axis]

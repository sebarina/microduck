"""Stage events of a run: appended to <run>/events.jsonl (the web UI polls this file)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class Events:
    def __init__(self, path: Path, callback=None):
        self.path, self.callback, self.t0 = Path(path), callback, time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def __call__(self, stage: str, status: str, **data) -> None:
        e = {"t": round(time.time() - self.t0, 2), "stage": stage, "status": status, **data}
        with self.path.open("a") as f:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        if self.callback:
            self.callback(e)

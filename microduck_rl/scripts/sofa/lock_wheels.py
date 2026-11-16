#!/usr/bin/env python3
"""Emit the exact wheel list that uv.lock resolves for linux/x86_64/cp312.

Reads uv.lock (which records every candidate wheel's URL, sha256 and size),
evaluates each package's resolution markers against the training environment
and picks the single wheel uv would install. Output: one
``<filename> <sha256hex> <url>`` line per wheel, ready for curl -C - plus
``sha256sum -c``.

Usage:
    python3 lock_wheels.py [path/to/uv.lock] > wheels_linux_x86_64.txt
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

# The environment we resolve for: ModelScope notebook, linux x86_64, CPython 3.12.
ENV = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
    "extra": "",
}

# Python tags we accept, best first (used when several wheels match a package).
PY_TAG_PREF = ("cp312", "cp311", "abi3", "py3", "py2.py3")


def wheel_python_rank(filename: str) -> int:
    """Rank a wheel filename by how well its interpreter tag fits cp312."""
    try:
        py_tag = filename.split("-")[2]  # name-version-pytag-abitag-plat.whl
    except IndexError:
        return len(PY_TAG_PREF)
    for rank, pref in enumerate(PY_TAG_PREF):
        if py_tag == pref:
            return rank
    return len(PY_TAG_PREF)


def wheel_matches(filename: str) -> bool:
    """True if this wheel filename targets linux x86_64."""
    if "none-any" in filename:  # pure-python wheel
        return True
    if "win" in filename or "macosx" in filename or "musllinux" in filename:
        return False
    if "linux" not in filename and "manylinux" not in filename:
        return False
    if "x86_64" not in filename:
        return False
    if any(arch in filename for arch in ("aarch64", "i686", "armv7l", "ppc64le", "s390x", "riscv64")):
        return False
    return True


def main() -> int:
    lock_path = Path(sys.argv[1] if len(sys.argv) > 1 else "uv.lock")
    with open(lock_path, "rb") as fh:
        lock = tomllib.load(fh)

    from packaging.markers import Marker  # noqa: PLC0415 -- only needed here

    sdists: list[str] = []
    rows: list[tuple[str, str, str]] = []
    total = 0

    for pkg in lock["package"]:
        name = pkg["name"]
        variants = pkg if "version" in pkg else [pkg]
        if "version" not in pkg:  # safety; uv.lock packages always carry version
            continue

        # A package may appear with several versions gated by resolution-markers
        # (e.g. nvidia-cublas 12.8.4.1 x86_64 vs 12.9.1.4 aarch64). Keep the
        # variant(s) whose markers apply to our target environment.
        applicable = []
        markers = pkg.get("resolution-markers")
        if markers:
            if any(Marker(m).evaluate(ENV) for m in markers):
                applicable.append(pkg)
        else:
            applicable.append(pkg)

        for variant in applicable:
            version = variant["version"]
            source = variant.get("source", {})
            kind = source.get("registry") or source.get("url") or ""
            if "git" in str(source):
                sdists.append(f"{name} {version} (git dep -- fetched by uv, not curl)")
                continue
            candidates = [
                w for w in variant.get("wheels", []) if wheel_matches(w["url"].rsplit("/", 1)[1])
            ]
            if not candidates:
                if variant.get("sdist"):
                    sdists.append(f"{name} {version} (no matching wheel, sdist only)")
                continue
            best = min(candidates, key=lambda w: wheel_python_rank(w["url"].rsplit("/", 1)[1]))
            fname = best["url"].rsplit("/", 1)[1]
            sha = best["hash"].removeprefix("sha256:")
            rows.append((fname, sha, best["url"]))
            total += best.get("size", 0)

    for fname, sha, url in sorted(rows):
        print(f"{fname} {sha} {url}")

    print(
        f"# {len(rows)} wheels, {total / 1e9:.2f} GB total "
        f"({len(sdists)} non-wheel items: {'; '.join(sdists) if sdists else 'none'})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

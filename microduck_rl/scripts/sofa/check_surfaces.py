#!/usr/bin/env python3
"""Measure every walkable surface the duck could be trained on, and decide
whether ONE circle policy can serve all of them.

Why this exists
---------------
The sofa task was built on geometry that was read off a bounding box instead of
measured, and four runs paid for it (README §16). Adding a second furniture type
multiplies that risk: a desk is a different height, a different contact regime,
and a different radius cap, and any of those can quietly invalidate a shared
command range.

This script answers the only question that matters before writing a multi-
surface env cfg: **does one ``CIRCLE_RADIUS`` range fit every surface?**

It is CUDA-free and mjlab-free on purpose -- it ray-casts the compiled assets
with plain ``mujoco``, so it runs on the laptop in ~5 s:

    .venv/bin/python scripts/sofa/check_surfaces.py
    .venv/bin/python scripts/sofa/check_surfaces.py --surfaces sofa,desk_mv

Exit code 0 = every declared constant matches its asset AND the shared radius
range fits all of them. 1 = something would have silently broken a run.

What it does NOT do: it does not touch the reward, the spawn maths or the
contact sensor. Those live in the env cfg and are gated by build_check.py.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mjlab_microduck.robot.microduck import desk_constants as D  # noqa: E402
from mjlab_microduck.robot.microduck import sofa_constants as S  # noqa: E402

# Grid pitch for the ray sweep. 1 cm over a 1.4 x 0.9 m area is ~12k rays, i.e.
# about a second per surface -- fine for a gate, and 10x finer than the
# GEOM_TOL the declarations are checked against.
GRID = 0.01
# A ray "lands on the surface" if it is within this of the measured top. The
# desks have their next surface 40 mm below the top plate (the drawer fronts),
# the sofa has bolsters 350 mm above the seat, so 8 mm cannot confuse them.
TOP_TOL = 0.008
# Declared rectangle must agree with the measured one to this. Declarations are
# the measured rectangle pulled 5 mm inward, so 15 mm is the honest bound.
GEOM_TOL = 0.015
# Height declaration tolerance.
Z_TOL = 0.005


@dataclass
class Surface:
    """One thing the duck can stand on, with everything the check needs."""

    name: str
    spec_fn: object
    top_z: float
    pocket_x: tuple[float, float]
    pocket_y: tuple[float, float]
    spawn_x: tuple[float, float]
    spawn_y: tuple[float, float]
    spawn_yaw: tuple[float, float]
    note: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        return (
            0.5 * (self.pocket_x[0] + self.pocket_x[1]),
            0.5 * (self.pocket_y[0] + self.pocket_y[1]),
        )

    @property
    def half(self) -> tuple[float, float]:
        return (
            0.5 * (self.pocket_x[1] - self.pocket_x[0]),
            0.5 * (self.pocket_y[1] - self.pocket_y[0]),
        )

    @property
    def radius_cap(self) -> float:
        return min(self.half) - S.BODY_REACH


def surfaces() -> dict[str, Surface]:
    """The registry. Sofa first: it is the binding constraint on every range."""
    out = {
        "sofa": Surface(
            name="sofa seat (odg_sofa_concept_a)",
            spec_fn=S.get_sofa_spec,
            top_z=S.SEAT_TOP_Z,
            pocket_x=S.SEAT_FREE_X,
            pocket_y=S.SEAT_FREE_Y,
            spawn_x=S.SPAWN_X,
            spawn_y=S.SPAWN_Y,
            spawn_yaw=S.SPAWN_YAW,
            note="soft contact (solref 0.06); the pocket is a T, this is its core",
        )
    }
    for key, desk in D.DESKS.items():
        short = "desk_mv" if key == "multiview" else f"desk_{key}"
        out[short] = Surface(
            name=f"desk {desk.key}",
            spec_fn=lambda d=desk: D.get_desk_spec(d),
            top_z=desk.top_z,
            pocket_x=desk.pocket_x,
            pocket_y=desk.pocket_y,
            spawn_x=(
                desk.centre[0] + D.SPAWN_X_JITTER[0],
                desk.centre[0] + D.SPAWN_X_JITTER[1],
            ),
            spawn_y=(
                desk.centre[1] + D.SPAWN_Y_JITTER[0],
                desk.centre[1] + D.SPAWN_Y_JITTER[1],
            ),
            spawn_yaw=D.SPAWN_YAW,
            note=f"hard contact (solref 0.02); {desk.n_col_geoms} collision boxes",
        )
    return out


class Prober:
    """Downward ray-caster over one compiled asset, in that asset's own frame."""

    def __init__(self, spec_fn):
        self.model = spec_fn().compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        # Collision geoms only. The desks carry 6-9 visual meshes with
        # contype=0; letting a ray hit one reports the underside of a drawer
        # front as if it were the tabletop.
        groups = np.zeros(6, dtype=np.uint8)
        for g in range(self.model.ngeom):
            if self.model.geom_contype[g] > 0 or self.model.geom_conaffinity[g] > 0:
                groups[self.model.geom_group[g]] = 1
        if not groups.any():
            groups[:] = 1
        self.groups = groups
        self.n_rays = 0

    def z(self, x: float, y: float, z0: float = 2.0) -> float | None:
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model, self.data, np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]),
            self.groups, 1, -1, gid,
        )
        self.n_rays += 1
        return None if dist < 0 else z0 - dist


def max_inscribed_rect(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle in a boolean grid.

    Histogram-per-row + monotone stack, i.e. O(h*w). This is the whole reason a
    desk can be measured safely: its bounding box and its walkable rectangle are
    NOT the same thing (the drawers desk loses a corner to a chamfer), and
    taking the bounding box is exactly the bug that cost four sofa runs.
    """
    best = (0, 0, 0, 0, 0)  # area, i0, i1, j0, j1
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    for i in range(h):
        heights = np.where(mask[i], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for j in range(w + 1):
            cur = heights[j] if j < w else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                if hh * (j - s) > best[0]:
                    best = (hh * (j - s), i - hh + 1, i, s, j - 1)
                start = s
            stack.append((start, cur))
    return best


def measure(pr: Prober, surf: Surface) -> dict:
    """Top height + largest inscribed rectangle, from rays only."""
    cx, cy = surf.centre
    top = pr.z(cx, cy)
    if top is None:
        raise RuntimeError(f"{surf.name}: no surface under its declared centre")

    half_x = max(abs(surf.pocket_x[0]), surf.pocket_x[1]) + 0.05
    half_y = max(abs(surf.pocket_y[0]), surf.pocket_y[1]) + 0.05
    xs = np.round(np.arange(cx - half_x, cx + half_x + 1e-9, GRID), 4)
    ys = np.round(np.arange(cy - half_y, cy + half_y + 1e-9, GRID), 4)
    Z = np.full((len(xs), len(ys)), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = pr.z(float(x), float(y))
            if z is not None:
                Z[i, j] = z

    mask = np.abs(Z - top) < TOP_TOL
    _, i0, i1, j0, j1 = max_inscribed_rect(mask)
    return {
        "top": top,
        "x": (float(xs[i0]), float(xs[i1])),
        "y": (float(ys[j0]), float(ys[j1])),
        "cells": int(mask.sum()),
    }


def read_shared_radius(default=(0.08, 0.12)) -> tuple[tuple[float, float], str]:
    """CIRCLE_RADIUS straight out of the circle cfg, without importing mjlab."""
    cfg = REPO / "src/mjlab_microduck/tasks/microduck_circle_sofa_env_cfg.py"
    try:
        tree = ast.parse(cfg.read_text())
        for node in tree.body:
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "CIRCLE_RADIUS":
                val = ast.literal_eval(node.value)
                return (float(val[0]), float(val[1])), f"{cfg.name}"
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CIRCLE_RADIUS":
                val = ast.literal_eval(node.value)
                return (float(val[0]), float(val[1])), f"{cfg.name}"
    except Exception as exc:  # noqa: BLE001 - a gate must not die on a parse quirk
        print(f"warn: could not read CIRCLE_RADIUS from the cfg ({exc}); "
              f"using {default}")
    return default, "fallback default (KEEP IN SYNC WITH THE CFG)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surfaces", default="sofa,desk_mv,desk_drawers",
                    help="comma-separated: " + ",".join(surfaces()))
    ap.add_argument("--no-spawn-sweep", action="store_true",
                    help="skip the spawn-footprint rays (the slow part)")
    args = ap.parse_args()

    reg = surfaces()
    wanted = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in reg]
    if unknown:
        print(f"unknown surface(s) {unknown}; have {sorted(reg)}")
        return 1

    (r_lo, r_hi), r_src = read_shared_radius()
    print(f"shared orbit range CIRCLE_RADIUS = ({r_lo}, {r_hi})  [{r_src}]")
    print(f"body reach BODY_REACH = {S.BODY_REACH} m "
          f"(worst footprint corner, stand pose)\n")

    bad = 0
    rows = []
    for key in wanted:
        surf = reg[key]
        pr = Prober(surf.spec_fn)
        m = measure(pr, surf)

        # --- height ---------------------------------------------------------
        ok_z = abs(m["top"] - surf.top_z) <= Z_TOL
        print(f"[{key}] {surf.name}   ({surf.note})")
        print(f"  {'ok ' if ok_z else 'FAIL'}: top z measured {m['top']:.4f} "
              f"declared {surf.top_z:.4f}  ({m['cells']} cells at that height, "
              f"{pr.n_rays} rays)")
        bad += not ok_z

        # --- inscribed rectangle --------------------------------------------
        for axis in ("x", "y"):
            measured, declared = m[axis], getattr(surf, f"pocket_{axis}")
            err = max(abs(measured[0] - declared[0]), abs(measured[1] - declared[1]))
            ok = err <= GEOM_TOL
            print(f"  {'ok ' if ok else 'FAIL'}: {axis} measured "
                  f"[{measured[0]:+.3f}, {measured[1]:+.3f}] "
                  f"({measured[1] - measured[0]:.3f} m) vs declared "
                  f"[{declared[0]:+.3f}, {declared[1]:+.3f}]  err {err * 1000:.0f} mm")
            bad += not ok

        # --- spawn footprint --------------------------------------------------
        if not args.no_spawn_sweep:
            foot = [(fx, fy) for fx in S.BODY_FOOT_X for fy in S.BODY_FOOT_Y]
            off, n = [], 0
            for i in range(5):
                sx = surf.spawn_x[0] + (surf.spawn_x[1] - surf.spawn_x[0]) * i / 4
                for j in range(5):
                    sy = surf.spawn_y[0] + (surf.spawn_y[1] - surf.spawn_y[0]) * j / 4
                    for k in range(5):
                        yaw = surf.spawn_yaw[0] + (surf.spawn_yaw[1] - surf.spawn_yaw[0]) * k / 4
                        c, s = np.cos(yaw), np.sin(yaw)
                        for fx, fy in foot:
                            px, py = sx + c * fx - s * fy, sy + s * fx + c * fy
                            n += 1
                            z = pr.z(px, py)
                            if z is None or abs(z - m["top"]) > TOP_TOL:
                                off.append((round(sx, 3), round(sy, 3), round(px, 3), round(py, 3), z))
            ok = not off
            print(f"  {'ok ' if ok else 'FAIL'}: spawn footprint 5 yaws x 5 x 5 "
                  f"positions x 4 corners = {n} rays, {len(off)} off the surface"
                  + ("" if ok else f"; first miss {off[0]}"))
            bad += not ok

        # --- the orbit this surface can carry -------------------------------
        cap = surf.radius_cap
        ok = r_hi <= cap
        print(f"  {'ok ' if ok else 'FAIL'}: orbit needs r {r_hi} + reach "
              f"{S.BODY_REACH} = {r_hi + S.BODY_REACH:.3f} m vs half-extent "
              f"{min(surf.half):.3f} m  ->  cap {cap:.3f} m, "
              f"slack {1000 * (cap - r_hi):.0f} mm")
        bad += not ok
        print()
        rows.append((key, m["top"], surf.half, cap))

    # --- summary -------------------------------------------------------------
    print("=" * 78)
    print(f"{'surface':<26}{'top z':>8}{'rect (m)':>16}{'r_cap':>8}{'slack vs ' + str(r_hi):>14}")
    for key, top, half, cap in rows:
        print(f"{key:<26}{top:>8.4f}{f'{2*half[0]:.2f} x {2*half[1]:.2f}':>16}"
              f"{cap:>8.3f}{1000 * (cap - r_hi):>11.0f} mm")
    worst_half = min(min(r[2]) for r in rows)
    print("=" * 78)
    print(f"worst surface half-extent {worst_half:.3f} m "
          f"-> a shared r <= {worst_half - S.BODY_REACH:.3f} m is walkable everywhere")

    if bad:
        print(f"\n{bad} check(s) FAILED -- do not write a multi-surface cfg on top of these numbers")
        return 1
    print("\nall surfaces agree with their declarations; one orbit range fits all of them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

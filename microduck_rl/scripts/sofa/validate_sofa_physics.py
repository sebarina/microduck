"""CPU gate for the sofa task: does the real duck actually rest on the seat?

Runs without the CUDA training stack (plain ``mujoco``), on the *real* robot
model + the *real* sofa asset, merged the same way mjlab merges them
(``MjSpec.attach`` under a name prefix). It answers the questions that would
otherwise only surface after a paid GPU run:

1. do the sofa's retuned collision geoms compile (group 0, solref 0.06)?
2. does the duck actually stand on the cushion, or fall through it?
3. does the mocap sofa leak contacts against the static ground plane? Those
   would burn the per-world ``nconmax`` budget and starve the robot's contacts.
4. does anything NaN when the duck collapses onto the seat (every bad episode
   starts this way)?

The robot is spawned exactly like the env cfg does (STAND pose, seat top +
standing height) with zero torque, so it folds and falls onto the cushion. The
walk model only carries collision geoms for feet/legs, so the *trunk* sinking
into the cushion during a fall is expected and harmless - the episode is
terminated on bad orientation long before that matters. What is asserted here
is contact quality, not pose.

Usage (no torch/warp needed):
    python scripts/sofa/validate_sofa_physics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from mjlab_microduck.robot.microduck.sofa_constants import (  # noqa: E402
    SEAT_TOP_Z,
    get_sofa_spec,
)

SCENE_XML = _REPO / "src" / "mjlab_microduck" / "robot" / "microduck" / "scene_walk.xml"
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def build(solref=None) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    sofa = get_sofa_spec()
    if solref is not None:
        for geom in sofa.geoms:
            if "_col_" in geom.name:
                geom.solref = list(solref)
    # Same attach path mjlab uses for entities: a fresh frame on the worldbody,
    # then the child spec under a name prefix.
    frame = spec.worldbody.add_frame()
    spec.attach(child=sofa, prefix="sofa/", frame=frame)
    return spec.compile()


def _sofa_pairs(model, data):
    pairs = {}
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
        pairs[tuple(sorted((n1, n2)))] = pairs.get(tuple(sorted((n1, n2))), 0) + 1
    return pairs


def run(model: mujoco.MjModel, seconds: float = 3.0):
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[2] += SEAT_TOP_Z       # root: seat top + standing height
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    z0 = float(data.xpos[trunk][2])
    min_z = z0
    min_dist = 0.0                    # most negative contact.dist = deepest penetration
    foot_on_seat = False
    foot_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS}

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return None
        min_z = min(min_z, float(data.xpos[trunk][2]))
        for i in range(data.ncon):
            min_dist = min(min_dist, float(data.contact[i].dist))
            c = data.contact[i]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if {c.geom1, c.geom2} & foot_geoms and ("sofa/" in n1 or "sofa/" in n2):
                foot_on_seat = True

    return dict(
        z0=z0,
        z_end=float(data.xpos[trunk][2]),
        min_z=min_z,
        min_dist=min_dist,
        foot_on_seat=foot_on_seat,
        pairs=_sofa_pairs(model, data),
        ncon=data.ncon,
        qpos=data.qpos.copy(),
    )


def main() -> int:
    print(f"mujoco {mujoco.__version__}, seat top {SEAT_TOP_Z:.3f} m")
    failures: list[str] = []

    model = build()
    sofa_geoms = [
        g
        for g in range(model.ngeom)
        if "_col_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        and "sofa/" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
    ]
    solref = model.geom_solref[sofa_geoms[0]]
    groups = {int(model.geom_group[g]) for g in sofa_geoms}
    print(f"sofa collision geoms: {len(sofa_geoms)} | solref {solref.round(3).tolist()} | groups {groups}")
    if len(sofa_geoms) != 26:
        failures.append(f"expected 26 sofa collision geoms, got {len(sofa_geoms)}")
    if not np.allclose(solref, [0.06, 1.0]):
        failures.append(f"seat solref not softened: {solref}")
    if groups != {0}:
        failures.append(f"seat geoms must be in group 0 for the ray sensors, got {groups}")

    # --- Collapse onto the cushion -------------------------------------------
    print("\n[collapse] zero torque, legs fold onto the seat:")
    for label, sr in (("foam (task cfg)   ", None), ("hard (asset default)", [0.02, 1.0])):
        res = run(build(sr), seconds=3.0)
        if res is None:
            failures.append(f"{label}: NaN during collapse")
            print(f"  {label}: NaN")
            continue
        leaked = {
            k: v
            for k, v in res["pairs"].items()
            if any(x.startswith("sofa/") for x in k) and any(x in ("floor", "terrain") for x in k)
        }
        print(
            f"  {label}: trunk min {res['min_z']:.3f}, end {res['z_end']:.3f} x,y=({res['qpos'][0]:+.2f},{res['qpos'][1]:+.2f}) "
            f"| deepest contact {res['min_dist'] * 1000:+.1f} mm | ncon {res['ncon']}"
        )
        print(f"    sofa pairs at end: {sorted(set(res['pairs']))}")
        if leaked:
            failures.append(f"{label}: sofa leaked contacts against the ground plane: {leaked}")
        if not res["foot_on_seat"]:
            failures.append(f"{label}: feet never contacted the seat")
        if res["min_dist"] < -0.04:
            failures.append(f"{label}: collapse drove penetration past 40 mm ({res['min_dist'] * 1000:.1f} mm)")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS: cushion contact model, seat group and contact budget all check out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

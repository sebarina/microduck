"""Sofa prop for the sofa-walking task: asset path + MuJoCo-side tuning.

Pure ``mujoco`` only (no mjlab import), so the contact model can be validated on
a CPU-only machine without the CUDA training stack:

    python scripts/sofa/validate_sofa_physics.py

The asset is the ODG `ondeskgen_static_v1` "box" variant: every collision geom
is an oriented box (fastest for MuJoCo Warp), 26 of them, plus 6 visual meshes.
Static, no freejoint, metres, Z-up, origin at floor centre.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco

_SOFA_DIR: Path = Path(__file__).resolve().parent / "assets" / "sofa"
SOFA_XML: Path = _SOFA_DIR / "odg_sofa_concept_a.xml"
assert SOFA_XML.exists(), f"sofa asset not found: {SOFA_XML}"

# ---------------------------------------------------------------------------
# Seat geometry -- MEASURED, not read off the asset's bounding box
# ---------------------------------------------------------------------------
# SEAT_TOP_Z: the two seat halves are col_0 / col_1, half-size z 0.2006 / 0.2015
# at centres 0.3156 / 0.3166 -> tops 0.5162 / 0.5181 m. The higher one is used
# so the robot never spawns intersecting the lower half.
#
# THE USABLE SURFACE IS NOT THE SEAT BOXES' BOUNDING BOX, and it is not even a
# rectangle. `scripts/sofa/build_check.py` ray-casts the compiled asset downward
# and fits the largest axis-aligned rectangle INSIDE the free region; the free
# region itself is a **T**:
#
#   x (depth) ->  -0.03 ...... +0.32 .................. +0.50
#   |y| <= 0.39        [======= deep pocket =======]         free
#   |y| 0.39..0.83                        [== front ledge ==] free
#   |y| > 0.83                                        BLOCKED (armrest)
#   x < -0.03                                BLOCKED (backrest, top 0.980)
#
#   deep pocket   x [-0.03, +0.50]  y [-0.41, +0.39]   0.530 x 0.800 m
#   front ledge   x [+0.32, +0.50]  y [-0.83, +0.82]   0.180 x 1.655 m
#   pocket centre (+0.235, -0.010)   <- note: NOT (0.225, 0), the pocket is
#                                      10 mm off-centre because the right-hand
#                                      bolster is the one with the cupholder
#
# What blocks the shoulders of the T: the backrest cushions col_11 / col_12
# (front face at x = -0.007 / -0.026, top 0.9797) own the whole rear strip, and
# the two seat bolsters col_3 / col_4 (tops 0.868 / 0.870) cut the pocket's
# length from 1.66 m down to 0.80 m. The "1.56 m long axis" this task was
# designed around therefore only exists on a 0.18 m wide ledge -- 2 cm more than
# the duck is wide. It cannot be walked.
#
# Getting this wrong is silent, which is why it survived four runs. The previous
# constants declared "x in [-0.45, 0.17]" (the seat boxes' AABB, minus nothing)
# with the spawn at x = -0.13: 0.35 m behind the pocket, INSIDE the backrest
# solid. Every sofa run up to 2026-11-14 -- straight v1, v2_bounded, v3_gait,
# circle c1 -- spawned the duck half-buried in a 0.98 m wall and then measured
# "the duck cannot walk" while it was in fact wedged. build_check.py now
# re-derives this from the asset on every build (section 0) and fails if a
# spawn point, a command segment, or an orbit leaves the measured pocket.
SEAT_TOP_Z: float = 0.518
SEAT_FREE_X = (-0.03, 0.50)   # inscribed rectangle, valid for SEAT_FREE_Y
SEAT_FREE_Y = (-0.41, 0.39)
SEAT_LEDGE_X = (0.32, 0.50)   # front ledge: documented, NOT walkable
SEAT_LEDGE_Y = (-0.83, 0.82)
SEAT_CENTER_XY = (0.235, -0.010)  # midpoint of the inscribed rectangle

# The duck's collision footprint at the stand pose, measured off
# robot_groundcontact.xml (81 mesh geoms, so the AABB of every mesh under its
# stand-pose transform). It is neither square nor centred on the root:
#     x in [-0.058, +0.093]   y in [-0.074, +0.088]   (0.151 x 0.162 m)
# The worst corner sits 0.128 m from the root in XY, and that -- not a half
# extent -- is what constrains an orbit, because the heading turns with the
# circle while the mesh offset does not.
BODY_FOOT_X = (-0.058, 0.093)
BODY_FOOT_Y = (-0.074, 0.088)
BODY_REACH: float = 0.128     # max |corner - root| in XY, stand pose
BODY_HALF_LEN: float = 0.093  # forward reach, along the travel direction

# Spawn. Centred on the deep pocket, because that is the only region with room
# in both directions: +-0.10 in y leaves 0.39 - 0.10 - 0.093 = 0.197 m of runway
# to the nearer bolster, which is one full straight command segment (see
# CMD_HOLD_S in microduck_velocity_sofa_env_cfg.py, and section 0d of
# build_check.py which asserts the inequality).
SPAWN_X = (0.16, 0.29)    # across the seat depth (pocket centre x = +0.235)
SPAWN_Y = (-0.10, 0.10)   # along the seat length (pocket centre y = -0.010)
SPAWN_YAW = (math.pi / 2 - 0.25, math.pi / 2 + 0.25)  # face +y, along the sofa

# Foam cushion contact model. 0.02 s is the asset's furniture-hard default;
# 0.06 s makes the seat spring 3x softer (roughly 1.5-2 cm sink under 800 g).
SOFA_SOLREF = [0.06, 1.0]
SOFA_SOLIMP = [0.85, 0.95, 0.002, 0.5, 2.0]
SOFA_FRICTION = 1.0     # cloth blend: mu ~ 1.0
SOFA_TORSIONAL = 0.01
SOFA_ROLLING = 0.001


def get_sofa_spec() -> mujoco.MjSpec:
    """Load the sofa MJCF and retune its collision geoms for a soft cushion."""
    spec = mujoco.MjSpec.from_file(str(SOFA_XML))
    tuned = 0
    for geom in spec.geoms:
        if "_col_" not in geom.name:
            continue  # visual meshes (contype=0, density=0) stay untouched
        geom.group = 0                 # walkable surface -> visible to ray sensors
        geom.condim = 3
        geom.friction = [SOFA_FRICTION, SOFA_TORSIONAL, SOFA_ROLLING]
        geom.solref = list(SOFA_SOLREF)
        geom.solimp = list(SOFA_SOLIMP)
        geom.contype = 1
        geom.conaffinity = 1
        tuned += 1
    if tuned == 0:
        raise RuntimeError(f"no '_col_' geoms found in {SOFA_XML}")
    return spec

"""Desk props for the on-desk walking task: asset paths + MEASURED geometry.

Companion to ``sofa_constants.py``. Same conventions:

  * asset = the ODG ``ondeskgen_static_v1`` "box" variant -- every collision geom
    is an oriented box, static, no freejoint, metres, Z up, origin at FLOOR
    centre, front facing +X;
  * pure ``mujoco`` here (no mjlab import) so the geometry can be re-measured on
    a CPU-only machine;
  * every number below was obtained by ray-casting the compiled asset, NOT by
    reading the XML's bounding boxes -- that mistake cost four sofa runs
    (see the geometry block in ``sofa_constants.py`` and README §16).

Why a desk at all
-----------------
The sofa pocket is 0.530 x 0.800 m and its walkable radius is capped at
0.137 m by the 0.53 m depth. Both desks are BIGGER and cleaner flat surfaces:

    surface             top z    inscribed rectangle      centre            r_cap
    sofa seat           0.518    0.530 x 0.800 m          (+0.235, -0.010)  0.137
    desk_multiview_a    0.7566   0.590 x 1.210 m          ( 0.000,  0.000)  0.167
    desk_drawers_a      0.7860   0.600 x 1.190 m          (-0.015,  0.000)  0.172

where ``r_cap = min(half_x, half_y) - BODY_REACH`` (0.128 m). Every surface
therefore accepts the sofa's ``CIRCLE_RADIUS = (0.08, 0.12)`` with 17-57 mm of
slack, which is what makes ONE circle policy on several surfaces possible at
all: the command range does not have to change per surface.

The other difference is CONTACT, and it is deliberate: the sofa is tuned soft
(solref 0.06, ~1.5-2 cm of sink) while a desk keeps the asset's furniture-hard
default (solref 0.02). Training on both is contact-domain randomisation -- the
gait cannot overfit to foam squish. Measured consequence: the same policy that
stands still on the cushion drifts its heading +0.21 rad/s there and
+0.35 rad/s on a rigid desk, i.e. the soft surface was MASKING a yaw bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from mjlab_microduck.robot.microduck.sofa_constants import BODY_REACH

_DESK_DIR: Path = Path(__file__).resolve().parent / "assets" / "desk"


@dataclass(frozen=True)
class DeskAsset:
    """One desk, with everything a task needs to place a duck on it."""

    key: str
    xml: Path
    # Height of the flat top, from mj_ray (not the XML's geom z + half-size:
    # desk_drawers has a 76 mm chamfer at the +x/+y corner that a bounding box
    # would ignore and the duck would fall through).
    top_z: float
    # Largest axis-aligned rectangle inscribed in the flat region.
    pocket_x: tuple[float, float]
    pocket_y: tuple[float, float]
    # Collision geom count: what this desk costs the broadphase per env.
    n_col_geoms: int
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
        """Largest centred-orbit radius whose body corners stay on the surface.

        The duck's furthest collision corner is BODY_REACH from its root and the
        heading turns with the circle, so that corner sweeps a disc of radius
        ``r + BODY_REACH`` around the orbit centre. It has to stay inside the
        inscribed rectangle's smaller half-extent.
        """
        return min(self.half) - BODY_REACH


DESKS: dict[str, DeskAsset] = {
    # 13 collision boxes: the cheap one. USE THIS FOR TRAINING -- the whole
    # point of a desk in the mix is a clean large surface, and 13 geoms is less
    # than half of the sofa's 26.
    "multiview": DeskAsset(
        key="odg_desk_multiview_a",
        xml=_DESK_DIR / "odg_desk_multiview_a.xml",
        top_z=0.7566,
        # Measured [-0.300, +0.300] x [-0.610, +0.610], then pulled 5 mm inward
        # on every side: the declaration is the rectangle the duck is allowed to
        # use, not the rectangle the mesh happens to have.
        pocket_x=(-0.295, 0.295),
        pocket_y=(-0.605, 0.605),
        n_col_geoms=13,
        note=(
            "flat top over its whole footprint; the rear cabinet block "
            "(col_0, top 0.756) is at the same height as the top plate, so the "
            "two boxes form ONE continuous surface with a 0.4 mm step at "
            "y ~ +0.17. Measured over a 1 cm grid: no hole, no lip."
        ),
    ),
    # 73 collision boxes: the expensive one. Keep for evaluation / for a
    # desk-with-drawers visual, not for a mixed training run -- see README §17.
    "drawers": DeskAsset(
        key="odg_desk_drawers_a",
        xml=_DESK_DIR / "odg_desk_drawers_a.xml",
        top_z=0.7860,
        # Measured [-0.320, +0.290] x [-0.600, +0.600], 5 mm inward. The x range
        # is NOT symmetric: see the note.
        pocket_x=(-0.315, 0.285),
        pocket_y=(-0.595, 0.595),
        n_col_geoms=73,
        note=(
            "the +x/+y corner is chamfered: rays at (0.32, 0.45) return 0.696 "
            "and at (0.30, 0.50) return nothing at all, so the usable depth "
            "stops at x = +0.285 while the far side goes to -0.315. Reading the "
            "asset's 1.225 x 0.602 bounding box instead would put the duck's "
            "orbit 3 cm past the chamfer."
        ),
    ),
}
DEFAULT_DESK = DESKS["multiview"]

# --- Contact model -----------------------------------------------------------
# Deliberately the asset's furniture-hard default, NOT the sofa's soft one.
# A desk is a rigid plate on four legs; pretending otherwise would throw away
# the only thing a desk adds to the mix. friction 1.0 / torsional 0.005 match
# the asset; condim 3 is what get_sofa_spec() also forces so the two surfaces
# differ in compliance alone and not in the contact basis.
DESK_SOLREF = [0.02, 1.0]
DESK_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]
DESK_FRICTION = 1.0
DESK_TORSIONAL = 0.005
DESK_ROLLING = 0.0001

# --- Spawn -------------------------------------------------------------------
# Centred on the flat top with a small jitter. The desk is 0.59 m deep, so the
# spawn box is NOT the binding constraint -- the orbit radius is (see
# radius_cap). +-0.10 keeps even the worst corner (0.10 + 0.128 = 0.228 m from
# the centre) inside the 0.295 m half-depth.
SPAWN_X_JITTER = (-0.10, 0.10)
SPAWN_Y_JITTER = (-0.10, 0.10)
# Standing trunk height above the surface: the same 0.115-0.125 used on the
# sofa (measured: the trunk sits 0.1204 m above the desk top when standing).
STAND_HEIGHT = 0.120
SPAWN_Z_OFFSET = (0.115, 0.125)

# Duck faces along the desk's LONG axis at spawn, same convention as the sofa
# (long axis = y for both desks, x = +-0.30 is the short side).
SPAWN_YAW = (0.0, 3.141592653589793)


def get_desk_spec(desk: DeskAsset | str = DEFAULT_DESK) -> mujoco.MjSpec:
    """Load a desk MJCF and make its collision geoms visible to ray sensors.

    Mirrors ``get_sofa_spec()`` except for the contact model: the sofa's
    solref/solimp are replaced with a soft-cushion pair, the desk keeps its own
    (hard) values. Both get ``group = 0`` so mjlab's ray-cast terrain sensors
    can see the walkable surface, and condim 3 so the contact basis is identical.
    """
    if isinstance(desk, str):
        desk = DESKS[desk]
    spec = mujoco.MjSpec.from_file(str(desk.xml))
    tuned = 0
    for geom in spec.geoms:
        if "_col_" not in geom.name:
            continue  # visual meshes (contype=0, density=0) stay untouched
        geom.group = 0                 # walkable surface -> visible to ray sensors
        geom.condim = 3
        geom.friction = [DESK_FRICTION, DESK_TORSIONAL, DESK_ROLLING]
        geom.solref = list(DESK_SOLREF)
        geom.solimp = list(DESK_SOLIMP)
        geom.contype = 1
        geom.conaffinity = 1
        tuned += 1
    if tuned == 0:
        raise RuntimeError(f"no '_col_' geoms found in {desk.xml}")
    return spec

#!/usr/bin/env python3
"""Build the sofa training env on CPU and PROVE the tricky wiring resolves.

A 30-second CPU-only gate that exists because run 2026-11-13 crashed at
`scene.compile()` with:

    ValueError: Error: unrecognized name '.*odg_sofa_concept_a.*' of object
    Element name 'feet_ground_contact_left_foot_collision_found', id 6

Root cause: `ContactMatch(pattern=r".*odg_sofa_concept_a.*")` WITHOUT `entity=`
is a LITERAL name in mjlab 1.3.0 (see sensor/contact_sensor.py:
`_resolve_single_secondary` returns the pattern unchanged when `entity` is
None; regex syntax then reaches MjSpec.compile and MuJoCo refuses it).

The fix is `entity="sofa"`, which expands the regex against the sofa entity's
own bodies and writes the resolved, prefixed name (`sofa/odg_sofa_concept_a`)
into the spec. This script asserts exactly that, BEFORE any GPU time is spent:

  1. the env builds and compiles;
  2. every feet_ground_contact sensor of type CONTACT has a refname that
     resolves to the sofa body (objid -> body name contains "odg_sofa");

For the circle task (`--task Mjlab-Circle-Sofa-MicroDuck`) it additionally
asserts the geometry that task depends on and that no CPU-only test can
otherwise reach:

  3. after reset the robot sits ON its circle (|dist - radius| < 1 mm), at seat
     height, facing along the tangent;
  4. the emitted command is exactly (omega*r, 0, omega) and consistent with the
     sampled circle;
  5. two consecutive resets produce different circles (the sampler is live);
  6. a duck that has not moved yet scores ~0 on BOTH circle terms, while the
     ungated radius term still reads ~1 (i.e. the c2 "no free reward for
     standing" gate is actually wired).

And for EVERY task, before any of that, it re-derives the seat geometry from the
asset (section 0 below). That gate exists because of a bug that survived four
training runs and was invisible in every log:

  0. ray-cast the compiled sofa downward on a 2 cm grid and check the declared
     constants (`sofa_constants.py`) against the measurement: pocket bounds,
     centre, seat height. Then check every spawn point -- plus the duck's four
     footprint corners, rotated by the spawn yaw -- lands on seat (z ~ 0.518)
     and not on backrest (0.980) or bolster (0.868). Then check that a whole
     commanded segment, and a whole orbit, fits inside the measured pocket.

Before 2026-11-14 the spawn sat at x = -0.13, which is 0.35 m behind the free
seat and INSIDE the backrest solid. Nothing failed; the duck was simply launched
out of a wall in every episode of every sofa run, and the resulting policies were
read as "the duck cannot walk". A ray-cast proves otherwise in 30 seconds.

    .venv/bin/python scripts/sofa/build_check.py
    .venv/bin/python scripts/sofa/build_check.py --task Mjlab-Circle-Sofa-MicroDuck
    TASK=... bash scripts/sofa/modelscope_run.sh check     # runs it for you

Exit code: 0 = env builds and the wiring is correct, 1 = broken.
"""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_TASK = "Mjlab-Velocity-Sofa-MicroDuck"
CIRCLE_TASK = "Mjlab-Circle-Sofa-MicroDuck"

# --- Geometry probe tuning ---------------------------------------------------
# 5 mm marching resolution: fine enough that the pocket bounds it reports can be
# compared against the declared constants to +-1 cm, coarse enough that the whole
# sweep is a few thousand rays (~1 s on CPU).
PROBE_STEP = 0.005
# A ray "lands on the seat" if it comes back within this of SEAT_TOP_Z. The three
# candidate heights are 0.518 (seat), 0.868 (bolster) and 0.980 (backrest), so
# 3 cm separates them with a factor of 10 to spare.
SEAT_Z_TOL = 0.03
# Declared constants must match the measurement to this. The probe marches in
# 5 mm steps, so anything larger means the constant is simply wrong.
GEOM_TOL = 0.01


class SeatProbe:
    """Downward ray-caster over the compiled sofa asset, in the asset's frame.

    The asset's frame is the one the spawn constants live in: origin at floor
    centre, sofa at (0, 0), Z up. mjlab places the *entity* at each env origin
    and `pose_range` is relative to that, so a point measured here is directly
    comparable to ``SPAWN_X`` / ``SEAT_CENTER_XY`` without any env offset.
    """

    def __init__(self, mujoco):
        import numpy as np

        from mjlab_microduck.robot.microduck.sofa_constants import get_sofa_spec

        self.np = np
        self.mujoco = mujoco
        self.model = get_sofa_spec().compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        # Only the COLLISION geoms. The asset also carries 6 visual meshes, one of
        # which (the seat's seam strip, vis_5) dips ~6 cm below the cushion at
        # y = +0.005 -- a ray that is allowed to hit it reports a 0.458 m "seat"
        # and the whole sweep collapses to y >= 0. Restricting mj_ray to the
        # groups that actually carry contype > 0 is the only reliable filter;
        # a post-hoc check of geomid cannot work, because mj_ray stops at the
        # first hit and would never see past the mesh.
        groups = np.zeros(6, dtype=np.uint8)
        for g in range(self.model.ngeom):
            if self.model.geom_contype[g] > 0 or self.model.geom_conaffinity[g] > 0:
                groups[self.model.geom_group[g]] = 1
        if not groups.any():  # defensive: fall back to everything
            groups[:] = 1
        self.geomgroup = groups
        self._up = np.array([0.0, 0.0, -1.0])
        self.n_rays = 0

    def z(self, x: float, y: float, z0: float = 1.6) -> float | None:
        """Height of the topmost collidable surface at (x, y), or None.

        The asset is built from two seat boxes (col_0 / col_1) that meet at
        y = +0.005 m. That seam is zero-width, so it resolves contacts perfectly
        but a ray aimed exactly along it falls straight through and reports
        "nothing here" -- which made the first version of this probe measure the
        pocket as y >= 0, i.e. half a seat. Four 1 mm nudges around the sample
        make the probe see through such cracks; the worst-case positional error
        they can introduce is 1 mm, against a 5 mm march step.
        """
        np = self.np
        for dx, dy in ((0.0, 0.0), (0.0, 1e-3), (0.0, -1e-3), (1e-3, 0.0), (-1e-3, 0.0)):
            geomid = np.zeros(1, dtype=np.int32)
            dist = self.mujoco.mj_ray(
                self.model, self.data, np.array([x + dx, y + dy, z0]), self._up,
                self.geomgroup, 1, -1, geomid,
            )
            self.n_rays += 1
            if dist >= 0:
                return z0 - dist
        return None

    def on_seat(self, x: float, y: float) -> tuple[bool, float | None]:
        from mjlab_microduck.robot.microduck.sofa_constants import SEAT_TOP_Z

        z = self.z(x, y)
        return (z is not None and abs(z - SEAT_TOP_Z) <= SEAT_Z_TOL), z

    def march(self, x0: float, y0: float, dx: float, dy: float) -> float:
        """Distance from (x0, y0) to the first off-seat point along (dx, dy).

        Marches outwards from a point known to be on the seat and returns the
        last distance that was still on it. This is how the pocket bounds are
        measured: no grid quantisation, no bounding boxes.
        """
        dist = 0.0
        while dist < 1.5:
            nxt = dist + PROBE_STEP
            if not self.on_seat(x0 + nxt * dx, y0 + nxt * dy)[0]:
                return dist
            dist = nxt
        return dist

    def span_x(self, y: float, x_seed: float) -> tuple[float, float]:
        return (
            x_seed - self.march(x_seed, y, -1.0, 0.0),
            x_seed + self.march(x_seed, y, +1.0, 0.0),
        )

    def span_y(self, x: float, y_seed: float = 0.0) -> tuple[float, float]:
        return (
            y_seed - self.march(x, y_seed, 0.0, -1.0),
            y_seed + self.march(x, y_seed, 0.0, +1.0),
        )


def measure_pocket(pr: "SeatProbe") -> dict:
    """The largest axis-aligned rectangle INSCRIBED in the walkable region.

    Naively taking the min/max of every free cell gives the bounding box of the
    T -- 0.54 x 1.66 m -- which is the 1.56 x 0.62 m lie again in a new dress: no
    single axis-aligned rectangle of that size is free, and a duck spawned at its
    corner lands on an armrest.

    So: measure the depth on a row that is inside the backrest (y = +0.20, NOT
    y = 0, because the two back cushions leave a gap at y ~ 0 that a 351 mm ray
    happily shoots through), then measure the length just inside that rear wall
    -- the narrowest place, where the seat bolsters pinch the pocket -- then
    re-measure the depth on the rows that length left us. Two passes reach the
    fixed point; the third is an assertion.
    """
    x_lo, x_hi = pr.span_x(y=+0.20, x_seed=0.25)
    y_lo, y_hi = pr.span_y(x=x_lo + 0.02, y_seed=0.0)
    # Tighten the depth on the rows the length allows (skip the very edge rows:
    # the march returns the last ON-seat sample, so the boundary is real there).
    for y in (y_lo + 0.02, 0.0, y_hi - 0.02):
        lo, hi = pr.span_x(y=y, x_seed=0.25)
        x_lo, x_hi = max(x_lo, lo), min(x_hi, hi)
    return {
        "x": (x_lo, x_hi),
        "y": (y_lo, y_hi),
        "centre": (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)),
    }



def check_seat_geometry(mujoco) -> int:
    """Re-derive the walkable seat from the asset and gate the spawn on it.

    Every number this prints is measured, so a diff between this output and the
    constants in ``robot/microduck/sofa_constants.py`` IS the bug report.
    """
    import math

    from mjlab_microduck.robot.microduck import sofa_constants as K
    from mjlab_microduck.tasks.microduck_velocity_sofa_env_cfg import (
        CMD_HOLD_S,
        LIN_VEL_X,
    )

    pr = SeatProbe(mujoco)
    bad = 0

    # --- 0a. The seat height itself -----------------------------------------
    z_seat = pr.z(*K.SEAT_CENTER_XY)
    ok = z_seat is not None and abs(z_seat - K.SEAT_TOP_Z) <= GEOM_TOL
    print(f"{'ok ' if ok else 'FAIL'}: seat top at pocket centre "
          f"z = {z_seat if z_seat is None else round(z_seat, 4)} "
          f"(declared SEAT_TOP_Z = {K.SEAT_TOP_Z})")
    bad += not ok

    # --- 0b. The pocket is a T, so measure the inscribed rectangle (see
    #         measure_pocket) plus the front ledge, which is documented but not
    #         used: it is only 0.17 m wide, 2 cm more than the duck. -----------
    pocket = measure_pocket(pr)
    x_span, y_span = pocket["x"], pocket["y"]
    ledge_x = 0.5 * (K.SEAT_LEDGE_X[0] + K.SEAT_LEDGE_X[1])
    ledge_span = pr.span_y(x=ledge_x)

    print(f"     measured pocket: x[{x_span[0]:+.3f}, {x_span[1]:+.3f}] "
          f"({x_span[1] - x_span[0]:.3f} m deep)   "
          f"y[{y_span[0]:+.3f}, {y_span[1]:+.3f}] "
          f"({y_span[1] - y_span[0]:.3f} m long)   [{pr.n_rays} rays]")
    print(f"     measured ledge : x~{ledge_x:+.2f} "
          f"y[{ledge_span[0]:+.3f}, {ledge_span[1]:+.3f}] "
          f"({ledge_span[1] - ledge_span[0]:.3f} m long) -- documented only")

    for label, measured, declared in (
        ("pocket x (depth)", x_span, K.SEAT_FREE_X),
        ("pocket y (length)", y_span, K.SEAT_FREE_Y),
        ("ledge y (length)", ledge_span, K.SEAT_LEDGE_Y),
    ):
        err = max(abs(measured[0] - declared[0]), abs(measured[1] - declared[1]))
        ok = err <= GEOM_TOL
        print(f"{'ok ' if ok else 'FAIL'}: {label} declared "
              f"[{declared[0]:+.3f}, {declared[1]:+.3f}] vs measured "
              f"[{measured[0]:+.3f}, {measured[1]:+.3f}]  err {err * 1000:.0f} mm")
        bad += not ok

    cx_declared, cy_declared = K.SEAT_CENTER_XY
    cx_measured, cy_measured = pocket["centre"]
    ok = (
        abs(cx_declared - cx_measured) <= GEOM_TOL
        and abs(cy_declared - cy_measured) <= GEOM_TOL
    )
    print(f"{'ok ' if ok else 'FAIL'}: pocket centre declared "
          f"({cx_declared:+.3f}, {cy_declared:+.3f}) vs measured "
          f"({cx_measured:+.3f}, {cy_measured:+.3f})")
    bad += not ok

    # --- 0c. Every spawn point, and the whole footprint under it, on seat ----
    # Four corners of the stand-pose collision box, rotated by the sampled yaw.
    foot = [
        (fx, fy)
        for fx in K.BODY_FOOT_X
        for fy in K.BODY_FOOT_Y
    ]
    yaw_lo, yaw_hi = K.SPAWN_YAW
    n_yaw = 5
    off_seat = []
    n_probed = 0
    for i in range(5):
        sx = K.SPAWN_X[0] + (K.SPAWN_X[1] - K.SPAWN_X[0]) * i / 4
        for j in range(5):
            sy = K.SPAWN_Y[0] + (K.SPAWN_Y[1] - K.SPAWN_Y[0]) * j / 4
            for k in range(n_yaw):
                yaw = yaw_lo + (yaw_hi - yaw_lo) * k / (n_yaw - 1)
                c, s = math.cos(yaw), math.sin(yaw)
                for fx, fy in foot:
                    px, py = sx + c * fx - s * fy, sy + s * fx + c * fy
                    n_probed += 1
                    good, z = pr.on_seat(px, py)
                    if not good:
                        off_seat.append((round(sx, 3), round(sy, 3),
                                         round(yaw, 2), round(px, 3), round(py, 3), z))
    ok = not off_seat
    print(f"{'ok ' if ok else 'FAIL'}: spawn footprint {n_yaw} yaws x 5 x 5 "
          f"positions x 4 corners = {n_probed} rays, "
          f"{len(off_seat)} off the seat")
    for row in off_seat[:8]:
        print(f"       spawn ({row[0]:+.3f},{row[1]:+.3f}) yaw {row[2]:+.2f} "
              f"-> corner ({row[3]:+.3f},{row[4]:+.3f}) hits z={row[5]}")
    bad += not ok

    # The old constants, kept as a regression trap: if a future edit widens the
    # spawn back to these, print exactly what it costs instead of failing mute.
    for label, (px, py) in (("old spawn x=-0.13", (-0.13, 0.0)),
                            ("old spawn y=+0.45", (0.225, 0.45))):
        good, z = pr.on_seat(px, py)
        print(f"     {label}: z={'none' if z is None else round(z, 3)} "
              f"-> {'on seat' if good else 'OFF SEAT (this is what every run before 2026-11-14 spawned into)'}")

    # --- 0d. A straight command segment must fit the pocket -----------------
    # travel + |spawn_y| + forward reach <= the NEARER of the two bolsters. The
    # pocket is asymmetric (0.39 to +y, 0.41 to -y) because the right-hand
    # bolster is the one with the cupholder in it, so take the tight side.
    runway = (
        CMD_HOLD_S[1] * max(abs(v) for v in LIN_VEL_X)
        + max(abs(v) for v in K.SPAWN_Y)
        + K.BODY_HALF_LEN
    )
    have = min(abs(y_span[0]), abs(y_span[1]))
    ok = runway <= have
    print(f"{'ok ' if ok else 'FAIL'}: straight segment needs {runway:.3f} m "
          f"({CMD_HOLD_S[1]} s x {max(abs(v) for v in LIN_VEL_X):.2f} m/s + "
          f"spawn + reach) vs {have:.3f} m to the nearer bolster")
    bad += not ok

    # --- 0e. A whole orbit must fit the pocket ------------------------------
    # The heading turns with the circle, so the binding quantity is the duck's
    # worst footprint corner (BODY_REACH), not a half extent.
    from mjlab_microduck.tasks.microduck_circle_sofa_env_cfg import (
        CIRCLE_OMEGA,
        CIRCLE_RADIUS,
        CIRCLE_RADIUS_MAX,
    )

    half_depth = 0.5 * (x_span[1] - x_span[0])
    r_max_measured = half_depth - K.BODY_REACH
    ok = CIRCLE_RADIUS[1] <= r_max_measured + 1e-9
    print(f"{'ok ' if ok else 'FAIL'}: orbit r <= {CIRCLE_RADIUS[1]} m + body "
          f"reach {K.BODY_REACH} m = {CIRCLE_RADIUS[1] + K.BODY_REACH:.3f} m "
          f"vs half-depth {half_depth:.3f} m "
          f"(measured cap {r_max_measured:.3f}, declared cap "
          f"{CIRCLE_RADIUS_MAX:.3f})")
    bad += not ok
    ok = abs(CIRCLE_RADIUS_MAX - r_max_measured) <= GEOM_TOL
    print(f"{'ok ' if ok else 'FAIL'}: CIRCLE_RADIUS_MAX derived in the cfg "
          f"agrees with the asset ({CIRCLE_RADIUS_MAX:.3f} vs "
          f"{r_max_measured:.3f})")
    bad += not ok

    # Sweep the orbit: at 16 angles, place the duck tangent to the circle and
    # probe its four footprint corners. This is the check that would have caught
    # a circle centred on the seat boxes' bounding box.
    worst = None
    n_orbits = 0
    for sign in (+1.0, -1.0):
        for r in (CIRCLE_RADIUS[0], CIRCLE_RADIUS[1]):
            for k in range(16):
                th = -math.pi + 2.0 * math.pi * k / 16
                px = cx_measured + r * math.cos(th)
                py = r * math.sin(th)
                yaw = th + sign * math.pi / 2.0
                c, s = math.cos(yaw), math.sin(yaw)
                for fx, fy in foot:
                    qx, qy = px + c * fx - s * fy, py + s * fx + c * fy
                    n_orbits += 1
                    good, z = pr.on_seat(qx, qy)
                    if not good:
                        worst = (round(r, 3), round(th, 2), round(qx, 3),
                                 round(qy, 3), z)
    ok = worst is None
    print(f"{'ok ' if ok else 'FAIL'}: orbit footprint sweep ({n_orbits} rays "
          f"over 2 directions x 2 radii x 16 angles) stays on seat"
          + ("" if ok else f"; worst corner {worst}"))
    bad += not ok

    print(f"     omega range {CIRCLE_OMEGA} -> tangential speed "
          f"{CIRCLE_OMEGA[0] * CIRCLE_RADIUS[0]:.3f}-"
          f"{CIRCLE_OMEGA[1] * CIRCLE_RADIUS[1]:.3f} m/s, "
          f"lap {2 * math.pi / CIRCLE_OMEGA[1]:.1f}-"
          f"{2 * math.pi / CIRCLE_OMEGA[0]:.1f} s")
    return bad


def check_contact_sensor(mujoco, model) -> int:
    """Every feet_ground_contact CONTACT sensor must reference the sofa body."""
    bad = 0
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if not name.startswith("feet_ground_contact"):
            continue
        if model.sensor_type[i] != mujoco.mjtSensor.mjSENS_CONTACT:
            continue
        objname = mujoco.mj_id2name(model, model.sensor_objtype[i], model.sensor_objid[i])
        if model.sensor_reftype[i] == mujoco.mjtObj.mjOBJ_UNKNOWN:
            print(f"FAIL: sensor '{name}' has NO reference (refname=<none>)")
            bad += 1
            continue
        refname = mujoco.mj_id2name(model, model.sensor_reftype[i], model.sensor_refid[i])
        ok = refname is not None and "odg_sofa" in refname
        print(f"{'ok ' if ok else 'FAIL'}: {name}  obj={objname}  ref={refname}")
        if not ok:
            bad += 1
    return bad


def check_circle_geometry(env, n_resets: int = 24) -> int:
    """The circle task is worthless if the spawn is off the circle.

    `num_envs=1`, so a single reset only ever shows one sample: the loop below
    resets repeatedly to cover the sampler's range and both orbit directions.
    """
    import torch

    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.microduck_circle_sofa_env_cfg import (
        CIRCLE_OMEGA,
        CIRCLE_RADIUS,
        SEAT_CENTER_XY,
    )
    from mjlab_microduck.robot.microduck.sofa_constants import SEAT_TOP_Z

    unb = env.unwrapped
    robot = unb.scene["robot"]
    term = unb.command_manager.get_term("twist")
    cfg = term.cfg

    radii, omegas, centers, thetas, poses, cmds = [], [], [], [], [], []
    for _ in range(n_resets):
        unb.reset()
        st = microduck_mdp.circle_state(unb)
        radii.append(st.radius.clone())
        omegas.append(st.omega.clone())
        centers.append(st.center.clone())
        thetas.append(st.theta_prev.clone())
        poses.append(robot.data.data.qpos.clone())
        cmds.append(term.command.clone())

    radius = torch.cat(radii)
    omega = torch.cat(omegas)
    center = torch.cat(centers)
    theta = torch.cat(thetas)
    qpos = torch.cat(poses)
    cmd = torch.cat(cmds)

    pos = qpos[:, 0:3]
    qw, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    bad = 0

    dist = torch.norm(pos[:, :2] - center, dim=-1)
    radial_err = (dist - radius).abs().max().item()
    print(f"{'ok ' if radial_err < 1e-3 else 'FAIL'}: spawn "
          f"|dist - r| max = {radial_err:.6f} m "
          f"(r in [{radius.min():.3f}, {radius.max():.3f}])")
    bad += radial_err >= 1e-3

    # Tangent heading: the body x axis must point along +-the circle tangent.
    #
    # Compare as VECTORS, not as an angle difference. yaw comes from atan2 (so
    # it is wrapped to [-pi, pi]) while a spawn angle is free to be anything;
    # an unwrapped `yaw - theta` then reports a 2*pi branch-cut jump as a 360 deg
    # heading error. That is exactly how this check first failed -- 6.283185 rad
    # on a spawn that was geometrically perfect. A dot product has no branch cut.
    tang = torch.sign(omega)
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    tangent = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    tangent = tangent * tang.unsqueeze(-1)
    cos_err = (heading * tangent).sum(-1)
    worst_cos = cos_err.min().item()
    # Signed angle error, wrapped, purely as a human-readable companion number.
    ang_err = torch.remainder(
        yaw - theta - tang * (torch.pi / 2.0) + torch.pi, 2.0 * torch.pi
    ) - torch.pi
    print(f"{'ok ' if worst_cos > 0.999 else 'FAIL'}: tangent heading "
          f"min cos = {worst_cos:+.6f} "
          f"(max |ang err| = {ang_err.abs().max().item():.2e} rad)")
    bad += worst_cos <= 0.999

    z_err = (pos[:, 2] - (SEAT_TOP_Z + 0.12)).abs().max().item()
    print(f"{'ok ' if z_err < 0.02 else 'FAIL'}: spawn z max err = {z_err:.4f} m "
          f"(want {SEAT_TOP_Z + 0.12:.3f} +-0.02)")
    bad += z_err >= 0.02

    # Command must be the orbit rate, not a random twist draw.
    cmd_err = (
        (cmd[:, 0] - omega * radius).abs().max().item(),
        (cmd[:, 1]).abs().max().item(),
        (cmd[:, 2] - omega).abs().max().item(),
    )
    ok_cmd = max(cmd_err) < 1e-5
    print(f"{'ok ' if ok_cmd else 'FAIL'}: command (v, 0, w) err = "
          f"({cmd_err[0]:.2e}, {cmd_err[1]:.2e}, {cmd_err[2]:.2e})")
    bad += not ok_cmd

    # The sampled circle must be inside the declared envelope.
    r_lo, r_hi = CIRCLE_RADIUS
    w_lo, w_hi = CIRCLE_OMEGA
    in_range = (
        bool((radius >= r_lo - 1e-6).all() and (radius <= r_hi + 1e-6).all())
        and bool((omega.abs() >= w_lo - 1e-6).all())
        and bool((omega.abs() <= w_hi + 1e-6).all())
        and bool((omega.abs() > 0).all())
    )
    print(f"{'ok ' if in_range else 'FAIL'}: sampled r/w inside "
          f"r{r_lo}-{r_hi}, |w|{w_lo}-{w_hi} over {n_resets} resets")
    bad += not in_range

    # Both directions must occur across a batch (a one-way batch would train a
    # policy that can only turn left).
    both = bool((omega > 0).any() and (omega < 0).any())
    print(f"{'ok ' if both else 'FAIL'}: both orbit directions sampled "
          f"({int((omega > 0).sum())} ccw / {int((omega < 0).sum())} cw)")
    bad += not both

    # The circle must fit the measured pocket. The literal numbers that used to
    # live here (x in [-0.45, 0.17], y in [-0.78, 0.78]) were the seat boxes'
    # bounding box, i.e. the wrong surface -- see check_seat_geometry above.
    from mjlab_microduck.robot.microduck.sofa_constants import (
        BODY_REACH,
        SEAT_FREE_X,
        SEAT_FREE_Y,
    )

    cx, cy = SEAT_CENTER_XY
    ox = unb.scene.env_origins[0, 0].item()
    x_lo = (center[:, 0] - radius).min().item() - ox - BODY_REACH
    x_hi = (center[:, 0] + radius).max().item() - ox + BODY_REACH
    y_lo = (center[:, 1] - radius).min().item() - BODY_REACH
    y_hi = (center[:, 1] + radius).max().item() + BODY_REACH
    fits = (
        x_lo >= SEAT_FREE_X[0]
        and x_hi <= SEAT_FREE_X[1]
        and y_lo >= SEAT_FREE_Y[0]
        and y_hi <= SEAT_FREE_Y[1]
    )
    print(f"{'ok ' if fits else 'FAIL'}: orbit footprint (r + body reach "
          f"{BODY_REACH} m) x[{x_lo:+.3f}, {x_hi:+.3f}] y[{y_lo:+.3f}, "
          f"{y_hi:+.3f}] inside pocket x[{SEAT_FREE_X[0]:+.3f}, "
          f"{SEAT_FREE_X[1]:+.3f}] y[{SEAT_FREE_Y[0]:+.3f}, "
          f"{SEAT_FREE_Y[1]:+.3f}] (centre x={cx:+.3f})")
    bad += not fits

    # A frozen sampler would silently train one single orbit forever.
    distinct = len({round(float(r), 5) for r in radius})
    print(f"{'ok ' if distinct > 1 else 'FAIL'}: {distinct} distinct radii "
          f"over {n_resets} resets")
    bad += distinct <= 1

    # --- Reward wiring: a stander must score ~0 on both circle terms ---------
    # Cheap, but it is the one thing that decides the whole run. Run c1 spent
    # 4000 iterations learning to march in place because standing still on the
    # circle collected a full radius reward (1.0/step) and a Gaussian arc term
    # that barely left its floor. The c2 recipe gates radius credit on actual
    # travel and uses a linear arc term, so RIGHT AFTER A RESET -- duck exactly
    # on the circle, zero motion -- both must read ~0, while the ungated radius
    # term still reads ~1. If someone drops `gate_ratio` or passes the gaussian
    # function by mistake, this prints FAIL in 30 seconds instead of costing a
    # GPU-night.
    unb.reset()
    if hasattr(microduck_mdp, "circle_progress_linear_reward"):
        arc = microduck_mdp.circle_progress_linear_reward(unb)
        rad_gated = microduck_mdp.circle_radius_reward(unb, gate_ratio=0.25)
        rad_free = microduck_mdp.circle_radius_reward(unb, gate_ratio=0.0)
        arc_max = float(arc.max())
        gated_max = float(rad_gated.max())
        free_min = float(rad_free.min())
        print(f"{'ok ' if arc_max < 0.15 else 'FAIL'}: standing still scores "
              f"arc_rate = {arc_max:.4f} (want ~0)")
        bad += arc_max >= 0.15
        print(f"{'ok ' if gated_max < 0.15 else 'FAIL'}: standing still scores "
              f"gated radius = {gated_max:.4f} (want ~0; ungated = {free_min:.3f})")
        bad += gated_max >= 0.15
        print(f"{'ok ' if free_min > 0.85 else 'FAIL'}: ungated radius on the "
              f"circle = {free_min:.3f} (want ~1: the duck IS on it)")
        bad += free_min <= 0.85

    print(f"     command cfg radius_range={cfg.radius_range} "
          f"omega_range={cfg.omega_range}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default=os.environ.get("TASK", DEFAULT_TASK),
                    help=f"mjlab task id (default: $TASK or {DEFAULT_TASK})")
    args = ap.parse_args()
    task_id = args.task

    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    print(f"building env for task {task_id} (num_envs=1, CPU)")
    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

    model = env.unwrapped.sim.mj_model
    print(f"compiled: nbody={model.nbody} ngeom={model.ngeom} nsensor={model.nsensor}")

    # 1. The sofa body must exist under its prefixed scene name.
    sofa_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sofa/odg_sofa_concept_a")
    if sofa_id < 0:
        print("FAIL: body 'sofa/odg_sofa_concept_a' not found in the compiled model")
        print("  sofa-ish bodies:", [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
            if "sofa" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
        ])
        env.close()
        return 1
    print(f"ok: body 'sofa/odg_sofa_concept_a' id={sofa_id}")

    # 0. Seat geometry, re-derived from the asset. Runs for every task: the
    #    spawn constants are shared, and this is the check that would have
    #    caught the duck being spawned inside the backrest for four runs.
    print()
    print("seat geometry (ray-cast of the compiled asset):")
    try:
        bad = check_seat_geometry(mujoco)
    except Exception:  # noqa: BLE001 - report and fail the gate
        import traceback
        traceback.print_exc()
        print("FAIL: seat geometry check raised (see traceback above)")
        bad = 1

    # 2. Feet contact sensors must resolve against the cushion.
    bad += check_contact_sensor(mujoco, model)

    # 3. Task-specific geometry (circle only for now).
    if task_id == CIRCLE_TASK:
        print()
        print("circle geometry checks:")
        try:
            bad += check_circle_geometry(env)
        except Exception:  # noqa: BLE001 - report and fail the gate
            import traceback
            traceback.print_exc()
            print("FAIL: circle geometry checks raised (see traceback above)")
            bad += 1

    env.close()

    if bad:
        print(f"\nFAIL: {bad} check(s) failed.")
        print("  Sensor wiring: src/mjlab_microduck/tasks/microduck_velocity_sofa_env_cfg.py")
        print("  (must be ContactMatch(mode='body', pattern=..., entity='sofa'),")
        print("   NOT a bare regex without entity).")
        print("  Circle spawn/reward: src/mjlab_microduck/tasks/mdp.py")
        print("  (reset_on_circle + CircleVelocityCommand) and")
        print("  src/mjlab_microduck/tasks/microduck_circle_sofa_env_cfg.py")
        return 1

    n_feet = sum(
        1
        for i in range(model.nsensor)
        if model.sensor(i).name.startswith("feet_ground_contact")
    )
    print(f"\nPASS: env builds on CPU; {n_feet} feet_ground_contact sensors all")
    print("      resolve foot contacts against the sofa cushion. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

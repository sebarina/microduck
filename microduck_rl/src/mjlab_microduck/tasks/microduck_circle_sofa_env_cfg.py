"""Microduck circle-walk task on a sofa seat — walk small laps on the cushion.

Why this task exists
--------------------
Straight-line velocity tracking on this cushion is geometrically self-defeating:
the walkable pocket is only 0.53 x 0.84 m, so even a 0.12 m/s command runs out
of seat inside one 1.5 s command segment, and "track well" fights "don't fall
off". The optimal policy is to stand still. Four runs (2026-11-12, 11-13
v2_bounded, 11-14 v3_gait, and circle c1) all converged to standing or to
marching in place; see scripts/sofa/README.md §13 and §15.

A circle removes the conflict entirely: the commanded path returns to its own
start, so a policy can track perfectly forever without approaching an edge. Of
the 0.53 m of seat depth available, an orbit of radius r plus the duck's 0.128 m
body reach has to fit inside +-0.265 m around the pocket centre (+0.225, 0), so
r <= 0.137 m is the hard limit (build_check.py asserts it). No other path shape
on this seat has that much room.

What the policy is asked to do
------------------------------
  * Spawn ON the circle, facing along the tangent (see
    mdp.reset_on_circle). The geometry is exact from step 0 — the duck does not
    have to discover the orbit, only the gait that stays on it.
  * Obey a constant command (v_tangent, 0, omega) with v_tangent = omega * r,
    where omega ~ ±U(0.50, 0.90) rad/s and the sign (CW/CCW) is re-drawn every
    episode. That is 0.040-0.108 m/s of tangential speed and 7-13 s per lap,
    i.e. about one lap per 12 s episode. (c1 used r<=0.20 / omega 0.35-0.55 /
    20 s episodes, and its 0.078 m/s peak happened to be in the same band -- the
    orbit it was asked for simply did not fit the seat.)
  * Keep the radius: the arc-rate reward alone would be satisfied by an orbit of
    the right curvature somewhere else.

Reward design (the part that differs from the straight-line task)
-----------------------------------------------------------------
`circle_progress` scores the ARC ACTUALLY SWEPT AROUND THE CENTRE, measured
from the robot's world position — not the yaw rate from the gyro. That single
choice kills both failure modes at once: a standing duck sweeps 0 rad/s of arc,
and a duck spinning on the spot also sweeps 0 rad/s of arc, because it never
moves. Only walking the circle earns anything. It also makes
`track_angular_velocity` redundant, and worse than redundant — it pays for
spinning in place — so it is switched OFF here (weight 0.0). That is the one
place where this file deliberately diverges from the stock recipe beyond
magnitudes.

c1 scored the arc rate with a Gaussian around ratio=1. It was run to
completion (4000 iters, 2026-11-14) and the policy marched in place:
`air_time` 0.37/step, `circle_progress` ~0.02, and in verify `w_ach` = 0.01 rad/s
against a 0.4-0.5 rad/s command. The Gaussian's slope at a standstill is 8x too
weak to beat the ~2.4/step a stander collects from upright+pose+radius+air_time.
c2 uses a LINEAR arc rate (constant slope, zero at standstill) and gates the
radius credit on actually travelling — see README §14.11.

The observation contract is untouched: 61-D obs / 14-D actions, and the
command channel still carries (v, 0, omega). Deployment therefore needs no
surgery — feed a constant (omega*r, 0, omega) and any yaw-setpoint loop on top
just re-aims the circle.

Diagnostics (what to watch in tensorboard)
-----------------------------------------
  * `Episode_Reward/circle_progress` — the whole task in one number. With the
    c2 linear term: ~1.0 means orbiting at the commanded rate, ~0.5 means half
    speed, exactly 0 means it never moved.
  * `Episode_Reward/circle_radius` — 1.0 on the circle while travelling; it is
    gated on the arc rate, so it reads ~0 for a stander BY DESIGN (a stander
    that shows 1.0 here means the gate is not wired).
  * `Episode_Reward/air_time` — the gait-bootstrap signal. It was exactly 0.000
    for all 4000 iterations of both straight-line runs; a non-zero value here
    within a few hundred iterations is the first proof of life.

Validated on CPU (no GPU needed) by ``scripts/sofa/build_check.py``, which
constructs this env and asserts the sensor/asset wiring resolves.
"""

from __future__ import annotations

import copy

from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp as envs_mdp

from mjlab_microduck.robot.microduck.sofa_constants import (
    BODY_REACH,
    SEAT_CENTER_XY,
    SEAT_FREE_X,
    SEAT_TOP_Z,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab_microduck.tasks.microduck_velocity_sofa_env_cfg import (
    FALL_OFF_MARGIN,
    make_microduck_velocity_sofa_env_cfg,
)

# --- Circle geometry ---------------------------------------------------------
# Centre of the WALKABLE pocket (SEAT_CENTER_XY = +0.225, 0), measured by
# ray-casting the compiled asset -- NOT the seat boxes' bounding-box centre,
# which sits under the backrest. See the geometry block in
# robot/microduck/sofa_constants.py: every sofa run before 2026-11-14 used
# centre (-0.14, 0) and therefore trained with the duck buried in the backrest.
#
# The radius cap is DERIVED, not tuned. The pocket is 0.53 m deep, so an orbit
# leaves at most 0.265 m on either side of the centre; the duck's furthest
# collision corner is 0.128 m from its root and the heading turns with the
# circle, so that corner can point anywhere:
#     r + 0.128 <= 0.265   ->   r <= 0.137
# 0.12 is used, which keeps 17 mm of slack for tracking error and for the
# body roll the BAM actuator adds. (The old 0.14-0.20 range came from a 0.62 m
# "depth" that does not exist -- it was the seat boxes' AABB, whose rear third
# is solid backrest.)
CIRCLE_RADIUS_MAX: float = (SEAT_FREE_X[1] - SEAT_FREE_X[0]) / 2 - BODY_REACH
CIRCLE_RADIUS = (0.08, 0.12)

# Orbit rate. Chosen so the TANGENTIAL speed stays in the band the cushion
# tolerated in the A/B experiments (README §13): v = omega*r then spans
# 0.040-0.108 m/s. A lap is 2*pi/omega = 7-13 s, i.e. roughly one lap per
# (12 s) episode -- the horizon over which open-loop drift is still harmless.
CIRCLE_OMEGA = (0.50, 0.90)

V_TANGENT_MAX = CIRCLE_OMEGA[1] * CIRCLE_RADIUS[1]  # 0.108 m/s

# Import-time geometry gate. Cheap, and it fails in the editor rather than
# 30 GPU-minutes later at `scene.compile()`. build_check.py does the expensive
# version: it re-measures the pocket from the asset instead of trusting these
# two constants.
assert CIRCLE_RADIUS[1] <= CIRCLE_RADIUS_MAX + 1e-9, (
    f"CIRCLE_RADIUS max {CIRCLE_RADIUS[1]} m exceeds the seat: the pocket is "
    f"{SEAT_FREE_X[1] - SEAT_FREE_X[0]:.2f} m deep and the body reaches "
    f"{BODY_REACH} m from its root, so r <= {CIRCLE_RADIUS_MAX:.3f} m"
)

# --- Reward tuning ----------------------------------------------------------
# c1 used a Gaussian of width 0.5 around ratio=1: perfect 1.0, motionless 0.018.
# Its slope at a standstill is 2*exp(-4)/sigma = 0.147 per unit ratio, i.e. with
# weight 2.0 only 0.29 -- against the ~2.4/step of income a stander collected.
# That is why c1 converged to marching in place. c2 uses the LINEAR arc rate
# instead: constant slope from zero, so exploration that translates at all is
# paid immediately.
PROGRESS_WEIGHT = 2.5
PROGRESS_CAP = 1.5
# Radial tolerance. 2 cm is a sixth of the 0.04 m radius range and a small
# fraction of the 0.265 m the pocket offers: inside it the duck is essentially on
# the commanded path, outside it the orbit it is walking is a different circle.
# (c1 used 0.04 against a 0.06 m-wide range; the range is narrower now because
# the seat is, so the tolerance has to be too, or the term stops discriminating.)
RADIUS_STD = 0.02
RADIUS_WEIGHT = 0.6
# Radius credit is multiplied by clip(ratio / this, 0, 1): a stander sitting
# exactly on the circle earned 1.0 of free reward in c1, which was the largest
# single piece of income holding its standing basin in place. 0.25 means "any
# quarter-speed motion counts as travelling", so the gate never punishes a slow
# but genuine gait -- only standing still.
RADIUS_GATE_RATIO = 0.25

# Tracking std for the (demoted) linear term: same 0.12 used by v3. Kept only as
# a "keep pushing forward" nudge; `circle_progress` is what actually decides.
TRACK_LIN_STD = 0.12

# c1 paid air_time 3.0. That term is what taught the duck to lift its feet, and
# it stayed alive at 0.37/step for run c1 -- but it is also satisfied perfectly
# by marching in place, which is exactly what the policy settled into. Kept
# (the gait has to exist before it can be steered) but halved, so the arc rate
# is worth 2.5x more than the gait itself.
AIR_TIME_WEIGHT = 1.5

# Episodes are 12 s, not 20. Rationale: c1's walkers covered ~0.6 laps before
# drifting off; at 12 s the slowest orbit (omega 0.50) is 0.95 lap and the
# fastest (0.90) 1.72, so a policy that can walk at all finishes the episode on
# the cushion and every episode contains at least one full lap. The task is
# "walk a lap", not "survive an open-loop orbit for 20 s with no position
# feedback" -- the latter is not achievable on foam without an estimator.
EPISODE_LENGTH_S = 12.0


def make_microduck_circle_sofa_env_cfg(play: bool = False):
    """Circle-walk variant of the sofa velocity task."""
    # Start from the sofa task so the scene, the foam contact model, the feet
    # contact sensor (secondary re-pointed at the cushion body) and the
    # `fell_off_sofa` termination are all inherited rather than duplicated.
    cfg = make_microduck_velocity_sofa_env_cfg(play=play)

    # --- Command: constant orbit, fixed for the whole episode ----------------
    # A circle is meaningless if the radius/rate resample mid-episode: the duck
    # would be asked to step from one orbit onto another without ever being told
    # that the orbit changed. So the command never resamples on its own; it is
    # re-drawn exactly once per episode, by `reset_on_circle`.
    old = copy.deepcopy(cfg.commands["twist"])
    kwargs = vars(old)
    for k in ("radius_range", "omega_range"):
        kwargs.pop(k, None)
    command = microduck_mdp.CircleVelocityCommandCfg(**kwargs)
    command.radius_range = CIRCLE_RADIUS
    command.omega_range = CIRCLE_OMEGA
    command.resampling_time_range = (1e6, 1e6)

    # `ranges` is only a clip/sanity envelope here (the cfg above is the source
    # of truth). Keep it truthful so scripts/sofa/verify_circle.py can pin a
    # command through it and tools that print the cfg show real numbers.
    command.ranges.lin_vel_x = (-V_TANGENT_MAX, V_TANGENT_MAX)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-CIRCLE_OMEGA[1], CIRCLE_OMEGA[1])
    # Nothing may silently rewrite the orbit: no standing bucket, no world-frame
    # replay, no forced-forward override. `CircleVelocityCommand` already forces
    # these masks False on every resample; setting the fractions to 0 as well
    # keeps the printed cfg honest.
    command.rel_standing_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_world_envs = 0.0
    command.rel_forward_envs = 0.0
    command.init_velocity_prob = 0.0
    cfg.commands["twist"] = command

    # --- Spawn: on the circle, tangent to it ---------------------------------
    # This replaces reset_root_state_uniform outright. The stock reset samples
    # (x, y, yaw) independently, which lands the duck at a random place and
    # heading — on a circle task that means most episodes start with a large
    # positional error and the policy spends its first seconds recovering
    # instead of walking.
    cfg.events["reset_base"] = EventTermCfg(
        func=microduck_mdp.reset_on_circle,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "center_xy": SEAT_CENTER_XY,
            "radius_range": CIRCLE_RADIUS,
            "omega_range": CIRCLE_OMEGA,
            "seat_top_z": SEAT_TOP_Z,
            "stand_height_range": (0.115, 0.125),
        },
    )

    # --- Rewards ------------------------------------------------------------
    # The geometric terms. Weights are relative to the inherited ones: upright
    # is 1.0, pose 1.0, air_time 1.5. c2's arc rate is deliberately LINEAR --
    # see `circle_progress_linear_reward` in mdp.py for why the Gaussian had no
    # usable slope at a standstill.
    cfg.rewards["circle_progress"] = RewardTermCfg(
        func=microduck_mdp.circle_progress_linear_reward,
        weight=PROGRESS_WEIGHT,
        params={"cap": PROGRESS_CAP},
    )
    cfg.rewards["circle_radius"] = RewardTermCfg(
        func=microduck_mdp.circle_radius_reward,
        weight=RADIUS_WEIGHT,
        params={"std": RADIUS_STD, "gate_ratio": RADIUS_GATE_RATIO},
    )

    # Spinning in place satisfies a yaw-rate objective perfectly while making
    # zero progress around the circle. Off, not merely reduced.
    cfg.rewards["track_angular_velocity"].weight = 0.0
    # Linear tracking kept but demoted: it is the "keep pushing" half of the
    # command, and unlike the yaw term it cannot be satisfied without moving.
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_STD

    # Gait incentive. The stock 0.125 s threshold needs a foot airborne for 6
    # consecutive control steps; this duck shuffles on foam and never got there,
    # so that term paid exactly 0.000 for two full runs. 0.04 s = 2 steps: any
    # real foot lift counts. Half weight in c2: it must not outbid the arc rate
    # (it is fully satisfied by marching on the spot).
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = 0.04

    # Pose std is selected by `walking_threshold` against the command norm.
    # Orbit commands sit at 0.05-0.15, so the stock 0.05 boundary would flip
    # between the standing and walking std dicts at random. Pin it to walking.
    cfg.rewards["pose"].params["walking_threshold"] = 0.01

    # Keep the head from becoming a competing objective (it must oscillate while
    # stepping anyway); same demotion as v3.
    if "head_pose_tracking" in cfg.rewards:
        cfg.rewards["head_pose_tracking"].weight = 0.3

    # --- Curriculum ---------------------------------------------------------
    # `command_vel` would rewrite `ranges` mid-run. Harmless for the orbit (the
    # command term ignores ranges) but it makes the logged cfg lie about the
    # task, and a future refactor that starts honouring ranges would silently
    # widen the orbit. Drop it.
    cfg.curriculum.pop("command_vel", None)
    # No standing bucket, and no curriculum that would re-ramp one.
    cfg.curriculum.pop("standing_envs", None)
    # action_rate: cap at the stage-1 value. The full ramp to -1.0 is tuned for
    # flat ground with an existing gait; here it suppresses exactly the
    # oscillation a bootstrapping gait needs.
    if "action_rate_weight" in cfg.curriculum:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": -0.1},
            {"step": 1000 * 24, "weight": -0.2},
            {"step": 2000 * 24, "weight": -0.3},
        ]

    # --- Terminations -------------------------------------------------------
    # `fell_off_sofa` is inherited. Re-declare it so the margin is visible here
    # and not only in the parent module.
    cfg.terminations["fell_off_sofa"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        time_out=False,
        params={
            "minimum_height": SEAT_TOP_Z - FALL_OFF_MARGIN,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- Horizon ------------------------------------------------------------
    # Inherited from the sofa task: 20 s. Shortened to one lap. The duck gets a
    # constant (v, omega) with no information about where it is on the circle,
    # so radial error accumulates; a longer horizon only asks it to survive the
    # accumulation. One lap is the requirement, so make one lap the episode.
    cfg.episode_length_s = EPISODE_LENGTH_S

    return cfg


# Same network/PPO recipe; only the experiment name differs so runs land in
# their own logs/ + wandb group.
MicroduckCircleRlCfg = copy.deepcopy(MicroduckRlCfg)
MicroduckCircleRlCfg.experiment_name = "velocity_circle"
# Provenance stamp, same convention as the sofa task: any run dir that does not
# say `c2_geo` was not trained on the current circle recipe. `c1_circle` is kept
# readable in report -- it is the Gaussian-arc, 20 s, r<=0.20 run whose policy
# marched in place (README §14.11). `c2_geo` = c2's linear arc + gated radius,
# now also on the MEASURED seat pocket (the `_geo` suffix marks the 2026-11-14
# geometry fix, which changed the centre by 0.365 m, the radius range, and the
# observation that c1 was never actually standing on the cushion).
MicroduckCircleRlCfg.run_name = "c2_geo"
MicroduckCircleRlCfg.max_iterations = 4_000
MicroduckCircleRlCfg.save_interval = 100

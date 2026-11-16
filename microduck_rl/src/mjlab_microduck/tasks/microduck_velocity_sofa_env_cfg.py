"""Microduck velocity task on a sofa seat (ODG `odg_sofa_concept_a`, box colliders).

The training recipe itself is the stock velocity env, reused verbatim: same
rewards, same domain randomisation, same curricula, same BAM actuator, same
61-D observation contract (`obs[1,61] -> actions[1,14]`). Only the *scene* and a
handful of task knobs change, so a policy trained here can be hot-swapped into
`robotd` exactly like the official `velstand.onnx`.

What changes and why
--------------------
* **The sofa is a fixed-base entity.** mjlab wraps fixed-base entities in a
  mocap body automatically (``mjlab.utils.spec.auto_wrap_fixed_base_mocap``), so
  every environment gets its own copy placed at its ``env_origin`` by the
  ``reset_sofa`` event. It must be an *entity* rather than something glued into
  the scene spec: a scene-spec prop would exist once, at the world origin, and
  only env 0 would have a sofa.
* **Foam seat contacts** — see ``robot/microduck/sofa_constants.py``
  (``solref`` 0.06 vs the asset's furniture-hard 0.02, cloth friction, seat
  geoms moved into geom group 0 so the foot-height rays see the cushion instead
  of the floor 0.5 m below it).
* **Feet contact sensor: secondary swapped to the sofa body.** The stock config
  matches the literal body ``terrain``, which on this task only ever fires after
  the duck has fallen off the cushion — so ``air_time`` (weight 3.0) would be
  permanently dead and there is no gait incentive at all. The secondary is
  therefore re-pointed at the sofa body via ``entity="sofa"`` (mjlab 1.3.0
  expands a ``ContactMatch`` regex **within the named entity**; without
  ``entity`` the pattern is passed to MuJoCo as a literal name and any regex
  syntax fails spec compile — see the 2026-11-13 incident note in
  ``scripts/sofa/README.md``).
* **Spawn centred on the walkable pocket**, facing along the sofa's long axis,
  with both the command range *and the command hold time* shrunk to what fits
  on the 0.53 x 0.80 m of seat that is actually free: at 0.12 m/s a 1.45 s
  segment covers 0.174 m against 0.197 m of runway. Three runs still produced a
  stander -- see ``CMD_HOLD_S`` and README §13 -- and the geometry post-mortem
  in §15 explains why this task cannot be rescued by tuning: the free seat is
  not a 1.56 x 0.62 m rectangle, it is a T whose long arm is only 0.17 m wide.
  The circle task exists for that reason.

Validated on CPU before spending GPU time:
``scripts/sofa/validate_sofa_physics.py`` drops the real robot model onto the
seat and asserts no penetration/NaN, and that the mocap sofa produces no
contacts against the static ground plane (so it costs no ``nconmax`` budget).
"""

from __future__ import annotations

import copy

from mjlab.entity import EntityCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch

from mjlab_microduck.robot.microduck.sofa_constants import (  # noqa: F401  (re-exported)
    BODY_HALF_LEN,
    SEAT_FREE_X,
    SEAT_FREE_Y,
    SEAT_TOP_Z,
    SOFA_FRICTION,
    SOFA_ROLLING,
    SOFA_SOLIMP,
    SOFA_SOLREF,
    SOFA_TORSIONAL,
    SOFA_XML,
    SPAWN_X,
    SPAWN_Y,
    SPAWN_YAW,
    get_sofa_spec,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Commands that fit on the free seat. The stock (-0.4, 0.4) m/s range would
# chase the duck off the edge before it can learn a gait, but squeezing it too
# far is what killed run 2026-11-12: at +-0.12 m/s a *motionless* duck still
# scores exp(-(0.12/0.316)^2) = 0.87 of the Gaussian tracking reward, so standing
# was a near-free lunch (see the anti-standing block below). +-0.12 m/s leaves a
# real gap between standing (6%) and tracking (~95%) while still being a slow,
# pocket-safe walk. The ceiling came down from 0.20 to 0.12 with the 2026-11-14
# geometry fix: the free pocket is 0.53 m deep, not 0.62, and 0.20 m/s simply
# does not fit inside any command segment (see CMD_HOLD_S).
LIN_VEL_X = (-0.12, 0.12)
LIN_VEL_Y = (-0.05, 0.05)   # v3: was +-0.10 -- lateral drift is what walks a
                            # non-steering gait off the seat edge
ANG_VEL_Z = (-0.6, 0.6)

# v3: command magnitudes never near zero (see the "v3: no env is ever asked to
# stand" block in make_microduck_velocity_sofa_env_cfg).
VX_ABS_RANGE = (0.06, 0.12)
WZ_ABS_RANGE = (0.15, 0.60)

# --- How long a command is held: this is a *pocket geometry* constraint ------
# The stock velocity task holds a command for 3-8 s (mjlab
# velocity_env_cfg.py: resampling_time_range=(3.0, 8.0)). On open ground that is
# fine. Here it is fatal, and it is the reason the 2026-11-13 run still stood:
#
#   free pocket is 0.80 m long (y), spawn was y in [-0.45, +0.45]
#   -> the old spawn was 0.05 m OUTSIDE the pocket, on a bolster; even a legal
#      +-0.20 spawn would have had only ~0.10 m left before the bolster
#   -> at 0.20 m/s a *perfect* tracker reaches it in 0.5 s
#   -> but the command is held for 3-8 s, so it had to keep pushing
#   -> `fell_off_sofa` terminates the episode (time_out=False)
#
# So tracking well meant dying inside 2 s, while standing still survived the
# full episode and collected pose(1.0) + upright(2.0) for free. No reward
# weighting can outbid that: the optimum really was "don't move".
#
# Fix: every commanded segment must be physically completable inside the
# pocket. The binding constraint is the bolster face at y = 0.42:
#
#   travel + |spawn_y| + forward reach <= 0.39
#   hold_max * |vx|max + 0.10 + 0.093 <= 0.39      (BODY_HALF_LEN = 0.093)
#   1.45 s * 0.12 m/s + 0.193 = 0.367 <= 0.39      -> 23 mm of slack
#
# build_check.py re-derives that inequality from the asset, so shrinking the
# seat or widening the command range fails the gate instead of a GPU run.
CMD_HOLD_S = (0.8, 1.45)

# Walked off the seat: the trunk drops below seat-top minus this margin, long
# before it reaches the floor (0.12 m standing height on the ground plane).
FALL_OFF_MARGIN = 0.20

SOFA_ENTITY_CFG = EntityCfg(
    spec_fn=get_sofa_spec,
    # Fixed-base + non-articulated: mjlab wraps this in a mocap body, and the
    # reset event below walks it to each env origin.
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)


def make_microduck_velocity_sofa_env_cfg(play: bool = False):
    """Create the sofa variant of the velocity task."""
    cfg = make_microduck_velocity_env_cfg(play=play)

    # --- Scene: add the sofa next to the robot -------------------------------
    cfg.scene.entities = {**cfg.scene.entities, "sofa": SOFA_ENTITY_CFG}
    # The sofa is 2.10 m long; the default 2.0 m grid would let neighbouring
    # sofas touch (and warp's contact detection would then pair them).
    cfg.scene.env_spacing = 3.0

    # --- Feet contact sensor: the ground is the seat, not the plane ----------
    # The secondary below resolves (inside the "sofa" entity) to the single
    # body ``odg_sofa_concept_a`` and is written into the compiled spec as
    # ``sofa/odg_sofa_concept_a`` -- so the air_time reward counts exactly the
    # foot-on-cushion contacts, nothing else. If the sofa asset is ever
    # swapped, update the pattern AND run scripts/sofa/build_check.py, which
    # asserts the sensor actually resolved.
    sensors = []
    for sns in cfg.scene.sensors:
        if sns.name == "feet_ground_contact":
            sns = copy.deepcopy(sns)
            # Point the secondary side at the SOFA body (the stock config matches
            # the literal body "terrain", which on this task can only ever fire
            # after the duck has already fallen off).
            # mjlab 1.3.0 ContactMatch semantics (sensor/contact_sensor.py):
            #   entity=None  -> pattern is a LITERAL MuJoCo name; regex syntax is
            #                   passed straight into sensor refname and fails
            #                   MjSpec.compile ("unrecognized name ... of object").
            #   entity="sofa"-> pattern is expanded with ent.find_bodies() inside
            #                   the sofa entity, then prefixed ("sofa/<name>") when
            #                   written into the spec. Exactly one body must match
            #                   with policy="first".
            # NOTE: the first sofa run used ``secondary=None`` and the air_time
            # reward (weight 3.0) stayed at 0.000 for all 4000 iterations (dead
            # gait incentive). The second run used a regex WITHOUT entity and
            # crashed at env build. Do not regress to either.
            sns.secondary = ContactMatch(
                mode="body", pattern=r"^odg_sofa_concept_a$", entity="sofa"
            )
            sns.secondary_policy = "first"
        sensors.append(sns)
    cfg.scene.sensors = tuple(sensors)

    # --- Reset: robot on the cushion, sofa at the env origin -----------------
    pose_range = cfg.events["reset_base"].params["pose_range"]
    pose_range["x"] = SPAWN_X
    pose_range["y"] = SPAWN_Y
    # z is an *offset* added to the entity's default root state (0,0,0) plus the
    # env origin, so this is "seat top + standing height".
    pose_range["z"] = (SEAT_TOP_Z + 0.115, SEAT_TOP_Z + 0.125)
    pose_range["yaw"] = SPAWN_YAW

    # Mocap props are only positioned by a reset event — without this the sofa
    # would sit at the world origin and only env 0 would have one.
    cfg.events["reset_sofa"] = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},  # no jitter: the sofa must not move between episodes
            "asset_cfg": SceneEntityCfg("sofa"),
        },
    )

    # --- Commands that fit on the cushion -----------------------------------
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X
    command.ranges.lin_vel_y = LIN_VEL_Y
    command.ranges.ang_vel_z = ANG_VEL_Z
    # A commanded segment has to be completable without walking off the seat;
    # see the CMD_HOLD_S block at the top of this module.
    command.resampling_time_range = CMD_HOLD_S
    # `rel_forward_envs` is inherited as 0.2 from the stock task, and its
    # `_resample_command` applies `abs(lin_vel_x).clamp(min=0.3)`. That commands
    # 20% of the batch at 0.30 m/s -- above this task's declared 0.20 m/s
    # ceiling, and precisely the command that leaves the cushion fastest. It
    # also silently breaks the "command is inside `ranges`" invariant that
    # scripts/sofa/verify_mjlab.py asserts. Disable it.
    command.rel_forward_envs = 0.0

    # --- v3: no env is ever asked to stand ----------------------------------
    # Both 4000-iter runs converged to standing. The mechanism (2026-11-14
    # post-mortem, scripts/sofa/README.md §13): with uniform ±0.2 sampling,
    # most commands are near zero, where a motionless duck earns ~1.0 of the
    # Gaussian tracking reward -- standing is what the reward signal says most
    # of the time. Three changes:
    #   1. |vx| ~ U(0.08, 0.20), |wz| ~ U(0.15, 0.80), random sign (fields on
    #      VelocityCommandCommandOnlyCfg): every env is always asked to move;
    #   2. rel_standing_envs = 0 AND its curriculum term removed (the stock
    #     curriculum is a *runtime writer* and would ramp it back up);
    #   3. the turn-in-place bucket off: its lin=0 command is exactly the
    #      standing-shaped command this task must stop rewarding. Turning is
    #      still trained via the |wz| >= 0.15 commands.
    command.rel_standing_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.lin_vel_x_abs_range = VX_ABS_RANGE
    command.ang_vel_z_abs_range = WZ_ABS_RANGE

    # --- Anti-standing recipe (v2 -- kept for provenance, superseded by v3) --
    # Symptoms: 4000 iterations, never a single `fell_off_sofa` termination (the
    # duck never crossed the seat), `air_time` reward stuck at 0.000, and in
    # deployment 0.00 m/s at *every* command up to 1.0 m/s (command sensitivity
    # 0.077 vs 0.914 for the official walker -- see scripts/sofa/probe_policy.py).
    #
    # Mechanism: the stock Gaussian tracking reward (weight 2.0, std=sqrt(0.1)
    # =0.316 m/s) gives a motionless duck exp(-(0.12/0.316)^2)=87% of the reward
    # at the old +-0.12 command ceiling, while every step on a compliant cushion
    # risks a fall and costs an action-rate penalty. Add a competing 2.0-weight
    # head-pose objective that a standing duck satisfies perfectly, and standing
    # is the optimum. Three fixes, all standard:
    #
    #  1. Tighter tracking std (0.316 -> 0.12): at 0.2 m/s, standing now earns
    #     exp(-(0.2/0.12)^2)=0.06 of the term while tracking earns ~0.95.
    #  2. Head-pose tracking demoted (2.0 -> 0.3): it is a secondary comfort
    #     objective here, not the point of the task; left at 2.0 it competes
    #     with locomotion for the same 14 actions.
    #  3. Standing-env curriculum capped (was ramped to 25% of envs by iter
    #     2000): a quarter of the batch rewarding stillness is fine on flat
    #     ground, but here it teaches "freeze" as the default answer.
    #
    # v2 still produced a stander (run 2026-11-13_21-35-25_v2_bounded): with the
    # command distribution uniform over +-0.2, the *average* Gaussian a
    # motionless duck collects is ~0.42 (weight 2.0 -> 0.84, logged 0.824 --
    # matched the standing prediction almost exactly), because near-zero
    # commands are common and cheap. Hence the v3 block above.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.12
    cfg.rewards["track_angular_velocity"].params["std"] = 0.35
    if "head_pose_tracking" in cfg.rewards:
        cfg.rewards["head_pose_tracking"].weight = 0.3

    # --- v3: make stepping pay ----------------------------------------------
    # `feet_air_time` pays 1.0 per foot whose CURRENT air time is inside
    # (threshold_min, threshold_max). The stock threshold_min=0.125 s means a
    # foot must be airborne >6 control steps (50 Hz) to count -- a small duck's
    # shuffle on compliant foam never gets there, so the 3.0-weight gait
    # incentive was dead in both runs (Episode_Reward/air_time == 0.000 for all
    # 4000 iters; Metrics/air_time_mean 0.042 -> 0.000). 0.04 s = 2 control
    # steps: any real foot lift starts paying.
    cfg.rewards["air_time"].params["threshold_min"] = 0.04

    # The stock action_rate curriculum ramps the penalty to -1.0 by iter 1500.
    # That is tuned for flat ground where a gait already exists; here it punishes
    # exactly the oscillation a bootstrapping gait needs (run v2 ended with the
    # weight at -1.0 and a mean action rate of ~0.5 -- flailing while standing).
    # Cap it at the stage-1 value: smoothness pressure without gait suppression.
    if "action_rate_weight" in cfg.curriculum:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": -0.1},
            {"step": 1000 * 24, "weight": -0.2},
            {"step": 2000 * 24, "weight": -0.3},
        ]

    # No standing fraction at all, and no curriculum that would re-ramp it:
    # the stock `standing_envs` curriculum writes rel_standing_envs at runtime
    # (0.02 -> 0.25 on flat), which would silently undo the zero above.
    cfg.curriculum.pop("standing_envs", None)

    # --- Pushes: halved ------------------------------------------------------
    # The stock +-0.3 m/s kick was tuned for open ground; on a cushion it throws
    # the duck off the seat more often than it teaches recovery.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
        }

    # --- Terminations: leaving the cushion ends the episode ------------------
    cfg.terminations["fell_off_sofa"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        time_out=False,
        params={
            "minimum_height": SEAT_TOP_Z - FALL_OFF_MARGIN,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # --- Solver: cushion seams + backrest give more simultaneous contacts ----
    cfg.sim.nconmax = 60          # was 35 (velocity default)
    cfg.sim.mujoco.iterations = 20     # was 10
    cfg.sim.mujoco.ls_iterations = 30  # was 20

    return cfg


# Same network/PPO recipe as the flat velocity task; only the experiment name
# differs so runs land in their own logs/ + wandb group.
MicroduckSofaRlCfg = copy.deepcopy(MicroduckRlCfg)
MicroduckSofaRlCfg.experiment_name = "velocity_sofa"
# The run name is part of the log directory
# (logs/rsl_rl/velocity_sofa/<timestamp>_<run_name>), so it doubles as a
# provenance stamp: any run dir that does *not* say `v3_gait` was trained
# before the 2026-11-14 gait-bootstrap fix. scripts/sofa/ckpt_report.py reads
# the real knobs out of params/env.yaml; this just makes it obvious at `ls`.
# History: (none) = pre-fix, `v2_bounded` = bounded-cushion fix (still a
# stander), `v3_gait` = abs-range commands + air_time threshold + curriculum
# caps (still a stander), `v4_geo` = the 2026-11-14 geometry fix: spawn moved
# 0.365 m forward out of the backrest, command ceiling 0.20 -> 0.12 m/s and hold
# 2.4 -> 1.5 s so a segment fits the real 0.53 x 0.84 m pocket. v4 is kept
# self-consistent but is NOT expected to produce a walker -- see README §15;
# the circle task (`c2_geo`) is the one that can.
MicroduckSofaRlCfg.run_name = "v4_geo"
MicroduckSofaRlCfg.max_iterations = 4_000
MicroduckSofaRlCfg.save_interval = 100

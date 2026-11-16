#!/usr/bin/env python3
"""Measure a trained sofa checkpoint's ACTUAL velocity tracking in the mjlab env.

Runs before export: if the policy does not translate in the very environment it
was trained in, no amount of ONNX/deploy debugging will help.

    .venv/bin/python scripts/sofa/verify_mjlab.py                # latest checkpoint
    .venv/bin/python scripts/sofa/verify_mjlab.py --checkpoint logs/rsl_rl/velocity_sofa/xxx/model_3999.pt
    .venv/bin/python scripts/sofa/verify_mjlab.py --commands 0.2 -0.2 --steps 100

Before reading a FAIL: scripts/sofa/ckpt_report.py prints which config trained
the checkpoint. A run from before the 2026-11-13 task fix *is* a stander, and
re-measuring it with these gates cannot change that.

Why a sweep and not a single command
------------------------------------
The 2026-11-12 run (`velocity_sofa`, 4000 iters) produced a policy that stands:
`air_time` reward 0.000 for the whole run, zero `fell_off_sofa` terminations, and
in deployment 0.00 m/s at every command. A single-command check at the *mean*
commanded speed can hide this, because a stander still scores ~87% of the
Gaussian tracking reward at small commands. Sweeping negative/positive commands
and comparing "achieved / commanded" exposes it immediately: a stander scores
~0 on both.

mjlab 1.3.0 API notes (learned the hard way -- see scripts/sofa/README.md §10)
---------------------------------------------------------------------------
* `RslRlVecEnvWrapper.reset()` -> `(TensorDict obs, extras)`; `.step()` returns a
  4-tuple `(obs, rew, dones, extras)`. The raw env returns a *dict* obs, which
  the rsl_rl policy cannot consume, so every env interaction goes through the
  wrapper.
* `CommandManager.reset()` **requires** `env_ids` (a tensor). We never call it
  directly: `env.reset()` resamples the command for us.
* Root state is exposed as `robot.data.root_link_pose_w` (7: pos+quat),
  `root_link_lin_vel_b` / `root_link_ang_vel_b` (body frame). There is no
  `root_state_w` on the entity.
* `UniformVelocityCommand._resample_command` rewrites the command behind your
  back: `rel_forward_envs` forces lin_vel_x >= 0.3 and zeroes the rest,
  `rel_world_envs` rotates it, `rel_standing_envs` zeroes it, and
  `init_velocity_prob` injects it into the robot state. With one env that is
  effectively random, so this script disables all of them and then *asserts*
  the command it thinks it is testing.

Exit code: 0 = tracking acceptable, 1 = looks like a standing policy.
"""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import asdict
from pathlib import Path

TASK_ID = "Mjlab-Velocity-Sofa-MicroDuck"

# Room available in front of a centred spawn before the armrest, in metres
# (cushion y in [-0.78, 0.78], spawn y in [-0.20, 0.20]; see sofa_constants.py).
# A sweep that asks for more travel than this judges a *good* policy as FELL,
# because it walks into the armrest while faithfully tracking the command.
CUSHION_ROOM_M = 0.58


def latest_checkpoint() -> Path:
    logs = sorted(glob.glob("logs/rsl_rl/velocity_sofa/*/model_*.pt"))
    if not logs:
        raise SystemExit(
            "no checkpoint found under logs/rsl_rl/velocity_sofa/*/model_*.pt -- "
            "pass --checkpoint explicitly"
        )
    return Path(max(logs, key=lambda p: (p.rsplit("_", 1)[-1].split(".")[0], p)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="model_XXXX.pt (default: latest)")
    ap.add_argument("--commands", type=float, nargs="+", default=[0.20, -0.20],
                    help="forward velocity commands to sweep (m/s)")
    ap.add_argument("--yaw", type=float, nargs="*", default=[0.0],
                    help="yaw-rate commands to sweep (rad/s); 0 is always included")
    ap.add_argument("--steps", type=int, default=100,
                    help="control steps measured per command (default 100 = 2 s: "
                         "0.20 m/s * 2 s = 0.40 m, which fits the cushion)")
    ap.add_argument("--warmup", type=int, default=25,
                    help="steps discarded per command (default 25 = 0.5 s)")
    ap.add_argument("--min-ratio", type=float, default=0.35,
                    help="pass if |achieved| / |commanded| >= this (default 0.35). "
                         "Calibrated 2026-11-14 against measured references on THIS "
                         "cushion: the official flat-trained alpha_walking gait "
                         "achieves ~0.25-0.55 (bursty) and drifts off the edge, and "
                         "~0.3-0.4 on flat ground. 0.55 was unachievable physics, "
                         "not a policy standard; 0.35 still catches a stander "
                         "(ratio ~0) with a wide margin.")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    print(f"checkpoint: {ckpt}\ndevice:     {device}\ntask:       {TASK_ID}\n")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    # A single env, so the sweep is deterministic and quick.
    twist = env_cfg.commands["twist"]
    twist.resampling_time_range = (1e9, 1e9)  # never resample; ranges pin the cmd
    # Neutralise every resampling rule that would silently rewrite our command
    # (all of these are fields of UniformVelocityCommandCfg).
    # NOTE: do NOT touch `heading_command` -- the command term enforces
    # `heading_command == (ranges.heading is not None)` and raises otherwise
    # ("ranges.heading is set but heading_command=False"). Zeroing
    # `rel_heading_envs` is enough: `is_heading_env` stays all-False, so the
    # heading controller never overwrites ang_vel_z.
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0
    # microduck's own command term (VelocityCommandCommandOnlyCfg) adds a
    # turn-in-place bucket: with probability p it forces lin=0 and
    # |ang| in [0.4*max, max]. The sofa task sets p=0.10, so without this a
    # 0.20 m/s case can silently become a pure spin.
    if hasattr(twist, "rel_turn_in_place_envs"):
        twist.rel_turn_in_place_envs = 0.0
    # v3 anti-standing: the sofa task sets *_abs_range so training samples
    # |command| away from zero. The resampler applies those AFTER the uniform
    # draw from `ranges.*` and would overwrite our pinned command mid-sweep --
    # disable them here (verify pins exact commands via `ranges`).
    for attr in ("lin_vel_x_abs_range", "lin_vel_y_abs_range", "ang_vel_z_abs_range"):
        if hasattr(twist, attr):
            setattr(twist, attr, None)
    # Curricula are *runtime writers* to the same fields: `standing_envs` ramps
    # rel_standing_envs from 0.02 up to 0.05 on this task, which would undo the
    # zeros above mid-sweep. There is nothing to learn in an eval run, so drop
    # all curricula.
    env_cfg.curriculum = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = env.unwrapped.scene["robot"]
    # The built command term (the cfg above is only the template; this object is
    # what resamples and what `.command` reads).
    twist_term = env.unwrapped.command_manager.get_term("twist")
    want_device = torch.device(device)

    yaws = [0.0] + [y for y in args.yaw if y != 0.0]
    cases = [(vx, wy) for wy in yaws for vx in args.commands]

    print(f"{'cmd vx':>8s} {'cmd wyaw':>9s} {'ach vx':>8s} {'ach yaw':>8s} "
          f"{'ratio':>6s} {'trunk z':>8s}  verdict")
    print("-" * 66)

    results = []
    for vx, wy in cases:
        # Pin the command for this segment.
        twist.ranges.lin_vel_x = (vx, vx)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (wy, wy)
        # env.reset() -> command_manager.reset(env_ids) -> _resample inside the
        # ranges above, so the sampled command is exactly (vx, 0, wy).
        obs, _extras = wrapped.reset()
        # CommandManager.reset() only resamples; `_update_command()` (which is
        # where standing/world/heading post-processing happens) runs on the next
        # compute(). dt=0 applies it without advancing any timer, so the
        # assertion below checks the command that will actually be used.
        env.unwrapped.command_manager.compute(0.0)

        # Guard against a silently rewritten command: a zero command would make
        # a walking policy look like a stander, and a forced +0.3 would make a
        # stander look alive. Fail loudly instead of reporting a bogus ratio.
        cmd_now = twist_term.command[0].detach()
        cmd_want = torch.tensor([vx, 0.0, wy], device=want_device)
        if not torch.allclose(cmd_now, cmd_want, atol=1e-5):
            print(f"{vx:8.2f} {wy:9.2f} {'-':>8s} {'-':>8s} {'-':>6s} {'-':>8s}  "
                  f"BAD COMMAND (got {[round(v, 3) for v in cmd_now.tolist()]})")
            results.append((vx, wy, 0.0, 0.0, 0.0, 0.0, True))
            continue

        # A sweep longer than the cushion is not a stricter test, it is a wrong
        # one: the duck reaches the armrest while faithfully tracking, and the
        # case gets reported as FELL. Cap the window at the available room.
        step_dt = float(env.unwrapped.step_dt)
        n_steps = args.warmup + args.steps
        if abs(vx) > 1e-6:
            travel = abs(vx) * n_steps * step_dt
            if travel > CUSHION_ROOM_M:
                n_steps = max(
                    int(CUSHION_ROOM_M / (abs(vx) * step_dt)), args.warmup + 20
                )
                print(f"note: {vx:+.2f} m/s over {args.warmup + args.steps} steps asks "
                      f"for {travel:.2f} m of travel but only {CUSHION_ROOM_M:.2f} m "
                      f"of cushion exists -- measuring {n_steps} steps instead")

        vx_acc, wy_acc, z_acc, n = 0.0, 0.0, 0.0, 0
        fell = False
        for step in range(n_steps):
            with torch.inference_mode():
                action = policy(obs)
            obs, _rew, dones, _extras = wrapped.step(action)
            z = float(robot.data.root_link_pose_w[0, 2].item())
            if step >= args.warmup:
                vx_acc += float(robot.data.root_link_lin_vel_b[0, 0].item())
                wy_acc += float(robot.data.root_link_ang_vel_b[0, 2].item())
                z_acc += z
                n += 1
            if z < 0.30:  # way below seat top (0.518) -> it left the cushion
                fell = True
                break
            if bool(dones[0].item()):
                break

        if n == 0:
            results.append((vx, wy, 0.0, 0.0, 0.0, 0.0, fell))
            print(f"{vx:8.2f} {wy:9.2f} {'-':>8s} {'-':>8s} {'-':>6s} {'-':>8s}  "
                  f"{'FELL/TERMINATED' if fell else 'no data'}")
            continue

        ach_vx = vx_acc / n
        ach_wy = wy_acc / n
        z_mean = z_acc / n
        ref = abs(vx) + abs(wy)
        ratio = (abs(ach_vx) + abs(ach_wy)) / ref if ref > 1e-6 else 1.0
        # Sign matters: walking the wrong way is not tracking.
        if vx * ach_vx < 0 or wy * ach_wy < 0:
            ratio = 0.0
        ok = ratio >= args.min_ratio and not fell
        verdict = "ok" if ok else ("FELL" if fell else "NOT TRACKING")
        results.append((vx, wy, ach_vx, ach_wy, z_mean, ratio, fell))
        print(f"{vx:8.2f} {wy:9.2f} {ach_vx:8.3f} {ach_wy:8.3f} {ratio:6.2f} "
              f"{z_mean:8.3f}  {verdict}")

    env.close()

    moving_cases = [r for r in results if abs(r[0]) + abs(r[1]) > 1e-6]
    passed = [r for r in moving_cases if r[5] >= args.min_ratio and not r[6]]
    print()
    if not moving_cases:
        print("no moving command tested -- nothing to judge")
        return 1
    print(f"tracking pass rate: {len(passed)}/{len(moving_cases)} commands "
          f">= {args.min_ratio:.0%} of commanded speed")
    if len(passed) == len(moving_cases):
        print("PASS: the checkpoint actually walks in the training env.")
        print("      next: bash scripts/sofa/modelscope_run.sh export")
        return 0
    print("FAIL: the checkpoint does not track the commanded velocity in its own")
    print("      training env -- it is (at least partly) a standing policy.")
    print("      Do NOT export/upload it. Check the v3 anti-standing settings in")
    print("      src/mjlab_microduck/tasks/microduck_velocity_sofa_env_cfg.py")
    print("      (vx/wz abs-range sampling, air_time threshold_min, standing")
    print("      curriculum removal, action_rate cap) and retrain. First run")
    print("      'bash scripts/sofa/modelscope_run.sh report' to confirm which")
    print("      config trained this checkpoint.")
    return 1


if __name__ == "__main__":
    import traceback

    try:
        _code = main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - any crash is a script/env bug
        traceback.print_exc()
        print()
        print("=" * 70)
        print("SCRIPT/ENVIRONMENT ERROR (exit 2) -- this is NOT a verdict on the")
        print("checkpoint. The policy was never actually measured. Fix the")
        print("verify script or the env config first; do not retrain because of")
        print("this message.")
        print("=" * 70)
        _code = 2
    sys.exit(_code)

#!/usr/bin/env python3
"""Measure whether a circle-task checkpoint actually ORBITS the cushion.

The straight-line gate (`verify_mjlab.py`) cannot judge this task: it pins a
velocity through `ranges`, while a circle checkpoint's command comes from the
per-episode orbit geometry. More importantly, its pass criterion is "body
speed over ground", and a duck can spin on the spot and satisfy a yaw-rate
command while never moving — the exact local optimum the circle task is built
to avoid. So this script judges the only thing that matters here:

    did the robot's POSITION sweep around the circle centre?

It is measured from the world trajectory, not from the gyro. For each env:

  * `w ratio`  = (arc actually swept / elapsed time) / |omega_cmd|
                 ~0 standing or spinning in place, 1.0 perfect, <0 backwards
  * `r err`    = RMS of |  dist-to-centre - r_cmd |
                 ~0 on the commanded orbit, >0.05 m = a different circle
  * `laps`     = net signed revolutions in the measurement window
  * `z`        = mean trunk height (drops well below the seat when it falls)

Pass = every surviving env has w ratio >= --min-ratio, r err <= --max-r-err,
and never left the cushion.

    .venv/bin/python scripts/sofa/verify_circle.py
    .venv/bin/python scripts/sofa/verify_circle.py --checkpoint logs/rsl_rl/velocity_circle/<run>/model_3999.pt
    .venv/bin/python scripts/sofa/verify_circle.py --envs 16 --steps 600

Exit codes: 0 = orbits, 1 = does not orbit (stander/spinner/drifts off),
2 = script or environment error (NOT a verdict on the checkpoint).
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from dataclasses import asdict
from pathlib import Path

TASK_ID = "Mjlab-Circle-Sofa-MicroDuck"

# Below the seat top (0.518) by 0.2 m -> it left the cushion.
FELL_Z = 0.30


def latest_checkpoint() -> Path:
    logs = sorted(glob.glob("logs/rsl_rl/velocity_circle/*/model_*.pt"))
    if not logs:
        raise SystemExit(
            "no checkpoint under logs/rsl_rl/velocity_circle/*/model_*.pt -- "
            "pass --checkpoint explicitly"
        )
    return Path(max(logs, key=lambda p: (p.rsplit("_", 1)[-1].split(".")[0], p)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="model_XXXX.pt (default: latest)")
    ap.add_argument("--envs", type=int, default=8, help="parallel envs measured (default 8)")
    ap.add_argument("--steps", type=int, default=500,
                    help="measured control steps per env (default 500 = 10 s at 50 Hz)")
    ap.add_argument("--warmup", type=int, default=50,
                    help="steps discarded first (default 50 = 1 s)")
    ap.add_argument("--min-ratio", type=float, default=0.40,
                    help="pass if achieved/commanded orbit rate >= this (default 0.40). "
                         "A stander or an in-place spinner scores ~0, so the gate is "
                         "about detecting 'moved around the circle at all', not about "
                         "a tight speed spec -- on foam, 0.4 is already a real gait.")
    ap.add_argument("--max-r-err", type=float, default=0.05,
                    help="pass if RMS radial error <= this, metres (default 0.05)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner

    from mjlab_microduck.tasks import mdp as microduck_mdp

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    torch.manual_seed(args.seed)

    print(f"checkpoint: {ckpt}\ndevice:     {device}\ntask:       {TASK_ID}\n")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.envs
    # An eval run must be reproducible and undisturbed:
    #  * curricula rewrite ranges/masks at runtime -- nothing to learn here;
    #  * push_robot fires every 0.5-1.0 s in play mode and would be measured as
    #    a radial excursion, i.e. a good policy reported as drifting.
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    agent_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    unb = env.unwrapped
    robot = unb.scene["robot"]
    st = microduck_mdp.circle_state(unb)

    obs, _extras = wrapped.reset()
    # The reset event already placed every env on its orbit; snapshot it before
    # the first step, because the state is overwritten on the next reset.
    radius = st.radius.clone()
    omega = st.omega.clone()
    center = st.center.clone()
    theta0 = st.theta0.clone()

    dt = float(unb.step_dt)
    total = args.warmup + args.steps
    z_acc = torch.zeros(args.envs, device=device)
    z_n = torch.zeros(args.envs, device=device)
    pts = []

    for step in range(total):
        with torch.inference_mode():
            action = policy(obs)
        obs, _rew, dones, _extras = wrapped.step(action)
        pos = robot.data.root_link_pos_w[:, :2].clone()
        z = robot.data.root_link_pose_w[:, 2].clone()
        died = (z < FELL_Z) | dones.bool()
        if step >= args.warmup:
            pts.append((pos, died))
            z_acc += torch.where(died, torch.zeros_like(z), z)
            z_n += (~died).float()

    if not pts:
        env.close()
        print("FAIL: no steps measured (warmup >= steps?)")
        return 1

    # Two passes over the recorded trajectory, per env:
    #   1. arc swept between consecutive poses (skipped on the step an env dies:
    #      the post-termination pose belongs to the NEXT episode's circle and
    #      would read as a giant angular jump);
    #   2. RMS radial error against the commanded radius.
    swept = torch.zeros(args.envs, device=device)
    alive = torch.ones(args.envs, dtype=torch.bool, device=device)
    live_masks = []
    prev_theta = None
    for pos, died in pts:
        theta = torch.atan2(pos[:, 1] - center[:, 1], pos[:, 0] - center[:, 0])
        if prev_theta is not None:
            d = torch.atan2(torch.sin(theta - prev_theta), torch.cos(theta - prev_theta))
            swept += torch.where(alive, d, torch.zeros_like(d))
        prev_theta = theta
        live_masks.append(alive.clone())
        alive &= ~died

    # Radial error over the same window, masked by the SAME "still on the first
    # episode" flag: `dones` is per-step, so after an auto-reset an env stands on
    # a NEW circle and its dist-to-OLD-centre says nothing about this run.
    r_err_sq = torch.zeros(args.envs, device=device)
    r_n = torch.zeros(args.envs, device=device)
    for (pos, died), live in zip(pts, live_masks):
        err = torch.norm(pos - center, dim=-1) - radius
        live = live & ~died
        r_err_sq += torch.where(live, err * err, torch.zeros_like(err))
        r_n += live.float()

    elapsed = args.steps * dt
    w_cmd = omega.abs()
    # Signed projection onto the commanded direction: orbiting BACKWARDS must
    # score ~0, not "fast". |swept| would let a policy that circles the wrong
    # way pass the gate with flying colours.
    w_ach_signed = swept / elapsed
    ratio = (w_ach_signed * torch.sign(omega)) / w_cmd.clamp(min=1e-6)
    w_ach = w_ach_signed.abs()
    r_err = torch.sqrt((r_err_sq / r_n.clamp(min=1.0)).clamp(min=0.0))
    laps = swept * torch.sign(omega) / (2.0 * math.pi)
    z_mean = z_acc / z_n.clamp(min=1.0)

    print(f"{'env':>3s} {'r cmd':>6s} {'w cmd':>7s} {'w ach':>7s} {'ratio':>6s} "
          f"{'r err':>7s} {'laps':>6s} {'z':>6s}  verdict")
    print("-" * 62)

    passed, failed = 0, 0
    for i in range(args.envs):
        ok_alive = bool(alive[i])
        ok_ratio = float(ratio[i]) >= args.min_ratio
        ok_radius = float(r_err[i]) <= args.max_r_err
        z = float(z_mean[i])
        ok_z = z > 0.35
        ok = ok_alive and ok_ratio and ok_radius and ok_z
        if not ok_alive:
            verdict = "DIED (left cushion / timeout)"
        elif not ok_z:
            verdict = "FELL"
        elif not ok_ratio:
            verdict = "NO ORBIT (stander/spinner)"
        elif not ok_radius:
            verdict = "DRIFTED OFF RADIUS"
        else:
            verdict = "ok"
            passed += 1
        if not ok:
            failed += 1
        # A dead env's numbers are meaningless; print them anyway but flagged.
        print(f"{i:3d} {float(radius[i]):6.3f} {float(omega[i]):+7.3f} "
              f"{float(w_ach[i]):7.3f} {float(ratio[i]):6.2f} "
              f"{float(r_err[i]):7.4f} {float(laps[i]):+6.2f} {z:6.3f}  {verdict}")

    env.close()

    n = args.envs
    print()
    print(f"orbit pass rate: {passed}/{n} envs  "
          f"(w ratio >= {args.min_ratio:.2f}, r err <= {args.max_r_err:.3f} m)")
    if passed == n:
        print("PASS: the checkpoint walks its circle(s) in the training env.")
        print("      next: bash scripts/sofa/modelscope_run.sh export")
        return 0

    print("FAIL: the checkpoint does not orbit the cushion in its own training")
    print("      env. Do NOT export it.")
    print("      Read the curve `Episode_Reward/circle_progress` in the run dir:")
    print("        ~1.0 -> orbits (this script disagrees => env/verify mismatch)")
    print("        ~0.02 -> stands or spins in place => reward never reached it")
    print("      Then check, in order: air_time threshold_min (does the gait ever")
    print("      pay?), circle_progress weight vs upright+pose, and the arc-rate")
    print("      sigma. A run named anything but `c1_circle` was not trained on")
    print("      this task at all -- run 'bash scripts/sofa/modelscope_run.sh report'.")
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
        print("checkpoint. The policy was never actually measured.")
        print("=" * 70)
        _code = 2
    sys.exit(_code)

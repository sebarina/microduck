#!/usr/bin/env python3
"""Probe an exported microduck policy for command sensitivity.

Why this exists
---------------
A velocity policy can *look* fine in the viewer (it stands, it balances, trunk
height is stable) while actually being a standing policy: PPO found that a
motionless stance still collects most of the Gaussian tracking reward when the
commanded speeds are small, and it avoids every fall risk. That failure was
shipped once already (run `velocity_sofa` 2026-11-12: 4000 iterations, 0
`fell_off_sofa` terminations, 0.00 m/s at every command in deployment).

This probe answers one question without a GPU, a simulator, or a trained env:
**does the policy's action actually depend on the velocity command?** It feeds
synthetic observations (61-D contract) with only the command slot varying and
measures how much the action moves. A walking policy reacts strongly; a
standing policy is nearly invariant.

Usage
-----
    python scripts/sofa/probe_policy.py policy.onnx
    python scripts/sofa/probe_policy.py policy.onnx --baseline other.onnx

Exit code 0 = policy is command-sensitive enough, 1 = likely a standing policy
(so a CI/upload gate can fail on it).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

N_OBS = 61
N_ACT = 14
CMD_SLICE = slice(48, 51)  # [vx, vy, wyaw] then head_pose(4) + body_pose(6)


def load(path: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    if int(shape[-1]) != N_OBS:
        raise SystemExit(f"{path}: input is {shape}, expected [1,{N_OBS}]")
    return sess, name


def act(sess, name, obs):
    out = sess.run(None, {name: obs.astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float64).reshape(-1)


def probe(path: str, verbose: bool = True):
    sess, name = load(path)

    # Neutral stance: joints at default (joint_pos obs is relative to default,
    # so zeros == the keyframe pose), gravity straight down, no motion.
    base = np.zeros(N_OBS, dtype=np.float32)
    base[5] = -1.0  # projected gravity z

    a0 = act(sess, name, base[None, :])

    rows = []
    for vx in (-0.30, -0.15, 0.0, 0.15, 0.30):
        obs = base.copy()
        obs[CMD_SLICE][0] = vx
        a = act(sess, name, obs[None, :])
        rows.append(("vx", vx, a))

    for wyaw in (-0.8, 0.0, 0.8):
        obs = base.copy()
        obs[CMD_SLICE][2] = wyaw
        a = act(sess, name, obs[None, :])
        rows.append(("wyaw", wyaw, a))

    # Sensitivity = RMS action difference vs the zero-command action, in rad,
    # normalised by the action RMS of the zero-command response. A policy that
    # ignores the command stays near 0; a real velocity policy is O(1).
    a0_rms = float(np.sqrt(np.mean(a0**2))) + 1e-9
    lin_sens, ang_sens = [], []
    if verbose:
        print(f"policy: {path}")
        print(f"  zero-cmd action rms = {a0_rms:.4f} rad, max |a| = {np.abs(a0).max():.4f}")
        print(f"  {'cmd':>8s} {'value':>7s} {'dtRMS':>8s} {'sens':>6s}  first 4 actions")
    for kind, val, a in rows:
        d = float(np.sqrt(np.mean((a - a0) ** 2)))
        sens = d / a0_rms
        if kind == "vx" and val != 0.0:
            lin_sens.append(abs(sens))
        if kind == "wyaw" and val != 0.0:
            ang_sens.append(abs(sens))
        if verbose:
            print(f"  {kind:>8s} {val:+7.2f} {d:8.4f} {sens:6.3f}  {np.round(a[:4], 3)}")

    lin = float(np.mean(lin_sens)) if lin_sens else 0.0
    ang = float(np.mean(ang_sens)) if ang_sens else 0.0
    if verbose:
        print(f"\nlinear-command sensitivity = {lin:.3f}   yaw-command sensitivity = {ang:.3f}")
    return lin, ang


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy", help="exported ONNX policy (obs[1,61] -> actions[1,14])")
    ap.add_argument("--baseline", help="reference policy to compare against (e.g. the official walker)")
    ap.add_argument("--min-lin", type=float, default=0.35,
                    help="minimum linear-command sensitivity to pass (default 0.35)")
    ap.add_argument("--min-ang", type=float, default=0.35,
                    help="minimum yaw-command sensitivity to pass (default 0.35)")
    args = ap.parse_args()

    lin, ang = probe(args.policy)
    if args.baseline:
        print("\n--- baseline ---")
        blin, bang = probe(args.baseline)
        print(f"\nbaseline: lin {blin:.3f} / yaw {bang:.3f}   ->   "
              f"candidate: lin {lin:.3f} / yaw {ang:.3f}")

    ok = lin >= args.min_lin and ang >= args.min_ang
    print()
    if ok:
        print("PASS: policy reacts to the velocity command like a walking policy.")
    else:
        print("FAIL: policy barely reacts to the velocity command -- it is very "
              "likely a standing policy.\n"
              "      Do NOT ship it. Fix the task spec (command range / tracking "
              "std / competing rewards) and retrain.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

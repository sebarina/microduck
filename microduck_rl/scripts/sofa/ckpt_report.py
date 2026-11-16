#!/usr/bin/env python3
"""Which config trained this checkpoint -- and how did that run actually go?

Why this exists
---------------
The 2026-11-13 fix (wider command range, tighter track std, demoted head-pose
weight, contact sensor re-pointed at the sofa) only means anything if the
checkpoint under test was trained **after** it. Twice now a gate has been
argued with while it was pointing at the previous run's weights, so this script
answers the provenance question from the artefacts the trainer writes next to
the weights, not from memory:

    logs/rsl_rl/<experiment>/<timestamp>_<run_name>/params/env.yaml  <- exact cfg
    logs/rsl_rl/<experiment>/<timestamp>_<run_name>/events.out.*     <- metrics

It prints, for the selected checkpoint:

  * path / mtime / iteration, and every other run of the same experiment
    (so "there are two runs and you picked the old one" is visible at a glance);
  * the four anti-standing knobs as they were **at training time**;
  * the training metrics that tell a stander apart from a walker
    (air_time reward, tracking reward, fell_off_sofa count, episode length).

Usage
-----
    python scripts/sofa/ckpt_report.py                  # newest checkpoint
    python scripts/sofa/ckpt_report.py --all            # one line per run
    python scripts/sofa/ckpt_report.py --checkpoint logs/rsl_rl/.../model_3999.pt

Exit codes: 0 = trained with the post-2026-11-13 config (or provenance
unreadable -- see the printed note), 3 = trained BEFORE the fix, i.e. the
numbers it produces say nothing about the fix.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENT = "velocity_sofa"
RUN_GLOB = f"logs/rsl_rl/{EXPERIMENT}/*/model_*.pt"

# Task id -> run directory (=== RslRlOnPolicyRunnerCfg.experiment_name).
# `modelscope_run.sh` exports $TASK, so every stage (verify's provenance gate
# included) inspects the experiment it is actually about instead of whichever
# run happens to have the newest mtime.
TASK_EXPERIMENTS = {
    "Mjlab-Velocity-Sofa-MicroDuck": "velocity_sofa",
    "Mjlab-Circle-Sofa-MicroDuck": "velocity_circle",
}


def set_experiment(name: str) -> None:
    global EXPERIMENT, RUN_GLOB
    EXPERIMENT = name
    RUN_GLOB = f"logs/rsl_rl/{EXPERIMENT}/*/model_*.pt"


def experiment_for_task(task_id: str | None) -> str | None:
    return TASK_EXPERIMENTS.get(task_id or "")

# Post-fix values. Three generations exist (see scripts/sofa/README.md §13/§15):
#   v2_bounded (2026-11-13): bounded commands + tight track std  -> still stood
#   v3_gait    (2026-11-14): the v2 knobs PLUS the gait-bootstrap knobs below.
#   v4_geo     (2026-11-14): the geometry fix -- spawn moved 0.365 m forward out
#              of the backrest, command ceiling 0.20 -> 0.12 m/s, hold 2.4 ->
#              1.45 s. EVERY earlier checkpoint was trained with the duck spawned
#              inside a 0.98 m backrest, so 2.4 / 0.2 below are the values that
#              make an old run look "current" only by accident; v4's numbers are
#              what a valid straight-line sofa run must now show.
# A checkpoint counts as current only when every knob matches the v4 values.
POST_FIX = {
    "track_std": 0.12,
    "head_pose_weight": 0.3,
    "cmd_hold_max_s": 1.45,
    "rel_forward_envs": 0.0,
    # v3:
    "air_threshold_min": 0.04,
    "rel_standing_envs": 0.0,
    # v4_geo: 0.2 -> 0.12 m/s, because the free pocket is 0.80 m long and
    # 1.45 s * 0.20 m/s + 0.10 (spawn) + 0.093 (body) = 0.483 m does not fit.
    "vx_abs_range_max": 0.12,
}

# Circle task (`velocity_circle`): a different task, so a different set of
# knobs. Notably `resampling_time_range` is 1e6 s there on purpose (the orbit is
# fixed for the episode), which would fail the straight-line `cmd_hold` check --
# hence a separate table rather than one shared list.
#
# Two circle recipes exist and BOTH stay readable, because the first one is the
# evidence for the second:
#   c1_circle  Gaussian arc rate, r<=0.20, 20 s. Ran 4000 iters, policy marched
#              in place (air_time 0.37, circle_progress 0.02, verify w_ach 0.01).
#   c2_short   linear arc rate, radius gated on travel, r<=0.15, 12 s.
#   c2_geo     the same recipe on the MEASURED seat pocket: centre (+0.235,
#              -0.010) instead of (-0.14, 0), r<=0.12, omega 0.50-0.90, 12 s.
#              (c2_geo is what `run_name` now stamps; the recipe key stays "c2"
#              because the reward function name is what identifies a recipe, and
#              it did not change -- only the geometry it is evaluated on.)
# Detected from the reward FUNCTION NAME recorded in env.yaml, not from a run
# name: the function is what the reward actually did.
CIRCLE_RECIPES = {
    "c1": {
        "progress_func": "circle_progress_reward",
        "progress_weight": 2.0,
        "radius_weight": 1.0,
        "radius_max": 0.20,
        "air_weight": 3.0,
    },
    "c2": {
        "progress_func": "circle_progress_linear_reward",
        "progress_weight": 2.5,
        "radius_weight": 0.6,
        # 0.12 since the 2026-11-14 geometry fix. The pocket is 0.53 m deep and
        # the duck's worst footprint corner is 0.128 m from its root, so an orbit
        # of r + 0.128 has to fit in +-0.265 m: r <= 0.137, and 0.12 leaves
        # 17 mm. The earlier 0.15 (and c1's 0.20) came from a 0.62 m "depth"
        # that was the seat boxes' AABB, whose rear third is solid backrest.
        # build_check.py section 0e asserts this against a ray-cast of the asset.
        "radius_max": 0.12,
        "air_weight": 1.5,
    },
}


def circle_recipe_of(func: str | None) -> str | None:
    """Which circle recipe a run used, from its arc-rate function name."""
    if not func:
        return None
    for name, spec in CIRCLE_RECIPES.items():
        if func.endswith(spec["progress_func"]):
            return name
    return None


# --------------------------------------------------------------------------- #
# artifact resolution
# --------------------------------------------------------------------------- #
def iteration_of(path: Path) -> int:
    m = re.search(r"model_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def all_checkpoints() -> list[Path]:
    return [Path(p) for p in glob.glob(RUN_GLOB)]


def newest_checkpoint() -> Path:
    """Newest by (iteration, mtime) across every run of the experiment."""
    ckpts = all_checkpoints()
    if not ckpts:
        raise SystemExit(
            f"no checkpoint under {RUN_GLOB} -- pass --checkpoint explicitly"
        )
    return max(ckpts, key=lambda p: (iteration_of(p), p.stat().st_mtime))


# --------------------------------------------------------------------------- #
# env.yaml (yaml.dump of a dataclass -> python object tags, so parse as text)
# --------------------------------------------------------------------------- #
def _block(text: str, key: str) -> str | None:
    """Return the indented body that follows the first `key:` line."""
    m = re.search(rf"^(\s*){re.escape(key)}:[ \t]*(.*)$", text, re.M)
    if not m:
        return None
    indent = len(m.group(1))
    out: list[str] = []
    for line in text[m.end():].splitlines():
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        out.append(line)
    return "\n".join(out)


def _scalar(text: str | None, key: str) -> str | None:
    """Read `key:` from a yaml.dump fragment, tolerating python object tags.

    mjlab dumps the env cfg with ``yaml.dump`` on a dataclass, so scalars can be
    written as ``key: !!python/tuple`` followed by ``- 3.0`` list items, and
    strings may carry ``!!python/object:...`` tags. Plain ``yaml.safe_load``
    cannot read that, so parse the lines instead.
    """
    if not text:
        return None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(key)}:[ \t]*(.*)$", line)
        if not m:
            continue
        indent, rest = len(m.group(1)), m.group(2).strip()
        if rest.startswith("!!python/name:"):
            # A function reference is written as
            #     func: !!python/name:mjlab_microduck.tasks.mdp.reset_on_circle ''
            # i.e. the NAME IS IN THE TAG and the payload is just an empty
            # string. Dropping the whole tag (as the generic branch below does)
            # turns every callable into None, which made this report declare a
            # correctly-spawned circle run "PRE-CIRCLE".
            rest = rest.split("!!python/name:", 1)[1].split(" ")[0]
        elif rest.startswith("!!"):
            tok, _, tail = rest.partition(" ")
            rest = tail.strip() if tail else ""
        if rest and not rest.startswith("-"):
            return rest.strip("\"'")
        vals = []
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            depth = len(nxt) - len(nxt.lstrip())
            item = re.match(r"^\s*-\s*(\S.*?)\s*$", nxt)
            # yaml.dump puts block sequences at the *same* indent as their key,
            # so a `- value` line at `depth == indent` still belongs to us.
            if item and depth >= indent:
                vals.append(item.group(1).strip("\"'"))
                continue
            if depth <= indent:
                break
            break
        return ", ".join(vals) if vals else (rest or None)
    return None


def _num(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s.split(",")[0].strip())
    except ValueError:
        return None


def _num_max(s: str | None) -> float | None:
    """Largest number in a value like ``1.2, 2.4`` (a `(min, max)` range)."""
    if not s:
        return None
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]
    return max(vals) if vals else None


def _circle_markers(text: str) -> list[str]:
    """Identifiers that can ONLY come from the circle task's cfg.

    The provenance gate needs one piece of positive evidence it can trust. The
    parsed structure is brittle (functions are `!!python/name:` tags, matchers
    are tagged objects, nesting depth is whatever yaml.dump chose), but these
    identifiers are literal strings: `circle_progress` is a reward term this
    task invented, so seeing it in the run's own env.yaml proves the run was
    trained on the circle cfg -- no matter how the rest of the file parses.
    """
    return [
        needle
        for needle in (
            "circle_progress",
            "reset_on_circle",
            "CircleVelocityCommand",
            "circle_radius",
        )
        if needle in text
    ]


def read_env_yaml(run_dir: Path) -> dict[str, object]:
    p = run_dir / "params" / "env.yaml"
    if not p.exists():
        return {}
    text = p.read_text(errors="replace")
    rewards = _block(text, "rewards") or ""
    commands = _block(text, "commands") or ""
    twist = _block(commands, "twist") or ""
    air_time = _block(rewards, "air_time") or ""
    sensors = _block(text, "sensors") or ""
    contact = _block(sensors, "feet_ground_contact") or ""
    # `secondary` is a ContactMatch object. Its nesting depth under the sensor
    # term depends on how yaml.dump tags it, so fall back to the whole sensor
    # subtree instead of insisting on one level: both keys are unique there.
    secondary = _block(contact, "secondary") or contact
    events = _block(text, "events") or ""
    reset_base = _block(events, "reset_base") or ""
    progress = _block(rewards, "circle_progress") or ""
    radius_range = _scalar(twist, "radius_range")
    return {
        "track_std": _scalar(_block(rewards, "track_linear_velocity") or "", "std"),
        "head_pose_weight": _scalar(_block(rewards, "head_pose_tracking") or "", "weight"),
        "air_threshold_min": _scalar(air_time, "threshold_min"),
        "cmd_hold": _scalar(twist, "resampling_time_range"),
        "lin_vel_x": _scalar(twist, "lin_vel_x"),
        "vx_abs_range": _scalar(twist, "lin_vel_x_abs_range"),
        "rel_forward_envs": _scalar(twist, "rel_forward_envs"),
        "rel_standing_envs": _scalar(twist, "rel_standing_envs"),
        "contact_pattern": _scalar(secondary, "pattern"),
        "contact_entity": _scalar(secondary, "entity"),
        # Circle task only.
        "progress_func": _scalar(progress, "func"),
        "circle_progress_weight": _scalar(progress, "weight"),
        "radius_weight": _scalar(_block(rewards, "circle_radius") or "", "weight"),
        "air_weight": _scalar(air_time, "weight"),
        "track_ang_weight": _scalar(_block(rewards, "track_angular_velocity") or "", "weight"),
        "radius_range": radius_range,
        "omega_range": _scalar(twist, "omega_range"),
        "episode_length_s": _scalar(text, "episode_length_s"),
        # The reset term is what proves the spawn is on the circle: the stock
        # `reset_root_state_uniform` would place the duck at a random pose.
        "reset_func": _scalar(reset_base, "func"),
        "markers": _circle_markers(text),
        "raw_text": text,
        "raw_path": str(p),
    }


# --------------------------------------------------------------------------- #
# tensorboard metrics
# --------------------------------------------------------------------------- #
WANT = (
    "air_time",
    "track_lin",
    "track_ang",
    "fell_off",
    "time_out",
    "mean_reward",
    "episode_length",
    "standing",
    "action_rate",
    # Circle task. `circle_progress` is THE curve for this task: ~1.0 means it
    # walks the orbit, ~0.02 means it stands still or spins on the spot. It was
    # missing from this list, so `report` -- the one command the docs tell you to
    # run when a gate fails -- could not answer its own question.
    "circle_progress",
    "circle_radius",
)


def _scalar_of(value) -> float | None:
    """Numeric value of a Summary.Value, or None if it is not a scalar.

    TensorBoard >= 2.x *migrates* scalars written as the deprecated
    ``simple_value`` into a 1-element ``tensor`` on write, so both encodings
    occur in the wild (torch's SummaryWriter and tensorboard's own writer both
    do it). Handle both.
    """
    kind = value.WhichOneof("value")
    if kind == "simple_value":
        return float(value.simple_value)
    if kind == "tensor":
        try:
            from tensorboard.util import tensor_util

            arr = tensor_util.make_ndarray(value.tensor).reshape(-1)
        except Exception:  # noqa: BLE001 - fall back to the raw float list
            arr = list(value.tensor.float_val)
        if len(arr) == 1:
            return float(arr[0])
    return None


def read_metrics(run_dir: Path, max_rows: int = 20) -> list[tuple[str, float, float]]:
    """First/last value of the scalars that separate a stander from a walker.

    Reads the raw event file with ``EventFileLoader`` instead of
    ``EventAccumulator``: the accumulator's import path moved between
    tensorboard releases (``tensorboard.backend.event_accumulator`` no longer
    exists in 2.21), and we only need "first" and "last".
    """
    evs = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    if not evs:
        return []
    try:
        from tensorboard.backend.event_processing.event_file_loader import (
            EventFileLoader,
        )
    except Exception:  # noqa: BLE001 - tensorboard absent/renamed
        return []

    first: dict[str, float] = {}
    last: dict[str, float] = {}
    order: list[str] = []
    try:
        for ev in EventFileLoader(evs[-1]).Load():
            for value in ev.summary.value:
                tag = value.tag or ""
                if not tag or not any(k in tag for k in WANT):
                    continue
                num = _scalar_of(value)
                if num is None:
                    continue  # histograms / non-scalar summaries
                if tag not in first:
                    first[tag] = num
                    order.append(tag)
                last[tag] = num
    except Exception:  # noqa: BLE001 - corrupt/partial event file
        return []
    return [(t, first[t], last[t]) for t in order[:max_rows]]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def describe_run(ckpt: Path) -> str:
    run_dir = ckpt.parent
    cfg = read_env_yaml(run_dir)
    std = _num(cfg.get("track_std"))
    thr = _num(cfg.get("air_threshold_min"))
    if cfg.get("circle_progress_weight") is not None:
        # Different task, different yardstick: the straight-line "V3-CURRENT"
        # label would be meaningless here (it grades knobs this task does not
        # have, and hides the one it does).
        tag = "CIRCLE"
    elif std is None:
        tag = "unknown"
    elif abs(std - POST_FIX["track_std"]) >= 0.02:
        tag = "PRE-FIX"
    elif thr is None or abs(thr - POST_FIX["air_threshold_min"]) > 0.01:
        tag = "V2-OLD"  # v2_bounded: bounded commands, but the gait knobs are absent
    else:
        tag = "V3-CURRENT"
    ts = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%m-%d %H:%M")
    return (
        f"{tag:<11s} {ckpt.name:<14s} iter={iteration_of(ckpt):>5d}  "
        f"mtime={ts}  std={cfg.get('track_std') or '?'}  "
        f"thr={cfg.get('air_threshold_min') or '?'}  "
        f"hold={cfg.get('cmd_hold') or '?'}  {run_dir.name}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=None, help="model_XXXX.pt (default: newest)")
    ap.add_argument("--all", action="store_true", help="list every run, one line each")
    ap.add_argument("--quiet", action="store_true", help="provenance only, no metrics")
    ap.add_argument("--experiment", default=None,
                    help="run directory to inspect (default: from $TASK, else "
                         "velocity_sofa): velocity_sofa | velocity_circle")
    args = ap.parse_args()

    exp = (args.experiment
           or experiment_for_task(os.environ.get("TASK"))
           or EXPERIMENT)
    set_experiment(exp)
    is_circle = exp == "velocity_circle"

    ckpts = sorted(all_checkpoints(), key=lambda p: p.stat().st_mtime)
    if args.all:
        if not ckpts:
            print(f"no checkpoints under {RUN_GLOB}")
            return 0
        print(f"{len(ckpts)} checkpoint(s) under logs/rsl_rl/{EXPERIMENT}/\n")
        for c in ckpts:
            print(describe_run(c))
        return 0

    ckpt = Path(args.checkpoint) if args.checkpoint else newest_checkpoint()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    run_dir = ckpt.parent

    print("=" * 78)
    print("CHECKPOINT PROVENANCE")
    print("=" * 78)
    print(f"checkpoint : {ckpt}")
    print(f"iteration  : {iteration_of(ckpt)}")
    print(f"mtime      : {datetime.fromtimestamp(ckpt.stat().st_mtime):%Y-%m-%d %H:%M:%S}")
    print(f"run dir    : {run_dir}")
    others = [c for c in ckpts if c.parent != run_dir]
    if others:
        print(f"\n!! {len(others)} checkpoint(s) from OTHER runs also exist:")
        for c in others[-4:]:
            print(f"   {describe_run(c)}")

    cfg = read_env_yaml(run_dir)
    print("\n" + "-" * 78)
    print("CONFIG THIS RUN WAS TRAINED WITH")
    print("-" * 78)
    if not cfg:
        print("params/env.yaml not found -- cannot verify provenance.")
        print("(mjlab writes it on rank 0 at startup; older or copied run dirs may")
        print(" lack it. The knobs below are then simply unknown, not assumed bad.)")
        return 0

    if is_circle:
        # Which recipe did this run actually use? The FUNCTION NAME is the
        # honest marker: it says what the reward computed, while the run dir
        # name only says what someone typed.
        stale = 0
        unknown = 0
        func = str(cfg.get("progress_func") or "")
        recipe = circle_recipe_of(func)
        if recipe is None:
            print(f"  {'arc-rate function':<28s} "
                  f"{func.split('.')[-1][-22:] or '?':>22s}   "
                  f"? (unknown recipe, not counted)")
            print("     expected circle_progress_reward (c1) or "
                  "circle_progress_linear_reward (c2)")
            unknown += 1
        else:
            want = CIRCLE_RECIPES[recipe]
            print(f"  {'circle recipe':<28s} {recipe:>22s}   "
                  f"from {func.split('.')[-1]}")
            checks = [
                ("air_time threshold_min", cfg["air_threshold_min"], 0.04),
                ("air_time weight", cfg["air_weight"], want["air_weight"]),
                ("arc_rate weight", cfg["circle_progress_weight"],
                 want["progress_weight"]),
                ("radius weight", cfg["radius_weight"], want["radius_weight"]),
                # Zero on purpose: a yaw-rate objective pays for spinning in
                # place, which makes zero progress around the circle.
                ("track_angular weight", cfg["track_ang_weight"], 0.0),
                ("orbit radius max", cfg["radius_range"], want["radius_max"]),
            ]
            stale = 0
            for name, val, want_v in checks:
                v = _num_max(val) if "max" in name else _num(val)
                w = float(want_v)
                if v is None:
                    verdict = "? (unreadable, not counted)"
                    unknown += 1
                elif abs(v - w) <= max(0.02, 0.1 * abs(w)):
                    verdict = "ok"
                else:
                    verdict = f"PRE-{recipe.upper()} (want {want_v})"
                    stale += 1
                print(f"  {name:<28s} {str(val or '?'):>22s}   {verdict}")

        # Positive evidence that this run used the circle cfg at all. A run
        # without it is a different task wearing the same directory name.
        markers = list(cfg.get("markers") or [])
        print(f"  {'circle term markers':<28s} {', '.join(markers) or 'NONE':>22s}   "
              f"{'ok' if markers else 'PRE-CIRCLE (no circle term in env.yaml)'}")
        stale += 0 if markers else 1

        # The spawn must come from reset_on_circle; with the stock uniform reset
        # the duck starts at a random pose and never sees the commanded orbit.
        #
        # Three outcomes, not two:
        #   read a name containing reset_on_circle -> ok
        #   read a DIFFERENT name                  -> real evidence of a stock reset
        #   read nothing (yaml shape we cannot parse) -> UNKNOWN, and unknown must
        #     never be reported as a mismatch. This gate has now twice aborted a
        #     good run on its own parsing blindness; the marker check above
        #     already covers the "wrong task" case with evidence.
        reset_func = str(cfg.get("reset_func") or "")
        raw_text = str(cfg.get("raw_text") or "")
        if "reset_on_circle" in reset_func:
            ok_reset, reset_note = True, "ok"
        elif reset_func:
            ok_reset, reset_note = False, "PRE-CIRCLE (want reset_on_circle)"
        elif "reset_on_circle" in raw_text:
            ok_reset, reset_note = True, "ok (found in raw yaml)"
        else:
            ok_reset, reset_note = None, "? (unreadable, not counted)"
        shown = reset_func.split(".")[-1] or ("reset_on_circle" if ok_reset else "?")
        print(f"  {'reset_base func':<28s} {shown:>22s}   {reset_note}")
        if ok_reset is False:
            stale += 1
        elif ok_reset is None:
            unknown += 1
        print(f"  {'orbit r / omega':<28s} "
              f"{(str(cfg['radius_range']) + ' / ' + str(cfg['omega_range'])):>22s}")
        print(f"  {'episode length (s)':<28s} "
              f"{str(cfg['episode_length_s'] or '?'):>22s}")
        print(f"  {'contact sensor secondary':<28s} "
              f"{str(cfg['contact_pattern'] or '?') + ' @ ' + str(cfg['contact_entity'] or '-')}")
        if unknown:
            print(f"  ({unknown} value(s) could not be parsed; they are reported as"
                  f" '?' and do NOT count as mismatches)")
    else:
        checks = [
            ("track_linear_velocity std", cfg["track_std"], POST_FIX["track_std"]),
            ("head_pose_tracking weight", cfg["head_pose_weight"], POST_FIX["head_pose_weight"]),
            ("command hold max (s)", cfg["cmd_hold"], POST_FIX["cmd_hold_max_s"]),
            ("rel_forward_envs", cfg["rel_forward_envs"], POST_FIX["rel_forward_envs"]),
            # v3 gait-bootstrap knobs (2026-11-14): a run without these is v2_bounded,
            # which still converged to standing.
            ("air_time threshold_min", cfg["air_threshold_min"], POST_FIX["air_threshold_min"]),
            ("rel_standing_envs", cfg["rel_standing_envs"], POST_FIX["rel_standing_envs"]),
            ("lin_vel_x_abs_range max", cfg["vx_abs_range"], POST_FIX["vx_abs_range_max"]),
        ]
        stale = 0
        for name, val, want in checks:
            # `command hold` / `abs_range` are (min, max); the constraint is on the max.
            v = _num_max(val) if ("max" in name) else _num(val)
            w = float(want)
            if v is None:
                verdict = "?"
            elif abs(v - w) <= max(0.02, 0.1 * abs(w)):
                verdict = "ok"
            else:
                verdict = f"PRE-V3 (want {want})"
                stale += 1
            print(f"  {name:<28s} {str(val or '?'):>22s}   {verdict}")
        print(f"  {'lin_vel_x range':<28s} {str(cfg['lin_vel_x'] or '?'):>22s}")
        print(f"  {'rel_standing_envs':<28s} {str(cfg['rel_standing_envs'] or '?'):>22s}")
        print(f"  {'contact sensor secondary':<28s} "
              f"{str(cfg['contact_pattern'] or '?') + ' @ ' + str(cfg['contact_entity'] or '-')}")

    if not args.quiet:
        rows = read_metrics(run_dir)
        print("\n" + "-" * 78)
        print("HOW THE RUN WENT  (first -> last logged value)")
        print("-" * 78)
        if not rows:
            print("no tensorboard scalars found (event file missing or unreadable)")
        else:
            print(f"  {'tag':<46s} {'first':>10s} {'last':>10s}")
            for tag, first, last in rows:
                print(f"  {tag[-46:]:<46s} {first:10.4f} {last:10.4f}")

            # The circle task lives or dies on one number, so say it out loud
            # instead of leaving the reader to match tag names.
            if is_circle:
                cp = next((r for r in rows if r[0].endswith("circle_progress")), None)
                if cp is not None:
                    v = cp[2]
                    if v >= 0.60:
                        hint = "orbits the commanded circle"
                    elif v >= 0.20:
                        hint = "moves, but well short of the commanded arc rate"
                    else:
                        hint = ("STANDS or SPINS IN PLACE -- the arc reward never "
                                "reached it")
                    print(f"\n  circle_progress = {v:.3f}  ->  {hint}")
                    if recipe == "c2":
                        print("  (c2 linear term: 1.0 = commanded rate, 0 = never moved)")
                    else:
                        print("  (c1 gaussian term: 1.0 perfect, 0.018 = zero arc swept)")

    print()
    if stale == 0:
        if is_circle:
            label = recipe.upper() if recipe else "CIRCLE"
            print(f"VERDICT: {label}-CURRENT config -- numbers from this checkpoint")
            print("         are meaningful for this circle recipe (README §14).")
            print("         Judge it with scripts/sofa/verify_circle.py, not")
            print("         verify_mjlab.py: a spin-in-place policy passes the")
            print("         latter and makes zero progress around the orbit.")
        else:
            print("VERDICT: V3-CURRENT config -- numbers from this checkpoint are")
            print("         meaningful for the 2026-11-14 gait-bootstrap fix.")
        return 0
    if is_circle:
        print(f"VERDICT: PRE-{recipe.upper() if recipe else 'CIRCLE'} config "
              f"({stale} knob(s) differ).")
        print("         This checkpoint was trained with a different circle recipe,")
        print("         so a standing policy here is expected and says nothing")
        print("         about the current one. Retrain first:")
        print("             bash scripts/sofa/modelscope_run.sh full")
        return 3
    print(f"VERDICT: PRE-V3 config ({stale} knob(s) differ).")
    print("         This checkpoint predates the v3 gait fix (both the pre-fix")
    print("         run and v2_bounded converged to standing, so a standing")
    print("         policy here is EXPECTED). Retrain with the current bundle")
    print("         before reading anything into verify's numbers:")
    print("             bash scripts/sofa/modelscope_run.sh full")
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - never let a reporter crash a gate run
        import traceback

        traceback.print_exc()
        print("\nckpt_report could not read this run (exit 0: no verdict).")
        sys.exit(0)

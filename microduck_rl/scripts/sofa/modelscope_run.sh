#!/usr/bin/env bash
# Train a MicroDuck locomotion policy for a sofa seat, on a ModelScope (魔搭) GPU
# instance.
#
# Two tasks share this driver (pick one with $TASK, default = circle):
#   Mjlab-Circle-Sofa-MicroDuck    walk small laps on the cushion   (README §14)
#   Mjlab-Velocity-Sofa-MicroDuck  straight-line velocity tracking  (README §13)
# The circle task is the default because straight-line tracking on a 1.56 x
# 0.62 m cushion is geometrically self-defeating: a 0.20 m/s command runs off
# the seat in <3 s, so "track well" and "stay on" contradict each other and
# every straight-line run converged to standing. An orbit returns to its own
# start, so tracking it perfectly never approaches an edge.
#
#   bash scripts/sofa/modelscope_run.sh prefetch  # curl -C - ALL 140 wheels
#                                                 #   (resumable + sha256-verified,
#                                                 #   mirror-chained per file)
#   bash scripts/sofa/modelscope_run.sh check      # env sanity + CPU physics gate
#   bash scripts/sofa/modelscope_run.sh smoke      # 64 envs x 5 iters (minutes, catches cfg bugs)
#   bash scripts/sofa/modelscope_run.sh full       # CPU build gate + 4096 envs x 4000 iters
#                                                 #   (SKIP_BUILD_CHECK=1 to skip the gate)
#   bash scripts/sofa/modelscope_run.sh verify     # provenance gate + task-specific
#                                                 #   motion gate in the TRAINING env
#                                                 #   (catches a standing policy)
#   bash scripts/sofa/modelscope_run.sh report     # which config trained the
#                                                 #   checkpoint + how the run went
#                                                 #   (`report --all`: all runs)
#   bash scripts/sofa/modelscope_run.sh export     # latest checkpoint -> policy.onnx (+probe)
#   bash scripts/sofa/modelscope_run.sh probe      # ONNX-only command-sensitivity gate
#   bash scripts/sofa/modelscope_run.sh upload     # push run dir + onnx to ModelScope
#                                                 #   (policy.onnx is exported
#                                                 #   first if it is missing;
#                                                 #   FORCE_UPLOAD=1 skips the
#                                                 #   probe gate, ONNX_ONLY=1
#                                                 #   pushes the .onnx alone)
#   bash scripts/sofa/modelscope_run.sh all        # check -> smoke -> full -> verify -> export -> upload
#
# Tunables (env vars):
#   TASK=Mjlab-Velocity-Sofa-MicroDuck (default: Mjlab-Circle-Sofa-MicroDuck)
#   NUM_ENVS=4096  ITERS=4000  MS_REPO=cartman/microduck-sofa-walk
#   MODELSCOPE_API_TOKEN=ms-...        (only needed for `upload`)
#   FORCE_UPLOAD=1                     upload without passing the probe gate
#                                      (deliberate experiments only -- the run
#                                      never proved it responds to the command)
#   ONNX_ONLY=1                        upload only policy.onnx, not the run dir
#   SKIP_BUILD_CHECK=1                 skip the CPU physics gate before `full`
#   UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/   (default; ModelScope
#                                       instances run on Aliyun, so this mirror
#                                       is by far the fastest there. Override to
#                                       e.g. the Tsinghua mirror if needed.)
#   GH_PROXY=                          (override the GitHub proxy used for the
#                                       bam git dependency; tried in order below)
#
# Why this exists: the repo's `--hf-jobs` path submits to Hugging Face Jobs, and
# huggingface.co is unreachable from the dev machine. The *training recipe* has
# no HF coupling, so this runs the exact same `uv run train <task>` locally on a
# ModelScope instance and uploads artefacts to a ModelScope model repo instead.
#
# IMPORTANT (2026-11-13): uv is used to CREATE the venv, nothing else. Every
# stage then runs `$REPO_ROOT/.venv/bin/python` directly, because `uv run`
# re-validates the lock and does an implicit `uv sync` first -- which goes to
# the index and can 403 (aliyun does not have every file, e.g. networkx-3.6.1).
# See section 1 for the full story.
set -euo pipefail

STAGE="${1:-all}"
# Which task to train. Two are supported:
#   Mjlab-Velocity-Sofa-MicroDuck  straight-line velocity tracking (the task
#                                  that kept converging to standing)
#   Mjlab-Circle-Sofa-MicroDuck    walk small laps on the cushion (default,
#                                  see README §14)
# Override per invocation: TASK=Mjlab-Velocity-Sofa-MicroDuck bash ...
TASK="${TASK:-Mjlab-Circle-Sofa-MicroDuck}"
export TASK
NUM_ENVS="${NUM_ENVS:-4096}"
ITERS="${ITERS:-4000}"
MS_REPO="${MS_REPO:-cartman/microduck-sofa-walk}"
UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Optional reference policies (e.g. the official alpha_walking.onnx) used by the
# probe stage to print a "your policy vs a real walker" comparison. Absent ->
# the probe simply reports absolute sensitivity.
POLICIES_REF="${POLICIES_REF:-$REPO_ROOT/.refpolicies}"

# uv installs to ~/.local/bin; every stage (not just check) needs it on PATH.
export PATH="$HOME/.local/bin:$PATH"

# All 140 wheels the lock resolves for linux/x86_64/cp312 (~4.5 GB), with URLs
# and sha256 taken from uv.lock. curl -C - resumes mid-file, uv cannot.
WHEELHOUSE="$REPO_ROOT/.wheelhouse"
WHEEL_LIST="$REPO_ROOT/scripts/sofa/wheels_linux_x86_64.txt"

# PyPI indexes to try, in order. Aliyun is intranet-fast on ModelScope but
# really does return 403 for files it has not mirrored (networkx-3.6.1,
# 2026-11-13), and uv has no per-file fallback: one 403 aborts the whole sync.
# Hence a chain rather than a single index.
PYPI_INDEXES=(
  "$UV_INDEX_URL"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.cloud.tencent.com/pypi/simple"
)

# The venv is the source of truth after the first `uv sync`; see section 1.
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_TRAIN="$VENV_DIR/bin/train"

# UV_HTTP_TIMEOUT is a *read* timeout (max gap between two reads), not a total
# transfer deadline: uv deliberately has no per-file total timeout, precisely so
# that an 858 MB torch wheel can finish on a slow link. Keep it modest so a
# truly stalled connection is detected quickly, and let sync_deps retry instead.
export UV_INDEX_URL UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" UV_LINK_MODE=copy
# uv has no byte-level resume, and a failure discards *every* in-flight transfer,
# so keep concurrency low: 4 in-flight means at most 4 wasted partial files per
# failure, and on a slow mirror a single stream is usually faster anyway.
# Override with UV_CONCURRENT_DOWNLOADS=3 for an even gentler setting.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
# uv re-downloads the full torch+CUDA stack (~4.5 GB) into its cache; keep that
# cache on the big data disk so a small system disk can't fill up and stall.
if [ -z "${UV_CACHE_DIR:-}" ] && [ -d /mnt/workspace ]; then
  export UV_CACHE_DIR=/mnt/workspace/.uvcache
fi
# uv downloads its standalone CPython from GitHub releases, which is unreachable
# on CN egress -- point it at the NJU mirror (same layout, per-tag dirs).
export UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
# Abort git clones that stall on github.com so the proxy fallback can kick in,
# instead of hanging forever (git ignores UV_HTTP_TIMEOUT).
git config --global http.lowSpeedLimit 1000 2>/dev/null || true
git config --global http.lowSpeedTime 20 2>/dev/null || true
# No wandb account needed: tensorboard logs land in logs/ (rsl_rl's other logger).
LOG_ARGS=(--agent.logger tensorboard)

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. uv + Python 3.12
# ---------------------------------------------------------------------------
ensure_uv() {
  command -v uv >/dev/null 2>&1 && { say "uv $(uv --version)"; return; }
  say "installing uv"
  # astral.sh can hang behind CN egress: hard-timeout the installer fetch.
  curl -m 30 -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 && { say "uv $(uv --version)"; return; }
  # Fallback: the PyPI mirror (astral.sh unreachable is common behind CN egress).
  python3 -m pip install -q -U uv --index-url "$UV_INDEX_URL"
  command -v uv >/dev/null 2>&1 || command -v "$HOME/.local/bin/uv" >/dev/null || die "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
}

ensure_python312() {
  # pyproject requires >=3.12,<3.13; ModelScope images ship 3.11.
  command -v python3.12 >/dev/null 2>&1 && { say "python3.12 $(python3.12 -V 2>&1)"; return; }
  if uv python find 3.12 >/dev/null 2>&1; then
    say "uv-managed python 3.12 present"
    return
  fi
  say "fetching a standalone CPython 3.12 via uv"
  if ! uv python install 3.12; then
    say "uv download failed, trying conda"
    if command -v conda >/dev/null 2>&1; then
      conda create -y -n py312 python=3.12 >/dev/null || die "conda create failed"
      say "conda env py312 created -- activate it and re-run: conda activate py312"
      die "re-run this script inside the py312 env"
    fi
    die "no python 3.12 and no conda. Install python3.12 (apt/conda) and re-run."
  fi
}

# ---------------------------------------------------------------------------
# 1. the venv: uv creates it ONCE; every stage then runs it DIRECTLY
# ---------------------------------------------------------------------------
# Why not `uv run <cmd>` everywhere (2026-11-13)? `uv run` re-validates the lock
# and performs an implicit `uv sync` before exec'ing. That implicit sync went to
# the index and hard-failed *inside the CPU build gate*:
#
#     error: Failed to fetch ... networkx-3.6.1-py3-none-any.whl
#     cause: HTTP status client error (403 Forbidden) for url
#            (https://mirrors.aliyun.com/pypi/packages/9e/c9/.../networkx-3.6.1)
#
# Two bugs in one line: (a) the Aliyun mirror serves 403 for files it has not
# cached, and uv has no per-file fallback, so one miss aborts everything; (b) the
# failure surfaced as "build gate failed -- fix the task spec", pointing at the
# task config for a pure dependency-availability problem. The venv was already
# complete at that point, so involving uv at all was unnecessary risk:
# `$VENV_PY` does no lock check, no resolution, no network.
venv_ready() {
  [ -x "$VENV_PY" ] || return 1
  # find_spec, not import: this runs before every stage and `import mjlab`
  # pulls in warp/MuJoCo (seconds, and it can touch the GPU).
  "$VENV_PY" -c 'import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ("mjlab", "torch", "mjlab_microduck")) else 1)' 2>/dev/null
}

# Run a script inside the existing venv. Never `uv run` (see above).
run_py() { "$VENV_PY" "$@"; }
run_train() {
  [ -x "$VENV_TRAIN" ] || die "no $VENV_TRAIN -- run 'bash scripts/sofa/modelscope_run.sh check' first"
  "$VENV_TRAIN" "$@"
}

ensure_env() {
  # Check the venv FIRST: if it is already complete, uv is not needed at all --
  # not even installed. That makes "a gate reaches out to the network" (the
  # 2026-11-13 403) structurally impossible rather than merely unlikely.
  if venv_ready; then
    say "venv ready: $("$VENV_PY" -V 2>&1) -- stages run it directly (no uv, no network)"
    return 0
  fi
  say "venv missing or incomplete -- syncing dependencies"
  ensure_uv
  ensure_python312
  sync_deps
  venv_ready || die "the venv still cannot import mjlab. Re-run: bash scripts/sofa/modelscope_run.sh prefetch"
}

# `--frozen` resolves by URL: uv looks up the exact URL recorded in uv.lock, so
# neither --find-links (resolution-only) nor --offline (cache lookup by URL) can
# feed it a wheelhouse. Rewriting those URLs to file:// in the lock is the only
# way to make a frozen lock consume local files -- uv still verifies each wheel
# against the lock's sha256. The original lock is kept at $WHEELHOUSE/uv.lock.remote.
localize_lock() {
  local pristine="$WHEELHOUSE/uv.lock.remote"
  [ -f "$pristine" ] || cp uv.lock "$pristine"
  cp -f "$pristine" uv.lock
  python3 - "$WHEEL_LIST" "$WHEELHOUSE" <<'PY'
import pathlib, sys
wheel_list, wheelhouse = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve()
lock = pathlib.Path("uv.lock")
text = lock.read_text()
patched = 0
for line in wheel_list.read_text().splitlines():
    parts = line.split()
    if len(parts) != 3 or parts[0].startswith("#"):
        continue
    fname, _, url = parts
    if url in text:
        text = text.replace(url, f"file://{wheelhouse}/{fname}")
        patched += 1
lock.write_text(text)
left = text.count("https://files.pythonhosted.org/packages/")
print(f"localized {patched} wheel URLs -> file://{wheelhouse} ({left} remote URLs left)")
PY
}

# Complete = every wheel named in the list exists and is non-empty. Cheap check;
# a truncated file is caught later by "uv sync fails" and then by re-fetching.
wheelhouse_ok() {
  [ -f "$WHEEL_LIST" ] && [ -d "$WHEELHOUSE" ] || return 1
  local f _rest
  while read -r f _rest; do
    case "$f" in "" | "#"*) continue ;; esac
    [ -s "$WHEELHOUSE/$f" ] || { WHEELHOUSE_MISSING="$f"; return 1; }
  done < "$WHEEL_LIST"
  return 0
}

# Candidate URLs for one wheel, cheapest first. The lock records the
# pythonhosted URL; the CN mirrors serve the identical path, so swapping the
# host is enough. Aliyun 403s on files it has not cached -> hence the chain.
wheel_urls() {
  local url="$1"
  printf '%s\n' \
    "${url/https:\/\/files.pythonhosted.org\/packages/https://mirrors.aliyun.com/pypi/web/packages}" \
    "${url/https:\/\/files.pythonhosted.org\/packages/https://pypi.tuna.tsinghua.edu.cn/packages}" \
    "$url"
}

# Fetch/resume every wheel that is missing or fails its sha256. Resumable
# (curl -C -), so an interruption costs nothing: just run it again.
fetch_wheels() {
  [ -f "$WHEEL_LIST" ] || return 1
  mkdir -p "$WHEELHOUSE"
  local total i=0 fname hash url cand ok
  total="$(wc -l < "$WHEEL_LIST" | tr -d ' ')"
  while read -r fname hash url; do
    case "$fname" in "" | "#"*) continue ;; esac
    i=$((i + 1))
    if [ -s "$WHEELHOUSE/$fname" ] \
      && (cd "$WHEELHOUSE" && echo "$hash  $fname" | sha256sum -c - >/dev/null 2>&1); then
      continue
    fi
    say "[$i/$total] fetch $fname"
    ok=0
    while read -r cand; do
      if curl -L -C - -f --retry 20 --retry-all-errors --retry-delay 3 \
           --connect-timeout 15 -o "$WHEELHOUSE/$fname" "$cand"; then
        ok=1
        break
      fi
      say "  ! ${cand%%/packages/*} refused -- next source"
    done < <(wheel_urls "$url")
    if [ "$ok" != 1 ]; then
      rm -f "$WHEELHOUSE/$fname"
      say "  !! no source could serve $fname (re-run prefetch to resume)"
      return 1
    fi
  done < "$WHEEL_LIST"
  return 0
}

# Size>0 is not integrity: an interrupted curl leaves a 800 MB torch wheel at
# 300 MB, which passes wheelhouse_ok and then fails uv's sha256 check with a
# message that never names the file. Run this after the first failed sync.
repair_corrupt_wheels() {
  [ -f "$WHEEL_LIST" ] && [ -d "$WHEELHOUSE" ] || return 0
  local bad f
  bad="$(cd "$WHEELHOUSE" && awk '{print $2"  "$1}' "$WHEEL_LIST" 2>/dev/null \
    | sha256sum -c - 2>/dev/null | grep -v ': OK$' | sed 's/: .*//' || true)"
  if [ -z "$bad" ]; then
    say "wheelhouse checksums all match"
    return 0
  fi
  say "corrupt/truncated wheels: $(echo $bad)"
  for f in $bad; do rm -f "$WHEELHOUSE/$f"; done
  fetch_wheels
}

sync_deps() {
  # Prefer the local wheelhouse: zero index requests, zero 403s. Repair it
  # first if a previous prefetch was interrupted (that used to fall through to
  # "wheelhouse detected" -> `uv sync --frozen` -> "Network connectivity is
  # disabled" with no hint about the missing file).
  if [ -f "$WHEEL_LIST" ]; then
    if wheelhouse_ok; then
      say "wheelhouse complete ($(du -sh "$WHEELHOUSE" 2>/dev/null | cut -f1)) -- installing offline"
    else
      say "wheelhouse incomplete (first missing: ${WHEELHOUSE_MISSING:-?}) -- repairing"
      fetch_wheels || say "repair incomplete -- will fall back to the index"
    fi
    if wheelhouse_ok; then localize_lock; fi
  fi

  # uv cannot resume a half-downloaded wheel, and one failed transfer aborts the
  # whole sync -- but every *completed* wheel stays in the cache forever. So just
  # run it again, and rotate mirrors between rounds: each completed file is
  # cached, so a package only ever has to succeed on ONE of the indexes.
  local attempt idx
  for attempt in 1 2 3; do
    for idx in "${PYPI_INDEXES[@]}"; do
      if UV_INDEX_URL="$idx" uv sync --frozen; then
        say "uv sync ok (${idx}, attempt ${attempt})"
        return 0
      fi
      say "uv sync failed on ${idx} -- next source"
    done
    # First full round failed. If we were installing from the wheelhouse, the
    # likely cause is a truncated file that passed the size check -- verify it
    # byte-for-byte and re-fetch before burning more retries on the index.
    if [ "$attempt" = 1 ] && [ -f "$WHEELHOUSE/uv.lock.remote" ]; then
      say "offline install is failing -- verifying the wheelhouse against the lock"
      repair_corrupt_wheels || true
      if wheelhouse_ok; then localize_lock; fi
    fi
    say "all indexes failed (attempt ${attempt}) -- completed wheels are cached, retrying in $((attempt * 10))s"
    sleep $((attempt * 10))
  done

  say "plain retries exhausted -- switching github.com to a proxy for the bam git dependency"
  local p
  local proxies=("${GH_PROXY:-}" "https://ghfast.top/" "https://ghproxy.net/" "https://kkgithub.com/")
  for p in "${proxies[@]}"; do
    [ -z "$p" ] && continue
    say "git url rewrite -> ${p}https://github.com/"
    git config --global --unset-all url."https://github.com/".insteadOf 2>/dev/null || true
    git config --global url."${p}https://github.com/".insteadOf "https://github.com/"
    if uv sync --frozen; then say "uv sync ok via $p"; return 0; fi
  done
  git config --global --unset-all url."https://github.com/".insteadOf 2>/dev/null || true
  die "uv sync failed. Manual options: (a) set GH_PROXY to a working mirror, (b) run the prefetch stage and retry (installs fully offline), (c) clone Rhoban/bam by hand into the uv cache."
}

# ---------------------------------------------------------------------------
# 2. stages
# ---------------------------------------------------------------------------
stage_prefetch() {
  [ -f "$WHEEL_LIST" ] || die "wheel list missing: $WHEEL_LIST"
  local total; total="$(wc -l < "$WHEEL_LIST" | tr -d ' ')"
  say "prefetch: $total wheels -> $WHEELHOUSE"
  say "(per-file mirror chain: aliyun -> tsinghua -> pythonhosted, resumable)"
  fetch_wheels || die "download failed -- re-run prefetch to resume (files already fetched are kept)"
  say "verifying all $total checksums..."
  (cd "$WHEELHOUSE" && awk '{print $2"  "$1}' "$WHEEL_LIST" | sha256sum -c -) \
    || die "checksum mismatch -- delete the offending .whl in $WHEELHOUSE and re-run prefetch"
  say "wheelhouse complete: $(du -sh "$WHEELHOUSE" | cut -f1) in $WHEELHOUSE"
  say "now re-run: bash scripts/sofa/modelscope_run.sh check  (sync switches to offline-first)"
}

stage_check() {
  say "hardware"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || say "no nvidia-smi (!)"
  python3 -c 'import os,sys; print("cpus", os.cpu_count())'
  free -g | head -2 || true
  df -h . | tail -1

  ensure_env

  say "CPU physics gate (sofa contact model, ~30 s)"
  run_py scripts/sofa/validate_sofa_physics.py
}

stage_smoke() {
  say "smoke: 64 envs x 5 iters (this is the step that catches 95% of cfg errors)"
  run_train "$TASK" --env.scene.num-envs 64 --agent.max_iterations 5 "${LOG_ARGS[@]}"
  say "smoke ok -- the task id, scene and data shapes all line up"
}

stage_full() {
  # CPU env-build gate (~1 min): builds the FULL task env, RE-MEASURES the seat
  # geometry from the asset by ray-casting, and asserts the contact sensor wiring
  # compiles + resolves.
  #   * 2026-11-13: a bad ContactMatch only exploded at scene.compile().
  #   * 2026-11-14: four runs were trained with the duck spawned 0.35 m inside
  #     the backrest, because the spawn constants were derived from the seat
  #     boxes' bounding box instead of the walkable surface (README §16). The
  #     geometry section of build_check.py now fails that in 30 s on CPU.
  # Either way it dies in seconds, not after waiting for the GPU queue.
  if [ "${SKIP_BUILD_CHECK:-0}" != "1" ]; then
    say "build gate: constructing the training env on CPU (SKIP_BUILD_CHECK=1 to skip)"
    # Runs on the existing venv only. A dependency/import error here is NOT a
    # task-spec problem, and the message says so: on 2026-11-13 a 403 from the
    # package index inside this gate was reported as "fix the task spec", which
    # sent us looking at the sofa XML for a broken mirror.
    run_py scripts/sofa/build_check.py --task "$TASK" || die "build gate failed -- see the traceback above (a MuJoCo/sensor error means the task spec; an ImportError or a fetch error means the environment: run 'bash scripts/sofa/modelscope_run.sh check')"
  else
    say "build gate SKIPPED (SKIP_BUILD_CHECK=1)"
  fi
  say "full: $NUM_ENVS envs x $ITERS iters"
  run_train "$TASK" --env.scene.num-envs "$NUM_ENVS" --agent.max_iterations "$ITERS" "${LOG_ARGS[@]}"
}

# Run directory (=== experiment_name) for the selected task. Checkpoints for two
# tasks can coexist under logs/rsl_rl/, so "newest by mtime" alone can hand the
# wrong task's weights to verify/export -- which is how a gate ends up arguing
# about the previous run.
task_experiment() {
  case "$TASK" in
    *Circle*) printf 'velocity_circle\n' ;;
    *Sofa*)   printf 'velocity_sofa\n' ;;
    *)        printf '\n' ;;
  esac
}

latest_ckpt() {
  # rsl_rl writes logs/rsl_rl/<experiment>/<timestamp>_<run_name>/model_<iter>.pt
  # (mjlab's train.py makes the run dir one level below the experiment dir).
  # NOTE: with `set -o pipefail` a non-matching glob makes the pipeline fail, so
  # each form is tried separately -- otherwise a single-run layout would look
  # like "no checkpoint found". Only one path is ever printed.
  local exp p
  exp="$(task_experiment)"
  if [ -n "$exp" ]; then
    p="$(ls -t "logs/rsl_rl/$exp"/*/model_*.pt 2>/dev/null | head -1 || true)"
    [ -n "$p" ] && { printf '%s\n' "$p"; return; }
  fi
  p="$(ls -t logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1 || true)"
  if [ -z "$p" ]; then
    p="$(ls -t logs/rsl_rl/*/model_*.pt 2>/dev/null | head -1 || true)"
  fi
  printf '%s\n' "$p"
}

run_report() {
  # Provenance + training metrics for one checkpoint. Returns 3 when the
  # checkpoint predates the current task config, 0 otherwise (a crash inside the
  # reporter is *not* a verdict, so it also returns 0).
  local ckpt="${1:-}"
  if [ -z "$ckpt" ]; then
    ckpt="$(latest_ckpt || true)"
  fi
  [ -n "${ckpt:-}" ] || { say "no checkpoint under logs/rsl_rl"; return 0; }
  set +e
  run_py scripts/sofa/ckpt_report.py --checkpoint "$ckpt" "${@:2}"
  local rc=$?
  set -e
  return "$rc"
}

stage_verify() {
  ensure_env
  local ckpt; ckpt="$(latest_ckpt || true)"
  [ -n "${ckpt:-}" ] || die "no checkpoint under logs/rsl_rl -- train first"

  # The straight-line gate measures ground speed; the circle gate measures arc
  # swept around the orbit centre. They are not interchangeable -- a duck that
  # spins in place passes the former and makes zero progress on the latter.
  local vscript descr
  case "$TASK" in
    *Circle*)
      vscript="scripts/sofa/verify_circle.py"
      descr="orbiting the cushion"
      ;;
    *)
      vscript="scripts/sofa/verify_mjlab.py"
      descr="sweeping velocity commands"
      ;;
  esac
  say "verify: $descr in the training env (${ckpt##*/}, task=$TASK)"

  # Provenance gate FIRST. Three times now a gate has been argued with while it
  # pointed at the previous run's weights: a check that fails on a pre-fix
  # checkpoint says nothing about the fix. Exit 3 = the run config differs from
  # what the current task spec produces.
  set +e
  run_report "$ckpt" --quiet
  local prov=$?
  set -e
  if [ "$prov" -eq 3 ]; then
    say "verify ABORTED -- this checkpoint was trained with a DIFFERENT config"
    say "  (or on a different task). A failing policy here is expected, not"
    say "  informative. Retrain first:"
    say "    TASK=$TASK bash scripts/sofa/modelscope_run.sh full"
    die "stale checkpoint"
  fi

  # Gate: a policy that does not move in its own training env will not move on
  # the robot either. Run 2026-11-12 shipped exactly such a standing policy;
  # this stage exists so that cannot happen again.
  # Exit codes: 0 = moves/tracks, 1 = stander, 2 = the SCRIPT/env crashed (no
  # verdict). Keeping 2 separate matters: twice now a verify crash was reported
  # as "verify FAIL -- standing policy", sending us off to retrain for a bug
  # that was in the measurement itself.
  set +e
  run_py "$vscript" --checkpoint "$ckpt" "${@}"
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    say "verify PASS"
  elif [ "$rc" -eq 2 ]; then
    say "verify could NOT RUN (script/env error, exit 2)"
    say "  the checkpoint was never measured -- this is not a policy verdict."
    say "  fix the error above (usually an mjlab API mismatch in $vscript)"
    die "verify error"
  else
    say "verify FAIL -- $vscript rejected this checkpoint"
    say "  run 'bash scripts/sofa/modelscope_run.sh report' for the training"
    say "  metrics that explain why (air_time, circle_progress, fell_off_sofa,"
    say "  episode length)"
    die "verify failed"
  fi
}

run_probe() {
  # Returns the probe's exit status (0 = command-sensitive, 1 = stander).
  local onnx="${1:-$REPO_ROOT/policy.onnx}"
  [ -f "$onnx" ] || { say "no policy at $onnx"; return 1; }
  local extra=()
  [ -f "$POLICIES_REF/alpha_walking.onnx" ] && extra=(--baseline "$POLICIES_REF/alpha_walking.onnx")
  case "$TASK" in
    *Circle*)
      # A circle policy has only ever seen v = omega*r, so v and wz are coupled
      # in its training data; sweeping vx alone with wz fixed at 0 probes an
      # input combination it was never asked about, and its response is
      # legitimately weaker. Keep the gate (a stander still scores ~0) but at a
      # level that does not punish the coupling.
      extra+=(--min-lin 0.20 --min-ang 0.35)
      ;;
  esac
  run_py scripts/sofa/probe_policy.py "$onnx" "${extra[@]}"
}

stage_probe() {
  ensure_env
  say "probe: does the policy actually react to the velocity command?"
  # Pure ONNX, a few MB and ~2 s: no sim, no GPU. Catches "standing policy"
  # before it reaches the robot.
  if run_probe "$@"; then
    say "probe PASS -- the policy responds to the command like a walker"
  else
    say "probe FAIL -- the policy is (nearly) command-blind: a standing policy"
    say "  do NOT upload it. Retrain with the anti-standing settings in"
    say "  src/mjlab_microduck/tasks/microduck_velocity_sofa_env_cfg.py"
    die "probe failed"
  fi
}

stage_export() {
  ensure_env
  local ckpt; ckpt="$(latest_ckpt || true)"
  [ -n "${ckpt:-}" ] || die "no checkpoint under logs/rsl_rl -- train first"
  say "exporting $ckpt -> policy.onnx"
  run_py scripts/export.py "$TASK" --checkpoint-file "$ckpt" --num-envs 1 --onnx-file "$REPO_ROOT/policy.onnx"
  run_py - <<'PY'
import onnx
m = onnx.load("policy.onnx")
for i in m.graph.input:
    print("input ", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("output", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
print("obs must be [1,61] and actions [1,14] -- robotd refuses anything else")
PY
  say "policy.onnx ready ($(du -h policy.onnx | cut -f1))"
  # Contract size is not behaviour: probe the graph for command sensitivity now,
  # so a standing policy never reaches the upload stage. Soft here (the upload
  # stage is the hard gate) so an intentional experiment can still be exported.
  if run_probe "$REPO_ROOT/policy.onnx"; then
    say "probe PASS"
  else
    say "WARNING: probe FAILED -- this looks like a standing policy."
    say "  upload will refuse it unless FORCE_UPLOAD=1."
  fi
  say "next: bash scripts/sofa/modelscope_run.sh upload"
}

# ---------------------------------------------------------------------------
# 5. the ModelScope SDK: its own venv, built with the STDLIB, never with uv
# ---------------------------------------------------------------------------
# `upload` is the only stage that needs a package the training venv lacks
# (modelscope). It used to build `.msvenv` with `uv venv` + `uv pip install`,
# which fails on exactly the boxes this script creates: `ensure_env` SKIPS uv
# whenever .venv is already complete (by design, see section 1), so a fully
# provisioned instance has no `uv` binary at all --
#
#     scripts/sofa/modelscope_run.sh: 行 606: uv: 未找到命令
#
# `$VENV_PY -m venv` needs nothing but the stdlib (ensurepip ships pip), and
# `.msvenv/bin/python -m pip` then installs through the same mirror chain. The
# separate venv still exists on purpose: the training env is pinned by uv.lock,
# and the ModelScope SDK drags in its own requests/urllib3 ranges.
MSVENV="$REPO_ROOT/.msvenv"
MSVENV_PY="$MSVENV/bin/python"
MSLIB="$REPO_ROOT/.mslib"
MS_RUN=()

ms_importable() { [ -x "$1" ] && "$1" -c 'import modelscope' >/dev/null 2>&1; }

msvenv_ready() {
  [ -x "$MSVENV_PY" ] || return 1
  # pip is checked separately: a venv created with --without-pip (or by an
  # interrupted ensurepip) imports fine but cannot install anything.
  "$MSVENV_PY" -m pip --version >/dev/null 2>&1 || return 1
  ms_importable "$MSVENV_PY"
}

# Sets MS_RUN to an interpreter that can `import modelscope`, installing the
# SDK if necessary. Offline boxes: prefetch cannot help here (modelscope is not
# in uv.lock), so every path goes to a PyPI mirror.
ensure_ms_sdk() {
  if ms_importable "$VENV_PY"; then
    say "ModelScope SDK already importable in the training venv -- reusing it"
    MS_RUN=("$VENV_PY")
    return 0
  fi
  if msvenv_ready; then
    say "ModelScope SDK ready in .msvenv"
    MS_RUN=("$MSVENV_PY")
    return 0
  fi
  # A half-built .msvenv (interrupted install, or a symlink left pointing at a
  # Python that no longer exists after the image was rebuilt) is worse than no
  # venv: recreate rather than try to repair it.
  rm -rf "$MSVENV"
  say "creating .msvenv with the stdlib venv module (no uv)"
  "$VENV_PY" -m venv "$MSVENV" || die "could not create $MSVENV -- is $VENV_PY a working interpreter?"
  # Keep pip's download cache off the system disk; modelscope drags in ~50 MB.
  if [ -z "${PIP_CACHE_DIR:-}" ] && [ -d /mnt/workspace ]; then
    export PIP_CACHE_DIR=/mnt/workspace/.pipcache
  fi
  local idx
  for idx in "${PYPI_INDEXES[@]}"; do
    say "installing the ModelScope SDK from $idx"
    if "$MSVENV_PY" -m pip install -q --disable-pip-version-check \
         --timeout 60 --retries 2 --index-url "$idx" modelscope; then
      MS_RUN=("$MSVENV_PY")
      return 0
    fi
    say "install failed on $idx -- next source"
  done
  # Last resort: keep the pinned training env untouched and drop the SDK into a
  # directory that only upload puts on PYTHONPATH.
  say "no venv could take modelscope -- falling back to a --target install in .mslib"
  rm -rf "$MSLIB"
  for idx in "${PYPI_INDEXES[@]}"; do
    if "$VENV_PY" -m pip install -q --disable-pip-version-check \
         --timeout 60 --retries 2 --index-url "$idx" --target "$MSLIB" modelscope; then
      MS_RUN=("$VENV_PY")
      export PYTHONPATH="$MSLIB${PYTHONPATH:+:$PYTHONPATH}"
      return 0
    fi
  done
  die "could not install modelscope from any index (tried: ${PYPI_INDEXES[*]}). Manual: $VENV_PY -m pip install --target $MSLIB modelscope"
}

stage_upload() {
  ensure_env
  [ -n "${MODELSCOPE_API_TOKEN:-}" ] || die "set MODELSCOPE_API_TOKEN=ms-... for the upload stage"
  local ckpt; ckpt="$(latest_ckpt || true)"

  # The onnx is the artefact the robot actually loads, so make sure it exists
  # instead of merely claiming to upload it: the old version printed
  # "+ policy.onnx" unconditionally and then silently shipped the run dir alone.
  if [ ! -f "$REPO_ROOT/policy.onnx" ]; then
    [ -n "${ckpt:-}" ] || die "no policy.onnx here and no checkpoint to export one from"
    say "policy.onnx missing -- exporting it from $(basename "$ckpt") first"
    stage_export
  fi

  # Hard gate (override with FORCE_UPLOAD=1 for an intentional experiment).
  if [ "${FORCE_UPLOAD:-0}" = "1" ]; then
    say "upload gate SKIPPED (FORCE_UPLOAD=1)"
    say "  publishing a checkpoint that has NOT passed the probe gate."
    say "  the run's own tensorboard curves are the only evidence behind it."
  else
    say "upload gate: probing policy.onnx before publishing"
    stage_probe "$REPO_ROOT/policy.onnx"
  fi

  local extra=() run_dir=""
  if [ "${ONNX_ONLY:-0}" = "1" ]; then
    # Only the graph ships, so no checkpoint is required: this path is also how
    # you re-publish an onnx that was copied back from another box.
    say "ONNX_ONLY=1 -- publishing policy.onnx alone (run dir not uploaded)"
    extra=(--only-onnx)
    run_dir="$REPO_ROOT"
  else
    [ -n "${ckpt:-}" ] || die "no checkpoint to upload (set ONNX_ONLY=1 to ship just policy.onnx)"
    run_dir="$(dirname "$ckpt")"
    say "uploading $run_dir + policy.onnx to $MS_REPO"
  fi

  ensure_ms_sdk
  MODELSCOPE_API_TOKEN="$MODELSCOPE_API_TOKEN" "${MS_RUN[@]}" scripts/sofa/ms_upload.py \
    --repo "$MS_REPO" \
    --run-dir "$run_dir" \
    --onnx "$REPO_ROOT/policy.onnx" \
    "${extra[@]}"
}

case "$STAGE" in
  check)  stage_check ;;
  prefetch) stage_prefetch ;;
  smoke)  ensure_env; stage_smoke ;;
  full)   ensure_env; stage_full ;;
  verify) shift || true; stage_verify "$@" ;;
  report|watch) shift || true; ensure_env
                if [ "${1:-}" = "--all" ] || [ "${1:-}" = "-a" ]; then
                  run_py scripts/sofa/ckpt_report.py --all
                else
                  run_report "${1:-}" || true
                fi ;;
  probe)  shift || true; stage_probe "${1:-$REPO_ROOT/policy.onnx}" ;;
  export) stage_export ;;
  upload) stage_upload ;;
  all)
    stage_check
    stage_smoke
    stage_full
    stage_verify
    stage_export
    if [ -n "${MODELSCOPE_API_TOKEN:-}" ]; then stage_upload; else say "skipping upload (no MODELSCOPE_API_TOKEN)"; fi
    ;;
  *) die "unknown stage '$STAGE' (check|smoke|full|verify|report|probe|export|upload|all)" ;;
esac
say "done: $STAGE"

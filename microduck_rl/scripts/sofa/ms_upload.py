"""Push a finished sofa run (checkpoints + ONNX) to a ModelScope model repo.

Stand-in for the HF Jobs checkpoint uploader: the training host is a ModelScope
instance, so artefacts live in a ModelScope model repo instead of
``<namespace>/<run-name>`` on the Hub.

    MODELSCOPE_API_TOKEN=ms-... .msvenv/bin/python scripts/sofa/ms_upload.py \
        --repo cartman/microduck-sofa-walk \
        --run-dir logs/rsl_rl/velocity_sofa/2026-11-11_01-02-03 \
        --onnx policy.onnx

Pass ``--only-onnx`` to publish just the exported graph (the file the robot
loads) without the checkpoint directory -- that is the fast path when the run
dir is already in the repo and only the export changed.

The ModelScope SDK's upload API has churned across releases (``upload_folder``
only exists in newer versions), so this tries the modern call and falls back to
per-file uploads; nothing here is version-pinned on purpose.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep the upload small: the rsl_rl run dir also holds tfevents/ and the agent
# yaml, which are useful provenance but not worth several hundred MB.
SKIP_SUFFIXES = {".tfevents", ".tmp"}
SKIP_GLOBS = ("events.out.*",)


def _files(run_dir: Path, onnx: Path | None) -> list[Path]:
    out = [
        p
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
        and p.suffix not in SKIP_SUFFIXES
        and not any(p.match(g) for g in SKIP_GLOBS)
    ]
    if onnx and onnx.exists():
        out.append(onnx)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="model repo id, e.g. cartman/microduck-sofa-walk")
    ap.add_argument("--run-dir", default=None, help="rsl_rl run directory holding model_*.pt (required unless --only-onnx)")
    ap.add_argument("--onnx", default=None, help="exported policy.onnx to upload alongside")
    ap.add_argument("--path-in-repo", default="run", help="subfolder inside the model repo")
    ap.add_argument(
        "--only-onnx",
        action="store_true",
        help="upload only --onnx (skip the checkpoint directory)",
    )
    args = ap.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("error: set MODELSCOPE_API_TOKEN (modelscope.cn -> 个人中心 -> 访问令牌)", file=sys.stderr)
        return 2

    onnx = Path(args.onnx) if args.onnx else None
    # An explicitly requested onnx that is not there is an error, not a silent
    # skip: the old code published the run dir and reported success while the
    # robot-ready graph stayed on the training box.
    if onnx is not None and not onnx.exists():
        print(f"error: --onnx not found: {onnx} (run the export stage first)", file=sys.stderr)
        return 2

    # --only-onnx needs no checkpoint at all: that path exists to re-publish the
    # graph alone (e.g. an onnx copied back from another machine).
    if args.only_onnx and onnx is None:
        print("error: --only-onnx needs --onnx", file=sys.stderr)
        return 2

    run_dir: Path | None = None
    if not args.only_onnx:
        if not args.run_dir:
            print("error: --run-dir is required unless --only-onnx", file=sys.stderr)
            return 2
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"error: --run-dir not found: {run_dir}", file=sys.stderr)
            return 2

    from modelscope.hub.api import HubApi

    api = HubApi()
    try:
        api.login(token)
    except TypeError:
        api.login(access_token=token)

    # The repo may already exist; creation failures are only fatal if the repo
    # cannot be reached at upload time, which the upload itself will report.
    try:
        api.create_model(args.repo, visibility=5)  # 5 = private
        print(f"[ms] created model repo {args.repo} (private)")
    except Exception as e:  # noqa: BLE001 - existing repo raises here
        print(f"[ms] create_model skipped ({type(e).__name__}: {e})")

    if args.only_onnx:
        print(f"[ms] uploading {onnx.name} alone ({onnx.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(onnx),
            path_in_repo="policy.onnx",
            commit_message="exported ONNX (only-onnx)",
        )
        print("[ms] done (only-onnx)")
        return 0

    # Past this point run_dir is guaranteed (the --only-onnx branch returned).
    assert run_dir is not None
    files = _files(run_dir, onnx)
    if not files:
        print("error: nothing to upload", file=sys.stderr)
        return 2
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"[ms] uploading {len(files)} files ({total_mb:.1f} MB) to {args.repo}/{args.path_in_repo}")

    # Modern SDK: one atomic folder commit. The tfevents payload is excluded
    # because it is hundreds of MB of provenance nobody reads back; an SDK that
    # rejects the filter (TypeError) falls through to the per-file path, which
    # applies SKIP_GLOBS instead.
    try:
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(run_dir),
            path_in_repo=args.path_in_repo,
            ignore_file_pattern=[r".*\.tfevents$", r"^events\.out\..*$"],
            commit_message=f"sofa run {run_dir.name}",
        )
        if onnx is not None:
            api.upload_file(
                repo_id=args.repo,
                path_or_fileobj=str(onnx),
                path_in_repo="policy.onnx",
                commit_message="exported ONNX",
            )
        print("[ms] done (upload_folder)")
        return 0
    except (AttributeError, TypeError) as e:
        print(f"[ms] upload_folder unavailable ({e}); falling back to per-file upload")

    for p in files:
        if p.name == "policy.onnx":
            path_in_repo = "policy.onnx"
        else:
            path_in_repo = f"{args.path_in_repo}/{p.relative_to(run_dir)}"
        print(f"  -> {path_in_repo} ({p.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            commit_message=f"sofa run {run_dir.name}",
        )
    print("[ms] done (per-file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

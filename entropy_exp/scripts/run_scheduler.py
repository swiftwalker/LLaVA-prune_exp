#!/usr/bin/env python3
"""CLI entrypoint for the entropy_exp experiment scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LLAVA_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(LLAVA_ROOT))

from entropy_exp.src.scheduler import (
    DEFAULT_STATE_BASE_DIR,
    ExperimentScheduler,
    finalize_attempt_result,
    load_scheduler_plan,
    sanitize_name,
    scheduler_dry_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent tmux-based scheduler for entropy_exp runs")
    parser.add_argument("--plan", help="Scheduler plan YAML")
    parser.add_argument("--pool-size", type=int, help="Override plan pool size")
    parser.add_argument("--state-dir", help="State directory for scheduler state")
    parser.add_argument("--dry-run", action="store_true", help="Expand queue and print jobs without launching tmux")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing state directory")

    subparsers = parser.add_subparsers(dest="internal_command")
    finalize = subparsers.add_parser("__finalize_attempt", help=argparse.SUPPRESS)
    finalize.add_argument("--state-dir", required=True)
    finalize.add_argument("--job-id", required=True)
    finalize.add_argument("--attempt", required=True, type=int)
    finalize.add_argument("--run-prefix", required=True)
    finalize.add_argument("--dataset", required=True)
    finalize.add_argument("--started-at", required=True, type=float)
    finalize.add_argument("--exit-code", required=True, type=int)
    finalize.add_argument("--log-path", required=True)
    finalize.add_argument("--gpu", required=True, type=int)
    finalize.add_argument("--tmux-session", required=True)
    finalize.add_argument("--tmux-window", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.internal_command == "__finalize_attempt":
        finalize_attempt_result(
            repo_root=LLAVA_ROOT,
            state_dir=Path(args.state_dir),
            job_id=args.job_id,
            attempt=args.attempt,
            run_prefix=args.run_prefix,
            dataset=args.dataset,
            started_at=args.started_at,
            exit_code=args.exit_code,
            log_path=args.log_path,
            gpu=args.gpu,
            tmux_session=args.tmux_session,
            tmux_window=args.tmux_window,
        )
        return 0

    state_dir = Path(args.state_dir) if args.state_dir else None
    if args.resume:
        if state_dir is None:
            if not args.plan:
                parser.error("--resume requires --state-dir or --plan")
            plan = load_scheduler_plan(Path(args.plan))
            state_dir = DEFAULT_STATE_BASE_DIR / sanitize_name(plan.label)
        scheduler = ExperimentScheduler.resume(
            repo_root=LLAVA_ROOT,
            state_dir=state_dir,
            scheduler_script=Path(__file__),
            scheduler_python=Path(sys.executable),
            pool_size_override=args.pool_size,
        )
        if args.dry_run:
            scheduler.emit_progress()
            return 0
        scheduler.run()
        return 0

    if not args.plan:
        parser.error("--plan is required unless using --resume with an existing --state-dir")

    plan_path = Path(args.plan)
    plan = load_scheduler_plan(plan_path)
    if args.dry_run:
        scheduler_dry_run(LLAVA_ROOT, plan, pool_size_override=args.pool_size)
        return 0

    scheduler = ExperimentScheduler.create(
        repo_root=LLAVA_ROOT,
        plan_path=plan_path,
        state_dir=state_dir,
        pool_size_override=args.pool_size,
        scheduler_script=Path(__file__),
        scheduler_python=Path(sys.executable),
    )
    scheduler.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Synchronize TM01 LeRobot metadata with its canonical policy prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.tasks.tm01.domain.contracts import (  # noqa: E402
    CANONICAL_PROMPT,
    TASK_ID,
)
from scripts.annotate.rewrite_tm02_task_prompt import rewrite_roots  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        action="append",
        required=True,
        help="TM01 LeRobot dataset root; may be supplied more than once.",
    )
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    rewrite_roots(
        args.lerobot_root,
        expected_episodes=args.expected_episodes,
        task_id=TASK_ID,
        canonical_prompt=CANONICAL_PROMPT,
    )


if __name__ == "__main__":
    main()

"""Installed evaluation entry point with recorded suite-family lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_suite(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if (ROOT / path).is_file():
        return (ROOT / path).resolve()
    if ":" in value:
        family, task = value.split(":", 1)
        key = "".join(c for c in task.casefold() if c.isalnum())
        inventory = json.loads(
            (ROOT / "configs/benchmark/evaluation_suite_sources.json").read_text()
        )
        for entry in inventory["entries"]:
            if (
                entry["family"] == family
                and "".join(c for c in entry["task_id"].casefold() if c.isalnum()) == key
            ):
                return ROOT / entry["suite_file"]
    raise ValueError(
        f"Unknown suite {value!r}; use a YAML path or family:task, for example pi05:wx01, act:wa01, diffusion:ts02"
    )


def main(argv=None) -> int:
    from scripts.eval.run_remote import main as run

    args = list(sys.argv[1:] if argv is None else argv)
    if "--suite" in args:
        index = args.index("--suite") + 1
        if index < len(args):
            args[index] = str(resolve_suite(args[index]))
    else:
        for index, value in enumerate(args):
            if value.startswith("--suite="):
                args[index] = "--suite=" + str(resolve_suite(value.split("=", 1)[1]))
                break
        else:
            if not any(value in ("--help", "-h") for value in args):
                raise ValueError(
                    "Choose the recorded model configuration explicitly with --suite family:task"
                )
    return run(args)

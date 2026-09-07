#!/usr/bin/env python3
"""Test a lower lift on an action-replayed WR-08 prefix, without recording data.

All ordinary prefix-probe arguments are forwarded. The override is confined to
this diagnostic process; the canonical task and active recorders are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.tasks.wr08.execution.pick_controller import WR08ManipulationController
from bench.tasks.wr08.execution import pick_controller as controller_module
from scripts.analysis import probe_wr08_execution_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--lift-height", type=float)
    parser.add_argument("--selector-file", type=Path)
    parser.add_argument("--executor-file", type=Path)
    args, forwarded = parser.parse_known_args()
    if args.lift_height is None and args.selector_file is None and args.executor_file is None:
        raise ValueError("provide --lift-height, --selector-file or --executor-file")
    if args.lift_height is not None and (not math.isfinite(args.lift_height) or args.lift_height <= 0):
        raise ValueError("--lift-height must be finite and positive")
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output_args, _ = output_parser.parse_known_args(forwarded)
    original_pick = WR08ManipulationController.pick
    original_init = WR08ManipulationController.__init__
    original_argv = sys.argv
    original_executor = controller_module.WorldAnchoredPickExecutor
    selector_factory = None
    if args.selector_file is not None:
        spec = importlib.util.spec_from_file_location("wr08_diagnostic_selector", args.selector_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selector_factory = module.build_pick_payload_selector
    if args.executor_file is not None:
        spec = importlib.util.spec_from_file_location("wr08_diagnostic_executor", args.executor_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        controller_module.WorldAnchoredPickExecutor = module.WorldAnchoredPickExecutor

    def diagnostic_pick(self, object_name, **kwargs):
        if args.lift_height is not None:
            kwargs = {**kwargs, "lift_height": args.lift_height}
        return original_pick(self, object_name, **kwargs)

    def diagnostic_init(self, *positional, **kwargs):
        original_init(self, *positional, **kwargs)
        if selector_factory is not None:
            self.pick_payload_selector = selector_factory()

    WR08ManipulationController.pick = diagnostic_pick
    WR08ManipulationController.__init__ = diagnostic_init
    sys.argv = [__file__, *forwarded]
    try:
        result = probe_wr08_execution_prefix.main()
    finally:
        WR08ManipulationController.pick = original_pick
        WR08ManipulationController.__init__ = original_init
        controller_module.WorldAnchoredPickExecutor = original_executor
        sys.argv = original_argv
    if output_args.output.is_file():
        payload = json.loads(output_args.output.read_text())
        payload["diagnostic_pick_overrides"] = {
            "lift_height": args.lift_height,
            "selector_file": str(args.selector_file.resolve()) if args.selector_file else None,
            "selector_sha256": hashlib.sha256(args.selector_file.read_bytes()).hexdigest() if args.selector_file else None,
            "executor_file": str(args.executor_file.resolve()) if args.executor_file else None,
            "executor_sha256": hashlib.sha256(args.executor_file.read_bytes()).hexdigest() if args.executor_file else None,
        }
        output_args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return 0 if (payload.get("result") or {}).get("success") else 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())

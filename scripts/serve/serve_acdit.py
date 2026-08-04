#!/usr/bin/env python
"""Reserved AC-DiT server entrypoint."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.policy.backends.acdit import ACDiTBackend  # noqa: E402


def main() -> None:
    ACDiTBackend.from_checkpoint()


if __name__ == "__main__":
    main()

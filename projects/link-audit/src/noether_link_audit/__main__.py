from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sample link-loss audit on synthetic telemetry")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True, help="directory for bulk synthetic telemetry")
    args = parser.parse_args()
    payload = run_sample(args.config, args.output, args.work)
    audit = payload["summary"]["audit"]
    print(
        f"simulation-only sample audit complete: {payload['experiment_id']}; "
        f"delivered {audit['delivered_gb']:.0f} of {audit['possible_gb']:.0f} Gb; report: {args.output / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()

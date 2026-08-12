"""Trigger a full DVC pipeline retrain and push updated artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    run(["dvc", "repro"])
    run(["dvc", "push"])

    metrics_path = Path("metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print("New evaluation metrics:")
        print(json.dumps(metrics, indent=2))
    else:
        print("Warning: metrics.json not found after dvc repro", file=sys.stderr)


if __name__ == "__main__":
    main()

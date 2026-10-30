"""Run the prespecified benchmark grid and save audit-ready raw results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from datasentry.experiment import ExperimentConfig, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="One seed and compact grid for smoke tests")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Run a seed batch and write it to a separate output CSV")
    parser.add_argument("--mechanisms", nargs="+", choices=("mcar", "mar", "extreme"))
    parser.add_argument("--levels", nargs="+", type=float)
    parser.add_argument("--strategies", nargs="+", choices=("median", "knn", "iqr", "adaptive"))
    parser.add_argument("--models", nargs="+", choices=("ridge", "forest", "mlp"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    config = ExperimentConfig(
        seeds=tuple(args.seeds) if args.seeds else ((11,) if args.quick else (11, 23, 37, 53, 71)),
        levels=tuple(args.levels) if args.levels else ((0.0, 0.15) if args.quick else (0.0, 0.05, 0.15, 0.30)),
        mechanisms=tuple(args.mechanisms) if args.mechanisms else (("mcar", "extreme") if args.quick else ("mcar", "mar", "extreme")),
        strategies=tuple(args.strategies) if args.strategies else (("median", "iqr", "adaptive") if args.quick else ("median", "knn", "iqr", "adaptive")),
        models=tuple(args.models) if args.models else (("ridge", "forest") if args.quick else ("ridge", "forest", "mlp")),
        drop_columns=tuple(args.drop),
    )
    result = run_experiment(frame, args.target, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    manifest = {
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "input_rows": len(frame),
        "target": args.target,
        "config": vars(config),
        "result_rows": len(result),
        "pandas": pd.__version__,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"result_rows": len(result), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

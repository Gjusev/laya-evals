"""Regression gate CLI: compare a measurement JSON against a baseline.

Exits 0 when every metric is within tolerance, 1 on regression, 2 on usage
or input errors. Metrics are ``path:goal[:tolerance]`` where path is a
dotted path into the JSON (keys containing dots are matched first), goal is
max (accuracy-like) or min (ECE-like), and tolerance defaults to 0.02.

Usage:
    python scripts/check_regression.py --current results/sst2-comparison.json \
        --baseline results/baselines/sst2-comparison.json \
        --metric accuracy_vs_gold:max --metric ece_15_bins:min
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from laya_evals.baseline import Metric, check_regression


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True,
                        help="fresh measurement JSON")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="committed baseline JSON")
    parser.add_argument("--metric", action="append", required=True, metavar="PATH:GOAL[:TOL]",
                        help="e.g. accuracy_vs_gold:max or ece_15_bins:min:0.01")
    return parser.parse_args(argv)


def parse_metric(spec: str) -> Metric:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"metric spec must be path:goal[:tolerance], got {spec!r}")
    path, goal = parts[0], parts[1]
    tolerance = float(parts[2]) if len(parts) == 3 else None
    return Metric(path=path, goal=goal, tolerance=tolerance) if tolerance is not None \
        else Metric(path=path, goal=goal)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        current = json.loads(args.current.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        metrics = [parse_metric(spec) for spec in args.metric]
        check = check_regression(current, baseline, metrics)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(check.summary())
    return 1 if check.regressed else 0


if __name__ == "__main__":
    sys.exit(main())

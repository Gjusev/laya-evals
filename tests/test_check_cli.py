"""CLI tests for the regression gate: exit codes and output."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_regression.py"

BASELINE = {"accuracy": 0.90, "ece": 0.05}


def run_cli(current, baseline, metrics):
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--current", str(current), "--baseline", str(baseline)]
        + [arg for m in metrics for arg in ("--metric", m)],
        capture_output=True, text=True,
    )


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestCheckCli:
    def test_passing_report_exits_zero(self, tmp_path):
        current = write(tmp_path, "current.json", {"accuracy": 0.895, "ece": 0.05})
        baseline = write(tmp_path, "baseline.json", BASELINE)
        result = run_cli(current, baseline, ["accuracy:max", "ece:min"])
        assert result.returncode == 0
        assert "REGRESS" not in result.stdout

    def test_regression_exits_one_and_names_the_metric(self, tmp_path):
        current = write(tmp_path, "current.json", {"accuracy": 0.80, "ece": 0.05})
        baseline = write(tmp_path, "baseline.json", BASELINE)
        result = run_cli(current, baseline, ["accuracy:max", "ece:min"])
        assert result.returncode == 1
        assert "accuracy" in result.stdout
        assert "REGRESS" in result.stdout

    def test_tolerance_in_metric_spec(self, tmp_path):
        # drop of 0.03 passes with tolerance 0.05, fails with the default
        current = write(tmp_path, "current.json", {"accuracy": 0.87, "ece": 0.05})
        baseline = write(tmp_path, "baseline.json", BASELINE)
        loose = run_cli(current, baseline, ["accuracy:max:0.05", "ece:min"])
        strict = run_cli(current, baseline, ["accuracy:max:0.01", "ece:min"])
        assert loose.returncode == 0
        assert strict.returncode == 1

    def test_missing_file_exits_two(self, tmp_path):
        baseline = write(tmp_path, "baseline.json", BASELINE)
        result = run_cli(tmp_path / "nope.json", baseline, ["accuracy:max"])
        assert result.returncode == 2

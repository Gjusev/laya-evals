"""Reproduce laya's published MASSIVE and XNLI benchmark claims on Kaggle.

Clones laya-evals from GitHub, installs it with the compare extra, and
re-runs the reproduction pack under the upstream notebook's exact protocol
(seed 13, gold + 19 distractors, first 300 test rows per language, all
14 MASSIVE and 15 XNLI languages) on CPU. Output lands in
/kaggle/working/reproduction.json next to a printed verdict table.

Roughly 20-30 minutes of CPU inference.
"""

import json
import subprocess
import sys


def run(command):
    print(f"$ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


run("git clone --depth 1 https://github.com/Gjusev/laya-evals.git")
run(f"{sys.executable} -m pip install -q --no-input './laya-evals[compare]'")
run(f"{sys.executable} laya-evals/scripts/reproduction_pack.py "
    "--per-lang 300 --out /kaggle/working/reproduction.json")

with open("/kaggle/working/reproduction.json", encoding="utf-8") as handle:
    report = json.load(handle)

print("\n=== What reproduces ===")
for key, entry in report["claims"].items():
    line = (f"{key}: claimed {entry['claimed']:.3f} "
            f"measured {entry['measured']:.4f} ({entry['verdict']})")
    if "caveat" in entry:
        line += f" -- {entry['caveat']}"
    print(line)

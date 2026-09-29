# laya-evals

> Cheap LLM-as-judge for CI plus a calibration auditor: score eval sets with a local System 1 decision model and audit confidence before automating on it.

Status: early development. Built on [laya](https://github.com/NandhaKishorM/laya),
the open-source System 1 decision engine (Apache 2.0).

## Why

- The gaps are filed as open issues in the upstream repo: confidence thresholds do not transfer across option counts (#394), the published ECE columns no longer reproduce (#208), and independent evals had to be hand-rolled (#555, #450).
- In one large independent eval, answers with confidence >= 0.9 were only 72.2% accurate: automating on confidence without auditing calibration is the classic silent failure.
- Unofficial community tool, not the laya-evals CLI shipped inside the upstream package.
- Every project in this portfolio needs honest evals; this one is that capability, productized.

## Roadmap

- [ ] Calibration core: ECE, Brier, reliability bins and coverage/accuracy curves implemented from scratch, tested against synthetic distributions with known values
- [ ] Judge: rubric-based score and choice questions as a drop-in cheap replacement for LLM judges, batched
- [ ] Judge comparison: agreement (kappa) and cost per 1k judgments vs an LLM judge on a public set
- [ ] Reproduction pack: re-run public benchmark claims (MASSIVE and XNLI subsets) and publish what reproduces
- [ ] Threshold advisor: recommended min_confidence per question shape at a target accuracy, by option count
- [ ] GitHub Action: fail the build when accuracy or calibration regresses

## Development setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or any venv + pip):

```bash
uv venv
uv pip install -e ".[dev]"
pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).

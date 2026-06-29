# Evaluation Artifacts

This repository keeps evaluation reproducible through configuration and small summaries, not through
large generated prediction tables.

## Versionable

- `configs/model.yaml`, including fold definitions, seed, bootstrap iterations and the declared
  selected goal model.
- Markdown reports under `artifacts/evaluation/**/report.md`.
- Small CSV/JSON summaries under `artifacts/evaluation/**`, such as fold reports, search metrics,
  paired-comparison summaries and selected configs.

## Regenerable and ignored

- Out-of-fold and holdout prediction Parquet files under `artifacts/evaluation/**/predictions_*.parquet`.
- Binary or full model artifacts under `artifacts/models/**`.

These files can be regenerated locally with:

```bash
uv run wc2026 model dixon-coles
uv run wc2026 evaluate dixon-coles
```

## Temporal naming

- `out_of_fold` means a validation prediction produced by the configured temporal folds.
- `holdout_2026` means a retrospective 2026 evaluation set with observed results available. It is
  excluded from hyperparameter selection.
- `prospective` is reserved for predictions where the kickoff is after the recorded `data_cutoff`
  and the result is not yet available.

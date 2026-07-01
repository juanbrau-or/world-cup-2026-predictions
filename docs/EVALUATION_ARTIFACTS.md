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
uv run wc2026 model contextual-challenger
uv run wc2026 evaluate contextual-challenger
```

## Temporal naming

- `out_of_fold` means a validation prediction produced by the configured temporal folds.
- `holdout_2026` means a retrospective 2026 evaluation set with observed results available. It is
  excluded from hyperparameter selection.
- `prospective` is reserved for predictions where the kickoff is after the recorded `data_cutoff`
  and the result is not yet available.

## Prospective operational scorecard

`uv run wc2026 evaluate prospective` builds a canonical append-only ledger from immutable
prediction snapshots and then applies a versioned official selection policy. The current official
policy is `early_v1_2026_06_30`: for context `early_v1`, select the latest valid prediction at least
6 hours before kickoff; if none exists, select the earliest valid prediction before kickoff. This
selection does not inspect the result.

The official 1X2 metric uses `result_90` only. Extra time and penalties are reported as separate
columns in the match-level output and are not mixed into the 1X2 outcome.

## Contextual challenger shadow artifacts

`contextual_logit_v1` and `contextual_lgbm_v1` are challenger models only. They are evaluated
against `poisson_goal_v1` on the same temporal folds and the same `match_id` set, but they do not
replace the official model.

Versionable inputs:

- `configs/contextual_challenger_features.yaml`, the explicit feature whitelist.
- `configs/model.yaml`, section `contextual_challenger`.
- `configs/shadow_contextual_evaluation.yaml`, the shadow prospective selection policy.

Generated evaluation outputs:

- `artifacts/evaluation/contextual_challenger/fold_metrics.csv`
- `artifacts/evaluation/contextual_challenger/paired_comparison.csv`
- `artifacts/evaluation/contextual_challenger/bootstrap_report.json`
- `artifacts/evaluation/contextual_challenger/calibration_report.json`
- `artifacts/evaluation/contextual_challenger/ablation_report.csv`
- `artifacts/evaluation/contextual_challenger/feature_importance.csv`
- `artifacts/evaluation/contextual_challenger/report.md`
- `artifacts/evaluation/contextual_challenger/predictions_out_of_fold.parquet`

The Parquet prediction files remain ignored and regenerable. Feature importance is diagnostic only
and must not be read as causal evidence.

Versionable configuration:

- `configs/prospective_evaluation.yaml`, including horizon buckets, official policy and baselines.

Generated outputs:

- `predictions/prospective_scorecard.json`
- `predictions/prospective_scorecard.md`
- `predictions/prospective_matches.csv`
- `predictions/prediction_ledger.parquet`

Shadow contextual prospective outputs are separate:

- `predictions/shadow/contextual_latest.csv`
- `predictions/shadow/contextual_latest.parquet`
- `predictions/shadow/contextual_upcoming.md`
- `predictions/shadow/contextual_ledger.parquet`
- `predictions/shadow/contextual_scorecard.json`
- `predictions/shadow/contextual_scorecard.md`
- `predictions/shadow/contextual_comparison.md`

The Parquet ledger is an operational artifact, not a `predictions-data` branch file. Aggregates must
always be read with their sample size; when the official sample is below the configured threshold,
the Markdown report explicitly warns that the metrics are monitoring diagnostics rather than
evidence of statistical improvement.

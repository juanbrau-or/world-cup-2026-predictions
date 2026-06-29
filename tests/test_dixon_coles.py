from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.evaluation.dixon_coles_backtest import (
    DixonColesBacktestError,
    DixonColesEvaluationConfig,
    DixonColesEvaluationOutputConfig,
    DixonColesModelConfig,
    DixonColesSearchConfig,
    run_dixon_coles_evaluation,
)
from worldcup2026.evaluation.elo_backtest import (
    BacktestFoldConfig,
    CompetitionWeightsProfile,
    EloEvaluationConfig,
    EloEvaluationOutputConfig,
    EloParameterSearchConfig,
)
from worldcup2026.models.dixon_coles import (
    DixonColesGoalModel,
    score_distribution,
    temporal_weights,
)


def test_poisson_model_identifiability_and_score_normalization() -> None:
    rows = [
        _match_row("m1", "2019-01-01", home_team_id="a", away_team_id="b", home_goals_90=2),
        _match_row("m2", "2019-01-08", home_team_id="b", away_team_id="a", away_goals_90=2),
        _match_row("m3", "2019-01-15", home_team_id="c", away_team_id="d", away_goals_90=1),
        _match_row("m4", "2019-01-22", home_team_id="d", away_team_id="c", home_goals_90=1),
    ]
    model = DixonColesGoalModel(
        model_type="poisson",
        half_life_days=None,
        max_goals=3,
        regularization_strength=0.1,
    )

    model.fit(rows, cutoff=date(2020, 1, 1))

    assert model.parameters is not None
    assert sum(model.parameters.attack.values()) == pytest.approx(0.0)
    assert sum(model.parameters.defense.values()) == pytest.approx(0.0)
    distribution = score_distribution(
        model.parameters,
        home_team="a",
        away_team="b",
        competition_category="friendly",
        home_advantage_eligible=True,
    )
    assert distribution.expected_home_goals > 0
    assert distribution.expected_away_goals > 0
    assert (
        distribution.prob_home_win + distribution.prob_draw + distribution.prob_away_win
    ) == pytest.approx(1.0)
    assert distribution.score_probability_mass + distribution.residual_probability == pytest.approx(
        1.0
    )
    assert distribution.residual_probability > 0.0
    assert set(distribution.score_probabilities) == {
        f"{home}-{away}" for home in range(4) for away in range(4)
    }


def test_temporal_weights_half_life_uses_only_past_distance() -> None:
    rows = [
        _match_row("old", "2019-07-05"),
        _match_row("recent", "2020-01-01"),
    ]

    weights = temporal_weights(rows, cutoff=date(2020, 1, 1), half_life_days=180)

    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(1.0)


def test_dixon_coles_evaluation_writes_oof_predictions_and_compares_elo(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "matches.parquet"
    _write_matches(input_path, _sample_rows(include_2026=True))
    model_config = _model_config(tmp_path, input_path)
    elo_config = _elo_evaluation_config(tmp_path, input_path)
    dc_config = _dc_evaluation_config(tmp_path, input_path)
    _write_elo_metrics(elo_config.outputs.metrics_by_fold_path)
    _write_elo_predictions(elo_config.outputs.out_of_fold_predictions_path)

    result = run_dixon_coles_evaluation(model_config, elo_config, dc_config)
    predictions = pq.read_table(result.out_of_fold_predictions_path).to_pylist()
    selected_config = json.loads(result.selected_config_path.read_text(encoding="utf-8"))
    comparison_rows = list(csv.DictReader(result.comparison_with_elo_path.open()))
    paired_summary = list(csv.DictReader(result.paired_comparison_summary_path.open()))
    search_rows = list(csv.DictReader(result.search_metrics_path.open()))

    assert result.validation_matches == 2
    assert result.holdout_2026_matches == 1
    assert selected_config["validation_folds"] == ["mini_cup"]
    assert {row["half_life_days"] for row in search_rows} == {"180.0", "none"}
    assert {row["model_type"] for row in search_rows} == {"poisson", "dixon_coles"}
    assert comparison_rows[0]["matches"] == "2"
    assert float(comparison_rows[0]["elo_log_loss"]) > 0
    assert {row["pair"] for row in paired_summary} == {
        "dixon_coles_minus_elo",
        "poisson_minus_dixon_coles",
        "poisson_minus_elo",
    }
    for row in predictions:
        probabilities = [
            row["prob_home_win"],
            row["prob_draw"],
            row["prob_away_win"],
        ]
        assert sum(probabilities) == pytest.approx(1.0)
        assert row["data_cutoff"] == "2020-02-01"
        assert row["prediction_status"] == "out_of_fold"
        assert row["generated_at"]
        score_probabilities = json.loads(row["score_probabilities_json"])
        assert sum(score_probabilities.values()) + row["residual_probability"] == pytest.approx(
            1.0
        )
        assert row["goal_log_likelihood"] <= 0.0


def test_dixon_coles_evaluation_fails_when_elo_match_ids_differ(tmp_path: Path) -> None:
    input_path = tmp_path / "matches.parquet"
    _write_matches(input_path, _sample_rows(include_2026=False))
    model_config = _model_config(tmp_path, input_path)
    elo_config = _elo_evaluation_config(tmp_path, input_path)
    dc_config = _dc_evaluation_config(tmp_path, input_path)
    _write_elo_metrics(elo_config.outputs.metrics_by_fold_path)
    _write_elo_predictions(elo_config.outputs.out_of_fold_predictions_path, match_ids=("m4",))

    with pytest.raises(DixonColesBacktestError, match="match_id set"):
        run_dixon_coles_evaluation(model_config, elo_config, dc_config)


def test_dixon_coles_cli_model_and_evaluate(tmp_path: Path) -> None:
    input_path = tmp_path / "matches.parquet"
    _write_matches(input_path, _sample_rows(include_2026=False))
    config_path = tmp_path / "model.yaml"
    model_config = _model_config(tmp_path, input_path)
    elo_config = _elo_evaluation_config(tmp_path, input_path)
    dc_config = _dc_evaluation_config(tmp_path, input_path)
    _write_elo_metrics(elo_config.outputs.metrics_by_fold_path)
    _write_elo_predictions(elo_config.outputs.out_of_fold_predictions_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "dixon_coles": model_config.model_dump(mode="json"),
                "elo_evaluation": elo_config.model_dump(mode="json"),
                "dixon_coles_evaluation": dc_config.model_dump(mode="json"),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    model_result = runner.invoke(app, ["model", "dixon-coles", "--config", str(config_path)])
    eval_result = runner.invoke(app, ["evaluate", "dixon-coles", "--config", str(config_path)])

    assert model_result.exit_code == 0
    assert "Model type:" in model_result.stdout
    assert model_config.output_model_path.is_file()
    assert eval_result.exit_code == 0
    assert "Selected model:" in eval_result.stdout
    assert dc_config.outputs.out_of_fold_predictions_path.is_file()


def _model_config(tmp_path: Path, input_path: Path) -> DixonColesModelConfig:
    return DixonColesModelConfig(
        input_matches_path=input_path,
        output_model_path=tmp_path / "models" / "model.json",
        report_output=tmp_path / "models" / "report.json",
        final_holdout_start=date(2026, 1, 1),
        model_type="dixon_coles",
        time_decay_half_life_days=365,
        regularization_strength=0.1,
        max_goals=4,
    )


def _dc_evaluation_config(tmp_path: Path, input_path: Path) -> DixonColesEvaluationConfig:
    root = tmp_path / "dc_eval"
    return DixonColesEvaluationConfig(
        input_matches_path=input_path,
        bootstrap_iterations=200,
        search=DixonColesSearchConfig(
            model_types=("poisson", "dixon_coles"),
            half_life_days=(180.0, None),
            regularization_strength=(0.1,),
            max_goals=4,
        ),
        outputs=DixonColesEvaluationOutputConfig(
            root=root,
            selected_config_path=root / "selected_config.json",
            search_metrics_path=root / "search_metrics.csv",
            metrics_by_fold_path=root / "metrics_by_fold.csv",
            comparison_with_elo_path=root / "comparison_with_elo.csv",
            fold_report_path=root / "fold_report.csv",
            paired_match_comparisons_path=root / "paired_match_comparisons.csv",
            paired_comparison_summary_path=root / "paired_comparison_summary.csv",
            evaluation_summary_path=root / "evaluation_summary.json",
            out_of_fold_predictions_path=root / "predictions_out_of_fold.parquet",
            holdout_2026_predictions_path=root / "predictions_2026_holdout.parquet",
            report_path=root / "report.md",
        ),
    )


def _elo_evaluation_config(tmp_path: Path, input_path: Path) -> EloEvaluationConfig:
    root = tmp_path / "elo_eval"
    return EloEvaluationConfig(
        input_matches_path=input_path,
        random_seed=2026,
        calibration_bins=4,
        final_holdout_start=date(2026, 1, 1),
        folds=(
            BacktestFoldConfig(
                name="mini_cup",
                start=date(2020, 2, 1),
                end=date(2020, 2, 2),
                competitions=("Mini Cup",),
            ),
        ),
        search=EloParameterSearchConfig(
            k_base=(20,),
            home_advantage=(0,),
            competition_weight_profiles=(
                CompetitionWeightsProfile(
                    name="flat",
                    weights={
                        "friendly": 1,
                        "world_cup_qualifier": 1,
                        "confederation_qualifier": 1,
                        "world_cup": 1,
                        "confederation_championship": 1,
                        "nations_league": 1,
                        "other_official": 1,
                        "other": 1,
                    },
                ),
            ),
        ),
        outputs=EloEvaluationOutputConfig(
            root=root,
            selected_config_path=root / "selected_config.json",
            search_metrics_path=root / "search_metrics.csv",
            metrics_by_fold_path=root / "metrics_by_fold.csv",
            segment_metrics_path=root / "metrics_by_segment.csv",
            out_of_fold_predictions_path=root / "predictions_out_of_fold.parquet",
            holdout_2026_predictions_path=root / "predictions_2026_holdout.parquet",
            calibration_curves_path=root / "calibration_curves.csv",
            report_path=root / "report.md",
        ),
    )


def _write_elo_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("fold", "log_loss", "brier_score", "ranked_probability_score"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "fold": "mini_cup",
                "log_loss": "1.2",
                "brier_score": "0.7",
                "ranked_probability_score": "0.3",
            }
        )


def _write_elo_predictions(path: Path, *, match_ids: tuple[str, ...] = ("m4", "m5")) -> None:
    rows = []
    for match_id in match_ids:
        actual_result = "home_win" if match_id == "m4" else "away_win"
        rows.append(
            {
                "fold": "mini_cup",
                "match_id": match_id,
                "match_date": "2020-02-01" if match_id == "m4" else "2020-02-02",
                "kickoff_utc": "2020-02-01T12:00:00+00:00",
                "data_cutoff": "2020-02-01",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "prediction_status": "out_of_fold",
                "home_team_id": "b" if match_id == "m4" else "a",
                "away_team_id": "d",
                "competition": "Mini Cup",
                "competition_category": "friendly",
                "actual_result": actual_result,
                "prob_home_win": 0.5,
                "prob_draw": 0.25,
                "prob_away_win": 0.25,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _sample_rows(*, include_2026: bool) -> list[dict[str, Any]]:
    rows = [
        _match_row("m1", "2019-01-01", home_team_id="a", away_team_id="b", home_goals_90=2),
        _match_row(
            "m2",
            "2019-01-08",
            home_team_id="c",
            away_team_id="d",
            home_goals_90=0,
            away_goals_90=1,
        ),
        _match_row("m3", "2019-01-15", home_team_id="a", away_team_id="c", away_goals_90=1),
        _match_row(
            "m4",
            "2020-02-01",
            home_team_id="b",
            away_team_id="d",
            competition="Mini Cup",
        ),
        _match_row(
            "m5",
            "2020-02-02",
            home_team_id="a",
            away_team_id="d",
            competition="Mini Cup",
            home_goals_90=0,
            away_goals_90=2,
        ),
    ]
    if include_2026:
        rows.append(
            _match_row(
                "m6",
                "2026-01-01",
                home_team_id="a",
                away_team_id="b",
                competition="Future Cup",
            )
        )
    return rows


def _write_matches(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _match_row(
    match_id: str,
    match_date: str,
    *,
    home_team_id: str = "home",
    away_team_id: str = "away",
    home_goals_90: int = 1,
    away_goals_90: int = 0,
    competition: str = "Friendly",
    competition_category: str = "friendly",
) -> dict[str, Any]:
    parsed_date = date.fromisoformat(match_date)
    return {
        "match_id": match_id,
        "match_date": parsed_date,
        "kickoff_utc": datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            12,
            tzinfo=UTC,
        ),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals_90": home_goals_90,
        "away_goals_90": away_goals_90,
        "competition": competition,
        "competition_category": competition_category,
        "model_eligible": True,
        "neutral_site": False,
        "home_advantage_eligible": True,
        "home_advantage_status": "home_team",
    }

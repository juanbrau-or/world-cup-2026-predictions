from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.evaluation.elo_backtest import (
    BacktestFoldConfig,
    CompetitionWeightsProfile,
    EloEvaluationConfig,
    EloEvaluationError,
    EloEvaluationOutputConfig,
    EloParameterSearchConfig,
    build_walk_forward_folds,
    run_elo_evaluation,
)
from worldcup2026.features.elo import EloRatingsConfig, load_elo_ratings_config


def test_walk_forward_fold_rejects_future_training_rows() -> None:
    rows = [
        _match_row("train", "2020-01-01", competition="Friendly"),
        _match_row("test", "2020-02-01", competition="Test Cup"),
    ]
    fold = BacktestFoldConfig(
        name="test_cup",
        start=date(2020, 2, 1),
        end=date(2020, 2, 1),
        competitions=("Test Cup",),
    )
    folds = build_walk_forward_folds(rows, (fold,))

    assert folds[0]["train_ids"] == {"train"}
    assert folds[0]["test_ids"] == {"test"}
    assert all(
        _row_by_id(rows, match_id)["match_date"] < fold.start
        for match_id in folds[0]["train_ids"]
    )


def test_walk_forward_fold_fails_when_no_past_training_rows() -> None:
    rows = [_match_row("test", "2020-02-01", competition="Test Cup")]
    fold = BacktestFoldConfig(
        name="test_cup",
        start=date(2020, 2, 1),
        end=date(2020, 2, 1),
        competitions=("Test Cup",),
    )

    with pytest.raises(EloEvaluationError):
        build_walk_forward_folds(rows, (fold,))


def test_elo_evaluation_writes_probabilities_that_sum_to_one(tmp_path: Path) -> None:
    input_path = tmp_path / "matches.parquet"
    _write_matches(
        input_path,
        [
            _match_row("m1", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row("m2", "2020-01-08", home_team_id="c", away_team_id="d", home_goals_90=0),
            _match_row("m3", "2020-01-15", home_team_id="a", away_team_id="c", away_goals_90=1),
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
                away_goals_90=1,
            ),
            _match_row(
                "m6",
                "2026-01-01",
                home_team_id="a",
                away_team_id="b",
                competition="Future Cup",
            ),
        ],
    )
    base_config = _base_elo_config(tmp_path)
    eval_config = _evaluation_config(tmp_path, input_path)

    result = run_elo_evaluation(base_config, eval_config)
    rows = pq.read_table(result.out_of_fold_predictions_path).to_pylist()

    assert rows
    assert result.holdout_2026_matches == 1
    for row in rows:
        total = row["prob_home_win"] + row["prob_draw"] + row["prob_away_win"]
        assert total == pytest.approx(1.0)
        assert row["data_cutoff"] == "2020-02-01"
        assert row["prediction_status"] == "out_of_fold"
        assert row["generated_at"]


def test_evaluate_elo_cli_writes_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "matches.parquet"
    _write_matches(
        input_path,
        [
            _match_row("m1", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row("m2", "2020-01-08", home_team_id="c", away_team_id="d", home_goals_90=0),
            _match_row(
                "m3",
                "2020-02-01",
                home_team_id="a",
                away_team_id="d",
                competition="Mini Cup",
            ),
            _match_row(
                "m4",
                "2020-02-02",
                home_team_id="b",
                away_team_id="c",
                competition="Mini Cup",
                home_goals_90=0,
                away_goals_90=1,
            ),
        ],
    )
    base_config = _base_elo_config(tmp_path)
    eval_config = _evaluation_config(tmp_path, input_path)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        json.dumps(
            {
                "elo": base_config.model_dump(mode="json"),
                "elo_evaluation": eval_config.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["evaluate", "elo", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Selected method:" in result.stdout
    assert eval_config.outputs.selected_config_path.is_file()
    assert eval_config.outputs.calibration_curves_path.is_file()


def _base_elo_config(tmp_path: Path) -> EloRatingsConfig:
    config = load_elo_ratings_config().model_copy(
        update={
            "input_matches_path": tmp_path / "unused.parquet",
            "output_match_ratings_path": tmp_path / "unused_match.parquet",
            "output_current_ratings_path": tmp_path / "unused_current.parquet",
            "report_output": tmp_path / "unused_report.json",
            "competition_importance": {
                "friendly": 1,
                "world_cup_qualifier": 1,
                "confederation_qualifier": 1,
                "world_cup": 1,
                "confederation_championship": 1,
                "nations_league": 1,
                "other_official": 1,
                "other": 1,
            },
        }
    )
    return EloRatingsConfig.model_validate(config.model_dump(mode="python"))


def _evaluation_config(tmp_path: Path, input_path: Path) -> EloEvaluationConfig:
    root = tmp_path / "evaluation"
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


def _write_matches(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _row_by_id(rows: list[dict[str, Any]], match_id: str) -> dict[str, Any]:
    return next(row for row in rows if row["match_id"] == match_id)


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

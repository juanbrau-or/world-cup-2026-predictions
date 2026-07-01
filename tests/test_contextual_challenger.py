from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.evaluation.contextual_challenger import (
    ContextualChallengerError,
    TemperatureCalibrationConfig,
    _fit_temperature_scaler,
    load_feature_whitelist,
    run_contextual_challenger_evaluation,
)


def test_feature_whitelist_rejects_prohibited_operational_features(tmp_path: Path) -> None:
    path = tmp_path / "features.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "contextual_challenger_feature_whitelist_v1",
                "feature_set_version": "bad",
                "description": "bad",
                "base_features": ["base_poisson_prob_home_win"],
                "contextual_features": ["home_travel_distance_km"],
                "categorical_features": [],
                "excluded_features": ["home_travel_distance_km"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContextualChallengerError, match="excluded features"):
        load_feature_whitelist(path)


def test_temperature_scaling_keeps_identity_when_not_better() -> None:
    scaler = _fit_temperature_scaler(
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        np.array([0, 1]),
        config=TemperatureCalibrationConfig(candidate_temperatures=(1.0, 2.0)),
    )

    assert scaler.method == "identity"
    assert scaler.temperature == 1.0


def test_contextual_challenger_evaluation_writes_artifacts(tmp_path: Path) -> None:
    config_path, matches_path, contextual_path = _write_challenger_fixture(tmp_path)

    result = run_contextual_challenger_evaluation(config_path=config_path)

    assert result.validation_matches == 2
    assert result.selected_model_name in {"contextual_logit_v1", "contextual_lgbm_v1"}
    assert result.out_of_fold_predictions_path.is_file()
    assert result.fold_metrics_path.is_file()
    assert result.bootstrap_report_path.is_file()
    assert result.ablation_report_path.is_file()
    predictions = pq.read_table(result.out_of_fold_predictions_path).to_pylist()
    assert {row["ablation"] for row in predictions} == {
        "poisson_official",
        "logistic_stack",
        "contextual_logistic",
        "lgbm_stack",
        "lgbm_contextual",
    }
    for row in predictions:
        total = row["prob_home_win"] + row["prob_draw"] + row["prob_away_win"]
        assert total == pytest.approx(1.0)
        assert row["match_id"] in {"m13", "m14"}
    selected = json.loads(result.selected_config_path.read_text(encoding="utf-8"))
    assert selected["official_baseline_model_version"] == "poisson_goal_v1"
    assert "home_travel_distance_km" not in selected["contextual_features"]
    assert matches_path.is_file()
    assert contextual_path.is_file()


def test_contextual_challenger_cli_commands(tmp_path: Path) -> None:
    config_path, _, _ = _write_challenger_fixture(tmp_path)
    runner = CliRunner()

    model_result = runner.invoke(
        app,
        ["model", "contextual-challenger", "--config", str(config_path)],
    )
    eval_result = runner.invoke(
        app,
        ["evaluate", "contextual-challenger", "--config", str(config_path)],
    )

    assert model_result.exit_code == 0
    assert "Selected model:" in model_result.stdout
    assert eval_result.exit_code == 0
    assert "Selected shadow model:" in eval_result.stdout


def _write_challenger_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    matches_path = tmp_path / "matches.parquet"
    contextual_path = tmp_path / "contextual.parquet"
    features_path = tmp_path / "features.yaml"
    config_path = tmp_path / "model.yaml"
    rows = [_match_row(f"m{index}", date(2020, 1, index)) for index in range(1, 13)]
    rows.extend(
        [
            _match_row("m13", date(2020, 3, 1), competition="Mini Cup", home_goals_90=2),
            _match_row(
                "m14",
                date(2020, 3, 2),
                competition="Mini Cup",
                home_team_id="c",
                away_team_id="d",
                home_goals_90=0,
                away_goals_90=1,
            ),
            _match_row("m15", date(2026, 1, 1), competition="Future Cup"),
        ]
    )
    _write_rows(matches_path, rows)
    _write_rows(contextual_path, [_contextual_row(row) for row in rows])
    features_path.write_text(
        Path("configs/contextual_challenger_features.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = {
        "dixon_coles": {
            "model_version": "poisson_goal_v1",
            "input_matches_path": str(matches_path),
            "output_model_path": str(tmp_path / "model.json"),
            "report_output": str(tmp_path / "model_report.json"),
            "final_holdout_start": "2026-01-01",
            "model_type": "poisson",
            "time_decay_half_life_days": 730,
            "regularization_strength": 0.05,
            "max_goals": 4,
        },
        "elo": {
            "initial_rating": 1500,
            "k_base": 20,
            "home_advantage": 0,
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
            "margin_of_victory": {"enabled": False, "goal_difference_weight": 0},
            "rating_regression_after_inactivity": {
                "enabled": False,
                "inactivity_days": 365,
                "regression_fraction": 0,
            },
            "model_version": "elo_v1",
            "input_matches_path": str(matches_path),
            "output_match_ratings_path": str(tmp_path / "elo.parquet"),
            "output_current_ratings_path": str(tmp_path / "elo_current.parquet"),
            "report_output": str(tmp_path / "elo_report.json"),
        },
        "elo_evaluation": {
            "input_matches_path": str(matches_path),
            "random_seed": 2026,
            "calibration_bins": 4,
            "final_holdout_start": "2026-01-01",
            "folds_version": "tournament_validation_v1",
            "folds": [
                {
                    "name": "mini_cup",
                    "start": "2020-03-01",
                    "end": "2020-03-02",
                    "competitions": ["Mini Cup"],
                }
            ],
            "search": {
                "k_base": [20],
                "home_advantage": [0],
                "competition_weight_profiles": [
                    {
                        "name": "flat",
                        "weights": {
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
                ],
            },
        },
        "contextual_challenger": {
            "schema_version": "contextual_challenger_config_v1",
            "official_baseline_model_version": "poisson_goal_v1",
            "sanity_model_name": "contextual_logit_v1",
            "primary_model_name": "contextual_lgbm_v1",
            "shadow_prediction_context": "shadow_contextual_v1",
            "input_matches_path": str(matches_path),
            "contextual_match_features_path": str(contextual_path),
            "feature_whitelist_path": str(features_path),
            "final_holdout_start": "2026-01-01",
            "folds_version": "tournament_validation_v1",
            "random_seed": 2026,
            "bootstrap_iterations": 20,
            "max_threads": 1,
            "split": {
                "tuning_window_days": 20,
                "calibration_window_days": 10,
                "minimum_train_matches": 2,
                "base_oof_blocks": 2,
            },
            "calibration": {
                "method": "temperature_scaling",
                "candidate_temperatures": [0.75, 1.0, 1.5],
            },
            "logistic_grid": [{"name": "logit", "c": 1.0, "max_iter": 200}],
            "lightgbm_grid": [
                {
                    "name": "tiny",
                    "learning_rate": 0.1,
                    "n_estimators": 5,
                    "num_leaves": 3,
                    "max_depth": 2,
                    "min_child_samples": 1,
                    "feature_fraction": 1.0,
                    "bagging_fraction": 1.0,
                    "bagging_freq": 0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                }
            ],
            "shadow_selection": {
                "selected_model_name": "contextual_lgbm_v1",
                "selected_ablation": "lgbm_contextual",
                "selected_feature_set": "contextual_level_a_v1",
                "selected_hyperparameters": {
                    "name": "tiny",
                    "learning_rate": 0.1,
                    "n_estimators": 5,
                    "num_leaves": 3,
                    "max_depth": 2,
                    "min_child_samples": 1,
                    "feature_fraction": 1.0,
                    "bagging_fraction": 1.0,
                    "bagging_freq": 0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                },
                "calibration_method": "temperature_scaling",
                "training_cutoff": "2026-01-01",
                "promotion_status": "shadow_monitoring",
            },
            "outputs": {
                "root": str(tmp_path / "eval"),
                "selected_config_path": str(tmp_path / "eval" / "selected_config.json"),
                "model_manifest_path": str(tmp_path / "eval" / "model_manifest.json"),
                "fold_metrics_path": str(tmp_path / "eval" / "fold_metrics.csv"),
                "segment_metrics_path": str(tmp_path / "eval" / "segment_metrics.csv"),
                "paired_comparison_path": str(tmp_path / "eval" / "paired.csv"),
                "bootstrap_report_path": str(tmp_path / "eval" / "bootstrap.json"),
                "calibration_report_path": str(tmp_path / "eval" / "calibration.json"),
                "ablation_report_path": str(tmp_path / "eval" / "ablation.csv"),
                "feature_importance_path": str(tmp_path / "eval" / "importance.csv"),
                "out_of_fold_predictions_path": str(tmp_path / "eval" / "predictions_oof.parquet"),
                "holdout_2026_predictions_path": str(tmp_path / "eval" / "holdout.parquet"),
                "report_path": str(tmp_path / "eval" / "report.md"),
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, matches_path, contextual_path


def _match_row(
    match_id: str,
    match_date: date,
    *,
    competition: str = "Friendly",
    home_team_id: str = "a",
    away_team_id: str = "b",
    home_goals_90: int = 1,
    away_goals_90: int = 0,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "match_status": "played",
        "match_date": match_date,
        "kickoff_utc": datetime(match_date.year, match_date.month, match_date.day, 12, tzinfo=UTC),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals_90": home_goals_90,
        "away_goals_90": away_goals_90,
        "result_90": "home_win"
        if home_goals_90 > away_goals_90
        else "draw"
        if home_goals_90 == away_goals_90
        else "away_win",
        "competition": competition,
        "stage": None,
        "competition_category": "friendly",
        "neutral_site": False,
        "home_advantage_eligible": True,
        "home_advantage_status": "home_team",
        "model_eligible": True,
    }


def _contextual_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "match_id": row["match_id"],
        "home_rest_days": 7.0,
        "away_rest_days": 6.0,
        "home_matches_last_7d": 0,
        "away_matches_last_7d": 1,
        "home_matches_last_14d": 1,
        "away_matches_last_14d": 1,
        "home_matches_last_30d": 2,
        "away_matches_last_30d": 2,
        "home_previous_match_penalty_shootout": False,
        "away_previous_match_penalty_shootout": False,
        "home_consecutive_matches_without_7d_rest": 0,
        "away_consecutive_matches_without_7d_rest": 1,
        "home_tournament_match_number": 1,
        "away_tournament_match_number": 1,
        "home_is_first_tournament_match": True,
        "away_is_first_tournament_match": True,
        "is_neutral_venue": False,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)

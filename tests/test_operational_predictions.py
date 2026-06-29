from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]

from worldcup2026.evaluation.prospective import run_prospective_evaluation
from worldcup2026.pipelines.operational_predictions import run_predict_upcoming


def test_predict_upcoming_filters_fixtures_and_writes_versioned_outputs(tmp_path: Path) -> None:
    config_path = _write_model_config(tmp_path)
    modeling_path = tmp_path / "modeling.parquet"
    live_path = tmp_path / "live.parquet"
    ingest_report_path = tmp_path / "ingest_report.json"
    predictions_root = tmp_path / "predictions"
    cutoff = datetime(2026, 6, 20, 10, tzinfo=UTC)
    _write_rows(modeling_path, _historical_rows())
    _write_rows(live_path, _live_rows(cutoff=cutoff))
    ingest_report_path.write_text(
        json.dumps(
            {
                "data_cutoff_utc": cutoff.isoformat(),
                "snapshot_checksum": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    first = run_predict_upcoming(
        model_config_path=config_path,
        modeling_matches_path=modeling_path,
        live_matches_path=live_path,
        ingest_report_path=ingest_report_path,
        predictions_root=predictions_root,
        created_at=datetime(2026, 6, 20, 12, tzinfo=UTC),
    )
    second = run_predict_upcoming(
        model_config_path=config_path,
        modeling_matches_path=modeling_path,
        live_matches_path=live_path,
        ingest_report_path=ingest_report_path,
        predictions_root=predictions_root,
        created_at=datetime(2026, 6, 20, 12, 1, tzinfo=UTC),
    )

    assert len(first.predictions) == 1
    prediction = first.predictions[0]
    assert prediction["source_fixture_id"] == "future-known"
    assert prediction["prediction_status"] == "prospective"
    assert prediction["prediction_context"] == "early_v1"
    assert prediction["dataset_revision"].startswith("operational_dataset_v1:")
    assert prediction["live_snapshot_checksum"] == "a" * 64
    assert prediction["model_family"] == "poisson"
    assert json.loads(prediction["selected_config_json"])["time_decay_half_life_days"] == 730
    assert prediction["live_finished_2026_matches"] == 1
    probability_sum = (
        prediction["probability_home_win"]
        + prediction["probability_draw"]
        + prediction["probability_away_win"]
    )
    assert probability_sum == pytest.approx(1.0)
    assert first.latest_csv_path.is_file()
    assert first.latest_parquet_path.is_file()
    assert first.history_path.is_file()
    assert first.report_path.is_file()
    assert second.history_path != first.history_path
    assert len(list((predictions_root / "history").glob("*.parquet"))) == 2
    latest_rows = pq.read_table(second.latest_parquet_path).to_pylist()
    assert latest_rows[0]["prediction_run_id"] == second.predictions[0]["prediction_run_id"]
    assert latest_rows[0]["prediction_run_id"] != first.predictions[0]["prediction_run_id"]
    assert second.excluded_fixtures["finished"] == 1
    assert second.excluded_fixtures["in_progress"] == 1
    assert second.excluded_fixtures["cancelled"] == 1
    assert second.excluded_fixtures["unresolved_team"] == 1


def test_prospective_evaluation_reports_zero_when_no_finished_predictions(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    live_path = tmp_path / "live.parquet"
    _write_rows(live_path, [_scheduled_live_row()])

    result = run_prospective_evaluation(
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
    )

    assert result.evaluable_predictions == 0
    assert result.log_loss is None
    assert result.report_path.is_file()
    assert result.json_path.is_file()


def test_prospective_evaluation_scores_saved_predictions(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    _write_rows(history / "run.parquet", [_prediction_history_row()])
    _write_rows(live_path, [_finished_live_row("future-known", home_goals=2, away_goals=0)])

    result = run_prospective_evaluation(
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
    )

    assert result.evaluable_predictions == 1
    assert result.log_loss == pytest.approx(-math.log(0.6))
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["predictions"] == 1
    assert payload["metrics"]["evaluation_sets"]["prospective_real"]["predictions"] == 1


def _write_model_config(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    config = {
        "dixon_coles": {
            "model_version": "poisson_goal_v1",
            "input_matches_path": str(tmp_path / "unused.parquet"),
            "output_model_path": str(tmp_path / "model.json"),
            "report_output": str(tmp_path / "report.json"),
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
                "world_cup": 5,
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
            "input_matches_path": str(tmp_path / "unused.parquet"),
            "output_match_ratings_path": str(tmp_path / "elo_match.parquet"),
            "output_current_ratings_path": str(tmp_path / "elo_current.parquet"),
            "report_output": str(tmp_path / "elo_report.json"),
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _historical_rows() -> list[dict[str, Any]]:
    return [
        _modeling_row("m1", "2020-01-01", "a", "b", 2, 0),
        _modeling_row("m2", "2020-01-08", "b", "a", 1, 1),
        _modeling_row("m3", "2020-01-15", "c", "d", 0, 1),
        _modeling_row("m4", "2020-01-22", "d", "c", 2, 1),
    ]


def _modeling_row(
    match_id: str,
    match_date: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> dict[str, Any]:
    parsed = date.fromisoformat(match_date)
    return {
        "match_id": match_id,
        "match_status": "played",
        "match_date": parsed,
        "kickoff_utc": datetime(parsed.year, parsed.month, parsed.day, 12, tzinfo=UTC),
        "home_team_id": home,
        "away_team_id": away,
        "home_goals_90": home_goals,
        "away_goals_90": away_goals,
        "competition": "Friendly",
        "competition_category": "friendly",
        "model_eligible": True,
        "neutral_site": False,
        "home_advantage_eligible": False,
        "home_advantage_status": "neutral",
    }


def _live_rows(*, cutoff: datetime) -> list[dict[str, Any]]:
    return [
        _finished_live_row("finished", home_goals=1, away_goals=0, cutoff=cutoff),
        _scheduled_live_row(source_fixture_id="future-known", cutoff=cutoff),
        _scheduled_live_row(
            source_fixture_id="in-play",
            status="in_progress",
            kickoff=datetime(2026, 6, 21, 20, tzinfo=UTC),
            cutoff=cutoff,
        ),
        _scheduled_live_row(
            source_fixture_id="cancelled",
            status="cancelled",
            kickoff=datetime(2026, 6, 22, 20, tzinfo=UTC),
            cutoff=cutoff,
        ),
        _scheduled_live_row(
            source_fixture_id="unresolved",
            home_team_id=None,
            kickoff=datetime(2026, 6, 23, 20, tzinfo=UTC),
            cutoff=cutoff,
        ),
    ]


def _scheduled_live_row(
    *,
    source_fixture_id: str = "future-known",
    status: str = "scheduled",
    kickoff: datetime = datetime(2026, 6, 21, 18, tzinfo=UTC),
    cutoff: datetime = datetime(2026, 6, 20, 10, tzinfo=UTC),
    home_team_id: str | None = "a",
) -> dict[str, Any]:
    return {
        "match_id": f"world_cup_2026_football_data:{source_fixture_id}",
        "source_match_id": source_fixture_id,
        "source": "world_cup_2026_football_data",
        "match_status": status,
        "match_date": kickoff.date(),
        "kickoff_utc": kickoff,
        "data_cutoff_utc": cutoff,
        "home_team_id": home_team_id,
        "away_team_id": "b",
        "home_team_name_original": "Team A" if home_team_id is not None else None,
        "away_team_name_original": "Team B",
        "home_goals_90": None,
        "away_goals_90": None,
        "competition": "FIFA World Cup",
        "stage": "GROUP_STAGE",
    }


def _finished_live_row(
    source_fixture_id: str,
    *,
    home_goals: int,
    away_goals: int,
    cutoff: datetime = datetime(2026, 6, 20, 10, tzinfo=UTC),
) -> dict[str, Any]:
    kickoff = datetime(2026, 6, 19, 18, tzinfo=UTC)
    return {
        **_scheduled_live_row(
            source_fixture_id=source_fixture_id,
            status="played",
            kickoff=kickoff,
            cutoff=cutoff,
        ),
        "home_goals_90": home_goals,
        "away_goals_90": away_goals,
        "home_team_id": "c",
        "away_team_id": "d",
        "home_team_name_original": "Team C",
        "away_team_name_original": "Team D",
    }


def _prediction_history_row() -> dict[str, Any]:
    kickoff = datetime(2026, 6, 19, 18, tzinfo=UTC)
    created = datetime(2026, 6, 18, 12, tzinfo=UTC)
    return {
        "prediction_id": "prediction-1",
        "prediction_run_id": "run-1",
        "source_fixture_id": "future-known",
        "prediction_status": "prospective",
        "kickoff_utc": kickoff,
        "prediction_created_at_utc": created,
        "hours_before_kickoff": 30.0,
        "probability_home_win": 0.6,
        "probability_draw": 0.2,
        "probability_away_win": 0.2,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)

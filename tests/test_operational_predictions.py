from __future__ import annotations

import csv
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]

from worldcup2026.evaluation.prospective import (
    ProspectiveEvaluationError,
    build_prediction_ledger_rows,
    load_prospective_evaluation_config,
    metrics_for_rows,
    run_prospective_evaluation,
    select_official_predictions,
)
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
    config_path = _write_prospective_config(tmp_path)
    _write_rows(live_path, [_scheduled_live_row()])

    result = run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    assert result.evaluable_predictions == 0
    assert result.log_loss is None
    assert result.report_path.is_file()
    assert result.json_path.is_file()
    assert result.matches_path.is_file()
    assert result.ledger_path.is_file()


def test_prospective_evaluation_scores_saved_predictions(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    _write_rows(history / "run.parquet", [_prediction_history_row()])
    _write_rows(live_path, [_finished_live_row("future-known", home_goals=2, away_goals=0)])

    result = run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    assert result.evaluable_predictions == 1
    assert result.log_loss == pytest.approx(-math.log(0.6))
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["matches"] == 1
    assert payload["official_selection_policy"]["policy_version"] == "early_v1_test"
    uniform = payload["baselines"]["uniform_1x2"]["metrics"]
    assert uniform["log_loss"] == pytest.approx(-math.log(1 / 3))


def test_official_selection_uses_latest_prediction_at_least_six_hours(
    tmp_path: Path,
) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    kickoff = datetime(2026, 6, 21, 18, tzinfo=UTC)
    rows = [
        _prediction_history_row(
            prediction_id="p10",
            created=kickoff.replace(hour=8),
            kickoff=kickoff,
        ),
        _prediction_history_row(
            prediction_id="p6",
            created=kickoff.replace(hour=12),
            kickoff=kickoff,
        ),
        _prediction_history_row(
            prediction_id="p5",
            created=kickoff.replace(hour=13),
            kickoff=kickoff,
        ),
    ]

    ledger = build_prediction_ledger_rows(rows, config=config)
    selected = select_official_predictions(ledger, config=config)

    assert len(selected) == 1
    assert selected[0]["prediction_id"] == "p6"
    assert selected[0]["official_selection_rule"] == "latest_valid_at_least_6h_before_kickoff"


def test_official_selection_falls_back_to_earliest_before_kickoff(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    kickoff = datetime(2026, 6, 21, 18, tzinfo=UTC)
    rows = [
        _prediction_history_row(
            prediction_id="p5",
            created=kickoff.replace(hour=13),
            kickoff=kickoff,
        ),
        _prediction_history_row(
            prediction_id="p2",
            created=kickoff.replace(hour=16),
            kickoff=kickoff,
        ),
    ]

    ledger = build_prediction_ledger_rows(rows, config=config)
    selected = select_official_predictions(ledger, config=config)

    assert selected[0]["prediction_id"] == "p5"
    assert selected[0]["official_selection_rule"] == "earliest_valid_before_kickoff"


def test_official_selection_does_not_require_known_result(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    _write_rows(history / "run.parquet", [_prediction_history_row()])
    _write_rows(live_path, [_scheduled_live_row()])

    result = run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    assert result.official_predictions_selected == 1
    assert result.evaluable_predictions == 0


def test_ledger_rejects_prediction_after_kickoff(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    row = _prediction_history_row(
        created=datetime(2026, 6, 19, 18, tzinfo=UTC),
        kickoff=datetime(2026, 6, 19, 18, tzinfo=UTC),
    )

    with pytest.raises(ProspectiveEvaluationError, match="prediction_created_at_not_before"):
        build_prediction_ledger_rows([row], config=config)


def test_ledger_rejects_cutoff_after_creation(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    row = _prediction_history_row(
        created=datetime(2026, 6, 18, 12, tzinfo=UTC),
        cutoff=datetime(2026, 6, 18, 13, tzinfo=UTC),
    )

    with pytest.raises(ProspectiveEvaluationError, match="data_cutoff_after"):
        build_prediction_ledger_rows([row], config=config)


def test_ledger_rejects_invalid_probabilities(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    row = _prediction_history_row(probabilities=(0.7, 0.2, 0.2))

    with pytest.raises(ProspectiveEvaluationError, match="probabilities_do_not_sum"):
        build_prediction_ledger_rows([row], config=config)


def test_ledger_accepts_exact_duplicate_idempotently(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    row = _prediction_history_row()

    ledger = build_prediction_ledger_rows([row, dict(row)], config=config)

    assert len(ledger) == 1


def test_ledger_rejects_inconsistent_duplicate_prediction_id(tmp_path: Path) -> None:
    config = load_prospective_evaluation_config(_write_prospective_config(tmp_path))
    first = _prediction_history_row(prediction_id="same")
    second = _prediction_history_row(prediction_id="same", probabilities=(0.5, 0.3, 0.2))

    with pytest.raises(ProspectiveEvaluationError, match="duplicate prediction_id"):
        build_prediction_ledger_rows([first, second], config=config)


def test_scorecard_uses_90_minute_result_when_extra_time_is_played(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    _write_rows(history / "run.parquet", [_prediction_history_row(probabilities=(0.2, 0.6, 0.2))])
    _write_rows(
        live_path,
        [
            _finished_live_row(
                "future-known",
                home_goals=1,
                away_goals=1,
                extra_time_played=True,
                home_goals_after_extra_time=2,
                away_goals_after_extra_time=1,
            )
        ],
    )

    result = run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    assert result.log_loss == pytest.approx(-math.log(0.6))
    match_rows = list(csv.DictReader((tmp_path / "matches.csv").open()))
    assert match_rows[0]["metric_result_1x2"] == "draw"
    assert match_rows[0]["qualification_winner"] == "home"


def test_scorecard_keeps_penalty_result_separate_from_1x2(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    _write_rows(history / "run.parquet", [_prediction_history_row(probabilities=(0.2, 0.6, 0.2))])
    _write_rows(
        live_path,
        [
            _finished_live_row(
                "future-known",
                home_goals=0,
                away_goals=0,
                extra_time_played=True,
                home_goals_after_extra_time=0,
                away_goals_after_extra_time=0,
                penalty_shootout=True,
                home_penalty_goals=3,
                away_penalty_goals=4,
            )
        ],
    )

    run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    match_rows = list(csv.DictReader((tmp_path / "matches.csv").open()))
    assert match_rows[0]["metric_result_1x2"] == "draw"
    assert match_rows[0]["qualification_winner"] == "away"


def test_cancelled_fixture_is_not_evaluated(tmp_path: Path) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    _write_rows(history / "run.parquet", [_prediction_history_row()])
    _write_rows(live_path, [_scheduled_live_row(status="cancelled")])

    result = run_prospective_evaluation(
        config_path=config_path,
        predictions_history_root=history,
        live_matches_path=live_path,
        report_path=tmp_path / "prospective.md",
        json_path=tmp_path / "prospective.json",
        matches_csv_path=tmp_path / "matches.csv",
        ledger_path=tmp_path / "ledger.parquet",
    )

    assert result.official_predictions_selected == 1
    assert result.evaluable_predictions == 0


def test_metrics_known_values_and_uniform_baseline(tmp_path: Path) -> None:
    config_path = _write_prospective_config(tmp_path)
    config = load_prospective_evaluation_config(config_path)
    rows = [
        {
            **_prediction_history_row(probabilities=(0.6, 0.2, 0.2)),
            "metric_result_1x2": "home_win",
        },
        {
            **_prediction_history_row(
                prediction_id="prediction-2",
                source_fixture_id="future-known-2",
                probabilities=(0.2, 0.5, 0.3),
            ),
            "metric_result_1x2": "draw",
        },
    ]

    metrics = metrics_for_rows(rows, minimum_calibration_matches=config.minimum_calibration_matches)

    assert metrics["matches"] == 2
    assert metrics["log_loss"] == pytest.approx((-math.log(0.6) - math.log(0.5)) / 2)


def test_prospective_evaluation_is_reproducible_with_fixed_generated_at(
    tmp_path: Path,
) -> None:
    history = tmp_path / "predictions" / "history"
    history.mkdir(parents=True)
    live_path = tmp_path / "live.parquet"
    config_path = _write_prospective_config(tmp_path)
    generated_at = datetime(2026, 6, 21, tzinfo=UTC)
    _write_rows(history / "run.parquet", [_prediction_history_row()])
    _write_rows(live_path, [_finished_live_row("future-known", home_goals=2, away_goals=0)])

    kwargs = {
        "config_path": config_path,
        "predictions_history_root": history,
        "live_matches_path": live_path,
        "generated_at": generated_at,
    }
    run_prospective_evaluation(
        **kwargs,
        report_path=tmp_path / "first.md",
        json_path=tmp_path / "first.json",
        matches_csv_path=tmp_path / "first.csv",
        ledger_path=tmp_path / "first.parquet",
    )
    run_prospective_evaluation(
        **kwargs,
        report_path=tmp_path / "second.md",
        json_path=tmp_path / "second.json",
        matches_csv_path=tmp_path / "second.csv",
        ledger_path=tmp_path / "second.parquet",
    )

    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()


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


def _write_prospective_config(tmp_path: Path) -> Path:
    path = tmp_path / "prospective_evaluation.yaml"
    config = {
        "schema_version": "prospective_evaluation_config_v1",
        "result_metric_basis": "result_90",
        "minimum_calibration_matches": 30,
        "small_sample_warning_threshold": 30,
        "horizons": {
            "version": "prediction_horizons_test",
            "buckets": [
                {
                    "id": "gt_24h",
                    "label": "> 24h",
                    "min_hours": 24,
                    "min_inclusive": False,
                    "max_hours": None,
                    "max_inclusive": False,
                },
                {
                    "id": "6_to_24h",
                    "label": "6 to 24h",
                    "min_hours": 6,
                    "min_inclusive": True,
                    "max_hours": 24,
                    "max_inclusive": True,
                },
                {
                    "id": "lt_6h",
                    "label": "< 6h",
                    "min_hours": 0,
                    "min_inclusive": True,
                    "max_hours": 6,
                    "max_inclusive": False,
                },
            ],
        },
        "official_selection": {
            "policy_id": "early_v1",
            "policy_version": "early_v1_test",
            "prediction_context": "early_v1",
            "primary_rule": {
                "id": "latest_valid_at_least_6h_before_kickoff",
                "min_hours_before_kickoff": 6,
            },
            "fallback_rule": {"id": "earliest_valid_before_kickoff"},
        },
        "baselines": {
            "uniform_1x2": {"enabled": True},
            "historical_frequency": {
                "enabled": True,
                "input_matches_path": str(tmp_path / "modeling_baseline.parquet"),
                "cutoff_utc": "2026-01-01T00:00:00Z",
            },
            "elo_operational": {"enabled": False},
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _write_rows(
        tmp_path / "modeling_baseline.parquet",
        [
            _modeling_row("b1", "2020-01-01", "a", "b", 2, 0),
            _modeling_row("b2", "2020-01-02", "a", "b", 1, 1),
            _modeling_row("b3", "2020-01-03", "a", "b", 0, 2),
        ],
    )
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
        "result_90": None,
        "extra_time_played": False,
        "home_goals_after_extra_time": None,
        "away_goals_after_extra_time": None,
        "penalty_shootout": False,
        "home_penalty_goals": None,
        "away_penalty_goals": None,
        "competition": "FIFA World Cup",
        "stage": "GROUP_STAGE",
    }


def _finished_live_row(
    source_fixture_id: str,
    *,
    home_goals: int,
    away_goals: int,
    cutoff: datetime = datetime(2026, 6, 20, 10, tzinfo=UTC),
    extra_time_played: bool = False,
    home_goals_after_extra_time: int | None = None,
    away_goals_after_extra_time: int | None = None,
    penalty_shootout: bool = False,
    home_penalty_goals: int | None = None,
    away_penalty_goals: int | None = None,
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
        "result_90": "home_win"
        if home_goals > away_goals
        else "away_win"
        if away_goals > home_goals
        else "draw",
        "extra_time_played": extra_time_played,
        "home_goals_after_extra_time": home_goals_after_extra_time,
        "away_goals_after_extra_time": away_goals_after_extra_time,
        "penalty_shootout": penalty_shootout,
        "home_penalty_goals": home_penalty_goals,
        "away_penalty_goals": away_penalty_goals,
        "home_team_id": "c",
        "away_team_id": "d",
        "home_team_name_original": "Team C",
        "away_team_name_original": "Team D",
    }


def _prediction_history_row(
    *,
    prediction_id: str = "prediction-1",
    source_fixture_id: str = "future-known",
    created: datetime = datetime(2026, 6, 18, 12, tzinfo=UTC),
    cutoff: datetime = datetime(2026, 6, 18, 10, tzinfo=UTC),
    kickoff: datetime = datetime(2026, 6, 19, 18, tzinfo=UTC),
    probabilities: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, Any]:
    return {
        "schema_version": "world_cup_prediction_v1",
        "prediction_id": prediction_id,
        "prediction_run_id": "run-1",
        "source_fixture_id": source_fixture_id,
        "match_id": f"world_cup_2026_football_data:{source_fixture_id}",
        "source": "world_cup_2026_football_data",
        "prediction_created_at_utc": created,
        "data_cutoff_utc": cutoff,
        "kickoff_utc": kickoff,
        "hours_before_kickoff": (kickoff - created).total_seconds() / 3600,
        "home_team_id": "c",
        "away_team_id": "d",
        "home_team_name": "Team C",
        "away_team_name": "Team D",
        "home_elo_pre": 1500.0,
        "away_elo_pre": 1500.0,
        "expected_home_goals": 1.2,
        "expected_away_goals": 1.0,
        "probability_home_win": probabilities[0],
        "probability_draw": probabilities[1],
        "probability_away_win": probabilities[2],
        "modal_score": "1-0",
        "score_probabilities_json": "{}",
        "score_probability_mass": 1.0,
        "residual_probability": 0.0,
        "model_family": "poisson",
        "model_version": "poisson_goal_v1",
        "selected_config_json": "{\"model_type\":\"poisson\"}",
        "selected_config_checksum": "b" * 64,
        "dataset_revision": "operational_dataset_v1:test",
        "dataset_checksum": "c" * 64,
        "live_snapshot_checksum": "a" * 64,
        "prediction_context": "early_v1",
        "prediction_status": "prospective",
        "competition": "FIFA World Cup",
        "stage": "GROUP_STAGE",
        "training_matches": 3,
        "live_finished_2026_matches": 0,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)

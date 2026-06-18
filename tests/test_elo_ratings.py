from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.features.elo import (
    EloRatingsConfig,
    MarginOfVictoryConfig,
    RatingRegressionAfterInactivityConfig,
    load_elo_ratings_config,
    run_elo_ratings,
)


def test_elo_ratings_use_pre_match_ratings_without_result_leakage(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("first", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row("second", "2020-01-02", home_team_id="a", away_team_id="c"),
        ],
        home_advantage=0,
    )
    rows = _table_rows(result.report.match_ratings_path)

    assert rows[0]["home_elo_pre"] == 1500
    assert rows[0]["away_elo_pre"] == 1500
    assert rows[0]["home_elo_post"] > rows[0]["home_elo_pre"]
    assert rows[1]["home_elo_pre"] == rows[0]["home_elo_post"]


def test_elo_ratings_batch_same_date_before_updates(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("a-first", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row("a-second", "2020-01-01", home_team_id="a", away_team_id="c"),
        ],
        home_advantage=0,
    )
    rows = _table_rows(result.report.match_ratings_path)

    assert [row["match_id"] for row in rows] == ["a-first", "a-second"]
    assert rows[0]["home_elo_pre"] == 1500
    assert rows[1]["home_elo_pre"] == 1500
    assert rows[0]["home_elo_post"] == rows[1]["home_elo_post"]


def test_elo_ratings_apply_home_advantage_only_when_eligible(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row(
                "neutral",
                "2020-01-01",
                home_team_id="a",
                away_team_id="b",
                home_advantage_eligible=False,
                home_advantage_status="neutral",
            ),
            _match_row(
                "home",
                "2020-01-02",
                home_team_id="c",
                away_team_id="d",
                home_advantage_eligible=True,
                home_advantage_status="home_team",
            ),
            _match_row(
                "away",
                "2020-01-03",
                home_team_id="e",
                away_team_id="f",
                home_advantage_eligible=True,
                home_advantage_status="away_team",
            ),
        ],
    )
    rows = {row["match_id"]: row for row in _table_rows(result.report.match_ratings_path)}

    assert rows["neutral"]["elo_difference_pre"] == 0
    assert rows["neutral"]["home_expected_score"] == 0.5
    assert rows["home"]["elo_difference_pre"] == 75
    assert rows["home"]["home_expected_score"] > 0.5
    assert rows["away"]["elo_difference_pre"] == -75
    assert rows["away"]["home_expected_score"] < 0.5


def test_elo_ratings_can_regress_after_inactivity(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("first", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row("inactive", "2020-01-11", home_team_id="a", away_team_id="c"),
        ],
        home_advantage=0,
        rating_regression_after_inactivity=RatingRegressionAfterInactivityConfig(
            enabled=True,
            inactivity_days=5,
            regression_fraction=0.5,
        ),
    )
    rows = _table_rows(result.report.match_ratings_path)
    first_post = rows[0]["home_elo_post"]
    expected_regressed_rating = 1500 + (first_post - 1500) * 0.5

    assert rows[1]["home_elo_pre"] == expected_regressed_rating


def test_elo_ratings_margin_of_victory_can_scale_updates(tmp_path: Path) -> None:
    one_goal = _run_with_rows(
        tmp_path / "one",
        [_match_row("one-goal", "2020-01-01", home_goals_90=1, away_goals_90=0)],
        home_advantage=0,
        margin_of_victory=MarginOfVictoryConfig(enabled=True, goal_difference_weight=0.5),
    )
    three_goal = _run_with_rows(
        tmp_path / "three",
        [_match_row("three-goal", "2020-01-01", home_goals_90=3, away_goals_90=0)],
        home_advantage=0,
        margin_of_victory=MarginOfVictoryConfig(enabled=True, goal_difference_weight=0.5),
    )
    one_goal_row = _table_rows(one_goal.report.match_ratings_path)[0]
    three_goal_row = _table_rows(three_goal.report.match_ratings_path)[0]

    assert three_goal_row["rating_change"] > one_goal_row["rating_change"]


def test_elo_ratings_handle_draws(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [_match_row("draw", "2020-01-01", home_goals_90=1, away_goals_90=1)],
        home_advantage=0,
    )
    row = _table_rows(result.report.match_ratings_path)[0]

    assert row["home_expected_score"] == 0.5
    assert row["rating_change"] == 0
    assert row["home_elo_post"] == 1500
    assert row["away_elo_post"] == 1500


def test_elo_ratings_exclude_model_ineligible_matches(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("eligible", "2020-01-01", home_team_id="a", away_team_id="b"),
            _match_row(
                "excluded",
                "2020-01-02",
                home_team_id="a",
                away_team_id="c",
                model_eligible=False,
                exclusion_reason="team_status_not_allowed",
            ),
        ],
        home_advantage=0,
    )
    rows = _table_rows(result.report.match_ratings_path)

    assert [row["match_id"] for row in rows] == ["eligible"]
    assert result.report.processed_matches == 1
    assert result.report.excluded_matches == 1
    assert result.report.excluded_by_reason == {"team_status_not_allowed": 1}


def test_elo_ratings_are_idempotent(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        home_advantage=0,
    )
    _write_matches(
        config.input_matches_path,
        [
            _match_row("b", "2020-01-02", home_team_id="a", away_team_id="b"),
            _match_row("a", "2020-01-01", home_team_id="c", away_team_id="d"),
        ],
    )

    first = run_elo_ratings(config)
    first_match_bytes = first.report.match_ratings_path.read_bytes()
    first_current_bytes = first.report.current_ratings_path.read_bytes()
    second = run_elo_ratings(config)

    assert second.report == first.report
    assert second.report.match_ratings_path.read_bytes() == first_match_bytes
    assert second.report.current_ratings_path.read_bytes() == first_current_bytes


def test_model_elo_ratings_cli_writes_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path, home_advantage=0)
    _write_matches(config.input_matches_path, [_match_row("cli", "2020-01-01")])
    config_path = tmp_path / "model.yaml"
    _write_config(config, config_path)
    runner = CliRunner()

    result = runner.invoke(app, ["model", "elo-ratings", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Matches processed: 1" in result.stdout
    assert config.output_match_ratings_path.is_file()
    assert config.output_current_ratings_path.is_file()
    assert config.report_output.is_file()


def _run_with_rows(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    **config_updates: Any,
) -> Any:
    config = _config(tmp_path, **config_updates)
    _write_matches(config.input_matches_path, rows)
    return run_elo_ratings(config)


def _config(tmp_path: Path, **updates: Any) -> EloRatingsConfig:
    config = load_elo_ratings_config().model_copy(
        update={
            "input_matches_path": tmp_path / "modeling.parquet",
            "output_match_ratings_path": tmp_path / "elo_match_ratings.parquet",
            "output_current_ratings_path": tmp_path / "elo_current_ratings.parquet",
            "report_output": tmp_path / "elo_report.json",
            **updates,
        }
    )
    return EloRatingsConfig.model_validate(config.model_dump(mode="python"))


def _write_config(config: EloRatingsConfig, path: Path) -> None:
    payload = {"elo": config.model_dump(mode="json")}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_matches(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _table_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _match_row(
    match_id: str,
    match_date: str,
    *,
    home_team_id: str = "home",
    away_team_id: str = "away",
    home_goals_90: int = 1,
    away_goals_90: int = 0,
    competition_category: str = "friendly",
    model_eligible: bool = True,
    exclusion_reason: str | None = None,
    home_advantage_eligible: bool = True,
    home_advantage_status: str = "home_team",
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "match_date": date.fromisoformat(match_date),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals_90": home_goals_90,
        "away_goals_90": away_goals_90,
        "competition_category": competition_category,
        "model_eligible": model_eligible,
        "exclusion_reason": exclusion_reason,
        "home_advantage_eligible": home_advantage_eligible,
        "home_advantage_status": home_advantage_status,
    }

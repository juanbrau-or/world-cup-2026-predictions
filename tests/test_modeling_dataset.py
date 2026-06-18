from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.data.modeling_dataset import (
    ModelingDatasetConfig,
    NeutralMatchPolicy,
    load_modeling_dataset_config,
    run_modeling_dataset_preparation,
)


def test_modeling_dataset_applies_minimum_rating_date(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("m-1999", "1999-12-31", competition="Friendly"),
            _match_row("m-2000", "2000-01-01", competition="Friendly"),
        ],
    )
    rows = _table_rows(result.report.output_path)

    assert rows[0]["modeling_period"] == "pre_rating_history"
    assert rows[0]["model_eligible"] is False
    assert rows[0]["exclusion_reason"] == "before_rating_history_start"
    assert rows[1]["modeling_period"] == "rating_history"
    assert rows[1]["model_eligible"] is True


def test_modeling_dataset_excludes_special_teams_without_dropping_rows(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row(
                "special-team",
                "2020-01-01",
                home_team_id="basque_country",
                home_team_name_original="Basque Country",
                away_team_id="brazil",
                away_team_name_original="Brazil",
                competition="Friendly",
            )
        ],
    )
    rows = _table_rows(result.report.output_path)

    assert len(rows) == 1
    assert rows[0]["home_team_status"] == "special"
    assert rows[0]["home_team_type"] == "regional_representative"
    assert rows[0]["model_eligible"] is False
    assert rows[0]["exclusion_reason"] == "team_status_not_allowed"


def test_modeling_dataset_reports_unknown_competition_categories(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [_match_row("unknown-cup", "2020-01-01", competition="Mystery Cup")],
    )
    rows = _table_rows(result.report.output_path)

    assert rows[0]["competition_category"] is None
    assert rows[0]["model_eligible"] is False
    assert rows[0]["exclusion_reason"] == "unresolved_competition_category"
    assert result.report.unresolved_competitions == ("Mystery Cup",)


def test_modeling_dataset_home_advantage_flags_neutrality(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("home", "2020-01-01", neutral_site=False, home_advantage_status="home_team"),
            _match_row("neutral", "2020-01-02", neutral_site=True, home_advantage_status="neutral"),
        ],
    )
    rows = _table_rows(result.report.output_path)

    assert rows[0]["home_advantage_eligible"] is True
    assert rows[1]["home_advantage_eligible"] is False
    assert result.report.neutral_matches == 1


def test_modeling_dataset_can_exclude_neutral_matches_by_policy(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        neutral_match_policy=NeutralMatchPolicy.EXCLUDE,
    )
    _write_matches(
        config.input_matches_path,
        [_match_row("neutral", "2020-01-02", neutral_site=True, home_advantage_status="neutral")],
    )

    result = run_modeling_dataset_preparation(config)
    rows = _table_rows(result.report.output_path)

    assert rows[0]["model_eligible"] is False
    assert rows[0]["exclusion_reason"] == "neutral_site_excluded"


def test_modeling_dataset_output_order_is_deterministic(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("b", "2020-01-02", competition="Friendly"),
            _match_row("c", "2020-01-01", competition="Friendly"),
            _match_row("a", "2020-01-01", competition="Friendly"),
        ],
    )
    rows = _table_rows(result.report.output_path)

    assert [row["match_id"] for row in rows] == ["a", "c", "b"]
    assert [row["same_date_batch_id"] for row in rows] == [
        "2020-01-01",
        "2020-01-01",
        "2020-01-02",
    ]


def test_modeling_dataset_is_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = [
        _match_row("b", "2020-01-02", competition="FIFA World Cup qualification"),
        _match_row("a", "2020-01-01", competition="Friendly"),
    ]
    _write_matches(config.input_matches_path, rows)

    first = run_modeling_dataset_preparation(config)
    first_bytes = first.report.output_path.read_bytes()
    second = run_modeling_dataset_preparation(config)

    assert second.report == first.report
    assert second.report.output_path.read_bytes() == first_bytes


def test_modeling_dataset_exclusion_reasons_are_primary_and_reported(tmp_path: Path) -> None:
    result = _run_with_rows(
        tmp_path,
        [
            _match_row("unknown", "2020-01-01", competition="Mystery Cup"),
            _match_row("not-allowed", "2020-01-02", competition="CONIFA World Football Cup"),
            _match_row("incomplete", "2020-01-03", competition="Friendly", home_goals_90=None),
        ],
    )
    rows = _table_rows(result.report.output_path)

    assert [row["exclusion_reason"] for row in rows] == [
        "unresolved_competition_category",
        "competition_category_not_allowed",
        "incomplete_record",
    ]
    assert result.report.exclusions_by_reason == {
        "competition_category_not_allowed": 1,
        "incomplete_record": 1,
        "unresolved_competition_category": 1,
    }


def test_prepare_modeling_data_cli_writes_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_matches(config.input_matches_path, [_match_row("cli", "2020-01-01")])
    config_path = tmp_path / "modeling_data.yaml"
    _write_config(config, config_path)
    runner = CliRunner()

    result = runner.invoke(app, ["prepare", "modeling-data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Rows total: 1" in result.stdout
    assert "Rows eligible: 1" in result.stdout
    assert config.output_matches_path.is_file()
    assert config.report_output.is_file()


def _run_with_rows(
    tmp_path: Path,
    rows: list[dict[str, Any]],
) -> Any:
    config = _config(tmp_path)
    _write_matches(config.input_matches_path, rows)
    return run_modeling_dataset_preparation(config)


def _config(tmp_path: Path, **updates: Any) -> ModelingDatasetConfig:
    config = load_modeling_dataset_config().model_copy(
        update={
            "input_matches_path": tmp_path / "input.parquet",
            "output_matches_path": tmp_path / "modeling.parquet",
            "report_output": tmp_path / "report.json",
            **updates,
        }
    )
    return ModelingDatasetConfig.model_validate(config.model_dump(mode="python"))


def _write_config(config: ModelingDatasetConfig, path: Path) -> None:
    payload = {"modeling_data": config.model_dump(mode="json")}
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
    competition: str = "Friendly",
    home_team_id: str = "united_states",
    away_team_id: str = "brazil",
    home_team_name_original: str = "United States",
    away_team_name_original: str = "Brazil",
    neutral_site: bool | None = False,
    home_advantage_status: str = "home_team",
    home_goals_90: int | None = 1,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "schema_version": "international_match_v1",
        "match_status": "played",
        "match_date": date.fromisoformat(match_date),
        "kickoff_utc": None,
        "kickoff_local_time": None,
        "kickoff_timezone": None,
        "kickoff_time_status": "date_only",
        "home_team_name_original": home_team_name_original,
        "away_team_name_original": away_team_name_original,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_goals_90": home_goals_90,
        "away_goals_90": 0,
        "result_90": "home_win",
        "extra_time_played": False,
        "home_goals_after_extra_time": None,
        "away_goals_after_extra_time": None,
        "penalty_shootout": False,
        "home_penalty_goals": None,
        "away_penalty_goals": None,
        "competition": competition,
        "stage": None,
        "match_type": "friendly",
        "city": "Austin",
        "host_country": "United States",
        "venue_name_original": None,
        "neutral_site": neutral_site,
        "home_advantage_status": home_advantage_status,
        "source": "test",
        "source_match_id": match_id,
        "retrieved_at_utc": datetime(2026, 6, 18, tzinfo=UTC),
    }

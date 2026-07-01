from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]

from worldcup2026.pipelines.shadow_contextual import (
    ShadowContextualError,
    _assert_same_fixtures,
    run_evaluate_shadow_contextual,
)


def test_shadow_contextual_evaluation_reports_zero_observations(tmp_path: Path) -> None:
    config_path = _write_shadow_config(tmp_path)
    live_path = tmp_path / "live.parquet"
    predictions_root = tmp_path / "predictions"
    predictions_root.mkdir()
    _write_rows(live_path, [_scheduled_live_row()])
    _write_official_scorecard(predictions_root / "prospective_scorecard.json")

    result = run_evaluate_shadow_contextual(
        config_path=config_path,
        official_scorecard_path=predictions_root / "prospective_scorecard.json",
        shadow_history_root=predictions_root / "shadow" / "history",
        live_matches_path=live_path,
        predictions_root=predictions_root,
    )

    assert result.evaluable_predictions == 0
    assert result.paired_matches == 0
    assert result.ledger_path == predictions_root / "shadow" / "contextual_ledger.parquet"
    assert result.ledger_path.is_file()
    assert result.comparison_path.is_file()
    payload = json.loads(result.scorecard_json_path.read_text(encoding="utf-8"))
    assert payload["official_selection_policy"]["prediction_context"] == "shadow_contextual_v1"


def test_shadow_fixture_mismatch_is_detected() -> None:
    with pytest.raises(ShadowContextualError, match="do not match"):
        _assert_same_fixtures(
            [{"source_fixture_id": "fixture-a"}],
            [{"source_match_id": "fixture-b"}],
        )


def _write_shadow_config(tmp_path: Path) -> Path:
    path = tmp_path / "shadow.yaml"
    config = {
        "schema_version": "prospective_evaluation_config_v1",
        "result_metric_basis": "result_90",
        "minimum_calibration_matches": 30,
        "small_sample_warning_threshold": 30,
        "horizons": {
            "version": "shadow_test",
            "buckets": [
                {
                    "id": "gt_6h",
                    "label": "> 6h",
                    "min_hours": 6,
                    "min_inclusive": True,
                    "max_hours": None,
                    "max_inclusive": False,
                }
            ],
        },
        "official_selection": {
            "policy_id": "shadow_contextual_early_v1",
            "policy_version": "shadow_contextual_early_v1_test",
            "prediction_context": "shadow_contextual_v1",
            "primary_rule": {
                "id": "latest_valid_at_least_6h_before_kickoff",
                "min_hours_before_kickoff": 6,
            },
            "fallback_rule": {"id": "earliest_valid_before_kickoff"},
        },
        "baselines": {
            "uniform_1x2": {"enabled": True},
            "historical_frequency": {
                "enabled": False,
                "input_matches_path": str(tmp_path / "missing.parquet"),
                "cutoff_utc": "2026-01-01T00:00:00Z",
            },
            "elo_operational": {"enabled": False},
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _scheduled_live_row() -> dict[str, object]:
    kickoff = datetime(2026, 6, 20, 18, tzinfo=UTC)
    return {
        "match_id": "wc_fixture",
        "match_status": "scheduled",
        "match_date": kickoff.date(),
        "kickoff_utc": kickoff,
        "home_team_id": "canada",
        "away_team_id": "japan",
        "home_team_name_original": "Canada",
        "away_team_name_original": "Japan",
        "source": "world_cup_2026_football_data",
        "source_match_id": "fixture-a",
        "competition": "FIFA World Cup",
        "stage": "Group Stage",
        "data_cutoff_utc": datetime(2026, 6, 19, 10, tzinfo=UTC),
    }


def _write_official_scorecard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "prospective_scorecard_v1",
                "official_selection_policy": {"prediction_context": "early_v1"},
                "metrics": {"matches": 0},
                "matches": [],
            }
        ),
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)

from __future__ import annotations

import csv
import gzip
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from worldcup2026.pipelines.publication import (
    PublicationError,
    assert_allowed_publication_path,
    prepare_predictions_publication,
)

GENERATED_AT = datetime(2026, 6, 29, 20, 0, tzinfo=UTC)
CUTOFF = "2026-06-29T18:00:00Z"


def test_prepare_publication_writes_allowed_outputs_and_manifest(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    output_root = tmp_path / "branch"

    result = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )

    assert result.changed is True
    assert result.prediction_count == 1
    assert result.prospective_evaluation_observations == 2
    assert result.history_path is not None
    assert result.history_path.is_file()
    assert not (output_root / "latest.parquet").exists()
    published_files = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    )
    assert published_files == [
        "history/20260629T180000Z_98c5baae3cef.csv.gz",
        "latest.csv",
        "latest.json",
        "manifest.json",
        "prospective_evaluation.json",
        "prospective_evaluation.md",
        "upcoming.md",
    ]

    manifest = _json(output_root / "manifest.json")
    assert manifest["generated_at"] == "2026-06-29T20:00:00Z"
    assert manifest["data_cutoff"] == CUTOFF
    assert manifest["model"] == {"family": "poisson", "version": "poisson_goal_v1"}
    assert manifest["prediction_count"] == 1
    assert manifest["prospective_evaluation_observations"] == 2
    assert manifest["history_path"] == "history/20260629T180000Z_98c5baae3cef.csv.gz"
    assert manifest["checksum"] == result.checksum
    assert "latest.csv" in manifest["checksums"]


def test_latest_json_matches_latest_csv(tmp_path: Path) -> None:
    row = _prediction_row(source_fixture_id="fixture-2", home_team_name="Mexico")
    predictions_root = _write_predictions_root(tmp_path, rows=[row])
    output_root = tmp_path / "branch"

    prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )

    latest = _json(output_root / "latest.json")
    assert latest["schema_version"] == "predictions_latest_v1"
    assert latest["prediction_count"] == 1
    assert latest["predictions"][0]["source_fixture_id"] == "fixture-2"
    assert latest["predictions"][0]["home_team_name"] == "Mexico"


def test_history_is_immutable(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    output_root = tmp_path / "branch"
    first = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )
    assert first.history_path is not None
    first.history_path.write_bytes(b"corrupt")

    with pytest.raises(PublicationError, match="immutable history collision"):
        prepare_predictions_publication(
            predictions_root=predictions_root,
            output_root=output_root,
            generated_at=GENERATED_AT,
        )


def test_identical_content_is_not_republished(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    output_root = tmp_path / "branch"

    first = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )
    manifest_before = (output_root / "manifest.json").read_text(encoding="utf-8")
    second = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=datetime(2026, 6, 29, 21, 0, tzinfo=UTC),
    )

    assert first.changed is True
    assert second.changed is False
    assert (output_root / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert len(list((output_root / "history").glob("*.csv.gz"))) == 1


def test_prepare_publication_without_predictions_keeps_latest_and_manifest(
    tmp_path: Path,
) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[])
    output_root = tmp_path / "branch"

    result = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )

    assert result.prediction_count == 0
    assert result.history_path is None
    assert (output_root / "latest.csv").is_file()
    assert (output_root / "latest.json").is_file()
    assert not (output_root / "history").exists()
    latest = _json(output_root / "latest.json")
    assert latest["predictions"] == []
    assert latest["data_cutoff"] == CUTOFF


def test_empty_data_branch_is_created(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    output_root = tmp_path / "missing" / "branch"

    result = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=output_root,
        generated_at=GENERATED_AT,
    )

    assert result.manifest_path.is_file()
    assert output_root.is_dir()


def test_secret_values_are_rejected(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(
        tmp_path,
        rows=[_prediction_row(home_team_name="SECRET_VALUE_123456")],
    )

    with pytest.raises(PublicationError, match="secret value detected"):
        prepare_predictions_publication(
            predictions_root=predictions_root,
            output_root=tmp_path / "branch",
            generated_at=GENERATED_AT,
            secret_values=("SECRET_VALUE_123456",),
        )


def test_disallowed_existing_paths_are_rejected(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    output_root = tmp_path / "branch"
    (output_root / "data" / "raw").mkdir(parents=True)
    (output_root / "data" / "raw" / "provider.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PublicationError, match="raw snapshots and models"):
        prepare_predictions_publication(
            predictions_root=predictions_root,
            output_root=output_root,
            generated_at=GENERATED_AT,
        )


def test_allowed_path_validation_rejects_parquet_raw_and_large_models() -> None:
    assert_allowed_publication_path(Path("latest.csv"), size_bytes=100)
    assert_allowed_publication_path(Path("history/20260629T180000Z_abc123.csv.gz"), size_bytes=100)

    with pytest.raises(PublicationError, match="Parquet"):
        assert_allowed_publication_path(Path("latest.parquet"), size_bytes=100)
    with pytest.raises(PublicationError, match="raw snapshots and models"):
        assert_allowed_publication_path(Path("data/raw/snapshot.json"), size_bytes=100)
    with pytest.raises(PublicationError, match="too large"):
        assert_allowed_publication_path(Path("history/big.csv.gz"), size_bytes=2_000_001)
    with pytest.raises(PublicationError, match="raw snapshots and models"):
        assert_allowed_publication_path(Path("artifacts/models/model.json"), size_bytes=100)


def test_timestamps_must_be_utc(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(
        tmp_path,
        rows=[_prediction_row(data_cutoff_utc="2026-06-29T18:00:00-05:00")],
    )

    with pytest.raises(PublicationError, match="timezone-aware UTC"):
        prepare_predictions_publication(
            predictions_root=predictions_root,
            output_root=tmp_path / "branch",
            generated_at=GENERATED_AT,
        )

    with pytest.raises(PublicationError, match="timezone-aware UTC"):
        prepare_predictions_publication(
            predictions_root=_write_predictions_root(tmp_path / "other", rows=[_prediction_row()]),
            output_root=tmp_path / "branch-2",
            generated_at=datetime(2026, 6, 29, 20, 0),
        )


def test_compressed_history_contains_latest_csv(tmp_path: Path) -> None:
    predictions_root = _write_predictions_root(tmp_path, rows=[_prediction_row()])
    result = prepare_predictions_publication(
        predictions_root=predictions_root,
        output_root=tmp_path / "branch",
        generated_at=GENERATED_AT,
    )
    assert result.history_path is not None

    assert gzip.decompress(result.history_path.read_bytes()) == (
        predictions_root / "latest.csv"
    ).read_bytes()


def _write_predictions_root(tmp_path: Path, *, rows: list[dict[str, str]]) -> Path:
    root = tmp_path / "predictions"
    root.mkdir(parents=True)
    _write_latest_csv(root / "latest.csv", rows=rows)
    (root / "upcoming.md").write_text(_upcoming_report(rows=rows), encoding="utf-8")
    (root / "prospective_evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": "prospective_evaluation_v1",
                "saved_predictions_seen": 3,
                "metrics": {"predictions": 2},
                "evaluated_predictions": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "prospective_evaluation.md").write_text("# Prospective Evaluation\n", encoding="utf-8")
    return root


def _write_latest_csv(path: Path, *, rows: list[dict[str, str]]) -> None:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_fieldnames())
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _fieldnames() -> list[str]:
    return [
        "prediction_id",
        "source_fixture_id",
        "prediction_created_at_utc",
        "data_cutoff_utc",
        "kickoff_utc",
        "hours_before_kickoff",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_elo_pre",
        "away_elo_pre",
        "expected_home_goals",
        "expected_away_goals",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
        "modal_score",
        "model_family",
        "model_version",
        "dataset_revision",
        "live_snapshot_checksum",
        "prediction_context",
        "prediction_status",
    ]


def _prediction_row(
    *,
    source_fixture_id: str = "fixture-1",
    home_team_name: str = "Canada",
    data_cutoff_utc: str = CUTOFF,
) -> dict[str, str]:
    return {
        "prediction_id": "prediction-1",
        "source_fixture_id": source_fixture_id,
        "prediction_created_at_utc": "2026-06-29T19:00:00Z",
        "data_cutoff_utc": data_cutoff_utc,
        "kickoff_utc": "2026-06-30T18:00:00Z",
        "hours_before_kickoff": "23.0",
        "home_team_id": "canada",
        "away_team_id": "japan",
        "home_team_name": home_team_name,
        "away_team_name": "Japan",
        "home_elo_pre": "1800.0",
        "away_elo_pre": "1810.0",
        "expected_home_goals": "1.2",
        "expected_away_goals": "1.1",
        "probability_home_win": "0.400000",
        "probability_draw": "0.300000",
        "probability_away_win": "0.300000",
        "modal_score": "1-1",
        "model_family": "poisson",
        "model_version": "poisson_goal_v1",
        "dataset_revision": "operational_dataset_v1:abc",
        "live_snapshot_checksum": "a" * 64,
        "prediction_context": "early_v1",
        "prediction_status": "prospective",
    }


def _upcoming_report(*, rows: list[dict[str, str]]) -> str:
    if rows:
        return "# Upcoming World Cup 2026 Predictions\n"
    return (
        "# Upcoming World Cup 2026 Predictions\n\n"
        f"Data cutoff UTC: {CUTOFF}\n"
        "Model: poisson (poisson_goal_v1)\n"
    )


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload

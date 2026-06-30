from __future__ import annotations

import csv
import json
import tarfile
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from worldcup2026.pipelines.publication import prepare_predictions_publication


def test_operational_predictions_workflow_syntax_and_contract() -> None:
    payload = _workflow_payload()

    triggers = _mapping(payload["on"])
    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "0 */4 * * *"}]
    assert payload["concurrency"]["cancel-in-progress"] == "false"
    assert payload["permissions"] == {"contents": "read"}

    jobs = _mapping(payload["jobs"])
    run_pipeline = _mapping(jobs["run-pipeline"])
    publish_data = _mapping(jobs["publish-data"])
    assert "permissions" not in run_pipeline
    assert publish_data["permissions"] == {"contents": "write"}

    run_steps = _step_names(run_pipeline)
    assert "Run operational pipeline" in run_steps
    assert "Test publisher" in run_steps
    assert "Validate publisher output and secret scan" in run_steps
    assert "Upload Parquet artifacts" in run_steps
    assert "Upload logs" in run_steps
    assert "Upload reports" in run_steps
    assert "Upload manifests" in run_steps

    publish_steps = _step_names(publish_data)
    assert "Prepare predictions-data worktree" in publish_steps
    assert "Prepare branch-safe publication" in publish_steps
    assert "Commit and push data branch" in publish_steps


def test_operational_pipeline_rebuilds_clean_checkout_inputs_before_prediction() -> None:
    payload = _workflow_payload()
    run_pipeline = _mapping(_mapping(payload["jobs"])["run-pipeline"])
    script = _step_run(run_pipeline, "Run operational pipeline")
    expected_commands = [
        "uv run wc2026 doctor",
        "uv run wc2026 ingest historical",
        "uv run wc2026 prepare modeling-data",
        "uv run wc2026 model dixon-coles",
        "uv run wc2026 ingest world-cup",
        "uv run wc2026 predict upcoming",
        "uv run wc2026 evaluate prospective",
    ]

    positions = [script.index(command) for command in expected_commands]

    assert positions == sorted(positions)
    assert "|| true" not in script
    assert "data/processed/modeling_matches.parquet" not in script


def test_operational_workflow_uses_current_official_action_versions() -> None:
    payload = _workflow_payload()
    jobs = _mapping(payload["jobs"])
    run_pipeline = _mapping(jobs["run-pipeline"])
    publish_data = _mapping(jobs["publish-data"])

    assert _step_uses(run_pipeline, "Checkout main") == "actions/checkout@v7.0.0"
    assert _step_uses(run_pipeline, "Set up Python") == "actions/setup-python@v6.3.0"
    assert _step_uses(run_pipeline, "Install uv") == "astral-sh/setup-uv@v8.2.0"
    assert _step_uses(publish_data, "Checkout main") == "actions/checkout@v7.0.0"
    assert _step_uses(publish_data, "Set up Python") == "actions/setup-python@v6.3.0"
    assert _step_uses(publish_data, "Install uv") == "astral-sh/setup-uv@v8.2.0"
    assert _step_uses(publish_data, "Download publication payload") == (
        "actions/download-artifact@v8.0.1"
    )
    for step_name in (
        "Upload Parquet artifacts",
        "Upload logs",
        "Upload reports",
        "Upload manifests",
        "Upload audit models",
        "Upload branch publication payload",
    ):
        assert _step_uses(run_pipeline, step_name) == "actions/upload-artifact@v7.0.1"


def test_publication_payload_extracts_under_predictions_root(tmp_path: Path) -> None:
    payload = _workflow_payload()
    run_pipeline = _mapping(_mapping(payload["jobs"])["run-pipeline"])
    members = _publication_payload_members(_step_run(run_pipeline, "Package publication inputs"))
    source_repo = tmp_path / "run-pipeline"
    publish_repo = tmp_path / "publish-data" / "repo"
    tarball = tmp_path / "predictions-data-inputs.tgz"

    _write_publication_inputs(source_repo / "predictions")
    with tarfile.open(tarball, "w:gz") as archive:
        for member in members:
            archive.add(source_repo / member, arcname=member)

    publish_repo.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as archive:
        archive.extractall(publish_repo)

    result = prepare_predictions_publication(
        predictions_root=publish_repo / "predictions",
        output_root=tmp_path / "predictions-data",
        generated_at=datetime(2026, 6, 30, 12, tzinfo=UTC),
    )

    assert sorted(members) == [
        "predictions/latest.csv",
        "predictions/prospective_evaluation.json",
        "predictions/prospective_evaluation.md",
        "predictions/upcoming.md",
    ]
    assert (publish_repo / "predictions" / "latest.csv").is_file()
    assert result.manifest_path.is_file()
    assert result.latest_json_path.is_file()


def _workflow_payload() -> dict[str, Any]:
    path = Path(".github/workflows/operational-predictions.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _step_names(job: dict[str, Any]) -> set[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return {str(step["name"]) for step in steps if isinstance(step, dict) and "name" in step}


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    steps = job["steps"]
    assert isinstance(steps, list)
    for step in steps:
        if isinstance(step, dict) and step.get("name") == name:
            return step
    raise AssertionError(f"workflow step not found: {name}")


def _step_run(job: dict[str, Any], name: str) -> str:
    value = _step(job, name).get("run")
    assert isinstance(value, str)
    return value


def _step_uses(job: dict[str, Any], name: str) -> str:
    value = _step(job, name).get("uses")
    assert isinstance(value, str)
    return value


def _publication_payload_members(script: str) -> list[str]:
    members: list[str] = []
    for line in script.splitlines():
        item = line.strip().removesuffix(" \\")
        if item.startswith("predictions/"):
            members.append(item)
    return members


def _write_publication_inputs(predictions_root: Path) -> None:
    predictions_root.mkdir(parents=True)
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "prediction_id",
            "source_fixture_id",
            "data_cutoff_utc",
            "kickoff_utc",
            "probability_home_win",
            "probability_draw",
            "probability_away_win",
            "model_family",
            "model_version",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "prediction_id": "prediction-1",
            "source_fixture_id": "fixture-1",
            "data_cutoff_utc": "2026-06-30T10:00:00Z",
            "kickoff_utc": "2026-06-30T18:00:00Z",
            "probability_home_win": "0.400000",
            "probability_draw": "0.300000",
            "probability_away_win": "0.300000",
            "model_family": "poisson",
            "model_version": "poisson_goal_v1",
        }
    )
    (predictions_root / "latest.csv").write_text(buffer.getvalue(), encoding="utf-8")
    (predictions_root / "upcoming.md").write_text(
        "# Upcoming World Cup 2026 Predictions\n",
        encoding="utf-8",
    )
    (predictions_root / "prospective_evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": "prospective_evaluation_v1",
                "saved_predictions_seen": 0,
                "metrics": {"predictions": 0},
                "evaluated_predictions": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (predictions_root / "prospective_evaluation.md").write_text(
        "# Prospective Evaluation\n",
        encoding="utf-8",
    )

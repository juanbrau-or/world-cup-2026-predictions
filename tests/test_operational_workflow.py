from __future__ import annotations

import csv
import json
import tarfile
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from worldcup2026.pipelines.operational_summary import write_operational_step_summary
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
    assert "Write operational step summary" in run_steps
    assert "Package manifests" in run_steps
    assert "Upload Parquet artifacts" in run_steps
    assert "Upload logs" in run_steps
    assert "Upload reports" in run_steps
    assert "Upload manifests" in run_steps

    publish_steps = _step_names(publish_data)
    assert "Prepare predictions-data worktree" in publish_steps
    assert "Prepare branch-safe publication" in publish_steps
    assert "Commit and push data branch" in publish_steps


def test_predictions_data_worktree_uses_authenticated_remote() -> None:
    payload = _workflow_payload()
    publish_data = _mapping(_mapping(payload["jobs"])["publish-data"])
    step = _step(publish_data, "Prepare predictions-data worktree")
    script = _step_run(publish_data, "Prepare predictions-data worktree")
    env = _mapping(step["env"])

    assert env["GITHUB_TOKEN"] == "${{ github.token }}"
    assert "GITHUB_TOKEN is required for predictions-data push" in script
    assert "x-access-token:${GITHUB_TOKEN}" in script
    assert "${GITHUB_REPOSITORY}.git" in script
    assert "$GITHUB_SERVER_URL/$GITHUB_REPOSITORY.git" not in script


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
        "predictions/prospective_matches.csv",
        "predictions/prospective_scorecard.json",
        "predictions/prospective_scorecard.md",
        "predictions/upcoming.md",
    ]
    assert (publish_repo / "predictions" / "latest.csv").is_file()
    assert result.manifest_path.is_file()
    assert result.latest_json_path.is_file()


def test_manifest_artifact_is_packaged_with_portable_upload_path() -> None:
    payload = _workflow_payload()
    run_pipeline = _mapping(_mapping(payload["jobs"])["run-pipeline"])
    package_script = _step_run(run_pipeline, "Package manifests")
    upload_step = _step(run_pipeline, "Upload manifests")
    upload_with = _mapping(upload_step["with"])

    assert "dist/operational-manifests.tgz" in package_script
    assert "find data/raw dist/predictions-data" in package_script
    assert "--null -czf dist/operational-manifests.tgz --files-from -" in package_script
    assert upload_with["path"] == "dist/operational-manifests.tgz"
    assert upload_with["if-no-files-found"] == "error"


def test_operational_step_summary_reports_core_monitoring_fields(tmp_path: Path) -> None:
    predictions_root = tmp_path / "predictions"
    interim_root = tmp_path / "interim"
    publication_root = tmp_path / "publication"
    logs_root = tmp_path / "logs"
    interim_root.mkdir()
    publication_root.mkdir()
    logs_root.mkdir()
    _write_publication_inputs(predictions_root)
    (interim_root / "world_cup_2026_ingest_report.json").write_text(
        json.dumps(
            {
                "provider_fixtures_received": 104,
                "fixtures_with_tbd_participants": 13,
                "pending_fixtures": 13,
                "freshness": {"next_kickoff_utc": "2026-06-30T17:00:00Z"},
                "validation_discrepancies": [
                    {"kind": "secondary_provider_unavailable", "detail": "timeout"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (publication_root / "manifest.json").write_text("{}", encoding="utf-8")
    summary_path = tmp_path / "summary.md"

    result = write_operational_step_summary(
        summary_path=summary_path,
        predictions_root=predictions_root,
        interim_root=interim_root,
        publication_root=publication_root,
        logs_root=logs_root,
    )

    text = summary_path.read_text(encoding="utf-8")
    assert result.predictable_fixtures == 1
    assert result.official_selected == 1
    assert result.official_evaluated == 0
    assert "| Fixtures received | 104 |" in text
    assert "| Fixtures TBD | 13 |" in text
    assert "| Predictable fixtures | 1 |" in text
    assert "| Official predictions selected | 1 |" in text
    assert "| Official matches evaluable | 0 |" in text
    assert "Secondary provider unavailable" in text


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
    (predictions_root / "prospective_scorecard.json").write_text(
        json.dumps(
            {
                "schema_version": "prospective_scorecard_v1",
                "results_cutoff_utc": None,
                "ledger": {
                    "schema_version": "prediction_ledger_v1",
                    "predictions": 1,
                    "unique_fixtures": 1,
                    "snapshots": 1,
                    "validity_counts": {"valid": 1},
                    "invalidity_counts": {},
                },
                "official_selection_policy": {
                    "policy_id": "early_v1",
                    "policy_version": "early_v1_2026_06_30",
                    "prediction_context": "early_v1",
                },
                "official_predictions_selected": 1,
                "official_predictions_evaluated": 0,
                "metrics": {"matches": 0},
                "matches": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (predictions_root / "prospective_scorecard.md").write_text(
        "# Prospective Scorecard\n",
        encoding="utf-8",
    )
    (predictions_root / "prospective_matches.csv").write_text(
        "source_fixture_id,prediction_id\n",
        encoding="utf-8",
    )

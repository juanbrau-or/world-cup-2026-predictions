from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def test_operational_predictions_workflow_syntax_and_contract() -> None:
    path = Path(".github/workflows/operational-predictions.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)

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


def _mapping(value: Any) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


def _step_names(job: dict[str, Any]) -> set[str]:
    steps = job["steps"]
    assert isinstance(steps, list)
    return {str(step["name"]) for step in steps if isinstance(step, dict) and "name" in step}

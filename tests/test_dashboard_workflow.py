from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def test_dashboard_workflow_triggers_permissions_and_concurrency() -> None:
    payload = _workflow_payload()

    triggers = _mapping(payload["on"])
    assert "workflow_dispatch" in triggers
    assert triggers["workflow_run"]["workflows"] == ["Operational Predictions"]
    assert triggers["workflow_run"]["types"] == ["completed"]
    assert triggers["schedule"] == [{"cron": "30 */12 * * *"}]
    assert payload["permissions"] == {"contents": "read"}
    assert payload["concurrency"] == {
        "group": "world-cup-2026-static-dashboard",
        "cancel-in-progress": "true",
    }

    deploy = _mapping(_mapping(payload["jobs"])["deploy"])
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"]["name"] == "github-pages"


def test_dashboard_workflow_checks_operational_success_and_fetches_data_branch() -> None:
    build = _mapping(_mapping(_workflow_payload()["jobs"])["build"])

    check_script = _step_run(build, "Check Operational Predictions success")
    assert "github.event.workflow_run.conclusion" in check_script
    assert "operational-predictions.yml" in check_script
    assert '--repo "$GITHUB_REPOSITORY"' in check_script
    assert "exit 1" in check_script
    assert '!= "success"' in check_script

    fetch_script = _step_run(build, "Fetch predictions-data")
    assert "git fetch --depth=1 origin predictions-data" in fetch_script
    assert "git archive origin/predictions-data" in fetch_script
    assert "test -f predictions-data/manifest.json" in fetch_script

    build_script = _step_run(build, "Validate and build dashboard")
    assert "uv run wc2026 site build --data-root predictions-data --output-root site-dist" in (
        build_script
    )


def test_dashboard_workflow_uses_official_pages_actions() -> None:
    payload = _workflow_payload()
    build = _mapping(_mapping(payload["jobs"])["build"])
    deploy = _mapping(_mapping(payload["jobs"])["deploy"])

    assert _step_uses(build, "Checkout main") == "actions/checkout@v7.0.0"
    assert _step_uses(build, "Set up Python") == "actions/setup-python@v6.3.0"
    assert _step_uses(build, "Install uv") == "astral-sh/setup-uv@v8.2.0"
    assert _step_uses(build, "Configure GitHub Pages") == "actions/configure-pages@v6"
    assert _step_uses(build, "Upload Pages artifact") == "actions/upload-pages-artifact@v5"
    assert _step_uses(deploy, "Deploy Pages artifact") == "actions/deploy-pages@v5"

    upload = _step(build, "Upload Pages artifact")
    assert upload["with"]["path"] == "site-dist"
    summary_script = _step_run(deploy, "Write deployment summary")
    assert "GITHUB_STEP_SUMMARY" in summary_script
    assert "steps.deployment.outputs.page_url" in summary_script


def _workflow_payload() -> dict[str, Any]:
    path = Path(".github/workflows/deploy-dashboard.yml")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    assert isinstance(value, dict)
    return value


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

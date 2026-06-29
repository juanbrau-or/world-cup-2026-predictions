"""Prospective evaluation for saved operational World Cup predictions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EPSILON = 1e-15
HOME_CLASS = 0
DRAW_CLASS = 1
AWAY_CLASS = 2
CLASS_LABELS = (HOME_CLASS, DRAW_CLASS, AWAY_CLASS)


class ProspectiveEvaluationError(RuntimeError):
    """Raised when saved prospective predictions cannot be evaluated."""


@dataclass(frozen=True)
class ProspectiveEvaluationResult:
    """Summary returned by the prospective evaluation pipeline."""

    evaluable_predictions: int
    log_loss: float | None
    brier_score: float | None
    ranked_probability_score: float | None
    accuracy: float | None
    kickoff_range: tuple[str | None, str | None]
    average_hours_before_kickoff: float | None
    report_path: Path
    json_path: Path


def run_prospective_evaluation(
    *,
    predictions_history_root: Path = Path("predictions/history"),
    live_matches_path: Path = Path("data/processed/world_cup_2026/matches.parquet"),
    report_path: Path = Path("predictions/prospective_evaluation.md"),
    json_path: Path = Path("predictions/prospective_evaluation.json"),
) -> ProspectiveEvaluationResult:
    """Evaluate previously saved prospective predictions for matches now finished."""

    prediction_rows = _read_prediction_history(predictions_history_root)
    live_rows = _read_live_matches(live_matches_path)
    evaluation_rows = _evaluable_rows(prediction_rows, live_rows)
    metrics = _metrics(evaluation_rows) if evaluation_rows else _empty_metrics()
    _write_outputs(
        metrics,
        evaluation_rows=evaluation_rows,
        report_path=report_path,
        json_path=json_path,
        prediction_rows=prediction_rows,
    )
    return ProspectiveEvaluationResult(
        evaluable_predictions=int(metrics["predictions"]),
        log_loss=_optional_float(metrics["log_loss"]),
        brier_score=_optional_float(metrics["brier_score"]),
        ranked_probability_score=_optional_float(metrics["ranked_probability_score"]),
        accuracy=_optional_float(metrics["accuracy"]),
        kickoff_range=(metrics["kickoff_start"], metrics["kickoff_end"]),
        average_hours_before_kickoff=_optional_float(metrics["average_hours_before_kickoff"]),
        report_path=report_path,
        json_path=json_path,
    )


def _read_prediction_history(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.parquet")):
        try:
            table = pq.read_table(path)  # type: ignore[no-untyped-call]
        except (OSError, pa.ArrowInvalid) as exc:
            msg = f"failed to read prediction history {path}: {exc}"
            raise ProspectiveEvaluationError(msg) from exc
        for row in table.to_pylist():
            item = dict(row)
            item["_history_path"] = str(path)
            rows.append(item)
    return rows


def _read_live_matches(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read live matches {path}: {exc}"
        raise ProspectiveEvaluationError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _evaluable_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    live_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    live_by_fixture = {
        str(row["source_match_id"]): row
        for row in live_rows
        if row.get("source_match_id") is not None
    }
    output: list[dict[str, Any]] = []
    seen_versions: set[tuple[str, str, str]] = set()
    for prediction in prediction_rows:
        if prediction.get("prediction_status") != "prospective":
            continue
        fixture_id = str(prediction.get("source_fixture_id") or "")
        live = live_by_fixture.get(fixture_id)
        if live is None or live.get("match_status") != "played":
            continue
        if not _has_result(live):
            continue
        version_key = (
            str(prediction.get("prediction_run_id") or ""),
            str(prediction.get("prediction_id") or ""),
            fixture_id,
        )
        if version_key in seen_versions:
            continue
        seen_versions.add(version_key)
        row = dict(prediction)
        row["actual_result"] = _actual_result(live)
        row["actual_home_goals"] = int(live["home_goals_90"])
        row["actual_away_goals"] = int(live["away_goals_90"])
        output.append(row)
    output.sort(
        key=lambda row: (
            _datetime_or_string(row.get("kickoff_utc")),
            str(row.get("prediction_created_at_utc")),
            str(row.get("source_fixture_id")),
        )
    )
    return output


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = np.asarray(
        [
            [
                row["probability_home_win"],
                row["probability_draw"],
                row["probability_away_win"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    targets = np.asarray([_class_from_name(str(row["actual_result"])) for row in rows])
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    one_hot = np.eye(len(CLASS_LABELS))[targets]
    kickoffs = [_datetime_or_string(row.get("kickoff_utc")) for row in rows]
    return {
        "evaluation_sets": {
            "out_of_fold": {"evaluated_by_this_command": False, "predictions": 0},
            "holdout_retrospective": {"evaluated_by_this_command": False, "predictions": 0},
            "prospective_real": {"evaluated_by_this_command": True, "predictions": len(rows)},
            "fixtures_offline": {"evaluated_by_this_command": False, "predictions": 0},
        },
        "predictions": len(rows),
        "log_loss": float(-np.mean(np.log(np.clip(true_probabilities, EPSILON, 1.0)))),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ranked_probability_score": _ranked_probability_score(probabilities, targets),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
        "kickoff_start": min(kickoffs) if kickoffs else None,
        "kickoff_end": max(kickoffs) if kickoffs else None,
        "average_hours_before_kickoff": float(
            np.mean([float(row["hours_before_kickoff"]) for row in rows])
        ),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "evaluation_sets": {
            "out_of_fold": {"evaluated_by_this_command": False, "predictions": 0},
            "holdout_retrospective": {"evaluated_by_this_command": False, "predictions": 0},
            "prospective_real": {"evaluated_by_this_command": True, "predictions": 0},
            "fixtures_offline": {"evaluated_by_this_command": False, "predictions": 0},
        },
        "predictions": 0,
        "log_loss": None,
        "brier_score": None,
        "ranked_probability_score": None,
        "accuracy": None,
        "kickoff_start": None,
        "kickoff_end": None,
        "average_hours_before_kickoff": None,
    }


def _ranked_probability_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    order = [AWAY_CLASS, DRAW_CLASS, HOME_CLASS]
    ordered_probabilities = probabilities[:, order]
    mapping = {AWAY_CLASS: 0, DRAW_CLASS: 1, HOME_CLASS: 2}
    ordered_targets = np.asarray([mapping[int(value)] for value in targets], dtype=int)
    true = np.eye(len(CLASS_LABELS))[ordered_targets]
    return float(
        np.mean(
            np.sum(
                np.square(np.cumsum(ordered_probabilities, axis=1) - np.cumsum(true, axis=1)),
                axis=1,
            )
            / 2
        )
    )


def _write_outputs(
    metrics: Mapping[str, Any],
    *,
    evaluation_rows: Sequence[Mapping[str, Any]],
    report_path: Path,
    json_path: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": "prospective_evaluation_v1",
        "saved_predictions_seen": len(prediction_rows),
        "metrics": dict(metrics),
        "evaluated_predictions": [
            {
                "prediction_id": row.get("prediction_id"),
                "prediction_run_id": row.get("prediction_run_id"),
                "source_fixture_id": row.get("source_fixture_id"),
                "kickoff_utc": _json_value(row.get("kickoff_utc")),
                "actual_result": row.get("actual_result"),
            }
            for row in evaluation_rows
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = _markdown_report(metrics, saved_predictions=len(prediction_rows))
    report_path.write_text(report, encoding="utf-8")


def _markdown_report(metrics: Mapping[str, Any], *, saved_predictions: int) -> str:
    lines = [
        "# Prospective Evaluation",
        "",
        f"Saved predictions seen: {saved_predictions}",
        f"Evaluable prospective predictions: {metrics['predictions']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| log loss | {_metric_value(metrics['log_loss'])} |",
        f"| Brier score | {_metric_value(metrics['brier_score'])} |",
        f"| RPS | {_metric_value(metrics['ranked_probability_score'])} |",
        f"| accuracy | {_metric_value(metrics['accuracy'])} |",
        "| average hours before kickoff | "
        f"{_metric_value(metrics['average_hours_before_kickoff'])} |",
        "",
        "## Evaluation Sets",
        "",
        "| Set | Evaluated here | Predictions |",
        "| --- | --- | ---: |",
    ]
    evaluation_sets = metrics["evaluation_sets"]
    if isinstance(evaluation_sets, Mapping):
        for name, values in evaluation_sets.items():
            if isinstance(values, Mapping):
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(name),
                            str(values.get("evaluated_by_this_command")),
                            str(values.get("predictions")),
                        )
                    )
                    + " |"
                )
    lines.append("")
    return "\n".join(lines)


def _metric_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _actual_result(row: Mapping[str, Any]) -> str:
    home = int(row["home_goals_90"])
    away = int(row["away_goals_90"])
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def _class_from_name(name: str) -> int:
    if name == "home_win":
        return HOME_CLASS
    if name == "draw":
        return DRAW_CLASS
    if name == "away_win":
        return AWAY_CLASS
    msg = f"unknown result class: {name}"
    raise ProspectiveEvaluationError(msg)


def _has_result(row: Mapping[str, Any]) -> bool:
    return _is_int(row.get("home_goals_90")) and _is_int(row.get("away_goals_90"))


def _is_int(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool)


def _datetime_or_string(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, float):
        return value
    return None

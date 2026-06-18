"""Elo rating engine for pre-match football ratings."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class MarginOfVictoryConfig(BaseModel):
    """Optional goal-margin multiplier for Elo updates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    goal_difference_weight: float = 0.0

    @model_validator(mode="after")
    def validate_weight(self) -> Self:
        """Require non-negative margin scaling."""

        if self.goal_difference_weight < 0:
            msg = "goal_difference_weight must be non-negative"
            raise ValueError(msg)
        return self


class RatingRegressionAfterInactivityConfig(BaseModel):
    """Optional regression toward the initial rating after long inactivity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    inactivity_days: int = 365
    regression_fraction: float = 0.0

    @model_validator(mode="after")
    def validate_regression(self) -> Self:
        """Require a usable inactivity window and regression fraction."""

        if self.inactivity_days < 1:
            msg = "inactivity_days must be at least 1"
            raise ValueError(msg)
        if not 0 <= self.regression_fraction <= 1:
            msg = "regression_fraction must be between 0 and 1"
            raise ValueError(msg)
        return self


class EloRatingsConfig(BaseModel):
    """Declarative configuration for the Elo rating engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_rating: float = 1500.0
    k_base: float = 20.0
    home_advantage: float = 75.0
    competition_importance: Mapping[str, float] = Field(default_factory=dict)
    margin_of_victory: MarginOfVictoryConfig = Field(default_factory=MarginOfVictoryConfig)
    rating_regression_after_inactivity: RatingRegressionAfterInactivityConfig = Field(
        default_factory=RatingRegressionAfterInactivityConfig
    )
    model_version: str = "elo_v1"
    input_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    output_match_ratings_path: Path = Path("data/processed/elo_match_ratings.parquet")
    output_current_ratings_path: Path = Path("data/processed/elo_current_ratings.parquet")
    report_output: Path = Path("data/interim/elo_ratings_report.json")

    @model_validator(mode="after")
    def validate_parameters(self) -> Self:
        """Require finite positive Elo parameters and complete importance weights."""

        if not math.isfinite(self.initial_rating):
            msg = "initial_rating must be finite"
            raise ValueError(msg)
        if not math.isfinite(self.k_base) or self.k_base <= 0:
            msg = "k_base must be a positive finite number"
            raise ValueError(msg)
        if not math.isfinite(self.home_advantage):
            msg = "home_advantage must be finite"
            raise ValueError(msg)
        if not self.model_version.strip():
            msg = "model_version cannot be blank"
            raise ValueError(msg)
        invalid_weights = [
            key
            for key, value in self.competition_importance.items()
            if not math.isfinite(value) or value < 0
        ]
        if invalid_weights:
            msg = "competition_importance weights must be finite and non-negative: " + ", ".join(
                sorted(invalid_weights)
            )
            raise ValueError(msg)
        return self


class EloRatingsReport(BaseModel):
    """Quality report emitted by the Elo rating stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: Path
    match_ratings_path: Path
    current_ratings_path: Path
    model_version: str
    total_rows: int
    processed_matches: int
    excluded_matches: int
    excluded_by_reason: Mapping[str, int]
    date_range: Mapping[str, str | None]
    teams_rated: int
    top_ratings: tuple[Mapping[str, Any], ...]


class EloRatingsResult(BaseModel):
    """Return value for programmatic Elo rating runs."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    match_ratings_table: Any
    current_ratings_table: Any
    report: EloRatingsReport


class EloRatingsError(RuntimeError):
    """Raised when Elo ratings cannot be produced."""


def load_elo_ratings_config(config_path: Path = Path("configs/model.yaml")) -> EloRatingsConfig:
    """Load Elo rating configuration from the model config file."""

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read Elo config {config_path}: {exc}"
        raise EloRatingsError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse Elo config {config_path}: {exc}"
        raise EloRatingsError(msg) from exc
    if not isinstance(config, dict):
        msg = f"Elo config {config_path} must contain a YAML mapping"
        raise EloRatingsError(msg)
    section = config.get("elo")
    if not isinstance(section, dict):
        msg = f"{config_path} is missing the elo section"
        raise EloRatingsError(msg)
    try:
        return EloRatingsConfig.model_validate(section)
    except ValidationError as exc:
        msg = f"Elo config {config_path} is invalid: {exc}"
        raise EloRatingsError(msg) from exc


def run_elo_ratings(
    config: EloRatingsConfig,
    *,
    input_path: Path | None = None,
    output_match_ratings_path: Path | None = None,
    output_current_ratings_path: Path | None = None,
    report_path: Path | None = None,
) -> EloRatingsResult:
    """Build pre-match Elo ratings and write match/current rating tables."""

    effective_input_path = input_path or config.input_matches_path
    effective_match_path = output_match_ratings_path or config.output_match_ratings_path
    effective_current_path = output_current_ratings_path or config.output_current_ratings_path
    effective_report_path = report_path or config.report_output

    rows = _read_modeling_matches(effective_input_path)
    eligible_rows = [row for row in rows if row.get("model_eligible") is True]
    match_rating_rows, current_rating_rows = _rate_matches(eligible_rows, config=config)

    match_ratings_table = pa.Table.from_pylist(match_rating_rows, schema=_match_ratings_schema())
    current_ratings_table = pa.Table.from_pylist(
        current_rating_rows,
        schema=_current_ratings_schema(),
    )
    effective_match_path.parent.mkdir(parents=True, exist_ok=True)
    effective_current_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(match_ratings_table, effective_match_path)  # type: ignore[no-untyped-call]
    pq.write_table(current_ratings_table, effective_current_path)  # type: ignore[no-untyped-call]

    report = _build_report(
        rows,
        match_rating_rows,
        current_rating_rows,
        input_path=effective_input_path,
        match_ratings_path=effective_match_path,
        current_ratings_path=effective_current_path,
        model_version=config.model_version,
    )
    _write_report(report, effective_report_path)
    return EloRatingsResult(
        match_ratings_table=match_ratings_table,
        current_ratings_table=current_ratings_table,
        report=report,
    )


def _read_modeling_matches(path: Path) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read modeling matches parquet {path}: {exc}"
        raise EloRatingsError(msg) from exc
    return [dict(row) for row in table.to_pylist()]


def _rate_matches(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: EloRatingsConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ratings: defaultdict[str, float] = defaultdict(lambda: config.initial_rating)
    matches_played: Counter[str] = Counter()
    last_match_date: dict[str, date] = {}
    output_rows: list[dict[str, Any]] = []

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _require_date(row["match_date"]),
            str(row.get("match_id") or ""),
        ),
    )
    grouped_rows: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted_rows:
        grouped_rows[_require_date(row["match_date"])].append(row)

    for match_date in sorted(grouped_rows):
        batch = sorted(grouped_rows[match_date], key=lambda row: str(row.get("match_id") or ""))
        _apply_inactivity_regression(
            batch,
            match_date=match_date,
            ratings=ratings,
            last_match_date=last_match_date,
            config=config,
        )

        pending_updates: defaultdict[str, float] = defaultdict(float)
        batch_outputs: list[dict[str, Any]] = []
        for row in batch:
            home_team_id = _require_str(row, "home_team_id")
            away_team_id = _require_str(row, "away_team_id")
            home_elo_pre = ratings[home_team_id]
            away_elo_pre = ratings[away_team_id]
            home_advantage = _home_advantage_adjustment(row, config.home_advantage)
            elo_difference_pre = home_elo_pre + home_advantage - away_elo_pre
            home_expected_score = _expected_score(elo_difference_pre)
            actual_home_score = _actual_home_score(row)
            k_factor = config.k_base * _competition_importance(row, config)
            margin_multiplier = _margin_of_victory_multiplier(row, config.margin_of_victory)
            home_delta = k_factor * margin_multiplier * (actual_home_score - home_expected_score)

            pending_updates[home_team_id] += home_delta
            pending_updates[away_team_id] -= home_delta
            batch_outputs.append(
                {
                    "match_id": _require_str(row, "match_id"),
                    "match_date": match_date,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_elo_pre": home_elo_pre,
                    "away_elo_pre": away_elo_pre,
                    "elo_difference_pre": elo_difference_pre,
                    "home_expected_score": home_expected_score,
                    "home_elo_post": None,
                    "away_elo_post": None,
                    "rating_change": home_delta,
                    "model_version": config.model_version,
                }
            )

        for team_id, rating_delta in pending_updates.items():
            ratings[team_id] += rating_delta
            matches_played[team_id] += sum(
                1
                for row in batch
                if row.get("home_team_id") == team_id or row.get("away_team_id") == team_id
            )
            last_match_date[team_id] = match_date

        for row in batch_outputs:
            row["home_elo_post"] = ratings[str(row["home_team_id"])]
            row["away_elo_post"] = ratings[str(row["away_team_id"])]
        output_rows.extend(batch_outputs)

    current_rating_rows = [
        {
            "canonical_team_id": team_id,
            "elo_rating": rating,
            "matches_processed": matches_played[team_id],
            "last_match_date": last_match_date.get(team_id),
            "model_version": config.model_version,
        }
        for team_id, rating in sorted(ratings.items())
        if matches_played[team_id] > 0
    ]
    current_rating_rows.sort(key=_current_rating_sort_key)
    return output_rows, current_rating_rows


def _current_rating_sort_key(row: Mapping[str, Any]) -> tuple[float, str]:
    return (-float(row["elo_rating"]), str(row["canonical_team_id"]))


def _apply_inactivity_regression(
    batch: Iterable[Mapping[str, Any]],
    *,
    match_date: date,
    ratings: defaultdict[str, float],
    last_match_date: Mapping[str, date],
    config: EloRatingsConfig,
) -> None:
    regression = config.rating_regression_after_inactivity
    if not regression.enabled or regression.regression_fraction == 0:
        return

    team_ids = {
        _require_str(row, field_name)
        for row in batch
        for field_name in ("home_team_id", "away_team_id")
    }
    for team_id in team_ids:
        previous_date = last_match_date.get(team_id)
        if previous_date is None:
            continue
        inactive_days = (match_date - previous_date).days
        if inactive_days >= regression.inactivity_days:
            distance_from_initial = ratings[team_id] - config.initial_rating
            ratings[team_id] = (
                config.initial_rating
                + distance_from_initial * (1 - regression.regression_fraction)
            )


def _home_advantage_adjustment(row: Mapping[str, Any], home_advantage: float) -> float:
    if row.get("home_advantage_eligible") is not True:
        return 0.0
    status = str(row.get("home_advantage_status") or "")
    if status == "home_team":
        return home_advantage
    if status == "away_team":
        return -home_advantage
    return 0.0


def _expected_score(elo_difference: float) -> float:
    return 1 / (1 + 10 ** (-elo_difference / 400))


def _actual_home_score(row: Mapping[str, Any]) -> float:
    home_goals = _require_int(row, "home_goals_90")
    away_goals = _require_int(row, "away_goals_90")
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _competition_importance(row: Mapping[str, Any], config: EloRatingsConfig) -> float:
    category = row.get("competition_category")
    if category is None:
        msg = f"eligible match {row.get('match_id')!r} is missing competition_category"
        raise EloRatingsError(msg)
    try:
        return config.competition_importance[str(category)]
    except KeyError as exc:
        msg = f"Elo config is missing competition_importance for {category!r}"
        raise EloRatingsError(msg) from exc


def _margin_of_victory_multiplier(
    row: Mapping[str, Any],
    config: MarginOfVictoryConfig,
) -> float:
    if not config.enabled or config.goal_difference_weight == 0:
        return 1.0
    goal_difference = abs(_require_int(row, "home_goals_90") - _require_int(row, "away_goals_90"))
    if goal_difference <= 1:
        return 1.0
    return 1 + (goal_difference - 1) * config.goal_difference_weight


def _build_report(
    input_rows: Iterable[Mapping[str, Any]],
    match_rating_rows: Iterable[Mapping[str, Any]],
    current_rating_rows: Iterable[Mapping[str, Any]],
    *,
    input_path: Path,
    match_ratings_path: Path,
    current_ratings_path: Path,
    model_version: str,
) -> EloRatingsReport:
    input_row_list = list(input_rows)
    match_row_list = list(match_rating_rows)
    current_row_list = list(current_rating_rows)
    exclusion_counts = Counter(
        str(row.get("exclusion_reason") or "model_eligible_false")
        for row in input_row_list
        if row.get("model_eligible") is not True
    )
    match_dates = [_require_date(row["match_date"]) for row in match_row_list]
    top_ratings = tuple(
        {
            "canonical_team_id": row["canonical_team_id"],
            "elo_rating": round(float(row["elo_rating"]), 3),
            "matches_processed": row["matches_processed"],
            "last_match_date": (
                row["last_match_date"].isoformat()
                if isinstance(row["last_match_date"], date)
                else None
            ),
        }
        for row in current_row_list[:10]
    )
    return EloRatingsReport(
        input_path=input_path,
        match_ratings_path=match_ratings_path,
        current_ratings_path=current_ratings_path,
        model_version=model_version,
        total_rows=len(input_row_list),
        processed_matches=len(match_row_list),
        excluded_matches=len(input_row_list) - len(match_row_list),
        excluded_by_reason=dict(sorted(exclusion_counts.items())),
        date_range={
            "min": min(match_dates).isoformat() if match_dates else None,
            "max": max(match_dates).isoformat() if match_dates else None,
        },
        teams_rated=len(current_row_list),
        top_ratings=top_ratings,
    )


def _write_report(report: EloRatingsReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected date value, got {value!r}"
    raise EloRatingsError(msg)


def _require_str(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if isinstance(value, str) and value:
        return value
    msg = f"required field {field_name} is missing or blank for match {row.get('match_id')!r}"
    raise EloRatingsError(msg)


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, int):
        return value
    msg = f"required integer field {field_name} is missing for match {row.get('match_id')!r}"
    raise EloRatingsError(msg)


def _match_ratings_schema() -> pa.Schema:
    return pa.schema(
        [
            ("match_id", pa.string()),
            ("match_date", pa.date32()),
            ("home_team_id", pa.string()),
            ("away_team_id", pa.string()),
            ("home_elo_pre", pa.float64()),
            ("away_elo_pre", pa.float64()),
            ("elo_difference_pre", pa.float64()),
            ("home_expected_score", pa.float64()),
            ("home_elo_post", pa.float64()),
            ("away_elo_post", pa.float64()),
            ("rating_change", pa.float64()),
            ("model_version", pa.string()),
        ]
    )


def _current_ratings_schema() -> pa.Schema:
    return pa.schema(
        [
            ("canonical_team_id", pa.string()),
            ("elo_rating", pa.float64()),
            ("matches_processed", pa.int64()),
            ("last_match_date", pa.date32()),
            ("model_version", pa.string()),
        ]
    )

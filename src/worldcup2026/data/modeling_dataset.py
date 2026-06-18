"""Deterministic modeling dataset preparation from canonical historical matches."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from worldcup2026.data.contracts import TeamStatus, TeamType
from worldcup2026.data.historical_ingest import load_team_catalog


class CompetitionCategory(StrEnum):
    """Modeling taxonomy for source tournament names."""

    FRIENDLY = "friendly"
    WORLD_CUP_QUALIFIER = "world_cup_qualifier"
    CONFEDERATION_QUALIFIER = "confederation_qualifier"
    WORLD_CUP = "world_cup"
    CONFEDERATION_CHAMPIONSHIP = "confederation_championship"
    NATIONS_LEAGUE = "nations_league"
    OTHER_OFFICIAL = "other_official"
    OTHER = "other"


class NeutralMatchPolicy(StrEnum):
    """How neutral-site matches affect main modeling eligibility."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


class IncompleteRecordPolicy(StrEnum):
    """How incomplete canonical rows affect main modeling eligibility."""

    EXCLUDE = "exclude"


class ModelingPeriod(StrEnum):
    """Time bucket used by future rating/model evaluation stages."""

    PRE_RATING_HISTORY = "pre_rating_history"
    RATING_HISTORY = "rating_history"
    EVALUATION = "evaluation"


class ExclusionReason(StrEnum):
    """Primary reason a row is not eligible for the main modeling dataset."""

    BEFORE_RATING_HISTORY_START = "before_rating_history_start"
    INCOMPLETE_RECORD = "incomplete_record"
    UNRESOLVED_COMPETITION_CATEGORY = "unresolved_competition_category"
    COMPETITION_CATEGORY_NOT_ALLOWED = "competition_category_not_allowed"
    TEAM_STATUS_NOT_ALLOWED = "team_status_not_allowed"
    TEAM_TYPE_NOT_ALLOWED = "team_type_not_allowed"
    NEUTRAL_SITE_EXCLUDED = "neutral_site_excluded"


class ModelingDatasetConfig(BaseModel):
    """Declarative configuration for modeling dataset preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rating_history_start: date = date(2000, 1, 1)
    evaluation_start: date = date(2014, 1, 1)
    allowed_team_statuses: tuple[TeamStatus, ...] = (TeamStatus.CURRENT, TeamStatus.HISTORICAL)
    allowed_team_types: tuple[TeamType, ...] = (
        TeamType.NATIONAL_TEAM,
        TeamType.TERRITORY_TEAM,
        TeamType.DEFUNCT_STATE_TEAM,
    )
    allowed_competition_categories: tuple[CompetitionCategory, ...] = (
        CompetitionCategory.FRIENDLY,
        CompetitionCategory.WORLD_CUP_QUALIFIER,
        CompetitionCategory.CONFEDERATION_QUALIFIER,
        CompetitionCategory.WORLD_CUP,
        CompetitionCategory.CONFEDERATION_CHAMPIONSHIP,
        CompetitionCategory.NATIONS_LEAGUE,
        CompetitionCategory.OTHER_OFFICIAL,
    )
    neutral_match_policy: NeutralMatchPolicy = NeutralMatchPolicy.INCLUDE
    incomplete_record_policy: IncompleteRecordPolicy = IncompleteRecordPolicy.EXCLUDE
    competition_importance: Mapping[CompetitionCategory, int] = Field(default_factory=dict)
    input_matches_path: Path = Path("data/processed/international_matches.parquet")
    output_matches_path: Path = Path("data/processed/modeling_matches.parquet")
    report_output: Path = Path("data/interim/modeling_data_report.json")
    teams_path: Path = Path("data/static/teams.csv")
    competition_mapping_path: Path = Path("configs/competition_categories.yaml")

    @model_validator(mode="after")
    def validate_modeling_windows_and_importance(self) -> Self:
        """Require coherent temporal windows and complete category importance values."""

        if self.evaluation_start < self.rating_history_start:
            msg = "evaluation_start cannot be earlier than rating_history_start"
            raise ValueError(msg)
        missing_importance = sorted(
            category.value
            for category in CompetitionCategory
            if category not in self.competition_importance
        )
        if missing_importance:
            msg = "competition_importance is missing categories: " + ", ".join(
                missing_importance
            )
            raise ValueError(msg)
        return self


class ModelingDataReport(BaseModel):
    """Quality report emitted by the modeling dataset preparation stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_path: Path
    output_path: Path
    total_rows: int
    eligible_rows: int
    exclusions_by_reason: Mapping[str, int]
    matches_by_category: Mapping[str, int]
    date_range: Mapping[str, str | None]
    teams_included: int
    included_team_ids: tuple[str, ...]
    neutral_matches: int
    matches_by_year: Mapping[str, int]
    unresolved_competitions: tuple[str, ...]


class ModelingDatasetResult(BaseModel):
    """Return value for programmatic modeling dataset preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    table: Any
    report: ModelingDataReport


class ModelingDatasetError(RuntimeError):
    """Raised when modeling dataset preparation cannot be completed."""


def load_modeling_dataset_config(
    config_path: Path = Path("configs/modeling_data.yaml"),
) -> ModelingDatasetConfig:
    """Load the declarative modeling dataset configuration."""

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read modeling dataset config {config_path}: {exc}"
        raise ModelingDatasetError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse modeling dataset config {config_path}: {exc}"
        raise ModelingDatasetError(msg) from exc
    if not isinstance(config, dict):
        msg = f"modeling dataset config {config_path} must contain a YAML mapping"
        raise ModelingDatasetError(msg)
    section = config.get("modeling_data")
    if not isinstance(section, dict):
        msg = f"{config_path} is missing the modeling_data section"
        raise ModelingDatasetError(msg)
    try:
        return ModelingDatasetConfig.model_validate(section)
    except ValidationError as exc:
        msg = f"modeling dataset config {config_path} is invalid: {exc}"
        raise ModelingDatasetError(msg) from exc


def run_modeling_dataset_preparation(
    config: ModelingDatasetConfig,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
) -> ModelingDatasetResult:
    """Build the deterministic modeling dataset and write Parquet plus a JSON report."""

    effective_input_path = input_path or config.input_matches_path
    effective_output_path = output_path or config.output_matches_path
    effective_report_path = report_path or config.report_output
    matches = _read_matches(effective_input_path)
    teams_by_id = {team.canonical_team_id: team for team in load_team_catalog(config.teams_path)}
    competition_mapping = load_competition_mapping(config.competition_mapping_path)

    rows = [
        _prepare_row(
            row,
            config=config,
            teams_by_id=teams_by_id,
            competition_mapping=competition_mapping,
        )
        for row in matches
    ]
    sorted_rows = sorted(rows, key=lambda row: (row["match_date"], row["match_id"]))
    table = pa.Table.from_pylist(sorted_rows, schema=_modeling_parquet_schema())
    effective_output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, effective_output_path)  # type: ignore[no-untyped-call]

    report = _build_report(
        sorted_rows,
        input_path=effective_input_path,
        output_path=effective_output_path,
    )
    write_modeling_report(report, effective_report_path)
    return ModelingDatasetResult(table=table, report=report)


def load_competition_mapping(path: Path) -> dict[str, CompetitionCategory]:
    """Load exact tournament-name mappings to modeling categories."""

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read competition mapping {path}: {exc}"
        raise ModelingDatasetError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse competition mapping {path}: {exc}"
        raise ModelingDatasetError(msg) from exc
    if not isinstance(config, dict):
        msg = f"competition mapping {path} must contain a YAML mapping"
        raise ModelingDatasetError(msg)
    section = config.get("competition_categories")
    if not isinstance(section, dict):
        msg = f"{path} is missing the competition_categories section"
        raise ModelingDatasetError(msg)

    mapping: dict[str, CompetitionCategory] = {}
    duplicates: set[str] = set()
    for raw_category, raw_competitions in section.items():
        category = CompetitionCategory(str(raw_category))
        if not isinstance(raw_competitions, list):
            msg = f"competition category {category.value} must be a list"
            raise ModelingDatasetError(msg)
        for raw_competition in raw_competitions:
            competition = str(raw_competition).strip()
            if not competition:
                msg = f"competition category {category.value} contains a blank tournament name"
                raise ModelingDatasetError(msg)
            if competition in mapping:
                duplicates.add(competition)
            mapping[competition] = category
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        msg = f"competition mapping contains duplicate tournament names: {duplicate_list}"
        raise ModelingDatasetError(msg)
    return mapping


def write_modeling_report(report: ModelingDataReport, path: Path) -> None:
    """Write the modeling data quality report as deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_matches(path: Path) -> list[dict[str, Any]]:
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        msg = f"failed to read canonical matches parquet {path}: {exc}"
        raise ModelingDatasetError(msg) from exc
    rows = table.to_pylist()
    return [dict(row) for row in rows]


def _prepare_row(
    row: Mapping[str, Any],
    *,
    config: ModelingDatasetConfig,
    teams_by_id: Mapping[str, Any],
    competition_mapping: Mapping[str, CompetitionCategory],
) -> dict[str, Any]:
    competition = str(row.get("competition") or "")
    category = competition_mapping.get(competition)
    match_date = _require_date(row["match_date"])
    modeling_period = _modeling_period(
        match_date,
        rating_history_start=config.rating_history_start,
        evaluation_start=config.evaluation_start,
    )
    home_team = teams_by_id.get(str(row.get("home_team_id")))
    away_team = teams_by_id.get(str(row.get("away_team_id")))
    home_advantage_status = str(row.get("home_advantage_status") or "")
    neutral_site = row.get("neutral_site")
    home_advantage_eligible = home_advantage_status in {"home_team", "away_team"}

    exclusion_reason = _exclusion_reason(
        row,
        config=config,
        match_date=match_date,
        category=category,
        home_team=home_team,
        away_team=away_team,
        neutral_site=neutral_site,
    )
    output = dict(row)
    output.update(
        {
            "home_team_status": home_team.team_status.value if home_team is not None else None,
            "away_team_status": away_team.team_status.value if away_team is not None else None,
            "home_team_type": home_team.team_type.value if home_team is not None else None,
            "away_team_type": away_team.team_type.value if away_team is not None else None,
            "model_eligible": exclusion_reason is None,
            "exclusion_reason": exclusion_reason.value if exclusion_reason is not None else None,
            "competition_category": category.value if category is not None else None,
            "competition_importance": (
                config.competition_importance[category] if category is not None else None
            ),
            "home_advantage_eligible": home_advantage_eligible,
            "modeling_period": modeling_period.value,
            "same_date_batch_id": match_date.isoformat(),
        }
    )
    return output


def _exclusion_reason(
    row: Mapping[str, Any],
    *,
    config: ModelingDatasetConfig,
    match_date: date,
    category: CompetitionCategory | None,
    home_team: Any,
    away_team: Any,
    neutral_site: object,
) -> ExclusionReason | None:
    if match_date < config.rating_history_start:
        return ExclusionReason.BEFORE_RATING_HISTORY_START
    if _is_incomplete_record(row, home_team=home_team, away_team=away_team):
        return ExclusionReason.INCOMPLETE_RECORD
    if category is None:
        return ExclusionReason.UNRESOLVED_COMPETITION_CATEGORY
    if category not in config.allowed_competition_categories:
        return ExclusionReason.COMPETITION_CATEGORY_NOT_ALLOWED
    team_statuses = {home_team.team_status, away_team.team_status}
    if not team_statuses.issubset(set(config.allowed_team_statuses)):
        return ExclusionReason.TEAM_STATUS_NOT_ALLOWED
    team_types = {home_team.team_type, away_team.team_type}
    if not team_types.issubset(set(config.allowed_team_types)):
        return ExclusionReason.TEAM_TYPE_NOT_ALLOWED
    if config.neutral_match_policy is NeutralMatchPolicy.EXCLUDE and neutral_site is True:
        return ExclusionReason.NEUTRAL_SITE_EXCLUDED
    return None


def _is_incomplete_record(row: Mapping[str, Any], *, home_team: Any, away_team: Any) -> bool:
    required_fields = (
        "match_id",
        "match_status",
        "match_date",
        "home_team_id",
        "away_team_id",
        "home_goals_90",
        "away_goals_90",
        "result_90",
        "competition",
        "source",
        "source_match_id",
    )
    if any(row.get(field_name) is None for field_name in required_fields):
        return True
    return (
        row.get("match_status") != "played"
        or home_team is None
        or away_team is None
        or not isinstance(row.get("neutral_site"), bool)
    )


def _modeling_period(
    match_date: date,
    *,
    rating_history_start: date,
    evaluation_start: date,
) -> ModelingPeriod:
    if match_date < rating_history_start:
        return ModelingPeriod.PRE_RATING_HISTORY
    if match_date < evaluation_start:
        return ModelingPeriod.RATING_HISTORY
    return ModelingPeriod.EVALUATION


def _build_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
) -> ModelingDataReport:
    row_list = list(rows)
    eligible_rows = [row for row in row_list if row["model_eligible"] is True]
    exclusion_counts = Counter(
        str(row["exclusion_reason"])
        for row in row_list
        if row["exclusion_reason"] is not None
    )
    category_counts = Counter(
        str(row["competition_category"])
        for row in row_list
        if row["competition_category"] is not None
    )
    match_dates = [_require_date(row["match_date"]) for row in row_list]
    team_ids = sorted(
        {
            str(row[field_name])
            for row in eligible_rows
            for field_name in ("home_team_id", "away_team_id")
        }
    )
    matches_by_year = Counter(str(_require_date(row["match_date"]).year) for row in row_list)
    unresolved_competitions = sorted(
        {
            str(row["competition"])
            for row in row_list
            if row["competition_category"] is None
        }
    )
    return ModelingDataReport(
        input_path=input_path,
        output_path=output_path,
        total_rows=len(row_list),
        eligible_rows=len(eligible_rows),
        exclusions_by_reason=dict(sorted(exclusion_counts.items())),
        matches_by_category=dict(sorted(category_counts.items())),
        date_range={
            "min": min(match_dates).isoformat() if match_dates else None,
            "max": max(match_dates).isoformat() if match_dates else None,
        },
        teams_included=len(team_ids),
        included_team_ids=tuple(team_ids),
        neutral_matches=sum(1 for row in row_list if row["neutral_site"] is True),
        matches_by_year=dict(sorted(matches_by_year.items())),
        unresolved_competitions=tuple(unresolved_competitions),
    )


def _require_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"expected date value, got {value!r}"
    raise ModelingDatasetError(msg)


def _modeling_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            ("match_id", pa.string()),
            ("schema_version", pa.string()),
            ("match_status", pa.string()),
            ("match_date", pa.date32()),
            ("kickoff_utc", pa.timestamp("us", tz="UTC")),
            ("kickoff_local_time", pa.string()),
            ("kickoff_timezone", pa.string()),
            ("kickoff_time_status", pa.string()),
            ("home_team_name_original", pa.string()),
            ("away_team_name_original", pa.string()),
            ("home_team_id", pa.string()),
            ("away_team_id", pa.string()),
            ("home_goals_90", pa.int64()),
            ("away_goals_90", pa.int64()),
            ("result_90", pa.string()),
            ("extra_time_played", pa.bool_()),
            ("home_goals_after_extra_time", pa.int64()),
            ("away_goals_after_extra_time", pa.int64()),
            ("penalty_shootout", pa.bool_()),
            ("home_penalty_goals", pa.int64()),
            ("away_penalty_goals", pa.int64()),
            ("competition", pa.string()),
            ("stage", pa.string()),
            ("match_type", pa.string()),
            ("city", pa.string()),
            ("host_country", pa.string()),
            ("venue_name_original", pa.string()),
            ("neutral_site", pa.bool_()),
            ("home_advantage_status", pa.string()),
            ("source", pa.string()),
            ("source_match_id", pa.string()),
            ("retrieved_at_utc", pa.timestamp("us", tz="UTC")),
            ("home_team_status", pa.string()),
            ("away_team_status", pa.string()),
            ("home_team_type", pa.string()),
            ("away_team_type", pa.string()),
            ("model_eligible", pa.bool_()),
            ("exclusion_reason", pa.string()),
            ("competition_category", pa.string()),
            ("competition_importance", pa.int64()),
            ("home_advantage_eligible", pa.bool_()),
            ("modeling_period", pa.string()),
            ("same_date_batch_id", pa.string()),
        ]
    )

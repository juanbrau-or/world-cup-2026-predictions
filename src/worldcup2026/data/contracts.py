"""Canonical data contracts for international football matches."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

MATCH_SCHEMA_VERSION: Literal["international_match_v1"] = "international_match_v1"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CanonicalTeamId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=32,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
LocalKickoffTime = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^\d{2}:\d{2}$")
]
UTC_OFFSET_PATTERN = re.compile(r"^[+-](\d{2}):(\d{2})$")


class MatchStatus(StrEnum):
    """Lifecycle status for a canonical match record."""

    PLAYED = "played"
    SCHEDULED = "scheduled"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    ABANDONED = "abandoned"


class KickoffTimeStatus(StrEnum):
    """Precision available for the match kickoff time."""

    EXACT_UTC = "exact_utc"
    DATE_ONLY = "date_only"
    LOCAL_TIME_WITHOUT_TIMEZONE = "local_time_without_timezone"


class Result90(StrEnum):
    """Result after regulation time, before extra time or penalties."""

    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


class HomeAdvantageStatus(StrEnum):
    """Whether either listed team had real host-team advantage."""

    HOME_TEAM = "home_team"
    AWAY_TEAM = "away_team"
    NEUTRAL = "neutral"
    SHARED_HOST = "shared_host"
    UNKNOWN = "unknown"


class MatchType(StrEnum):
    """Documented high-level category of an international match."""

    FRIENDLY = "friendly"
    QUALIFIER = "qualifier"
    CONTINENTAL_TOURNAMENT = "continental_tournament"
    WORLD_CUP = "world_cup"
    OTHER = "other"


class TeamAlias(BaseModel):
    """Canonical team alias valid for a source name and optional date range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_team_id: CanonicalTeamId
    canonical_name: NonEmptyStr
    source: NonEmptyStr
    source_name: NonEmptyStr
    valid_from: date | None = None
    valid_to: date | None = None

    @model_validator(mode="after")
    def validate_validity_range(self) -> Self:
        """Reject inverted alias validity windows."""

        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to < self.valid_from
        ):
            msg = "alias valid_to cannot be earlier than valid_from"
            raise ValueError(msg)
        return self

    def applies_to(self, *, source: str, source_name: str, match_date: date) -> bool:
        """Return whether this alias applies to a source team name on a match date."""

        if self.source != source or self.source_name != source_name:
            return False
        if self.valid_from is not None and match_date < self.valid_from:
            return False
        return not (self.valid_to is not None and match_date > self.valid_to)


class CanonicalMatch(BaseModel):
    """Validated canonical representation of one international national-team match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_id: NonEmptyStr
    schema_version: Literal["international_match_v1"] = MATCH_SCHEMA_VERSION
    match_status: MatchStatus

    match_date: date
    kickoff_utc: datetime | None = None
    kickoff_local_time: LocalKickoffTime | None = None
    kickoff_timezone: NonEmptyStr | None = None
    kickoff_time_status: KickoffTimeStatus

    home_team_name_original: NonEmptyStr
    away_team_name_original: NonEmptyStr
    home_team_id: CanonicalTeamId
    away_team_id: CanonicalTeamId

    home_goals_90: NonNegativeInt | None = None
    away_goals_90: NonNegativeInt | None = None
    result_90: Result90 | None = None
    extra_time_played: StrictBool
    home_goals_after_extra_time: NonNegativeInt | None = None
    away_goals_after_extra_time: NonNegativeInt | None = None
    penalty_shootout: StrictBool
    home_penalty_goals: NonNegativeInt | None = None
    away_penalty_goals: NonNegativeInt | None = None

    competition: NonEmptyStr
    stage: NonEmptyStr | None = None
    match_type: MatchType
    city: NonEmptyStr | None = None
    host_country: NonEmptyStr | None = None
    venue_name_original: NonEmptyStr | None = None
    neutral_site: StrictBool | None = None
    home_advantage_status: HomeAdvantageStatus

    source: NonEmptyStr
    source_match_id: NonEmptyStr
    retrieved_at_utc: datetime

    @field_validator("kickoff_utc", "retrieved_at_utc")
    @classmethod
    def datetime_must_be_utc(cls, value: datetime | None) -> datetime | None:
        """Require timezone-aware UTC datetimes whenever a timestamp is present."""

        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            msg = "datetime values must be timezone-aware UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @field_validator("kickoff_timezone")
    @classmethod
    def timezone_must_be_iana_or_offset(cls, value: str | None) -> str | None:
        """Accept only IANA timezones or explicit UTC offsets."""

        if value is None:
            return None
        if _is_valid_utc_offset(value):
            return value
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            msg = "kickoff_timezone must be an IANA timezone or UTC offset"
            raise ValueError(msg) from exc
        return value

    @field_validator("kickoff_local_time")
    @classmethod
    def local_time_must_be_valid_clock_time(cls, value: str | None) -> str | None:
        """Reject HH:MM strings that match the pattern but are not valid clock times."""

        if value is None:
            return None
        if re.fullmatch(r"\d{2}:\d{2}", value) is None:
            msg = "kickoff_local_time must use HH:MM"
            raise ValueError(msg)
        hour, minute = (int(part) for part in value.split(":"))
        if hour > 23 or minute > 59:
            msg = "kickoff_local_time must be a valid 24-hour HH:MM time"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_canonical_match(self) -> Self:
        """Validate cross-field rules that keep match facts temporally explicit."""

        self._validate_team_ids()
        self._validate_kickoff_precision()
        self._validate_scores_for_status()
        self._validate_result_90()
        self._validate_extra_time()
        self._validate_penalties()
        self._validate_home_advantage()
        return self

    def _validate_team_ids(self) -> None:
        if self.home_team_id == self.away_team_id:
            msg = "home_team_id and away_team_id must be different"
            raise ValueError(msg)

    def _validate_kickoff_precision(self) -> None:
        if self.kickoff_time_status is KickoffTimeStatus.EXACT_UTC:
            if self.kickoff_utc is None:
                msg = "kickoff_utc is required when kickoff_time_status is exact_utc"
                raise ValueError(msg)
            return

        if self.kickoff_time_status is KickoffTimeStatus.DATE_ONLY:
            if (
                self.kickoff_utc is not None
                or self.kickoff_local_time is not None
                or self.kickoff_timezone is not None
            ):
                msg = "date_only records must not include kickoff_utc, local time, or timezone"
                raise ValueError(msg)
            return

        if self.kickoff_utc is not None:
            msg = "local_time_without_timezone records must not include kickoff_utc"
            raise ValueError(msg)
        if self.kickoff_local_time is None:
            msg = "local_time_without_timezone records require kickoff_local_time"
            raise ValueError(msg)
        if self.kickoff_timezone is not None:
            msg = "local_time_without_timezone records must not include kickoff_timezone"
            raise ValueError(msg)

    def _validate_scores_for_status(self) -> None:
        score_fields = (
            self.home_goals_90,
            self.away_goals_90,
            self.home_goals_after_extra_time,
            self.away_goals_after_extra_time,
            self.home_penalty_goals,
            self.away_penalty_goals,
        )
        if self.match_status is MatchStatus.PLAYED:
            if self.home_goals_90 is None or self.away_goals_90 is None:
                msg = "played matches require both 90-minute goal fields"
                raise ValueError(msg)
            if self.result_90 is None:
                msg = "played matches require result_90"
                raise ValueError(msg)
            return

        if any(value is not None for value in score_fields) or self.result_90 is not None:
            msg = "matches that were not played must not include scores or result_90"
            raise ValueError(msg)
        if self.extra_time_played or self.penalty_shootout:
            msg = "matches that were not played must not include extra time or penalties"
            raise ValueError(msg)

    def _validate_result_90(self) -> None:
        if self.home_goals_90 is None or self.away_goals_90 is None or self.result_90 is None:
            return
        expected = _result_from_scores(self.home_goals_90, self.away_goals_90)
        if self.result_90 is not expected:
            msg = "result_90 must match home_goals_90 and away_goals_90"
            raise ValueError(msg)

    def _validate_extra_time(self) -> None:
        extra_scores = (self.home_goals_after_extra_time, self.away_goals_after_extra_time)
        if not self.extra_time_played:
            if any(value is not None for value in extra_scores):
                msg = "extra-time goal fields require extra_time_played=true"
                raise ValueError(msg)
            return

        if self.home_goals_after_extra_time is None or self.away_goals_after_extra_time is None:
            msg = "extra_time_played=true requires both extra-time goal fields"
            raise ValueError(msg)
        if self.home_goals_90 is None or self.away_goals_90 is None:
            return
        if self.home_goals_90 != self.away_goals_90:
            msg = "extra_time_played=true requires tied 90-minute goals"
            raise ValueError(msg)
        if (
            self.home_goals_after_extra_time < self.home_goals_90
            or self.away_goals_after_extra_time < self.away_goals_90
        ):
            msg = "extra-time goals cannot be lower than 90-minute goals"
            raise ValueError(msg)

    def _validate_penalties(self) -> None:
        penalty_scores = (self.home_penalty_goals, self.away_penalty_goals)
        if not self.penalty_shootout:
            if any(value is not None for value in penalty_scores):
                msg = "penalty goal fields require penalty_shootout=true"
                raise ValueError(msg)
            return

        if self.home_penalty_goals is None or self.away_penalty_goals is None:
            msg = "penalty_shootout=true requires both penalty goal fields"
            raise ValueError(msg)
        if self.home_penalty_goals == self.away_penalty_goals:
            msg = "penalty shootout goals must identify a winner"
            raise ValueError(msg)

        if self.extra_time_played:
            if self.home_goals_after_extra_time != self.away_goals_after_extra_time:
                msg = "penalty shootouts after extra time require tied extra-time goals"
                raise ValueError(msg)
            return

        if self.home_goals_90 != self.away_goals_90:
            msg = "penalty shootouts without extra time require tied 90-minute goals"
            raise ValueError(msg)

    def _validate_home_advantage(self) -> None:
        if self.neutral_site is True and self.home_advantage_status not in {
            HomeAdvantageStatus.NEUTRAL,
            HomeAdvantageStatus.SHARED_HOST,
        }:
            msg = "neutral_site=true requires neutral or shared_host home_advantage_status"
            raise ValueError(msg)
        if self.neutral_site is False and self.home_advantage_status is HomeAdvantageStatus.NEUTRAL:
            msg = "neutral_site=false is incompatible with neutral home_advantage_status"
            raise ValueError(msg)


def validate_match_records(
    records: Iterable[CanonicalMatch | Mapping[str, Any]],
    *,
    require_temporal_order: bool = False,
    team_aliases: Iterable[TeamAlias | Mapping[str, Any]] | None = None,
) -> list[CanonicalMatch]:
    """Validate a collection of match records and reject duplicate identifiers.

    Args:
        records: CanonicalMatch instances or mappings that can be parsed into one.
        require_temporal_order: When true, require records to be ordered by their known
            kickoff time, falling back to date comparisons when kickoff precision is missing.
        team_aliases: Optional alias records used to verify canonical team IDs and historical
            source names by match date.

    Returns:
        A list of validated CanonicalMatch instances.

    Raises:
        ValueError: If duplicate IDs, duplicate source keys, or temporal ordering errors appear.
        pydantic.ValidationError: If an individual record violates the canonical contract.
    """

    matches = [
        record if isinstance(record, CanonicalMatch) else CanonicalMatch.model_validate(record)
        for record in records
    ]
    _reject_duplicate_match_ids(matches)
    _reject_duplicate_source_keys(matches)
    if team_aliases is not None:
        aliases = [
            alias if isinstance(alias, TeamAlias) else TeamAlias.model_validate(alias)
            for alias in team_aliases
        ]
        _validate_match_aliases(matches, aliases)
    if require_temporal_order:
        _validate_temporal_order(matches)
    return matches


def resolve_team_alias(
    *,
    source: str,
    source_name: str,
    match_date: date,
    aliases: Iterable[TeamAlias],
) -> str:
    """Resolve a source team name to one canonical ID for a specific match date."""

    matches = [
        alias
        for alias in aliases
        if alias.applies_to(source=source, source_name=source_name, match_date=match_date)
    ]
    if not matches:
        msg = f"missing team alias for {source}:{source_name} on {match_date.isoformat()}"
        raise ValueError(msg)
    canonical_ids = {alias.canonical_team_id for alias in matches}
    if len(canonical_ids) > 1:
        msg = f"ambiguous team alias for {source}:{source_name} on {match_date.isoformat()}"
        raise ValueError(msg)
    return matches[0].canonical_team_id


def _result_from_scores(home_goals: int, away_goals: int) -> Result90:
    if home_goals > away_goals:
        return Result90.HOME_WIN
    if home_goals < away_goals:
        return Result90.AWAY_WIN
    return Result90.DRAW


def _is_valid_utc_offset(value: str) -> bool:
    match = UTC_OFFSET_PATTERN.fullmatch(value)
    if match is None:
        return False
    hours = int(match.group(1))
    minutes = int(match.group(2))
    return hours < 14 or (hours == 14 and minutes == 0)


def _validate_match_aliases(matches: Iterable[CanonicalMatch], aliases: list[TeamAlias]) -> None:
    for match in matches:
        expected_home_id = resolve_team_alias(
            source=match.source,
            source_name=match.home_team_name_original,
            match_date=match.match_date,
            aliases=aliases,
        )
        if expected_home_id != match.home_team_id:
            msg = (
                f"home_team_id {match.home_team_id} does not match alias "
                f"{expected_home_id} for {match.match_id}"
            )
            raise ValueError(msg)
        expected_away_id = resolve_team_alias(
            source=match.source,
            source_name=match.away_team_name_original,
            match_date=match.match_date,
            aliases=aliases,
        )
        if expected_away_id != match.away_team_id:
            msg = (
                f"away_team_id {match.away_team_id} does not match alias "
                f"{expected_away_id} for {match.match_id}"
            )
            raise ValueError(msg)


def _reject_duplicate_match_ids(matches: Iterable[CanonicalMatch]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for match in matches:
        if match.match_id in seen:
            duplicates.add(match.match_id)
        seen.add(match.match_id)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        msg = f"duplicate match_id values: {duplicate_list}"
        raise ValueError(msg)


def _reject_duplicate_source_keys(matches: Iterable[CanonicalMatch]) -> None:
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for match in matches:
        source_key = (match.source, match.source_match_id)
        if source_key in seen:
            duplicates.add(source_key)
        seen.add(source_key)
    if duplicates:
        duplicate_list = ", ".join(f"{source}:{source_id}" for source, source_id in duplicates)
        msg = f"duplicate source/source_match_id values: {duplicate_list}"
        raise ValueError(msg)


def _validate_temporal_order(matches: Iterable[CanonicalMatch]) -> None:
    previous_match: CanonicalMatch | None = None
    previous_match_id: str | None = None
    for match in matches:
        if previous_match is not None and _match_is_before(match, previous_match):
            msg = (
                "matches must be ordered by kickoff_utc or match_date; "
                f"{match.match_id} appears before {previous_match_id}"
            )
            raise ValueError(msg)
        previous_match = match
        previous_match_id = match.match_id


def _match_is_before(current: CanonicalMatch, previous: CanonicalMatch) -> bool:
    if current.kickoff_utc is not None and previous.kickoff_utc is not None:
        return current.kickoff_utc < previous.kickoff_utc
    return current.match_date < previous.match_date

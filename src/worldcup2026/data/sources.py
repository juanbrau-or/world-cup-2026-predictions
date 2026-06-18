"""Typed interfaces for historical match data sources."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from worldcup2026.data.contracts import CanonicalMatch, TeamAlias

RAW_SNAPSHOT_MANIFEST_VERSION: Literal["raw_snapshot_manifest_v1"] = "raw_snapshot_manifest_v1"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Hex = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^[0-9a-f]{64}$")]


class HistoricalSourceCandidate(BaseModel):
    """Documented choice for a historical source considered in Phase 1A."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: NonEmptyStr
    role: Literal["selected", "alternative"]
    reason: NonEmptyStr
    user_confirmation_needed: tuple[NonEmptyStr, ...]


class HistoricalFetchRequest(BaseModel):
    """Logical request for a historical source snapshot.

    This request intentionally avoids physical endpoints. Phase 1B can bind the logical URI to a
    local file, authenticated download, or API only after the user confirms licensing and access.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: NonEmptyStr
    logical_uri: NonEmptyStr
    cache_dir: Path


class RawSnapshotManifest(BaseModel):
    """Manifest that makes a raw snapshot reproducible and idempotent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["raw_snapshot_manifest_v1"] = RAW_SNAPSHOT_MANIFEST_VERSION
    source: NonEmptyStr
    logical_uri: NonEmptyStr
    retrieved_at_utc: datetime
    content_sha256: Sha256Hex
    cache_key: NonEmptyStr
    raw_path: Path

    @field_validator("retrieved_at_utc")
    @classmethod
    def retrieved_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require and normalize timezone-aware UTC retrieval timestamps."""

        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            msg = "retrieved_at_utc must be timezone-aware UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


class RawSnapshot(BaseModel):
    """Raw bytes plus manifest metadata before any normalization occurs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: RawSnapshotManifest
    content: bytes


class HistoricalSourceRecord(BaseModel):
    """One raw source record after parsing a snapshot, before canonical normalization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: NonEmptyStr
    source_match_id: NonEmptyStr
    payload: Mapping[str, object]


class HistoricalDataClient(Protocol):
    """Fetch a raw historical snapshot without mutating the canonical dataset."""

    def fetch_snapshot(self, request: HistoricalFetchRequest) -> RawSnapshot:
        """Return raw source bytes and a manifest for the requested logical source."""


class HistoricalMatchAdapter(Protocol):
    """Convert parsed source records to canonical matches."""

    def adapt_records(
        self,
        records: Iterable[HistoricalSourceRecord],
        *,
        aliases: Iterable[TeamAlias],
    ) -> list[CanonicalMatch]:
        """Normalize source rows into canonical matches using explicit team aliases."""


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 checksum used by raw snapshot manifests."""

    return hashlib.sha256(content).hexdigest()

"""Historical international match ingestion pipeline."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Self

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from worldcup2026.data.contracts import (
    MATCH_SCHEMA_VERSION,
    CanonicalMatch,
    HomeAdvantageStatus,
    KickoffTimeStatus,
    MatchStatus,
    MatchType,
    Result90,
    TeamAlias,
    resolve_team_alias,
    validate_match_records,
)
from worldcup2026.data.sources import RawSnapshot, RawSnapshotManifest, sha256_bytes

SOURCE_NAME = "international_results_csv"
SOURCE_FILE_NAMES: tuple[Literal["results", "shootouts"], ...] = ("results", "shootouts")


class SourceFileConfig(BaseModel):
    """Configuration for one downloadable source file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_uri: str
    url: str
    filename: str
    expected_columns: tuple[str, ...]


class HistoricalCsvSourceConfig(BaseModel):
    """Declarative configuration for the selected historical CSV source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str
    license: str
    homepage: str
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    snapshot_retrieved_at_utc: datetime
    timeout_seconds: float = Field(gt=0)
    retries: int = Field(ge=0)
    files: Mapping[Literal["results", "shootouts"], SourceFileConfig]
    raw_root: Path
    processed_output: Path
    quarantine_output: Path
    report_output: Path
    aliases_path: Path

    @model_validator(mode="after")
    def file_urls_must_use_source_revision(self) -> Self:
        """Require each configured remote file to be pinned to the documented revision."""

        revision_suffix = f"@{self.source_revision}"
        revision_path = f"/{self.source_revision}/"
        for file_name, file_config in self.files.items():
            if not file_config.logical_uri.endswith(revision_suffix):
                msg = (
                    f"{file_name} logical_uri must end with the configured "
                    f"source_revision {revision_suffix!r}"
                )
                raise ValueError(msg)
            if revision_path not in file_config.url:
                msg = (
                    f"{file_name} url must include the configured source_revision "
                    f"{self.source_revision!r}"
                )
                raise ValueError(msg)
        return self

    @field_validator("snapshot_retrieved_at_utc")
    @classmethod
    def snapshot_retrieved_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require the pinned snapshot timestamp to be timezone-aware UTC."""

        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            msg = "snapshot_retrieved_at_utc must be timezone-aware UTC"
            raise ValueError(msg)
        return value.astimezone(UTC)


class InvalidHistoricalRecord(BaseModel):
    """A source row that could not be safely represented in the canonical contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    source_file: str
    logical_uri: str
    source_revision: str
    row_number: int | None
    stage: str
    source_match_id: str | None
    reason: str
    payload: Mapping[str, object]


class HistoricalIngestReport(BaseModel):
    """Quality report emitted by the historical ingestion pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    source_homepage: str
    source_license: str
    retrieved_at_utc: datetime
    results_rows_downloaded: int
    results_rows_processed: int
    shootout_rows_downloaded: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    normalized_rows: int
    snapshot_manifests: tuple[RawSnapshotManifest, ...]
    output_path: Path | None
    quarantine_path: Path | None
    dry_run: bool


class HistoricalIngestResult(BaseModel):
    """Return value for programmatic ingestion runs and tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[CanonicalMatch, ...]
    invalid_records: tuple[InvalidHistoricalRecord, ...]
    report: HistoricalIngestReport


class HistoricalIngestError(RuntimeError):
    """Raised when the ingestion source cannot be fetched or parsed."""


@dataclass(frozen=True)
class ParsedHistoricalRow:
    """One CSV row plus immutable source coordinates for quarantine traceability."""

    source_file: str
    logical_uri: str
    source_revision: str
    row_number: int
    payload: Mapping[str, object]


@dataclass(frozen=True)
class FetchedSourceFile:
    """Raw source bytes plus their effective origin before snapshot persistence."""

    file_config: SourceFileConfig
    content: bytes
    input_uri: str


@dataclass(frozen=True)
class ParsedCsvSnapshot:
    """Parsed CSV rows and row-level parse errors for one raw snapshot."""

    row_count: int
    rows: tuple[ParsedHistoricalRow, ...]
    invalid_records: tuple[InvalidHistoricalRecord, ...]


def load_historical_source_config(
    config_path: Path = Path("configs/sources.yaml"),
    *,
    source_name: str = SOURCE_NAME,
) -> HistoricalCsvSourceConfig:
    """Load the declarative historical source configuration."""

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read source config {config_path}: {exc}"
        raise HistoricalIngestError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"failed to parse source config {config_path}: {exc}"
        raise HistoricalIngestError(msg) from exc
    if not isinstance(config, dict):
        msg = f"source config {config_path} must contain a YAML mapping"
        raise HistoricalIngestError(msg)

    historical = config.get("historical")
    if not isinstance(historical, dict):
        msg = "configs/sources.yaml is missing the historical source section"
        raise HistoricalIngestError(msg)
    if historical.get("provider") != source_name:
        msg = f"historical provider must be {source_name!r}"
        raise HistoricalIngestError(msg)

    sources = historical.get("sources")
    if not isinstance(sources, dict) or source_name not in sources:
        msg = f"historical source {source_name!r} is not configured"
        raise HistoricalIngestError(msg)
    try:
        return HistoricalCsvSourceConfig.model_validate(sources[source_name])
    except ValidationError as exc:
        msg = f"historical source {source_name!r} in {config_path} is invalid: {exc}"
        raise HistoricalIngestError(msg) from exc


def run_historical_ingest(
    config: HistoricalCsvSourceConfig,
    *,
    results_file: Path | None = None,
    shootouts_file: Path | None = None,
    raw_root: Path | None = None,
    output_path: Path | None = None,
    quarantine_path: Path | None = None,
    report_path: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> HistoricalIngestResult:
    """Run fetch, snapshot, parse, normalize, validate, deduplicate, write, and report stages."""

    if limit is not None and not dry_run:
        msg = "limit is only allowed with dry_run=true to avoid writing partial datasets"
        raise HistoricalIngestError(msg)

    effective_raw_root = raw_root or config.raw_root
    fetched_files = _fetch_source_files(
        config,
        local_files={
            "results": results_file,
            "shootouts": shootouts_file,
        },
    )
    snapshots = _snapshot_source_files(
        fetched_files=fetched_files,
        source_revision=config.source_revision,
        retrieved_at_utc=config.snapshot_retrieved_at_utc,
        raw_root=effective_raw_root,
        write_snapshots=not dry_run,
    )
    results_snapshot = snapshots["results"]
    shootouts_snapshot = snapshots["shootouts"]

    retrieved_at_utc = config.snapshot_retrieved_at_utc
    result_parse = _parse_csv_snapshot(
        results_snapshot,
        file_config=config.files["results"],
        source_revision=config.source_revision,
    )
    result_rows = list(result_parse.rows)
    processing_rows = result_rows[:limit] if limit is not None else result_rows
    shootout_parse = _parse_csv_snapshot(
        shootouts_snapshot,
        file_config=config.files["shootouts"],
        source_revision=config.source_revision,
    )
    shootout_rows = list(shootout_parse.rows)
    result_keys = {_date_team_key(row.payload) for row in result_rows}
    shootout_keys, shootout_invalid_records = _validate_shootout_rows(
        shootout_rows,
        result_keys=result_keys,
    )
    aliases = load_team_aliases(config.aliases_path)

    deduplicated_rows, duplicate_records = _deduplicate_rows(processing_rows)
    matches: list[CanonicalMatch] = []
    invalid_records: list[InvalidHistoricalRecord] = [
        *result_parse.invalid_records,
        *shootout_parse.invalid_records,
        *shootout_invalid_records,
        *duplicate_records,
    ]
    for row in deduplicated_rows:
        source_match_id = _source_match_id(row.payload)
        try:
            if _date_team_key(row.payload) in shootout_keys:
                msg = (
                    "shootout row lacks 90-minute score and penalty goals required by "
                    "the canonical contract"
                )
                raise ValueError(msg)
            matches.append(
                _row_to_canonical_match(
                    row.payload,
                    aliases=aliases,
                    retrieved_at_utc=retrieved_at_utc,
                )
            )
        except (ValueError, ValidationError) as exc:
            invalid_records.append(
                _invalid_record(
                    row,
                    source_match_id=source_match_id,
                    stage="normalize",
                    reason=str(exc),
                )
            )

    valid_matches = validate_match_records(matches, team_aliases=aliases)
    sorted_matches = tuple(
        sorted(valid_matches, key=lambda match: (match.match_date, match.match_id))
    )

    effective_output_path = output_path or config.processed_output
    effective_quarantine_path = quarantine_path or config.quarantine_output
    effective_report_path = report_path or config.report_output
    report = HistoricalIngestReport(
        source=SOURCE_NAME,
        source_homepage=config.homepage,
        source_license=config.license,
        retrieved_at_utc=retrieved_at_utc,
        results_rows_downloaded=result_parse.row_count,
        results_rows_processed=len(processing_rows),
        shootout_rows_downloaded=shootout_parse.row_count,
        valid_rows=len(sorted_matches),
        invalid_rows=len(invalid_records),
        duplicate_rows=len(duplicate_records),
        normalized_rows=len(sorted_matches),
        snapshot_manifests=(results_snapshot.manifest, shootouts_snapshot.manifest),
        output_path=None if dry_run else effective_output_path,
        quarantine_path=None if dry_run else effective_quarantine_path,
        dry_run=dry_run,
    )

    if not dry_run:
        write_matches_parquet(sorted_matches, effective_output_path)
        write_invalid_records(invalid_records, effective_quarantine_path)
        write_report(report, effective_report_path)

    return HistoricalIngestResult(
        matches=sorted_matches,
        invalid_records=tuple(invalid_records),
        report=report,
    )


def load_team_aliases(path: Path) -> list[TeamAlias]:
    """Load explicit team aliases from the repository static CSV."""

    aliases: list[TeamAlias] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                aliases.append(
                    TeamAlias.model_validate(
                        {
                            "canonical_team_id": row["canonical_team_id"],
                            "canonical_name": row["canonical_name"],
                            "source": row["source"],
                            "source_name": row["source_name"],
                            "valid_from": row["valid_from"] or None,
                            "valid_to": row["valid_to"] or None,
                        }
                    )
                )
    except OSError as exc:
        msg = f"failed to read team aliases {path}: {exc}"
        raise HistoricalIngestError(msg) from exc
    except (KeyError, ValidationError) as exc:
        msg = f"team aliases file {path} is invalid: {exc}"
        raise HistoricalIngestError(msg) from exc
    return aliases


def write_matches_parquet(matches: Iterable[CanonicalMatch], path: Path) -> None:
    """Write canonical matches to Parquet using stable column names."""

    rows = [match.model_dump(mode="python") for match in matches]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_canonical_parquet_schema())
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def write_invalid_records(records: Iterable[InvalidHistoricalRecord], path: Path) -> None:
    """Write quarantined source rows as JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def write_report(report: HistoricalIngestReport, path: Path) -> None:
    """Write the quality report as deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fetch_source_files(
    config: HistoricalCsvSourceConfig,
    *,
    local_files: Mapping[Literal["results", "shootouts"], Path | None],
) -> dict[Literal["results", "shootouts"], FetchedSourceFile]:
    fetched_files: dict[Literal["results", "shootouts"], FetchedSourceFile] = {}
    for file_name in SOURCE_FILE_NAMES:
        file_config = config.files[file_name]
        local_file = local_files[file_name]
        if local_file is not None:
            try:
                content = local_file.read_bytes()
            except OSError as exc:
                msg = f"failed to read local source file {local_file}: {exc}"
                raise HistoricalIngestError(msg) from exc
            input_uri = local_file.resolve().as_uri()
        else:
            content = _download_bytes(
                file_config.url,
                timeout_seconds=config.timeout_seconds,
                retries=config.retries,
            )
            input_uri = file_config.url
        fetched_files[file_name] = FetchedSourceFile(
            file_config=file_config,
            content=content,
            input_uri=input_uri,
        )
    return fetched_files


def _snapshot_source_files(
    *,
    fetched_files: Mapping[Literal["results", "shootouts"], FetchedSourceFile],
    source_revision: str,
    retrieved_at_utc: datetime,
    raw_root: Path,
    write_snapshots: bool,
) -> dict[Literal["results", "shootouts"], RawSnapshot]:
    bundle_key = _snapshot_bundle_key(fetched_files, source_revision=source_revision)
    snapshot_dir = raw_root / SOURCE_NAME / _snapshot_dir_name(retrieved_at_utc)
    manifests: dict[Literal["results", "shootouts"], RawSnapshotManifest] = {}
    for file_name in SOURCE_FILE_NAMES:
        fetched_file = fetched_files[file_name]
        file_config = fetched_file.file_config
        manifests[file_name] = RawSnapshotManifest(
            source=SOURCE_NAME,
            logical_uri=file_config.logical_uri,
            source_revision=source_revision,
            retrieved_at_utc=retrieved_at_utc,
            content_sha256=sha256_bytes(fetched_file.content),
            cache_key=bundle_key,
            raw_path=snapshot_dir / file_config.filename,
            input_uri=fetched_file.input_uri,
        )

    if write_snapshots:
        manifests = _write_or_validate_snapshot_bundle(manifests, fetched_files)

    return {
        file_name: RawSnapshot(
            manifest=manifests[file_name],
            content=fetched_files[file_name].content,
        )
        for file_name in SOURCE_FILE_NAMES
    }


def _write_or_validate_snapshot_bundle(
    manifests: Mapping[Literal["results", "shootouts"], RawSnapshotManifest],
    fetched_files: Mapping[Literal["results", "shootouts"], FetchedSourceFile],
) -> dict[Literal["results", "shootouts"], RawSnapshotManifest]:
    first_manifest = manifests["results"]
    snapshot_dir = first_manifest.raw_path.parent
    existing_pairs = {
        file_name: (
            manifest.raw_path.exists(),
            _manifest_path(manifest.raw_path).exists(),
        )
        for file_name, manifest in manifests.items()
    }
    for file_name, (raw_exists, manifest_exists) in existing_pairs.items():
        if raw_exists != manifest_exists:
            missing = (
                _manifest_path(manifests[file_name].raw_path)
                if raw_exists
                else manifests[file_name].raw_path
            )
            msg = f"incomplete immutable snapshot; missing paired file: {missing}"
            raise HistoricalIngestError(msg)

    any_existing = any(
        raw_exists or manifest_exists for raw_exists, manifest_exists in existing_pairs.values()
    )
    all_existing = all(
        raw_exists and manifest_exists for raw_exists, manifest_exists in existing_pairs.values()
    )
    if any_existing and not all_existing:
        missing_files = [
            str(path)
            for file_name, manifest in manifests.items()
            for path in (manifest.raw_path, _manifest_path(manifest.raw_path))
            if not path.exists()
        ]
        msg = "incomplete immutable snapshot bundle; missing files: " + ", ".join(missing_files)
        raise HistoricalIngestError(msg)

    if all_existing:
        existing_manifests: dict[Literal["results", "shootouts"], RawSnapshotManifest] = {}
        for file_name, manifest in manifests.items():
            existing_content = manifest.raw_path.read_bytes()
            if existing_content != fetched_files[file_name].content:
                msg = f"snapshot path collision with different content: {manifest.raw_path}"
                raise HistoricalIngestError(msg)
            existing_manifest = _read_existing_manifest(_manifest_path(manifest.raw_path))
            _validate_existing_manifest(
                existing_manifest,
                expected_logical_uri=manifest.logical_uri,
                expected_source_revision=manifest.source_revision,
                expected_retrieved_at_utc=manifest.retrieved_at_utc,
                expected_content_sha256=manifest.content_sha256,
                expected_cache_key=manifest.cache_key,
                expected_raw_path=manifest.raw_path,
            )
            existing_manifests[file_name] = existing_manifest
        return existing_manifests

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for file_name, manifest in manifests.items():
        manifest.raw_path.write_bytes(fetched_files[file_name].content)
        _manifest_path(manifest.raw_path).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return dict(manifests)


def _snapshot_bundle_key(
    fetched_files: Mapping[Literal["results", "shootouts"], FetchedSourceFile],
    *,
    source_revision: str,
) -> str:
    identity_parts = [SOURCE_NAME, source_revision, MATCH_SCHEMA_VERSION]
    for file_name in SOURCE_FILE_NAMES:
        fetched_file = fetched_files[file_name]
        identity_parts.extend(
            [
                file_name,
                fetched_file.file_config.logical_uri,
                sha256_bytes(fetched_file.content),
            ]
        )
    return sha256_bytes("\n".join(identity_parts).encode())[:16]


def _snapshot_dir_name(retrieved_at_utc: datetime) -> str:
    return retrieved_at_utc.isoformat().replace("+00:00", "Z")


def _manifest_path(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.name}.manifest.json")


def _read_existing_manifest(path: Path) -> RawSnapshotManifest:
    try:
        return RawSnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"failed to read snapshot manifest {path}: {exc}"
        raise HistoricalIngestError(msg) from exc
    except ValidationError as exc:
        msg = f"snapshot manifest {path} is invalid: {exc}"
        raise HistoricalIngestError(msg) from exc


def _validate_existing_manifest(
    manifest: RawSnapshotManifest,
    *,
    expected_logical_uri: str,
    expected_source_revision: str,
    expected_retrieved_at_utc: datetime,
    expected_content_sha256: str,
    expected_cache_key: str,
    expected_raw_path: Path,
) -> None:
    expected_values = {
        "source": SOURCE_NAME,
        "logical_uri": expected_logical_uri,
        "source_revision": expected_source_revision,
        "retrieved_at_utc": expected_retrieved_at_utc,
        "content_sha256": expected_content_sha256,
        "cache_key": expected_cache_key,
        "raw_path": expected_raw_path,
    }
    for field_name, expected_value in expected_values.items():
        actual_value = getattr(manifest, field_name)
        if actual_value != expected_value:
            msg = (
                f"snapshot manifest mismatch for {expected_raw_path}: {field_name} "
                f"is {actual_value!r}, expected {expected_value!r}"
            )
            raise HistoricalIngestError(msg)


def _parse_csv_snapshot(
    snapshot: RawSnapshot,
    *,
    file_config: SourceFileConfig,
    source_revision: str,
) -> ParsedCsvSnapshot:
    try:
        text = snapshot.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        msg = f"{snapshot.manifest.raw_path} is not valid UTF-8 CSV: {exc}"
        raise HistoricalIngestError(msg) from exc
    reader = csv.DictReader(text.splitlines())
    expected = tuple(file_config.expected_columns)
    fieldnames = tuple(reader.fieldnames or ())
    missing = [column for column in expected if column not in fieldnames]
    if missing:
        missing_columns = ", ".join(missing)
        msg = f"{snapshot.manifest.raw_path} is missing required columns: {missing_columns}"
        raise HistoricalIngestError(msg)

    rows: list[ParsedHistoricalRow] = []
    invalid_records: list[InvalidHistoricalRecord] = []
    row_count = 0
    for row_number, row in enumerate(reader, start=2):
        row_count += 1
        row_data: Mapping[object, object] = row
        payload = {column: _strip_csv_value(row_data.get(column)) for column in expected}
        parsed_row = ParsedHistoricalRow(
            source_file=file_config.filename,
            logical_uri=file_config.logical_uri,
            source_revision=source_revision,
            row_number=row_number,
            payload=payload,
        )
        extra_fields = row_data.get(None)
        missing_row_columns = [column for column in expected if row_data.get(column) is None]
        if extra_fields is not None or missing_row_columns:
            invalid_payload: dict[str, object] = dict(payload)
            reasons: list[str] = []
            if extra_fields is not None:
                invalid_payload["_extra_fields"] = extra_fields
                reasons.append("extra CSV fields")
            if missing_row_columns:
                invalid_payload["_missing_columns"] = missing_row_columns
                reasons.append("missing CSV fields")
            invalid_records.append(
                _invalid_record(
                    ParsedHistoricalRow(
                        source_file=file_config.filename,
                        logical_uri=file_config.logical_uri,
                        source_revision=source_revision,
                        row_number=row_number,
                        payload=invalid_payload,
                    ),
                    source_match_id=_source_match_id_or_none(payload),
                    stage="parse",
                    reason=", ".join(reasons),
                )
            )
            continue
        rows.append(parsed_row)
    return ParsedCsvSnapshot(
        row_count=row_count,
        rows=tuple(rows),
        invalid_records=tuple(invalid_records),
    )


def _invalid_record(
    row: ParsedHistoricalRow,
    *,
    source_match_id: str | None,
    stage: str,
    reason: str,
) -> InvalidHistoricalRecord:
    return InvalidHistoricalRecord(
        source=SOURCE_NAME,
        source_file=row.source_file,
        logical_uri=row.logical_uri,
        source_revision=row.source_revision,
        row_number=row.row_number,
        stage=stage,
        source_match_id=source_match_id,
        reason=reason,
        payload=row.payload,
    )


def _download_bytes(url: str, *, timeout_seconds: float, retries: int) -> bytes:
    last_error: Exception | None = None
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.content
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == attempts:
                break
    msg = f"failed to download {url!r} after {attempts} attempts: {last_error}"
    raise HistoricalIngestError(msg)


def _validate_shootout_rows(
    rows: Iterable[ParsedHistoricalRow],
    *,
    result_keys: set[tuple[str, str, str]],
) -> tuple[set[tuple[str, str, str]], list[InvalidHistoricalRecord]]:
    rows_by_key: dict[tuple[str, str, str], list[ParsedHistoricalRow]] = {}
    for row in rows:
        rows_by_key.setdefault(_date_team_key(row.payload), []).append(row)

    shootout_keys: set[tuple[str, str, str]] = set()
    invalid_records: list[InvalidHistoricalRecord] = []
    for key, key_rows in rows_by_key.items():
        if key in result_keys:
            shootout_keys.add(key)
            reason = (
                "shootout auxiliary row marks a result that lacks 90-minute score and "
                "penalty goals required by the canonical contract"
            )
        else:
            reason = "shootout auxiliary row has no matching results.csv row"

        if len(key_rows) > 1:
            duplicate_reason = "duplicate shootout row for date/home/away key"
            if key not in result_keys:
                duplicate_reason += "; no matching results.csv row"
            for row in key_rows:
                invalid_records.append(
                    _invalid_record(
                        row,
                        source_match_id=None,
                        stage="shootout_validate",
                        reason=duplicate_reason,
                    )
                )
            continue

        if key not in result_keys:
            for row in key_rows:
                invalid_records.append(
                    _invalid_record(
                        row,
                        source_match_id=None,
                        stage="shootout_validate",
                        reason=reason,
                    )
                )
            continue

        for row in key_rows:
            invalid_records.append(
                _invalid_record(
                    row,
                    source_match_id=None,
                    stage="shootout_validate",
                    reason=reason,
                )
            )

    return shootout_keys, invalid_records


def _deduplicate_rows(
    rows: Iterable[ParsedHistoricalRow],
) -> tuple[list[ParsedHistoricalRow], list[InvalidHistoricalRecord]]:
    rows_by_source_id: dict[str, list[ParsedHistoricalRow]] = {}
    source_id_order: list[str] = []
    for row in rows:
        source_match_id = _source_match_id(row.payload)
        if source_match_id not in rows_by_source_id:
            rows_by_source_id[source_match_id] = []
            source_id_order.append(source_match_id)
        rows_by_source_id[source_match_id].append(row)

    deduplicated_rows: list[ParsedHistoricalRow] = []
    duplicates: list[InvalidHistoricalRecord] = []
    for source_match_id in source_id_order:
        key_rows = rows_by_source_id[source_match_id]
        if len(key_rows) == 1:
            deduplicated_rows.append(key_rows[0])
            continue
        first_payload = key_rows[0].payload
        if all(row.payload == first_payload for row in key_rows):
            deduplicated_rows.append(key_rows[0])
            for row in key_rows[1:]:
                duplicates.append(
                    _invalid_record(
                        row,
                        source_match_id=source_match_id,
                        stage="deduplicate",
                        reason=(
                            "duplicate source_match_id in results snapshot with identical payload"
                        ),
                    )
                )
            continue

        for row in key_rows:
            duplicates.append(
                _invalid_record(
                    row,
                    source_match_id=source_match_id,
                    stage="deduplicate",
                    reason="conflicting duplicate source_match_id in results snapshot",
                )
            )
    return deduplicated_rows, duplicates


def _row_to_canonical_match(
    row: Mapping[str, object],
    *,
    aliases: Iterable[TeamAlias],
    retrieved_at_utc: datetime,
) -> CanonicalMatch:
    match_date = _parse_match_date(row["date"])
    home_name = _require_text(row, "home_team")
    away_name = _require_text(row, "away_team")
    source_match_id = _source_match_id(row)
    home_goals = _parse_non_negative_int(row["home_score"], field_name="home_score")
    away_goals = _parse_non_negative_int(row["away_score"], field_name="away_score")
    neutral_site = _parse_bool(row["neutral"], field_name="neutral")
    host_country = _nullable_text(row, "country")

    return CanonicalMatch(
        match_id=f"{SOURCE_NAME}-{source_match_id}",
        match_status=MatchStatus.PLAYED,
        match_date=match_date,
        kickoff_utc=None,
        kickoff_local_time=None,
        kickoff_timezone=None,
        kickoff_time_status=KickoffTimeStatus.DATE_ONLY,
        home_team_name_original=home_name,
        away_team_name_original=away_name,
        home_team_id=resolve_team_alias(
            source=SOURCE_NAME,
            source_name=home_name,
            match_date=match_date,
            aliases=aliases,
        ),
        away_team_id=resolve_team_alias(
            source=SOURCE_NAME,
            source_name=away_name,
            match_date=match_date,
            aliases=aliases,
        ),
        home_goals_90=home_goals,
        away_goals_90=away_goals,
        result_90=_result_from_scores(home_goals, away_goals),
        extra_time_played=False,
        home_goals_after_extra_time=None,
        away_goals_after_extra_time=None,
        penalty_shootout=False,
        home_penalty_goals=None,
        away_penalty_goals=None,
        competition=_require_text(row, "tournament"),
        stage=None,
        match_type=_match_type(_require_text(row, "tournament")),
        city=_nullable_text(row, "city"),
        host_country=host_country,
        venue_name_original=None,
        neutral_site=neutral_site,
        home_advantage_status=_home_advantage_status(
            neutral_site=neutral_site,
            host_country=host_country,
            home_name=home_name,
            away_name=away_name,
        ),
        source=SOURCE_NAME,
        source_match_id=source_match_id,
        retrieved_at_utc=retrieved_at_utc,
    )


def _source_match_id(row: Mapping[str, object]) -> str:
    identity_fields = (
        "date",
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country",
    )
    identity = "\n".join(str(row.get(field, "")).strip() for field in identity_fields)
    return sha256_bytes(identity.encode())[:20]


def _source_match_id_or_none(row: Mapping[str, object]) -> str | None:
    required_identity_fields = ("date", "home_team", "away_team")
    if any(not str(row.get(field, "")).strip() for field in required_identity_fields):
        return None
    return _source_match_id(row)


def _date_team_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("date", "")).strip(),
        str(row.get("home_team", "")).strip(),
        str(row.get("away_team", "")).strip(),
    )


def _strip_csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(item) for item in value).strip()
    return str(value).strip()


def _parse_match_date(value: object) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        msg = f"invalid date value {value!r}"
        raise ValueError(msg) from exc


def _require_text(row: Mapping[str, object], field_name: str) -> str:
    value = str(row.get(field_name, "")).strip()
    if not value:
        msg = f"missing required source field {field_name}"
        raise ValueError(msg)
    return value


def _nullable_text(row: Mapping[str, object], field_name: str) -> str | None:
    value = str(row.get(field_name, "")).strip()
    return value or None


def _parse_non_negative_int(value: object, *, field_name: str) -> int:
    try:
        parsed = int(str(value))
    except ValueError as exc:
        msg = f"invalid integer field {field_name}: {value!r}"
        raise ValueError(msg) from exc
    if parsed < 0:
        msg = f"negative integer field {field_name}: {value!r}"
        raise ValueError(msg)
    return parsed


def _parse_bool(value: object, *, field_name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    msg = f"invalid boolean field {field_name}: {value!r}"
    raise ValueError(msg)


def _result_from_scores(home_goals: int, away_goals: int) -> Result90:
    if home_goals > away_goals:
        return Result90.HOME_WIN
    if home_goals < away_goals:
        return Result90.AWAY_WIN
    return Result90.DRAW


def _match_type(tournament: str) -> MatchType:
    normalized = tournament.casefold()
    if "friendly" in normalized:
        return MatchType.FRIENDLY
    if normalized == "fifa world cup":
        return MatchType.WORLD_CUP
    if "qualification" in normalized or "qualifier" in normalized:
        return MatchType.QUALIFIER
    continental_terms = (
        "afc asian cup",
        "africa cup of nations",
        "copa america",
        "concacaf gold cup",
        "ofc nations cup",
        "uefa euro",
    )
    if any(term in normalized for term in continental_terms):
        return MatchType.CONTINENTAL_TOURNAMENT
    return MatchType.OTHER


def _home_advantage_status(
    *,
    neutral_site: bool,
    host_country: str | None,
    home_name: str,
    away_name: str,
) -> HomeAdvantageStatus:
    if neutral_site:
        return HomeAdvantageStatus.NEUTRAL
    if host_country == home_name:
        return HomeAdvantageStatus.HOME_TEAM
    if host_country == away_name:
        return HomeAdvantageStatus.AWAY_TEAM
    return HomeAdvantageStatus.UNKNOWN


def _canonical_parquet_schema() -> pa.Schema:
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
        ]
    )

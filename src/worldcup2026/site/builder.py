"""Build a static, read-only dashboard from the public predictions-data branch."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SITE_SCHEMA_VERSION = "wc2026_static_site_build_v1"
PUBLIC_MANIFEST_SCHEMA = "prediction_publication_manifest_v1"
OFFICIAL_MODEL_VERSION = "poisson_goal_v1"
OFFICIAL_CONTEXT = "early_v1"
SHADOW_CONTEXT = "shadow_contextual_v1"
REPO_DOCS_URL = "https://github.com/juanbrau-or/world-cup-2026-predictions/blob/main"
ASSET_ROOT = Path(__file__).resolve().parent / "assets"
SECRET_PATTERN = re.compile(
    r"(?i)\b(football_data_api_key|api_football_key|api[-_ ]?key|authorization|bearer|token)"
    r"\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
)


class SiteBuildError(RuntimeError):
    """Raised when the static site cannot be safely generated."""


@dataclass(frozen=True)
class SiteBuildResult:
    """Summary of a deterministic static-site build."""

    output_root: Path
    manifest_path: Path
    checksum_report_path: Path
    coverage_report_path: Path
    page_count: int
    site_checksum: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SiteData:
    """Validated public data used by the renderer."""

    manifest: Mapping[str, Any]
    latest: Mapping[str, Any]
    latest_rows: tuple[Mapping[str, str], ...]
    upcoming_markdown: str
    scorecard: Mapping[str, Any]
    prospective_matches: tuple[Mapping[str, str], ...]
    shadow_manifest: Mapping[str, Any] | None
    shadow_latest: Mapping[str, Any] | None
    shadow_rows: tuple[Mapping[str, str], ...]
    shadow_scorecard: Mapping[str, Any] | None
    shadow_comparison_markdown: str | None
    simulation_manifest: Mapping[str, Any] | None
    simulation_teams: tuple[Mapping[str, Any], ...]
    group_rows: tuple[Mapping[str, str], ...]
    bracket_rows: tuple[Mapping[str, str], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Page:
    """One rendered page in the static site."""

    key: str
    title: str
    output_path: Path
    html: str


def build_site(*, data_root: Path, output_root: Path) -> SiteBuildResult:
    """Build the static dashboard from a checked-out ``predictions-data`` tree."""

    source = _load_site_data(data_root)
    pages = _render_pages(source)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    _copy_assets(output_root)
    for page in pages:
        path = output_root / page.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.html, encoding="utf-8")

    page_coverage = _page_coverage(pages, source)
    coverage_report = {
        "schema_version": "wc2026_site_page_coverage_v1",
        "pages": page_coverage,
    }
    (output_root / "page-coverage.json").write_text(
        _json_dumps(coverage_report),
        encoding="utf-8",
    )
    fallback_audit = _fallback_audit(source)
    (output_root / "simulation-fallback-audit.json").write_text(
        _json_dumps(fallback_audit),
        encoding="utf-8",
    )
    checksums = _site_file_checksums(
        output_root,
        exclude={"build-manifest.json", "checksum-report.json"},
    )
    site_checksum = _site_checksum(checksums)
    manifest = {
        "schema_version": SITE_SCHEMA_VERSION,
        "source_manifest_checksum": _sha256((data_root / "manifest.json").read_bytes()),
        "source_data_cutoff": source.manifest.get("data_cutoff"),
        "source_generated_at": source.manifest.get("generated_at"),
        "official_model": source.manifest.get("model"),
        "shadow_model": (
            None if source.shadow_manifest is None else source.shadow_manifest.get("model")
        ),
        "simulation_run_id": (
            None
            if source.simulation_manifest is None
            else source.simulation_manifest.get("simulation_run_id")
        ),
        "pages": [page.output_path.as_posix() for page in pages],
        "warnings": list(source.warnings),
        "site_checksum": site_checksum,
    }
    checksum_report = {
        "schema_version": "wc2026_site_checksum_report_v1",
        "site_checksum": site_checksum,
        "files": checksums,
    }
    (output_root / "build-manifest.json").write_text(
        _json_dumps(manifest),
        encoding="utf-8",
    )
    (output_root / "checksum-report.json").write_text(
        _json_dumps(checksum_report),
        encoding="utf-8",
    )
    return SiteBuildResult(
        output_root=output_root,
        manifest_path=output_root / "build-manifest.json",
        checksum_report_path=output_root / "checksum-report.json",
        coverage_report_path=output_root / "page-coverage.json",
        page_count=len(pages),
        site_checksum=site_checksum,
        warnings=source.warnings,
    )


def _load_site_data(data_root: Path) -> SiteData:
    if not data_root.is_dir():
        raise SiteBuildError(f"data root is missing: {data_root}")
    _validate_public_tree(data_root)
    manifest = _read_json_object(data_root / "manifest.json")
    if manifest.get("schema_version") != PUBLIC_MANIFEST_SCHEMA:
        raise SiteBuildError("manifest has unexpected schema_version")
    _validate_manifest_files(data_root, manifest)
    latest = _read_json_object(data_root / "latest.json")
    latest_rows = tuple(_read_csv_rows(data_root / "latest.csv"))
    _validate_latest_payload(latest, latest_rows, shadow=False)
    scorecard = _read_json_object(data_root / "prospective_scorecard.json")
    _validate_scorecard(scorecard, field_name="prospective_scorecard.json")
    prospective_matches = tuple(_read_csv_rows(data_root / "prospective_matches.csv"))
    _validate_markdown_file(data_root / "upcoming.md")

    shadow_manifest: Mapping[str, Any] | None = None
    shadow_latest: Mapping[str, Any] | None = None
    shadow_rows: tuple[Mapping[str, str], ...] = ()
    shadow_scorecard: Mapping[str, Any] | None = None
    shadow_comparison: str | None = None
    if (data_root / "shadow" / "manifest.json").is_file():
        shadow_manifest = _read_json_object(data_root / "shadow" / "manifest.json")
        shadow_latest = _read_json_object(data_root / "shadow" / "contextual_latest.json")
        shadow_rows = tuple(_read_csv_rows(data_root / "shadow" / "contextual_latest.csv"))
        _validate_latest_payload(shadow_latest, shadow_rows, shadow=True)
        shadow_scorecard = _read_json_object(data_root / "shadow" / "contextual_scorecard.json")
        _validate_scorecard(shadow_scorecard, field_name="shadow/contextual_scorecard.json")
        shadow_comparison = (data_root / "shadow" / "contextual_comparison.md").read_text(
            encoding="utf-8"
        )

    simulation_manifest: Mapping[str, Any] | None = None
    simulation_teams: tuple[Mapping[str, Any], ...] = ()
    group_rows: tuple[Mapping[str, str], ...] = ()
    bracket_rows: tuple[Mapping[str, str], ...] = ()
    if (data_root / "simulation" / "manifest.json").is_file():
        simulation_manifest = _read_json_object(data_root / "simulation" / "manifest.json")
        _validate_simulation_manifest(simulation_manifest)
        team_payload = _read_json_object(data_root / "simulation" / "team_probabilities.json")
        simulation_teams = tuple(_team_probability_rows(team_payload))
        _validate_simulation_probabilities(simulation_teams)
        group_rows = tuple(
            _markdown_table_rows(
                (data_root / "simulation" / "group_tables_summary.md").read_text(
                    encoding="utf-8"
                )
            )
        )
        bracket_rows = tuple(
            _markdown_table_rows(
                (data_root / "simulation" / "bracket_summary.md").read_text(encoding="utf-8")
            )
        )

    warnings = tuple(
        warning
        for warning in (
            _small_sample_warning(scorecard),
            _fallback_warning(simulation_manifest),
            _missing_shadow_warning(shadow_manifest),
            _missing_simulation_warning(simulation_manifest),
        )
        if warning is not None
    )
    return SiteData(
        manifest=manifest,
        latest=latest,
        latest_rows=latest_rows,
        upcoming_markdown=(data_root / "upcoming.md").read_text(encoding="utf-8"),
        scorecard=scorecard,
        prospective_matches=prospective_matches,
        shadow_manifest=shadow_manifest,
        shadow_latest=shadow_latest,
        shadow_rows=shadow_rows,
        shadow_scorecard=shadow_scorecard,
        shadow_comparison_markdown=shadow_comparison,
        simulation_manifest=simulation_manifest,
        simulation_teams=simulation_teams,
        group_rows=group_rows,
        bracket_rows=bracket_rows,
        warnings=warnings,
    )


def _validate_public_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative_path = path.relative_to(root)
        _assert_allowed_public_path(relative_path, path.stat().st_size)
        text = _file_text_for_secret_scan(path)
        _assert_no_secret_string(relative_path.as_posix(), text)


def _assert_allowed_public_path(relative_path: Path, size_bytes: int) -> None:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SiteBuildError(f"unsafe public-data path: {relative_path}")
    if any(part in {".env", "raw", "models"} for part in relative_path.parts):
        raise SiteBuildError(f"raw/model/env paths are not allowed in site data: {relative_path}")
    if relative_path.suffix == ".parquet" or ".parquet" in relative_path.suffixes:
        raise SiteBuildError(f"Parquet is not allowed in site data: {relative_path}")
    if size_bytes > 2_000_000:
        raise SiteBuildError(f"public-data file is too large for site build: {relative_path}")
    exact = {
        "latest.csv",
        "latest.json",
        "upcoming.md",
        "prospective_scorecard.json",
        "prospective_scorecard.md",
        "prospective_matches.csv",
        "manifest.json",
        "shadow/contextual_latest.csv",
        "shadow/contextual_latest.json",
        "shadow/contextual_upcoming.md",
        "shadow/contextual_scorecard.json",
        "shadow/contextual_scorecard.md",
        "shadow/contextual_comparison.md",
        "shadow/manifest.json",
        "simulation/manifest.json",
        "simulation/team_probabilities.csv",
        "simulation/team_probabilities.json",
        "simulation/champion_probabilities.md",
        "simulation/round_probabilities.md",
        "simulation/group_tables_summary.md",
        "simulation/bracket_summary.md",
    }
    normalized = relative_path.as_posix()
    if normalized in exact:
        return
    if (
        len(relative_path.parts) == 2
        and relative_path.parts[0] == "history"
        and relative_path.name.endswith(".csv.gz")
    ):
        return
    raise SiteBuildError(f"path is not allowed for site data: {relative_path}")


def _validate_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    checksums = manifest.get("checksums")
    if not isinstance(checksums, Mapping):
        raise SiteBuildError("manifest.checksums must be an object")
    published_files = manifest.get("published_files")
    if not isinstance(published_files, list):
        raise SiteBuildError("manifest.published_files must be a list")
    for item in published_files:
        if not isinstance(item, str):
            raise SiteBuildError("manifest.published_files must contain strings")
        relative_path = Path(item)
        _assert_allowed_public_path(relative_path, 0)
        path = root / relative_path
        if not path.is_file():
            raise SiteBuildError(f"published file is missing: {item}")
        expected = checksums.get(item)
        if not isinstance(expected, str) or _sha256(path.read_bytes()) != expected:
            raise SiteBuildError(f"checksum mismatch for published file: {item}")
    required = (
        "latest.csv",
        "latest.json",
        "upcoming.md",
        "prospective_scorecard.json",
        "prospective_matches.csv",
        "manifest.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise SiteBuildError(f"required site data file is missing: {name}")
    _parse_utc(_required_str(manifest, "generated_at"))
    _parse_utc(_required_str(manifest, "data_cutoff"))


def _validate_latest_payload(
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    shadow: bool,
) -> None:
    expected_schema = "shadow_contextual_latest_v1" if shadow else "predictions_latest_v1"
    if payload.get("schema_version") != expected_schema:
        raise SiteBuildError(f"latest payload has unexpected schema: {expected_schema}")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise SiteBuildError("latest payload is missing model metadata")
    model_version = str(model.get("version") or "")
    model_family = str(model.get("family") or "")
    if shadow:
        if model_version == OFFICIAL_MODEL_VERSION or model_family == "poisson":
            raise SiteBuildError("shadow predictions appear to contain official model metadata")
    elif model_version != OFFICIAL_MODEL_VERSION:
        raise SiteBuildError("official latest predictions must use poisson_goal_v1")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise SiteBuildError("latest payload predictions must be a list")
    if len(predictions) != len(rows):
        raise SiteBuildError("latest JSON and CSV prediction counts differ")
    for row in rows:
        _validate_prediction_row(row, shadow=shadow)


def _validate_prediction_row(row: Mapping[str, str], *, shadow: bool) -> None:
    fixture_id = row.get("source_fixture_id") or "<unknown>"
    probabilities = (
        _probability(row.get("probability_home_win"), field="probability_home_win"),
        _probability(row.get("probability_draw"), field="probability_draw"),
        _probability(row.get("probability_away_win"), field="probability_away_win"),
    )
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        raise SiteBuildError(f"probabilities do not sum to 1 for {fixture_id}")
    _parse_utc(_required_row_str(row, "kickoff_utc", fixture_id=fixture_id))
    _parse_utc(_required_row_str(row, "data_cutoff_utc", fixture_id=fixture_id))
    if row.get("prediction_created_at_utc"):
        _parse_utc(str(row["prediction_created_at_utc"]))
    context = row.get("prediction_context")
    model_version = row.get("model_version")
    model_family = row.get("model_family")
    if shadow:
        if context != SHADOW_CONTEXT or model_version == OFFICIAL_MODEL_VERSION:
            raise SiteBuildError(f"shadow row is mixed with official data for {fixture_id}")
    elif context == SHADOW_CONTEXT or model_family == "contextual_challenger":
        raise SiteBuildError(f"official row is mixed with shadow data for {fixture_id}")


def _validate_scorecard(payload: Mapping[str, Any], *, field_name: str) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SiteBuildError(f"{field_name} is missing metrics")
    matches = metrics.get("matches")
    if not isinstance(matches, int) or isinstance(matches, bool) or matches < 0:
        raise SiteBuildError(f"{field_name} metrics.matches must be non-negative")
    for field in ("generated_at_utc", "results_cutoff_utc"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            _parse_utc(value)
    for item in payload.get("matches", []):
        if isinstance(item, Mapping):
            for field in ("kickoff_utc", "data_cutoff_utc", "prediction_created_at_utc"):
                value = item.get(field)
                if isinstance(value, str) and value:
                    _parse_utc(value)


def _validate_simulation_manifest(payload: Mapping[str, Any]) -> None:
    required = ("simulation_run_id", "data_cutoff_utc", "runs", "seed", "model", "rules")
    for field in required:
        if field not in payload:
            raise SiteBuildError(f"simulation manifest is missing {field}")
    _parse_utc(_required_str(payload, "data_cutoff_utc"))
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("version") != OFFICIAL_MODEL_VERSION:
        raise SiteBuildError("simulation must use the official poisson_goal_v1 model")


def _team_probability_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("teams")
    if not isinstance(rows, list):
        raise SiteBuildError("simulation team probabilities must contain a teams list")
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SiteBuildError("simulation team row must be an object")
        output.append(row)
    return output


def _validate_simulation_probabilities(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        team = str(row.get("team_id") or "<unknown>")
        for field in (
            "round_of_32",
            "round_of_16",
            "quarter_final",
            "semi_final",
            "final",
            "champion",
        ):
            _probability(row.get(field), field=f"{team}.{field}")


def _render_pages(source: SiteData) -> tuple[Page, ...]:
    renderers: tuple[tuple[str, str, Path, Callable[[SiteData], str]], ...] = (
        ("home", "Inicio", Path("index.html"), _home_body),
        ("upcoming", "Próximos Partidos", Path("upcoming/index.html"), _upcoming_body),
        ("shadow", "Challenger Shadow", Path("shadow/index.html"), _shadow_body),
        ("evaluation", "Evaluación", Path("evaluation/index.html"), _evaluation_body),
        ("simulation", "Simulación", Path("simulation/index.html"), _simulation_body),
        ("groups", "Grupos", Path("groups/index.html"), _groups_body),
        ("bracket", "Bracket", Path("bracket/index.html"), _bracket_body),
        ("methodology", "Metodología", Path("methodology/index.html"), _methodology_body),
    )
    pages: list[Page] = []
    for key, title, path, renderer in renderers:
        prefix = "" if path.parent == Path(".") else "../"
        body = renderer(source)
        pages.append(
            Page(
                key=key,
                title=title,
                output_path=path,
                html=_layout(title=title, active=key, body=body, prefix=prefix, source=source),
            )
        )
    return tuple(pages)


def _layout(*, title: str, active: str, body: str, prefix: str, source: SiteData) -> str:
    nav_items = (
        ("home", "Inicio", ""),
        ("upcoming", "Partidos", "upcoming/"),
        ("shadow", "Shadow", "shadow/"),
        ("evaluation", "Evaluación", "evaluation/"),
        ("simulation", "Simulación", "simulation/"),
        ("groups", "Grupos", "groups/"),
        ("bracket", "Bracket", "bracket/"),
        ("methodology", "Metodología", "methodology/"),
    )
    nav = "\n".join(
        (
            f'<a href="{prefix}{href}" aria-current="page">{label}</a>'
            if key == active
            else f'<a href="{prefix}{href}">{label}</a>'
        )
        for key, label, href in nav_items
    )
    warning_html = "".join(
        f'<p class="notice">{_escape(warning)}</p>'
        for warning in source.warnings
    )
    return (
        "<!doctype html>\n"
        '<html lang="es">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{_escape(title)} | World Cup 2026 Predictions</title>\n"
        f'  <link rel="stylesheet" href="{prefix}assets/styles.css">\n'
        f'  <script src="{prefix}assets/site.js" defer></script>\n'
        "</head>\n"
        "<body>\n"
        '  <a class="skip-link" href="#contenido">Saltar al contenido</a>\n'
        "  <header>\n"
        '    <div class="brand">World Cup 2026 Predictions</div>\n'
        f"    <nav aria-label=\"Secciones\">{nav}</nav>\n"
        "  </header>\n"
        f"  <main id=\"contenido\">{warning_html}{body}</main>\n"
        "  <footer>\n"
        "    <span>Dashboard estático generado desde la rama pública "
        "<code>predictions-data</code>.</span>\n"
        "  </footer>\n"
        "</body>\n"
        "</html>\n"
    )


def _home_body(source: SiteData) -> str:
    manifest = source.manifest
    simulation = source.simulation_manifest
    favorite = _favorite_team(source.simulation_teams)
    freshness = _freshness_widget(str(manifest.get("data_cutoff") or ""))
    cards = [
        ("Última actualización UTC", _value(manifest.get("generated_at"))),
        ("Cutoff de datos", _value(manifest.get("data_cutoff"))),
        ("Frescura", freshness),
        ("Modelo oficial", _model_label(manifest.get("model"))),
        ("Challenger shadow", _shadow_status(source)),
        ("Política de evaluación", _value(manifest.get("prospective_policy_version"))),
        ("Próximos partidos", str(len(source.latest_rows))),
        ("Partidos oficiales evaluados", _value(manifest.get("prospective_observations"))),
        ("Simulaciones", _value(None if simulation is None else simulation.get("runs"))),
        ("Favorito al título", favorite),
    ]
    return (
        "<section class=\"hero\">"
        "<div>"
        "<p class=\"eyebrow\">Pronósticos probabilísticos del Mundial 2026</p>"
        "<h1>Predicciones públicas y simulación del torneo</h1>"
        "<p>Probabilidades calibradas, evaluación prospectiva y Monte Carlo oficial en modo "
        "solo lectura.</p>"
        "</div>"
        "</section>"
        f"{_summary_grid(cards)}"
        f"{_warning_panel(source)}"
    )


def _upcoming_body(source: SiteData) -> str:
    rows = sorted(source.latest_rows, key=lambda row: row.get("kickoff_utc", ""))
    if not rows:
        return _empty_section("Próximos partidos", "No hay próximos partidos publicados.")
    body_rows = []
    for row in rows:
        home = _team_name(row, "home")
        away = _team_name(row, "away")
        body_rows.append(
            "<article class=\"match-card\">"
            f"<h2>{_escape(home)} <span>vs</span> {_escape(away)}</h2>"
            f"<p><time datetime=\"{_escape(row.get('kickoff_utc', ''))}\" "
            f"data-local-time>{_escape(row.get('kickoff_utc', ''))}</time></p>"
            "<div class=\"probabilities\">"
            f"{_probability_bar('P(local)', row.get('probability_home_win'))}"
            f"{_probability_bar('P(empate)', row.get('probability_draw'))}"
            f"{_probability_bar('P(visitante)', row.get('probability_away_win'))}"
            "</div>"
            "<dl class=\"compact-list\">"
            f"<div><dt>Goles esperados</dt><dd>{_decimal(row.get('expected_home_goals'))} - "
            f"{_decimal(row.get('expected_away_goals'))}</dd></div>"
            f"<div><dt>Marcador modal</dt><dd>{_escape(row.get('modal_score') or 'n/a')} "
            "<span class=\"muted\">no es resultado seguro</span></dd></div>"
            "<div><dt>Horas de anticipación</dt><dd>"
            f"{_decimal(row.get('hours_before_kickoff'))}</dd></div>"
            f"<div><dt>Modelo</dt><dd>{_escape(row.get('model_version') or '')}</dd></div>"
            f"<div><dt>Cutoff</dt><dd>{_escape(row.get('data_cutoff_utc') or '')}</dd></div>"
            "</dl>"
            "</article>"
        )
    return (
        "<section class=\"section-head\"><h1>Próximos partidos</h1>"
        "<p>Probabilidades oficiales 1X2 desde <code>latest.json</code>/<code>latest.csv</code>; "
        "<code>upcoming.md</code> queda como referencia textual publicada.</p></section>"
        f"<div class=\"match-grid\">{''.join(body_rows)}</div>"
    )


def _shadow_body(source: SiteData) -> str:
    if source.shadow_manifest is None:
        return _empty_section("Challenger shadow", "El challenger contextual no está publicado.")
    official_by_fixture = {row.get("source_fixture_id", ""): row for row in source.latest_rows}
    paired = [
        (official_by_fixture[key], row)
        for row in source.shadow_rows
        for key in [row.get("source_fixture_id", "")]
        if key in official_by_fixture
    ]
    rows = []
    for official, shadow in sorted(paired, key=lambda item: item[0].get("kickoff_utc", "")):
        match_label = (
            f"{_escape(_team_name(official, 'home'))} vs "
            f"{_escape(_team_name(official, 'away'))}"
        )
        home_delta = _pp_delta(
            shadow.get("probability_home_win"),
            official.get("probability_home_win"),
        )
        draw_delta = _pp_delta(
            shadow.get("probability_draw"),
            official.get("probability_draw"),
        )
        away_delta = _pp_delta(
            shadow.get("probability_away_win"),
            official.get("probability_away_win"),
        )
        rows.append(
            "<tr>"
            f"<td>{match_label}</td>"
            f"<td>{_percent(official.get('probability_home_win'))}</td>"
            f"<td>{_percent(shadow.get('probability_home_win'))}</td>"
            f"<td>{home_delta}</td>"
            f"<td>{_percent(official.get('probability_draw'))}</td>"
            f"<td>{_percent(shadow.get('probability_draw'))}</td>"
            f"<td>{draw_delta}</td>"
            f"<td>{_percent(official.get('probability_away_win'))}</td>"
            f"<td>{_percent(shadow.get('probability_away_win'))}</td>"
            f"<td>{away_delta}</td>"
            "</tr>"
        )
    table = (
        _table(
            headers=(
                "Partido",
                "Local oficial",
                "Local shadow",
                "Δ pp",
                "Empate oficial",
                "Empate shadow",
                "Δ pp",
                "Visitante oficial",
                "Visitante shadow",
                "Δ pp",
            ),
            rows=rows,
        )
        if rows
        else "<p class=\"empty\">No hay partidos disponibles en ambos sistemas.</p>"
    )
    summary = _summary_grid(
        (
            ("Partidos oficiales", str(len(source.latest_rows))),
            ("Partidos shadow", str(len(source.shadow_rows))),
            ("Disponibles en ambos", str(len(paired))),
            ("Estado", "shadow_monitoring"),
        )
    )
    return (
        "<section class=\"section-head\"><h1>Challenger shadow</h1>"
        "<p><strong>Modelo oficial</strong>: Poisson. <strong>Challenger experimental</strong>: "
        "contextual en <code>shadow_monitoring</code>. No se combinan probabilidades.</p></section>"
        f"{summary}"
        f"{table}"
    )


def _evaluation_body(source: SiteData) -> str:
    scorecard = source.scorecard
    metrics = _mapping(scorecard.get("metrics"))
    small = _is_small_sample(scorecard)
    metric_cards = [
        ("Partidos evaluados", _value(metrics.get("matches"))),
        ("Log loss", _metric_value(metrics.get("log_loss"), small=small)),
        ("Brier", _metric_value(metrics.get("brier_score"), small=small)),
        ("RPS", _metric_value(metrics.get("ranked_probability_score"), small=small)),
        ("Accuracy secundaria", _metric_value(metrics.get("accuracy"), small=small)),
    ]
    baseline_rows = []
    baselines = scorecard.get("baselines")
    if isinstance(baselines, Mapping):
        for name, raw in sorted(baselines.items()):
            baseline = _mapping(raw)
            status = _value(baseline.get("status"))
            base_metrics = _mapping(baseline.get("metrics"))
            baseline_rows.append(
                "<tr>"
                f"<td>{_escape(str(name))}</td>"
                f"<td>{_escape(status)}</td>"
                f"<td>{_metric_value(base_metrics.get('log_loss'), small=small)}</td>"
                f"<td>{_metric_value(base_metrics.get('brier_score'), small=small)}</td>"
                "<td>"
                f"{_metric_value(base_metrics.get('ranked_probability_score'), small=small)}"
                "</td>"
                "</tr>"
            )
    match_rows = []
    for match in scorecard.get("matches", []):
        if not isinstance(match, Mapping):
            continue
        match_rows.append(
            "<tr>"
            f"<td>{_escape(str(match.get('home_team_name') or ''))} vs "
            f"{_escape(str(match.get('away_team_name') or ''))}</td>"
            f"<td>{_escape(str(match.get('kickoff_utc') or ''))}</td>"
            f"<td>{_escape(str(match.get('actual_result_90') or ''))}</td>"
            f"<td>{_escape(str(match.get('predicted_result') or ''))}</td>"
            f"<td>{_percent(match.get('probability_home_win'))}</td>"
            f"<td>{_percent(match.get('probability_draw'))}</td>"
            f"<td>{_percent(match.get('probability_away_win'))}</td>"
            "</tr>"
        )
    comparison = (
        "<p class=\"empty\">La comparación oficial-shadow todavía no tiene muestra pareada "
        "prospectiva suficiente.</p>"
        if source.shadow_scorecard is None
        or int(_mapping(source.shadow_scorecard.get("metrics")).get("matches") or 0) == 0
        else (
            "<p>La comparación oficial-shadow tiene observaciones disponibles en el "
            "scorecard shadow.</p>"
        )
    )
    match_headers = (
        "Partido",
        "Kickoff UTC",
        "Resultado 90",
        "Predicho",
        "P local",
        "P empate",
        "P visitante",
    )
    return (
        "<section class=\"section-head\"><h1>Evaluación prospectiva</h1>"
        "<p>La política oficial usa predicciones válidas al menos seis horas antes del kickoff. "
        "Accuracy se reporta como métrica secundaria.</p></section>"
        f"{_summary_grid(metric_cards)}"
        f"{_scorecard_warning(scorecard)}"
        "<h2>Baselines</h2>"
        f"{_table(('Baseline', 'Estado', 'Log loss', 'Brier', 'RPS'), baseline_rows)}"
        "<h2>Comparación oficial-shadow</h2>"
        f"{comparison}"
        "<h2>Lista por partido</h2>"
        f"{_table(match_headers, match_rows)}"
    )


def _simulation_body(source: SiteData) -> str:
    if source.simulation_manifest is None:
        return _empty_section("Simulación del torneo", "La simulación todavía no está publicada.")
    manifest = source.simulation_manifest
    top = sorted(
        source.simulation_teams,
        key=lambda row: _float(row.get("champion")),
        reverse=True,
    )
    top_rows = [
        "<tr>"
        f"<td>{index}</td><td>{_escape(str(row.get('team_id') or ''))}</td>"
        f"<td>{_percent(row.get('champion'))}</td>"
        f"<td>{_percent(row.get('final'))}</td>"
        f"<td>{_percent(row.get('semi_final'))}</td>"
        f"<td>{_percent(row.get('quarter_final'))}</td>"
        f"<td>{_percent(row.get('round_of_16'))}</td>"
        f"<td>{_percent(row.get('round_of_32'))}</td>"
        "</tr>"
        for index, row in enumerate(top[:20], start=1)
    ]
    full_rows = [
        "<tr>"
        f"<td>{_escape(str(row.get('team_id') or ''))}</td>"
        f"<td>{_percent(row.get('champion'))}</td>"
        f"<td>{_percent(row.get('final'))}</td>"
        f"<td>{_percent(row.get('semi_final'))}</td>"
        f"<td>{_percent(row.get('quarter_final'))}</td>"
        f"<td>{_percent(row.get('round_of_16'))}</td>"
        f"<td>{_percent(row.get('round_of_32'))}</td>"
        "</tr>"
        for row in top
    ]
    mexico = next((row for row in top if row.get("team_id") == "mexico"), None)
    mexico_panel = (
        "<section class=\"mexico-panel\"><h2>México</h2>"
        f"<p>Campeón {_percent(mexico.get('champion'))}; final {_percent(mexico.get('final'))}; "
        f"semifinal {_percent(mexico.get('semi_final'))}; cuartos "
        f"{_percent(mexico.get('quarter_final'))}.</p></section>"
        if mexico is not None
        else ""
    )
    metadata = (
        ("Runs", _value(manifest.get("runs"))),
        ("Seed", _value(manifest.get("seed"))),
        ("Cutoff", _value(manifest.get("data_cutoff_utc"))),
        ("Model version", _model_label(manifest.get("model"))),
        ("Rule version", _rule_label(manifest.get("rules"))),
        ("Simulation run ID", _value(manifest.get("simulation_run_id"))),
        (
            "Fallback count",
            _value(_mapping(manifest.get("fallback_counts")).get("random_lot_proxy")),
        ),
    )
    round_headers = (
        "Rank",
        "Selección",
        "Campeón",
        "Final",
        "Semifinal",
        "Cuartos",
        "R16",
        "R32",
    )
    full_headers = ("Selección", "Campeón", "Final", "Semifinal", "Cuartos", "R16", "R32")
    return (
        "<section class=\"section-head\"><h1>Simulación del torneo</h1>"
        "<p>Monte Carlo oficial basado solo en <code>poisson_goal_v1</code>.</p></section>"
        f"{_summary_grid(metadata)}"
        f"{mexico_panel}"
        "<h2>Top campeón</h2>"
        f"{_table(round_headers, top_rows)}"
        "<h2>Probabilidades completas</h2>"
        f"{_table(full_headers, full_rows)}"
    )


def _groups_body(source: SiteData) -> str:
    if not source.group_rows:
        return _empty_section("Grupos", "No hay resumen de grupos publicado.")
    observed = _groups_look_observed(source.group_rows)
    group_headers = (
        "Equipo",
        "1º",
        "2º",
        "3º",
        "4º",
        "Directo",
        "Mejor 3º",
        "Eliminado",
    )
    groups: dict[str, list[Mapping[str, str]]] = {}
    for row in source.group_rows:
        groups.setdefault(row.get("Group", ""), []).append(row)
    sections = []
    for group, rows in sorted(groups.items()):
        table_rows = []
        for row in rows:
            table_rows.append(
                "<tr>"
                f"<td>{_escape(row.get('Team', ''))}</td>"
                f"<td>{_escape(row.get('1st', ''))}</td>"
                f"<td>{_escape(row.get('2nd', ''))}</td>"
                f"<td>{_escape(row.get('3rd', ''))}</td>"
                f"<td>{_escape(row.get('4th', ''))}</td>"
                f"<td>{_escape(row.get('Direct', ''))}</td>"
                f"<td>{_escape(row.get('Best 3rd', ''))}</td>"
                f"<td>{_escape(row.get('Eliminated', ''))}</td>"
                "</tr>"
            )
        sections.append(
            f"<section class=\"group-block\"><h2>Grupo {_escape(group)}</h2>"
            f"{_table(group_headers, table_rows)}"
            "</section>"
        )
    label = (
        "La fase de grupos aparece completamente observada; los 1.000/0.000 se muestran como "
        "resultado observado, no como probabilidades activas."
        if observed
        else "La fase de grupos todavía tiene posiciones probabilísticas relevantes."
    )
    return (
        "<section class=\"section-head\"><h1>Grupos</h1>"
        f"<p>{_escape(label)}</p></section>"
        + "".join(sections)
    )


def _bracket_body(source: SiteData) -> str:
    if not source.bracket_rows:
        return _empty_section("Bracket", "No hay resumen de bracket publicado.")
    fixed_rows: list[str] = []
    simulated_rows: list[str] = []
    for row in source.bracket_rows:
        probability = _float(row.get("Probability"))
        target = fixed_rows if probability >= 0.9995 else simulated_rows
        status = (
            "Participante observado/fijo"
            if probability >= 0.9995
            else "Cruce simulado frecuente"
        )
        target.append(
            "<tr>"
            f"<td>{_escape(row.get('Pair', ''))}</td>"
            f"<td>{_percent(probability)}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    finished = []
    for match in source.scorecard.get("matches", []):
        if isinstance(match, Mapping):
            finished.append(
                "<tr>"
                f"<td>{_escape(str(match.get('home_team_name') or ''))} vs "
                f"{_escape(str(match.get('away_team_name') or ''))}</td>"
                f"<td>{_escape(str(match.get('stage') or ''))}</td>"
                f"<td>{_escape(str(match.get('actual_result_90') or ''))}</td>"
                "</tr>"
            )
    upcoming = [
        "<tr>"
        f"<td>{_escape(_team_name(row, 'home'))} vs {_escape(_team_name(row, 'away'))}</td>"
        f"<td>{_escape(row.get('kickoff_utc') or '')}</td>"
        f"<td>{_escape(row.get('model_version') or '')}</td>"
        "</tr>"
        for row in source.latest_rows
    ]
    return (
        "<section class=\"section-head\"><h1>Bracket</h1>"
        "<p>La visualización usa los cruces preparados por el simulador; no reconstruye "
        "reglas en JavaScript.</p>"
        "</section>"
        "<h2>Participantes observados o fijos</h2>"
        f"{_table(('Cruce', 'Probabilidad', 'Estado'), fixed_rows)}"
        "<h2>Cruces frecuentes simulados</h2>"
        f"{_table(('Cruce', 'Probabilidad', 'Estado'), simulated_rows)}"
        "<h2>Resultados terminados</h2>"
        f"{_table(('Partido', 'Ronda', 'Resultado 90'), finished)}"
        "<h2>Próximos partidos</h2>"
        f"{_table(('Partido', 'Kickoff UTC', 'Modelo'), upcoming)}"
    )


def _methodology_body(_source: SiteData) -> str:
    links = (
        ("Modelo Poisson", f"{REPO_DOCS_URL}/docs/EVALUATION_ARTIFACTS.md"),
        ("Operación", f"{REPO_DOCS_URL}/docs/OPERATIONS.md"),
        ("Simulación", f"{REPO_DOCS_URL}/docs/TOURNAMENT_SIMULATION.md"),
        ("Features contextuales", f"{REPO_DOCS_URL}/docs/CONTEXTUAL_FEATURES.md"),
    )
    link_html = "".join(
        f'<li><a href="{href}">{_escape(label)}</a></li>'
        for label, href in links
    )
    return (
        "<section class=\"section-head\"><h1>Metodología</h1>"
        "<p>Resumen operativo de los artefactos publicados.</p></section>"
        "<div class=\"method-grid\">"
        "<section><h2>Modelo oficial</h2><p>Poisson de goles con vida media de 730 días. "
        "Produce probabilidades 1X2 y matriz de marcadores para partidos futuros.</p></section>"
        "<section><h2>Challenger contextual</h2><p>Modelo experimental en shadow mode; se publica "
        "separado y no reemplaza al oficial.</p></section>"
        "<section><h2>Evaluación</h2><p>La política oficial selecciona predicciones prospectivas "
        "al menos seis horas antes del kickoff cuando existen.</p></section>"
        "<section><h2>Monte Carlo</h2><p>La simulación conserva resultados observados, simula lo "
        "pendiente, usa prórroga con goles escalados 30/90 y penales 50/50.</p></section>"
        "<section><h2>Desempates</h2><p>Los criterios modelados cubren puntos, diferencia, goles "
        "y head-to-head. Fair play y ranking FIFA no se inventan; si faltan datos se reporta "
        "<code>random_lot_proxy</code>.</p></section>"
        "<section><h2>Limitaciones</h2><p>No incluye clima, alineaciones, lesiones ni datos de "
        "jugadores. La muestra prospectiva pequeña se marca explícitamente.</p></section>"
        "</div>"
        f"<h2>Documentación</h2><ul class=\"doc-links\">{link_html}</ul>"
    )


def _copy_assets(output_root: Path) -> None:
    target = output_root / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("styles.css", "site.js"):
        source = ASSET_ROOT / name
        if not source.is_file():
            raise SiteBuildError(f"missing site asset: {source}")
        shutil.copyfile(source, target / name)


def _page_coverage(pages: Sequence[Page], source: SiteData) -> list[Mapping[str, Any]]:
    data_flags = {
        "upcoming_predictions": len(source.latest_rows),
        "shadow_predictions": len(source.shadow_rows),
        "scorecard_matches": _mapping(source.scorecard.get("metrics")).get("matches", 0),
        "simulation_teams": len(source.simulation_teams),
        "group_rows": len(source.group_rows),
        "bracket_rows": len(source.bracket_rows),
    }
    return [
        {
            "page": page.output_path.as_posix(),
            "title": page.title,
            "bytes": len(page.html.encode("utf-8")),
            "data_flags": data_flags,
        }
        for page in pages
    ]


def _fallback_audit(source: SiteData) -> Mapping[str, Any]:
    manifest = source.simulation_manifest
    if manifest is None:
        return {
            "schema_version": "wc2026_simulation_fallback_audit_v1",
            "status": "simulation_missing",
            "random_lot_proxy_count": 0,
        }
    fallback_counts = _mapping(manifest.get("fallback_counts"))
    count = int(fallback_counts.get("random_lot_proxy") or 0)
    runs = int(manifest.get("runs") or 0)
    return {
        "schema_version": "wc2026_simulation_fallback_audit_v1",
        "simulation_run_id": manifest.get("simulation_run_id"),
        "runs": runs,
        "random_lot_proxy_count": count,
        "runs_with_at_least_one_fallback": count if count == runs else None,
        "interpretation": (
            "The public manifest reports aggregate fallback decisions. The audited 2026-07-01 "
            "run had one best-third ranking fallback per trajectory and no measured effect on "
            "qualifiers, Annex C crossings or champion probabilities."
        )
        if count
        else "No random_lot_proxy fallback is reported.",
        "warning": _fallback_warning(manifest),
    }


def _site_file_checksums(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _site_checksum(checksums: Mapping[str, str]) -> str:
    payload = _json_dumps(checksums).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SiteBuildError(f"required JSON file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SiteBuildError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SiteBuildError(f"JSON file must contain an object: {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SiteBuildError(f"required CSV file is missing: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise SiteBuildError(f"CSV file must include a header: {path}")
        return [dict(row) for row in reader]


def _markdown_table_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = _markdown_cells(lines[0])
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = _markdown_cells(line)
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _validate_markdown_file(path: Path) -> None:
    if not path.is_file():
        raise SiteBuildError(f"required Markdown file is missing: {path}")
    path.read_text(encoding="utf-8")


def _file_text_for_secret_scan(path: Path) -> str:
    if path.suffix == ".gz":
        try:
            return gzip.decompress(path.read_bytes()).decode("utf-8", errors="ignore")
        except OSError as exc:
            raise SiteBuildError(f"invalid gzip file: {path}") from exc
    return path.read_text(encoding="utf-8", errors="ignore")


def _assert_no_secret_string(relative_path: str, text: str) -> None:
    if SECRET_PATTERN.search(text):
        raise SiteBuildError(f"secret-like assignment detected in {relative_path}")
    for env_name in ("FOOTBALL_DATA_API_KEY", "API_FOOTBALL_KEY"):
        value = os.environ.get(env_name)
        if value and len(value) >= 8 and value in text:
            raise SiteBuildError(f"secret value detected in {relative_path}")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SiteBuildError(f"timestamp is not valid ISO 8601 UTC: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SiteBuildError(f"timestamp must be timezone-aware UTC: {value}")
    return parsed.astimezone(UTC)


def _required_str(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise SiteBuildError(f"field is required: {field}")
    return value


def _required_row_str(row: Mapping[str, str], field: str, *, fixture_id: str) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise SiteBuildError(f"{field} is required for {fixture_id}")
    return value


def _probability(value: object, *, field: str) -> float:
    parsed = _float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise SiteBuildError(f"invalid probability for {field}")
    return parsed


def _float(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SiteBuildError(f"expected finite numeric value: {value!r}") from exc
    if not math.isfinite(parsed):
        raise SiteBuildError(f"expected finite numeric value: {value!r}")
    return parsed


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _small_sample_warning(scorecard: Mapping[str, Any]) -> str | None:
    warning = scorecard.get("small_sample_warning")
    if isinstance(warning, Mapping) and warning.get("applies") is True:
        message = warning.get("message")
        return str(message) if isinstance(message, str) and message else None
    return None


def _is_small_sample(scorecard: Mapping[str, Any]) -> bool:
    warning = scorecard.get("small_sample_warning")
    return isinstance(warning, Mapping) and warning.get("applies") is True


def _fallback_warning(simulation_manifest: Mapping[str, Any] | None) -> str | None:
    if simulation_manifest is None:
        return None
    count = int(_mapping(simulation_manifest.get("fallback_counts")).get("random_lot_proxy") or 0)
    if count <= 0:
        return None
    return (
        f"La simulación reporta random_lot_proxy={count}; fair play/ranking FIFA no se inventan."
    )


def _missing_shadow_warning(shadow_manifest: Mapping[str, Any] | None) -> str | None:
    if shadow_manifest is None:
        return "El challenger shadow no está publicado en este snapshot."
    return None


def _missing_simulation_warning(simulation_manifest: Mapping[str, Any] | None) -> str | None:
    if simulation_manifest is None:
        return "La simulación del torneo no está publicada en este snapshot."
    return None


def _summary_grid(items: Sequence[tuple[str, str]]) -> str:
    cards = "".join(
        f"<div class=\"metric-card\"><dt>{_escape(label)}</dt><dd>{value}</dd></div>"
        for label, value in items
    )
    return f"<dl class=\"metric-grid\">{cards}</dl>"


def _warning_panel(source: SiteData) -> str:
    if not source.warnings:
        return ""
    items = "".join(f"<li>{_escape(warning)}</li>" for warning in source.warnings)
    return f"<section class=\"warning-panel\"><h2>Advertencias</h2><ul>{items}</ul></section>"


def _scorecard_warning(scorecard: Mapping[str, Any]) -> str:
    warning = _small_sample_warning(scorecard)
    if warning is None:
        return ""
    threshold = _mapping(scorecard.get("small_sample_warning")).get("threshold")
    return (
        "<p class=\"notice\">"
        f"{_escape(warning)} Umbral configurado: {_escape(str(threshold))} partidos."
        "</p>"
    )


def _empty_section(title: str, message: str) -> str:
    return (
        f"<section class=\"section-head\"><h1>{_escape(title)}</h1></section>"
        f"<p class=\"empty\">{_escape(message)}</p>"
    )


def _table(headers: Sequence[str], rows: Sequence[str]) -> str:
    head = "".join(f"<th scope=\"col\">{_escape(header)}</th>" for header in headers)
    body = "".join(rows) if rows else (
        f"<tr><td colspan=\"{len(headers)}\" class=\"empty\">Sin datos publicados.</td></tr>"
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _probability_bar(label: str, value: object) -> str:
    probability = _probability(value, field=label)
    width = max(0.0, min(100.0, probability * 100.0))
    return (
        "<div class=\"prob-row\">"
        f"<span>{_escape(label)}</span>"
        f"<div class=\"prob-track\" aria-label=\"{_escape(label)} {_percent(probability)}\">"
        f"<i style=\"width: {width:.1f}%\"></i></div>"
        f"<strong>{_percent(probability)}</strong>"
        "</div>"
    )


def _freshness_widget(cutoff: str) -> str:
    return f'<span data-freshness data-cutoff="{_escape(cutoff)}">Pendiente</span>'


def _favorite_team(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "n/a"
    top = max(rows, key=lambda row: _float(row.get("champion")))
    return f"{_escape(str(top.get('team_id') or 'n/a'))} ({_percent(top.get('champion'))})"


def _model_label(value: object) -> str:
    model = _mapping(value)
    family = str(model.get("family") or "n/a")
    version = str(model.get("version") or "n/a")
    return f"{_escape(family)} <span class=\"muted\">{_escape(version)}</span>"


def _rule_label(value: object) -> str:
    rules = _mapping(value)
    return _escape(str(rules.get("version") or "n/a"))


def _shadow_status(source: SiteData) -> str:
    if source.shadow_manifest is None:
        return "no publicado"
    model = _mapping(source.shadow_manifest.get("model"))
    version = _escape(str(model.get("version") or ""))
    return f'shadow_monitoring <span class="muted">{version}</span>'


def _team_name(row: Mapping[str, str], side: str) -> str:
    return row.get(f"{side}_team_name") or row.get(f"{side}_team_id") or "TBD"


def _groups_look_observed(rows: Sequence[Mapping[str, str]]) -> bool:
    fields = ("1st", "2nd", "3rd", "4th")
    if not rows:
        return False
    for row in rows:
        values = [_float(row.get(field)) for field in fields]
        if sorted(values) != [0.0, 0.0, 0.0, 1.0]:
            return False
    return True


def _metric_value(value: object, *, small: bool) -> str:
    if value is None or value == "":
        return "n/a"
    if small:
        return f"n/a <span class=\"muted\">diagnóstico: {_decimal(value)}</span>"
    return _decimal(value)


def _decimal(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    parsed = _float(value)
    return f"{parsed:.3f}"


def _percent(value: object) -> str:
    parsed = _float(value)
    return f"{parsed * 100.0:.1f}%"


def _pp_delta(left: object, right: object) -> str:
    return f"{(_float(left) - _float(right)) * 100.0:+.1f} pp"


def _value(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    return _escape(str(value))


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

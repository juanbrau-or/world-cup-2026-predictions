"""Command-line interface."""

import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from worldcup2026.config import get_settings
from worldcup2026.data.historical_ingest import (
    HistoricalIngestError,
    audit_team_aliases,
    load_historical_source_config,
    run_historical_ingest,
)
from worldcup2026.data.modeling_dataset import (
    ModelingDatasetError,
    load_modeling_dataset_config,
    run_modeling_dataset_preparation,
)
from worldcup2026.data.world_cup_ingest import (
    WorldCupIngestError,
    offline_provider,
    provider_from_settings,
    run_world_cup_ingest,
    secondary_provider_from_settings,
)
from worldcup2026.evaluation.contextual_challenger import (
    ContextualChallengerError,
    run_contextual_challenger_evaluation,
    run_contextual_challenger_model,
)
from worldcup2026.evaluation.dixon_coles_backtest import (
    DixonColesBacktestError,
    load_dixon_coles_config,
    load_dixon_coles_evaluation_config,
    run_dixon_coles_evaluation,
    run_dixon_coles_model,
)
from worldcup2026.evaluation.elo_backtest import (
    EloEvaluationError,
    load_elo_evaluation_config,
    run_elo_evaluation,
)
from worldcup2026.evaluation.prospective import (
    ProspectiveEvaluationError,
    run_prospective_evaluation,
)
from worldcup2026.features.elo import EloRatingsError, load_elo_ratings_config, run_elo_ratings
from worldcup2026.models.dixon_coles import DixonColesModelError
from worldcup2026.pipelines.contextual_features import (
    ContextualFeaturePipelineError,
    audit_contextual_feature_outputs,
    contextual_feature_report_summary,
    run_contextual_feature_pipeline,
)
from worldcup2026.pipelines.operational_predictions import (
    OperationalPredictionError,
    run_predict_upcoming,
)
from worldcup2026.pipelines.operational_summary import (
    OperationalSummaryError,
    write_operational_step_summary,
)
from worldcup2026.pipelines.publication import PublicationError, prepare_predictions_publication
from worldcup2026.pipelines.shadow_contextual import (
    ShadowContextualError,
    run_evaluate_shadow_contextual,
    run_predict_shadow_contextual,
)
from worldcup2026.simulation.tournament import (
    TournamentSimulationError,
    audit_simulation_outputs,
    run_tournament_simulation,
    simulation_report_summary,
)
from worldcup2026.site import SiteBuildError, build_site

app = typer.Typer(no_args_is_help=True)
ingest_app = typer.Typer(no_args_is_help=True)
audit_app = typer.Typer(no_args_is_help=True)
prepare_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
predict_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
publish_app = typer.Typer(no_args_is_help=True)
operational_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
simulate_app = typer.Typer(no_args_is_help=True)
site_app = typer.Typer(no_args_is_help=True)
console = Console()

MIN_PYTHON = (3, 11)
REQUIRED_DIRECTORIES = (
    "artifacts/models",
    "configs",
    "data/cache",
    "data/interim",
    "data/processed",
    "data/raw",
    "data/static",
    "docs",
    "predictions/history",
    "src",
    "tests",
)
REQUIRED_FILES = (
    ".env.example",
    "README.md",
    "docs/DATA_CONTRACT.md",
    "docs/ROADMAP.md",
    "pyproject.toml",
    "uv.lock",
)


@app.callback()
def main() -> None:
    """World Cup 2026 prediction utilities."""


app.add_typer(ingest_app, name="ingest", help="Ingest source data into project contracts.")
app.add_typer(audit_app, name="audit", help="Audit source data and static normalization tables.")
app.add_typer(prepare_app, name="prepare", help="Prepare deterministic derived datasets.")
app.add_typer(model_app, name="model", help="Build model-stage derived artifacts.")
app.add_typer(predict_app, name="predict", help="Generate operational predictions.")
app.add_typer(evaluate_app, name="evaluate", help="Evaluate probabilistic model stages.")
app.add_typer(publish_app, name="publish", help="Prepare branch-safe prediction publications.")
app.add_typer(operational_app, name="operational", help="Operational workflow helpers.")
app.add_typer(report_app, name="report", help="Read generated reports.")
app.add_typer(simulate_app, name="simulate", help="Run tournament simulations.")
app.add_typer(site_app, name="site", help="Build the static public dashboard.")


@app.command()
def doctor() -> None:
    """Check that the initial local environment is usable."""

    settings = get_settings()
    missing_directories = [name for name in REQUIRED_DIRECTORIES if not Path(name).is_dir()]
    missing_files = [name for name in REQUIRED_FILES if not Path(name).is_file()]
    python_ok = sys.version_info >= MIN_PYTHON

    console.print(f"Python: {platform.python_version()}")
    console.print(f"Platform: {platform.platform()}")
    console.print(f"Project root: {Path.cwd()}")
    console.print(
        "Football data key configured: "
        f"{'yes' if settings.football_data_api_key or settings.api_football_key else 'no'}"
    )
    open_meteo_configured = "yes" if settings.open_meteo_base_url else "no"
    console.print(f"Open-Meteo base URL configured: {open_meteo_configured}")

    if not python_ok:
        console.print("[red]Python 3.11 or newer is required.[/red]")
    if missing_directories:
        console.print(f"[red]Missing directories: {', '.join(missing_directories)}[/red]")
    if missing_files:
        console.print(f"[red]Missing files: {', '.join(missing_files)}[/red]")

    if not python_ok or missing_directories or missing_files:
        raise typer.Exit(code=1)

    console.print("[green]Initial environment looks healthy for Phase 0.[/green]")


@site_app.command("build")
def site_build(
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root",
            help="Checked-out predictions-data root containing manifest.json.",
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Static site output directory for GitHub Pages artifact upload.",
        ),
    ] = Path("site-dist"),
) -> None:
    """Build the static dashboard from public prediction outputs."""

    try:
        result = build_site(data_root=data_root, output_root=output_root)
    except SiteBuildError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Site output: {result.output_root}")
    console.print(f"Pages: {result.page_count}")
    console.print(f"Build manifest: {result.manifest_path}")
    console.print(f"Checksum report: {result.checksum_report_path}")
    console.print(f"Site checksum: {result.site_checksum}")
    for warning in result.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


@ingest_app.command("historical")
def ingest_historical(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative source configuration.",
        ),
    ] = Path("configs/sources.yaml"),
    results_file: Annotated[
        Path | None,
        typer.Option(
            "--results-file",
            help="Use a local results.csv equivalent instead of downloading it.",
        ),
    ] = None,
    shootouts_file: Annotated[
        Path | None,
        typer.Option(
            "--shootouts-file",
            help="Use a local shootouts.csv equivalent instead of downloading it.",
        ),
    ] = None,
    raw_root: Annotated[
        Path | None,
        typer.Option(
            "--raw-root",
            help="Override the raw snapshot root directory.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Override the canonical Parquet output path.",
        ),
    ] = None,
    quarantine_path: Annotated[
        Path | None,
        typer.Option(
            "--quarantine",
            help="Override the invalid-record JSONL output path.",
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Override the quality report JSON output path.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Fetch, parse, and validate without writing raw or processed artifacts.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Process only the first N results rows for controlled validation runs.",
        ),
    ] = None,
) -> None:
    """Ingest historical international matches into the canonical match contract."""

    try:
        config = load_historical_source_config(config_path)
        result = run_historical_ingest(
            config,
            results_file=results_file,
            shootouts_file=shootouts_file,
            raw_root=raw_root,
            output_path=output_path,
            quarantine_path=quarantine_path,
            report_path=report_path,
            dry_run=dry_run,
            limit=limit,
        )
    except HistoricalIngestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    report = result.report
    console.print(f"Source: {report.source}")
    console.print(f"Results rows: {report.results_rows_downloaded}")
    if report.results_rows_processed != report.results_rows_downloaded:
        console.print(f"Results rows processed: {report.results_rows_processed}")
    console.print(f"Shootout rows: {report.shootout_rows_downloaded}")
    console.print(f"Normalized rows: {report.normalized_rows}")
    console.print(f"Invalid rows: {report.invalid_rows}")
    console.print(f"Duplicate rows: {report.duplicate_rows}")
    if report.dry_run:
        console.print("[yellow]Dry run: processed outputs were not written.[/yellow]")
    else:
        console.print(f"Canonical dataset: {report.output_path}")
        console.print(f"Quarantine: {report.quarantine_path}")


@ingest_app.command("world-cup")
def ingest_world_cup(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Fetch and validate without writing snapshots or derived views."
        ),
    ] = False,
    offline_fixture: Annotated[
        bool,
        typer.Option("--offline-fixture", help="Use the bundled offline Football-Data fixture."),
    ] = False,
    offline_fixture_path: Annotated[
        Path,
        typer.Option("--offline-fixture-path", help="Override the bundled offline fixture path."),
    ] = Path("tests/fixtures/world_cup_2026/football_data_matches.json"),
    raw_root: Annotated[
        Path,
        typer.Option("--raw-root", help="Override append-only raw snapshot storage."),
    ] = Path("data/raw"),
    processed_root: Annotated[
        Path,
        typer.Option(
            "--processed-root", help="Override the operational and historical live views."
        ),
    ] = Path("data/processed/world_cup_2026"),
    interim_root: Annotated[
        Path,
        typer.Option("--interim-root", help="Override report output storage."),
    ] = Path("data/interim"),
) -> None:
    """Ingest traceable live World Cup 2026 fixtures and results."""

    settings = get_settings()
    try:
        if offline_fixture:
            provider = offline_provider(offline_fixture_path)
            secondary = None
        else:
            provider = provider_from_settings(
                settings, cache_root=Path("data/cache/world_cup_2026")
            )
            secondary = secondary_provider_from_settings(
                settings,
                cache_root=Path("data/cache/world_cup_2026"),
            )
        result = run_world_cup_ingest(
            provider,
            raw_root=raw_root,
            processed_root=processed_root,
            interim_root=interim_root,
            secondary_provider=secondary,
            dry_run=dry_run,
        )
    except WorldCupIngestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    report = result.report
    console.print(f"Provider: {report.provider}")
    console.print(f"Provider fixtures received: {report.provider_fixtures_received}")
    console.print(f"Known participants: {report.fixtures_with_known_participants}")
    console.print(
        f"Partially known participants: {report.fixtures_with_partially_known_participants}"
    )
    console.print(f"TBD participants: {report.fixtures_with_tbd_participants}")
    console.print(f"Invalid provider fixtures: {report.invalid_provider_fixtures}")
    console.print(f"Canonical matches: {report.canonical_matches}")
    console.print(f"Pending: {report.pending_fixtures}")
    console.print(f"In progress: {report.in_progress_matches}")
    console.print(f"Finished: {report.finished_matches}")
    console.print(f"Interrupted: {report.interrupted_matches}")
    console.print(f"Unresolved teams: {len(report.unresolved_teams)}")
    console.print(
        f"Snapshot differences: {len(report.freshness.differences_from_previous_snapshot)}"
    )
    if report.dry_run:
        console.print("[yellow]Dry run: no snapshots or derived views were written.[/yellow]")
    else:
        console.print(f"Operational table: {report.operational_table}")


@audit_app.command("aliases")
def audit_aliases(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative source configuration.",
        ),
    ] = Path("configs/sources.yaml"),
    results_file: Annotated[
        Path | None,
        typer.Option(
            "--results-file",
            help="Use a local results.csv equivalent instead of downloading it.",
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Write the alias audit report as JSON.",
        ),
    ] = None,
) -> None:
    """Audit exact source team aliases against the canonical team catalog."""

    try:
        config = load_historical_source_config(config_path)
        report = audit_team_aliases(
            config,
            results_file=results_file,
            report_path=report_path,
        )
    except HistoricalIngestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Source: {report.source}")
    console.print(f"Canonical teams: {report.catalog_rows}")
    console.print(f"Alias rows: {report.alias_rows}")
    console.print(f"Original team names: {report.original_team_names}")
    console.print(f"Resolved team names: {report.resolved_team_names}")
    console.print(f"Unresolved team names: {len(report.unresolved_team_names)}")
    console.print(
        "Rows with resolved team names: "
        f"{report.rows_with_resolved_team_names}/{report.result_rows} "
        f"({report.team_name_row_coverage:.2%})"
    )
    console.print(f"Orphan canonical IDs: {len(report.orphan_canonical_team_ids)}")
    if report_path is not None:
        console.print(f"Audit report: {report_path}")
    if report.unresolved_team_names or report.orphan_canonical_team_ids:
        raise typer.Exit(code=1)


@audit_app.command("simulation")
def audit_simulation(
    simulations_root: Annotated[
        Path,
        typer.Option("--simulations-root", help="Simulation output root."),
    ] = Path("simulations"),
) -> None:
    """Audit latest tournament simulation outputs."""

    try:
        summary = audit_simulation_outputs(simulations_root=simulations_root)
    except TournamentSimulationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Simulation run ID: {summary['simulation_run_id']}")
    console.print(f"Team rows: {summary['team_rows']}")
    console.print(f"Rules: {summary['rule_version']}")
    console.print(f"Model: {summary['model_version']}")
    console.print(f"Fallback counts: {summary['fallback_counts']}")
    if summary["runs_with_random_lot_proxy"] is not None:
        console.print(f"Runs with random_lot_proxy: {summary['runs_with_random_lot_proxy']}")


@audit_app.command("contextual-features")
def audit_contextual_features(
    team_fixture_path: Annotated[
        Path,
        typer.Option("--team-fixture", help="Team-fixture contextual feature Parquet."),
    ] = Path("data/processed/contextual_features/team_fixture_contextual_features.parquet"),
    match_path: Annotated[
        Path,
        typer.Option("--match", help="Match-level contextual feature Parquet."),
    ] = Path("data/processed/contextual_features/match_contextual_features.parquet"),
) -> None:
    """Audit contextual features for leakage and schema corruption."""

    try:
        result = audit_contextual_feature_outputs(
            team_fixture_path=team_fixture_path,
            match_path=match_path,
        )
    except ContextualFeaturePipelineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Team-fixture rows: {result.team_fixture_rows}")
    console.print(f"Match rows: {result.match_rows}")
    console.print(f"Leakage audit passed: {'yes' if result.leakage_audit_passed else 'no'}")
    console.print(f"Violations: {result.violations}")


@prepare_app.command("modeling-data")
def prepare_modeling_data(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative modeling dataset configuration.",
        ),
    ] = Path("configs/modeling_data.yaml"),
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Override the canonical historical Parquet input path.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Override the modeling Parquet output path.",
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Override the modeling data quality report JSON path.",
        ),
    ] = None,
) -> None:
    """Prepare the deterministic modeling dataset from normalized historical matches."""

    try:
        config = load_modeling_dataset_config(config_path)
        result = run_modeling_dataset_preparation(
            config,
            input_path=input_path,
            output_path=output_path,
            report_path=report_path,
        )
    except ModelingDatasetError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    report = result.report
    console.print(f"Rows total: {report.total_rows}")
    console.print(f"Rows eligible: {report.eligible_rows}")
    console.print(f"Date range: {report.date_range['min']} to {report.date_range['max']}")
    console.print(f"Teams included: {report.teams_included}")
    console.print(f"Neutral matches: {report.neutral_matches}")
    console.print(f"Modeling dataset: {report.output_path}")
    console.print(f"Quality report: {report_path or config.report_output}")
    if report.unresolved_competitions:
        console.print(
            "[yellow]Unresolved competitions: "
            f"{len(report.unresolved_competitions)}[/yellow]"
        )


@prepare_app.command("contextual-features")
def prepare_contextual_features(
    historical_matches_path: Annotated[
        Path,
        typer.Option("--historical-matches", help="Canonical historical Parquet input."),
    ] = Path("data/processed/international_matches.parquet"),
    live_matches_path: Annotated[
        Path,
        typer.Option("--live-matches", help="Canonical World Cup live Parquet input."),
    ] = Path("data/processed/world_cup_2026/matches.parquet"),
    venue_catalog_path: Annotated[
        Path,
        typer.Option("--venue-catalog", help="Audited World Cup 2026 venue catalog CSV."),
    ] = Path("data/static/venues.csv"),
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Contextual feature Parquet output directory."),
    ] = Path("data/processed/contextual_features"),
    interim_root: Annotated[
        Path,
        typer.Option("--interim-root", help="Contextual report output directory."),
    ] = Path("data/interim/contextual_features"),
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Feature generation timestamp, UTC ISO 8601."),
    ] = None,
    data_cutoff: Annotated[
        str | None,
        typer.Option("--data-cutoff", help="Maximum source data cutoff, UTC ISO 8601."),
    ] = None,
    include_historical: Annotated[
        bool,
        typer.Option("--historical/--no-historical", help="Include historical canonical matches."),
    ] = True,
    include_live: Annotated[
        bool,
        typer.Option("--live/--no-live", help="Include current World Cup live matches."),
    ] = True,
    offline_fixture: Annotated[
        bool,
        typer.Option("--offline-fixture", help="Build live input from an offline fixture JSON."),
    ] = False,
    offline_fixture_path: Annotated[
        Path,
        typer.Option(
            "--offline-fixture-path",
            help="Football-Data-compatible offline fixture JSON.",
        ),
    ] = Path("tests/fixtures/world_cup_2026/football_data_matches.json"),
) -> None:
    """Prepare auditable as-of contextual features."""

    try:
        generated_at = _parse_optional_utc(as_of, field_name="--as-of")
        cutoff = _parse_optional_utc(data_cutoff, field_name="--data-cutoff")
        effective_live_matches_path: Path | None = live_matches_path
        if offline_fixture:
            fetched_at = cutoff or generated_at or datetime.now(UTC).replace(microsecond=0)
            offline_processed_root = output_root / "offline_world_cup_2026"
            run_world_cup_ingest(
                offline_provider(offline_fixture_path),
                raw_root=output_root / "offline_raw",
                processed_root=offline_processed_root,
                interim_root=interim_root / "offline_ingest",
                dry_run=False,
                fetched_at=fetched_at,
            )
            effective_live_matches_path = offline_processed_root / "matches.parquet"
            if generated_at is None:
                generated_at = fetched_at
        result = run_contextual_feature_pipeline(
            historical_matches_path=historical_matches_path if include_historical else None,
            live_matches_path=effective_live_matches_path if include_live else None,
            venue_catalog_path=venue_catalog_path,
            output_root=output_root,
            interim_root=interim_root,
            feature_generated_at_utc=generated_at,
            data_cutoff_utc=cutoff,
            include_historical=include_historical,
            include_live=include_live,
        )
    except (ContextualFeaturePipelineError, WorldCupIngestError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Team-fixture rows: {result.team_fixture_rows}")
    console.print(f"Match rows: {result.match_rows}")
    console.print(f"Team-fixture Parquet: {result.team_fixture_parquet}")
    console.print(f"Match Parquet: {result.match_parquet}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"Coverage report: {result.coverage_markdown_path}")
    console.print(f"Missing-data report: {result.missing_data_report_path}")
    console.print(f"Leakage audit: {result.leakage_audit_path}")


@model_app.command("elo-ratings")
def model_elo_ratings(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative model configuration.",
        ),
    ] = Path("configs/model.yaml"),
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Override the modeling matches Parquet input path.",
        ),
    ] = None,
    output_match_ratings_path: Annotated[
        Path | None,
        typer.Option(
            "--match-ratings-output",
            help="Override the per-match Elo ratings Parquet output path.",
        ),
    ] = None,
    output_current_ratings_path: Annotated[
        Path | None,
        typer.Option(
            "--current-ratings-output",
            help="Override the current Elo ratings Parquet output path.",
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Override the Elo ratings report JSON path.",
        ),
    ] = None,
) -> None:
    """Build chronological Elo ratings from the modeling match dataset."""

    try:
        config = load_elo_ratings_config(config_path)
        result = run_elo_ratings(
            config,
            input_path=input_path,
            output_match_ratings_path=output_match_ratings_path,
            output_current_ratings_path=output_current_ratings_path,
            report_path=report_path,
        )
    except EloRatingsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    report = result.report
    console.print(f"Model version: {report.model_version}")
    console.print(f"Input rows: {report.total_rows}")
    console.print(f"Matches processed: {report.processed_matches}")
    console.print(f"Matches excluded: {report.excluded_matches}")
    console.print(f"Date range: {report.date_range['min']} to {report.date_range['max']}")
    console.print(f"Teams rated: {report.teams_rated}")
    console.print(f"Match ratings: {report.match_ratings_path}")
    console.print(f"Current ratings: {report.current_ratings_path}")
    console.print(f"Quality report: {report_path or config.report_output}")
    if report.top_ratings:
        console.print("Top ratings:")
        for row in report.top_ratings[:5]:
            console.print(
                f"  {row['canonical_team_id']}: {row['elo_rating']} "
                f"({row['matches_processed']} matches)"
            )


@model_app.command("dixon-coles")
def model_dixon_coles(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative model configuration.",
        ),
    ] = Path("configs/model.yaml"),
) -> None:
    """Fit the configured Poisson or Dixon-Coles goal model."""

    try:
        config = load_dixon_coles_config(config_path)
        result = run_dixon_coles_model(config)
    except (DixonColesBacktestError, DixonColesModelError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Model type: {result.model_type}")
    console.print(f"Half-life days: {result.half_life_days}")
    console.print(f"Training matches: {result.training_matches}")
    console.print(f"Teams: {result.teams}")
    console.print(f"Model: {result.model_path}")
    console.print(f"Report: {result.report_path}")


@model_app.command("contextual-challenger")
def model_contextual_challenger(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the declarative model configuration."),
    ] = Path("configs/model.yaml"),
) -> None:
    """Write the frozen contextual challenger shadow manifest."""

    try:
        result = run_contextual_challenger_model(config_path=config_path)
    except ContextualChallengerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Selected model: {result.selected_model_name}")
    console.print(f"Selected ablation: {result.selected_ablation}")
    console.print(f"Feature set: {result.feature_set_version}")
    console.print(f"Training cutoff: {result.training_cutoff.isoformat()}")
    console.print(f"Manifest: {result.manifest_path}")


@predict_app.command("upcoming")
def predict_upcoming(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the declarative model configuration."),
    ] = Path("configs/model.yaml"),
    modeling_matches_path: Annotated[
        Path,
        typer.Option("--modeling-matches", help="Eligible historical modeling Parquet input."),
    ] = Path("data/processed/modeling_matches.parquet"),
    live_matches_path: Annotated[
        Path,
        typer.Option("--live-matches", help="Current World Cup canonical live Parquet input."),
    ] = Path("data/processed/world_cup_2026/matches.parquet"),
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Prediction output root."),
    ] = Path("predictions"),
) -> None:
    """Generate upcoming World Cup predictions for known future fixtures."""

    try:
        result = run_predict_upcoming(
            model_config_path=config_path,
            modeling_matches_path=modeling_matches_path,
            live_matches_path=live_matches_path,
            predictions_root=predictions_root,
        )
    except OperationalPredictionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Model: {result.model_family} ({result.model_version})")
    console.print(f"Half-life days: {result.half_life_days}")
    console.print(f"Data cutoff UTC: {result.data_cutoff_utc.isoformat()}")
    console.print(f"Training matches: {result.training_matches}")
    console.print(f"World Cup 2026 finished matches used: {result.live_finished_2026_matches}")
    console.print(f"Dataset revision: {result.dataset_revision}")
    console.print(f"Predictions: {len(result.predictions)}")
    console.print(f"Latest CSV: {result.latest_csv_path}")
    console.print(f"Latest Parquet: {result.latest_parquet_path}")
    console.print(f"History: {result.history_path}")
    console.print(f"Report: {result.report_path}")


@predict_app.command("shadow-contextual")
def predict_shadow_contextual(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the declarative model configuration."),
    ] = Path("configs/model.yaml"),
    modeling_matches_path: Annotated[
        Path,
        typer.Option("--modeling-matches", help="Eligible historical modeling Parquet input."),
    ] = Path("data/processed/modeling_matches.parquet"),
    live_matches_path: Annotated[
        Path,
        typer.Option("--live-matches", help="Current World Cup canonical live Parquet input."),
    ] = Path("data/processed/world_cup_2026/matches.parquet"),
    contextual_match_features_path: Annotated[
        Path,
        typer.Option("--contextual-matches", help="Match-level contextual feature Parquet."),
    ] = Path("data/processed/contextual_features/match_contextual_features.parquet"),
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Prediction output root."),
    ] = Path("predictions"),
) -> None:
    """Generate shadow contextual challenger predictions for official fixtures."""

    try:
        result = run_predict_shadow_contextual(
            model_config_path=config_path,
            modeling_matches_path=modeling_matches_path,
            live_matches_path=live_matches_path,
            contextual_match_features_path=contextual_match_features_path,
            predictions_root=predictions_root,
        )
    except (ShadowContextualError, ContextualChallengerError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Model: {result.model_family} ({result.model_version})")
    console.print(f"Data cutoff UTC: {result.data_cutoff_utc.isoformat()}")
    console.print(f"Official baseline fixtures: {result.baseline_fixture_count}")
    console.print(f"Shadow predictions: {len(result.predictions)}")
    console.print(f"Training matches: {result.training_matches}")
    console.print(f"World Cup 2026 finished matches used: {result.live_finished_2026_matches}")
    console.print(f"Latest CSV: {result.latest_csv_path}")
    console.print(f"Latest Parquet: {result.latest_parquet_path}")
    console.print(f"History: {result.history_path}")
    console.print(f"Report: {result.report_path}")


@simulate_app.command("tournament")
def simulate_tournament(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to simulation configuration."),
    ] = Path("configs/simulation.yaml"),
    runs: Annotated[
        int | None,
        typer.Option("--runs", min=0, help="Override Monte Carlo simulation count."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", min=0, help="Override deterministic random seed."),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Simulation data cutoff timestamp, UTC ISO 8601."),
    ] = None,
    offline_fixture: Annotated[
        bool,
        typer.Option("--offline-fixture", help="Use a local raw Football-Data fixture snapshot."),
    ] = False,
    offline_fixture_path: Annotated[
        Path | None,
        typer.Option("--offline-fixture-path", help="Football-Data-compatible fixture JSON."),
    ] = None,
    official_model: Annotated[
        bool,
        typer.Option(
            "--official-model/--no-official-model",
            help="Require official Poisson model.",
        ),
    ] = True,
    shadow_contextual: Annotated[
        bool,
        typer.Option(
            "--shadow-contextual",
            help="Attempt a separate contextual shadow simulation if valid.",
        ),
    ] = False,
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Official prediction output root."),
    ] = Path("predictions"),
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Override simulation output root."),
    ] = None,
) -> None:
    """Run the official World Cup 2026 tournament simulation."""

    try:
        offline_fixture_override = (
            offline_fixture_path if offline_fixture or offline_fixture_path is not None else None
        )
        result = run_tournament_simulation(
            config_path=config_path,
            runs=runs,
            seed=seed,
            as_of=_parse_optional_utc(as_of, field_name="--as-of"),
            offline_fixture=offline_fixture_override,
            official_model=official_model,
            shadow_contextual=shadow_contextual,
            predictions_root=predictions_root,
            output_root=output_root,
        )
    except TournamentSimulationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Simulation run ID: {result.simulation_run_id}")
    console.print(f"Data cutoff UTC: {result.cutoff.isoformat()}")
    console.print(f"Runs: {result.runs}")
    console.print(f"Seed: {result.seed}")
    if result.favorite_team_id is not None and result.favorite_champion_probability is not None:
        console.print(
            "Title favorite: "
            f"{result.favorite_team_id} ({result.favorite_champion_probability:.4f})"
        )
    console.print(f"Fallbacks: {dict(result.fallback_counts)}")
    console.print(f"Latest outputs: {result.latest_root}")
    console.print(f"Manifest: {result.manifest_path}")


@evaluate_app.command("elo")
def evaluate_elo(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative model and evaluation configuration.",
        ),
    ] = Path("configs/model.yaml"),
) -> None:
    """Run walk-forward validation and calibration for Elo 1X2 probabilities."""

    try:
        base_elo_config, evaluation_config = load_elo_evaluation_config(config_path)
        result = run_elo_evaluation(base_elo_config, evaluation_config)
    except EloEvaluationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Selected method: {result.selected_method}")
    console.print(f"Validation log loss: {result.validation_log_loss:.6f}")
    console.print(f"Validation matches: {result.validation_matches}")
    console.print(f"Holdout 2026 matches: {result.holdout_2026_matches}")
    console.print(f"Selected config: {result.selected_config_path}")
    console.print(f"Fold metrics: {result.metrics_by_fold_path}")
    console.print(f"Out-of-fold predictions: {result.out_of_fold_predictions_path}")
    console.print(f"Calibration curves: {result.calibration_curves_path}")
    console.print(f"Report: {result.report_path}")


@evaluate_app.command("contextual-challenger")
def evaluate_contextual_challenger(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Path to the declarative model and evaluation config."),
    ] = Path("configs/model.yaml"),
) -> None:
    """Evaluate contextual challengers with nested temporal validation."""

    try:
        result = run_contextual_challenger_evaluation(config_path=config_path)
    except ContextualChallengerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Selected shadow model: {result.selected_model_name}")
    console.print(f"Selected ablation: {result.selected_ablation}")
    console.print(f"Promotion status: {result.promotion_status}")
    console.print(f"Validation matches: {result.validation_matches}")
    console.print(f"Holdout 2026 matches: {result.holdout_2026_matches}")
    console.print(f"Selected config: {result.selected_config_path}")
    console.print(f"Fold metrics: {result.fold_metrics_path}")
    console.print(f"Paired comparison: {result.paired_comparison_path}")
    console.print(f"Bootstrap report: {result.bootstrap_report_path}")
    console.print(f"Report: {result.report_path}")


@evaluate_app.command("prospective")
def evaluate_prospective(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Prospective evaluation policy configuration."),
    ] = Path("configs/prospective_evaluation.yaml"),
    predictions_history_root: Annotated[
        Path,
        typer.Option("--predictions-history", help="Prediction history directory."),
    ] = Path("predictions/history"),
    live_matches_path: Annotated[
        Path,
        typer.Option("--live-matches", help="Current World Cup canonical live Parquet input."),
    ] = Path("data/processed/world_cup_2026/matches.parquet"),
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Prediction output root."),
    ] = Path("predictions"),
    published_history_root: Annotated[
        Path | None,
        typer.Option(
            "--published-history",
            help="Restored predictions-data history directory with published CSV gzip snapshots.",
        ),
    ] = Path("predictions/published-history/history"),
) -> None:
    """Evaluate saved prospective predictions whose fixtures are now finished."""

    try:
        result = run_prospective_evaluation(
            config_path=config_path,
            predictions_history_root=predictions_history_root,
            live_matches_path=live_matches_path,
            report_path=predictions_root / "prospective_scorecard.md",
            json_path=predictions_root / "prospective_scorecard.json",
            matches_csv_path=predictions_root / "prospective_matches.csv",
            ledger_path=predictions_root / "prediction_ledger.parquet",
            published_history_root=published_history_root,
        )
    except ProspectiveEvaluationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Ledger predictions: {result.ledger_predictions}")
    console.print(f"Official predictions selected: {result.official_predictions_selected}")
    console.print(f"Evaluable official predictions: {result.evaluable_predictions}")
    console.print(f"Log loss: {result.log_loss if result.log_loss is not None else 'n/a'}")
    console.print(f"Brier score: {result.brier_score if result.brier_score is not None else 'n/a'}")
    rps = result.ranked_probability_score if result.ranked_probability_score is not None else "n/a"
    console.print(f"RPS: {rps}")
    console.print(f"Accuracy: {result.accuracy if result.accuracy is not None else 'n/a'}")
    console.print(f"Scorecard JSON: {result.json_path}")
    console.print(f"Scorecard report: {result.report_path}")
    console.print(f"Matches CSV: {result.matches_path}")
    console.print(f"Ledger Parquet: {result.ledger_path}")


@evaluate_app.command("shadow-contextual")
def evaluate_shadow_contextual(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="Shadow prospective evaluation policy configuration."),
    ] = Path("configs/shadow_contextual_evaluation.yaml"),
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Prediction output root."),
    ] = Path("predictions"),
    live_matches_path: Annotated[
        Path,
        typer.Option("--live-matches", help="Current World Cup canonical live Parquet input."),
    ] = Path("data/processed/world_cup_2026/matches.parquet"),
) -> None:
    """Evaluate shadow contextual prospective predictions separately."""

    try:
        result = run_evaluate_shadow_contextual(
            config_path=config_path,
            live_matches_path=live_matches_path,
            predictions_root=predictions_root,
        )
    except ShadowContextualError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Evaluable shadow predictions: {result.evaluable_predictions}")
    console.print(f"Paired official/shadow matches: {result.paired_matches}")
    console.print(f"Scorecard JSON: {result.scorecard_json_path}")
    console.print(f"Scorecard report: {result.scorecard_report_path}")
    console.print(f"Ledger Parquet: {result.ledger_path}")
    console.print(f"Comparison: {result.comparison_path}")


@evaluate_app.command("dixon-coles")
def evaluate_dixon_coles(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the declarative model and evaluation configuration.",
        ),
    ] = Path("configs/model.yaml"),
) -> None:
    """Evaluate Poisson and Dixon-Coles goal models with the Elo temporal folds."""

    try:
        model_config, elo_evaluation_config, evaluation_config = (
            load_dixon_coles_evaluation_config(config_path)
        )
        result = run_dixon_coles_evaluation(
            model_config,
            elo_evaluation_config,
            evaluation_config,
        )
    except DixonColesBacktestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Selected model: {result.selected_model_type}")
    console.print(f"Selected half-life days: {result.selected_half_life_days}")
    console.print(f"Validation log loss: {result.validation_log_loss:.6f}")
    console.print(f"Validation matches: {result.validation_matches}")
    console.print(f"Holdout 2026 matches: {result.holdout_2026_matches}")
    console.print(f"Selected config: {result.selected_config_path}")
    console.print(f"Fold metrics: {result.metrics_by_fold_path}")
    console.print(f"Comparison with Elo: {result.comparison_with_elo_path}")
    console.print(f"Paired summary: {result.paired_comparison_summary_path}")
    console.print(f"Evaluation summary: {result.evaluation_summary_path}")
    console.print(f"Out-of-fold predictions: {result.out_of_fold_predictions_path}")
    console.print(f"Report: {result.report_path}")


@publish_app.command("prepare")
def publish_prepare(
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Root containing generated prediction outputs."),
    ] = Path("predictions"),
    simulations_root: Annotated[
        Path,
        typer.Option("--simulations-root", help="Root containing generated simulation outputs."),
    ] = Path("simulations"),
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Data-branch worktree or staging output root."),
    ] = Path("dist/predictions-data"),
) -> None:
    """Prepare small branch-safe files for predictions-data."""

    try:
        result = prepare_predictions_publication(
            predictions_root=predictions_root,
            simulations_root=simulations_root,
            output_root=output_root,
        )
    except PublicationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Output root: {result.output_root}")
    console.print(f"Changed: {result.changed}")
    console.print(f"Data cutoff UTC: {result.data_cutoff}")
    console.print(f"Predictions: {result.prediction_count}")
    console.print(
        "Prospective scorecard observations: "
        f"{result.prospective_scorecard_observations}"
    )
    console.print(f"Checksum: {result.checksum}")
    console.print(f"Manifest: {result.manifest_path}")
    if result.history_path is not None:
        console.print(f"History: {result.history_path}")
    if result.simulation_run_id is not None:
        console.print(f"Simulation run ID: {result.simulation_run_id}")


@operational_app.command("summary")
def operational_summary(
    predictions_root: Annotated[
        Path,
        typer.Option("--predictions-root", help="Prediction output root."),
    ] = Path("predictions"),
    simulations_root: Annotated[
        Path,
        typer.Option("--simulations-root", help="Simulation output root."),
    ] = Path("simulations"),
    interim_root: Annotated[
        Path,
        typer.Option("--interim-root", help="Interim report root."),
    ] = Path("data/interim"),
    publication_root: Annotated[
        Path,
        typer.Option("--publication-root", help="Prepared publication root."),
    ] = Path("dist/predictions-data"),
    logs_root: Annotated[
        Path,
        typer.Option("--logs-root", help="Operational logs root."),
    ] = Path("logs"),
    summary_path: Annotated[
        Path | None,
        typer.Option("--summary-path", help="Override GitHub Step Summary path."),
    ] = None,
) -> None:
    """Write the GitHub Step Summary for an operational run."""

    try:
        result = write_operational_step_summary(
            summary_path=summary_path,
            predictions_root=predictions_root,
            simulations_root=simulations_root,
            interim_root=interim_root,
            publication_root=publication_root,
            logs_root=logs_root,
        )
    except OperationalSummaryError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Step summary: {result.summary_path}")
    console.print(f"Predictable fixtures: {result.predictable_fixtures}")
    console.print(f"Publication ready: {result.publication_ready}")


@report_app.command("contextual-features")
def report_contextual_features(
    coverage_report_path: Annotated[
        Path,
        typer.Option("--coverage", help="Contextual coverage JSON report."),
    ] = Path("data/interim/contextual_features/contextual_features_coverage.json"),
    leakage_audit_path: Annotated[
        Path,
        typer.Option("--leakage-audit", help="Contextual leakage audit JSON report."),
    ] = Path("data/interim/contextual_features/contextual_features_leakage_audit.json"),
) -> None:
    """Print a compact contextual feature report summary."""

    try:
        summary = contextual_feature_report_summary(
            coverage_report_path=coverage_report_path,
            leakage_audit_path=leakage_audit_path,
        )
    except ContextualFeaturePipelineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Rows total: {summary['rows_total']}")
    console.print(f"Leakage audit passed: {'yes' if summary['leakage_audit_passed'] else 'no'}")
    console.print(f"Leakage violations: {summary['leakage_violations']}")
    classifications = summary.get("feature_classification")
    if isinstance(classifications, dict):
        counts: dict[str, int] = {}
        for item in classifications.values():
            if isinstance(item, dict):
                category = str(item.get("category"))
                counts[category] = counts.get(category, 0) + 1
        for category, count in sorted(counts.items()):
            console.print(f"{category}: {count}")


@report_app.command("simulation")
def report_simulation(
    simulations_root: Annotated[
        Path,
        typer.Option("--simulations-root", help="Simulation output root."),
    ] = Path("simulations"),
) -> None:
    """Print a compact latest simulation summary."""

    try:
        summary = simulation_report_summary(simulations_root=simulations_root)
    except TournamentSimulationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Simulation run ID: {summary['simulation_run_id']}")
    console.print(f"Data cutoff UTC: {summary['data_cutoff_utc']}")
    console.print(f"Runs: {summary['runs']}")
    console.print(f"Seed: {summary['seed']}")
    console.print(f"Fallbacks: {summary['fallback_counts']}")
    console.print("Top champion probabilities:")
    top_champions = summary.get("top_champions")
    if isinstance(top_champions, list):
        for row in top_champions:
            if isinstance(row, dict):
                console.print(f"  {row['team_id']}: {float(str(row['champion'])):.4f}")
    mexico = summary.get("mexico")
    if isinstance(mexico, dict):
        console.print(
            "Mexico: "
            f"champion={float(str(mexico['champion'])):.4f}, "
            f"final={float(str(mexico['final'])):.4f}, "
            f"round_of_16={float(str(mexico['round_of_16'])):.4f}"
        )


def _parse_optional_utc(value: str | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContextualFeaturePipelineError(f"{field_name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.min.replace(tzinfo=UTC).utcoffset():
        raise ContextualFeaturePipelineError(f"{field_name} must be timezone-aware UTC")
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    app()

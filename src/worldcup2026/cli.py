"""Command-line interface."""

import platform
import sys
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
from worldcup2026.evaluation.elo_backtest import (
    EloEvaluationError,
    load_elo_evaluation_config,
    run_elo_evaluation,
)
from worldcup2026.features.elo import EloRatingsError, load_elo_ratings_config, run_elo_ratings

app = typer.Typer(no_args_is_help=True)
ingest_app = typer.Typer(no_args_is_help=True)
audit_app = typer.Typer(no_args_is_help=True)
prepare_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
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
app.add_typer(evaluate_app, name="evaluate", help="Evaluate probabilistic model stages.")


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
    console.print(f"Prospective 2026 matches: {result.prospective_2026_matches}")
    console.print(f"Selected config: {result.selected_config_path}")
    console.print(f"Fold metrics: {result.metrics_by_fold_path}")
    console.print(f"Out-of-fold predictions: {result.out_of_fold_predictions_path}")
    console.print(f"Calibration curves: {result.calibration_curves_path}")
    console.print(f"Report: {result.report_path}")


if __name__ == "__main__":
    app()

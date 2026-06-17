"""Command-line interface."""

import platform
from pathlib import Path

import typer
from rich.console import Console

from worldcup2026.config import get_settings

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """World Cup 2026 prediction utilities."""


@app.command()
def doctor() -> None:
    """Check that the initial local environment is usable."""

    settings = get_settings()
    required = ["configs", "data", "docs", "src", "tests"]
    missing = [name for name in required if not Path(name).exists()]

    console.print(f"Python: {platform.python_version()}")
    console.print(f"Platform: {platform.platform()}")
    console.print(
        "Football data key configured: "
        f"{'yes' if settings.football_data_api_key or settings.api_football_key else 'no'}"
    )

    if missing:
        console.print(f"[red]Missing directories: {', '.join(missing)}[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Initial environment looks healthy.[/green]")


if __name__ == "__main__":
    app()

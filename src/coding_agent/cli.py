"""Command-line bootstrap for the coding agent."""

import typer

from coding_agent import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def health() -> None:
    """Report that the CLI is available."""
    typer.echo("ok")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)

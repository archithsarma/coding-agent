from typer.testing import CliRunner

from coding_agent import __version__
from coding_agent.cli import app


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_health_command() -> None:
    result = CliRunner().invoke(app, ["health"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"

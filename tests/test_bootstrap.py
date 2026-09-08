from typer.testing import CliRunner

from coding_agent import __version__
from coding_agent.cli import app, render_result


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_health_command() -> None:
    result = CliRunner().invoke(app, ["health"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"


def test_help_exposes_chat_and_single_request_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "chat" in result.stdout
    assert "run" in result.stdout


def test_invalid_workspace_is_a_concise_startup_error(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        ["run", "run tests", "--workspace", str(tmp_path / "missing")],
    )

    assert result.exit_code == 2
    assert "Startup error" in result.output
    assert "Traceback" not in result.output


def test_render_result_exposes_model_configuration_failure() -> None:
    assert (
        render_result(
            {
                "failure": {
                    "code": "model_not_configured",
                    "message": "set OPENAI_API_KEY",
                }
            }
        )
        == "Failed [model_not_configured]: set OPENAI_API_KEY"
    )


def test_render_result_shows_grounded_explore_answer() -> None:
    assert (
        render_result({"explore": {"answer": "Tasks are created by create_task."}})
        == "Tasks are created by create_task."
    )

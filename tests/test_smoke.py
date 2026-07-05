"""Smoke tests: the package imports and the CLI runs."""

from agent_hotwash import __version__
from agent_hotwash.cli import app, main


def test_version_is_set() -> None:
    assert __version__


def test_app_is_typer() -> None:
    assert app.info.name == "agent-hotwash"


def test_main_version_runs_clean() -> None:
    assert main(["version"]) == 0

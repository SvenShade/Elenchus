from __future__ import annotations

from typer.testing import CliRunner

from llm_mcts.cli import app
from tests.conftest import minimal_game_data, write_game


runner = CliRunner()


def test_validate_cli_accepts_example_game():
    result = runner.invoke(app, ["validate", "--game", "examples/elenchus_v0_1.yaml"])

    assert result.exit_code == 0
    assert "Valid game" in result.stdout


def test_plan_cli_runs_with_mock_llm(tmp_path):
    result = runner.invoke(
        app,
        [
            "plan",
            "--game",
            "examples/elenchus_v0_1.yaml",
            "--mock-llm",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 0
    assert "Selected action" in result.stdout
    assert (tmp_path / "run" / "calls.jsonl").exists()
    assert (tmp_path / "run" / "tree.json").exists()


def test_play_cli_manual_mode_one_exchange(tmp_path):
    game_path = write_game(tmp_path / "game.yaml", minimal_game_data())
    result = runner.invoke(
        app,
        [
            "play",
            "--game",
            str(game_path),
            "--mock-llm",
            "--p2-mode",
            "manual",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        input="Manual P2 reply\n",
    )

    assert result.exit_code == 0
    assert "Manual P2 reply" in result.stdout

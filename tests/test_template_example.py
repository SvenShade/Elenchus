from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_mcts.cli import app
from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.mcts import MCTSPlanner
from llm_mcts.mock import DeterministicMockLLM


TEMPLATE_GAME = Path("examples/template.yaml")
runner = CliRunner()


def test_template_example_loads_with_action_card_shape():
    config = GameConfig.load(TEMPLATE_GAME)

    assert config.metadata.name == "Template Conversation Game"
    assert config.initial_state.transcript == []
    assert config.llm.turn_transport == "completion"
    assert config.mcts.simulations == 8
    assert config.mcts.max_rollout_depth == 2
    assert config.mcts.progressive_widening.enabled is False
    assert config.prompts.state_analysis
    assert config.prompts.candidate_actions
    assert config.prompts.rollout_reflection
    assert config.prompts.finalize_p1_move
    assert config.prompts.exploration_summary

    action_ids = {action.id for action in config.actions}
    assert {
        "clarify",
        "surface_assumption",
        "test_evidence",
        "surface_risk",
        "offer_synthesis",
        "invite_pushback",
    } == action_ids
    assert all(action.metadata for action in config.actions)
    assert all("agency_policy" in action.metadata for action in config.actions)


def test_template_cli_validate_succeeds():
    result = runner.invoke(app, ["validate", "--game", str(TEMPLATE_GAME)])

    assert result.exit_code == 0
    assert "Valid game" in result.stdout


def test_template_example_runs_mock_mcts_plan():
    config = GameConfig.load(TEMPLATE_GAME)
    env = TwoPlayerConversationEnv(config, DeterministicMockLLM())
    planner = MCTSPlanner(env)

    state = env.initial_state().append(
        "P2",
        "I know the situation is messy, but I am not sure which uncertainty matters most.",
    )
    result = planner.plan(state, simulations=2, max_rollout_depth=1)

    legal_action_ids = {action.id for action in env.legal_actions(state)}
    assert result.chosen_action_id in legal_action_ids
    assert result.search_p1_utterance
    assert result.p1_utterance
    assert result.root_stats


def test_template_cli_plan_runs_with_mock_llm(tmp_path):
    result = runner.invoke(
        app,
        [
            "plan",
            "--game",
            str(TEMPLATE_GAME),
            "--mock-llm",
            "--run-dir",
            str(tmp_path / "template-run"),
        ],
    )

    assert result.exit_code == 0
    assert "Selected action" in result.stdout
    assert (tmp_path / "template-run" / "calls.jsonl").exists()
    assert (tmp_path / "template-run" / "tree.json").exists()

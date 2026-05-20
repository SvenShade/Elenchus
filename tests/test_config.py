from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_mcts.config import GameConfig
from tests.conftest import minimal_game_data, write_game


def test_game_config_loads_minimal_yaml(tmp_path):
    path = write_game(tmp_path / "game.yaml", minimal_game_data())
    config = GameConfig.load(path)

    assert config.metadata.name == "minimal"
    assert [action.id for action in config.actions] == ["clarify", "accept"]
    assert config.actions[0].metadata == {}
    assert config.players["P1"].role == "planner"


def test_game_config_accepts_action_card_metadata():
    data = minimal_game_data()
    data["actions"][0]["metadata"] = {
        "objective": "clarify",
        "target_types": ["claim"],
        "diagnostic_operator": "disconfirmation_probe",
        "safety_constraints": ["preserve autonomy"],
    }

    config = GameConfig.model_validate(data)

    assert config.actions[0].metadata["diagnostic_operator"] == "disconfirmation_probe"


def test_game_config_rejects_missing_required_prompt():
    data = minimal_game_data()
    del data["prompts"]["judge_rollout_state"]

    with pytest.raises(ValidationError):
        GameConfig.model_validate(data)


def test_game_config_accepts_optional_exploration_summary_prompt():
    data = minimal_game_data()
    data["prompts"]["exploration_summary"] = "Summarize {{exploration_evidence}}"

    config = GameConfig.model_validate(data)

    assert config.prompts.exploration_summary == "Summarize {{exploration_evidence}}"


def test_game_config_accepts_optional_state_analysis_prompt():
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = "Analyze {{transcript}}"

    config = GameConfig.model_validate(data)

    assert config.prompts.state_analysis == "Analyze {{transcript}}"


def test_game_config_accepts_optional_candidate_actions_prompt():
    data = minimal_game_data()
    data["prompts"]["candidate_actions"] = "Generate {{action_catalog}}"

    config = GameConfig.model_validate(data)

    assert config.prompts.candidate_actions == "Generate {{action_catalog}}"


def test_game_config_accepts_progressive_widening_config():
    data = minimal_game_data()
    data["mcts"]["progressive_widening"] = {
        "enabled": True,
        "k": 2.5,
        "alpha": 0.4,
        "initial_children": 3,
        "expand_batch": 2,
        "max_candidates": 12,
        "candidate_refresh_visits": [0, 3, 9],
    }

    config = GameConfig.model_validate(data)

    widening = config.mcts.progressive_widening
    assert widening.enabled is True
    assert widening.k == 2.5
    assert widening.initial_children == 3
    assert widening.candidate_refresh_visits == [0, 3, 9]


def test_game_config_accepts_optional_rollout_reflection_prompts():
    data = minimal_game_data()
    data["prompts"]["rollout_reflection"] = "Reflect on {{rollout_evidence}}"
    data["prompts"]["finalize_p1_move"] = "Use {{rollout_reflection}}\nP1: "

    config = GameConfig.model_validate(data)

    assert config.prompts.rollout_reflection == "Reflect on {{rollout_evidence}}"
    assert config.prompts.finalize_p1_move == "Use {{rollout_reflection}}\nP1: "

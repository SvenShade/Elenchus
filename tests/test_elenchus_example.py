from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.mcts import MCTSPlanner
from llm_mcts.mock import DeterministicMockLLM


ELENCHUS_GAME = Path("examples/elenchus_v0_1.yaml")


class RecordingTurnLLM:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []
        self.json_prompts: list[tuple[str, str]] = []

    def complete(self, *, prompt_name: str, prompt: str, **kwargs: Any) -> str:
        self.prompts.append((prompt_name, prompt))
        if prompt_name == "realize_p1_move":
            return "What, precisely, are you protecting by keeping that claim vague?"
        return "I think I am avoiding having to admit what I already know."

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.json_prompts.append((prompt_name, "\n".join(message["content"] for message in messages)))
        if prompt_name == "rollout_reflection":
            return {
                "summary": "Projection evidence suggests a possible evasion.",
                "promising_questions": ["What would test the claim?"],
            }
        return kwargs.get("fallback", {})


def test_elenchus_example_loads_with_expected_shape():
    config = GameConfig.load(ELENCHUS_GAME)

    assert config.metadata.name == "Elenchus, the Mirror V0.1"
    assert config.metadata.max_real_turns == 8
    assert config.initial_state.transcript == []
    assert config.mcts.simulations == 12
    assert config.mcts.max_rollout_depth == 2
    assert config.mcts.progressive_widening.enabled is True
    assert config.llm.turn_max_tokens == 96
    assert config.prompts.state_analysis
    assert config.prompts.candidate_actions
    assert config.prompts.rollout_reflection
    assert config.prompts.finalize_p1_move
    assert len(config.players["P1"].private_context["cognitive_failure_library"]) == 23

    action_ids = {action.id for action in config.actions}
    assert {
        "definition_probe",
        "disconfirmation_probe",
        "fresh_start_test",
        "stakes_separation",
        "double_crux_probe",
        "source_tagging",
    }.issubset(action_ids)
    assert all(action.metadata for action in config.actions)

    judge_prompt = config.prompts.judge_rollout_state
    assert "hidden_premise_exposed" in judge_prompt
    assert "manipulation_risk" in judge_prompt
    assert "rhetorical_victory_without_understanding" in judge_prompt
    assert "pseudo-diagnosis" in judge_prompt
    assert "P1-private state sketch" in judge_prompt


def test_elenchus_turn_prompts_preserve_continuation_and_observation_boundaries():
    config = GameConfig.load(ELENCHUS_GAME)
    llm = RecordingTurnLLM()
    env = TwoPlayerConversationEnv(config, llm)
    state = env.initial_state().append(
        "P2",
        "I say I want to write, but I keep choosing distractions.",
    )
    action = next(action for action in env.legal_actions(state) if action.id == "stakes_separation")

    env.realize_p1_move(state, action)
    p1_prompt = llm.prompts[-1][1]

    assert p1_prompt.endswith("P1: ")
    assert "Return only valid JSON" not in p1_prompt
    assert "stakes_separation" in p1_prompt
    assert "failure_family_ids" in p1_prompt
    assert "P1-private state sketch" in p1_prompt
    assert "belief_graph" in p1_prompt
    assert "cognitive_failure_library" in p1_prompt
    assert "This is dialectic reflection, not therapy" in p1_prompt

    after_p1 = state.append(
        "P1",
        "What reward do you get from keeping the writing desire untested?",
        action_id=action.id,
    )
    env.imagine_p2_reply(after_p1)
    p2_prompt = llm.prompts[-1][1]

    assert p2_prompt.endswith("P2: ")
    assert "P2 profile:" in p2_prompt
    assert "Return only valid JSON" not in p2_prompt
    assert "Reflective user seeking sharper self-awareness" in p2_prompt
    assert "The true user context is not known in advance" not in p2_prompt
    assert "cognitive_failure_library" not in p2_prompt
    assert "P1-private state sketch" not in p2_prompt
    assert "failure_hypotheses" not in p2_prompt
    assert "ACTION_ID:" not in p2_prompt

    env.reflect_on_rollouts(after_p1, {"branch_samples": []})
    reflection_prompt = llm.json_prompts[-1][1]

    assert "Rollout evidence:" in reflection_prompt
    assert "The true user context is not known in advance" not in reflection_prompt

    env.finalize_p1_move(after_p1, action, "Original search turn.", {"summary": "projection"})
    finalize_prompt = llm.prompts[-1][1]

    assert finalize_prompt.endswith("P1: ")
    assert "Original search turn." in finalize_prompt
    assert "label_policy" in finalize_prompt
    assert "The true user context is not known in advance" not in finalize_prompt


def test_elenchus_example_runs_mock_mcts_plan():
    config = GameConfig.load(ELENCHUS_GAME)
    env = TwoPlayerConversationEnv(config, DeterministicMockLLM())
    planner = MCTSPlanner(env)

    state = env.initial_state().append(
        "P2",
        "I keep insisting I am being realistic, but I suspect I am just afraid to try.",
    )
    result = planner.plan(state, simulations=2, max_rollout_depth=1)

    legal_action_ids = {action.id for action in env.legal_actions(state)}
    assert result.chosen_action_id in legal_action_ids
    assert result.p1_utterance
    assert result.root_stats

from __future__ import annotations

import json
from typing import Any

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.mcts import MCTSPlanner
from llm_mcts.trace import TraceWriter
from tests.conftest import minimal_game_data


class DeterministicPlannerLLM:
    def complete(self, *, prompt_name: str, prompt: str, **kwargs) -> str:
        if prompt_name == "realize_p1_move":
            if "clarify - " in prompt:
                return "Clarify the cache and migration uncertainty."
            return "Accept the current direction."
        if prompt_name == "imagine_p2_reply":
            return "I can answer that and verify the next step."
        return ""

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs) -> Any:
        content = "\n".join(message["content"] for message in messages)
        if prompt_name == "policy_prior":
            return {"clarify": 0.8, "accept": 0.2}
        if prompt_name == "judge_rollout_state":
            utility = 1.0 if "Clarify the cache" in content else -0.25
            return {
                "utility": utility,
                "task_progress": utility,
                "coordination_quality": utility,
            }
        return kwargs.get("fallback", {})


def test_mcts_selects_high_value_action_and_writes_tree(tmp_path):
    data = minimal_game_data()
    data["actions"][0]["metadata"] = {
        "objective": "clarify",
        "diagnostic_operator": "disconfirmation_probe",
    }
    config = GameConfig.model_validate(data)
    tracer = TraceWriter(run_dir=tmp_path / "run", enabled=True)
    env = TwoPlayerConversationEnv(config, DeterministicPlannerLLM())
    planner = MCTSPlanner(env, tracer=tracer)

    result = planner.plan(env.initial_state())

    assert result.chosen_action_id == "clarify"
    assert "Clarify the cache" in result.p1_utterance
    assert result.estimated_utility > 0
    assert tracer.tree_path.exists()
    clarify_stats = next(item for item in result.root_stats if item["action_id"] == "clarify")
    assert clarify_stats["action_metadata"]["diagnostic_operator"] == "disconfirmation_probe"
    tree = json.loads(tracer.tree_path.read_text(encoding="utf-8"))
    clarify_node = next(node for node in tree["nodes"] if node["action_id"] == "clarify")
    assert clarify_node["action_metadata"]["objective"] == "clarify"

from __future__ import annotations

from typing import Any

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.mcts import MCTSPlanner
from tests.conftest import minimal_game_data


class WideningLLM:
    def complete(self, *, prompt_name: str, prompt: str, **kwargs: Any) -> str:
        if prompt_name == "imagine_p2_reply":
            return "I can answer that."
        return "What detail would change the next step?"

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if prompt_name == "state_analysis":
            return {
                "belief_graph": {"claims": [{"id": "C1", "text": "Unclear next step"}]},
                "claim_types": [{"target_id": "C1", "type": "strategic"}],
                "failure_hypotheses": [{"id": "H1", "family_id": "D5", "target_id": "C1"}],
                "latent_user_particles": [{"id": "U1", "cluster": "uncertain"}],
            }
        if prompt_name == "policy_prior":
            return {"clarify": 0.6, "accept": 0.4}
        if prompt_name == "candidate_actions":
            return {
                "candidates": [
                    {
                        "base_action_id": "clarify",
                        "target_ids": ["C1"],
                        "failure_hypothesis_ids": ["H1"],
                        "diagnostic_operator": "ask_missing_detail",
                        "repair_operator": "make_specific",
                        "dialogue_form": "question",
                        "prior": 0.6,
                        "compatibility_score": 0.8,
                    },
                    {
                        "base_action_id": "accept",
                        "target_ids": ["C1"],
                        "failure_hypothesis_ids": ["H1"],
                        "diagnostic_operator": "commit",
                        "repair_operator": "confirm_step",
                        "dialogue_form": "summary",
                        "prior": 0.4,
                        "compatibility_score": 0.5,
                    },
                ]
            }
        if prompt_name == "judge_rollout_state":
            return {"utility": 0.5, "specificity": 0.7}
        return kwargs.get("fallback", {})


class ReflectingWideningLLM(WideningLLM):
    def __init__(self) -> None:
        self.reflection_prompt = ""

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if prompt_name == "rollout_reflection":
            self.reflection_prompt = messages[0]["content"]
            return {
                "summary": "Deferred passages remain visible.",
                "deferred_passages": [{"passage": "accept", "note": "unopened"}],
                "refused_passages": [{"passage": "unsafe", "note": "refused"}],
                "factor_insights": ["question forms performed best"],
            }
        return super().chat_json(prompt_name=prompt_name, messages=messages, **kwargs)


def widening_game() -> GameConfig:
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = "Analyze {{transcript}}"
    data["prompts"]["candidate_actions"] = "Generate:\n{{p1_state_analysis}}\n{{action_catalog}}"
    data["mcts"]["simulations"] = 3
    data["mcts"]["max_rollout_depth"] = 1
    data["mcts"]["progressive_widening"] = {
        "enabled": True,
        "k": 2.0,
        "alpha": 0.5,
        "initial_children": 1,
        "expand_batch": 1,
        "max_candidates": 4,
        "candidate_refresh_visits": [0, 2],
    }
    return GameConfig.model_validate(data)


def test_progressive_widening_adds_candidates_incrementally():
    env = TwoPlayerConversationEnv(widening_game(), WideningLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    result = planner.plan(env.initial_state(), observer=events.append)

    root_candidate_adds = [
        event for event in events if event["type"] == "candidate_added" and event["node_id"] == 0
    ]
    assert [event["candidate"]["base_action_id"] for event in root_candidate_adds] == [
        "clarify",
        "accept",
    ]
    assert root_candidate_adds[0]["candidate_rank"] == 1
    assert root_candidate_adds[0]["widening_limit"] >= 1
    assert "parent_visits" in root_candidate_adds[0]
    assert "expanded_children" in root_candidate_adds[0]
    assert any(event["type"] == "state_analyzed" for event in events)
    assert any(event["type"] == "candidate_pool_generated" for event in events)
    assert any(event["type"] == "widening_limit_updated" for event in events)
    assert any(event["type"] == "abstract_stats" for event in events)
    assert result.chosen_candidate
    assert result.chosen_candidate["base_action_id"] in {"clarify", "accept"}


def test_abstract_stats_update_during_backup():
    env = TwoPlayerConversationEnv(widening_game(), WideningLLM())
    planner = MCTSPlanner(env)

    planner.plan(env.initial_state())

    compact = planner.stats_store.compact()
    assert any(key.startswith("candidate:") for key in compact)
    assert "base_action:clarify" in compact
    assert "diagnostic_operator:ask_missing_detail" in compact


def test_rollout_reflection_evidence_includes_deferred_candidate_space():
    config_data = widening_game().model_dump(mode="json")
    config_data["prompts"]["rollout_reflection"] = "Reflect on {{rollout_evidence}}"
    config = GameConfig.model_validate(config_data)
    llm = ReflectingWideningLLM()
    env = TwoPlayerConversationEnv(config, llm)
    planner = MCTSPlanner(env)

    result = planner.plan(env.initial_state(), simulations=2)

    assert "candidate_space" in llm.reflection_prompt
    assert "top_deferred_root_candidates" in llm.reflection_prompt
    assert "abstract_stats" in llm.reflection_prompt
    assert result.rollout_reflection
    assert result.rollout_reflection["deferred_passages"]
    assert result.rollout_reflection["factor_insights"] == ["question forms performed best"]

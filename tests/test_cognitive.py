from __future__ import annotations

from typing import Any

from llm_mcts.cognitive import normalize_cognitive_state
from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from tests.conftest import minimal_game_data


class CandidateLLM:
    def __init__(self) -> None:
        self.json_prompts: list[tuple[str, str]] = []

    def complete(self, **kwargs: Any) -> str:
        return "A focused utterance."

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        content = "\n".join(message["content"] for message in messages)
        self.json_prompts.append((prompt_name, content))
        if prompt_name == "state_analysis":
            return {
                "belief_graph": {
                    "claims": [{"id": "C1", "text": "The next step is unclear."}],
                    "edges": [{"source": "E1", "target": "C1", "type": "supports"}],
                },
                "claim_types": [{"target_id": "C1", "type": "strategic"}],
                "failure_hypotheses": [{"id": "H1", "family_id": "D5", "target_id": "C1"}],
                "latent_user_particles": [{"id": "P1", "cluster": "uncertain_collaborator"}],
                "action_affordances": [{"action_id": "clarify", "fit": 0.9}],
            }
        if prompt_name == "policy_prior":
            return {"clarify": 0.7, "accept": 0.3}
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
                        "prior": 2.0,
                        "compatibility_score": 0.9,
                    },
                    {
                        "base_action_id": "not_legal",
                        "prior": 1.0,
                    },
                    {
                        "base_action_id": "accept",
                        "prior": 1.0,
                        "safety_flags": ["unsafe"],
                    },
                ]
            }
        return kwargs.get("fallback", {})


def test_typed_cognitive_state_accepts_partial_llm_json():
    state = normalize_cognitive_state(
        {
            "belief_graph": {
                "claims": [{"id": "C1", "text": "A claim."}],
                "edges": [{"source": "E1", "target": "C1", "type": "supports"}],
            },
            "failure_hypotheses": [{"id": "H1", "family_id": "F4"}],
        },
        state_id="abc",
    )

    assert state.source_state_id == "abc"
    assert state.belief_graph.claims[0].id == "C1"
    assert state.belief_graph.edges[0].type == "supports"
    assert state.failure_hypotheses[0].family_id == "F4"
    assert state.summary()["belief_graph"]["claims"][0]["text"] == "A claim."


def test_candidate_generation_filters_invalid_candidates_and_stabilizes_ids():
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = "Analyze {{transcript}}"
    data["prompts"]["candidate_actions"] = "Candidates:\n{{p1_state_analysis}}\n{{action_catalog}}"
    config = GameConfig.model_validate(data)
    env = TwoPlayerConversationEnv(config, CandidateLLM())
    state = env.initial_state()

    first = env.candidate_actions(state)
    second = env.candidate_actions(state)

    assert [candidate.base_action_id for candidate in first] == ["clarify"]
    assert first[0].candidate_id == second[0].candidate_id
    assert first[0].target_ids == ["C1"]
    assert first[0].prior == 1.0
    rejections = env.last_candidate_rejections[state.state_hash()]
    assert {item["reason"] for item in rejections} == {
        "unknown_base_action",
        "unsafe_candidate",
    }


def test_cognitive_memory_is_private_to_p1_context():
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = "Analyze {{transcript}}\n{{p1_cognitive_memory}}"
    config = GameConfig.model_validate(data)
    env = TwoPlayerConversationEnv(config, CandidateLLM())
    state = env.initial_state().append("P1", "Question?", action_id="clarify")

    env.remember_cognitive_state(state)

    p1_context = env.p1_planning_context(state)
    p2_context = env.actual_p2_context(state)
    assert "The next step is unclear" in p1_context["p1_cognitive_memory"]
    assert "The next step is unclear" not in str(p2_context)

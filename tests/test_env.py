from __future__ import annotations

import json

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv, clean_transcript_completion
from tests.conftest import minimal_game_data


class TinyLLM:
    def complete(self, **kwargs):
        return kwargs.get("fallback", "")

    def chat_json(self, **kwargs):
        return kwargs.get("fallback", {})


class RecordingTurnLLM:
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def complete(self, *, prompt_name: str, prompt: str, **kwargs):
        self.prompts.append((prompt_name, prompt))
        return f"{'P1' if prompt_name == 'realize_p1_move' else 'P2'}: cleaned turn"

    def chat_json(self, **kwargs):
        return kwargs.get("fallback", {})


class StateAnalysisLLM:
    def __init__(self):
        self.json_prompts: list[tuple[str, str]] = []

    def complete(self, **kwargs):
        return "cleaned turn"

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs):
        content = "\n".join(message["content"] for message in messages)
        self.json_prompts.append((prompt_name, content))
        if prompt_name == "state_analysis":
            return {
                "belief_graph": {"claims": [{"id": "C1", "text": "Needs clarity"}]},
                "claim_types": [{"target_id": "C1", "type": "strategic", "confidence": 0.7}],
                "failure_hypotheses": [
                    {
                        "id": "H1",
                        "family_id": "F4_confirmation_myside_filtering",
                        "target_id": "C1",
                    }
                ],
                "affective_state": {"reactance_risk": "low"},
                "safety_state": {"constraints": ["preserve autonomy"]},
                "action_affordances": [{"action_id": "clarify", "fit": 0.8}],
            }
        return kwargs.get("fallback", {})


def test_p1_planning_context_excludes_true_p2_private_context():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, TinyLLM())

    context = env.p1_planning_context(env.initial_state())
    serialized = json.dumps(context)

    assert "SECRET_TOKEN" not in serialized
    assert "must-not-leak" not in serialized
    assert "needs bounded next step" in serialized
    assert "visible-to-p1" in serialized


def test_state_analysis_is_p1_private_and_cached():
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = (
        "Public transcript:\n{{transcript}}\n"
        "P1 notes:\n{{p1_profile}}\n"
        "P1 beliefs:\n{{p1_beliefs_about_p2}}"
    )
    config = GameConfig.model_validate(data)
    llm = StateAnalysisLLM()
    env = TwoPlayerConversationEnv(config, llm)
    state = env.initial_state()

    first_context = env.p1_planning_context(state)
    second_context = env.p1_planning_context(state)
    p2_context = env.actual_p2_context(state)

    assert "Needs clarity" in first_context["p1_state_analysis"]
    assert "F4_confirmation_myside_filtering" in second_context["p1_state_analysis"]
    assert len([name for name, _ in llm.json_prompts if name == "state_analysis"]) == 1
    assert "Needs clarity" not in json.dumps(p2_context)
    assert "SECRET_TOKEN" not in llm.json_prompts[0][1]
    assert "must-not-leak" not in llm.json_prompts[0][1]


def test_rollout_reflection_memory_clears_state_analysis_cache():
    data = minimal_game_data()
    data["prompts"]["state_analysis"] = "Reflections:\n{{p1_rollout_reflections}}"
    config = GameConfig.model_validate(data)
    llm = StateAnalysisLLM()
    env = TwoPlayerConversationEnv(config, llm)
    state = env.initial_state()

    env.p1_planning_context(state)
    env.remember_rollout_reflection({"summary": "New private reflection."})
    env.p1_planning_context(state)

    state_analysis_prompts = [content for name, content in llm.json_prompts if name == "state_analysis"]
    assert len(state_analysis_prompts) == 2
    assert "New private reflection." in state_analysis_prompts[-1]


def test_action_catalog_and_selected_prompt_include_action_card_metadata():
    data = minimal_game_data()
    data["actions"][0]["metadata"] = {
        "objective": "clarify",
        "diagnostic_operator": "disconfirmation_probe",
    }
    data["prompts"]["realize_p1_move"] = (
        "Action metadata:\n{{action_metadata}}\n"
        "Recorded conversation:\n{{dialogue_transcript}}\nP1: "
    )
    config = GameConfig.model_validate(data)
    llm = RecordingTurnLLM()
    env = TwoPlayerConversationEnv(config, llm)
    state = env.initial_state()
    action = env.legal_actions(state)[0]

    context = env.p1_planning_context(state)
    assert "- clarify:" in context["action_catalog"]
    assert "disconfirmation_probe" in context["action_catalog"]
    assert "disconfirmation_probe" in env.p1_planning_context(state, action)["action_metadata"]

    env.realize_p1_move(state, action)
    p1_prompt = llm.prompts[-1][1]

    assert "disconfirmation_probe" in p1_prompt
    assert '"objective": "clarify"' in p1_prompt


def test_manual_p2_response_appends_verbatim():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, TinyLLM())
    state = env.initial_state().append("P1", "Can you verify the cache path?", action_id="clarify")

    next_state = env.append_manual_p2_response(state, "Manual answer with odd punctuation?!")

    assert next_state.transcript[-1].speaker == "P2"
    assert next_state.transcript[-1].content == "Manual answer with odd punctuation?!"
    assert next_state.transcript[-1].imagined is False


def test_completion_cleaning_formats_without_rewriting():
    assert (
        clean_transcript_completion('P2: I can check that. It might be cache order.', "P2", "P1")
        == "I can check that. It might be cache order."
    )
    assert (
        clean_transcript_completion("I can check that.\nP1: Great.", "P2", "P1")
        == "I can check that."
    )
    assert (
        clean_transcript_completion("```json\n{\"utterance\": \"P1: Let's inspect the failing branch.\"}\n```", "P1", "P2")
        == "Let's inspect the failing branch."
    )
    assert (
        clean_transcript_completion(
            "Could you describe the failing path?\n\nP1's next move:\nP1: Explain why.",
            "P1",
            "P2",
        )
        == "Could you describe the failing path?"
    )
    assert (
        clean_transcript_completion("What would test that belief?\nElenchus: Answer me.", "P1", "P2")
        == "What would test that belief?"
    )
    assert clean_transcript_completion("P1:", "P2", "P1", fallback="fallback") == "fallback"


def test_turn_generation_prompts_are_transcript_continuations():
    config = GameConfig.model_validate(minimal_game_data())
    llm = RecordingTurnLLM()
    env = TwoPlayerConversationEnv(config, llm)
    state = env.initial_state()
    action = env.legal_actions(state)[0]

    env.realize_p1_move(state, action)
    p1_prompt = llm.prompts[-1][1]

    assert p1_prompt.endswith("P1: ")
    assert "Return only valid JSON" not in p1_prompt
    assert "P2: I am unsure where to patch." in p1_prompt

    after_p1 = state.append("P1", "Let's inspect the failing branch.", action_id="clarify")
    env.imagine_p2_reply(after_p1)
    p2_prompt = llm.prompts[-1][1]

    assert p2_prompt.endswith("P2: ")
    assert "Return only valid JSON" not in p2_prompt
    assert "SECRET_TOKEN" not in p2_prompt
    assert "must-not-leak" not in p2_prompt
    assert "needs bounded next step" in p2_prompt


def test_rollout_reflection_memory_is_private_to_p1_context():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, TinyLLM())
    env.remember_rollout_reflection(
        {
            "summary": "Clarify the projected evasion.",
            "promising_questions": ["What would change your mind?"],
        }
    )
    state = env.initial_state()

    p1_context = env.p1_planning_context(state)
    p2_context = env.actual_p2_context(state)

    assert "Clarify the projected evasion" in p1_context["p1_rollout_reflections"]
    assert "Clarify the projected evasion" not in json.dumps(p2_context)

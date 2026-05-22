from __future__ import annotations

import threading
from typing import Any

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.mcts import MCTSCancelled, MCTSPlanner
from tests.conftest import minimal_game_data


class ObserverLLM:
    def complete(self, *, prompt_name: str, prompt: str, **kwargs) -> str:
        if prompt_name == "realize_p1_move":
            return "Clarify now." if "clarify" in prompt else "Accept now."
        if prompt_name == "imagine_p2_reply":
            return "P2 imagined reply."
        return ""

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs) -> Any:
        if prompt_name == "policy_prior":
            return {"clarify": 0.6, "accept": 0.4}
        if prompt_name == "judge_rollout_state":
            return {"utility": 0.5, "task_progress": 0.5}
        return kwargs.get("fallback", {})


class ReflectionLLM:
    def complete(self, *, prompt_name: str, prompt: str, **kwargs) -> str:
        if prompt_name == "realize_p1_move":
            return "Search utterance: clarify the protected claim."
        if prompt_name == "finalize_p1_move":
            return "Final utterance: what exact claim are you protecting from a test?"
        if prompt_name == "imagine_p2_reply":
            return "I might be avoiding the falsifiable version."
        return ""

    def chat_json(self, *, prompt_name: str, messages: list[dict[str, str]], **kwargs) -> Any:
        if prompt_name == "policy_prior":
            return {"clarify": 0.9, "accept": 0.1}
        if prompt_name == "judge_rollout_state":
            return {"utility": 0.6, "specificity": 0.7}
        if prompt_name == "rollout_reflection":
            return {
                "summary": "Projection suggests P2 avoids a testable claim.",
                "interlocutor_hypotheses": [
                    {"hypothesis": "P2 may protect self-image.", "node_ids": [1]}
                ],
                "projected_contradictions": ["P2 wants clarity but resists falsification."],
                "branch_sensitivities": ["Clarify branches produce more specificity."],
                "promising_questions": ["What would falsify the safer story?"],
                "overreach_risks": ["Do not assert motive as fact."],
                "next_move_guidance": "Ask one discriminating question.",
            }
        return kwargs.get("fallback", {})


def test_mcts_observer_emits_ordered_events_and_honors_simulation_override():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    result = planner.plan(env.initial_state(), simulations=2, observer=events.append)

    event_types = [event["type"] for event in events]
    assert event_types[0] == "run_started"
    assert event_types.count("node_evaluated") == 2
    assert event_types.count("backup") >= 2
    assert event_types[-1] == "plan_completed"
    assert result.chosen_action_id in {"clarify", "accept"}
    assert result.chosen_node_id is not None
    assert result.chosen_path == [0, result.chosen_node_id]
    completed = events[-1]
    assert completed["chosen_node_id"] == result.chosen_node_id
    assert completed["chosen_path"] == result.chosen_path


def test_mcts_plan_honors_depth_override():
    data = minimal_game_data()
    data["mcts"]["max_rollout_depth"] = 1
    config = GameConfig.model_validate(data)
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    planner.plan(env.initial_state(), simulations=1, max_rollout_depth=3, observer=events.append)

    run_started = next(event for event in events if event["type"] == "run_started")
    rollout_depths = [event["depth"] for event in events if event["type"] == "rollout_step"]
    assert run_started["max_rollout_depth"] == 3
    assert max(rollout_depths) == 3


def test_split_depth_materializes_tree_then_extends_hidden_rollout():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    planner.plan(
        env.initial_state(),
        simulations=4,
        max_tree_depth=3,
        rollout_extension_depth=2,
        observer=events.append,
    )

    run_started = next(event for event in events if event["type"] == "run_started")
    node_depths = [event["depth"] for event in events if event["type"] == "node_added"]
    rollout_depths = [event["depth"] for event in events if event["type"] == "rollout_step"]
    rollout_offsets = [event["rollout_offset"] for event in events if event["type"] == "rollout_step"]

    assert run_started["depth_mode"] == "split"
    assert run_started["labyrinth_depth"] == 3
    assert run_started["lantern_range"] == 2
    assert run_started["max_rollout_depth"] == 5
    assert max(node_depths) == 3
    assert max(rollout_depths) == 5
    assert max(rollout_offsets) == 2


def test_split_depth_can_evaluate_leaf_without_hidden_rollout():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    planner.plan(
        env.initial_state(),
        simulations=1,
        max_tree_depth=1,
        rollout_extension_depth=0,
        observer=events.append,
    )

    assert not [event for event in events if event["type"] == "rollout_step"]
    evaluated = next(event for event in events if event["type"] == "node_evaluated")
    assert evaluated["leaf_depth"] == 1
    assert evaluated["rollout_extension_depth"] == 0


def test_split_depth_keeps_tree_shallow_while_judging_hidden_future():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    planner.plan(
        env.initial_state(),
        simulations=2,
        max_tree_depth=1,
        rollout_extension_depth=2,
        observer=events.append,
    )

    node_depths = [event["depth"] for event in events if event["type"] == "node_added"]
    rollout_depths = [event["depth"] for event in events if event["type"] == "rollout_step"]
    assert max(node_depths) == 1
    assert max(rollout_depths) == 3


def test_mcts_rollout_reflection_finalizes_p1_without_rewriting_search_stats():
    data = minimal_game_data()
    data["prompts"]["rollout_reflection"] = "Reflect:\n{{rollout_evidence}}"
    data["prompts"]["finalize_p1_move"] = (
        "Reflection:\n{{rollout_reflection}}\n"
        "Original:\n{{search_p1_utterance}}\n"
        "P1: "
    )
    config = GameConfig.model_validate(data)
    env = TwoPlayerConversationEnv(config, ReflectionLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    result = planner.plan(env.initial_state(), simulations=2, observer=events.append)

    event_types = [event["type"] for event in events]
    assert event_types.index("rollout_reflection_started") < event_types.index("rollout_reflection")
    assert event_types.index("rollout_reflection") < event_types.index("final_p1_move_started")
    assert event_types.index("final_p1_move") < event_types.index("plan_completed")
    assert result.chosen_action_id == "clarify"
    assert result.search_p1_utterance.startswith("Search utterance")
    assert result.p1_utterance.startswith("Final utterance")
    assert result.rollout_reflection
    clarify_stats = next(item for item in result.root_stats if item["action_id"] == "clarify")
    assert clarify_stats["p1_utterance"].startswith("Search utterance")
    assert clarify_stats["visits"] > 0


def test_mcts_can_skip_finalization_for_reflexion_first_pass():
    data = minimal_game_data()
    data["prompts"]["rollout_reflection"] = "Reflect:\n{{rollout_evidence}}"
    data["prompts"]["finalize_p1_move"] = (
        "Reflection:\n{{rollout_reflection}}\n"
        "P1: "
    )
    config = GameConfig.model_validate(data)
    env = TwoPlayerConversationEnv(config, ReflectionLLM())
    planner = MCTSPlanner(env)
    events: list[dict[str, Any]] = []

    result = planner.plan(env.initial_state(), simulations=1, observer=events.append, finalize=False)

    event_types = [event["type"] for event in events]
    assert "rollout_reflection" in event_types
    assert "final_p1_move" not in event_types
    assert result.p1_utterance == result.search_p1_utterance
    assert result.rollout_reflection


def test_mcts_honors_cancellation_event():
    config = GameConfig.model_validate(minimal_game_data())
    env = TwoPlayerConversationEnv(config, ObserverLLM())
    planner = MCTSPlanner(env)
    cancel_event = threading.Event()
    events: list[dict[str, Any]] = []

    def observer(event: dict[str, Any]) -> None:
        events.append(event)
        if event["type"] == "node_added":
            cancel_event.set()

    try:
        planner.plan(env.initial_state(), simulations=5, observer=observer, cancel_event=cancel_event)
    except MCTSCancelled:
        pass
    else:
        raise AssertionError("planner did not stop after cancellation")

    assert cancel_event.is_set()
    assert "plan_completed" not in [event["type"] for event in events]

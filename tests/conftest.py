from __future__ import annotations

from pathlib import Path

import yaml


def minimal_game_data() -> dict:
    return {
        "metadata": {"name": "minimal", "max_real_turns": 1},
        "llm": {"model": "mock"},
        "players": {
            "P1": {
                "role": "planner",
                "goals": ["coordinate"],
                "skills": ["planning"],
                "private_context": {"private_p1": "visible-to-p1"},
                "beliefs_about_teammate": {"uncertainty": "needs bounded next step"},
            },
            "P2": {
                "role": "responder",
                "goals": ["help"],
                "skills": ["implementation"],
                "private_context": {"SECRET_TOKEN": "must-not-leak"},
            },
        },
        "actions": [
            {
                "id": "clarify",
                "description": "Ask a targeted question.",
                "allowed_players": ["P1"],
                "realization_guidance": "Ask one precise question.",
            },
            {
                "id": "accept",
                "description": "Accept and proceed.",
                "allowed_players": ["P1"],
                "realization_guidance": "Confirm the next step.",
            },
        ],
        "prompts": {
            "policy_prior": "Task: {{task}}\nActions:\n{{action_catalog}}",
            "realize_p1_move": (
                "Public task:\n{{task}}\n\n"
                "Speaker notes available to P1:\n{{p1_profile}}\n\n"
                "P1's current model of P2:\n{{p1_beliefs_about_p2}}\n\n"
                "Strategic move assigned to P1's continuation:\n"
                "{{action_id}} - {{action_description}}\n{{action_guidance}}\n\n"
                "Recorded conversation:\n{{dialogue_transcript}}\nP1: "
            ),
            "imagine_p2_reply": (
                "Public task:\n{{task}}\n\n"
                "P2 profile:\n{{p1_model_of_p2}}\n\n"
                "Recorded conversation:\n{{dialogue_transcript}}\nP2: "
            ),
            "judge_rollout_state": "Transcript:\n{{transcript}}",
            "actual_p2_reply": (
                "Public task:\n{{task}}\n\n"
                "Recorded conversation:\n{{dialogue_transcript}}\nP2: "
            ),
        },
        "mcts": {
            "simulations": 4,
            "max_rollout_depth": 1,
            "c_puct": 1.5,
            "root_action_selection": "visits",
            "random_seed": 3,
            "trace": True,
        },
        "initial_state": {
            "task": "Coordinate a small bug fix.",
            "transcript": [{"speaker": "P2", "content": "I am unsure where to patch."}],
        },
    }


def write_game(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path

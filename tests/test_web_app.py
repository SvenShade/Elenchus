from __future__ import annotations

import time

from fastapi.testclient import TestClient

from llm_mcts.web_app import WebRuntimeConfig, create_app
from tests.conftest import minimal_game_data, write_game


def make_client(tmp_path):
    data = minimal_game_data()
    data["mcts"]["trace"] = False
    data["mcts"]["simulations"] = 2
    path = write_game(tmp_path / "game.yaml", data)
    app = create_app(
        WebRuntimeConfig(
            game_path=path,
            mock_llm=True,
            max_simulations=5,
        )
    )
    return TestClient(app)


def make_reflection_client(tmp_path):
    data = minimal_game_data()
    data["players"]["P1"]["role"] = "Elenchus dialectic examiner"
    data["mcts"]["trace"] = False
    data["mcts"]["simulations"] = 2
    data["prompts"]["rollout_reflection"] = (
        "Elenchus should reflect on projection evidence only.\n"
        "{{rollout_evidence}}"
    )
    data["prompts"]["finalize_p1_move"] = (
        "Elenchus should finalize without mentioning rollouts.\n"
        "{{rollout_reflection}}\n"
        "P1: "
    )
    path = write_game(tmp_path / "reflection_game.yaml", data)
    app = create_app(
        WebRuntimeConfig(
            game_path=path,
            mock_llm=True,
            max_simulations=5,
        )
    )
    return TestClient(app)


def test_session_create_and_manual_p2_append(tmp_path):
    client = make_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post(f"/api/sessions/{session_id}/p2", json={"content": "Manual reply"})

    assert response.status_code == 200
    transcript = response.json()["transcript"]
    assert transcript[-1]["speaker"] == "P2"
    assert transcript[-1]["content"] == "Manual reply"


def test_mocked_plan_appends_p1_and_websocket_receives_events(tmp_path):
    client = make_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]

    initial_summary = client.get(f"/api/sessions/{session_id}").json()
    assert initial_summary["depth_slider"] == {"min": 1, "max": 5, "default": 1}

    response = client.post(
        f"/api/sessions/{session_id}/plan",
        json={"simulations": 2, "max_rollout_depth": 3},
    )
    assert response.status_code == 200
    assert response.json()["max_rollout_depth"] == 3
    deadline = time.monotonic() + 5
    summary = client.get(f"/api/sessions/{session_id}").json()
    while summary["planning"] and time.monotonic() < deadline:
        time.sleep(0.05)
        summary = client.get(f"/api/sessions/{session_id}").json()

    event_count = len(client.app.state.sessions.get(session_id).event_log)
    seen = []
    with client.websocket_connect(f"/api/sessions/{session_id}/events") as websocket:
        for _ in range(event_count):
            event = websocket.receive_json()
            seen.append(event["type"])

    assert "node_added" in seen
    assert "plan_completed" in seen
    assert "exploration_summary" in seen
    assert "planning_finished" in seen
    session = client.app.state.sessions.get(session_id)
    run_started = next(event for event in session.event_log if event["type"] == "run_started")
    assert run_started["max_rollout_depth"] == 3
    assert seen.index("plan_completed") < seen.index("exploration_summary")
    assert seen.index("exploration_summary") < seen.index("planning_finished")
    summary_event = next(event for event in session.event_log if event["type"] == "exploration_summary")
    assert "clarifying uncertainty" in summary_event["summary"]
    assert summary_event["themes"]
    assert summary["transcript"][-1]["speaker"] == "P1"


def test_reflection_plan_events_and_memory_are_used_by_web_session(tmp_path):
    client = make_reflection_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/plan",
        json={"simulations": 2, "max_rollout_depth": 1},
    )
    assert response.status_code == 200
    deadline = time.monotonic() + 5
    summary = client.get(f"/api/sessions/{session_id}").json()
    while summary["planning"] and time.monotonic() < deadline:
        time.sleep(0.05)
        summary = client.get(f"/api/sessions/{session_id}").json()

    session = client.app.state.sessions.get(session_id)
    seen = [event["type"] for event in session.event_log]

    assert "rollout_reflection" in seen
    assert "final_p1_move" in seen
    assert "exploration_summary" in seen
    assert "planning_finished" in seen
    assert seen.index("rollout_reflection") < seen.index("plan_completed")
    assert seen.index("final_p1_move") < seen.index("plan_completed")
    assert "separate realism from avoidance" in summary["transcript"][-1]["content"]
    assert session.env.p1_rollout_reflections
    context = session.env.p1_planning_context(session.state)
    assert "simulated routes" in context["p1_rollout_reflections"]


def test_reflexion_runs_second_tree_from_first_reflection(tmp_path):
    client = make_reflection_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]

    initial_summary = client.get(f"/api/sessions/{session_id}").json()
    assert initial_summary["reflexion"] == {"available": True, "default": False}

    response = client.post(
        f"/api/sessions/{session_id}/plan",
        json={"simulations": 2, "max_rollout_depth": 1, "reflexion": True},
    )
    assert response.status_code == 200
    assert response.json()["reflexion"] is True
    deadline = time.monotonic() + 5
    summary = client.get(f"/api/sessions/{session_id}").json()
    while summary["planning"] and time.monotonic() < deadline:
        time.sleep(0.05)
        summary = client.get(f"/api/sessions/{session_id}").json()

    session = client.app.state.sessions.get(session_id)
    seen = [event["type"] for event in session.event_log]

    assert seen.count("run_started") == 2
    assert seen.count("rollout_reflection") == 2
    assert seen.count("plan_completed") == 1
    assert "reflexion_pass_completed" in seen
    assert "reflexion_replan_started" in seen
    assert seen.index("reflexion_pass_completed") < seen.index("reflexion_replan_started")
    assert seen.index("reflexion_replan_started") < seen.index("plan_completed")
    assert len(session.env.p1_rollout_reflections) == 2
    assert "separate realism from avoidance" in summary["transcript"][-1]["content"]


def test_planning_lock_rejects_concurrent_plan(tmp_path):
    client = make_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]
    client.app.state.sessions.get(session_id).planning = True

    response = client.post(f"/api/sessions/{session_id}/plan", json={"simulations": 5})

    assert response.status_code == 409


def test_cancel_endpoint_marks_active_plan_for_cancellation(tmp_path):
    client = make_client(tmp_path)
    session_id = client.post("/api/sessions").json()["session_id"]
    session = client.app.state.sessions.get(session_id)
    session.planning = True
    session.current_run_id = "run-cancel"

    response = client.post(f"/api/sessions/{session_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert session.cancel_event.is_set()
    assert session.cancel_reason == "browser closed or refreshed"
    assert client.app.state.sessions.get(session_id).event_log[-1]["type"] == "planning_cancelling"


def test_cors_allows_non_default_local_vite_port(tmp_path):
    client = make_client(tmp_path)

    response = client.options(
        "/api/sessions",
        headers={
            "Origin": "http://127.0.0.1:5177",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5177"

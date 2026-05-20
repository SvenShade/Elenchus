from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_mcts.config import GameConfig
from llm_mcts.env import TwoPlayerConversationEnv
from llm_mcts.llm import LLMClient
from llm_mcts.mcts import MCTSCancelled, MCTSPlanner
from llm_mcts.mock import DeterministicMockLLM
from llm_mcts.state import ConversationState
from llm_mcts.trace import TraceWriter


DEFAULT_EXPLORATION_SUMMARY_PROMPT = """
Summarize what P1's MCTS exploration discovered about possible cooperative
conversation Passages. Focus on simulated conversations, themes, interesting
routes, and potential conflicts. Use only the provided task, transcript, P1
visible context, and search evidence.

Task:
{{task}}

Transcript before P1's selected move:
{{transcript}}

P1 profile:
{{p1_profile}}

P1 beliefs about P2:
{{p1_beliefs_about_p2}}

Search evidence:
{{exploration_evidence}}

Return only valid JSON with this shape:
{
  "summary": "...",
  "themes": ["..."],
  "interesting_paths": ["..."],
  "potential_conflicts": ["..."]
}
""".strip()


class CreateSessionResponse(BaseModel):
    session_id: str


class PlanRequest(BaseModel):
    simulations: int = Field(ge=1)
    max_rollout_depth: int = Field(default=2, ge=1, le=5)
    reflexion: bool = False


class P2Request(BaseModel):
    content: str = Field(min_length=1)


@dataclass
class WebRuntimeConfig:
    game_path: Path
    base_url: str | None = None
    model: str | None = None
    mock_llm: bool = False
    max_simulations: int = 128
    web_origin: str | None = None


@dataclass
class SessionState:
    id: str
    config: GameConfig
    state: ConversationState
    env: TwoPlayerConversationEnv
    tracer: TraceWriter
    max_simulations: int
    planning: bool = False
    current_run_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str | None = None
    event_log: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[asyncio.Queue[dict[str, Any]]] = field(default_factory=list)
    mutex: threading.Lock = field(default_factory=threading.Lock)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "planning": self.planning,
            "current_run_id": self.current_run_id,
            "transcript": [turn.model_dump() for turn in self.state.transcript],
            "config": {
                "name": self.config.metadata.name,
                "description": self.config.metadata.description,
                "actions": [
                    {"id": action.id, "description": action.description}
                    for action in self.config.actions
                ],
            },
            "slider": {
                "min": 1,
                "max": self.max_simulations,
                "default": min(self.config.mcts.simulations, self.max_simulations),
            },
            "depth_slider": {
                "min": 1,
                "max": 5,
                "default": min(max(1, self.config.mcts.max_rollout_depth), 5),
            },
            "reflexion": {
                "available": bool(self.config.prompts.rollout_reflection),
                "default": False,
            },
        }


class SessionStore:
    def __init__(self, runtime: WebRuntimeConfig):
        self.runtime = runtime
        self.sessions: dict[str, SessionState] = {}

    def create_session(self) -> SessionState:
        config = _load_runtime_config(self.runtime)
        session_id = str(uuid4())
        tracer = TraceWriter(enabled=config.mcts.trace)
        llm = DeterministicMockLLM(tracer=tracer) if self.runtime.mock_llm else LLMClient(
            config.llm,
            tracer=tracer,
        )
        env = TwoPlayerConversationEnv(config, llm)
        session = SessionState(
            id=session_id,
            config=config,
            state=env.initial_state(),
            env=env,
            tracer=tracer,
            max_simulations=max(1, self.runtime.max_simulations),
        )
        self.sessions[session_id] = session
        session.event_log.append(_event("session_created", session_id=session_id))
        session.event_log.append(_event("transcript_updated", transcript=session.summary()["transcript"]))
        return session

    def get(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session


def create_app(runtime: WebRuntimeConfig) -> FastAPI:
    app = FastAPI(title="llm-mcts web")
    allow_origins = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    if runtime.web_origin:
        allow_origins.append(runtime.web_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=r"^http://(127\.0\.0\.1|localhost):\d+$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = SessionStore(runtime)
    app.state.sessions = store

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    async def create_session() -> CreateSessionResponse:
        session = store.create_session()
        return CreateSessionResponse(session_id=session.id)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        return store.get(session_id).summary()

    @app.post("/api/sessions/{session_id}/p2")
    async def append_p2(session_id: str, request: P2Request) -> dict[str, Any]:
        session = store.get(session_id)
        with session.mutex:
            if session.planning:
                raise HTTPException(status_code=409, detail="cannot append P2 while planning")
            session.state = session.env.append_manual_p2_response(session.state, request.content)
            transcript = [turn.model_dump() for turn in session.state.transcript]
        await publish(session, _event("transcript_updated", transcript=transcript))
        return session.summary()

    @app.post("/api/sessions/{session_id}/plan")
    async def plan(session_id: str, request: PlanRequest) -> dict[str, Any]:
        session = store.get(session_id)
        simulations = min(max(1, request.simulations), session.max_simulations)
        max_rollout_depth = min(max(1, request.max_rollout_depth), 5)
        run_id = str(uuid4())
        with session.mutex:
            if session.planning:
                raise HTTPException(status_code=409, detail="planning already in progress")
            session.cancel_event = threading.Event()
            session.cancel_reason = None
            session.planning = True
            session.current_run_id = run_id

        await publish(
            session,
            _event(
                "planning_started",
                run_id=run_id,
                simulations=simulations,
                max_rollout_depth=max_rollout_depth,
                reflexion=request.reflexion,
            ),
        )
        loop = asyncio.get_running_loop()
        threading.Thread(
            target=_run_plan_thread,
            args=(session, simulations, max_rollout_depth, bool(request.reflexion), run_id, loop),
            daemon=True,
        ).start()
        return {
            "status": "started",
            "run_id": run_id,
            "simulations": simulations,
            "max_rollout_depth": max_rollout_depth,
            "reflexion": bool(request.reflexion),
        }

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel(session_id: str) -> dict[str, Any]:
        session = store.get(session_id)
        requested = _request_cancel(session, "browser closed or refreshed")
        if requested:
            await publish(
                session,
                _event(
                    "planning_cancelling",
                    run_id=session.current_run_id,
                    reason=session.cancel_reason,
                ),
            )
        return {"status": "cancelling" if requested else "idle"}

    @app.websocket("/api/sessions/{session_id}/events")
    async def events(websocket: WebSocket, session_id: str) -> None:
        session = store.get(session_id)
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        session.subscribers.append(queue)
        try:
            for event in session.event_log:
                await websocket.send_json(event)
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            if queue in session.subscribers:
                session.subscribers.remove(queue)
            if not session.subscribers and session.planning:
                requested = _request_cancel(session, "browser event stream disconnected")
                if requested:
                    await publish(
                        session,
                        _event(
                            "planning_cancelling",
                            run_id=session.current_run_id,
                            reason=session.cancel_reason,
                        ),
                    )

    return app


def _run_plan_thread(
    session: SessionState,
    simulations: int,
    max_rollout_depth: int,
    reflexion: bool,
    run_id: str,
    loop: asyncio.AbstractEventLoop,
) -> None:
    plan_events: list[dict[str, Any]] = []
    current_pass = 1
    reflexion_available = bool(session.config.prompts.rollout_reflection)
    planned_passes = 2 if reflexion and reflexion_available else 1

    def observer(event: dict[str, Any]) -> None:
        nonlocal current_pass
        if session.cancel_event.is_set():
            raise MCTSCancelled(session.cancel_reason or "planning cancelled")
        event_type = event["type"]
        payload = _without_type(event)
        payload["reflexion_pass"] = current_pass
        payload["reflexion_passes"] = planned_passes
        forwarded_type = (
            "reflexion_pass_completed"
            if event_type == "plan_completed" and current_pass < planned_passes
            else event_type
        )
        recorded = {"type": forwarded_type, **payload}
        plan_events.append(recorded)
        publish_from_thread(
            session,
            _event(forwarded_type, run_id=run_id, **payload),
            loop,
        )

    try:
        planner = MCTSPlanner(session.env, tracer=session.tracer)
        state_snapshot = session.state.copy()
        result = planner.plan(
            state_snapshot,
            simulations=simulations,
            max_rollout_depth=max_rollout_depth,
            observer=observer,
            finalize=planned_passes == 1,
            cancel_event=session.cancel_event,
        )
        if session.cancel_event.is_set():
            raise MCTSCancelled(session.cancel_reason or "planning cancelled")
        if planned_passes > 1 and result.rollout_reflection:
            session.env.remember_rollout_reflection(result.rollout_reflection)
            publish_from_thread(
                session,
                _event(
                    "reflexion_replan_started",
                    run_id=run_id,
                    reflexion_pass=2,
                    reflexion_passes=planned_passes,
                ),
                loop,
            )
            current_pass = 2
            planner = MCTSPlanner(session.env, tracer=session.tracer)
            result = planner.plan(
                state_snapshot,
                simulations=simulations,
                max_rollout_depth=max_rollout_depth,
                observer=observer,
                finalize=True,
                cancel_event=session.cancel_event,
            )
        if session.cancel_event.is_set():
            raise MCTSCancelled(session.cancel_reason or "planning cancelled")
        if result.rollout_reflection:
            session.env.remember_rollout_reflection(result.rollout_reflection)
            exploration_summary = _exploration_summary_from_reflection(result.rollout_reflection)
        else:
            exploration_summary = _summarize_exploration(
                session,
                state_snapshot,
                result,
                plan_events,
            )
        publish_from_thread(
            session,
            _event("exploration_summary", run_id=run_id, **exploration_summary),
            loop,
        )
        with session.mutex:
            session.state = session.state.append(
                "P1",
                result.p1_utterance,
                action_id=result.chosen_action_id,
                imagined=False,
            )
            session.env.remember_cognitive_state(session.state)
            transcript = [turn.model_dump() for turn in session.state.transcript]
        publish_from_thread(
            session,
            _event(
                "transcript_updated",
                run_id=run_id,
                transcript=transcript,
            ),
            loop,
        )
        publish_from_thread(
            session,
            _event(
                "planning_finished",
                run_id=run_id,
                chosen_action_id=result.chosen_action_id,
                chosen_candidate=result.chosen_candidate,
                chosen_node_id=result.chosen_node_id,
                chosen_path=result.chosen_path,
                p1_utterance=result.p1_utterance,
                search_p1_utterance=result.search_p1_utterance,
                estimated_utility=result.estimated_utility,
                root_stats=result.root_stats,
                rollout_reflection=result.rollout_reflection,
            ),
            loop,
        )
        with session.mutex:
            session.planning = False
            session.current_run_id = None
            session.cancel_reason = None
    except MCTSCancelled as exc:
        publish_from_thread(
            session,
            _event(
                "planning_cancelled",
                run_id=run_id,
                reason=session.cancel_reason or str(exc),
            ),
            loop,
        )
        with session.mutex:
            session.planning = False
            session.current_run_id = None
    except Exception as exc:
        if session.cancel_event.is_set():
            publish_from_thread(
                session,
                _event(
                    "planning_cancelled",
                    run_id=run_id,
                    reason=session.cancel_reason or repr(exc),
                ),
                loop,
            )
            with session.mutex:
                session.planning = False
                session.current_run_id = None
            return
        publish_from_thread(
            session,
            _event("planning_failed", run_id=run_id, error=repr(exc)),
            loop,
        )
        with session.mutex:
            session.planning = False
            session.current_run_id = None


def _request_cancel(session: SessionState, reason: str) -> bool:
    with session.mutex:
        if not session.planning or session.cancel_event.is_set():
            return False
        session.cancel_reason = reason
        session.cancel_event.set()
    close = getattr(session.env.llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    return True


async def publish(session: SessionState, event: dict[str, Any]) -> None:
    session.event_log.append(event)
    for queue in list(session.subscribers):
        await queue.put(event)


def publish_from_thread(
    session: SessionState,
    event: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> None:
    session.event_log.append(event)
    for queue in list(session.subscribers):
        loop.call_soon_threadsafe(queue.put_nowait, event)


def _load_runtime_config(runtime: WebRuntimeConfig) -> GameConfig:
    config = GameConfig.load(runtime.game_path)
    updates = {}
    if runtime.base_url:
        updates["base_url"] = runtime.base_url
    if runtime.model:
        updates["model"] = runtime.model
    if updates:
        config = config.model_copy(
            update={"llm": config.llm.model_copy(update=updates)}
        )
    return config


def _exploration_summary_from_reflection(reflection: dict[str, Any]) -> dict[str, Any]:
    promising_questions = _reflection_items_to_strings(reflection.get("promising_questions"))
    hypotheses = _reflection_items_to_strings(reflection.get("interlocutor_hypotheses"))
    contradictions = _reflection_items_to_strings(reflection.get("projected_contradictions"))
    sensitivities = _reflection_items_to_strings(reflection.get("branch_sensitivities"))
    overreach_risks = _reflection_items_to_strings(reflection.get("overreach_risks"))
    return {
        "summary": _string_or_default(
            reflection.get("summary"),
            "Rollout reflection completed, but no detailed reflection was available.",
        ),
        "themes": (hypotheses + sensitivities)[:6],
        "interesting_paths": promising_questions[:6],
        "potential_conflicts": (contradictions + overreach_risks)[:6],
    }


def _summarize_exploration(
    session: SessionState,
    state_snapshot: ConversationState,
    result: Any,
    plan_events: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = _compact_exploration_evidence(result, plan_events)
    fallback = {
        "summary": "Exploration completed, but no detailed summary was available.",
        "themes": [],
        "interesting_paths": [],
        "potential_conflicts": [],
    }
    prompt = session.config.prompts.exploration_summary or DEFAULT_EXPLORATION_SUMMARY_PROMPT
    variables = session.env.p1_planning_context(state_snapshot)
    variables["exploration_evidence"] = json.dumps(evidence, indent=2, ensure_ascii=True, default=str)
    rendered = _render_template(prompt, variables)
    try:
        data = session.env.llm.chat_json(
            prompt_name="exploration_summary",
            messages=[{"role": "user", "content": rendered}],
            state_id=state_snapshot.state_hash(),
            fallback=fallback,
        )
    except Exception as exc:
        data = fallback | {"error": repr(exc)}
    return _normalize_exploration_summary(data)


def _compact_exploration_evidence(result: Any, plan_events: list[dict[str, Any]]) -> dict[str, Any]:
    evaluations = [
        event
        for event in plan_events
        if event.get("type") == "node_evaluated"
    ]
    sorted_high = sorted(
        evaluations,
        key=lambda event: float(event.get("utility", 0.0)),
        reverse=True,
    )
    sorted_low = sorted(
        evaluations,
        key=lambda event: float(event.get("utility", 0.0)),
    )
    action_counts: dict[str, int] = {}
    for event in evaluations:
        for step in event.get("rollout_trace", []) or []:
            action_id = step.get("action_id")
            if isinstance(action_id, str):
                action_counts[action_id] = action_counts.get(action_id, 0) + 1

    return {
        "chosen_action_id": result.chosen_action_id,
        "estimated_utility": result.estimated_utility,
        "top_root_actions": result.root_stats[:8],
        "high_utility_rollouts": [_compact_evaluation(event) for event in sorted_high[:5]],
        "low_utility_rollouts": [_compact_evaluation(event) for event in sorted_low[:3]],
        "rollout_action_counts": dict(
            sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _compact_evaluation(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": event.get("node_id"),
        "utility": event.get("utility"),
        "rubric": event.get("rubric"),
        "rollout_trace": event.get("rollout_trace"),
    }


def _normalize_exploration_summary(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    return {
        "summary": _string_or_default(
            data.get("summary"),
            "Exploration completed, but no detailed summary was available.",
        ),
        "themes": _string_list(data.get("themes")),
        "interesting_paths": _string_list(data.get("interesting_paths")),
        "potential_conflicts": _string_list(data.get("potential_conflicts")),
    }


def _string_or_default(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _reflection_items_to_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
        elif isinstance(item, dict):
            compact = ", ".join(
                f"{key}: {val}"
                for key, val in item.items()
                if isinstance(val, (str, int, float, bool))
            )
            if compact:
                items.append(compact)
    return items


def _render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def _event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {"type": event_type, **payload}


def _without_type(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "type"}

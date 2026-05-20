from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from llm_mcts.cognitive import (
    CognitiveState,
    LiftedActionCandidate,
    candidate_from_action,
    cognitive_transposition_key,
    normalize_cognitive_state,
    normalize_lifted_candidates,
)
from llm_mcts.config import ActionConfig, GameConfig, PlayerConfig
from llm_mcts.state import ConversationState


class JSONLLM(Protocol):
    def complete(
        self,
        *,
        prompt_name: str,
        prompt: str,
        state_id: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        ...

    def chat_json(
        self,
        *,
        prompt_name: str,
        messages: list[dict[str, str]],
        state_id: str | None = None,
        fallback: Any | None = None,
        max_repairs: int = 1,
        max_tokens: int | None = None,
    ) -> Any:
        ...


@dataclass
class JudgeResult:
    utility: float
    rubric: dict[str, Any] = field(default_factory=dict)


class TwoPlayerConversationEnv:
    def __init__(self, config: GameConfig, llm: JSONLLM):
        self.config = config
        self.llm = llm
        self.p1_rollout_reflections: list[dict[str, Any]] = []
        self.p1_cognitive_memory: list[dict[str, Any]] = []
        self.max_reflection_memory = 3
        self.max_cognitive_memory = 3
        self._state_analysis_cache: dict[str, CognitiveState] = {}
        self._candidate_cache: dict[str, list[LiftedActionCandidate]] = {}
        self.last_candidate_rejections: dict[str, list[dict[str, Any]]] = {}

    def initial_state(self) -> ConversationState:
        return ConversationState(
            task=self.config.initial_state.task,
            transcript=list(self.config.initial_state.transcript),
        )

    def legal_actions(self, state: ConversationState, player: str = "P1") -> list[ActionConfig]:
        return [action for action in self.config.actions if player in action.allowed_players]

    def p1_planning_context(
        self,
        state: ConversationState,
        action: ActionConfig | None = None,
        candidate: LiftedActionCandidate | None = None,
    ) -> dict[str, str]:
        p1 = self.config.players["P1"]
        variables = self._public_context(state)
        variables.update(
            {
                "p1_profile": _compact_json(self._p1_visible_profile(p1)),
                "p1_profile_full": _compact_json(self._p1_visible_profile_full(p1)),
                "p1_beliefs_about_p2": _compact_json(p1.beliefs_about_teammate),
                "p1_rollout_reflections": _compact_json(
                    _compact_rollout_reflection_memory(
                        self.p1_rollout_reflections[-self.max_reflection_memory :]
                    )
                ),
                "p1_cognitive_memory": _compact_json(
                    _compact_cognitive_memory(
                        self.p1_cognitive_memory[-self.max_cognitive_memory :]
                    )
                ),
                "p1_state_analysis": _compact_json(
                    _compact_state_analysis_summary(self.analyze_state(state))
                ),
                "action_catalog": self._action_catalog(self.legal_actions(state, "P1")),
                "action_catalog_full": self._action_catalog_full(self.legal_actions(state, "P1")),
                "candidate_json": "{}",
                "action_metadata": "{}",
            }
        )
        if action is not None:
            variables.update(
                {
                    "action_id": action.id,
                    "action_json": _compact_json(action.model_dump()),
                    "action_metadata": _compact_json(action.metadata),
                    "action_description": action.description,
                    "action_guidance": action.realization_guidance,
                }
            )
        if candidate is not None:
            variables.update(
                {
                "candidate_id": candidate.candidate_id,
                    "candidate_json": _compact_json(_compact_candidate_for_prompt(candidate)),
                    "base_action_id": candidate.base_action_id,
                    "target_ids": _compact_json(candidate.target_ids),
                    "failure_hypothesis_ids": _compact_json(candidate.failure_hypothesis_ids),
                    "diagnostic_operator": candidate.diagnostic_operator,
                    "repair_operator": candidate.repair_operator,
                    "dialogue_form": candidate.dialogue_form,
                    "affect_strategy": candidate.affect_strategy,
                    "directness": str(candidate.directness),
                    "abstraction": str(candidate.abstraction),
                    "agency_policy": candidate.agency_policy,
                    "label_policy": candidate.label_policy,
                    "expected_observations": _compact_json(candidate.expected_observations[:3]),
                    "candidate_prior": str(candidate.prior),
                    "candidate_compatibility_score": str(candidate.compatibility_score),
                    "action_metadata": _compact_json(
                        _compact_action_metadata_for_prompt(candidate.action_metadata)
                    ),
                }
            )
        return variables

    def actual_p2_context(self, state: ConversationState) -> dict[str, str]:
        return self._public_context(state)

    def analyze_state(self, state: ConversationState) -> dict[str, Any]:
        return self.analyze_cognitive_state(state).summary()

    def analyze_cognitive_state(self, state: ConversationState) -> CognitiveState:
        if not self.config.prompts.state_analysis:
            return CognitiveState(source_state_id=state.state_hash())
        state_hash = state.state_hash()
        if state_hash in self._state_analysis_cache:
            return self._state_analysis_cache[state_hash]
        fallback = {
            "belief_graph": {
                "claims": [],
                "reasons": [],
                "evidence": [],
                "definitions": [],
                "values": [],
                "decisions": [],
                "memories": [],
                "tensions": [],
            },
            "discourse_state": {},
            "claim_types": [],
            "failure_hypotheses": [],
            "affective_state": {},
            "safety_state": {},
            "action_affordances": [],
        }
        data = self.llm.chat_json(
            prompt_name="state_analysis",
            messages=[
                {
                    "role": "user",
                    "content": self._render(
                        self.config.prompts.state_analysis,
                        self._state_analysis_context(state),
                    ),
                }
            ],
            state_id=state_hash,
            fallback=fallback,
            max_tokens=self._json_max_tokens(512),
        )
        analysis = normalize_cognitive_state(data, state_id=state_hash)
        self._state_analysis_cache[state_hash] = analysis
        return analysis

    def remember_cognitive_state(self, state: ConversationState) -> None:
        if state.transcript and state.transcript[-1].imagined:
            return
        analysis = self.analyze_cognitive_state(state).summary()
        self.p1_cognitive_memory.append(analysis)
        self.p1_cognitive_memory = self.p1_cognitive_memory[-self.max_cognitive_memory :]

    def candidate_actions(
        self,
        state: ConversationState,
        node_visits: int = 0,
        limit: int | None = None,
    ) -> list[LiftedActionCandidate]:
        state_hash = state.state_hash()
        refresh_bucket = _candidate_refresh_bucket(
            node_visits,
            self.config.mcts.progressive_widening.candidate_refresh_visits,
        )
        cache_key = f"{state_hash}:{refresh_bucket}:{limit or 'all'}"
        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]
        legal = self.legal_actions(state, "P1")
        priors = self.policy_prior(state)
        if not self.config.prompts.candidate_actions:
            candidates = [
                candidate_from_action(
                    action,
                    prior=priors.get(action.id, 0.0),
                    compatibility_score=priors.get(action.id, 0.0),
                )
                for action in legal
            ]
            candidates.sort(
                key=lambda item: (-item.prior, -item.compatibility_score, item.base_action_id)
            )
            result = candidates[:limit] if limit is not None else candidates
            self._candidate_cache[cache_key] = result
            return result

        cognitive_state = self.analyze_cognitive_state(state)
        variables = self.p1_planning_context(state)
        variables["node_visits"] = str(node_visits)
        variables["candidate_limit"] = str(limit or "")
        variables["p1_state_analysis"] = _compact_json(
            _compact_state_analysis_summary(cognitive_state.summary())
        )
        data = self.llm.chat_json(
            prompt_name="candidate_actions",
            messages=[
                {
                    "role": "user",
                    "content": self._render(
                        self.config.prompts.candidate_actions,
                        variables,
                    ),
                }
            ],
            state_id=state_hash,
            fallback={"candidates": []},
            max_tokens=self._json_max_tokens(1024),
        )
        candidates, rejections = normalize_lifted_candidates(
            data,
            legal_actions=legal,
            fallback_priors=priors,
        )
        if not candidates:
            candidates = [
                candidate_from_action(
                    action,
                    prior=priors.get(action.id, 0.0),
                    compatibility_score=priors.get(action.id, 0.0),
                    source="fallback_action",
                )
                for action in legal
            ]
        self.last_candidate_rejections[state_hash] = rejections
        candidates = candidates[:limit] if limit is not None else candidates
        self._candidate_cache[cache_key] = candidates
        return candidates

    def policy_prior(self, state: ConversationState) -> dict[str, float]:
        legal = self.legal_actions(state, "P1")
        fallback = {action.id: 1.0 for action in legal}
        data = self.llm.chat_json(
            prompt_name="policy_prior",
            messages=[
                {
                    "role": "user",
                    "content": self._render(
                        self.config.prompts.policy_prior,
                        self.p1_planning_context(state),
                    ),
                }
            ],
            state_id=state.state_hash(),
            fallback=fallback,
            max_tokens=self._json_max_tokens(384),
        )
        return self._normalize_priors(data, legal)

    def realize_p1_move(
        self,
        state: ConversationState,
        action: ActionConfig,
        candidate: LiftedActionCandidate | None = None,
    ) -> str:
        fallback = f"I suggest we take the '{action.id}' step next."
        prompt = self._turn_prompt(
            self.config.prompts.realize_p1_move,
            self.p1_planning_context(state, action=action, candidate=candidate),
            "P1",
        )
        raw = self.llm.complete(
            prompt_name="realize_p1_move",
            prompt=prompt,
            state_id=state.state_hash(),
            temperature=self.config.llm.turn_temperature,
            max_tokens=self.config.llm.turn_max_tokens,
            stop=_turn_stop_sequences("P1"),
        )
        return clean_transcript_completion(raw, "P1", "P2", fallback=fallback)

    def imagine_p2_reply(self, state_after_p1: ConversationState) -> str:
        p1 = self.config.players["P1"]
        variables = self.p1_planning_context(state_after_p1)
        variables["p1_model_of_p2"] = _json(p1.beliefs_about_teammate)
        fallback = "I understand. I will respond with the relevant detail."
        prompt = self._turn_prompt(
            self.config.prompts.imagine_p2_reply,
            variables,
            "P2",
        )
        raw = self.llm.complete(
            prompt_name="imagine_p2_reply",
            prompt=prompt,
            state_id=state_after_p1.state_hash(),
            temperature=self.config.llm.turn_temperature,
            max_tokens=self.config.llm.turn_max_tokens,
            stop=_turn_stop_sequences("P2"),
        )
        return clean_transcript_completion(raw, "P2", "P1", fallback=fallback)

    def actual_p2_reply(self, state_after_p1: ConversationState) -> str:
        fallback = "I understand. Here is my response."
        prompt = self._turn_prompt(
            self.config.prompts.actual_p2_reply,
            self.actual_p2_context(state_after_p1),
            "P2",
        )
        raw = self.llm.complete(
            prompt_name="actual_p2_reply",
            prompt=prompt,
            state_id=state_after_p1.state_hash(),
            temperature=self.config.llm.turn_temperature,
            max_tokens=self.config.llm.turn_max_tokens,
            stop=_turn_stop_sequences("P2"),
        )
        return clean_transcript_completion(raw, "P2", "P1", fallback=fallback)

    def reflect_on_rollouts(
        self,
        state: ConversationState,
        evidence: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.config.prompts.rollout_reflection:
            return None
        fallback = {
            "summary": "Rollout reflection completed, but no detailed reflection was available.",
            "interlocutor_hypotheses": [],
            "projected_contradictions": [],
            "branch_sensitivities": [],
            "promising_questions": [],
            "overreach_risks": [],
            "next_move_guidance": "",
        }
        variables = self.p1_planning_context(state)
        evidence_json = _compact_json(evidence)
        variables["rollout_evidence"] = evidence_json
        variables["exploration_evidence"] = evidence_json
        data = self.llm.chat_json(
            prompt_name="rollout_reflection",
            messages=[
                {
                    "role": "user",
                    "content": self._render(
                        self.config.prompts.rollout_reflection,
                        variables,
                    ),
                }
            ],
            state_id=state.state_hash(),
            fallback=fallback,
            max_tokens=self._json_max_tokens(768),
        )
        return normalize_rollout_reflection(data)

    def finalize_p1_move(
        self,
        state: ConversationState,
        action: ActionConfig,
        search_p1_utterance: str,
        rollout_reflection: dict[str, Any] | None,
        candidate: LiftedActionCandidate | None = None,
    ) -> str:
        if not self.config.prompts.finalize_p1_move:
            return search_p1_utterance
        variables = self.p1_planning_context(state, action=action, candidate=candidate)
        variables["search_p1_utterance"] = search_p1_utterance
        variables["original_p1_utterance"] = search_p1_utterance
        variables["rollout_reflection"] = _compact_json(
            _compact_rollout_reflection_memory([rollout_reflection or {}])[0]
            if rollout_reflection
            else {}
        )
        prompt = self._turn_prompt(
            self.config.prompts.finalize_p1_move,
            variables,
            "P1",
        )
        raw = self.llm.complete(
            prompt_name="finalize_p1_move",
            prompt=prompt,
            state_id=state.state_hash(),
            temperature=self.config.llm.turn_temperature,
            max_tokens=self.config.llm.turn_max_tokens,
            stop=_turn_stop_sequences("P1"),
        )
        return clean_transcript_completion(
            raw,
            "P1",
            "P2",
            fallback=search_p1_utterance,
        )

    def remember_rollout_reflection(self, reflection: dict[str, Any] | None) -> None:
        if not reflection:
            return
        self.p1_rollout_reflections.append(normalize_rollout_reflection(reflection))
        self.p1_rollout_reflections = self.p1_rollout_reflections[-self.max_reflection_memory :]
        self._state_analysis_cache.clear()
        self._candidate_cache.clear()

    def judge_state(self, state: ConversationState) -> JudgeResult:
        fallback = {
            "utility": 0.0,
            "task_progress": 0.0,
            "coordination_quality": 0.0,
            "information_gain": 0.0,
            "risk": 1.0,
            "conversational_cost": 1.0,
            "rationale": "Fallback judge result after invalid LLM output.",
        }
        data = self.llm.chat_json(
            prompt_name="judge_rollout_state",
            messages=[
                {
                    "role": "user",
                    "content": self._render(
                        self.config.prompts.judge_rollout_state,
                        self.p1_planning_context(state),
                    ),
                }
            ],
            state_id=state.state_hash(),
            fallback=fallback,
            max_tokens=self._json_max_tokens(512),
        )
        if not isinstance(data, dict):
            data = fallback
        utility = _safe_float(data.get("utility"), 0.0)
        rubric = {key: value for key, value in data.items() if key != "utility"}
        return JudgeResult(utility=utility, rubric=rubric)

    def simulate_p1_p2(
        self,
        state: ConversationState,
        action: ActionConfig,
        imagined: bool = True,
    ) -> tuple[ConversationState, str, str]:
        p1_utterance = self.realize_p1_move(state, action)
        after_p1 = state.append(
            "P1",
            p1_utterance,
            action_id=action.id,
            imagined=imagined,
        )
        p2_reply = self.imagine_p2_reply(after_p1)
        after_p2 = after_p1.append("P2", p2_reply, imagined=imagined)
        return after_p2, p1_utterance, p2_reply

    def simulate_candidate_p1_p2(
        self,
        state: ConversationState,
        candidate: LiftedActionCandidate,
        imagined: bool = True,
    ) -> tuple[ConversationState, str, str]:
        action = self._action_from_candidate(candidate)
        p1_utterance = self.realize_p1_move(state, action, candidate=candidate)
        after_p1 = state.append(
            "P1",
            p1_utterance,
            action_id=candidate.base_action_id,
            imagined=imagined,
        )
        p2_reply = self.imagine_p2_reply(after_p1)
        after_p2 = after_p1.append("P2", p2_reply, imagined=imagined)
        return after_p2, p1_utterance, p2_reply

    def append_manual_p2_response(
        self,
        state_after_p1: ConversationState,
        response: str,
    ) -> ConversationState:
        next_state = state_after_p1.append("P2", response, imagined=False)
        self.remember_cognitive_state(next_state)
        return next_state

    def _public_context(self, state: ConversationState) -> dict[str, str]:
        return {
            "task": state.task,
            "transcript": state.transcript_text(),
            "dialogue_transcript": state.dialogue_text(),
            "state_json": _json(state.model_dump()),
            "turn_index": str(state.current_turn_index),
        }

    def _state_analysis_context(self, state: ConversationState) -> dict[str, str]:
        p1 = self.config.players["P1"]
        variables = self._public_context(state)
        variables.update(
            {
                "p1_profile": _compact_json(self._p1_visible_profile(p1)),
                "p1_profile_full": _compact_json(self._p1_visible_profile_full(p1)),
                "p1_beliefs_about_p2": _compact_json(p1.beliefs_about_teammate),
                "p1_rollout_reflections": _compact_json(
                    _compact_rollout_reflection_memory(
                        self.p1_rollout_reflections[-self.max_reflection_memory :]
                    )
                ),
                "p1_cognitive_memory": _compact_json(
                    _compact_cognitive_memory(
                        self.p1_cognitive_memory[-self.max_cognitive_memory :]
                    )
                ),
                "action_catalog": self._action_catalog(self.legal_actions(state, "P1")),
                "action_catalog_full": self._action_catalog_full(self.legal_actions(state, "P1")),
            }
        )
        return variables

    def transposition_key(self, state: ConversationState) -> str:
        recent = [
            turn.action_id
            for turn in state.transcript[-2:]
            if turn.speaker == "P1" and turn.action_id
        ]
        return cognitive_transposition_key(self.analyze_cognitive_state(state), recent)

    def _p1_visible_profile(self, p1: PlayerConfig) -> dict[str, Any]:
        return {
            "role": p1.role,
            "goals": p1.goals[:4],
            "skills": p1.skills[:6],
            "hierarchy": p1.hierarchy,
            "private_context": _compact_private_context(p1.private_context),
            "communication_style": p1.communication_style,
        }

    def _p1_visible_profile_full(self, p1: PlayerConfig) -> dict[str, Any]:
        return {
            "role": p1.role,
            "goals": p1.goals,
            "skills": p1.skills,
            "hierarchy": p1.hierarchy,
            "private_context": _compact_private_context(p1.private_context),
            "communication_style": p1.communication_style,
            "beliefs_about_teammate": p1.beliefs_about_teammate,
        }

    def _action_catalog(self, actions: list[ActionConfig]) -> str:
        blocks: list[str] = []
        for action in actions:
            block = f"- {action.id}: {_clip_text(action.description, 56)}"
            guidance = _clip_text(action.realization_guidance, 40)
            if guidance:
                block += f" | hint={guidance}"
            if action.metadata:
                block += f" | {_action_metadata_line(action.metadata)}"
            blocks.append(block)
        return "\n".join(blocks)

    def _action_catalog_full(self, actions: list[ActionConfig]) -> str:
        blocks: list[str] = []
        for action in actions:
            line = (
                f"- {action.id}: {_clip_text(action.description, 72)}"
                f" | guide={_clip_text(action.realization_guidance, 56)}"
            )
            if action.metadata:
                line += f" | {_action_card_line(action.metadata)}"
            blocks.append(line)
        return "\n".join(blocks)

    def _action_from_candidate(self, candidate: LiftedActionCandidate) -> ActionConfig:
        for action in self.config.actions:
            if action.id == candidate.base_action_id:
                metadata = dict(action.metadata or {})
                metadata.update(candidate.action_metadata)
                return ActionConfig(
                    id=action.id,
                    description=candidate.description or action.description,
                    allowed_players=action.allowed_players,
                    realization_guidance=(
                        candidate.realization_guidance or action.realization_guidance
                    ),
                    metadata=metadata,
                )
        return candidate.to_action_config()

    def _normalize_priors(
        self,
        data: Any,
        legal_actions: list[ActionConfig],
    ) -> dict[str, float]:
        legal_ids = {action.id for action in legal_actions}
        raw: dict[str, float] = {}
        if isinstance(data, dict) and isinstance(data.get("priors"), list):
            for item in data["priors"]:
                if isinstance(item, dict):
                    action_id = item.get("action_id") or item.get("id")
                    probability = item.get("probability", item.get("prior", item.get("p", 0.0)))
                    if action_id in legal_ids:
                        raw[action_id] = _safe_float(probability, 0.0)
        elif isinstance(data, dict) and isinstance(data.get("actions"), list):
            for item in data["actions"]:
                if isinstance(item, dict):
                    action_id = item.get("action_id") or item.get("id")
                    probability = item.get("probability", item.get("prior", item.get("p", 0.0)))
                    if action_id in legal_ids:
                        raw[action_id] = _safe_float(probability, 0.0)
        elif isinstance(data, dict):
            for action_id, probability in data.items():
                if action_id in legal_ids:
                    raw[action_id] = _safe_float(probability, 0.0)

        cleaned = {action.id: max(0.0, raw.get(action.id, 0.0)) for action in legal_actions}
        total = sum(cleaned.values())
        if total <= 0:
            return {action.id: 1.0 / len(legal_actions) for action in legal_actions}
        return {action_id: value / total for action_id, value in cleaned.items()}

    def _render(self, template: str, variables: dict[str, str]) -> str:
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
        return rendered

    def _turn_prompt(self, template: str, variables: dict[str, str], target_speaker: str) -> str:
        prompt = self._render(template, variables)
        stripped = prompt.rstrip()
        if stripped.endswith(f"{target_speaker}:"):
            return stripped + " "
        return prompt

    def _json_max_tokens(self, minimum: int) -> int:
        return max(int(self.config.llm.max_tokens), minimum)


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True, default=str)


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, default=str)


def _compact_private_context(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value)

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key == "cognitive_failure_library" and isinstance(item, list):
            compact[key] = [_failure_card_id(card) for card in item]
        else:
            compact[key] = _compact_value(item)
    return compact


def _failure_card_id(card: Any) -> Any:
    if not isinstance(card, dict):
        return _compact_value(card)
    return card.get("id", "")


def _compact_rollout_reflection_memory(reflections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for reflection in reflections[-2:]:
        if not isinstance(reflection, dict):
            continue
        compact.append(
            {
                "summary": _clip_text(reflection.get("summary"), 220),
                "promising_questions": _compact_list(reflection.get("promising_questions"), 3, 120),
                "overreach_risks": _compact_list(reflection.get("overreach_risks"), 3, 120),
                "next_move_guidance": _clip_text(reflection.get("next_move_guidance"), 180),
                "deferred_passages": _compact_list(reflection.get("deferred_passages"), 3, 120),
                "refused_passages": _compact_list(reflection.get("refused_passages"), 3, 120),
                "factor_insights": _compact_list(reflection.get("factor_insights"), 3, 120),
            }
        )
    return compact


def _compact_cognitive_memory(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_state_analysis_summary(memory) for memory in memories[-2:]]


def _compact_state_analysis_summary(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, string_limit=120, list_limit=3, dict_limit=8)

    graph = value.get("belief_graph") if isinstance(value.get("belief_graph"), dict) else {}
    compact_graph: dict[str, Any] = {}
    for key in ("claims", "reasons", "evidence", "definitions", "values", "decisions", "memories", "tensions"):
        compact_graph[key] = _compact_nodes(graph.get(key), limit=3)
    compact_graph["edges"] = _compact_edges(graph.get("edges"), limit=5)

    return {
        "belief_graph": compact_graph,
        "discourse_state": _compact_value(
            value.get("discourse_state", {}),
            string_limit=100,
            list_limit=3,
            dict_limit=6,
        ),
        "claim_types": _compact_value(
            value.get("claim_types", [])[:5] if isinstance(value.get("claim_types"), list) else [],
            string_limit=80,
            list_limit=5,
            dict_limit=6,
        ),
        "failure_hypotheses": _compact_failure_hypotheses(value.get("failure_hypotheses")),
        "affective_state": _compact_value(
            value.get("affective_state", {}),
            string_limit=80,
            list_limit=3,
            dict_limit=6,
        ),
        "safety_state": _compact_value(
            value.get("safety_state", {}),
            string_limit=100,
            list_limit=3,
            dict_limit=6,
        ),
        "latent_user_particles": _compact_latent_particles(value.get("latent_user_particles")),
        "action_affordances": _compact_action_affordances(value.get("action_affordances")),
    }


def _compact_candidate_for_prompt(candidate: Any) -> dict[str, Any]:
    payload = candidate.to_prompt_dict() if hasattr(candidate, "to_prompt_dict") else candidate
    if not isinstance(payload, dict):
        return {}
    return {
        "candidate_id": payload.get("candidate_id"),
        "base_action_id": payload.get("base_action_id"),
        "description": _clip_text(payload.get("description"), 120),
        "target_ids": _compact_list(payload.get("target_ids"), 3, 64),
        "failure_hypothesis_ids": _compact_list(payload.get("failure_hypothesis_ids"), 3, 64),
        "diagnostic_operator": _clip_text(payload.get("diagnostic_operator"), 64),
        "repair_operator": _clip_text(payload.get("repair_operator"), 64),
        "dialogue_form": _clip_text(payload.get("dialogue_form"), 48),
        "affect_strategy": _clip_text(payload.get("affect_strategy"), 48),
        "directness": payload.get("directness"),
        "abstraction": payload.get("abstraction"),
        "agency_policy": _clip_text(payload.get("agency_policy"), 80),
        "label_policy": _clip_text(payload.get("label_policy"), 80),
        "prior": payload.get("prior"),
        "compatibility_score": payload.get("compatibility_score"),
        "safety_flags": _compact_list(payload.get("safety_flags"), 3, 80),
        "expected_observations": _compact_list(payload.get("expected_observations"), 3, 100),
        "action_metadata": _compact_action_metadata_for_prompt(payload.get("action_metadata")),
    }


def _compact_action_metadata_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = (
        "objective",
        "target_types",
        "codex_quadrant",
        "failure_family_ids",
        "diagnostic_operator",
        "repair_operator",
        "dialogue_form",
        "affect_strategy",
        "directness",
        "abstraction",
        "cognitive_load",
        "agency_policy",
        "label_policy",
        "expected_user_signals",
        "safety_constraints",
        "contraindications",
    )
    compact: dict[str, Any] = {}
    for key in keep:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, list):
            compact[key] = [_clip_text(entry, 64) for entry in item[:3]]
        elif isinstance(item, str):
            compact[key] = _clip_text(item, 120)
        else:
            compact[key] = item
    return compact


def _compact_nodes(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: val
                for key, val in {
                    "id": _clip_text(item.get("id"), 48),
                    "text": _clip_text(item.get("text"), 140),
                    "type": _clip_text(item.get("type"), 48),
                    "confidence": item.get("confidence"),
                }.items()
                if val not in ("", None)
            }
        )
    return compact


def _compact_edges(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: val
                for key, val in {
                    "source": _clip_text(item.get("source"), 48),
                    "target": _clip_text(item.get("target"), 48),
                    "type": _clip_text(item.get("type"), 48),
                    "confidence": item.get("confidence"),
                }.items()
                if val not in ("", None)
            }
        )
    return compact


def _compact_failure_hypotheses(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: val
                for key, val in {
                    "id": _clip_text(item.get("id"), 48),
                    "family_id": _clip_text(item.get("family_id"), 80),
                    "target_id": _clip_text(item.get("target_id"), 48),
                    "confidence": item.get("confidence"),
                    "cues_for": _compact_list(item.get("cues_for"), 2, 80),
                    "diagnostic_tests": _compact_list(item.get("diagnostic_tests"), 2, 80),
                    "repair_operators": _compact_list(item.get("repair_operators"), 2, 80),
                    "user_facing_label_policy": _clip_text(
                        item.get("user_facing_label_policy"), 64
                    ),
                }.items()
                if val not in ("", None, [])
            }
        )
    return compact


def _compact_latent_particles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:2]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: val
                for key, val in {
                    "id": _clip_text(item.get("id"), 48),
                    "cluster": _clip_text(item.get("cluster"), 64),
                    "likely_failure_families": _compact_list(
                        item.get("likely_failure_families"), 3, 80
                    ),
                    "confidence": item.get("confidence"),
                }.items()
                if val not in ("", None, [])
            }
        )
    return compact


def _compact_action_affordances(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                key: val
                for key, val in {
                    "action_id": _clip_text(item.get("action_id"), 64),
                    "fit": item.get("fit"),
                    "reason": _clip_text(item.get("reason"), 100),
                    "target_ids": _compact_list(item.get("target_ids"), 3, 48),
                    "failure_hypothesis_ids": _compact_list(
                        item.get("failure_hypothesis_ids"), 3, 48
                    ),
                }.items()
                if val not in ("", None, [])
            }
        )
    return compact


def _compact_list(value: Any, limit: int, string_limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    compact: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact.append(
                _compact_value(item, string_limit=string_limit, list_limit=3, dict_limit=6)
            )
        else:
            compact.append(_clip_text(item, string_limit))
    return compact


def _action_metadata_line(metadata: dict[str, Any]) -> str:
    targets = _join_compact_list(metadata.get("target_types"), 1, item_limit=18)
    failures = _join_compact_list(metadata.get("failure_family_ids"), 1, item_limit=28)
    diagnostic = _clip_text(metadata.get("diagnostic_operator", ""), 22)
    repair = _clip_text(metadata.get("repair_operator", ""), 22)
    parts = [
        f"obj={_clip_text(metadata.get('objective', ''), 16)}",
        f"tgt={targets}",
        f"fail={failures}",
        f"op={diagnostic}>{repair}",
        f"form={_clip_text(metadata.get('dialogue_form', ''), 18)}",
    ]
    return "; ".join(str(part) for part in parts if str(part).strip())


def _action_card_line(metadata: dict[str, Any]) -> str:
    parts = [
        f"obj={_clip_text(metadata.get('objective', ''), 18)}",
        f"tgt={_join_compact_list(metadata.get('target_types'), 2, item_limit=18)}",
        f"fail={_join_compact_list(metadata.get('failure_family_ids'), 2, item_limit=30)}",
        (
            "op="
            f"{_clip_text(metadata.get('diagnostic_operator', ''), 24)}>"
            f"{_clip_text(metadata.get('repair_operator', ''), 24)}"
        ),
        f"form={_clip_text(metadata.get('dialogue_form', ''), 18)}",
        f"tone={_clip_text(metadata.get('affect_strategy', ''), 18)}",
        f"dir={metadata.get('directness', '')}",
        f"agency={_clip_text(metadata.get('agency_policy', ''), 24)}",
        f"label={_clip_text(metadata.get('label_policy', ''), 20)}",
        f"safety={_join_compact_list(metadata.get('safety_constraints'), 1, item_limit=28)}",
    ]
    return "; ".join(str(part) for part in parts if str(part).strip() and not str(part).endswith("="))


def _join_compact_list(value: Any, limit: int, *, item_limit: int = 30) -> str:
    if isinstance(value, list):
        return ",".join(_clip_text(item, item_limit) for item in value[:limit])
    if value is None:
        return ""
    return _clip_text(value, 48)


def _compact_value(
    value: Any,
    *,
    string_limit: int = 180,
    list_limit: int = 4,
    dict_limit: int = 10,
) -> Any:
    if isinstance(value, str):
        return _clip_text(value, string_limit)
    if isinstance(value, list):
        return [
            _compact_value(item, string_limit=string_limit, list_limit=list_limit)
            for item in value[:list_limit]
        ]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= dict_limit:
                break
            compact[str(key)] = _compact_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                dict_limit=dict_limit,
            )
        return compact
    return value


def _clip_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _candidate_refresh_bucket(visits: int, thresholds: list[int]) -> int:
    bucket = 0
    for threshold in sorted(thresholds):
        if visits >= threshold:
            bucket = threshold
    return bucket


def normalize_rollout_reflection(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    return {
        "summary": _string_or_default(
            data.get("summary"),
            "Rollout reflection completed, but no detailed reflection was available.",
        ),
        "interlocutor_hypotheses": _list_or_empty(data.get("interlocutor_hypotheses")),
        "projected_contradictions": _list_or_empty(data.get("projected_contradictions")),
        "branch_sensitivities": _list_or_empty(data.get("branch_sensitivities")),
        "promising_questions": _list_or_empty(data.get("promising_questions")),
        "overreach_risks": _list_or_empty(data.get("overreach_risks")),
        "next_move_guidance": _string_or_default(data.get("next_move_guidance"), ""),
        "deferred_passages": _list_or_empty(data.get("deferred_passages")),
        "refused_passages": _list_or_empty(data.get("refused_passages")),
        "factor_insights": _list_or_empty(data.get("factor_insights")),
    }


def normalize_state_analysis(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    return {
        "belief_graph": _dict_or_empty(data.get("belief_graph")),
        "discourse_state": _dict_or_empty(data.get("discourse_state")),
        "claim_types": _list_or_empty(data.get("claim_types")),
        "failure_hypotheses": _list_or_empty(data.get("failure_hypotheses")),
        "affective_state": _dict_or_empty(data.get("affective_state")),
        "safety_state": _dict_or_empty(data.get("safety_state")),
        "action_affordances": _list_or_empty(data.get("action_affordances")),
    }


def _string_or_default(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    cleaned: list[Any] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, dict):
            cleaned.append(item)
    return cleaned


def _utterance_from(data: Any, fallback: str) -> str:
    if isinstance(data, dict) and isinstance(data.get("utterance"), str):
        return data["utterance"].strip()
    if isinstance(data, str):
        return data.strip()
    return fallback


def clean_transcript_completion(
    raw: str,
    target_speaker: str,
    other_speaker: str,
    fallback: str = "",
) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _strip_code_fence(text)
    text = _json_utterance_or_original(text)
    text = _strip_wrapping_quotes(text.strip())
    text = _strip_leading_speaker_prefix(text, target_speaker)
    text = _truncate_at_speaker_marker(text, target_speaker, other_speaker)
    text = _truncate_at_non_dialogue_header(text, target_speaker, other_speaker)
    text = _strip_wrapping_quotes(text.strip())
    if not re.search(r"[A-Za-z0-9]", text):
        return fallback
    return text


def _turn_stop_sequences(target_speaker: str) -> list[str]:
    other_speaker = "P2" if target_speaker == "P1" else "P1"
    return [
        f"\n{other_speaker}:",
        "\nElenchus:",
        "\nUser:",
        "\nJudge:",
        "\nTranscript:",
        f"\n{target_speaker}'s next move:",
        f"\n{target_speaker}'s response:",
        f"\n{other_speaker}'s next move:",
        f"\n{other_speaker}'s response:",
    ]


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _json_utterance_or_original(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, dict) and isinstance(data.get("utterance"), str):
        return data["utterance"]
    if isinstance(data, str):
        return data
    return text


def _strip_wrapping_quotes(text: str) -> str:
    quote_pairs = [('"', '"'), ("'", "'"), ("“", "”")]
    stripped = text.strip()
    for left, right in quote_pairs:
        if stripped.startswith(left) and stripped.endswith(right) and len(stripped) >= 2:
            return stripped[1:-1].strip()
    return stripped


def _strip_leading_speaker_prefix(text: str, target_speaker: str) -> str:
    pattern = rf"^\s*(?:{re.escape(target_speaker)}\s*:\s*)+"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _truncate_at_speaker_marker(
    text: str,
    target_speaker: str,
    other_speaker: str,
) -> str:
    speaker_markers = {
        re.escape(target_speaker),
        re.escape(other_speaker),
        "Elenchus",
        "User",
    }
    pattern = rf"(?im)^\s*(?:{'|'.join(sorted(speaker_markers))})\s*:"
    match = re.search(pattern, text)
    if match:
        return text[: match.start()]
    return text


def _truncate_at_non_dialogue_header(
    text: str,
    target_speaker: str,
    other_speaker: str,
) -> str:
    speakers = f"{re.escape(target_speaker)}|{re.escape(other_speaker)}"
    pattern = rf"(?im)^\s*(?:{speakers})'s\s+(?:next\s+move|response)\s*:"
    match = re.search(pattern, text)
    if match:
        return text[: match.start()]
    return text


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

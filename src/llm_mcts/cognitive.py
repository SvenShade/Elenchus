from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_mcts.config import ActionConfig


class BeliefNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    text: str = ""
    type: str | None = None
    confidence: float | None = None


class BeliefEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = ""
    target: str = ""
    type: str = ""
    confidence: float | None = None


class BeliefGraph(BaseModel):
    model_config = ConfigDict(extra="allow")

    claims: list[BeliefNode] = Field(default_factory=list)
    reasons: list[BeliefNode] = Field(default_factory=list)
    evidence: list[BeliefNode] = Field(default_factory=list)
    definitions: list[BeliefNode] = Field(default_factory=list)
    values: list[BeliefNode] = Field(default_factory=list)
    decisions: list[BeliefNode] = Field(default_factory=list)
    memories: list[BeliefNode] = Field(default_factory=list)
    tensions: list[BeliefNode] = Field(default_factory=list)
    edges: list[BeliefEdge] = Field(default_factory=list)


class FailureHypothesis(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    family_id: str = ""
    codex_macro_problem: str = ""
    target_id: str = ""
    confidence: float = 0.0
    cues_for: list[Any] = Field(default_factory=list)
    cues_against: list[Any] = Field(default_factory=list)
    diagnostic_tests: list[str] = Field(default_factory=list)
    repair_operators: list[str] = Field(default_factory=list)
    contraindications: list[Any] = Field(default_factory=list)
    manipulation_risks: list[Any] = Field(default_factory=list)
    user_facing_label_policy: str = "never_name_bias"


class AffectiveState(BaseModel):
    model_config = ConfigDict(extra="allow")

    motivational_overlays: list[str] = Field(default_factory=list)
    reactance_risk: str = "unknown"
    shame_risk: str = "unknown"
    affect_band: str = "unknown"


class SafetyState(BaseModel):
    model_config = ConfigDict(extra="allow")

    constraints: list[Any] = Field(default_factory=list)
    overreach_risks: list[Any] = Field(default_factory=list)
    do_not_say: list[Any] = Field(default_factory=list)


class LatentUserParticle(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    cluster: str = "unknown"
    motivational_state: dict[str, Any] = Field(default_factory=dict)
    knowledge_state: dict[str, Any] = Field(default_factory=dict)
    likely_failure_families: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ActionAffordance(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_id: str = ""
    fit: float = 0.0
    reason: str = ""
    target_ids: list[str] = Field(default_factory=list)
    failure_hypothesis_ids: list[str] = Field(default_factory=list)


class CognitiveState(BaseModel):
    model_config = ConfigDict(extra="allow")

    belief_graph: BeliefGraph = Field(default_factory=BeliefGraph)
    discourse_state: dict[str, Any] = Field(default_factory=dict)
    claim_types: list[dict[str, Any]] = Field(default_factory=list)
    failure_hypotheses: list[FailureHypothesis] = Field(default_factory=list)
    affective_state: AffectiveState = Field(default_factory=AffectiveState)
    safety_state: SafetyState = Field(default_factory=SafetyState)
    latent_user_particles: list[LatentUserParticle] = Field(default_factory=list)
    action_affordances: list[ActionAffordance] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    source_state_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "belief_graph": {
                "claims": [node.model_dump(exclude_none=True) for node in self.belief_graph.claims[:6]],
                "evidence": [
                    node.model_dump(exclude_none=True) for node in self.belief_graph.evidence[:6]
                ],
                "values": [node.model_dump(exclude_none=True) for node in self.belief_graph.values[:4]],
                "decisions": [
                    node.model_dump(exclude_none=True) for node in self.belief_graph.decisions[:4]
                ],
                "memories": [
                    node.model_dump(exclude_none=True) for node in self.belief_graph.memories[:4]
                ],
                "tensions": [
                    node.model_dump(exclude_none=True) for node in self.belief_graph.tensions[:6]
                ],
                "edges": [
                    edge.model_dump(exclude_none=True) for edge in self.belief_graph.edges[:10]
                ],
            },
            "discourse_state": self.discourse_state,
            "claim_types": self.claim_types[:8],
            "failure_hypotheses": [
                hypothesis.model_dump(exclude_none=True)
                for hypothesis in self.failure_hypotheses[:8]
            ],
            "affective_state": self.affective_state.model_dump(exclude_none=True),
            "safety_state": self.safety_state.model_dump(exclude_none=True),
            "latent_user_particles": [
                particle.model_dump(exclude_none=True)
                for particle in self.latent_user_particles[:4]
            ],
            "action_affordances": [
                affordance.model_dump(exclude_none=True)
                for affordance in self.action_affordances[:8]
            ],
        }


class LiftedActionCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str = ""
    base_action_id: str
    description: str = ""
    realization_guidance: str = ""
    target_ids: list[str] = Field(default_factory=list)
    failure_hypothesis_ids: list[str] = Field(default_factory=list)
    diagnostic_operator: str = ""
    repair_operator: str = ""
    dialogue_form: str = ""
    affect_strategy: str = ""
    directness: float = 0.5
    abstraction: float = 0.5
    agency_policy: str = ""
    label_policy: str = "internal_only"
    prior: float = 0.0
    compatibility_score: float = 0.0
    safety_flags: list[str] = Field(default_factory=list)
    expected_observations: list[str] = Field(default_factory=list)
    action_metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "generated"

    def stable_identity_payload(self) -> dict[str, Any]:
        return {
            "base_action_id": self.base_action_id,
            "target_ids": self.target_ids,
            "failure_hypothesis_ids": self.failure_hypothesis_ids,
            "diagnostic_operator": self.diagnostic_operator,
            "repair_operator": self.repair_operator,
            "dialogue_form": self.dialogue_form,
            "affect_strategy": self.affect_strategy,
            "directness": round(float(self.directness), 2),
            "abstraction": round(float(self.abstraction), 2),
            "agency_policy": self.agency_policy,
            "label_policy": self.label_policy,
        }

    def with_stable_id(self) -> "LiftedActionCandidate":
        if self.candidate_id:
            return self
        payload = json.dumps(self.stable_identity_payload(), sort_keys=True, ensure_ascii=True)
        candidate_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return self.model_copy(update={"candidate_id": candidate_id})

    def to_prompt_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_action_config(self) -> ActionConfig:
        return ActionConfig(
            id=self.base_action_id,
            description=self.description or self.base_action_id,
            allowed_players=["P1"],
            realization_guidance=self.realization_guidance,
            metadata=self.action_metadata,
        )


def normalize_cognitive_state(data: Any, *, state_id: str | None = None) -> CognitiveState:
    raw = data if isinstance(data, dict) else {}
    payload = dict(raw)
    payload["raw"] = raw
    payload["source_state_id"] = state_id
    try:
        return CognitiveState.model_validate(payload)
    except Exception:
        return CognitiveState(raw=raw, source_state_id=state_id)


def candidate_from_action(
    action: ActionConfig,
    *,
    prior: float = 0.0,
    compatibility_score: float = 0.0,
    source: str = "static_action",
) -> LiftedActionCandidate:
    metadata = action.metadata or {}
    return LiftedActionCandidate(
        base_action_id=action.id,
        description=action.description,
        realization_guidance=action.realization_guidance,
        target_ids=_string_list(metadata.get("target_types")),
        failure_hypothesis_ids=[],
        diagnostic_operator=str(metadata.get("diagnostic_operator") or ""),
        repair_operator=str(metadata.get("repair_operator") or ""),
        dialogue_form=str(metadata.get("dialogue_form") or ""),
        affect_strategy=str(metadata.get("affect_strategy") or ""),
        directness=_safe_float(metadata.get("directness"), 0.5),
        abstraction=_safe_float(metadata.get("abstraction"), 0.5),
        agency_policy=str(metadata.get("agency_policy") or ""),
        label_policy=str(metadata.get("label_policy") or "internal_only"),
        prior=prior,
        compatibility_score=compatibility_score,
        safety_flags=_string_list(metadata.get("safety_constraints")),
        expected_observations=_string_list(metadata.get("expected_user_signals")),
        action_metadata=metadata,
        source=source,
    ).with_stable_id()


def normalize_lifted_candidates(
    data: Any,
    *,
    legal_actions: list[ActionConfig],
    fallback_priors: dict[str, float],
) -> tuple[list[LiftedActionCandidate], list[dict[str, Any]]]:
    legal = {action.id: action for action in legal_actions}
    raw_items = _candidate_items(data)
    candidates: list[LiftedActionCandidate] = []
    rejections: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in raw_items:
        if not isinstance(raw, dict):
            rejections.append({"reason": "candidate_not_object", "candidate": raw})
            continue
        base_action_id = (
            raw.get("base_action_id")
            or raw.get("action_id")
            or raw.get("id")
            or raw.get("mode")
            or ""
        )
        if base_action_id not in legal:
            rejections.append(
                {
                    "reason": "unknown_base_action",
                    "base_action_id": base_action_id,
                    "candidate": raw,
                }
            )
            continue
        safety_flags = _string_list(raw.get("safety_flags"))
        lowered_flags = {flag.lower() for flag in safety_flags}
        if {"unsafe", "reject", "rejected"} & lowered_flags:
            rejections.append(
                {
                    "reason": "unsafe_candidate",
                    "base_action_id": base_action_id,
                    "candidate": raw,
                }
            )
            continue

        action = legal[base_action_id]
        metadata = dict(action.metadata or {})
        metadata.update(_dict_or_empty(raw.get("action_metadata") or raw.get("metadata")))
        prior = _safe_float(raw.get("prior", raw.get("probability")), fallback_priors.get(base_action_id, 0.0))
        compatibility = _safe_float(
            raw.get("compatibility_score", raw.get("fit", raw.get("score"))),
            prior,
        )
        candidate = LiftedActionCandidate(
            candidate_id=str(raw.get("candidate_id") or ""),
            base_action_id=base_action_id,
            description=str(raw.get("description") or action.description),
            realization_guidance=str(raw.get("realization_guidance") or action.realization_guidance),
            target_ids=_string_list(raw.get("target_ids") or raw.get("targets")),
            failure_hypothesis_ids=_string_list(
                raw.get("failure_hypothesis_ids") or raw.get("failure_hypotheses")
            ),
            diagnostic_operator=str(
                raw.get("diagnostic_operator") or metadata.get("diagnostic_operator") or ""
            ),
            repair_operator=str(raw.get("repair_operator") or metadata.get("repair_operator") or ""),
            dialogue_form=str(raw.get("dialogue_form") or metadata.get("dialogue_form") or ""),
            affect_strategy=str(raw.get("affect_strategy") or metadata.get("affect_strategy") or ""),
            directness=_safe_float(raw.get("directness"), metadata.get("directness", 0.5)),
            abstraction=_safe_float(raw.get("abstraction"), metadata.get("abstraction", 0.5)),
            agency_policy=str(raw.get("agency_policy") or metadata.get("agency_policy") or ""),
            label_policy=str(raw.get("label_policy") or metadata.get("label_policy") or "internal_only"),
            prior=prior,
            compatibility_score=compatibility,
            safety_flags=safety_flags,
            expected_observations=_string_list(
                raw.get("expected_observations") or raw.get("expected_user_signals")
            ),
            action_metadata=metadata,
            source="generated",
        ).with_stable_id()
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        candidates.append(candidate)

    candidates = _normalize_candidate_priors(candidates)
    candidates.sort(key=lambda item: (-item.prior, -item.compatibility_score, item.base_action_id, item.candidate_id))
    return candidates, rejections


def _normalize_candidate_priors(
    candidates: list[LiftedActionCandidate],
) -> list[LiftedActionCandidate]:
    total = sum(max(0.0, candidate.prior) for candidate in candidates)
    if total <= 0:
        if not candidates:
            return []
        prior = 1.0 / len(candidates)
        return [candidate.model_copy(update={"prior": prior}) for candidate in candidates]
    return [
        candidate.model_copy(update={"prior": max(0.0, candidate.prior) / total})
        for candidate in candidates
    ]


def cognitive_transposition_key(state: CognitiveState, recent_move_bigram: list[str] | None = None) -> str:
    top_claim_type = _top_claim_type(state)
    tension_type = _first_node_type(state.belief_graph.tensions)
    failure_families = sorted(
        {hypothesis.family_id for hypothesis in state.failure_hypotheses if hypothesis.family_id}
    )[:2]
    affect_band = state.affective_state.affect_band or state.affective_state.reactance_risk or "unknown"
    particle_cluster = "unknown"
    if state.latent_user_particles:
        particle_cluster = state.latent_user_particles[0].cluster or "unknown"
    payload = {
        "target_type": top_claim_type,
        "tension_type": tension_type,
        "failure_family_top2": failure_families,
        "affect_band": affect_band,
        "user_particle_cluster": particle_cluster,
        "recent_move_bigram": (recent_move_bigram or [])[-2:],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _candidate_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("candidates", "actions", "lifted_actions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if "base_action_id" in data or "action_id" in data:
            return [data]
    return []


def _top_claim_type(state: CognitiveState) -> str:
    if state.claim_types:
        first = state.claim_types[0]
        if isinstance(first, dict):
            return str(first.get("type") or "unknown")
    for collection in (
        state.belief_graph.claims,
        state.belief_graph.decisions,
        state.belief_graph.memories,
    ):
        value = _first_node_type(collection)
        if value != "unknown":
            return value
    return "unknown"


def _first_node_type(nodes: list[BeliefNode]) -> str:
    for node in nodes:
        if node.type:
            return node.type
    return "unknown"


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

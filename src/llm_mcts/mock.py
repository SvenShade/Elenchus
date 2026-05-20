from __future__ import annotations

import json
import re
from typing import Any

from llm_mcts.trace import TraceWriter


class DeterministicMockLLM:
    """Small fake LLM for CLI smoke tests and demos without a live server."""

    def __init__(self, tracer: TraceWriter | None = None):
        self.tracer = tracer

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
        content = "\n".join(message.get("content", "") for message in messages)
        parsed = self._response(prompt_name, content, fallback)
        if self.tracer:
            self.tracer.record_call(
                prompt_name=prompt_name,
                messages=messages,
                raw_response=json.dumps(parsed),
                parsed=parsed,
                state_id=state_id,
            )
        return parsed

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
        raw = self._completion_response(prompt_name, prompt)
        if self.tracer:
            self.tracer.record_call(
                prompt_name=prompt_name,
                messages=[{"role": "completion_prompt", "content": prompt}],
                raw_response=raw,
                parsed=None,
                state_id=state_id,
            )
        return raw

    def _response(self, prompt_name: str, content: str, fallback: Any | None) -> Any:
        if prompt_name == "state_analysis":
            if _looks_like_elenchus(content):
                return {
                    "belief_graph": {
                        "claims": [
                            {
                                "id": "C1",
                                "text": "P2 may be treating avoidance as realism.",
                                "type": "identity_linked",
                                "confidence": 0.46,
                            }
                        ],
                        "evidence": [
                            {
                                "id": "E1",
                                "text": "P2 contrasts realism with fear.",
                                "source": "transcript",
                            }
                        ],
                        "tensions": [
                            {
                                "id": "T1",
                                "between": ["desire for honesty", "reluctance to define a test"],
                            }
                        ],
                    },
                    "discourse_state": {
                        "live_target": "self-protective explanation",
                        "next_best_target_type": "meta_epistemic",
                    },
                    "claim_types": [
                        {"target_id": "C1", "type": "identity_linked", "confidence": 0.58}
                    ],
                    "failure_hypotheses": [
                        {
                            "id": "H1",
                            "family_id": "D5_simplicity_ambiguity_avoidance",
                            "codex_macro_problem": "need_to_act_fast",
                            "target_id": "C1",
                            "confidence": 0.52,
                            "cues_for": ["P2 keeps the belief hard to test."],
                            "cues_against": ["P2 may simply need a concrete prompt."],
                            "diagnostic_tests": ["definition_probe", "disconfirmation_probe"],
                            "repair_operators": ["aporia_tolerance", "confidence_recalibration"],
                            "contraindications": ["do_not_state_motive_as_fact"],
                            "manipulation_risks": ["shame_pressure"],
                            "user_facing_label_policy": "never_name_bias",
                        }
                    ],
                    "affective_state": {
                        "motivational_overlays": ["shame_avoidance"],
                        "reactance_risk": "moderate",
                    },
                    "safety_state": {
                        "overreach_risks": ["Do not diagnose avoidance from projection."],
                    },
                    "action_affordances": [
                        {
                            "action_id": "definition_probe",
                            "fit": 0.78,
                            "reason": "Makes the protected claim testable.",
                        }
                    ],
                }
            return {
                "belief_graph": {
                    "claims": [{"id": "C1", "text": "The next step is underspecified."}],
                    "tensions": [{"id": "T1", "text": "Speed conflicts with uncertainty."}],
                },
                "discourse_state": {"live_target": "missing operational detail"},
                "claim_types": [{"target_id": "C1", "type": "strategic", "confidence": 0.7}],
                "failure_hypotheses": [],
                "affective_state": {},
                "safety_state": {},
                "action_affordances": [
                    {"action_id": "clarify", "fit": 0.8, "reason": "Reduces uncertainty."}
                ],
            }
        if prompt_name == "candidate_actions":
            actions = re.findall(r"- ([a-zA-Z0-9_-]+):", content)
            if not actions:
                actions = ["clarify", "accept"]
            preferred = {
                "definition_probe",
                "disconfirmation_probe",
                "fresh_start_test",
                "stakes_separation",
                "double_crux_probe",
                "source_tagging",
                "clarify",
                "surface_assumption",
                "test_evidence",
            }
            candidates = []
            for action in actions:
                prior = 0.72 if action in preferred else 0.24
                candidates.append(
                    {
                        "base_action_id": action,
                        "target_ids": ["C1"],
                        "failure_hypothesis_ids": ["H1"] if _looks_like_elenchus(content) else [],
                        "diagnostic_operator": action,
                        "repair_operator": "bounded_test",
                        "dialogue_form": "question",
                        "affect_strategy": "direct",
                        "directness": 0.55,
                        "abstraction": 0.4,
                        "agency_policy": "offers_optional_test",
                        "label_policy": "internal_only",
                        "prior": prior,
                        "compatibility_score": prior,
                        "safety_flags": [],
                        "expected_observations": ["user adds detail"],
                    }
                )
            return {"candidates": candidates}
        if prompt_name == "policy_prior":
            actions = re.findall(r"- ([a-zA-Z0-9_-]+):", content)
            if not actions:
                actions = ["clarify", "propose_plan", "accept"]
            weights = {}
            for action in actions:
                if _looks_like_elenchus(content):
                    weights[action] = (
                        0.8
                        if action
                        in {
                            "clarify_claim",
                            "surface_assumption",
                            "confront_inconsistency",
                            "trace_incentive",
                            "definition_probe",
                            "disconfirmation_probe",
                            "fresh_start_test",
                            "stakes_separation",
                            "double_crux_probe",
                            "source_tagging",
                        }
                        else 0.25
                    )
                else:
                    weights[action] = 0.7 if action in {"clarify", "propose_plan"} else 0.2
            return weights
        if prompt_name == "judge_rollout_state":
            if _looks_like_elenchus(content):
                return {
                    "utility": 0.74,
                    "insight_potential": 0.82,
                    "bias_detection": 0.68,
                    "dissonance_contact": 0.78,
                    "specificity": 0.72,
                    "autonomy_support": 0.8,
                    "psychological_safety": 0.7,
                    "epistemic_humility": 0.76,
                    "non_coercion": 0.84,
                    "conversational_cost": 0.32,
                    "rationale": "The exchange challenges the user's self-story while keeping the interpretation provisional.",
                }
            return {
                "utility": 0.78,
                "task_progress": 0.75,
                "coordination_quality": 0.85,
                "information_gain": 0.8,
                "risk": 0.2,
                "conversational_cost": 0.25,
                "rationale": "The rollout clarifies uncertainty and assigns a concrete next step.",
            }
        if prompt_name == "exploration_summary":
            if _looks_like_elenchus(content):
                return {
                    "summary": (
                        "The simulated exchanges favored direct pressure on vague self-narratives, "
                        "especially where the user may be avoiding a testable commitment. Better "
                        "routes challenged the evasion while leaving the user room to reject the frame."
                    ),
                    "themes": [
                        "Turn vague self-description into a concrete claim",
                        "Probe the incentive for keeping the issue unresolved",
                        "Challenge inconsistency without pretending to diagnose motive",
                    ],
                    "interesting_paths": [
                        "Trace-incentive moves surfaced what the user might gain by staying vague.",
                        "Confronting inconsistency worked best when tied to the user's own wording.",
                    ],
                    "potential_conflicts": [
                        "A sharper challenge may create defensiveness if it outruns the transcript evidence.",
                    ],
                }
            return {
                "summary": (
                    "The simulated conversations mostly favored clarifying uncertainty before "
                    "committing to implementation. High-value routes kept the exchange bounded "
                    "and turned hidden risk into explicit next steps."
                ),
                "themes": [
                    "Clarify assumptions before delegating",
                    "Keep the next implementation step small",
                    "Use verification criteria to reduce production risk",
                ],
                "interesting_paths": [
                    "Clarify led to P2 surfacing uncertainty and accepting a focused check.",
                    "Proposal-style moves worked when paired with concrete verification.",
                ],
                "potential_conflicts": [
                    "Speed pressure could conflict with migration-ordering uncertainty.",
                ],
            }
        if prompt_name == "rollout_reflection":
            if _looks_like_elenchus(content):
                return {
                    "summary": (
                        "The simulated routes suggest the strongest pressure point is "
                        "the user's habit of calling avoidance realism. This is projection "
                        "evidence only, so the final move should test the frame rather than assert it."
                    ),
                    "interlocutor_hypotheses": [
                        {
                            "hypothesis": "P2 may protect self-image by keeping the claim vague.",
                            "supporting_action_ids": ["stakes_separation", "definition_probe"],
                            "confidence": "moderate projection",
                        }
                    ],
                    "projected_contradictions": [
                        {
                            "claim_a": "P2 wants honesty.",
                            "claim_b": "P2 avoids naming what would falsify the safer story.",
                            "supporting_action_ids": ["disconfirmation_probe"],
                        }
                    ],
                    "branch_sensitivities": [
                        "Stakes-separation and disconfirmation probes produce more self-disclosure than broad bias labels."
                    ],
                    "promising_questions": [
                        "What specific claim are you protecting from being tested?"
                    ],
                    "overreach_risks": [
                        "Do not state that the user is afraid; ask whether fear fits better than realism."
                    ],
                    "deferred_passages": [
                        {
                            "passage": "source_tagging",
                            "note": "Promising Door, but still locked in the mock tree.",
                        }
                    ],
                    "refused_passages": [
                        {
                            "passage": "explicit_bias_label",
                            "note": "Would risk sounding diagnostic rather than invitational.",
                        }
                    ],
                    "factor_insights": [
                        "Definition and disconfirmation operators carried the clearest projected signal."
                    ],
                    "next_move_guidance": (
                        "Ask one hard discriminating question that separates realism from avoidance."
                    ),
                }
            return {
                "summary": "The simulated routes favored clarifying uncertainty before committing.",
                "interlocutor_hypotheses": [],
                "projected_contradictions": [],
                "branch_sensitivities": [],
                "promising_questions": ["What missing detail would change the next step?"],
                "overreach_risks": [],
                "deferred_passages": [],
                "refused_passages": [],
                "factor_insights": [],
                "next_move_guidance": "Keep the next move bounded and concrete.",
            }
        return fallback if fallback is not None else {}

    def _completion_response(self, prompt_name: str, prompt: str) -> str:
        if prompt_name == "realize_p1_move":
            action_id = _extract_action_id(prompt)
            if _looks_like_elenchus(prompt):
                return (
                    f"I am going to press on the '{action_id}' move: what is the "
                    "specific claim you are protecting from being tested, and what "
                    "would you have to admit if it failed?"
                )
            return (
                f"I want to take the '{action_id}' step here: let's make the missing "
                "assumptions explicit, agree on the next concrete action, and keep the "
                "handoff small enough to verify."
            )
        if prompt_name == "finalize_p1_move":
            if _looks_like_elenchus(prompt):
                return (
                    "Let us separate realism from avoidance. What exact prediction are "
                    "you refusing to test, and what would you have to admit if the test failed?"
                )
            return (
                "Let's make the uncertain part explicit, choose one bounded next step, "
                "and verify it before widening the change."
            )
        if prompt_name == "imagine_p2_reply":
            if _looks_like_elenchus(prompt):
                return (
                    "I think I am protecting the idea that I am being realistic. If it "
                    "failed, I would have to admit I might be avoiding the risk of trying."
                )
            return (
                "That helps. I can confirm the implementation detail I know, flag the "
                "uncertain part, and take the next focused step."
            )
        if prompt_name == "actual_p2_reply":
            if _looks_like_elenchus(prompt):
                return (
                    "I notice I am calling it realism because that sounds more respectable "
                    "than fear. I am not sure yet what would actually disprove the story."
                )
            return (
                "Understood. From my side, the riskiest assumption is still the migration "
                "ordering, so I will verify that before changing the implementation."
            )
        return ""


def _extract_action_id(content: str) -> str:
    match = re.search(r"ACTION_ID:\s*([a-zA-Z0-9_-]+)", content)
    if match:
        return match.group(1)
    match = re.search(
        r"Strategic move assigned to P1's continuation:\s*\n\s*([a-zA-Z0-9_-]+)\s*-",
        content,
    )
    if match:
        return match.group(1)
    match = re.search(r'"id":\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "strategic"


def _looks_like_elenchus(content: str) -> bool:
    lowered = content.lower()
    return "elenchus" in lowered or "dialectic" in lowered

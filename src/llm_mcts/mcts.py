from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from llm_mcts.cognitive import LiftedActionCandidate
from llm_mcts.config import ActionConfig
from llm_mcts.env import JudgeResult, TwoPlayerConversationEnv
from llm_mcts.search_stats import SearchStatsStore
from llm_mcts.state import ConversationState
from llm_mcts.trace import TraceWriter

MCTSEvent = dict[str, Any]
MCTSObserver = Callable[[MCTSEvent], None]


class MCTSCancelled(RuntimeError):
    pass


@dataclass
class PlanResult:
    chosen_action_id: str
    chosen_candidate: dict[str, Any] | None
    chosen_node_id: int | None
    chosen_path: list[int]
    p1_utterance: str
    search_p1_utterance: str
    estimated_utility: float
    root_stats: list[dict[str, Any]]
    rollout_reflection: dict[str, Any] | None = None
    trace_dir: Path | None = None
    calls_path: Path | None = None
    tree_path: Path | None = None


@dataclass
class MCTSNode:
    id: int
    state: ConversationState
    parent_id: int | None
    action_id: str | None = None
    candidate_id: str | None = None
    candidate: LiftedActionCandidate | None = None
    prior: float = 0.0
    depth: int = 0
    p1_utterance: str | None = None
    p2_reply: str | None = None
    action_metadata: dict[str, Any] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    candidate_pool: list[LiftedActionCandidate] = field(default_factory=list)
    candidate_rejections: list[dict[str, Any]] = field(default_factory=list)
    widening_limit: int = 0
    transposition_key: str | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass(frozen=True)
class SearchDepths:
    labyrinth_depth: int
    lantern_range: int
    max_rollout_depth: int
    legacy_absolute: bool

    @property
    def depth_mode(self) -> str:
        return "legacy_absolute" if self.legacy_absolute else "split"

    def rollout_steps_from_leaf(self, leaf_depth: int) -> int:
        if self.legacy_absolute:
            return max(0, self.max_rollout_depth - leaf_depth)
        return self.lantern_range


class MCTSPlanner:
    def __init__(
        self,
        env: TwoPlayerConversationEnv,
        tracer: TraceWriter | None = None,
    ):
        self.env = env
        self.config = env.config.mcts
        self.tracer = tracer
        self._rng = random.Random(self.config.random_seed)
        self.stats_store = SearchStatsStore()
        self._next_node_id = 0
        self._observer: MCTSObserver | None = None
        self._cancel_event: threading.Event | None = None

    def plan(
        self,
        state: ConversationState,
        simulations: int | None = None,
        max_rollout_depth: int | None = None,
        max_tree_depth: int | None = None,
        rollout_extension_depth: int | None = None,
        labyrinth_depth: int | None = None,
        lantern_range: int | None = None,
        observer: MCTSObserver | None = None,
        finalize: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> PlanResult:
        self._next_node_id = 0
        self._observer = observer
        self._cancel_event = cancel_event
        simulation_count = simulations if simulations is not None else self.config.simulations
        simulation_count = max(1, int(simulation_count))
        depths = self._resolve_depths(
            max_rollout_depth=max_rollout_depth,
            max_tree_depth=max_tree_depth,
            rollout_extension_depth=rollout_extension_depth,
            labyrinth_depth=labyrinth_depth,
            lantern_range=lantern_range,
        )
        root = self._new_node(state.copy(), parent_id=None, depth=0, prior=1.0)
        self._emit(
            "run_started",
            root_id=root.id,
            simulations=simulation_count,
            max_rollout_depth=depths.max_rollout_depth,
            labyrinth_depth=depths.labyrinth_depth,
            lantern_range=depths.lantern_range,
            depth_mode=depths.depth_mode,
            state=root.state.model_dump(),
        )
        self._emit("node_added", **self._event_node(root))

        try:
            for simulation_index in range(simulation_count):
                self._check_cancelled()
                path, leaf = self._select_and_expand(root, depths.labyrinth_depth)
                self._check_cancelled()
                self._emit(
                    "node_selected",
                    simulation=simulation_index + 1,
                    node_id=leaf.id,
                    path=[node.id for node in path],
                )
                rollout_extension = depths.rollout_steps_from_leaf(leaf.depth)
                rollout_state, rollout_trace = self._rollout_from(
                    leaf.state,
                    leaf.depth,
                    rollout_extension_depth=rollout_extension,
                    simulation_index=simulation_index + 1,
                )
                self._check_cancelled()
                judge = self.env.judge_state(rollout_state)
                evaluation = {
                    "utility": judge.utility,
                    "rubric": judge.rubric,
                    "rollout_trace": rollout_trace,
                    "rollout_state": rollout_state.model_dump(),
                }
                leaf.evaluations.append(evaluation)
                self._emit(
                    "node_evaluated",
                    simulation=simulation_index + 1,
                    node_id=leaf.id,
                    leaf_depth=leaf.depth,
                    rollout_extension_depth=rollout_extension,
                    utility=judge.utility,
                    rubric=judge.rubric,
                    rollout_trace=rollout_trace,
                )
                self._backpropagate(path, judge, simulation_index=simulation_index + 1)

            self._check_cancelled()
            chosen = self._select_root_action(root)
            if chosen is None:
                candidate = self.env.candidate_actions(state, limit=1)[0]
                action = self._action_for_id(state, candidate.base_action_id)
                search_utterance = self.env.realize_p1_move(state, action, candidate=candidate)
                root_stats: list[dict[str, Any]] = []
                utility = 0.0
                chosen_action_id = candidate.base_action_id
                chosen_candidate = candidate
                chosen_node_id = None
                chosen_path: list[int] = []
            else:
                search_utterance = chosen.p1_utterance or ""
                root_stats = self._root_stats(root)
                utility = chosen.q
                chosen_action_id = chosen.action_id or ""
                chosen_candidate = chosen.candidate
                chosen_node_id = chosen.id
                chosen_path = [root.id, chosen.id]
                action = self._action_for_id(state, chosen_action_id)

            self._emit("root_stats", stats=root_stats)

            self._check_cancelled()
            rollout_reflection = self._reflect_on_rollouts(
                root,
                state,
                chosen_action_id,
                search_utterance,
                root_stats,
                simulation_count,
                depths,
            )
            final_utterance = (
                self._finalize_p1_move(
                    state,
                    action,
                    search_utterance,
                    rollout_reflection,
                    chosen_candidate,
                )
                if finalize
                else search_utterance
            )
            self._check_cancelled()

            if self.tracer:
                self.tracer.save_tree(
                    self._tree_dict(
                        root,
                        chosen_action_id,
                        simulation_count,
                        depths,
                        search_utterance=search_utterance,
                        final_utterance=final_utterance,
                        rollout_reflection=rollout_reflection,
                        chosen_node_id=chosen_node_id,
                        chosen_path=chosen_path,
                        chosen_candidate=chosen_candidate,
                    )
                )

            result = PlanResult(
                chosen_action_id=chosen_action_id,
                chosen_candidate=(
                    chosen_candidate.to_prompt_dict() if chosen_candidate is not None else None
                ),
                chosen_node_id=chosen_node_id,
                chosen_path=chosen_path,
                p1_utterance=final_utterance,
                search_p1_utterance=search_utterance,
                estimated_utility=utility,
                root_stats=root_stats,
                rollout_reflection=rollout_reflection,
                trace_dir=self.tracer.run_dir if self.tracer else None,
                calls_path=self.tracer.calls_path if self.tracer else None,
                tree_path=self.tracer.tree_path if self.tracer else None,
            )
            self._emit(
                "plan_completed",
                chosen_action_id=result.chosen_action_id,
                chosen_candidate=result.chosen_candidate,
                chosen_node_id=result.chosen_node_id,
                chosen_path=result.chosen_path,
                p1_utterance=result.p1_utterance,
                search_p1_utterance=result.search_p1_utterance,
                estimated_utility=result.estimated_utility,
                root_stats=result.root_stats,
                rollout_reflection=result.rollout_reflection,
                max_rollout_depth=depths.max_rollout_depth,
                labyrinth_depth=depths.labyrinth_depth,
                lantern_range=depths.lantern_range,
                depth_mode=depths.depth_mode,
                trace_dir=str(result.trace_dir) if result.trace_dir else None,
            )
            return result
        except MCTSCancelled:
            raise
        except Exception as exc:
            self._emit("plan_failed", error=repr(exc))
            raise
        finally:
            self._observer = None
            self._cancel_event = None

    def _resolve_depths(
        self,
        *,
        max_rollout_depth: int | None,
        max_tree_depth: int | None,
        rollout_extension_depth: int | None,
        labyrinth_depth: int | None,
        lantern_range: int | None,
    ) -> SearchDepths:
        if labyrinth_depth is not None:
            max_tree_depth = labyrinth_depth
        if lantern_range is not None:
            rollout_extension_depth = lantern_range

        split_requested = (
            max_tree_depth is not None
            or rollout_extension_depth is not None
            or self.config.max_tree_depth is not None
            or self.config.rollout_extension_depth is not None
        )
        if not split_requested:
            absolute_depth = (
                max(1, int(max_rollout_depth))
                if max_rollout_depth is not None
                else self.config.max_rollout_depth
            )
            return SearchDepths(
                labyrinth_depth=absolute_depth,
                lantern_range=0,
                max_rollout_depth=absolute_depth,
                legacy_absolute=True,
            )

        tree_depth = (
            max_tree_depth
            if max_tree_depth is not None
            else self.config.max_tree_depth
            if self.config.max_tree_depth is not None
            else max_rollout_depth
            if max_rollout_depth is not None
            else self.config.max_rollout_depth
        )
        extension_depth = (
            rollout_extension_depth
            if rollout_extension_depth is not None
            else self.config.rollout_extension_depth
            if self.config.rollout_extension_depth is not None
            else 1
        )
        tree_depth = max(1, int(tree_depth))
        extension_depth = max(0, int(extension_depth))
        return SearchDepths(
            labyrinth_depth=tree_depth,
            lantern_range=extension_depth,
            max_rollout_depth=tree_depth + extension_depth,
            legacy_absolute=False,
        )

    def _select_and_expand(
        self,
        root: MCTSNode,
        labyrinth_depth: int,
    ) -> tuple[list[MCTSNode], MCTSNode]:
        node = root
        path = [node]
        while node.depth < labyrinth_depth:
            self._check_cancelled()
            if self._should_expand_or_widen(node):
                self._expand_or_widen(node)
                if not node.children:
                    return path, node
                node = self._select_child(node)
                self._emit(
                    "edge_traversed",
                    parent_id=node.parent_id,
                    node_id=node.id,
                    action_id=node.action_id,
                    depth=node.depth,
                )
                path.append(node)
                return path, node
            if not node.children:
                return path, node
            node = self._select_child(node)
            self._emit(
                "edge_traversed",
                parent_id=node.parent_id,
                node_id=node.id,
                action_id=node.action_id,
                depth=node.depth,
            )
            path.append(node)
        return path, node

    def _should_expand_or_widen(self, node: MCTSNode) -> bool:
        if not node.expanded:
            return True
        if not self.config.progressive_widening.enabled:
            return False
        if not node.candidate_pool:
            return True
        return len(node.children) < self._current_widening_limit(node)

    def _expand_or_widen(self, node: MCTSNode) -> None:
        self._check_cancelled()
        if not node.candidate_pool:
            self._generate_candidate_pool(node)
        pw = self.config.progressive_widening
        if pw.enabled:
            limit = self._current_widening_limit(node)
            if limit != node.widening_limit:
                node.widening_limit = limit
                self._emit(
                    "widening_limit_updated",
                    node_id=node.id,
                    depth=node.depth,
                    visits=node.visits,
                    widening_limit=limit,
                    expanded_children=len(node.children),
                    unexpanded_candidate_count=self._unexpanded_candidate_count(node),
                )
            add_limit = pw.initial_children if not node.expanded else pw.expand_batch
            target_count = min(limit, len(node.children) + add_limit)
        else:
            node.widening_limit = len(node.candidate_pool)
            target_count = len(node.candidate_pool)

        self._emit(
            "node_expanded",
            node_id=node.id,
            depth=node.depth,
            priors={candidate.candidate_id: candidate.prior for candidate in node.candidate_pool},
            action_ids=[candidate.base_action_id for candidate in node.candidate_pool],
            candidate_ids=[candidate.candidate_id for candidate in node.candidate_pool],
            widening_limit=node.widening_limit,
            unexpanded_candidate_count=self._unexpanded_candidate_count(node),
        )
        for candidate in node.candidate_pool:
            self._check_cancelled()
            if len(node.children) >= target_count:
                break
            if candidate.candidate_id in node.children:
                continue
            self._add_candidate_child(node, candidate)
        node.expanded = True

    def _current_widening_limit(self, node: MCTSNode) -> int:
        pw = self.config.progressive_widening
        if not pw.enabled:
            return len(node.candidate_pool)
        return min(
            len(node.candidate_pool),
            pw.max_candidates,
            max(
                pw.initial_children,
                int(math.floor(pw.k * (max(1, node.visits) ** pw.alpha))),
            ),
        )

    def _generate_candidate_pool(self, node: MCTSNode) -> None:
        self._check_cancelled()
        cognitive_state = self.env.analyze_cognitive_state(node.state)
        node.transposition_key = self.env.transposition_key(node.state)
        self._emit(
            "state_analyzed",
            node_id=node.id,
            state_id=node.state.state_hash(),
            cognitive_state=cognitive_state.summary(),
            transposition_key=node.transposition_key,
        )
        pw = self.config.progressive_widening
        limit = pw.max_candidates if pw.enabled else None
        node.candidate_pool = self.env.candidate_actions(
            node.state,
            node_visits=node.visits,
            limit=limit,
        )
        node.candidate_rejections = self.env.last_candidate_rejections.get(
            node.state.state_hash(),
            [],
        )
        self._emit(
            "candidate_pool_generated",
            node_id=node.id,
            depth=node.depth,
            candidate_count=len(node.candidate_pool),
            rejected_count=len(node.candidate_rejections),
            candidates=[self._compact_candidate(candidate) for candidate in node.candidate_pool],
            widening_limit=node.widening_limit,
            expanded_children=len(node.children),
            unexpanded_candidate_count=self._unexpanded_candidate_count(node),
        )
        for rejection in node.candidate_rejections[:8]:
            self._emit(
                "candidate_rejected",
                node_id=node.id,
                depth=node.depth,
                **rejection,
            )

    def _add_candidate_child(self, node: MCTSNode, candidate: LiftedActionCandidate) -> None:
        self._check_cancelled()
        child_state, p1_utterance, p2_reply = self.env.simulate_candidate_p1_p2(
            node.state,
            candidate,
            imagined=True,
        )
        child = self._new_node(
            child_state,
            parent_id=node.id,
            action_id=candidate.base_action_id,
            candidate_id=candidate.candidate_id,
            candidate=candidate,
            prior=candidate.prior,
            depth=node.depth + 1,
            p1_utterance=p1_utterance,
            p2_reply=p2_reply,
            action_metadata=candidate.action_metadata,
        )
        node.children[candidate.candidate_id] = child
        candidate_rank = self._candidate_rank(node, candidate)
        self._emit(
            "candidate_added",
            node_id=node.id,
            child_id=child.id,
            candidate_id=candidate.candidate_id,
            candidate=self._compact_candidate(candidate),
            candidate_rank=candidate_rank,
            parent_visits=node.visits,
            widening_limit=node.widening_limit,
            expanded_children=len(node.children),
            unexpanded_candidate_count=self._unexpanded_candidate_count(node),
        )
        self._emit("node_added", **self._event_node(child))

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        parent_visits = max(1, node.visits)

        def score(child: MCTSNode) -> tuple[float, float, str]:
            exploration = (
                self.config.c_puct
                * child.prior
                * math.sqrt(parent_visits)
                / (1 + child.visits)
            )
            exploit = child.q
            if child.visits == 0:
                exploit = self.stats_store.value_hint(child.candidate)
            return (exploit + exploration, child.prior, child.candidate_id or child.action_id or "")

        return max(node.children.values(), key=score)

    def _rollout_from(
        self,
        state: ConversationState,
        depth: int,
        rollout_extension_depth: int,
        simulation_index: int,
    ) -> tuple[ConversationState, list[dict[str, Any]]]:
        rollout_state = state.copy()
        rollout_trace: list[dict[str, Any]] = []
        rollout_offset = 0
        while rollout_offset < rollout_extension_depth and not rollout_state.terminal:
            self._check_cancelled()
            candidate = self._rollout_action(rollout_state)
            self._check_cancelled()
            rollout_state, p1_utterance, p2_reply = self.env.simulate_candidate_p1_p2(
                rollout_state,
                candidate,
                imagined=True,
            )
            rollout_offset += 1
            absolute_depth = depth + rollout_offset
            rollout_trace.append(
                {
                    "depth": absolute_depth,
                    "rollout_offset": rollout_offset,
                    "action_id": candidate.base_action_id,
                    "candidate_id": candidate.candidate_id,
                    "candidate": candidate.to_prompt_dict(),
                    "action_metadata": candidate.action_metadata,
                    "p1_utterance": p1_utterance,
                    "p2_reply": p2_reply,
                }
            )
            self._emit(
                "rollout_step",
                simulation=simulation_index,
                depth=absolute_depth,
                rollout_offset=rollout_offset,
                action_id=candidate.base_action_id,
                candidate_id=candidate.candidate_id,
                candidate=candidate.to_prompt_dict(),
                action_metadata=candidate.action_metadata,
                p1_utterance=p1_utterance,
                p2_reply=p2_reply,
                state_id=rollout_state.state_hash(),
            )
        return rollout_state, rollout_trace

    def _rollout_action(self, state: ConversationState) -> LiftedActionCandidate:
        self._check_cancelled()
        candidates = self.env.candidate_actions(state, limit=self.config.progressive_widening.max_candidates)
        if not candidates:
            raise RuntimeError("no rollout candidates available")

        def score(candidate: LiftedActionCandidate) -> tuple[float, float, str]:
            hint = self.stats_store.value_hint(candidate)
            return (
                candidate.prior + 0.15 * hint,
                candidate.compatibility_score,
                candidate.candidate_id,
            )

        max_score = max(score(candidate) for candidate in candidates)
        tied = [candidate for candidate in candidates if score(candidate) == max_score]
        return self._rng.choice(sorted(tied, key=lambda candidate: candidate.candidate_id))

    def _backpropagate(
        self,
        path: list[MCTSNode],
        judge: JudgeResult,
        simulation_index: int,
    ) -> None:
        for node in path:
            self._check_cancelled()
            node.visits += 1
            node.value_sum += judge.utility
            self.stats_store.backup(node.candidate, judge.utility)
            self._emit(
                "backup",
                simulation=simulation_index,
                node_id=node.id,
                visits=node.visits,
                value=node.q,
                value_sum=node.value_sum,
                utility=judge.utility,
            )
        self._emit(
            "abstract_stats",
            simulation=simulation_index,
            stats=self.stats_store.compact(),
        )

    def _select_root_action(self, root: MCTSNode) -> MCTSNode | None:
        if not root.children:
            return None
        if self.config.root_action_selection == "value":
            return max(root.children.values(), key=lambda child: (child.q, child.visits, child.prior))
        return max(root.children.values(), key=lambda child: (child.visits, child.q, child.prior))

    def _action_for_id(self, state: ConversationState, action_id: str) -> ActionConfig:
        legal_actions = self.env.legal_actions(state, "P1")
        for action in legal_actions:
            if action.id == action_id:
                return action
        return legal_actions[0]

    def _reflect_on_rollouts(
        self,
        root: MCTSNode,
        state: ConversationState,
        chosen_action_id: str,
        search_utterance: str,
        root_stats: list[dict[str, Any]],
        simulation_count: int,
        depths: SearchDepths,
    ) -> dict[str, Any] | None:
        if not self.env.config.prompts.rollout_reflection:
            return None
        evidence = self._rollout_reflection_evidence(
            root,
            chosen_action_id,
            search_utterance,
            root_stats,
            simulation_count,
            depths,
        )
        self._emit(
            "rollout_reflection_started",
            chosen_action_id=chosen_action_id,
            evidence_summary={
                "evaluations": evidence["evaluation_count"],
                "branch_samples": len(evidence["branch_samples"]),
            },
        )
        reflection = self.env.reflect_on_rollouts(state, evidence)
        if reflection:
            self._emit(
                "rollout_reflection",
                chosen_action_id=chosen_action_id,
                reflection=reflection,
                **reflection,
            )
        return reflection

    def _finalize_p1_move(
        self,
        state: ConversationState,
        action: ActionConfig,
        search_utterance: str,
        rollout_reflection: dict[str, Any] | None,
        candidate: LiftedActionCandidate | None = None,
    ) -> str:
        if not self.env.config.prompts.finalize_p1_move:
            return search_utterance
        self._emit(
            "final_p1_move_started",
            chosen_action_id=action.id,
            search_p1_utterance=search_utterance,
        )
        final_utterance = self.env.finalize_p1_move(
            state,
            action,
            search_utterance,
            rollout_reflection,
            candidate,
        )
        self._emit(
            "final_p1_move",
            chosen_action_id=action.id,
            p1_utterance=final_utterance,
            search_p1_utterance=search_utterance,
        )
        return final_utterance

    def _rollout_reflection_evidence(
        self,
        root: MCTSNode,
        chosen_action_id: str,
        search_utterance: str,
        root_stats: list[dict[str, Any]],
        simulation_count: int,
        depths: SearchDepths,
    ) -> dict[str, Any]:
        nodes = list(self._iter_nodes(root))
        evaluations: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}

        for node in nodes:
            if node.action_id:
                action_counts[node.action_id] = action_counts.get(node.action_id, 0) + node.visits
            for evaluation in node.evaluations:
                evaluations.append(self._compact_evaluation(node, evaluation))
                for step in evaluation.get("rollout_trace", []) or []:
                    action_id = step.get("action_id")
                    if isinstance(action_id, str):
                        action_counts[action_id] = action_counts.get(action_id, 0) + 1

        high_utility = sorted(
            evaluations,
            key=lambda item: float(item.get("utility", 0.0)),
            reverse=True,
        )
        low_utility = sorted(evaluations, key=lambda item: float(item.get("utility", 0.0)))
        selected_node = root.children.get(chosen_action_id)
        if selected_node is None:
            selected_node = next(
                (child for child in root.children.values() if child.action_id == chosen_action_id),
                None,
            )

        return {
            "chosen_action_id": chosen_action_id,
            "search_p1_utterance": search_utterance,
            "simulations": simulation_count,
            "max_rollout_depth": depths.max_rollout_depth,
            "labyrinth_depth": depths.labyrinth_depth,
            "lantern_range": depths.lantern_range,
            "depth_mode": depths.depth_mode,
            "top_root_actions": [_compact_root_stat(item) for item in root_stats[:6]],
            "selected_branch": (
                self._compact_node_for_reflection(selected_node) if selected_node else None
            ),
            "candidate_space": self._candidate_space_summary(root),
            "abstract_stats": self._abstract_stats_for_reflection(),
            "branch_samples": [
                self._compact_node_for_reflection(child)
                for child in sorted(root.children.values(), key=lambda item: (-item.visits, -item.q))[:4]
                if child.visits > 0
            ],
            "high_utility_rollouts": high_utility[:1],
            "low_utility_rollouts": low_utility[:1],
            "rollout_action_counts": dict(
                sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))[:12]
            ),
            "evaluation_count": len(evaluations),
        }

    def _candidate_space_summary(self, root: MCTSNode) -> dict[str, Any]:
        opened_ids = set(root.children)
        deferred = [
            self._compact_candidate_for_reflection(candidate)
            for candidate in root.candidate_pool
            if candidate.candidate_id not in opened_ids
        ]
        return {
            "widening_limit": root.widening_limit,
            "candidate_count": len(root.candidate_pool),
            "opened_count": len(root.children),
            "deferred_count": len(deferred),
            "opened_root_candidates": [
                self._compact_node_for_candidate_space(child)
                for child in sorted(root.children.values(), key=lambda item: (-item.visits, -item.q))
                [:4]
            ],
            "top_deferred_root_candidates": deferred[:3],
            "refused_candidates": [
                _compact_rejection(rejection) for rejection in root.candidate_rejections[:5]
            ],
            "factor_histograms": _compact_histograms(
                self._candidate_factor_histograms(root.candidate_pool)
            ),
        }

    def _candidate_factor_histograms(
        self,
        candidates: list[LiftedActionCandidate],
    ) -> dict[str, dict[str, int]]:
        histograms: dict[str, dict[str, int]] = {
            "base_action": {},
            "diagnostic_operator": {},
            "repair_operator": {},
            "failure_family": {},
            "target_type": {},
            "dialogue_form": {},
            "directness_bin": {},
            "safety_flag": {},
        }
        for candidate in candidates:
            metadata = candidate.action_metadata or {}
            _count(histograms["base_action"], candidate.base_action_id)
            _count(histograms["diagnostic_operator"], candidate.diagnostic_operator)
            _count(histograms["repair_operator"], candidate.repair_operator)
            _count(histograms["dialogue_form"], candidate.dialogue_form)
            _count(histograms["directness_bin"], _directness_bin(candidate.directness))
            for value in _values(metadata.get("failure_family_ids") or candidate.failure_hypothesis_ids):
                _count(histograms["failure_family"], value)
            for value in _values(metadata.get("target_types") or candidate.target_ids):
                _count(histograms["target_type"], value)
            for value in _values(candidate.safety_flags):
                _count(histograms["safety_flag"], value)
        return {key: value for key, value in histograms.items() if value}

    def _iter_nodes(self, node: MCTSNode) -> list[MCTSNode]:
        nodes = [node]
        for child in node.children.values():
            nodes.extend(self._iter_nodes(child))
        return nodes

    def _compact_node(self, node: MCTSNode | None) -> dict[str, Any] | None:
        if node is None:
            return None
        return {
            "node_id": node.id,
            "parent_id": node.parent_id,
            "action_id": node.action_id,
            "candidate_id": node.candidate_id,
            "candidate": self._compact_candidate(node.candidate),
            "action_metadata": _compact_action_metadata(node.action_metadata),
            "depth": node.depth,
            "visits": node.visits,
            "value": node.q,
            "prior": node.prior,
            "p1_utterance": _clip(node.p1_utterance, 120),
            "p2_reply": _clip(node.p2_reply, 120),
            "evaluation_count": len(node.evaluations),
        }

    def _compact_node_for_reflection(self, node: MCTSNode | None) -> dict[str, Any] | None:
        if node is None:
            return None
        return {
            "node_id": node.id,
            "parent_id": node.parent_id,
            "action_id": node.action_id,
            "candidate": self._compact_candidate_for_reflection(node.candidate),
            "depth": node.depth,
            "visits": node.visits,
            "value": node.q,
            "prior": node.prior,
            "p1_utterance": _clip(node.p1_utterance, 140),
            "p2_reply": _clip(node.p2_reply, 140),
            "evaluation_count": len(node.evaluations),
        }

    def _compact_node_for_candidate_space(self, node: MCTSNode | None) -> dict[str, Any] | None:
        if node is None:
            return None
        return {
            "node_id": node.id,
            "action_id": node.action_id,
            "candidate": self._compact_candidate_for_reflection(node.candidate),
            "visits": node.visits,
            "value": node.q,
            "prior": node.prior,
        }

    def _compact_evaluation(
        self,
        node: MCTSNode,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "node_id": node.id,
            "action_id": node.action_id,
            "candidate": self._compact_candidate_for_reflection(node.candidate),
            "depth": node.depth,
            "utility": evaluation.get("utility"),
            "rubric": _compact_rubric(evaluation.get("rubric")),
            "transition": {
                "p1_utterance": _clip(node.p1_utterance, 120),
                "p2_reply": _clip(node.p2_reply, 120),
            },
            "rollout_trace": [
                {
                    "depth": step.get("depth"),
                    "rollout_offset": step.get("rollout_offset"),
                    "action_id": step.get("action_id"),
                    "candidate": self._compact_candidate_for_reflection(step.get("candidate")),
                    "p1_utterance": _clip(step.get("p1_utterance"), 120),
                    "p2_reply": _clip(step.get("p2_reply"), 120),
                }
                for step in (evaluation.get("rollout_trace", []) or [])[:1]
            ],
        }

    def _root_stats(self, root: MCTSNode) -> list[dict[str, Any]]:
        stats = []
        for child in root.children.values():
            stats.append(
                {
                    "action_id": child.action_id,
                    "candidate_id": child.candidate_id,
                    "candidate": self._compact_candidate(child.candidate),
                    "action_metadata": child.action_metadata,
                    "visits": child.visits,
                    "value": child.q,
                    "prior": child.prior,
                    "p1_utterance": child.p1_utterance,
                    "imagined_p2_reply": child.p2_reply,
                    "unexpanded_candidate_count": self._unexpanded_candidate_count(root),
                }
            )
        return sorted(stats, key=lambda item: (-int(item["visits"]), -float(item["value"])))

    def _tree_dict(
        self,
        root: MCTSNode,
        chosen_action_id: str,
        simulation_count: int | None = None,
        depths: SearchDepths | None = None,
        search_utterance: str | None = None,
        final_utterance: str | None = None,
        rollout_reflection: dict[str, Any] | None = None,
        chosen_node_id: int | None = None,
        chosen_path: list[int] | None = None,
        chosen_candidate: LiftedActionCandidate | None = None,
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []

        def visit(node: MCTSNode) -> None:
            nodes.append(
                self._event_node(node)
                | {
                    "evaluations": node.evaluations,
                    "child_ids": [child.id for child in node.children.values()],
                }
            )
            for child in node.children.values():
                visit(child)

        visit(root)
        return {
            "chosen_action_id": chosen_action_id,
            "chosen_candidate": self._compact_candidate(chosen_candidate),
            "chosen_node_id": chosen_node_id,
            "chosen_path": chosen_path or [],
            "search_p1_utterance": search_utterance,
            "final_p1_utterance": final_utterance,
            "rollout_reflection": rollout_reflection,
            "simulations": simulation_count or self.config.simulations,
            "max_rollout_depth": (
                depths.max_rollout_depth if depths else self.config.max_rollout_depth
            ),
            "labyrinth_depth": (
                depths.labyrinth_depth if depths else self.config.max_tree_depth or self.config.max_rollout_depth
            ),
            "lantern_range": (
                depths.lantern_range if depths else self.config.rollout_extension_depth or 0
            ),
            "depth_mode": depths.depth_mode if depths else "legacy_absolute",
            "root_stats": self._root_stats(root),
            "search_stats": self.stats_store.compact(),
            "nodes": nodes,
        }

    def _new_node(
        self,
        state: ConversationState,
        parent_id: int | None,
        depth: int,
        prior: float,
        action_id: str | None = None,
        candidate_id: str | None = None,
        candidate: LiftedActionCandidate | None = None,
        p1_utterance: str | None = None,
        p2_reply: str | None = None,
        action_metadata: dict[str, Any] | None = None,
    ) -> MCTSNode:
        node = MCTSNode(
            id=self._next_node_id,
            state=state,
            parent_id=parent_id,
            action_id=action_id,
            candidate_id=candidate_id,
            candidate=candidate,
            prior=prior,
            depth=depth,
            p1_utterance=p1_utterance,
            p2_reply=p2_reply,
            action_metadata=action_metadata or {},
        )
        self._next_node_id += 1
        return node

    def _event_node(self, node: MCTSNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "parent_id": node.parent_id,
            "state_id": node.state.state_hash(),
            "action_id": node.action_id,
            "candidate_id": node.candidate_id,
            "candidate": self._compact_candidate(node.candidate),
            "action_metadata": node.action_metadata,
            "prior": node.prior,
            "depth": node.depth,
            "visits": node.visits,
            "value_sum": node.value_sum,
            "value": node.q,
            "expanded": node.expanded,
            "p1_utterance": node.p1_utterance,
            "p2_reply": node.p2_reply,
            "widening_limit": node.widening_limit,
            "unexpanded_candidate_count": self._unexpanded_candidate_count(node),
            "transposition_key": node.transposition_key,
        }

    def _compact_candidate(self, candidate: Any) -> dict[str, Any] | None:
        if candidate is None:
            return None
        if isinstance(candidate, LiftedActionCandidate):
            payload = candidate.to_prompt_dict()
        elif isinstance(candidate, dict):
            payload = candidate
        else:
            return None
        keep = {
            "candidate_id",
            "base_action_id",
            "description",
            "target_ids",
            "failure_hypothesis_ids",
            "diagnostic_operator",
            "repair_operator",
            "dialogue_form",
            "affect_strategy",
            "directness",
            "abstraction",
            "agency_policy",
            "label_policy",
            "prior",
            "compatibility_score",
            "safety_flags",
            "expected_observations",
            "action_metadata",
            "source",
        }
        compact = {key: value for key, value in payload.items() if key in keep}
        if "description" in compact:
            compact["description"] = _clip(compact["description"], 120)
        if "action_metadata" in compact:
            compact["action_metadata"] = _compact_action_metadata(compact["action_metadata"])
        return compact

    def _compact_candidate_for_reflection(self, candidate: Any) -> dict[str, Any] | None:
        if candidate is None:
            return None
        if isinstance(candidate, LiftedActionCandidate):
            payload = candidate.to_prompt_dict()
        elif isinstance(candidate, dict):
            payload = candidate
        else:
            return None
        return {
            key: value
            for key, value in {
                "base_action_id": payload.get("base_action_id"),
                "target_ids": _compact_list(payload.get("target_ids"), limit=3, string_limit=48),
                "failure_hypothesis_ids": _compact_list(
                    payload.get("failure_hypothesis_ids"), limit=3, string_limit=64
                ),
                "diagnostic_operator": _clip(payload.get("diagnostic_operator"), 64),
                "repair_operator": _clip(payload.get("repair_operator"), 64),
                "dialogue_form": _clip(payload.get("dialogue_form"), 48),
                "prior": payload.get("prior"),
                "compatibility_score": payload.get("compatibility_score"),
                "safety_flags": _compact_list(
                    payload.get("safety_flags"), limit=3, string_limit=72
                ),
            }.items()
            if value not in (None, "", [])
        }

    def _abstract_stats_for_reflection(self) -> dict[str, Any]:
        compact = self.stats_store.compact(limit=60)
        filtered = {
            key: value
            for key, value in compact.items()
            if not key.startswith("candidate:")
        }
        return dict(
            sorted(
                filtered.items(),
                key=lambda item: (-int(item[1].get("visits", 0)), item[0]),
            )[:12]
        )

    def _unexpanded_candidate_count(self, node: MCTSNode) -> int:
        expanded = set(node.children)
        return sum(1 for candidate in node.candidate_pool if candidate.candidate_id not in expanded)

    def _candidate_rank(self, node: MCTSNode, candidate: LiftedActionCandidate) -> int | None:
        for index, item in enumerate(node.candidate_pool, start=1):
            if item.candidate_id == candidate.candidate_id:
                return index
        return None

    def _emit(self, event_type: str, **payload: Any) -> None:
        self._check_cancelled()
        if self._observer is None:
            return
        self._observer({"type": event_type, **payload})

    def _check_cancelled(self) -> None:
        if getattr(self, "_cancel_event", None) is not None and self._cancel_event.is_set():
            raise MCTSCancelled("MCTS planning was cancelled")


def _clip(value: Any, limit: int = 420) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _compact_root_stat(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_id": item.get("action_id"),
        "visits": item.get("visits"),
        "value": item.get("value"),
        "prior": item.get("prior"),
    }


def _compact_action_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = (
        "objective",
        "target_types",
        "failure_family_ids",
        "diagnostic_operator",
        "repair_operator",
        "dialogue_form",
        "agency_policy",
        "label_policy",
    )
    compact: dict[str, Any] = {}
    for key in keep:
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, list):
            compact[key] = [_clip(entry, 48) for entry in item[:3]]
        elif isinstance(item, str):
            compact[key] = _clip(item, 80)
        else:
            compact[key] = item
    return compact


def _compact_rejection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"reason": _clip(value, 80)}
    candidate = value.get("candidate")
    base_action_id = value.get("base_action_id")
    if isinstance(candidate, dict):
        base_action_id = base_action_id or candidate.get("base_action_id") or candidate.get("action_id")
    return {
        key: item
        for key, item in {
            "reason": _clip(value.get("reason"), 80),
            "base_action_id": _clip(base_action_id, 80),
            "note": _clip(value.get("note") or value.get("message"), 120),
        }.items()
        if item not in ("", None)
    }


def _compact_histograms(histograms: dict[str, dict[str, int]], *, limit: int = 8) -> dict[str, dict[str, int]]:
    compact: dict[str, dict[str, int]] = {}
    for name, values in histograms.items():
        if not isinstance(values, dict):
            continue
        items = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
        compact[name] = {_clip(key, 80) or "unknown": count for key, count in items}
    return {key: value for key, value in compact.items() if value}


def _compact_list(value: Any, *, limit: int, string_limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    compact: list[Any] = []
    for item in value[:limit]:
        if isinstance(item, dict):
            compact.append(
                {
                    str(key): (_clip(val, string_limit) if isinstance(val, str) else val)
                    for key, val in list(item.items())[:6]
                }
            )
        else:
            compact.append(_clip(item, string_limit))
    return compact


def _compact_rubric(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = {
        "rationale",
        "insight_potential",
        "bias_detection",
        "dissonance_contact",
        "specificity",
        "autonomy_support",
        "psychological_safety",
        "epistemic_humility",
        "non_coercion",
        "conversational_cost",
        "hidden_premise_exposed",
        "contradiction_clarified",
        "definition_stabilized",
        "evidence_balance_improved",
        "probability_calibrated",
        "causal_model_improved",
        "value_tradeoff_clarified",
        "memory_uncertainty_mapped",
        "user_generated_insight",
        "user_agency_preserved",
        "rapport_preserved",
        "productive_aporia",
        "coercion_risk",
        "shame_risk",
        "reactance_increase",
        "misdiagnosis_risk",
        "manipulation_risk",
        "premature_closure",
        "rhetorical_victory_without_understanding",
    }
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key not in keep:
            continue
        if item is None:
            continue
        compact[key] = _clip(item, 120) if isinstance(item, str) else item
    return compact


def _compact_transcript(rollout_state: Any, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rollout_state, dict):
        return []
    transcript = rollout_state.get("transcript")
    if not isinstance(transcript, list):
        return []
    compact: list[dict[str, Any]] = []
    for turn in transcript[-limit:]:
        if not isinstance(turn, dict):
            continue
        compact.append(
            {
                "speaker": turn.get("speaker"),
                "action_id": turn.get("action_id"),
                "imagined": turn.get("imagined"),
                "content": _clip(turn.get("content"), 140),
            }
        )
    return compact


def _count(target: dict[str, int], value: Any) -> None:
    text = str(value or "").strip()
    if not text:
        return
    target[text] = target.get(text, 0) + 1


def _values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _directness_bin(value: Any) -> str:
    try:
        directness = float(value)
    except (TypeError, ValueError):
        directness = 0.5
    directness = max(0.0, min(1.0, directness))
    bucket = int(directness * 5)
    if bucket >= 5:
        bucket = 4
    return f"{bucket / 5:.1f}-{(bucket + 1) / 5:.1f}"

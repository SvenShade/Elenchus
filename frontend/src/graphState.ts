import type { TranscriptTurn } from "./api";

export type GraphNodeStatus = "fresh" | "selected" | "expanded" | "evaluated" | "chosen";

export type GraphNode = {
  id: number;
  parent_id: number | null;
  action_id: string | null;
  candidate_id?: string | null;
  candidate?: Record<string, unknown> | null;
  depth: number;
  prior: number;
  visits: number;
  value: number;
  status: GraphNodeStatus;
  action_metadata?: Record<string, unknown> | null;
  p1_utterance?: string | null;
  p2_reply?: string | null;
  widening_limit?: number;
  unexpanded_candidate_count?: number;
  transposition_key?: string | null;
  evaluations?: NodeEvaluation[];
};

export type GraphEdge = {
  id: string;
  source: number;
  target: number;
  action_id: string | null;
  active: boolean;
};

export type CandidateRecord = Record<string, unknown> & {
  candidate_id?: string;
  base_action_id?: string;
  description?: string;
  target_ids?: unknown[];
  failure_hypothesis_ids?: unknown[];
  diagnostic_operator?: string;
  repair_operator?: string;
  dialogue_form?: string;
  affect_strategy?: string;
  directness?: number;
  abstraction?: number;
  agency_policy?: string;
  label_policy?: string;
  prior?: number;
  compatibility_score?: number;
  safety_flags?: unknown[];
  expected_observations?: unknown[];
  action_metadata?: Record<string, unknown>;
};

export type CandidateRejection = Record<string, unknown> & {
  reason?: string;
  base_action_id?: string;
  candidate?: unknown;
};

export type CandidateOpening = {
  node_id: number;
  child_id: number | null;
  candidate_id: string;
  candidate: CandidateRecord | null;
  candidate_rank: number | null;
  parent_visits: number;
  widening_limit: number;
  expanded_children: number;
  unexpanded_candidate_count: number;
};

export type AbstractStat = {
  visits?: number;
  value?: number;
  value_sum?: number;
};

export type VisualEffect =
  | {
      id: number;
      type: "traversal_pulse";
      edge_id: string;
      source: number;
      target: number;
      target_prior: number;
      target_value: number;
    }
  | {
      id: number;
      type: "judgement_flash";
      node_id: number;
      utility: number;
    }
  | {
      id: number;
      type: "backup_wave";
      path: number[];
      utility: number;
    }
  | {
      id: number;
      type: "node_ripple";
      node_id: number;
      utility: number;
    }
  | {
      id: number;
      type: "transcript_wisp";
      node_id: number;
      text: string;
      source: "node_added" | "rollout_step";
    }
  | {
      id: number;
      type: "ghost_door_flare";
      node_id: number;
      candidate_id: string;
      candidate: CandidateRecord | null;
    }
  | {
      id: number;
      type: "singularity";
      reflexion_pass: number;
      reflexion_passes: number;
    };

export type CoalescenceState = {
  active: boolean;
  complete: boolean;
  reflection: RolloutReflection | null;
};

export type SingularityState = {
  id: number;
  active: boolean;
  reflexion_pass: number;
  reflexion_passes: number;
  snapshot_nodes: GraphNode[];
  snapshot_edges: GraphEdge[];
};

export type GraphState = {
  runId: string | null;
  nodes: Record<number, GraphNode>;
  edges: Record<string, GraphEdge>;
  selectedNodeId: number | null;
  chosenNodeId: number | null;
  chosenPath: number[];
  chosenActionId: string | null;
  rootStats: unknown[];
  candidatePoolsByNode: Record<number, CandidateRecord[]>;
  candidateRejectionsByNode: Record<number, CandidateRejection[]>;
  candidateOpeningsByCandidate: Record<string, CandidateOpening>;
  cognitiveStatesByNode: Record<number, Record<string, unknown>>;
  selectedCandidateId: string | null;
  abstractStats: Record<string, AbstractStat>;
  explorationSummary: ExplorationSummary | null;
  rolloutReflection: RolloutReflection | null;
  coalescence: CoalescenceState;
  singularity: SingularityState | null;
  visualEffects: VisualEffect[];
  nextEffectId: number;
  transcript: TranscriptTurn[];
  planning: boolean;
  status: string;
  error: string | null;
};

export type NodeEvaluation = {
  utility?: number;
  rubric?: unknown;
  rollout_trace?: unknown[];
};

export type ExplorationSummary = {
  summary: string;
  themes: string[];
  interesting_paths: string[];
  potential_conflicts: string[];
};

export type ReflectionItem = string | Record<string, unknown>;

export type RolloutReflection = {
  summary: string;
  interlocutor_hypotheses: ReflectionItem[];
  projected_contradictions: ReflectionItem[];
  branch_sensitivities: ReflectionItem[];
  promising_questions: ReflectionItem[];
  overreach_risks: ReflectionItem[];
  deferred_passages: ReflectionItem[];
  refused_passages: ReflectionItem[];
  factor_insights: ReflectionItem[];
  next_move_guidance: string;
};

export type MCTSEvent = {
  type: string;
  run_id?: string;
  id?: number;
  node_id?: number;
  child_id?: number;
  parent_id?: number | null;
  action_id?: string | null;
  candidate_id?: string | null;
  base_action_id?: string;
  candidate?: Record<string, unknown> | null;
  chosen_candidate?: Record<string, unknown> | null;
  action_metadata?: Record<string, unknown> | null;
  depth?: number;
  prior?: number;
  visits?: number;
  value?: number;
  p1_utterance?: string | null;
  search_p1_utterance?: string | null;
  p2_reply?: string | null;
  path?: number[];
  child_ids?: number[];
  chosen_action_id?: string;
  chosen_node_id?: number | null;
  chosen_path?: number[];
  root_stats?: unknown[];
  stats?: unknown;
  candidates?: CandidateRecord[];
  candidate_count?: number;
  rejected_count?: number;
  candidate_rank?: number | null;
  parent_visits?: number;
  widening_limit?: number;
  expanded_children?: number;
  unexpanded_candidate_count?: number;
  cognitive_state?: Record<string, unknown>;
  transposition_key?: string;
  transcript?: TranscriptTurn[];
  simulations?: number;
  max_rollout_depth?: number;
  utility?: number;
  rubric?: unknown;
  rollout_trace?: unknown[];
  summary?: string;
  reflection?: RolloutReflection;
  interlocutor_hypotheses?: ReflectionItem[];
  projected_contradictions?: ReflectionItem[];
  branch_sensitivities?: ReflectionItem[];
  promising_questions?: ReflectionItem[];
  overreach_risks?: ReflectionItem[];
  deferred_passages?: ReflectionItem[];
  refused_passages?: ReflectionItem[];
  factor_insights?: ReflectionItem[];
  next_move_guidance?: string;
  themes?: string[];
  interesting_paths?: string[];
  potential_conflicts?: string[];
  reflexion_pass?: number;
  reflexion_passes?: number;
  reason?: string;
  error?: string;
};

export const initialGraphState: GraphState = {
  runId: null,
  nodes: {},
  edges: {},
  selectedNodeId: null,
  chosenNodeId: null,
  chosenPath: [],
  chosenActionId: null,
  rootStats: [],
  candidatePoolsByNode: {},
  candidateRejectionsByNode: {},
  candidateOpeningsByCandidate: {},
  cognitiveStatesByNode: {},
  selectedCandidateId: null,
  abstractStats: {},
  explorationSummary: null,
  rolloutReflection: null,
  coalescence: {
    active: false,
    complete: false,
    reflection: null
  },
  singularity: null,
  visualEffects: [],
  nextEffectId: 1,
  transcript: [],
  planning: false,
  status: "Idle",
  error: null
};

export function graphReducer(state: GraphState, event: MCTSEvent): GraphState {
  switch (event.type) {
    case "session_created":
      return { ...state, status: "Session ready", error: null };
    case "planning_started":
      return {
        ...state,
        runId: event.run_id ?? state.runId,
        nodes: {},
        edges: {},
        selectedNodeId: null,
        chosenNodeId: null,
        chosenPath: [],
        chosenActionId: null,
        rootStats: [],
        candidatePoolsByNode: {},
        candidateRejectionsByNode: {},
        candidateOpeningsByCandidate: {},
        cognitiveStatesByNode: {},
        selectedCandidateId: null,
        abstractStats: {},
        explorationSummary: null,
        rolloutReflection: null,
        coalescence: {
          active: false,
          complete: false,
          reflection: null
        },
        singularity: null,
        visualEffects: [],
        nextEffectId: 1,
        planning: true,
        status: `Planning ${event.simulations ?? ""} simulations`.trim(),
        error: null
      };
    case "run_started": {
      const preserveSingularity =
        typeof event.reflexion_pass === "number" &&
        event.reflexion_pass > 1 &&
        state.singularity?.active;
      return {
        ...state,
        runId: event.run_id ?? state.runId,
        nodes: {},
        edges: {},
        selectedNodeId: null,
        chosenNodeId: null,
        chosenPath: [],
        chosenActionId: null,
        rootStats: [],
        candidatePoolsByNode: {},
        candidateRejectionsByNode: {},
        candidateOpeningsByCandidate: {},
        cognitiveStatesByNode: {},
        selectedCandidateId: null,
        abstractStats: {},
        explorationSummary: null,
        rolloutReflection: preserveSingularity ? state.rolloutReflection : null,
        singularity: preserveSingularity ? state.singularity : null,
        visualEffects: preserveSingularity ? state.visualEffects : [],
        planning: true,
        status: `Planning ${event.simulations ?? ""} simulations`.trim(),
        error: null
      };
    }
    case "node_added": {
      if (typeof event.id !== "number") return state;
      const node = toNode(event, state.nodes[event.id]?.status ?? "fresh");
      const edges = { ...state.edges };
      if (typeof event.parent_id === "number") {
        const id = edgeId(event.parent_id, event.id);
        edges[id] = {
          id,
          source: event.parent_id,
          target: event.id,
          action_id: event.action_id ?? null,
          active: false
        };
      }
      return {
        ...pushEffects(state, transcriptWispEffect(state, event, event.id, "node_added")),
        nodes: { ...state.nodes, [event.id]: node },
        edges,
        selectedNodeId: typeof event.parent_id === "number" ? event.parent_id : event.id,
        status: "Expanding search tree"
      };
    }
    case "edge_traversed": {
      if (typeof event.parent_id !== "number" || typeof event.node_id !== "number") return state;
      const id = edgeId(event.parent_id, event.node_id);
      return {
        ...pushEffects(state, {
          id: state.nextEffectId,
          type: "traversal_pulse",
          edge_id: id,
          source: event.parent_id,
          target: event.node_id,
          target_prior: state.nodes[event.node_id]?.prior ?? 0,
          target_value: state.nodes[event.node_id]?.value ?? 0
        }),
        selectedNodeId: event.node_id,
        edges: markActiveEdge(state.edges, id),
        nodes: markNode(state.nodes, event.node_id, "selected"),
        status: "Tracing Passages"
      };
    }
    case "node_selected":
      if (typeof event.node_id !== "number") return state;
      return {
        ...state,
        selectedNodeId: event.node_id,
        nodes: markNode(state.nodes, event.node_id, "selected")
      };
    case "scene_selected":
      if (typeof event.node_id !== "number") return state;
      return {
        ...state,
        selectedNodeId: event.node_id,
        selectedCandidateId: null,
        nodes: markNode(state.nodes, event.node_id, "selected")
      };
    case "candidate_selected":
      if (typeof event.candidate_id !== "string") return state;
      return {
        ...state,
        selectedCandidateId: event.candidate_id
      };
    case "node_expanded":
      if (typeof event.node_id !== "number") return state;
      return {
        ...state,
        selectedNodeId: event.node_id,
        nodes: updateNodeMeta(markNode(state.nodes, event.node_id, "expanded"), event.node_id, event),
        status: "Unlocking Doors"
      };
    case "state_analyzed":
      if (typeof event.node_id !== "number") return { ...state, status: "Reading the Floorplan" };
      return {
        ...state,
        nodes: updateNodeMeta(state.nodes, event.node_id, event),
        cognitiveStatesByNode: event.cognitive_state
          ? {
              ...state.cognitiveStatesByNode,
              [event.node_id]: event.cognitive_state
            }
          : state.cognitiveStatesByNode,
        status: "Reading the Floorplan"
      };
    case "candidate_pool_generated":
      if (typeof event.node_id !== "number") {
        return { ...state, status: "Finding Doors" };
      }
      return {
        ...state,
        nodes: updateNodeMeta(state.nodes, event.node_id, event),
        candidatePoolsByNode: {
          ...state.candidatePoolsByNode,
          [event.node_id]: event.candidates ?? state.candidatePoolsByNode[event.node_id] ?? []
        },
        status: `Found ${event.candidate_count ?? 0} Doors`
      };
    case "candidate_added": {
      if (typeof event.node_id !== "number" || typeof event.candidate_id !== "string") {
        return {
          ...state,
          status: "Unlocking a Door"
        };
      }
      const opening: CandidateOpening = {
        node_id: event.node_id,
        child_id: typeof event.child_id === "number" ? event.child_id : null,
        candidate_id: event.candidate_id,
        candidate: event.candidate ?? null,
        candidate_rank: typeof event.candidate_rank === "number" ? event.candidate_rank : null,
        parent_visits: event.parent_visits ?? 0,
        widening_limit: event.widening_limit ?? 0,
        expanded_children: event.expanded_children ?? 0,
        unexpanded_candidate_count: event.unexpanded_candidate_count ?? 0
      };
      return {
        ...pushEffects(state, {
          id: state.nextEffectId,
          type: "ghost_door_flare",
          node_id: event.node_id,
          candidate_id: event.candidate_id,
          candidate: event.candidate ?? null
        }),
        candidateOpeningsByCandidate: {
          ...state.candidateOpeningsByCandidate,
          [event.candidate_id]: opening
        },
        status: "Unlocking a Door"
      };
    }
    case "widening_limit_updated":
      if (typeof event.node_id !== "number") return state;
      return {
        ...state,
        nodes: updateNodeMeta(state.nodes, event.node_id, event),
        status: "Adding Keys to the Keyring"
      };
    case "candidate_rejected":
      if (typeof event.node_id !== "number") {
        return {
          ...state,
          status: "Refusing a Door"
        };
      }
      return {
        ...state,
        candidateRejectionsByNode: {
          ...state.candidateRejectionsByNode,
          [event.node_id]: [
            ...(state.candidateRejectionsByNode[event.node_id] ?? []),
            rejectionFromEvent(event)
          ]
        },
        status: "Refusing a Door"
      };
    case "backup":
      if (typeof event.node_id !== "number") return state;
      return {
        ...pushEffects(state, {
          id: state.nextEffectId,
          type: "node_ripple",
          node_id: event.node_id,
          utility: event.utility ?? 0
        }),
        nodes: {
          ...state.nodes,
          [event.node_id]: {
            ...(state.nodes[event.node_id] ?? emptyNode(event.node_id)),
            visits: event.visits ?? state.nodes[event.node_id]?.visits ?? 0,
            value: event.value ?? state.nodes[event.node_id]?.value ?? 0,
            status: "evaluated"
          }
        },
        status: "Carrying Omens to the Threshold"
      };
    case "node_evaluated":
      if (typeof event.node_id !== "number") return state;
      {
        const utility = event.utility ?? 0;
        const path = pathToRootFromNodes(state.nodes, event.node_id);
        const effects: VisualEffect[] = [
          {
            id: state.nextEffectId,
            type: "judgement_flash",
            node_id: event.node_id,
            utility
          },
          {
            id: state.nextEffectId + 1,
            type: "backup_wave",
            path,
            utility
          }
        ];
        return {
          ...pushEffects(state, effects),
          selectedNodeId: event.node_id,
          nodes: addEvaluationToNode(state.nodes, event.node_id, {
            utility: event.utility,
            rubric: event.rubric,
            rollout_trace: event.rollout_trace
          }),
          status: "Reading Omens"
        };
      }
    case "rollout_step":
      return {
        ...pushEffects(
          state,
          transcriptWispEffect(state, event, state.selectedNodeId ?? 0, "rollout_step")
        )
      };
    case "root_stats":
      return {
        ...state,
        rootStats: Array.isArray(event.stats)
          ? event.stats
          : event.root_stats ?? state.rootStats
      };
    case "abstract_stats":
      return {
        ...state,
        abstractStats: isStatsRecord(event.stats) ? event.stats : state.abstractStats
      };
    case "rollout_reflection_started":
      if (
        typeof event.reflexion_pass === "number" &&
        event.reflexion_pass > 1 &&
        state.coalescence.reflection
      ) {
        return {
          ...state,
          status: "Drawing Cartography"
        };
      }
      return {
        ...state,
        coalescence: {
          ...state.coalescence,
          active: true,
          complete: false
        },
        status: "Drawing Cartography"
      };
    case "rollout_reflection": {
      const reflection = normalizeRolloutReflection(event.reflection ?? event);
      const isFirstReflexionPass =
        typeof event.reflexion_pass === "number" &&
        typeof event.reflexion_passes === "number" &&
        event.reflexion_pass < event.reflexion_passes;
      return {
        ...state,
        rolloutReflection: reflection,
        coalescence: isFirstReflexionPass
          ? {
              active: true,
              complete: true,
              reflection
            }
          : state.coalescence.active
            ? {
                ...state.coalescence,
                complete: true,
                reflection: state.coalescence.reflection ?? reflection
              }
            : state.coalescence,
        status: "Rollout reflection ready"
      };
    }
    case "final_p1_move_started":
      return {
        ...state,
        status: "Synthesizing reflected P1 move"
      };
    case "final_p1_move":
      return {
        ...state,
        status: "Reflected P1 move ready"
      };
    case "reflexion_pass_completed":
      if (
        typeof event.reflexion_pass === "number" &&
        typeof event.reflexion_passes === "number" &&
        event.reflexion_pass < event.reflexion_passes
      ) {
        const effect: VisualEffect = {
          id: state.nextEffectId,
          type: "singularity",
          reflexion_pass: event.reflexion_pass,
          reflexion_passes: event.reflexion_passes
        };
        return {
          ...pushEffects(state, effect),
          singularity: {
            id: effect.id,
            active: true,
            reflexion_pass: event.reflexion_pass,
            reflexion_passes: event.reflexion_passes,
            snapshot_nodes: Object.values(state.nodes),
            snapshot_edges: Object.values(state.edges)
          },
          status: "Reflexion pass complete"
        };
      }
      return { ...state, status: "Reflexion pass complete" };
    case "reflexion_replan_started":
      return {
        ...state,
        status: "Building reflected tree"
      };
    case "planning_cancelling":
      return {
        ...state,
        status: "Cancelling thought"
      };
    case "planning_cancelled":
      return {
        ...state,
        planning: false,
        status: "Thought dissolved"
      };
    case "plan_completed":
      return {
        ...state,
        planning: false,
        chosenActionId: event.chosen_action_id ?? state.chosenActionId,
        chosenNodeId: chosenNodeIdFromEvent(event, state.nodes),
        chosenPath: chosenPathFromEvent(event, state.nodes),
        selectedNodeId: chosenNodeIdFromEvent(event, state.nodes) ?? state.selectedNodeId,
        rootStats: event.root_stats ?? state.rootStats,
        status: "P1 move selected"
      };
    case "planning_finished":
      return {
        ...state,
        planning: false,
        chosenActionId: event.chosen_action_id ?? state.chosenActionId,
        chosenNodeId: chosenNodeIdFromEvent(event, state.nodes),
        chosenPath: chosenPathFromEvent(event, state.nodes),
        rootStats: event.root_stats ?? state.rootStats,
        status: "P1 message appended"
      };
    case "transcript_updated":
      return { ...state, transcript: event.transcript ?? state.transcript };
    case "exploration_summary":
      return {
        ...state,
        explorationSummary: {
          summary: event.summary ?? "Exploration completed, but no detailed summary was available.",
          themes: event.themes ?? [],
          interesting_paths: event.interesting_paths ?? [],
          potential_conflicts: event.potential_conflicts ?? []
        },
        status: "Exploration summary ready"
      };
    case "plan_failed":
    case "planning_failed":
      return {
        ...state,
        planning: false,
        error: event.error ?? "Planning failed",
        status: "Planning failed"
      };
    default:
      return state;
  }
}

export function filterGraphByImportance(
  nodes: GraphNode[],
  edges: GraphEdge[],
  graphDetailPercent: number,
  selectedNodeId: number | null
) {
  const detail = Math.max(0, Math.min(100, graphDetailPercent));
  const threshold = (100 - detail) / 100;
  if (threshold <= 0) {
    return { nodes, edges };
  }

  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const rootVisits = Math.max(nodeMap.get(0)?.visits ?? 0, 1);
  const visibleIds = new Set<number>([0]);

  for (const node of nodes) {
    if (node.id === 0 || node.visits / rootVisits >= threshold) {
      visibleIds.add(node.id);
    }
  }

  for (const edge of edges) {
    if (edge.active) {
      visibleIds.add(edge.source);
      visibleIds.add(edge.target);
    }
  }

  if (selectedNodeId !== null) {
    let node = nodeMap.get(selectedNodeId);
    while (node) {
      visibleIds.add(node.id);
      node = typeof node.parent_id === "number" ? nodeMap.get(node.parent_id) : undefined;
    }
  }

  const filteredNodes = nodes.filter((node) => visibleIds.has(node.id));
  const filteredEdges = edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)
  );
  return { nodes: filteredNodes, edges: filteredEdges };
}

function toNode(event: MCTSEvent, status: GraphNodeStatus): GraphNode {
  return {
    id: event.id ?? 0,
    parent_id: event.parent_id ?? null,
    action_id: event.action_id ?? null,
    candidate_id: event.candidate_id ?? null,
    candidate: event.candidate ?? null,
    depth: event.depth ?? 0,
    prior: event.prior ?? 0,
    visits: event.visits ?? 0,
    value: event.value ?? 0,
    status,
    action_metadata: event.action_metadata ?? null,
    p1_utterance: event.p1_utterance,
    p2_reply: event.p2_reply,
    widening_limit: event.widening_limit,
    unexpanded_candidate_count: event.unexpanded_candidate_count,
    transposition_key: event.transposition_key ?? null
  };
}

function emptyNode(id: number): GraphNode {
  return {
    id,
    parent_id: null,
    action_id: null,
    candidate_id: null,
    candidate: null,
    depth: 0,
    prior: 0,
    visits: 0,
    value: 0,
    status: "fresh",
    action_metadata: null
  };
}

function updateNodeMeta(
  nodes: Record<number, GraphNode>,
  nodeId: number,
  event: MCTSEvent
) {
  const node = nodes[nodeId] ?? emptyNode(nodeId);
  return {
    ...nodes,
    [nodeId]: {
      ...node,
      widening_limit: event.widening_limit ?? node.widening_limit,
      unexpanded_candidate_count:
        event.unexpanded_candidate_count ?? node.unexpanded_candidate_count,
      transposition_key: event.transposition_key ?? node.transposition_key
    }
  };
}

function markNode(nodes: Record<number, GraphNode>, nodeId: number, status: GraphNodeStatus) {
  const node = nodes[nodeId] ?? emptyNode(nodeId);
  return { ...nodes, [nodeId]: { ...node, status } };
}

function addEvaluationToNode(
  nodes: Record<number, GraphNode>,
  nodeId: number,
  evaluation: NodeEvaluation
) {
  const node = nodes[nodeId] ?? emptyNode(nodeId);
  return {
    ...nodes,
    [nodeId]: {
      ...node,
      status: "evaluated" as const,
      evaluations: [...(node.evaluations ?? []), evaluation]
    }
  };
}

function markActiveEdge(edges: Record<string, GraphEdge>, edge: string) {
  return Object.fromEntries(
    Object.entries(edges).map(([id, value]) => [id, { ...value, active: id === edge }])
  );
}

function chosenNodeIdFromEvent(
  event: MCTSEvent,
  nodes: Record<number, GraphNode>
): number | null {
  if (typeof event.chosen_node_id === "number") {
    return event.chosen_node_id;
  }
  if (typeof event.chosen_action_id !== "string") {
    return null;
  }
  const rootChild = Object.values(nodes).find(
    (node) => node.parent_id === 0 && node.action_id === event.chosen_action_id
  );
  return rootChild?.id ?? null;
}

function chosenPathFromEvent(event: MCTSEvent, nodes: Record<number, GraphNode>): number[] {
  if (Array.isArray(event.chosen_path)) {
    return event.chosen_path.filter((nodeId): nodeId is number => typeof nodeId === "number");
  }
  const chosenNodeId = chosenNodeIdFromEvent(event, nodes);
  if (chosenNodeId === null) return [];
  const path: number[] = [];
  const seen = new Set<number>();
  let node: GraphNode | undefined = nodes[chosenNodeId];
  while (node && !seen.has(node.id)) {
    path.push(node.id);
    seen.add(node.id);
    node = typeof node.parent_id === "number" ? nodes[node.parent_id] : undefined;
  }
  return path.reverse();
}

function edgeId(source: number, target: number): string {
  return `${source}->${target}`;
}

function pushEffects(
  state: GraphState,
  effects: VisualEffect | VisualEffect[] | null
): GraphState {
  if (!effects) return state;
  const list = Array.isArray(effects) ? effects : [effects];
  if (list.length === 0) return state;
  return {
    ...state,
    visualEffects: [...state.visualEffects, ...list].slice(-96),
    nextEffectId: state.nextEffectId + list.length
  };
}

function transcriptWispEffect(
  state: GraphState,
  event: MCTSEvent,
  nodeId: number,
  source: "node_added" | "rollout_step"
): VisualEffect | null {
  const text = transcriptWispText(event);
  if (!text) return null;
  return {
    id: state.nextEffectId,
    type: "transcript_wisp",
    node_id: nodeId,
    text,
    source
  };
}

function transcriptWispText(event: MCTSEvent): string {
  const parts: string[] = [];
  if (event.p1_utterance) parts.push(`Elenchus: ${event.p1_utterance}`);
  if (event.p2_reply) parts.push(`You: ${event.p2_reply}`);
  return truncate(parts.join("\n"), 150);
}

function truncate(value: string, maxLength: number): string {
  const trimmed = value.replace(/\s+/g, " ").trim();
  if (trimmed.length <= maxLength) return trimmed;
  return `${trimmed.slice(0, maxLength - 1).trimEnd()}…`;
}

function pathToRootFromNodes(nodes: Record<number, GraphNode>, nodeId: number): number[] {
  const path: number[] = [];
  const seen = new Set<number>();
  let node: GraphNode | undefined = nodes[nodeId];
  while (node && !seen.has(node.id)) {
    path.push(node.id);
    seen.add(node.id);
    node = typeof node.parent_id === "number" ? nodes[node.parent_id] : undefined;
  }
  return path.reverse();
}

function normalizeRolloutReflection(value: unknown): RolloutReflection {
  const object = isRecord(value) ? value : {};
  return {
    summary: stringOrDefault(
      object.summary,
      "Rollout reflection completed, but no detailed reflection was available."
    ),
    interlocutor_hypotheses: reflectionItems(object.interlocutor_hypotheses),
    projected_contradictions: reflectionItems(object.projected_contradictions),
    branch_sensitivities: reflectionItems(object.branch_sensitivities),
    promising_questions: reflectionItems(object.promising_questions),
    overreach_risks: reflectionItems(object.overreach_risks),
    deferred_passages: reflectionItems(object.deferred_passages),
    refused_passages: reflectionItems(object.refused_passages),
    factor_insights: reflectionItems(object.factor_insights),
    next_move_guidance: stringOrDefault(object.next_move_guidance, "")
  };
}

function reflectionItems(value: unknown): ReflectionItem[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is ReflectionItem => {
    return (typeof item === "string" && item.trim().length > 0) || isRecord(item);
  });
}

function stringOrDefault(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function rejectionFromEvent(event: MCTSEvent): CandidateRejection {
  const rejection: CandidateRejection = {};
  for (const [key, value] of Object.entries(event)) {
    if (
      [
        "type",
        "run_id",
        "node_id",
        "reflexion_pass",
        "reflexion_passes"
      ].includes(key)
    ) {
      continue;
    }
    rejection[key] = value;
  }
  return rejection;
}

function isStatsRecord(value: unknown): value is Record<string, AbstractStat> {
  if (!isRecord(value)) return false;
  return Object.values(value).every((item) => {
    if (!isRecord(item)) return false;
    return (
      (typeof item.visits === "number" || typeof item.visits === "undefined") &&
      (typeof item.value === "number" || typeof item.value === "undefined") &&
      (typeof item.value_sum === "number" || typeof item.value_sum === "undefined")
    );
  });
}

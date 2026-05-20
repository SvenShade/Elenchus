import { describe, expect, it } from "vitest";
import { filterGraphByImportance, graphReducer, initialGraphState, type GraphEdge, type GraphNode } from "../src/graphState";

describe("graphReducer", () => {
  it("resets graph on a new planning run and accumulates nodes", () => {
    const started = graphReducer(initialGraphState, {
      type: "run_started",
      run_id: "run-1",
      simulations: 4
    });
    const withRoot = graphReducer(started, {
      type: "node_added",
      id: 0,
      parent_id: null,
      action_id: null,
      depth: 0,
      prior: 1,
      visits: 0,
      value: 0
    });
    const withChild = graphReducer(withRoot, {
      type: "node_added",
      id: 1,
      parent_id: 0,
      action_id: "clarify",
      action_metadata: { diagnostic_operator: "disconfirmation_probe" },
      depth: 1,
      prior: 0.7,
      visits: 0,
      value: 0
    });

    expect(withChild.runId).toBe("run-1");
    expect(Object.keys(withChild.nodes)).toHaveLength(2);
    expect(Object.keys(withChild.edges)).toEqual(["0->1"]);
    expect(withChild.selectedNodeId).toBe(0);
    expect(withChild.nodes[1].action_metadata).toEqual({
      diagnostic_operator: "disconfirmation_probe"
    });
  });

  it("updates transcript and finishes planning", () => {
    const withRoot = graphReducer(initialGraphState, {
      type: "node_added",
      id: 0,
      parent_id: null,
      action_id: null,
      depth: 0
    });
    const withChild = graphReducer(withRoot, {
      type: "node_added",
      id: 7,
      parent_id: 0,
      action_id: "clarify",
      depth: 1
    });
    const state = graphReducer(withChild, {
      type: "planning_finished",
      chosen_action_id: "clarify",
      chosen_node_id: 7,
      chosen_path: [0, 7],
      root_stats: []
    });
    const updated = graphReducer(state, {
      type: "transcript_updated",
      transcript: [{ speaker: "P1", content: "Can you clarify?", action_id: "clarify" }]
    });

    expect(updated.planning).toBe(false);
    expect(updated.chosenActionId).toBe("clarify");
    expect(updated.chosenNodeId).toBe(7);
    expect(updated.chosenPath).toEqual([0, 7]);
    expect(updated.transcript[0].content).toBe("Can you clarify?");
  });

  it("falls back to the chosen root child when older completion events omit node ids", () => {
    const withRoot = graphReducer(initialGraphState, {
      type: "node_added",
      id: 0,
      parent_id: null,
      action_id: null,
      depth: 0
    });
    const withChild = graphReducer(withRoot, {
      type: "node_added",
      id: 3,
      parent_id: 0,
      action_id: "accept",
      depth: 1
    });

    const completed = graphReducer(withChild, {
      type: "plan_completed",
      chosen_action_id: "accept",
      root_stats: []
    });

    expect(completed.chosenNodeId).toBe(3);
    expect(completed.chosenPath).toEqual([0, 3]);
    expect(completed.selectedNodeId).toBe(3);
  });

  it("stores rollout evidence and exploration summaries", () => {
    const withNode = graphReducer(initialGraphState, {
      type: "node_added",
      id: 1,
      parent_id: 0,
      action_id: "clarify",
      depth: 1,
      prior: 0.7,
      visits: 0,
      value: 0
    });
    const evaluated = graphReducer(withNode, {
      type: "node_evaluated",
      node_id: 1,
      utility: 0.8,
      rubric: { coordination_quality: 0.9 },
      rollout_trace: [{ action_id: "clarify" }]
    });
    const summarized = graphReducer(evaluated, {
      type: "exploration_summary",
      summary: "Clarification dominated the search.",
      themes: ["Clarify uncertainty"],
      interesting_paths: ["clarify -> focused reply"],
      potential_conflicts: ["speed vs certainty"]
    });

    expect(summarized.nodes[1].evaluations?.[0].utility).toBe(0.8);
    expect(summarized.explorationSummary?.themes).toEqual(["Clarify uncertainty"]);
    expect(evaluated.visualEffects.map((effect) => effect.type)).toEqual([
      "judgement_flash",
      "backup_wave"
    ]);
  });

  it("adds visual effects for traversal, backup, rollout wisps, and node wisps", () => {
    const withRoot = graphReducer(initialGraphState, {
      type: "node_added",
      id: 0,
      parent_id: null,
      depth: 0
    });
    const withChild = graphReducer(withRoot, {
      type: "node_added",
      id: 1,
      parent_id: 0,
      action_id: "clarify",
      depth: 1,
      p1_utterance: "Tell me the exact claim.",
      p2_reply: "I think I am avoiding it."
    });
    const traversed = graphReducer(withChild, {
      type: "edge_traversed",
      parent_id: 0,
      node_id: 1
    });
    const rippled = graphReducer(traversed, {
      type: "backup",
      node_id: 1,
      utility: 0.6,
      visits: 1,
      value: 0.6
    });
    const rollout = graphReducer(rippled, {
      type: "rollout_step",
      p1_utterance: "What would count as evidence?",
      p2_reply: "I would need a concrete case."
    });

    expect(withChild.visualEffects.at(-1)).toMatchObject({
      type: "transcript_wisp",
      node_id: 1,
      source: "node_added"
    });
    expect(traversed.visualEffects.at(-1)).toMatchObject({
      type: "traversal_pulse",
      edge_id: "0->1"
    });
    expect(rippled.visualEffects.at(-1)).toMatchObject({
      type: "node_ripple",
      node_id: 1
    });
    expect(rollout.visualEffects.at(-1)).toMatchObject({
      type: "transcript_wisp",
      node_id: 1,
      source: "rollout_step"
    });
  });

  it("stores rollout reflection details", () => {
    const reflecting = graphReducer(initialGraphState, {
      type: "rollout_reflection_started"
    });
    const reflected = graphReducer(reflecting, {
      type: "rollout_reflection",
      summary: "Projected P2 avoids the falsifiable claim.",
      interlocutor_hypotheses: [{ hypothesis: "P2 may be protecting self-image." }],
      projected_contradictions: ["Wants honesty but avoids the test."],
      branch_sensitivities: ["Trace incentive routes opened up."],
      promising_questions: ["What would disprove realism?"],
      overreach_risks: ["Do not assert fear as fact."],
      next_move_guidance: "Ask one discriminating question."
    });

    expect(reflecting.status).toBe("Drawing Cartography");
    expect(reflected.rolloutReflection?.summary).toContain("avoids");
    expect(reflected.rolloutReflection?.promising_questions).toEqual([
      "What would disprove realism?"
    ]);
    expect(reflected.rolloutReflection?.next_move_guidance).toBe(
      "Ask one discriminating question."
    );
  });

  it("stores progressive widening evidence and candidate selection", () => {
    const withRoot = graphReducer(initialGraphState, {
      type: "node_added",
      id: 0,
      parent_id: null,
      depth: 0
    });
    const analyzed = graphReducer(withRoot, {
      type: "state_analyzed",
      node_id: 0,
      cognitive_state: {
        belief_graph: {
          claims: [{ id: "C1", text: "I am being realistic." }],
          edges: []
        }
      }
    });
    const pooled = graphReducer(analyzed, {
      type: "candidate_pool_generated",
      node_id: 0,
      candidates: [
        {
          candidate_id: "cand-1",
          base_action_id: "definition_probe",
          target_ids: ["C1"],
          prior: 0.6,
          compatibility_score: 0.7
        }
      ],
      candidate_count: 1,
      widening_limit: 1,
      unexpanded_candidate_count: 1
    });
    const rejected = graphReducer(pooled, {
      type: "candidate_rejected",
      node_id: 0,
      reason: "unsafe_candidate",
      base_action_id: "confront"
    });
    const added = graphReducer(rejected, {
      type: "candidate_added",
      node_id: 0,
      child_id: 1,
      candidate_id: "cand-1",
      candidate: pooled.candidatePoolsByNode[0][0],
      candidate_rank: 1,
      parent_visits: 4,
      widening_limit: 2,
      expanded_children: 1,
      unexpanded_candidate_count: 0
    });
    const selected = graphReducer(added, {
      type: "candidate_selected",
      candidate_id: "cand-1"
    });
    const stats = graphReducer(selected, {
      type: "abstract_stats",
      stats: {
        "base_action:definition_probe": { visits: 2, value: 0.4, value_sum: 0.8 }
      }
    });

    expect(stats.cognitiveStatesByNode[0].belief_graph).toBeTruthy();
    expect(stats.candidatePoolsByNode[0]).toHaveLength(1);
    expect(stats.candidateRejectionsByNode[0][0].reason).toBe("unsafe_candidate");
    expect(stats.candidateOpeningsByCandidate["cand-1"].candidate_rank).toBe(1);
    expect(stats.visualEffects.at(-1)).toMatchObject({
      type: "ghost_door_flare",
      candidate_id: "cand-1"
    });
    expect(stats.selectedCandidateId).toBe("cand-1");
    expect(stats.abstractStats["base_action:definition_probe"].value).toBe(0.4);
  });

  it("stores first-pass Reflexion as coalescence and triggers singularity between trees", () => {
    const withRoot = graphReducer(initialGraphState, {
      type: "node_added",
      id: 0,
      parent_id: null,
      depth: 0
    });
    const withChild = graphReducer(withRoot, {
      type: "node_added",
      id: 2,
      parent_id: 0,
      action_id: "definition_probe",
      depth: 1
    });
    const reflecting = graphReducer(withChild, {
      type: "rollout_reflection_started",
      reflexion_pass: 1,
      reflexion_passes: 2
    });
    const reflected = graphReducer(reflecting, {
      type: "rollout_reflection",
      reflexion_pass: 1,
      reflexion_passes: 2,
      summary: "The first tree found an unstable definition."
    });
    const collapsed = graphReducer(reflected, {
      type: "reflexion_pass_completed",
      reflexion_pass: 1,
      reflexion_passes: 2
    });
    const secondRun = graphReducer(collapsed, {
      type: "run_started",
      reflexion_pass: 2,
      reflexion_passes: 2
    });

    expect(reflecting.coalescence.active).toBe(true);
    expect(reflected.coalescence.complete).toBe(true);
    expect(reflected.coalescence.reflection?.summary).toContain("unstable");
    expect(collapsed.singularity?.snapshot_nodes.map((node) => node.id).sort()).toEqual([0, 2]);
    expect(collapsed.visualEffects.at(-1)?.type).toBe("singularity");
    expect(secondRun.singularity?.active).toBe(true);
    expect(Object.keys(secondRun.nodes)).toEqual([]);
  });

  it("does not trigger singularity for ordinary single-pass planning", () => {
    const reflected = graphReducer(initialGraphState, {
      type: "rollout_reflection",
      reflexion_pass: 1,
      reflexion_passes: 1,
      summary: "Only one tree."
    });
    const completed = graphReducer(reflected, {
      type: "reflexion_pass_completed",
      reflexion_pass: 1,
      reflexion_passes: 1
    });

    expect(completed.singularity).toBeNull();
    expect(completed.visualEffects.some((effect) => effect.type === "singularity")).toBe(false);
  });

  it("tracks Reflexion pass statuses", () => {
    const firstDone = graphReducer(initialGraphState, {
      type: "reflexion_pass_completed"
    });
    const secondStarted = graphReducer(firstDone, {
      type: "reflexion_replan_started"
    });

    expect(firstDone.status).toBe("Reflexion pass complete");
    expect(secondStarted.status).toBe("Building reflected tree");
  });

  it("treats 100 graph detail as the full graph", () => {
    const nodes: GraphNode[] = [node(0, null, 10), node(1, 0, 8), node(2, 0, 1)];
    const edges: GraphEdge[] = [edge(0, 1, false), edge(0, 2, false)];

    const filtered = filterGraphByImportance(nodes, edges, 100, null);

    expect(filtered.nodes.map((item) => item.id).sort()).toEqual([0, 1, 2]);
    expect(filtered.edges.map((item) => item.id).sort()).toEqual(["0->1", "0->2"]);
  });

  it("filters graph by inverted detail while preserving root, selected ancestors, and active edge", () => {
    const nodes: GraphNode[] = [
      node(0, null, 10),
      node(1, 0, 8),
      node(2, 0, 1),
      node(3, 2, 0)
    ];
    const edges: GraphEdge[] = [
      edge(0, 1, false),
      edge(0, 2, false),
      edge(2, 3, true)
    ];

    const filtered = filterGraphByImportance(nodes, edges, 50, 3);

    expect(filtered.nodes.map((item) => item.id).sort()).toEqual([0, 1, 2, 3]);
    expect(filtered.edges.map((item) => item.id).sort()).toEqual(["0->1", "0->2", "2->3"]);

    const withoutSelected = filterGraphByImportance(nodes, edges, 50, null);
    expect(withoutSelected.nodes.map((item) => item.id).sort()).toEqual([0, 1, 2, 3]);
  });

  it("removes disconnected edges when low-visit nodes are hidden", () => {
    const nodes: GraphNode[] = [node(0, null, 10), node(1, 0, 8), node(2, 0, 1)];
    const edges: GraphEdge[] = [edge(0, 1, false), edge(0, 2, false)];

    const filtered = filterGraphByImportance(nodes, edges, 50, null);

    expect(filtered.nodes.map((item) => item.id).sort()).toEqual([0, 1]);
    expect(filtered.edges.map((item) => item.id)).toEqual(["0->1"]);
  });

  it("treats 0 graph detail as strongest pruning", () => {
    const nodes: GraphNode[] = [node(0, null, 10), node(1, 0, 8), node(2, 0, 1)];
    const edges: GraphEdge[] = [edge(0, 1, false), edge(0, 2, false)];

    const filtered = filterGraphByImportance(nodes, edges, 0, null);

    expect(filtered.nodes.map((item) => item.id).sort()).toEqual([0]);
    expect(filtered.edges).toEqual([]);
  });
});

function node(id: number, parent_id: number | null, visits: number): GraphNode {
  return {
    id,
    parent_id,
    action_id: id === 0 ? null : `action-${id}`,
    depth: parent_id === null ? 0 : 1,
    prior: 0,
    visits,
    value: 0,
    status: "fresh"
  };
}

function edge(source: number, target: number, active: boolean): GraphEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    action_id: null,
    active
  };
}

import { describe, expect, it } from "vitest";
import {
  buildAntechamber,
  buildPrismRows,
  candidatesTargeting,
  candidateFactorChips,
  ghostDoorCandidates,
  normalizeDreamGraph,
  openingExplanation
} from "../src/progressiveWidening";
import type { CandidateOpening, CandidateRecord, GraphNode } from "../src/graphState";

describe("progressive widening helpers", () => {
  it("derives unlocked, locked, and refused Door groups", () => {
    const selected = node(0, null);
    const child = node(4, 0, "definition_probe", "cand-open");
    const candidates = [candidate("cand-wait", "source_tagging", 0.2), candidate("cand-open", "definition_probe", 0.8)];
    const openings: Record<string, CandidateOpening> = {
      "cand-open": {
        node_id: 0,
        child_id: 4,
        candidate_id: "cand-open",
        candidate: candidates[1],
        candidate_rank: 1,
        parent_visits: 5,
        widening_limit: 2,
        expanded_children: 1,
        unexpanded_candidate_count: 1
      }
    };

    const model = buildAntechamber(selected, [selected, child], candidates, openings, [
      { reason: "unsafe_candidate", base_action_id: "confront" }
    ]);

    expect(model.opened.map((item) => item.candidateId)).toEqual(["cand-open"]);
    expect(model.waiting.map((item) => item.candidateId)).toEqual(["cand-wait"]);
    expect(model.refused[0].reason).toBe("unsafe_candidate");
    expect(ghostDoorCandidates(model)).toHaveLength(1);
    expect(openingExplanation(model.opened[0])).toContain("Unlocked Door");
  });

  it("normalizes a Floorplan belief graph and identifies targeted candidates", () => {
    const dream = normalizeDreamGraph({
      belief_graph: {
        claims: [{ id: "C1", text: "Realism protects me." }],
        tensions: [{ id: "T1", text: "Honesty versus comfort." }],
        edges: [{ source: "C1", target: "T1", type: "supports" }]
      }
    });

    expect(dream.nodes.map((item) => item.id).sort()).toEqual(["C1", "T1"]);
    expect(dream.edges[0]).toMatchObject({ source: "C1", target: "T1" });
    expect([...candidatesTargeting([candidate("cand-1", "definition_probe", 0.6, ["C1"])], "C1")]).toEqual([
      "cand-1"
    ]);
  });

  it("aggregates Lockwork rows over action factors and abstract stats", () => {
    const candidates = [
      candidate("cand-1", "definition_probe", 0.7, ["C1"]),
      candidate("cand-2", "source_tagging", 0.3, ["M1"])
    ];
    const nodes = [node(1, 0, "definition_probe", "cand-1", 3, 0.6)];
    const rows = buildPrismRows(candidates, nodes, {}, {
      "candidate:cand-deadbeef": { visits: 9, value: 0.7 },
      "directness_bin:0.4-0.6": { visits: 4, value: 0.2 },
      "repair_operator:stabilize_definition": { visits: 2, value: 0.5 }
    });

    expect(rows.some((row) => row.category === "Base Action" && row.factor === "Definition Probe")).toBe(true);
    expect(rows.some((row) => row.category === "Repair Operator" && row.factor === "Stabilize Definition")).toBe(true);
    expect(rows.some((row) => row.category === "Directness" && row.factor === "Balanced")).toBe(true);
    expect(rows.some((row) => row.category === "Candidate")).toBe(false);
    expect(candidateFactorChips(candidates[0])).toContain("C1");
  });
});

function candidate(
  candidate_id: string,
  base_action_id: string,
  prior: number,
  target_ids: string[] = []
): CandidateRecord {
  return {
    candidate_id,
    base_action_id,
    prior,
    compatibility_score: prior + 0.1,
    target_ids,
    failure_hypothesis_ids: ["F1"],
    diagnostic_operator: "test_definition",
    repair_operator: "stabilize_definition",
    dialogue_form: "socratic_question",
    directness: 0.45,
    safety_flags: ["agency_preserved"],
    action_metadata: {
      target_types: ["claim"],
      failure_family_ids: ["F1"]
    }
  };
}

function node(
  id: number,
  parent_id: number | null,
  action_id: string | null = null,
  candidate_id: string | null = null,
  visits = 0,
  value = 0
): GraphNode {
  return {
    id,
    parent_id,
    action_id,
    candidate_id,
    depth: parent_id === null ? 0 : 1,
    prior: 0,
    visits,
    value,
    status: "fresh"
  };
}

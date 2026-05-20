import { describe, expect, it } from "vitest";
import {
  actionLabel,
  branchEdgeIds,
  branchToRoot,
  edgeHeatForVisits,
  graphDistancesFromNode,
  isEdgeInBranch,
  labelOpacityForDistance,
  nodeIdPathToRoot,
  pathEdgeIds,
  sphereRadiusForNode,
  utilityFlashColor,
  visitHeatForNode,
  valueLabel
} from "../src/graphMetrics";
import type { GraphEdge, GraphNode } from "../src/graphState";

describe("graph metrics", () => {
  it("scales node spheres by utility and clamps extreme values", () => {
    const low = sphereRadiusForNode(node(1, 0, -1));
    const neutral = sphereRadiusForNode(node(2, 0, 0));
    const high = sphereRadiusForNode(node(3, 0, 1));
    const overHigh = sphereRadiusForNode(node(4, 0, 10));
    const selectedHigh = sphereRadiusForNode(node(5, 0, 1), true);

    expect(low).toBeLessThan(neutral);
    expect(neutral).toBeLessThan(high);
    expect(overHigh).toBe(high);
    expect(selectedHigh).toBeGreaterThan(high);
    expect(selectedHigh).toBeLessThanOrEqual(1.02);
  });

  it("returns the selected branch ordered from root to selected node", () => {
    const nodes = [node(0, null), node(1, 0), node(2, 1), node(3, 0)];

    const branch = branchToRoot(nodes, 2);

    expect(branch.map((item) => item.id)).toEqual([0, 1, 2]);
  });

  it("identifies edges on the selected branch", () => {
    const branch = branchToRoot([node(0, null), node(1, 0), node(2, 1), node(3, 0)], 2);
    const ids = branchEdgeIds(branch);

    expect(isEdgeInBranch(edge(0, 1), ids)).toBe(true);
    expect(isEdgeInBranch(edge(1, 2), ids)).toBe(true);
    expect(isEdgeInBranch(edge(0, 3), ids)).toBe(false);
  });

  it("builds edge ids from an explicit MCTS chosen path", () => {
    expect([...pathEdgeIds([0, 4, 9])]).toEqual(["0->4", "4->9"]);
  });

  it("computes label opacity by graph distance from the selected node", () => {
    const nodes = [node(0, null), node(1, 0), node(2, 1), node(3, 0)];
    const edges = [edge(0, 1), edge(1, 2), edge(0, 3)];

    const distances = graphDistancesFromNode(nodes, edges, 2);

    expect(labelOpacityForDistance(distances.get(2))).toBe(1);
    expect(labelOpacityForDistance(distances.get(1))).toBeGreaterThan(
      labelOpacityForDistance(distances.get(0))
    );
    expect(labelOpacityForDistance(distances.get(0))).toBeGreaterThan(
      labelOpacityForDistance(distances.get(3))
    );
    expect(labelOpacityForDistance(null)).toBe(0.88);
  });

  it("provides all nodes needed for a route inspector readout", () => {
    const nodes = [node(0, null), node(1, 0), node(2, 1), node(3, 2), node(4, 0)];

    const branch = branchToRoot(nodes, 3);

    expect(branch.map((item) => item.id)).toEqual([0, 1, 2, 3]);
    expect(branch).not.toContainEqual(expect.objectContaining({ id: 4 }));
  });

  it("maps utility values to human-readable lantern labels", () => {
    expect(valueLabel(-1)).toBe("Dark");
    expect(valueLabel(-0.6)).toBe("Dark");
    expect(valueLabel(-0.59)).toBe("Shadowed");
    expect(valueLabel(-0.2)).toBe("Dim");
    expect(valueLabel(0)).toBe("Dim");
    expect(valueLabel(0.2)).toBe("Glowing");
    expect(valueLabel(0.6)).toBe("Bright");
    expect(valueLabel(99)).toBe("Bright");
  });

  it("formats action ids for human-readable UI labels", () => {
    expect(actionLabel(null)).toBe("Threshold");
    expect(actionLabel("clarify_claim")).toBe("Clarify Claim");
    expect(actionLabel("name-bias_hypothesis")).toBe("Name Bias Hypothesis");
  });

  it("maps utility to judgement flash colors", () => {
    expect(utilityFlashColor(0.8)).toBe("#6ee7ff");
    expect(utilityFlashColor(0.3)).toBe("#ff3b6b");
    expect(utilityFlashColor(-0.5)).toBe("#ff6b8c");
    expect(utilityFlashColor(0)).toBe("#8a3a51");
  });

  it("maps visits to persistent heat", () => {
    const low = visitHeatForNode({ id: 1, visits: 1 }, 10);
    const high = visitHeatForNode({ id: 2, visits: 8 }, 10);
    const root = visitHeatForNode({ id: 0, visits: 10 }, 10);

    expect(high.opacity).toBeGreaterThan(low.opacity);
    expect(high.scale).toBeGreaterThan(low.scale);
    expect(root.opacity).toBeGreaterThan(low.opacity);
    expect(edgeHeatForVisits(5, 10)).toBe(0.5);
  });

  it("derives node id paths from leaf to root helpers", () => {
    const nodes = [node(0, null), node(1, 0), node(2, 1), node(3, 0)];

    expect(nodeIdPathToRoot(nodes, 2)).toEqual([0, 1, 2]);
    expect(nodeIdPathToRoot(nodes, null)).toEqual([]);
  });
});

function node(id: number, parent_id: number | null, value = 0): GraphNode {
  return {
    id,
    parent_id,
    action_id: id === 0 ? null : `action-${id}`,
    depth: parent_id === null ? 0 : 1,
    prior: 0,
    visits: 0,
    value,
    status: "fresh"
  };
}

function edge(source: number, target: number): GraphEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    action_id: null,
    active: false
  };
}

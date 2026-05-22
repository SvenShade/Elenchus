import { describe, expect, it } from "vitest";
import {
  ghostDoorLayouts,
  layoutLabyrinthGraph,
  passagePath
} from "../src/labyrinthLayout";
import type { GraphEdge, GraphNode } from "../src/graphState";

describe("labyrinth layout", () => {
  it("keeps the Threshold Chamber fixed at the centre", () => {
    const layout = layoutLabyrinthGraph([node(0, null)], []);

    expect(layout.nodeMap.get(0)).toMatchObject({ x: 0, y: 0, cellX: 0, cellY: 0 });
  });

  it("assigns stable positions across repeated layout calls", () => {
    const nodes = [node(0, null), node(1, 0), node(2, 0), node(3, 1)];
    const edges = [edge(0, 1), edge(0, 2), edge(1, 3)];

    const first = layoutLabyrinthGraph(nodes, edges);
    const second = layoutLabyrinthGraph(nodes, edges);

    expect(positions(first.nodes)).toEqual(positions(second.nodes));
  });

  it("resolves sibling collisions into unique cells", () => {
    const nodes = [
      node(0, null),
      node(1, 0),
      node(2, 0),
      node(3, 0),
      node(4, 0),
      node(5, 0),
      node(6, 0)
    ];

    const layout = layoutLabyrinthGraph(nodes, nodes.slice(1).map((item) => edge(0, item.id)));
    const cells = layout.nodes.map((item) => `${item.cellX},${item.cellY}`);

    expect(new Set(cells).size).toBe(cells.length);
  });

  it("generates valid orthogonal SVG passage paths", () => {
    expect(passagePath({ x: 0, y: 0 }, { x: 112, y: 112 })).toMatch(/^M 0 0 L/);
    expect(passagePath({ x: 0, y: 0 }, { x: 0, y: 112 })).toBe("M 0 0 L 0 112");
  });

  it("places ghost Doors around a selected Chamber without creating graph nodes", () => {
    const layout = layoutLabyrinthGraph([node(0, null)], []);
    const doors = ghostDoorLayouts(layout.nodeMap.get(0), ["cand-a", "cand-b"]);

    expect(doors.map((door) => door.candidateId)).toEqual(["cand-a", "cand-b"]);
    expect(doors[0].x).not.toBe(layout.nodeMap.get(0)?.x);
  });
});

function positions(nodes: Array<{ id: number; cellX: number; cellY: number }>) {
  return nodes.map((item) => [item.id, item.cellX, item.cellY]);
}

function node(id: number, parent_id: number | null): GraphNode {
  return {
    id,
    parent_id,
    action_id: id === 0 ? null : `action-${id}`,
    depth: parent_id === null ? 0 : 1,
    prior: 0,
    visits: 0,
    value: 0,
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

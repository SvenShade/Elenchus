import type { GraphEdge, GraphNode } from "./graphState";

const MIN_RADIUS = 0.38;
const MAX_RADIUS = 1.02;

export function sphereRadiusForNode(
  node: Pick<GraphNode, "id" | "value">,
  selected = false,
  hovered = false
): number {
  const value = clamp(Number.isFinite(node.value) ? node.value : 0, -1, 1);
  const base = node.id === 0 ? 0.62 : 0.42 + ((value + 1) / 2) * 0.44;
  const boost = selected ? 0.14 : hovered ? 0.08 : 0;
  return clamp(base + boost, MIN_RADIUS, MAX_RADIUS);
}

export function branchToRoot(nodes: GraphNode[], selectedNodeId: number | null): GraphNode[] {
  if (selectedNodeId === null) return [];
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const branch: GraphNode[] = [];
  const seen = new Set<number>();
  let node = nodeMap.get(selectedNodeId);

  while (node && !seen.has(node.id)) {
    branch.push(node);
    seen.add(node.id);
    node = typeof node.parent_id === "number" ? nodeMap.get(node.parent_id) : undefined;
  }

  return branch.reverse();
}

export function branchEdgeIds(branch: GraphNode[]): Set<string> {
  const ids = new Set<string>();
  for (let index = 1; index < branch.length; index += 1) {
    const node = branch[index];
    if (typeof node.parent_id === "number") {
      ids.add(edgeId(node.parent_id, node.id));
    }
  }
  return ids;
}

export function pathEdgeIds(path: number[]): Set<string> {
  const ids = new Set<string>();
  for (let index = 1; index < path.length; index += 1) {
    ids.add(edgeId(path[index - 1], path[index]));
  }
  return ids;
}

export function graphDistancesFromNode(
  nodes: GraphNode[],
  edges: GraphEdge[],
  selectedNodeId: number | null
): Map<number, number> {
  const distances = new Map<number, number>();
  if (selectedNodeId === null || !nodes.some((node) => node.id === selectedNodeId)) {
    return distances;
  }

  const visibleIds = new Set(nodes.map((node) => node.id));
  const adjacency = new Map<number, number[]>();
  for (const node of nodes) {
    adjacency.set(node.id, []);
  }
  for (const edge of edges) {
    if (!visibleIds.has(edge.source) || !visibleIds.has(edge.target)) continue;
    adjacency.get(edge.source)?.push(edge.target);
    adjacency.get(edge.target)?.push(edge.source);
  }

  const queue: number[] = [selectedNodeId];
  distances.set(selectedNodeId, 0);
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const nodeId = queue[cursor];
    const nextDistance = (distances.get(nodeId) ?? 0) + 1;
    for (const next of adjacency.get(nodeId) ?? []) {
      if (distances.has(next)) continue;
      distances.set(next, nextDistance);
      queue.push(next);
    }
  }
  return distances;
}

export function labelOpacityForDistance(distance: number | null | undefined): number {
  if (distance === null || typeof distance === "undefined") return 0.88;
  if (!Number.isFinite(distance)) return 0.18;
  if (distance <= 0) return 1;
  if (distance === 1) return 0.76;
  if (distance === 2) return 0.52;
  if (distance === 3) return 0.34;
  return 0.22;
}

export function valueLabel(value: number): string {
  const bounded = clamp(Number.isFinite(value) ? value : 0, -1, 1);
  if (bounded <= -0.6) return "Dark";
  if (bounded < -0.2) return "Shadowed";
  if (bounded < 0.2) return "Dim";
  if (bounded < 0.6) return "Glowing";
  return "Bright";
}

export function utilityFlashColor(value: number): string {
  const bounded = clamp(Number.isFinite(value) ? value : 0, -1, 1);
  if (bounded >= 0.6) return "#6ee7ff";
  if (bounded >= 0.2) return "#ff3b6b";
  if (bounded <= -0.2) return "#ff6b8c";
  return "#8a3a51";
}

export function visitHeatForNode(
  node: Pick<GraphNode, "id" | "visits">,
  rootVisits: number
): { opacity: number; scale: number } {
  if (node.id === 0) return { opacity: 0.34, scale: 1.18 };
  const share = clamp(node.visits / Math.max(rootVisits, 1), 0, 1);
  return {
    opacity: 0.08 + share * 0.48,
    scale: 1.08 + Math.sqrt(share) * 0.92
  };
}

export function edgeHeatForVisits(targetVisits: number, rootVisits: number): number {
  return clamp(targetVisits / Math.max(rootVisits, 1), 0, 1);
}

export function nodeIdPathToRoot(nodes: GraphNode[], selectedNodeId: number | null): number[] {
  return branchToRoot(nodes, selectedNodeId).map((node) => node.id);
}

export function actionLabel(actionId: string | null | undefined): string {
  if (!actionId) return "Threshold";
  return actionId
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function isEdgeInBranch(edge: GraphEdge, selectedBranchEdgeIds: Set<string>): boolean {
  return selectedBranchEdgeIds.has(edge.id) || selectedBranchEdgeIds.has(edgeId(edge.source, edge.target));
}

function edgeId(source: number, target: number): string {
  return `${source}->${target}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

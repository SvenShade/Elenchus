import type { GraphEdge, GraphNode } from "./graphState";

export type LabyrinthNode = GraphNode & {
  cellX: number;
  cellY: number;
  x: number;
  y: number;
  chamberSize: number;
};

export type LabyrinthEdge = GraphEdge & {
  path: string;
  sourcePoint: { x: number; y: number };
  targetPoint: { x: number; y: number };
};

export type LabyrinthLayout = {
  nodes: LabyrinthNode[];
  edges: LabyrinthEdge[];
  nodeMap: Map<number, LabyrinthNode>;
  edgeMap: Map<string, LabyrinthEdge>;
  bounds: LabyrinthBounds;
};

export type LabyrinthBounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
};

export type GhostDoorLayout = {
  candidateId: string;
  x: number;
  y: number;
  angle: number;
};

const CELL_SIZE = 112;
const CHAMBER_MIN = 34;
const CHAMBER_MAX = 66;
const DIRECTIONS = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 }
];

export function layoutLabyrinthGraph(nodes: GraphNode[], edges: GraphEdge[]): LabyrinthLayout {
  const children = childrenByParent(nodes);
  const occupied = new Set<string>();
  const cells = new Map<number, { x: number; y: number }>();
  const sorted = [...nodes].sort((a, b) => a.depth - b.depth || a.id - b.id);

  for (const node of sorted) {
    if (node.id === 0 || node.parent_id === null) {
      claimCell(node.id, 0, 0, cells, occupied);
      continue;
    }
    const parentCell = cells.get(node.parent_id) ?? { x: 0, y: 0 };
    const siblings = children.get(node.parent_id) ?? [];
    const siblingIndex = Math.max(0, siblings.findIndex((item) => item.id === node.id));
    const preferred = preferredCell(parentCell, node, siblingIndex);
    const cell = firstOpenCell(preferred, occupied);
    claimCell(node.id, cell.x, cell.y, cells, occupied);
  }

  const layoutNodes: LabyrinthNode[] = sorted.map((node) => {
    const cell = cells.get(node.id) ?? { x: 0, y: 0 };
    return {
      ...node,
      cellX: cell.x,
      cellY: cell.y,
      x: cell.x * CELL_SIZE,
      y: cell.y * CELL_SIZE,
      chamberSize: chamberSizeForValue(node)
    };
  });
  const nodeMap = new Map(layoutNodes.map((node) => [node.id, node]));
  const layoutEdges = edges
    .map((edge): LabyrinthEdge | null => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return null;
      return {
        ...edge,
        sourcePoint: { x: source.x, y: source.y },
        targetPoint: { x: target.x, y: target.y },
        path: passagePath(source, target)
      };
    })
    .filter((edge): edge is LabyrinthEdge => Boolean(edge));
  const edgeMap = new Map(layoutEdges.map((edge) => [edge.id, edge]));
  return {
    nodes: layoutNodes,
    edges: layoutEdges,
    nodeMap,
    edgeMap,
    bounds: boundsForNodes(layoutNodes)
  };
}

export function passagePath(source: Pick<LabyrinthNode, "x" | "y">, target: Pick<LabyrinthNode, "x" | "y">): string {
  if (source.x === target.x || source.y === target.y) {
    return `M ${source.x} ${source.y} L ${target.x} ${target.y}`;
  }
  const midX = source.x + (target.x - source.x) * 0.5;
  return `M ${source.x} ${source.y} L ${midX} ${source.y} L ${midX} ${target.y} L ${target.x} ${target.y}`;
}

export function ghostDoorLayouts(
  parent: LabyrinthNode | null | undefined,
  candidateIds: string[],
  radius = 54
): GhostDoorLayout[] {
  if (!parent) return [];
  return candidateIds.map((candidateId, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(candidateIds.length, 1) - Math.PI / 2;
    return {
      candidateId,
      x: parent.x + Math.cos(angle) * radius,
      y: parent.y + Math.sin(angle) * radius,
      angle
    };
  });
}

export function fitViewBox(bounds: LabyrinthBounds, padding = 130): LabyrinthBounds {
  const minX = bounds.minX - padding;
  const minY = bounds.minY - padding;
  const maxX = bounds.maxX + padding;
  const maxY = bounds.maxY + padding;
  return completeBounds(minX, minY, maxX, maxY);
}

export function centeredViewBox(x: number, y: number, width: number, height: number): LabyrinthBounds {
  return completeBounds(x - width / 2, y - height / 2, x + width / 2, y + height / 2);
}

function childrenByParent(nodes: GraphNode[]) {
  const children = new Map<number, GraphNode[]>();
  for (const node of nodes) {
    if (typeof node.parent_id !== "number") continue;
    const list = children.get(node.parent_id) ?? [];
    list.push(node);
    children.set(node.parent_id, list);
  }
  for (const list of children.values()) {
    list.sort((a, b) => a.depth - b.depth || a.id - b.id);
  }
  return children;
}

function preferredCell(
  parentCell: { x: number; y: number },
  node: GraphNode,
  siblingIndex: number
) {
  const direction = DIRECTIONS[(node.depth + siblingIndex + Math.abs(node.parent_id ?? 0)) % DIRECTIONS.length];
  const stride = 1 + Math.floor(siblingIndex / DIRECTIONS.length);
  return {
    x: parentCell.x + direction.x * stride,
    y: parentCell.y + direction.y * stride
  };
}

function firstOpenCell(preferred: { x: number; y: number }, occupied: Set<string>) {
  if (!occupied.has(cellKey(preferred.x, preferred.y))) return preferred;
  for (let radius = 1; radius <= 24; radius += 1) {
    for (const cell of ringCells(preferred.x, preferred.y, radius)) {
      if (!occupied.has(cellKey(cell.x, cell.y))) return cell;
    }
  }
  return preferred;
}

function ringCells(centerX: number, centerY: number, radius: number) {
  const cells: Array<{ x: number; y: number }> = [];
  for (let x = -radius; x <= radius; x += 1) {
    cells.push({ x: centerX + x, y: centerY - radius });
    cells.push({ x: centerX + x, y: centerY + radius });
  }
  for (let y = -radius + 1; y <= radius - 1; y += 1) {
    cells.push({ x: centerX - radius, y: centerY + y });
    cells.push({ x: centerX + radius, y: centerY + y });
  }
  return cells.sort((a, b) => {
    const angleA = Math.atan2(a.y - centerY, a.x - centerX);
    const angleB = Math.atan2(b.y - centerY, b.x - centerX);
    return angleA - angleB || a.x - b.x || a.y - b.y;
  });
}

function claimCell(
  nodeId: number,
  x: number,
  y: number,
  cells: Map<number, { x: number; y: number }>,
  occupied: Set<string>
) {
  cells.set(nodeId, { x, y });
  occupied.add(cellKey(x, y));
}

function cellKey(x: number, y: number) {
  return `${x},${y}`;
}

function chamberSizeForValue(node: GraphNode) {
  if (node.id === 0) return 66;
  const value = Math.max(-1, Math.min(1, Number.isFinite(node.value) ? node.value : 0));
  return CHAMBER_MIN + ((value + 1) / 2) * (CHAMBER_MAX - CHAMBER_MIN);
}

function boundsForNodes(nodes: LabyrinthNode[]): LabyrinthBounds {
  if (nodes.length === 0) return completeBounds(-180, -180, 180, 180);
  const minX = Math.min(...nodes.map((node) => node.x - node.chamberSize));
  const minY = Math.min(...nodes.map((node) => node.y - node.chamberSize));
  const maxX = Math.max(...nodes.map((node) => node.x + node.chamberSize));
  const maxY = Math.max(...nodes.map((node) => node.y + node.chamberSize));
  return completeBounds(minX, minY, maxX, maxY);
}

function completeBounds(minX: number, minY: number, maxX: number, maxY: number): LabyrinthBounds {
  const width = Math.max(360, maxX - minX);
  const height = Math.max(360, maxY - minY);
  return {
    minX,
    minY,
    maxX: minX + width,
    maxY: minY + height,
    width,
    height,
    centerX: minX + width / 2,
    centerY: minY + height / 2
  };
}

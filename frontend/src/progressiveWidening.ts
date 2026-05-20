import { actionLabel, valueLabel } from "./graphMetrics";
import type {
  AbstractStat,
  CandidateOpening,
  CandidateRecord,
  CandidateRejection,
  GraphNode
} from "./graphState";

export type AntechamberCandidate = {
  candidate: CandidateRecord;
  candidateId: string;
  label: string;
  rank: number;
  opening: CandidateOpening | null;
  child: GraphNode | null;
};

export type AntechamberModel = {
  opened: AntechamberCandidate[];
  waiting: AntechamberCandidate[];
  refused: CandidateRejection[];
  openedCount: number;
  wideningLimit: number;
  total: number;
  unopenedCount: number;
};

export type DreamNode = {
  id: string;
  type: string;
  text: string;
  confidence?: number;
  x: number;
  y: number;
};

export type DreamEdge = {
  source: string;
  target: string;
  type: string;
};

export type DreamGraph = {
  nodes: DreamNode[];
  edges: DreamEdge[];
};

export type PrismRow = {
  category: string;
  factor: string;
  assembled: number;
  opened: number;
  imagined: number;
  meanValue: number;
  mirror: string;
};

const BELIEF_TYPES = [
  "claims",
  "reasons",
  "evidence",
  "definitions",
  "values",
  "decisions",
  "memories",
  "tensions"
];

export function candidateId(candidate: CandidateRecord | null | undefined): string {
  return stringValue(candidate?.candidate_id);
}

export function candidateLabel(candidate: CandidateRecord | null | undefined): string {
  return actionLabel(stringValue(candidate?.base_action_id));
}

export function candidateFactorChips(candidate: CandidateRecord | null | undefined): string[] {
  if (!candidate) return [];
  const chips = [
    firstString(candidate.target_ids),
    firstString(candidate.failure_hypothesis_ids),
    stringValue(candidate.diagnostic_operator),
    stringValue(candidate.repair_operator),
    stringValue(candidate.dialogue_form),
    directnessLabel(candidate.directness),
    firstString(candidate.safety_flags)
  ].filter(Boolean);
  return Array.from(new Set(chips)).slice(0, 7);
}

export function buildAntechamber(
  selected: GraphNode | null,
  nodes: GraphNode[],
  candidatePool: CandidateRecord[] = [],
  openingsByCandidate: Record<string, CandidateOpening> = {},
  refused: CandidateRejection[] = []
): AntechamberModel {
  if (!selected) {
    return {
      opened: [],
      waiting: [],
      refused,
      openedCount: 0,
      wideningLimit: 0,
      total: 0,
      unopenedCount: 0
    };
  }

  const childByCandidate = new Map<string, GraphNode>();
  for (const node of nodes) {
    if (node.parent_id !== selected.id || !node.candidate_id) continue;
    childByCandidate.set(node.candidate_id, node);
  }

  const items = candidatePool
    .map((candidate, index): AntechamberCandidate => {
      const id = candidateId(candidate) || `candidate-${index}`;
      return {
        candidate,
        candidateId: id,
        label: candidateLabel(candidate),
        rank: index + 1,
        opening: openingsByCandidate[id] ?? null,
        child: childByCandidate.get(id) ?? null
      };
    })
    .sort(candidateSort);

  const opened = items.filter((item) => item.opening || item.child);
  const waiting = items.filter((item) => !item.opening && !item.child);
  return {
    opened,
    waiting,
    refused,
    openedCount: opened.length,
    wideningLimit: selected.widening_limit ?? opened.length,
    total: items.length,
    unopenedCount: selected.unexpanded_candidate_count ?? waiting.length
  };
}

export function ghostDoorCandidates(model: AntechamberModel, limit = 8): AntechamberCandidate[] {
  return model.waiting.slice(0, limit);
}

export function openingExplanation(candidate: AntechamberCandidate): string {
  const prior = numberValue(candidate.candidate.prior);
  const compatibility = numberValue(candidate.candidate.compatibility_score);
  const score = `prior ${formatScore(prior)}, compatibility ${formatScore(compatibility)}`;
  if (candidate.opening) {
    return `Unlocked Door rank ${candidate.opening.candidate_rank ?? candidate.rank} with Key ${candidate.opening.expanded_children} on the Keyring; ${score}.`;
  }
  return `Locked Door rank ${candidate.rank}, waiting for a Key; ${score}.`;
}

export function normalizeDreamGraph(cognitiveState: Record<string, unknown> | undefined): DreamGraph {
  const graph = recordValue(cognitiveState?.belief_graph);
  const nodes: DreamNode[] = [];
  for (const type of BELIEF_TYPES) {
    const items = Array.isArray(graph[type]) ? graph[type] : [];
    for (const item of items) {
      const object = recordValue(item);
      const id = stringValue(object.id);
      if (!id) continue;
      nodes.push({
        id,
        type: type.replace(/s$/, ""),
        text: stringValue(object.text) || id,
        confidence: numberOrUndefined(object.confidence),
        x: 0,
        y: 0
      });
    }
  }

  const radius = 38;
  const centerX = 50;
  const centerY = 50;
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(nodes.length, 1) - Math.PI / 2;
    node.x = centerX + Math.cos(angle) * radius;
    node.y = centerY + Math.sin(angle) * radius;
  });

  const ids = new Set(nodes.map((node) => node.id));
  const edges = (Array.isArray(graph.edges) ? graph.edges : [])
    .map((edge): DreamEdge => {
      const object = recordValue(edge);
      return {
        source: stringValue(object.source),
        target: stringValue(object.target),
        type: stringValue(object.type)
      };
    })
    .filter((edge) => ids.has(edge.source) && ids.has(edge.target));

  return { nodes, edges };
}

export function buildPrismRows(
  candidates: CandidateRecord[] = [],
  nodes: GraphNode[] = [],
  openingsByCandidate: Record<string, CandidateOpening> = {},
  abstractStats: Record<string, AbstractStat> = {}
): PrismRow[] {
  const rows = new Map<string, Omit<PrismRow, "mirror"> & { valueSum: number }>();
  const nodeByCandidate = new Map<string, GraphNode>();
  for (const node of nodes) {
    if (node.candidate_id) nodeByCandidate.set(node.candidate_id, node);
  }

  for (const candidate of candidates) {
    const id = candidateId(candidate);
    const opened = Boolean(openingsByCandidate[id] || nodeByCandidate.has(id));
    const node = nodeByCandidate.get(id);
    for (const [category, factor] of candidateFactors(candidate)) {
      const key = `${category}:${factor}`;
      const row =
        rows.get(key) ??
        {
          category,
          factor,
          assembled: 0,
          opened: 0,
          imagined: 0,
          meanValue: 0,
          valueSum: 0
        };
      row.assembled += 1;
      if (opened) row.opened += 1;
      if (node) {
        row.imagined += node.visits;
        row.valueSum += node.value;
      }
      rows.set(key, row);
    }
  }

  for (const [key, stat] of Object.entries(abstractStats)) {
    const [rawCategory, ...factorParts] = key.split(":");
    const factor = factorParts.join(":");
    if (!factor) continue;
    if (!isLockworkCategory(rawCategory)) continue;
    const category = rawCategory.replace(/_/g, " ");
    const row =
      rows.get(`${category}:${factor}`) ??
      {
        category,
        factor,
        assembled: 0,
        opened: 0,
        imagined: 0,
        meanValue: 0,
        valueSum: 0
      };
    row.imagined = Math.max(row.imagined, stat.visits ?? 0);
    if (typeof stat.value === "number") {
      row.meanValue = stat.value;
      row.valueSum = stat.value * Math.max(row.opened, 1);
    }
    rows.set(`${category}:${factor}`, row);
  }

  return [...rows.values()]
    .map((row) => {
      const meanValue =
        row.meanValue || (row.opened > 0 ? row.valueSum / Math.max(row.opened, 1) : 0);
      return {
        category: formatLockworkCategory(row.category),
        factor: formatLockworkFactor(row.category, row.factor),
        assembled: row.assembled,
        opened: row.opened,
        imagined: row.imagined,
        meanValue,
        mirror: valueLabel(meanValue)
      };
    })
    .sort((a, b) => b.assembled - a.assembled || b.imagined - a.imagined || a.factor.localeCompare(b.factor))
    .slice(0, 18);
}

export function candidatesTargeting(
  candidates: CandidateRecord[],
  targetId: string | null
): Set<string> {
  if (!targetId) return new Set();
  return new Set(
    candidates
      .filter((candidate) => stringArray(candidate.target_ids).includes(targetId))
      .map(candidateId)
      .filter(Boolean)
  );
}

function candidateFactors(candidate: CandidateRecord): Array<[string, string]> {
  const metadata = recordValue(candidate.action_metadata);
  const baseFactors: Array<[string, string]> = [
    ["base action", stringValue(candidate.base_action_id)],
    ["diagnostic operator", stringValue(candidate.diagnostic_operator)],
    ["repair operator", stringValue(candidate.repair_operator)],
    ["dialogue form", stringValue(candidate.dialogue_form)],
    ["directness bin", directnessBin(candidate.directness)],
    ...stringArray(metadata.failure_family_ids ?? candidate.failure_hypothesis_ids).map(
      (value): [string, string] => ["failure family", value]
    ),
    ...stringArray(metadata.target_types ?? candidate.target_ids).map(
      (value): [string, string] => ["target type", value]
    ),
    ...stringArray(candidate.safety_flags).map((value): [string, string] => ["safety flag", value])
  ];
  return baseFactors.filter(([, value]) => Boolean(value));
}

function isLockworkCategory(rawCategory: string): boolean {
  return new Set([
    "base_action",
    "base action",
    "diagnostic_operator",
    "diagnostic operator",
    "repair_operator",
    "repair operator",
    "dialogue_form",
    "dialogue form",
    "directness_bin",
    "directness bin",
    "failure_family",
    "failure family",
    "target_type",
    "target type",
    "codex_quadrant",
    "codex quadrant",
    "safety_flag",
    "safety flag"
  ]).has(rawCategory);
}

function formatLockworkCategory(value: string): string {
  const normalized = value.replace(/_/g, " ").trim();
  if (normalized === "directness bin") return "Directness";
  return titleCase(normalized);
}

function formatLockworkFactor(category: string, factor: string): string {
  const normalizedCategory = category.replace(/_/g, " ").trim();
  if (normalizedCategory === "directness bin") return directnessBinLabel(factor);
  return titleCase(factor);
}

function directnessBinLabel(value: string): string {
  const normalized = value.trim();
  if (normalized === "0.0-0.2") return "Very gentle";
  if (normalized === "0.2-0.4") return "Gentle";
  if (normalized === "0.4-0.6") return "Balanced";
  if (normalized === "0.6-0.8") return "Direct";
  if (normalized === "0.8-1.0") return "Very direct";
  return normalized;
}

function candidateSort(a: AntechamberCandidate, b: AntechamberCandidate): number {
  return (
    numberValue(b.candidate.prior) - numberValue(a.candidate.prior) ||
    numberValue(b.candidate.compatibility_score) - numberValue(a.candidate.compatibility_score) ||
    a.rank - b.rank
  );
}

function directnessLabel(value: unknown): string {
  if (typeof value !== "number") return "";
  if (value < 0.34) return "low directness";
  if (value > 0.67) return "high directness";
  return "medium directness";
}

function directnessBin(value: unknown): string {
  const directness = Math.max(0, Math.min(1, numberValue(value)));
  const bucket = Math.min(4, Math.floor(directness * 5));
  return `${(bucket / 5).toFixed(1)}-${((bucket + 1) / 5).toFixed(1)}`;
}

function formatScore(value: number): string {
  return value.toFixed(2);
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function firstString(value: unknown): string {
  return stringArray(value)[0] ?? stringValue(value);
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(stringValue).filter(Boolean);
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

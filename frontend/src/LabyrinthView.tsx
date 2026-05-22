import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type WheelEvent as ReactWheelEvent
} from "react";
import type { AntechamberCandidate } from "./progressiveWidening";
import { candidateLabel } from "./progressiveWidening";
import type { GraphEdge, GraphNode, VisualEffect } from "./graphState";
import {
  actionLabel,
  edgeHeatForVisits,
  isEdgeInBranch,
  labelOpacityForDistance,
  utilityFlashColor,
  valueLabel,
  visitHeatForNode
} from "./graphMetrics";
import {
  centeredViewBox,
  fitViewBox,
  ghostDoorLayouts,
  layoutLabyrinthGraph,
  type GhostDoorLayout,
  type LabyrinthBounds,
  type LabyrinthEdge,
  type LabyrinthLayout,
  type LabyrinthNode
} from "./labyrinthLayout";

type LabyrinthViewProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId: number | null;
  selectedBranchNodeIds: Set<number>;
  selectedBranchEdgeIds: Set<string>;
  mctsChosenEdgeIds: Set<string>;
  labelDistances: Map<number, number>;
  rootVisits: number;
  planning: boolean;
  singularityActive: boolean;
  resetSignal: number;
  rootOrbiting: boolean;
  birthGlowNodeIds: Set<number>;
  visualEffects: VisualEffect[];
  selectedGhostDoors: AntechamberCandidate[];
  selectedCandidateId: string | null;
  onSelectNode: (nodeId: number) => void;
  onSelectCandidate: (candidateId: string) => void;
  onManualInteraction: () => void;
};

export function LabyrinthView({
  nodes,
  edges,
  selectedNodeId,
  selectedBranchNodeIds,
  selectedBranchEdgeIds,
  mctsChosenEdgeIds,
  labelDistances,
  rootVisits,
  planning,
  singularityActive,
  resetSignal,
  rootOrbiting,
  birthGlowNodeIds,
  visualEffects,
  selectedGhostDoors,
  selectedCandidateId,
  onSelectNode,
  onSelectCandidate,
  onManualInteraction
}: LabyrinthViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const layout = useMemo(() => layoutLabyrinthGraph(nodes, edges), [nodes, edges]);
  const size = useElementSize(containerRef);
  const { viewBox, animateTo, setDirectly, stopAnimation } = useAnimatedViewBox(
    fitViewBox(layout.bounds)
  );
  const previousResetSignal = useRef(resetSignal);
  const previousNodeCount = useRef(layout.nodes.length);
  const [drag, setDrag] = useState<{
    pointerId: number;
    startX: number;
    startY: number;
    startViewBox: LabyrinthBounds;
  } | null>(null);

  useEffect(() => {
    previousNodeCount.current = Math.min(previousNodeCount.current, layout.nodes.length);
  }, [layout.nodes.length]);

  useEffect(() => {
    if (previousResetSignal.current !== resetSignal) {
      previousResetSignal.current = resetSignal;
      animateTo(fitViewBox(layout.bounds, 150));
      return;
    }

    const nodeCount = layout.nodes.length;
    if (planning && nodeCount > previousNodeCount.current) {
      previousNodeCount.current = nodeCount;
      const newest = [...layout.nodes].sort((a, b) => b.id - a.id)[0];
      const parent = newest && typeof newest.parent_id === "number"
        ? layout.nodeMap.get(newest.parent_id)
        : newest;
      animateTo(viewBoxForFocus(parent ?? null, layout.bounds));
      return;
    }
    previousNodeCount.current = nodeCount;

    if (planning || rootOrbiting || singularityActive) {
      const selected = selectedNodeId === null ? null : layout.nodeMap.get(selectedNodeId);
      const focus = singularityActive || rootOrbiting
        ? layout.nodeMap.get(0) ?? null
        : upstreamFocusNode(selected, layout.nodeMap);
      animateTo(
        singularityActive
          ? centeredViewBox(0, 0, 430, 430)
          : rootOrbiting
            ? fitViewBox(layout.bounds, 150)
            : viewBoxForFocus(focus, layout.bounds)
      );
    }
  }, [
    animateTo,
    layout.bounds,
    layout.nodeMap,
    layout.nodes,
    planning,
    resetSignal,
    rootOrbiting,
    selectedNodeId,
    singularityActive
  ]);

  const selected = selectedNodeId === null ? null : layout.nodeMap.get(selectedNodeId);
  const ghostDoors = useMemo(
    () =>
      ghostDoorLayouts(
        selected,
        selectedGhostDoors.map((candidate) => candidate.candidateId),
        (selected?.chamberSize ?? 56) * 0.88
      ),
    [selected, selectedGhostDoors]
  );

  const handleWheel = useCallback(
    (event: ReactWheelEvent<SVGSVGElement>) => {
      event.preventDefault();
      onManualInteraction();
      stopAnimation();
      const point = clientToWorld(svgRef, viewBox, event.clientX, event.clientY);
      const factor = event.deltaY > 0 ? 1.12 : 0.88;
      const nextWidth = clamp(viewBox.width * factor, 260, 4600);
      const nextHeight = clamp(viewBox.height * factor, 260, 4600);
      const xRatio = (point.x - viewBox.minX) / viewBox.width;
      const yRatio = (point.y - viewBox.minY) / viewBox.height;
      setDirectly({
        minX: point.x - nextWidth * xRatio,
        minY: point.y - nextHeight * yRatio,
        maxX: point.x - nextWidth * xRatio + nextWidth,
        maxY: point.y - nextHeight * yRatio + nextHeight,
        width: nextWidth,
        height: nextHeight,
        centerX: point.x - nextWidth * xRatio + nextWidth / 2,
        centerY: point.y - nextHeight * yRatio + nextHeight / 2
      });
    },
    [onManualInteraction, setDirectly, stopAnimation, viewBox]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (event.button !== 0) return;
      onManualInteraction();
      stopAnimation();
      event.currentTarget.setPointerCapture(event.pointerId);
      setDrag({
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        startViewBox: viewBox
      });
    },
    [onManualInteraction, stopAnimation, viewBox]
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (!drag) return;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const dx = -((event.clientX - drag.startX) / Math.max(rect.width, 1)) * drag.startViewBox.width;
      const dy = -((event.clientY - drag.startY) / Math.max(rect.height, 1)) * drag.startViewBox.height;
      setDirectly(translateBounds(drag.startViewBox, dx, dy));
    },
    [drag, setDirectly]
  );

  const endDrag = useCallback((event: ReactPointerEvent<SVGSVGElement>) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    setDrag(null);
  }, [drag]);

  return (
    <div
      ref={containerRef}
      className={`labyrinth-region ${singularityActive ? "singularity-active" : ""}`}
    >
      <svg
        ref={svgRef}
        className={`labyrinth-map ${drag ? "dragging" : ""}`}
        viewBox={viewBoxString(viewBox)}
        preserveAspectRatio="xMidYMid meet"
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        role="img"
        aria-label="Elenchus MCTS labyrinth map"
      >
        <defs>
          <filter id="labyrinth-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="4.6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <pattern id="labyrinth-stone" width="84" height="84" patternUnits="userSpaceOnUse">
            <path d="M0 42 H84 M42 0 V84" />
          </pattern>
        </defs>
        <rect
          className="labyrinth-floor"
          x={viewBox.minX - viewBox.width}
          y={viewBox.minY - viewBox.height}
          width={viewBox.width * 3}
          height={viewBox.height * 3}
        />
        <rect
          className="labyrinth-stone-pattern"
          x={viewBox.minX - viewBox.width}
          y={viewBox.minY - viewBox.height}
          width={viewBox.width * 3}
          height={viewBox.height * 3}
        />
        <g className="labyrinth-content">
          <g className="labyrinth-passages">
            {layout.edges.map((edge) => (
              <LabyrinthPassage
                key={edge.id}
                edge={edge}
                target={layout.nodeMap.get(edge.target)}
                rootVisits={rootVisits}
                selectedRoute={isEdgeInBranch(edge, selectedBranchEdgeIds)}
                mctsChosen={isEdgeInBranch(edge, mctsChosenEdgeIds)}
                drawing={birthGlowNodeIds.has(edge.target)}
              />
            ))}
          </g>
          <LabyrinthEffects
            effects={visualEffects}
            layout={layout}
          />
          <g className="labyrinth-doors">
            {ghostDoors.map((door) => {
              const candidate = selectedGhostDoors.find((item) => item.candidateId === door.candidateId);
              if (!candidate) return null;
              return (
                <GhostDoorGlyph
                  key={door.candidateId}
                  door={door}
                  candidate={candidate}
                  selected={door.candidateId === selectedCandidateId}
                  onSelectCandidate={onSelectCandidate}
                />
              );
            })}
          </g>
          <g className="labyrinth-chambers">
            {layout.nodes.map((node) => (
              <LabyrinthChamber
                key={node.id}
                node={node}
                selected={node.id === selectedNodeId}
                inSelectedRoute={selectedBranchNodeIds.has(node.id)}
                birthGlowing={birthGlowNodeIds.has(node.id)}
                visitHeat={visitHeatForNode(node, rootVisits)}
                onSelectNode={onSelectNode}
              />
            ))}
          </g>
        </g>
      </svg>
      <LabyrinthLabels
        layout={layout}
        viewBox={viewBox}
        size={size}
        selectedNodeId={selectedNodeId}
        selectedBranchNodeIds={selectedBranchNodeIds}
        labelDistances={labelDistances}
        onSelectNode={onSelectNode}
      />
    </div>
  );
}

function LabyrinthPassage({
  edge,
  target,
  rootVisits,
  selectedRoute,
  mctsChosen,
  drawing
}: {
  edge: LabyrinthEdge;
  target: LabyrinthNode | undefined;
  rootVisits: number;
  selectedRoute: boolean;
  mctsChosen: boolean;
  drawing: boolean;
}) {
  const heat = target ? edgeHeatForVisits(target.visits, rootVisits) : 0;
  return (
    <g
      className={[
        "labyrinth-passage",
        edge.active ? "active" : "",
        selectedRoute ? "selected-route" : "",
        mctsChosen ? "mcts-chosen" : "",
        drawing ? "drawing" : ""
      ].filter(Boolean).join(" ")}
      style={{ "--passage-heat": String(heat) } as CSSProperties}
    >
      <path className="passage-carve" d={edge.path} vectorEffect="non-scaling-stroke" />
      <path className="passage-light" d={edge.path} vectorEffect="non-scaling-stroke" />
    </g>
  );
}

function LabyrinthChamber({
  node,
  selected,
  inSelectedRoute,
  birthGlowing,
  visitHeat,
  onSelectNode
}: {
  node: LabyrinthNode;
  selected: boolean;
  inSelectedRoute: boolean;
  birthGlowing: boolean;
  visitHeat: { opacity: number; scale: number };
  onSelectNode: (nodeId: number) => void;
}) {
  return (
    <g
      className={[
        "labyrinth-chamber",
        node.id === 0 ? "threshold" : "",
        selected ? "selected" : "",
        inSelectedRoute ? "in-route" : "",
        birthGlowing ? "birth-glow" : "",
        node.status
      ].filter(Boolean).join(" ")}
      onClick={(event) => {
        event.stopPropagation();
        onSelectNode(node.id);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") onSelectNode(node.id);
      }}
    >
      <ellipse
        className="chamber-visit-halo"
        cx={node.x}
        cy={node.y}
        rx={node.chamberSize * 0.72 * visitHeat.scale}
        ry={node.chamberSize * 0.56 * visitHeat.scale}
        style={{ opacity: visitHeat.opacity }}
      />
      <path className="chamber-shape" d={chamberPath(node)} vectorEffect="non-scaling-stroke" />
      {node.id === 0 && (
        <>
          <circle className="threshold-sigil" cx={node.x} cy={node.y} r={node.chamberSize * 0.24} />
          <path
            className="threshold-cross"
            d={`M ${node.x - node.chamberSize * 0.2} ${node.y} H ${node.x + node.chamberSize * 0.2} M ${node.x} ${node.y - node.chamberSize * 0.2} V ${node.y + node.chamberSize * 0.2}`}
            vectorEffect="non-scaling-stroke"
          />
        </>
      )}
      <title>{node.id === 0 ? "Threshold Chamber" : `Chamber ${node.id}`}: {actionLabel(node.action_id)} · Lantern {valueLabel(node.value)}</title>
    </g>
  );
}

function GhostDoorGlyph({
  door,
  candidate,
  selected,
  onSelectCandidate
}: {
  door: GhostDoorLayout;
  candidate: AntechamberCandidate;
  selected: boolean;
  onSelectCandidate: (candidateId: string) => void;
}) {
  const rotation = (door.angle * 180) / Math.PI + 90;
  return (
    <g
      className={`labyrinth-door ${selected ? "selected" : ""}`}
      transform={`translate(${door.x} ${door.y}) rotate(${rotation})`}
      onClick={(event) => {
        event.stopPropagation();
        onSelectCandidate(door.candidateId);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") onSelectCandidate(door.candidateId);
      }}
    >
      <path className="door-frame" d="M -13 14 V -5 Q -13 -18 0 -18 Q 13 -18 13 -5 V 14" vectorEffect="non-scaling-stroke" />
      <line className="door-threshold" x1="-14" y1="14" x2="14" y2="14" vectorEffect="non-scaling-stroke" />
      <title>{candidate.label}: prior {score(candidate.candidate.prior)} · fit {score(candidate.candidate.compatibility_score)} · locked door, waiting for a key</title>
      {selected && (
        <foreignObject x="-88" y="-68" width="176" height="48">
          <div className="labyrinth-door-tooltip">
            <strong>{candidate.label}</strong>
            <span>waiting for a key</span>
          </div>
        </foreignObject>
      )}
    </g>
  );
}

function LabyrinthEffects({
  effects,
  layout
}: {
  effects: VisualEffect[];
  layout: LabyrinthLayout;
}) {
  return (
    <g className="labyrinth-effects">
      {effects.slice(-88).map((effect) => {
        if (effect.type === "traversal_pulse") {
          const edge = layout.edgeMap.get(effect.edge_id);
          if (!edge) return null;
          return (
            <circle
              key={effect.id}
              className="labyrinth-traversal-orb"
              r={5 + effect.target_prior * 4}
              style={{ "--omen-color": effect.target_prior > 0.22 ? "#6ee7ff" : utilityFlashColor(effect.target_value) } as CSSProperties}
            >
              <animateMotion dur="0.9s" fill="freeze" path={edge.path} />
            </circle>
          );
        }
        if (effect.type === "judgement_flash") {
          const node = layout.nodeMap.get(effect.node_id);
          if (!node) return null;
          return <OmenFlash key={effect.id} node={node} utility={effect.utility} />;
        }
        if (effect.type === "backup_wave") {
          return <BackupWave key={effect.id} path={effect.path} layout={layout} utility={effect.utility} />;
        }
        if (effect.type === "node_ripple") {
          const node = layout.nodeMap.get(effect.node_id);
          if (!node) return null;
          return <RouteRipple key={effect.id} node={node} utility={effect.utility} />;
        }
        if (effect.type === "transcript_wisp") {
          const node = layout.nodeMap.get(effect.node_id);
          if (!node) return null;
          return <LabyrinthWisp key={effect.id} node={node} text={effect.text} source={effect.source} />;
        }
        if (effect.type === "ghost_door_flare") {
          const node = layout.nodeMap.get(effect.node_id);
          if (!node) return null;
          return <DoorFlare key={effect.id} node={node} />;
        }
        return null;
      })}
    </g>
  );
}

function OmenFlash({ node, utility }: { node: LabyrinthNode; utility: number }) {
  return (
    <circle
      className="labyrinth-omen-flash"
      cx={node.x}
      cy={node.y}
      r={node.chamberSize * 0.64}
      style={{ "--omen-color": utilityFlashColor(utility) } as CSSProperties}
    />
  );
}

function RouteRipple({ node, utility }: { node: LabyrinthNode; utility: number }) {
  return (
    <circle
      className="labyrinth-route-ripple"
      cx={node.x}
      cy={node.y}
      r={node.chamberSize * 0.58}
      style={{ "--omen-color": utilityFlashColor(utility) } as CSSProperties}
    />
  );
}

function DoorFlare({ node }: { node: LabyrinthNode }) {
  return (
    <circle
      className="labyrinth-door-flare"
      cx={node.x}
      cy={node.y}
      r={node.chamberSize * 0.9}
    />
  );
}

function BackupWave({
  path,
  layout,
  utility
}: {
  path: number[];
  layout: LabyrinthLayout;
  utility: number;
}) {
  const ids = path.length > 1 ? path : [];
  return (
    <>
      {ids.slice(1).map((nodeId, index) => {
        const previous = ids[index];
        const edge = layout.edgeMap.get(`${previous}->${nodeId}`);
        if (!edge) return null;
        return (
          <path
            key={`${previous}->${nodeId}`}
            className="labyrinth-backup-wave"
            d={edge.path}
            vectorEffect="non-scaling-stroke"
            style={{
              "--omen-color": utilityFlashColor(utility),
              animationDelay: `${(ids.length - index - 1) * 90}ms`
            } as CSSProperties}
          />
        );
      })}
    </>
  );
}

function LabyrinthWisp({
  node,
  text,
  source
}: {
  node: LabyrinthNode;
  text: string;
  source: "node_added" | "rollout_step";
}) {
  return (
    <foreignObject
      className={`labyrinth-wisp-object ${source}`}
      x={node.x + node.chamberSize * 0.3}
      y={node.y - node.chamberSize * 1.05}
      width="170"
      height="104"
    >
      <div className={`labyrinth-wisp ${source}`}>
        {text}
      </div>
    </foreignObject>
  );
}

function LabyrinthLabels({
  layout,
  viewBox,
  size,
  selectedNodeId,
  selectedBranchNodeIds,
  labelDistances,
  onSelectNode
}: {
  layout: LabyrinthLayout;
  viewBox: LabyrinthBounds;
  size: { width: number; height: number };
  selectedNodeId: number | null;
  selectedBranchNodeIds: Set<number>;
  labelDistances: Map<number, number>;
  onSelectNode: (nodeId: number) => void;
}) {
  if (size.width === 0 || size.height === 0) return null;
  return (
    <div className="labyrinth-label-layer" aria-hidden="true">
      {layout.nodes.map((node) => {
        const position = worldToScreen(node.x, node.y + node.chamberSize * 0.68, viewBox, size);
        const opacity = labelOpacityForDistance(
          selectedNodeId === null ? null : labelDistances.get(node.id) ?? Number.POSITIVE_INFINITY
        );
        return (
          <button
            key={node.id}
            type="button"
            className={[
              "labyrinth-node-label",
              node.id === selectedNodeId ? "selected" : "",
              selectedBranchNodeIds.has(node.id) ? "in-route" : ""
            ].filter(Boolean).join(" ")}
            style={{
              left: position.x,
              top: position.y,
              opacity: node.id === selectedNodeId ? 1 : opacity
            }}
            onClick={() => onSelectNode(node.id)}
            aria-label={`Select ${node.id === 0 ? "Threshold Chamber" : `Chamber ${node.id}`}`}
          >
            {node.id === 0 ? "Threshold" : actionLabel(node.action_id)}
          </button>
        );
      })}
    </div>
  );
}

function useElementSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => {
      const rect = element.getBoundingClientRect();
      setSize({ width: rect.width, height: rect.height });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

function useAnimatedViewBox(initial: LabyrinthBounds) {
  const [viewBox, setViewBox] = useState(initial);
  const targetRef = useRef(initial);
  const rafRef = useRef<number | null>(null);

  const stopAnimation = useCallback(() => {
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const tick = useCallback(() => {
    setViewBox((current) => {
      const target = targetRef.current;
      const next = lerpBounds(current, target, 0.085);
      if (boundsClose(next, target)) {
        rafRef.current = null;
        return target;
      }
      rafRef.current = window.requestAnimationFrame(tick);
      return next;
    });
  }, []);

  const animateTo = useCallback((target: LabyrinthBounds) => {
    targetRef.current = target;
    if (rafRef.current === null) {
      rafRef.current = window.requestAnimationFrame(tick);
    }
  }, [tick]);

  const setDirectly = useCallback((target: LabyrinthBounds) => {
    stopAnimation();
    targetRef.current = target;
    setViewBox(target);
  }, [stopAnimation]);

  useEffect(() => stopAnimation, [stopAnimation]);

  return { viewBox, animateTo, setDirectly, stopAnimation };
}

function viewBoxForFocus(focus: LabyrinthNode | null, bounds: LabyrinthBounds): LabyrinthBounds {
  if (!focus) return fitViewBox(bounds, 150);
  const fitted = fitViewBox(bounds, 150);
  return centeredViewBox(focus.x, focus.y, Math.max(440, fitted.width), Math.max(420, fitted.height));
}

function upstreamFocusNode(
  selected: LabyrinthNode | null | undefined,
  nodeMap: Map<number, LabyrinthNode>
): LabyrinthNode | null {
  if (!selected) return nodeMap.get(0) ?? null;
  if (typeof selected.parent_id === "number") return nodeMap.get(selected.parent_id) ?? selected;
  return selected;
}

function chamberPath(node: LabyrinthNode): string {
  const width = node.chamberSize * (node.id === 0 ? 1.12 : 1);
  const height = node.chamberSize * 0.9;
  const x = node.x - width / 2;
  const y = node.y - height / 2;
  return [
    `M ${x} ${y + height * 0.92}`,
    `L ${x} ${y + height * 0.42}`,
    `C ${x} ${y + height * 0.16} ${x + width * 0.2} ${y} ${x + width * 0.5} ${y}`,
    `C ${x + width * 0.8} ${y} ${x + width} ${y + height * 0.16} ${x + width} ${y + height * 0.42}`,
    `L ${x + width} ${y + height * 0.92}`,
    `Q ${x + width} ${y + height} ${x + width * 0.88} ${y + height}`,
    `L ${x + width * 0.12} ${y + height}`,
    `Q ${x} ${y + height} ${x} ${y + height * 0.92}`,
    "Z"
  ].join(" ");
}

function clientToWorld(
  svgRef: RefObject<SVGSVGElement | null>,
  viewBox: LabyrinthBounds,
  clientX: number,
  clientY: number
) {
  const rect = svgRef.current?.getBoundingClientRect();
  if (!rect) return { x: viewBox.centerX, y: viewBox.centerY };
  return {
    x: viewBox.minX + ((clientX - rect.left) / Math.max(rect.width, 1)) * viewBox.width,
    y: viewBox.minY + ((clientY - rect.top) / Math.max(rect.height, 1)) * viewBox.height
  };
}

function worldToScreen(
  x: number,
  y: number,
  viewBox: LabyrinthBounds,
  size: { width: number; height: number }
) {
  return {
    x: ((x - viewBox.minX) / viewBox.width) * size.width,
    y: ((y - viewBox.minY) / viewBox.height) * size.height
  };
}

function viewBoxString(bounds: LabyrinthBounds) {
  return `${bounds.minX} ${bounds.minY} ${bounds.width} ${bounds.height}`;
}

function translateBounds(bounds: LabyrinthBounds, dx: number, dy: number): LabyrinthBounds {
  return {
    minX: bounds.minX + dx,
    minY: bounds.minY + dy,
    maxX: bounds.maxX + dx,
    maxY: bounds.maxY + dy,
    width: bounds.width,
    height: bounds.height,
    centerX: bounds.centerX + dx,
    centerY: bounds.centerY + dy
  };
}

function lerpBounds(current: LabyrinthBounds, target: LabyrinthBounds, alpha: number): LabyrinthBounds {
  const minX = current.minX + (target.minX - current.minX) * alpha;
  const minY = current.minY + (target.minY - current.minY) * alpha;
  const width = current.width + (target.width - current.width) * alpha;
  const height = current.height + (target.height - current.height) * alpha;
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

function boundsClose(a: LabyrinthBounds, b: LabyrinthBounds): boolean {
  return (
    Math.abs(a.minX - b.minX) < 0.5 &&
    Math.abs(a.minY - b.minY) < 0.5 &&
    Math.abs(a.width - b.width) < 0.5 &&
    Math.abs(a.height - b.height) < 0.5
  );
}

function score(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

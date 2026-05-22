import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { forceCenter, forceLink, forceManyBody, forceSimulation, forceZ } from "d3-force-3d";
import { Crosshair } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AdditiveBlending, MathUtils, Vector3, type BufferAttribute, type Group, type MeshBasicMaterial } from "three";
import {
  filterGraphByImportance,
  type AbstractStat,
  type CandidateOpening,
  type CandidateRecord,
  type CandidateRejection,
  type CoalescenceState,
  type ExplorationSummary,
  type GraphEdge,
  type GraphNode,
  type ReflectionItem,
  type RolloutReflection,
  type SingularityState,
  type VisualEffect
} from "./graphState";
import {
  actionLabel,
  branchEdgeIds,
  branchToRoot,
  edgeHeatForVisits,
  graphDistancesFromNode,
  isEdgeInBranch,
  labelOpacityForDistance,
  pathEdgeIds,
  sphereRadiusForNode,
  utilityFlashColor,
  visitHeatForNode,
  valueLabel
} from "./graphMetrics";
import { LabyrinthView } from "./LabyrinthView";
import {
  buildAntechamber,
  buildPrismRows,
  candidateFactorChips,
  candidateId,
  candidateLabel,
  candidatesTargeting,
  ghostDoorCandidates,
  normalizeDreamGraph,
  openingExplanation,
  type AntechamberCandidate,
  type AntechamberModel,
  type DreamGraph,
  type DreamNode,
  type PrismRow
} from "./progressiveWidening";

type LayoutNode = GraphNode & {
  x: number;
  y: number;
  z: number;
};

type GraphViewProps = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNodeId: number | null;
  chosenNodeId: number | null;
  chosenPath: number[];
  graphDetail: number;
  visualisation: "labyrinth" | "constellation";
  planning: boolean;
  explorationSummary: ExplorationSummary | null;
  rolloutReflection: RolloutReflection | null;
  coalescence: CoalescenceState;
  singularity: SingularityState | null;
  visualEffects: VisualEffect[];
  candidatePoolsByNode: Record<number, CandidateRecord[]>;
  candidateRejectionsByNode: Record<number, CandidateRejection[]>;
  candidateOpeningsByCandidate: Record<string, CandidateOpening>;
  cognitiveStatesByNode: Record<number, Record<string, unknown>>;
  selectedCandidateId: string | null;
  abstractStats: Record<string, AbstractStat>;
  onSelectNode: (nodeId: number) => void;
  onSelectCandidate: (candidateId: string) => void;
};

export function GraphView({
  nodes,
  edges,
  selectedNodeId,
  chosenNodeId,
  chosenPath,
  graphDetail,
  visualisation,
  planning,
  explorationSummary,
  rolloutReflection,
  coalescence,
  singularity,
  visualEffects,
  candidatePoolsByNode,
  candidateRejectionsByNode,
  candidateOpeningsByCandidate,
  cognitiveStatesByNode,
  selectedCandidateId,
  abstractStats,
  onSelectNode,
  onSelectCandidate
}: GraphViewProps) {
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);
  const [cameraResetSignal, setCameraResetSignal] = useState(0);
  const [cameraInterruptSignal, setCameraInterruptSignal] = useState(0);
  const [rootOrbiting, setRootOrbiting] = useState(false);
  const [birthGlowNodeIds, setBirthGlowNodeIds] = useState<Set<number>>(new Set());
  const [finishedSingularityIds, setFinishedSingularityIds] = useState<Set<number>>(new Set());
  const seenNodeIds = useRef<Set<number>>(new Set());
  const glowTimers = useRef<number[]>([]);
  const singularityActive = Boolean(singularity && !finishedSingularityIds.has(singularity.id));
  const visible = useMemo(
    () => filterGraphByImportance(nodes, edges, graphDetail, selectedNodeId),
    [nodes, edges, graphDetail, selectedNodeId]
  );
  const layout = useMemo(() => layoutGraph(visible.nodes, visible.edges), [visible.nodes, visible.edges]);
  const singularityLayout = useMemo(
    () =>
      singularity
        ? layoutGraph(singularity.snapshot_nodes, singularity.snapshot_edges)
        : null,
    [singularity]
  );
  const rootVisits = Math.max(nodes.find((node) => node.id === 0)?.visits ?? 0, 1);
  const selected = nodes.find((node) => node.id === selectedNodeId) ?? null;
  const selectedBranch = useMemo(() => branchToRoot(nodes, selectedNodeId), [nodes, selectedNodeId]);
  const selectedBranchNodeIds = useMemo(
    () => new Set(selectedBranch.map((node) => node.id)),
    [selectedBranch]
  );
  const selectedBranchEdgeIds = useMemo(() => branchEdgeIds(selectedBranch), [selectedBranch]);
  const mctsChosenEdgeIds = useMemo(() => {
    const explicitPath = pathEdgeIds(chosenPath);
    if (explicitPath.size > 0) return explicitPath;
    return branchEdgeIds(branchToRoot(nodes, chosenNodeId));
  }, [chosenNodeId, chosenPath, nodes]);
  const labelDistances = useMemo(
    () => graphDistancesFromNode(visible.nodes, visible.edges, selectedNodeId),
    [visible.nodes, visible.edges, selectedNodeId]
  );
  const selectedCandidatePool = selected ? candidatePoolsByNode[selected.id] ?? [] : [];
  const selectedAntechamber = useMemo(
    () =>
      buildAntechamber(
        selected,
        nodes,
        selectedCandidatePool,
        candidateOpeningsByCandidate,
        selected ? candidateRejectionsByNode[selected.id] ?? [] : []
      ),
    [candidateOpeningsByCandidate, candidateRejectionsByNode, nodes, selected, selectedCandidatePool]
  );
  const selectedGhostDoors = useMemo(
    () => ghostDoorCandidates(selectedAntechamber),
    [selectedAntechamber]
  );
  useEffect(() => {
    if (nodes.length === 0) {
      seenNodeIds.current.clear();
      glowTimers.current.forEach((timer) => window.clearTimeout(timer));
      glowTimers.current = [];
      setBirthGlowNodeIds(new Set());
      return;
    }
    const newIds = nodes
      .map((node) => node.id)
      .filter((id) => !seenNodeIds.current.has(id));
    if (newIds.length === 0) return;
    for (const id of newIds) seenNodeIds.current.add(id);
    setBirthGlowNodeIds((current) => {
      const next = new Set(current);
      for (const id of newIds) next.add(id);
      return next;
    });
    const timer = window.setTimeout(() => {
      setBirthGlowNodeIds((current) => {
        const next = new Set(current);
        for (const id of newIds) next.delete(id);
        return next;
      });
    }, 1500);
    glowTimers.current.push(timer);
  }, [nodes]);
  useEffect(() => {
    return () => {
      glowTimers.current.forEach((timer) => window.clearTimeout(timer));
    };
  }, []);
  useEffect(() => {
    if (!singularity) {
      setFinishedSingularityIds(new Set());
      return;
    }
    const timer = window.setTimeout(() => {
      setFinishedSingularityIds((current) => new Set(current).add(singularity.id));
    }, 1350);
    return () => window.clearTimeout(timer);
  }, [singularity]);

  return (
    <section className="graph-panel" aria-label="MCTS chamber visualization">
      <div className="graph-canvas-region">
        <div className="graph-status">
          <span>{visible.nodes.length}/{nodes.length} chambers</span>
          <span>{visible.edges.length}/{edges.length} passages</span>
          <span>{graphDetail}% map detail</span>
          <span>{visualisation === "labyrinth" ? "Labyrinth" : "Constellation"}</span>
        </div>
        {visualisation === "labyrinth" ? (
          <LabyrinthView
            nodes={visible.nodes}
            edges={visible.edges}
            selectedNodeId={selectedNodeId}
            selectedBranchNodeIds={selectedBranchNodeIds}
            selectedBranchEdgeIds={selectedBranchEdgeIds}
            mctsChosenEdgeIds={mctsChosenEdgeIds}
            labelDistances={labelDistances}
            rootVisits={rootVisits}
            planning={planning}
            singularityActive={singularityActive}
            resetSignal={cameraResetSignal}
            rootOrbiting={rootOrbiting}
            birthGlowNodeIds={birthGlowNodeIds}
            visualEffects={visualEffects}
            selectedGhostDoors={selectedGhostDoors}
            selectedCandidateId={selectedCandidateId}
            onSelectNode={(nodeId) => {
              setRootOrbiting(false);
              onSelectNode(nodeId);
            }}
            onSelectCandidate={onSelectCandidate}
            onManualInteraction={() => setRootOrbiting(false)}
          />
        ) : (
          <Canvas camera={{ position: [0, 0, 34], fov: 48 }} dpr={[1, 2]}>
            <color attach="background" args={["#09070b"]} />
            <ambientLight intensity={0.78} />
            <directionalLight position={[10, 12, 8]} intensity={1.55} />
            <OrbitControls
              enableDamping
              makeDefault
              autoRotate={planning || rootOrbiting || singularityActive}
              autoRotateSpeed={singularityActive ? 4.2 : 0.22}
              onStart={() => {
                setRootOrbiting(false);
                setCameraInterruptSignal((signal) => signal + 1);
              }}
            />
            <AnimatedCamera
              nodes={layout.nodes}
              nodeMap={layout.nodeMap}
              selectedNodeId={selectedNodeId}
              planning={planning}
              singularityActive={singularityActive}
              rootOrbiting={rootOrbiting}
              resetSignal={cameraResetSignal}
              interruptSignal={cameraInterruptSignal}
            />
            <group>
              {layout.edges.map((edge) => {
                const source = layout.nodeMap.get(edge.source);
                const target = layout.nodeMap.get(edge.target);
                if (!source || !target) return null;
                return (
                  <EdgeLine
                    key={edge.id}
                  source={source}
                  target={target}
                  traversalActive={planning && edge.active}
                  selectedBranchActive={isEdgeInBranch(edge, selectedBranchEdgeIds)}
                  mctsChosenActive={isEdgeInBranch(edge, mctsChosenEdgeIds)}
                  heat={edgeHeatForVisits(target.visits, rootVisits)}
                  />
                );
              })}
              <EffectLayer
                effects={visualEffects}
                nodeMap={layout.nodeMap}
                edgeMap={layout.edgeMap}
              />
              {layout.nodes.map((node) => (
                <GraphSphere
                  key={node.id}
                  node={node}
                  selected={node.id === selectedNodeId}
                  hovered={node.id === hoveredNode}
                  inSelectedBranch={selectedBranchNodeIds.has(node.id)}
                  birthGlowing={birthGlowNodeIds.has(node.id)}
                  visitHeat={visitHeatForNode(node, rootVisits)}
                  labelOpacity={labelOpacityForDistance(
                    selectedNodeId === null ? null : labelDistances.get(node.id) ?? Number.POSITIVE_INFINITY
                  )}
                  spawnPosition={spawnPositionForNode(node, layout.nodeMap)}
                  onPointerOver={() => setHoveredNode(node.id)}
                  onPointerOut={() => setHoveredNode(null)}
                  onClick={() => {
                    setRootOrbiting(false);
                    onSelectNode(node.id);
                  }}
                />
              ))}
              {selected && (
                <GhostDoorLayer
                  parent={layout.nodeMap.get(selected.id) ?? null}
                  candidates={selectedGhostDoors}
                  selectedCandidateId={selectedCandidateId}
                  onSelectCandidate={onSelectCandidate}
                />
              )}
            </group>
            {singularity && singularityLayout && singularityActive && (
              <SingularitySnapshot
                singularity={singularity}
                nodes={singularityLayout.nodes}
                edges={singularityLayout.edges}
                nodeMap={singularityLayout.nodeMap}
              />
            )}
          </Canvas>
        )}
        <button
          className="viewport-recenter-button"
          type="button"
          title="Recentre on the Threshold"
          aria-label="Recentre on the Threshold and zoom out"
          onClick={() => {
            setRootOrbiting(true);
            setCameraResetSignal((signal) => signal + 1);
          }}
        >
          <Crosshair size={18} />
        </button>
      </div>
      <BranchInspector
        selected={selected}
        branchNodes={selectedBranch}
        explorationSummary={explorationSummary}
        rolloutReflection={rolloutReflection}
        coalescence={coalescence}
        nodes={nodes}
        candidatePoolsByNode={candidatePoolsByNode}
        candidateRejectionsByNode={candidateRejectionsByNode}
        candidateOpeningsByCandidate={candidateOpeningsByCandidate}
        cognitiveStatesByNode={cognitiveStatesByNode}
        selectedCandidateId={selectedCandidateId}
        abstractStats={abstractStats}
        onSelectCandidate={onSelectCandidate}
      />
    </section>
  );
}

function GraphSphere({
  node,
  selected,
  hovered,
  inSelectedBranch,
  birthGlowing,
  visitHeat,
  labelOpacity,
  spawnPosition,
  onPointerOver,
  onPointerOut,
  onClick
}: {
  node: LayoutNode;
  selected: boolean;
  hovered: boolean;
  inSelectedBranch: boolean;
  birthGlowing: boolean;
  visitHeat: { opacity: number; scale: number };
  labelOpacity: number;
  spawnPosition: [number, number, number];
  onPointerOver: () => void;
  onPointerOut: () => void;
  onClick: () => void;
}) {
  const groupRef = useRef<Group>(null);
  const glowRef = useRef<Group>(null);
  const glowMaterialRef = useRef<MeshBasicMaterial>(null);
  const birthGlowStrength = useRef(birthGlowing ? 1 : 0);
  const radius = sphereRadiusForNode(node, selected, hovered);
  const color = colorForNode(node, selected);
  const emissiveIntensity = selected ? 0.46 : inSelectedBranch ? 0.22 : 0.06;
  useLayoutEffect(() => {
    groupRef.current?.position.set(...spawnPosition);
    groupRef.current?.scale.setScalar(0.16);
  }, [node.id]);
  useEffect(() => {
    if (birthGlowing) birthGlowStrength.current = 1;
  }, [birthGlowing]);
  useFrame((_, delta) => {
    const group = groupRef.current;
    if (!group) return;
    group.position.lerp(targetVectorForNode(node), 1 - Math.exp(-delta * 7.5));
    const targetScale = selected ? 1.1 : hovered ? 1.06 : 1;
    const scale = MathUtils.damp(group.scale.x, targetScale, 8, delta);
    group.scale.setScalar(scale);
    birthGlowStrength.current = MathUtils.damp(birthGlowStrength.current, 0, 3.1, delta);
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1.25 + (1 - birthGlowStrength.current) * 1.15);
    }
    if (glowMaterialRef.current) {
      glowMaterialRef.current.opacity = Math.pow(birthGlowStrength.current, 1.25) * 0.62;
    }
  });
  return (
    <group ref={groupRef}>
      <mesh raycast={() => null}>
        <sphereGeometry args={[radius * visitHeat.scale, 32, 20]} />
        <meshBasicMaterial
          color={visitHeat.opacity > 0.34 ? "#6ee7ff" : "#ff3b6b"}
          transparent
          opacity={visitHeat.opacity}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
      <group ref={glowRef}>
        <mesh raycast={() => null}>
          <sphereGeometry args={[radius * 1.82, 32, 20]} />
          <meshBasicMaterial
            ref={glowMaterialRef}
            color="#ff3b6b"
            transparent
            opacity={birthGlowing ? 0.62 : 0}
            depthWrite={false}
            blending={AdditiveBlending}
          />
        </mesh>
      </group>
      <mesh
        onPointerOver={(event) => {
          event.stopPropagation();
          onPointerOver();
        }}
        onPointerOut={(event) => {
          event.stopPropagation();
          onPointerOut();
        }}
        onClick={(event) => {
          event.stopPropagation();
          onClick();
        }}
      >
        <sphereGeometry args={[Math.max(radius * 1.85, 0.9), 24, 16]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      <mesh
        raycast={() => null}
        castShadow
      >
        <sphereGeometry args={[radius, 32, 24]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={emissiveIntensity} />
      </mesh>
      <Html center position={[0, -1.05, 0]} zIndexRange={[10, 0]}>
        <button
          className={`node-label ${selected ? "selected" : ""} ${inSelectedBranch ? "in-branch" : ""}`}
          data-testid={`node-label-${node.id}`}
          type="button"
          style={{ opacity: selected ? 1 : labelOpacity }}
          onClick={onClick}
        >
          {actionLabel(node.action_id)}
        </button>
      </Html>
      {hovered && (
        <Html center position={[0.9, 0.9, 0]} zIndexRange={[20, 0]}>
          <div className="graph-tooltip">
            <strong>{actionLabel(node.action_id)}</strong>
            <span>{valueLabel(node.value)} ({node.value.toFixed(2)})</span>
          </div>
        </Html>
      )}
    </group>
  );
}

function GhostDoorLayer({
  parent,
  candidates,
  selectedCandidateId,
  onSelectCandidate
}: {
  parent: LayoutNode | null;
  candidates: AntechamberCandidate[];
  selectedCandidateId: string | null;
  onSelectCandidate: (candidateId: string) => void;
}) {
  if (!parent || candidates.length === 0) return null;
  return (
    <group>
      {candidates.map((candidate, index) => (
        <GhostDoor
          key={candidate.candidateId}
          parent={parent}
          candidate={candidate}
          index={index}
          total={candidates.length}
          selected={candidate.candidateId === selectedCandidateId}
          onSelectCandidate={onSelectCandidate}
        />
      ))}
    </group>
  );
}

function GhostDoor({
  parent,
  candidate,
  index,
  total,
  selected,
  onSelectCandidate
}: {
  parent: LayoutNode;
  candidate: AntechamberCandidate;
  index: number;
  total: number;
  selected: boolean;
  onSelectCandidate: (candidateId: string) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const groupRef = useRef<Group>(null);
  const materialRef = useRef<MeshBasicMaterial>(null);
  const angle = (Math.PI * 2 * index) / Math.max(total, 1) - Math.PI / 2;
  const radius = 2.35 + Math.min(total, 8) * 0.035;
  const target = new Vector3(
    parent.x + Math.cos(angle) * radius,
    parent.y + Math.sin(angle) * radius,
    parent.z + 0.35
  );
  useFrame(({ clock }, delta) => {
    if (!groupRef.current) return;
    groupRef.current.position.lerp(target, 1 - Math.exp(-delta * 8));
    groupRef.current.rotation.z = angle + Math.PI / 2 + Math.sin(clock.elapsedTime * 1.2 + index) * 0.05;
    const scale = selected ? 1.18 : hovered ? 1.1 : 1;
    groupRef.current.scale.setScalar(MathUtils.damp(groupRef.current.scale.x, scale, 9, delta));
    if (materialRef.current) {
      materialRef.current.opacity = selected ? 0.72 : hovered ? 0.58 : 0.34;
    }
  });
  return (
    <group ref={groupRef} position={target}>
      <mesh
        onPointerOver={(event) => {
          event.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={(event) => {
          event.stopPropagation();
          setHovered(false);
        }}
        onClick={(event) => {
          event.stopPropagation();
          onSelectCandidate(candidate.candidateId);
        }}
      >
        <boxGeometry args={[0.5, 0.86, 0.08]} />
        <meshBasicMaterial
          ref={materialRef}
          color={selected ? "#6ee7ff" : "#ff3b6b"}
          transparent
          opacity={0.34}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
      {(hovered || selected) && (
        <Html center position={[0, 0.92, 0]} zIndexRange={[24, 0]}>
          <div className="ghost-door-label">
            <strong>{candidate.label}</strong>
            <span>prior {score(candidate.candidate.prior)} · fit {score(candidate.candidate.compatibility_score)}</span>
            <em>locked door · waiting for a key</em>
          </div>
        </Html>
      )}
    </group>
  );
}

function EffectLayer({
  effects,
  nodeMap,
  edgeMap
}: {
  effects: VisualEffect[];
  nodeMap: Map<number, LayoutNode>;
  edgeMap: Map<string, { source: LayoutNode; target: LayoutNode }>;
}) {
  return (
    <>
      {effects.slice(-80).map((effect) => {
        if (effect.type === "traversal_pulse") {
          const edge = edgeMap.get(effect.edge_id);
          if (!edge) return null;
          return <TraversalPulse key={effect.id} effect={effect} source={edge.source} target={edge.target} />;
        }
        if (effect.type === "judgement_flash") {
          const node = nodeMap.get(effect.node_id);
          if (!node) return null;
          return <JudgementFlash key={effect.id} node={node} utility={effect.utility} />;
        }
        if (effect.type === "backup_wave") {
          const nodes = effect.path
            .map((nodeId) => nodeMap.get(nodeId))
            .filter((node): node is LayoutNode => Boolean(node));
          return <BackupWave key={effect.id} nodes={nodes} utility={effect.utility} />;
        }
        if (effect.type === "node_ripple") {
          const node = nodeMap.get(effect.node_id);
          if (!node) return null;
          return <NodeRipple key={effect.id} node={node} utility={effect.utility} />;
        }
        if (effect.type === "transcript_wisp") {
          const node = nodeMap.get(effect.node_id);
          if (!node) return null;
          return <TranscriptWisp key={effect.id} node={node} text={effect.text} source={effect.source} />;
        }
        if (effect.type === "ghost_door_flare") {
          const node = nodeMap.get(effect.node_id);
          if (!node) return null;
          return <GhostDoorFlare key={effect.id} node={node} candidate={effect.candidate} />;
        }
        return null;
      })}
    </>
  );
}

function GhostDoorFlare({ node, candidate }: { node: LayoutNode; candidate: CandidateRecord | null }) {
  const meshRef = useRef<Group>(null);
  const materialRef = useRef<MeshBasicMaterial>(null);
  const start = useRef<number | null>(null);
  useFrame(({ clock }) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const age = clock.elapsedTime - start.current;
    const progress = MathUtils.clamp(age / 0.9, 0, 1);
    if (meshRef.current) {
      meshRef.current.position.set(node.x + 1.2, node.y + 0.6, node.z + 0.5);
      meshRef.current.scale.setScalar(0.6 + progress * 2.7);
    }
    if (materialRef.current) {
      materialRef.current.opacity = Math.sin(progress * Math.PI) * 0.7;
    }
  });
  return (
    <group ref={meshRef} frustumCulled={false}>
      <mesh raycast={() => null}>
        <boxGeometry args={[0.52, 0.9, 0.08]} />
        <meshBasicMaterial
          ref={materialRef}
          color={candidate?.base_action_id ? "#6ee7ff" : "#ff3b6b"}
          transparent
          opacity={0}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function TraversalPulse({
  effect,
  source,
  target
}: {
  effect: Extract<VisualEffect, { type: "traversal_pulse" }>;
  source: LayoutNode;
  target: LayoutNode;
}) {
  const meshRef = useRef<Group>(null);
  const materialRef = useRef<MeshBasicMaterial>(null);
  const start = useRef<number | null>(null);
  const color = effect.target_prior > 0.22 ? "#6ee7ff" : utilityFlashColor(effect.target_value);
  useFrame(({ clock }) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const age = clock.elapsedTime - start.current;
    const progress = MathUtils.clamp(age / 0.9, 0, 1);
    const position = targetVectorForNode(source).lerp(targetVectorForNode(target), progress);
    if (meshRef.current) {
      meshRef.current.position.copy(position);
      meshRef.current.scale.setScalar(0.9 + Math.sin(progress * Math.PI) * 1.35);
    }
    if (materialRef.current) {
      materialRef.current.opacity = Math.sin(progress * Math.PI) * 0.92;
    }
  });
  return (
    <group ref={meshRef} frustumCulled={false}>
      <mesh raycast={() => null}>
        <sphereGeometry args={[0.18 + effect.target_prior * 0.28, 18, 14]} />
        <meshBasicMaterial
          ref={materialRef}
          color={color}
          transparent
          opacity={0}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function JudgementFlash({ node, utility }: { node: LayoutNode; utility: number }) {
  return <NodePulse node={node} color={utilityFlashColor(utility)} duration={1} radiusMultiplier={2.45} />;
}

function BackupWave({ nodes, utility }: { nodes: LayoutNode[]; utility: number }) {
  return (
    <>
      {nodes.map((node, index) => (
        <NodePulse
          key={`${node.id}-${index}`}
          node={node}
          color={utilityFlashColor(utility)}
          duration={1.2}
          delay={(nodes.length - 1 - index) * 0.16}
          radiusMultiplier={1.95}
        />
      ))}
    </>
  );
}

function NodeRipple({ node, utility }: { node: LayoutNode; utility: number }) {
  return (
    <NodePulse
      node={node}
      color={utilityFlashColor(utility)}
      duration={0.75}
      radiusMultiplier={1.7}
      maxOpacity={0.42}
    />
  );
}

function NodePulse({
  node,
  color,
  duration,
  delay = 0,
  radiusMultiplier,
  maxOpacity = 0.68
}: {
  node: LayoutNode;
  color: string;
  duration: number;
  delay?: number;
  radiusMultiplier: number;
  maxOpacity?: number;
}) {
  const meshRef = useRef<Group>(null);
  const materialRef = useRef<MeshBasicMaterial>(null);
  const start = useRef<number | null>(null);
  const radius = sphereRadiusForNode(node);
  useFrame(({ clock }) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const age = clock.elapsedTime - start.current - delay;
    const progress = MathUtils.clamp(age / duration, 0, 1);
    if (meshRef.current) {
      meshRef.current.position.copy(targetVectorForNode(node));
      meshRef.current.scale.setScalar(0.7 + progress * 1.8);
    }
    if (materialRef.current) {
      const activeOpacity = Math.sin(progress * Math.PI) * maxOpacity;
      materialRef.current.opacity = age < 0 || age > duration ? 0 : activeOpacity;
    }
  });
  return (
    <group ref={meshRef} frustumCulled={false}>
      <mesh raycast={() => null}>
        <sphereGeometry args={[radius * radiusMultiplier, 28, 18]} />
        <meshBasicMaterial
          ref={materialRef}
          color={color}
          transparent
          opacity={0}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function TranscriptWisp({
  node,
  text,
  source
}: {
  node: LayoutNode;
  text: string;
  source: "node_added" | "rollout_step";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<number | null>(null);
  useFrame(({ clock }) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const age = clock.elapsedTime - start.current;
    const progress = MathUtils.clamp(age / 2.2, 0, 1);
    if (ref.current) {
      ref.current.style.opacity = String(Math.sin(progress * Math.PI) * (source === "node_added" ? 0.9 : 0.62));
      ref.current.style.transform = `translateY(${-18 * progress}px) scale(${0.96 + progress * 0.04})`;
    }
  });
  return (
    <Html center position={[node.x + 0.6, node.y + 1.25, node.z]} zIndexRange={[15, 0]}>
      <div ref={ref} className={`transcript-wisp ${source}`}>
        {text}
      </div>
    </Html>
  );
}

function BranchInspector({
  selected,
  branchNodes,
  explorationSummary,
  rolloutReflection,
  coalescence,
  nodes,
  candidatePoolsByNode,
  candidateRejectionsByNode,
  candidateOpeningsByCandidate,
  cognitiveStatesByNode,
  selectedCandidateId,
  abstractStats,
  onSelectCandidate
}: {
  selected: GraphNode | null;
  branchNodes: GraphNode[];
  explorationSummary: ExplorationSummary | null;
  rolloutReflection: RolloutReflection | null;
  coalescence: CoalescenceState;
  nodes: GraphNode[];
  candidatePoolsByNode: Record<number, CandidateRecord[]>;
  candidateRejectionsByNode: Record<number, CandidateRejection[]>;
  candidateOpeningsByCandidate: Record<string, CandidateOpening>;
  cognitiveStatesByNode: Record<number, Record<string, unknown>>;
  selectedCandidateId: string | null;
  abstractStats: Record<string, AbstractStat>;
  onSelectCandidate: (candidateId: string) => void;
}) {
  const [selectedDreamNodeId, setSelectedDreamNodeId] = useState<string | null>(null);
  const candidatePool = selected ? candidatePoolsByNode[selected.id] ?? [] : [];
  const antechamber = useMemo(
    () =>
      buildAntechamber(
        selected,
        nodes,
        candidatePool,
        candidateOpeningsByCandidate,
        selected ? candidateRejectionsByNode[selected.id] ?? [] : []
      ),
    [candidateOpeningsByCandidate, candidatePool, candidateRejectionsByNode, nodes, selected]
  );
  const dream = useMemo(
    () => normalizeDreamGraph(selected ? cognitiveStatesByNode[selected.id] : undefined),
    [cognitiveStatesByNode, selected]
  );
  const dreamTargetedCandidates = useMemo(
    () => candidatesTargeting(candidatePool, selectedDreamNodeId),
    [candidatePool, selectedDreamNodeId]
  );
  const selectedCandidate =
    findCandidateById(candidatePool, selectedCandidateId) ??
    (selected?.candidate as CandidateRecord | null | undefined) ??
    null;
  const selectedCandidateModel =
    selectedCandidate ?
      candidateToAntechamberCandidate(selectedCandidate, antechamber, candidateOpeningsByCandidate, nodes)
    : null;
  const prismRows = useMemo(
    () => buildPrismRows(candidatePool, nodes, candidateOpeningsByCandidate, abstractStats),
    [abstractStats, candidateOpeningsByCandidate, candidatePool, nodes]
  );
  return (
    <aside className="branch-inspector" aria-label="Selected route readout">
      <div className="inspector-heading">
        <span>Route readout</span>
        <strong>{selected ? (selected.id === 0 ? "Threshold Chamber" : `Chamber ${selected.id}`) : "No Chamber selected"}</strong>
      </div>
      {!selected ? (
        <p className="empty-detail">Select a Chamber to inspect the route back to the Threshold.</p>
      ) : (
        <>
          <div className="selected-summary">
            <div>
              <span>Door</span>
              <strong>{actionLabel(selected.action_id)}</strong>
            </div>
          </div>
          <PathPanel
            branchNodes={branchNodes}
            selected={selected}
            explorationSummary={explorationSummary}
            rolloutReflection={rolloutReflection}
            coalescence={coalescence}
          />
          <AntechamberPanel
            model={antechamber}
            selectedCandidateId={selectedCandidateId}
            highlightedCandidateIds={dreamTargetedCandidates}
            onSelectCandidate={onSelectCandidate}
          />
          <PassageMicroscope candidate={selectedCandidateModel} />
          <DreamPanel
            dream={dream}
            selectedDreamNodeId={selectedDreamNodeId}
            onSelectDreamNode={setSelectedDreamNodeId}
          />
          <PrismPanel rows={prismRows} />
        </>
      )}
    </aside>
  );
}

function PathPanel({
  branchNodes,
  selected,
  explorationSummary,
  rolloutReflection,
  coalescence
}: {
  branchNodes: GraphNode[];
  selected: GraphNode;
  explorationSummary: ExplorationSummary | null;
  rolloutReflection: RolloutReflection | null;
  coalescence: CoalescenceState;
}) {
  return (
    <InspectorFold title="Path" className="path-panel">
      <div className="branch-list">
        {branchNodes.map((node) => (
          <BranchNodeCard
            key={node.id}
            node={node}
            active={node.id === selected.id}
            showRootSummary={
              node.id === 0 && (selected.id === 0 || coalescence.active || Boolean(coalescence.reflection))
            }
            explorationSummary={explorationSummary}
            rolloutReflection={rolloutReflection}
            coalescence={coalescence}
          />
        ))}
      </div>
    </InspectorFold>
  );
}

function AntechamberPanel({
  model,
  selectedCandidateId,
  highlightedCandidateIds,
  onSelectCandidate
}: {
  model: AntechamberModel;
  selectedCandidateId: string | null;
  highlightedCandidateIds: Set<string>;
  onSelectCandidate: (candidateId: string) => void;
}) {
  const empty = model.total === 0 && model.refused.length === 0;
  return (
    <InspectorFold title="Doors" className="antechamber-panel">
      {empty ? (
        <p className="muted-detail">No Doors have gathered here yet.</p>
      ) : (
        <>
          <div className="aperture-meter">
            <span>{model.openedCount} unlocked / {model.wideningLimit} keys / {model.total} doors</span>
            <em>{model.unopenedCount} doors locked</em>
          </div>
          <CandidateGroup
            title="Unlocked"
            candidates={model.opened}
            selectedCandidateId={selectedCandidateId}
            highlightedCandidateIds={highlightedCandidateIds}
            onSelectCandidate={onSelectCandidate}
          />
          <CandidateGroup
            title="Locked"
            candidates={model.waiting}
            selectedCandidateId={selectedCandidateId}
            highlightedCandidateIds={highlightedCandidateIds}
            onSelectCandidate={onSelectCandidate}
          />
          <RefusedGroup refused={model.refused} />
        </>
      )}
    </InspectorFold>
  );
}

function CandidateGroup({
  title,
  candidates,
  selectedCandidateId,
  highlightedCandidateIds,
  onSelectCandidate
}: {
  title: string;
  candidates: AntechamberCandidate[];
  selectedCandidateId: string | null;
  highlightedCandidateIds: Set<string>;
  onSelectCandidate: (candidateId: string) => void;
}) {
  if (candidates.length === 0) return null;
  return (
    <div className="candidate-group">
      <strong>{title}</strong>
      <div className="candidate-list">
        {candidates.map((candidate) => (
          <button
            key={candidate.candidateId}
            type="button"
            className={`candidate-row ${candidate.candidateId === selectedCandidateId ? "selected" : ""} ${highlightedCandidateIds.has(candidate.candidateId) ? "target-highlight" : ""}`}
            onClick={() => onSelectCandidate(candidate.candidateId)}
          >
            <span>
              {candidate.label}
              {candidate.child && <em>Chamber {candidate.child.id}</em>}
            </span>
            <small>prior {score(candidate.candidate.prior)} · fit {score(candidate.candidate.compatibility_score)}</small>
            <div className="factor-chip-row">
              {candidateFactorChips(candidate.candidate).map((chip) => (
                <i key={chip}>{chip}</i>
              ))}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function RefusedGroup({ refused }: { refused: CandidateRejection[] }) {
  if (refused.length === 0) return null;
  return (
    <div className="candidate-group">
      <strong>Refused</strong>
      <div className="candidate-list">
        {refused.slice(0, 8).map((item, index) => (
          <div className="candidate-row refused" key={`${item.reason ?? "refused"}-${index}`}>
            <span>{actionLabel(String(item.base_action_id ?? "Refused Door"))}</span>
            <small>{String(item.reason ?? "declined by safety or validation")}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function InspectorFold({
  title,
  className = "",
  children
}: {
  title: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={`inspector-fold ${className}`.trim()}>
      <summary>{title}</summary>
      <div className="fold-body">{children}</div>
    </details>
  );
}

function PassageMicroscope({ candidate }: { candidate: AntechamberCandidate | null }) {
  if (!candidate) {
    return (
      <InspectorFold title="Door Mechanism">
        <p className="muted-detail">Select a Door or Chamber to inspect its mechanism.</p>
      </InspectorFold>
    );
  }
  const payload = candidate.candidate;
  return (
    <InspectorFold title="Door Mechanism" className="microscope-panel">
      <div className="microscope-title">
        <strong>{candidate.label}</strong>
        <span>{openingExplanation(candidate)}</span>
      </div>
      <dl className="microscope-grid">
        <MicroscopeField label="Targets" value={stringList(payload.target_ids)} />
        <MicroscopeField label="Hypotheses" value={stringList(payload.failure_hypothesis_ids)} />
        <MicroscopeField label="Diagnostic" value={valueText(payload.diagnostic_operator)} />
        <MicroscopeField label="Repair" value={valueText(payload.repair_operator)} />
        <MicroscopeField label="Form" value={valueText(payload.dialogue_form)} />
        <MicroscopeField label="Affect" value={valueText(payload.affect_strategy)} />
        <MicroscopeField label="Directness" value={score(payload.directness)} />
        <MicroscopeField label="Abstraction" value={score(payload.abstraction)} />
        <MicroscopeField label="Agency" value={valueText(payload.agency_policy)} />
        <MicroscopeField label="Label policy" value={valueText(payload.label_policy)} />
        <MicroscopeField label="Signals" value={stringList(payload.expected_observations)} />
        <MicroscopeField label="Safety" value={stringList(payload.safety_flags)} />
      </dl>
    </InspectorFold>
  );
}

function MicroscopeField({ label, value }: { label: string; value: string | string[] }) {
  const values = Array.isArray(value) ? value : value ? [value] : [];
  if (values.length === 0) return null;
  return (
    <div>
      <dt>{label}</dt>
      <dd>{values.slice(0, 4).join(", ")}</dd>
    </div>
  );
}

function DreamPanel({
  dream,
  selectedDreamNodeId,
  onSelectDreamNode
}: {
  dream: DreamGraph;
  selectedDreamNodeId: string | null;
  onSelectDreamNode: (nodeId: string | null) => void;
}) {
  return (
    <InspectorFold title="Floorplan" className="dream-panel">
      {dream.nodes.length === 0 ? (
        <p className="muted-detail">The Floorplan has not yet been drawn.</p>
      ) : (
        <>
          <svg viewBox="0 0 100 100" role="img" aria-label="Belief graph floorplan">
            {dream.edges.map((edge, index) => (
              <DreamEdgeLine key={`${edge.source}-${edge.target}-${index}`} edge={edge} dream={dream} />
            ))}
            {dream.nodes.map((node) => (
              <DreamNodeButton
                key={node.id}
                node={node}
                selected={node.id === selectedDreamNodeId}
                onClick={() => onSelectDreamNode(node.id === selectedDreamNodeId ? null : node.id)}
              />
            ))}
          </svg>
          <DreamLegend nodes={dream.nodes} selectedDreamNodeId={selectedDreamNodeId} />
        </>
      )}
    </InspectorFold>
  );
}

function DreamEdgeLine({ edge, dream }: { edge: { source: string; target: string; type: string }; dream: DreamGraph }) {
  const source = dream.nodes.find((node) => node.id === edge.source);
  const target = dream.nodes.find((node) => node.id === edge.target);
  if (!source || !target) return null;
  return <line x1={source.x} y1={source.y} x2={target.x} y2={target.y} className="dream-edge" />;
}

function DreamNodeButton({
  node,
  selected,
  onClick
}: {
  node: DreamNode;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <g
      className={`dream-node ${selected ? "selected" : ""} ${node.type}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") onClick();
      }}
    >
      <circle cx={node.x} cy={node.y} r={selected ? 4.8 : 3.8} />
      <title>{node.type}: {node.text}</title>
    </g>
  );
}

function DreamLegend({
  nodes,
  selectedDreamNodeId
}: {
  nodes: DreamNode[];
  selectedDreamNodeId: string | null;
}) {
  const selected = nodes.find((node) => node.id === selectedDreamNodeId);
  if (selected) {
    return (
      <p className="dream-caption">
        <strong>{selected.type}</strong> {selected.text}
      </p>
    );
  }
  return <p className="dream-caption">Select a Floorplan point to reveal which Doors target it.</p>;
}

function PrismPanel({ rows }: { rows: PrismRow[] }) {
  return (
    <InspectorFold title="Lockwork" className="prism-panel">
      {rows.length === 0 ? (
        <p className="muted-detail">The Lockwork has not assembled yet.</p>
      ) : (
        <div className="lockwork-cards">
          {rows.map((row) => (
            <article className="lockwork-card" key={`${row.category}-${row.factor}`}>
              <div>
                <span>{row.category}</span>
                <strong>{row.factor}</strong>
              </div>
              <small>Assembled {row.assembled} · Unlocked {row.opened} · Consulted {row.imagined}</small>
              <em className={`lantern-badge ${row.mirror.toLowerCase()}`}>Lantern {row.mirror}</em>
            </article>
          ))}
        </div>
      )}
    </InspectorFold>
  );
}

function BranchNodeCard({
  node,
  active,
  showRootSummary,
  explorationSummary,
  rolloutReflection,
  coalescence
}: {
  node: GraphNode;
  active: boolean;
  showRootSummary: boolean;
  explorationSummary: ExplorationSummary | null;
  rolloutReflection: RolloutReflection | null;
  coalescence: CoalescenceState;
}) {
  return (
    <article className={`branch-node ${active ? "active" : ""}`}>
      <div className="branch-node-header">
        <span>{node.id === 0 ? "Threshold" : "Chamber"}</span>
        <strong>{node.id === 0 ? "Threshold Chamber" : `Chamber ${node.id}`}</strong>
      </div>
      <div className="branch-node-action">{actionLabel(node.action_id)}</div>
      <div className="branch-metrics">
        <span>Imagined {node.visits} times.</span>
        <span>Lantern is {valueLabel(node.value)}.</span>
        <span>{node.evaluations?.length ?? 0} omens observed.</span>
        {typeof node.unexpanded_candidate_count === "number" && node.unexpanded_candidate_count > 0 && (
          <span>{node.unexpanded_candidate_count} doors locked.</span>
        )}
      </div>
      {showRootSummary ? (
        <RootSummary summary={explorationSummary} reflection={rolloutReflection} coalescence={coalescence} />
      ) : node.id === 0 ? (
        null
      ) : (
        <div className="branch-transition">
          <div className="detail-subtitle">Passage into this Chamber</div>
          {node.p1_utterance && <p className="imagined-utterance"><strong>Elenchus:</strong> {node.p1_utterance}</p>}
          {node.p2_reply && <p className="imagined-utterance"><strong>You:</strong> {node.p2_reply}</p>}
          {!node.p1_utterance && !node.p2_reply && <p className="muted-detail">No imagined utterance stored yet.</p>}
        </div>
      )}
    </article>
  );
}

function RootSummary({
  summary,
  reflection,
  coalescence
}: {
  summary: ExplorationSummary | null;
  reflection: RolloutReflection | null;
  coalescence: CoalescenceState;
}) {
  if (coalescence.active || coalescence.reflection) {
    return (
      <InspectorFold title="Cartography" className="coalescence-panel">
        <div className="coalescence-meter" aria-label="Cartography progress">
          <span className={coalescence.complete ? "complete" : "pending"} />
        </div>
        {coalescence.reflection ? (
          <div className="coalescence-body">
            <p>{coalescence.reflection.summary}</p>
            <ReflectionList title="Locked Doors" items={coalescence.reflection.deferred_passages} />
            <ReflectionList title="Refused Doors" items={coalescence.reflection.refused_passages} />
            <ReflectionList title="Lockwork insights" items={coalescence.reflection.factor_insights} />
          </div>
        ) : (
          <p className="muted-detail">Mapping the first route into Cartography.</p>
        )}
      </InspectorFold>
    );
  }
  if (!summary && !reflection) {
    return <p>Exploration summary pending.</p>;
  }
  if (reflection) {
    return (
      <InspectorFold title="Cartography" className="coalescence-panel">
        <div className="coalescence-body">
          <p>{reflection.summary}</p>
          <ReflectionList title="Locked Doors" items={reflection.deferred_passages} />
          <ReflectionList title="Refused Doors" items={reflection.refused_passages} />
          <ReflectionList title="Lockwork insights" items={reflection.factor_insights} />
        </div>
      </InspectorFold>
    );
  }
  if (!summary) return null;
  return (
    <div className="root-summary">
      <p>{summary.summary}</p>
      <SummaryList title="Themes" items={summary.themes} />
      <SummaryList title="Interesting passages" items={summary.interesting_paths} />
      <SummaryList title="Potential conflicts" items={summary.potential_conflicts} />
    </div>
  );
}

function SummaryList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="summary-list">
      <strong>{title}</strong>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function ReflectionList({ title, items }: { title: string; items: ReflectionItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="summary-list">
      <strong>{title}</strong>
      <ul>
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>{reflectionItemText(item)}</li>
        ))}
      </ul>
    </div>
  );
}

function EdgeLine({
  source,
  target,
  traversalActive,
  selectedBranchActive,
  mctsChosenActive,
  heat
}: {
  source: LayoutNode;
  target: LayoutNode;
  traversalActive: boolean;
  selectedBranchActive: boolean;
  mctsChosenActive: boolean;
  heat: number;
}) {
  const points = useRef(new Float32Array([source.x, source.y, source.z, source.x, source.y, source.z]));
  const positionAttribute = useRef<BufferAttribute | null>(null);
  const color = mctsChosenActive ? "#6ee7ff" : selectedBranchActive ? "#ff3b6b" : traversalActive ? "#8a3a51" : "#4f3340";
  const opacity = mctsChosenActive ? 0.98 : selectedBranchActive ? 0.92 : traversalActive ? 0.62 : 0.24 + heat * 0.42;
  useFrame((_, delta) => {
    const coordinates = points.current;
    const alpha = 1 - Math.exp(-delta * 8);
    coordinates[0] = MathUtils.lerp(coordinates[0], source.x, alpha);
    coordinates[1] = MathUtils.lerp(coordinates[1], source.y, alpha);
    coordinates[2] = MathUtils.lerp(coordinates[2], source.z, alpha);
    coordinates[3] = MathUtils.lerp(coordinates[3], target.x, alpha);
    coordinates[4] = MathUtils.lerp(coordinates[4], target.y, alpha);
    coordinates[5] = MathUtils.lerp(coordinates[5], target.z, alpha);
    if (positionAttribute.current) positionAttribute.current.needsUpdate = true;
  });
  return (
    <lineSegments frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute ref={positionAttribute} attach="attributes-position" args={[points.current, 3]} />
      </bufferGeometry>
      <lineBasicMaterial
        color={color}
        linewidth={mctsChosenActive ? 4 : selectedBranchActive ? 3 : traversalActive ? 2 : 1}
        transparent
        opacity={opacity}
        depthWrite={false}
      />
    </lineSegments>
  );
}

function SingularitySnapshot({
  singularity,
  nodes,
  edges,
  nodeMap
}: {
  singularity: SingularityState;
  nodes: LayoutNode[];
  edges: GraphEdge[];
  nodeMap: Map<number, LayoutNode>;
}) {
  const groupRef = useRef<Group>(null);
  const pointMaterialRef = useRef<MeshBasicMaterial>(null);
  const start = useRef<number | null>(null);
  useFrame(({ clock }, delta) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const age = clock.elapsedTime - start.current;
    const progress = MathUtils.clamp(age / 1.25, 0, 1);
    const eased = progress * progress * (3 - 2 * progress);
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * (5.5 + progress * 12);
      groupRef.current.rotation.x += delta * (1.5 + progress * 3);
      groupRef.current.scale.setScalar(Math.max(0.025, 1 - eased));
    }
    if (pointMaterialRef.current) {
      pointMaterialRef.current.opacity = Math.sin(progress * Math.PI) * 0.95;
    }
  });
  return (
    <group key={singularity.id} frustumCulled={false}>
      <group ref={groupRef} frustumCulled={false}>
        {edges.map((edge) => {
          const source = nodeMap.get(edge.source);
          const target = nodeMap.get(edge.target);
          if (!source || !target) return null;
          return <SnapshotEdge key={edge.id} source={source} target={target} />;
        })}
        {nodes.map((node) => (
          <group key={node.id} position={[node.x, node.y, node.z]}>
            <mesh raycast={() => null}>
              <sphereGeometry args={[sphereRadiusForNode(node) * 0.72, 18, 12]} />
              <meshBasicMaterial
                color={node.value >= 0.4 ? "#6ee7ff" : "#ff3b6b"}
                transparent
                opacity={0.64}
                depthWrite={false}
                blending={AdditiveBlending}
              />
            </mesh>
            <SingularityLabel text={actionLabel(node.action_id)} />
          </group>
        ))}
      </group>
      <mesh raycast={() => null}>
        <sphereGeometry args={[1.1, 32, 20]} />
        <meshBasicMaterial
          ref={pointMaterialRef}
          color="#fff1f5"
          transparent
          opacity={0}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

function SnapshotEdge({ source, target }: { source: LayoutNode; target: LayoutNode }) {
  const points = useMemo(
    () => new Float32Array([source.x, source.y, source.z, target.x, target.y, target.z]),
    [source.x, source.y, source.z, target.x, target.y, target.z]
  );
  return (
    <lineSegments frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[points, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color="#ff3b6b" transparent opacity={0.58} depthWrite={false} />
    </lineSegments>
  );
}

function SingularityLabel({ text }: { text: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<number | null>(null);
  useFrame(({ clock }) => {
    if (start.current === null) start.current = clock.elapsedTime;
    const progress = MathUtils.clamp((clock.elapsedTime - start.current) / 1.25, 0, 1);
    if (ref.current) ref.current.style.opacity = String(Math.max(0, 0.72 * (1 - progress * 1.5)));
  });
  return (
    <Html center position={[0, -0.95, 0]} zIndexRange={[12, 0]}>
      <div ref={ref} className="singularity-label">{text}</div>
    </Html>
  );
}

function AnimatedCamera({
  nodes,
  nodeMap,
  selectedNodeId,
  planning,
  singularityActive,
  rootOrbiting,
  resetSignal,
  interruptSignal
}: {
  nodes: LayoutNode[];
  nodeMap: Map<number, LayoutNode>;
  selectedNodeId: number | null;
  planning: boolean;
  singularityActive: boolean;
  rootOrbiting: boolean;
  resetSignal: number;
  interruptSignal: number;
}) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => (state as unknown as { controls?: unknown }).controls) as
    | { target: Vector3; update: () => void }
    | undefined;
  const bounds = useMemo(() => graphBounds(nodes), [nodes]);
  const previousSelectedNodeId = useRef<number | null>(null);
  const previousResetSignal = useRef(resetSignal);
  const previousInterruptSignal = useRef(interruptSignal);
  const transientFocusUntil = useRef(0);
  const transientFocusMode = useRef<"selected" | "root">("selected");
  useFrame(({ clock }, delta) => {
    if (!controls) return;
    const now = clock.elapsedTime;
    if (previousSelectedNodeId.current !== selectedNodeId) {
      previousSelectedNodeId.current = selectedNodeId;
      transientFocusUntil.current = now + 1.9;
      transientFocusMode.current = "selected";
    }
    if (previousResetSignal.current !== resetSignal) {
      previousResetSignal.current = resetSignal;
      transientFocusUntil.current = now + 2.2;
      transientFocusMode.current = "root";
    }
    if (previousInterruptSignal.current !== interruptSignal) {
      previousInterruptSignal.current = interruptSignal;
      if (!planning) transientFocusUntil.current = 0;
    }

    const shouldAutoFrame = planning || rootOrbiting || singularityActive || now < transientFocusUntil.current;
    if (!shouldAutoFrame) return;

    const target = cameraTarget({
      bounds,
      nodeMap,
      selectedNodeId,
      mode: rootOrbiting || singularityActive ? "root" : transientFocusMode.current,
      singularityActive
    });
    controls.target.lerp(target.center, 1 - Math.exp(-delta * target.targetEase));
    const offset = camera.position.clone().sub(controls.target);
    if (offset.lengthSq() < 0.001) offset.set(0, 0, 1);
    const distance = MathUtils.damp(offset.length(), target.distance, target.zoomEase, delta);
    offset.setLength(distance);
    camera.position.copy(controls.target).add(offset);
    camera.updateProjectionMatrix();
    controls.update();
  });
  return null;
}

function cameraTarget({
  bounds,
  nodeMap,
  selectedNodeId,
  mode,
  singularityActive = false
}: {
  bounds: GraphBounds;
  nodeMap: Map<number, LayoutNode>;
  selectedNodeId: number | null;
  mode: "selected" | "root";
  singularityActive?: boolean;
}) {
  if (singularityActive) {
    return {
      center: nodeMap.get(0) ? targetVectorForNode(nodeMap.get(0)!) : new Vector3(0, 0, 0),
      distance: 9,
      targetEase: 4.5,
      zoomEase: 4.2
    };
  }
  if (mode === "root") {
    return {
      center: nodeMap.get(0) ? targetVectorForNode(nodeMap.get(0)!) : bounds.center,
      distance: bounds.rootDistance,
      targetEase: 1.8,
      zoomEase: 1.15
    };
  }
  const focus = upstreamFocusNode(selectedNodeId, nodeMap);
  return {
    center: focus ? targetVectorForNode(focus) : bounds.center,
    distance: bounds.distance,
    targetEase: 2.15,
    zoomEase: 1.1
  };
}

function upstreamFocusNode(selectedNodeId: number | null, nodeMap: Map<number, LayoutNode>) {
  const selected = selectedNodeId === null ? null : nodeMap.get(selectedNodeId);
  if (!selected) return nodeMap.get(0) ?? null;
  if (typeof selected.parent_id === "number") return nodeMap.get(selected.parent_id) ?? selected;
  return selected;
}

function spawnPositionForNode(
  node: LayoutNode,
  nodeMap: Map<number, LayoutNode>
): [number, number, number] {
  const parent = typeof node.parent_id === "number" ? nodeMap.get(node.parent_id) : null;
  if (parent) return [parent.x, parent.y, parent.z];
  return [node.x, node.y, node.z];
}

function targetVectorForNode(node: LayoutNode): Vector3 {
  return new Vector3(node.x, node.y, node.z);
}

function findCandidateById(
  candidates: CandidateRecord[],
  selectedCandidateId: string | null
): CandidateRecord | null {
  if (!selectedCandidateId) return null;
  return candidates.find((candidate) => candidateId(candidate) === selectedCandidateId) ?? null;
}

function candidateToAntechamberCandidate(
  candidate: CandidateRecord,
  model: AntechamberModel,
  openingsByCandidate: Record<string, CandidateOpening>,
  nodes: GraphNode[]
): AntechamberCandidate {
  const id = candidateId(candidate);
  const existing = [...model.opened, ...model.waiting].find((item) => item.candidateId === id);
  if (existing) return existing;
  return {
    candidate,
    candidateId: id,
    label: candidateLabel(candidate),
    rank: 0,
    opening: openingsByCandidate[id] ?? null,
    child: nodes.find((node) => node.candidate_id === id) ?? null
  };
}

function score(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";
}

function valueText(value: unknown): string {
  return typeof value === "string" && value.trim().length > 0 ? actionLabel(value) : "";
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => (typeof item === "string" ? item.trim() : String(item ?? "").trim()))
    .filter(Boolean)
    .slice(0, 6);
}

function reflectionItemText(item: ReflectionItem): string {
  if (typeof item === "string") return item;
  const preferred = [
    "summary",
    "note",
    "passage",
    "hypothesis",
    "question",
    "risk",
    "rationale"
  ];
  for (const key of preferred) {
    const value = item[key];
    if (typeof value === "string" && value.trim()) {
      if (key === "passage") return `Door: ${actionLabel(value)}`;
      return value.trim();
    }
  }
  return Object.entries(item)
    .slice(0, 3)
    .map(([key, value]) => `${actionLabel(key)}: ${String(value)}`)
    .join("; ");
}

type GraphBounds = {
  center: Vector3;
  distance: number;
  rootDistance: number;
};

function graphBounds(nodes: LayoutNode[]): GraphBounds {
  if (nodes.length === 0) return { center: new Vector3(0, 0, 0), distance: 34, rootDistance: 34 };
  const center = nodes.reduce(
    (accumulator, node) => accumulator.add(new Vector3(node.x, node.y, node.z)),
    new Vector3(0, 0, 0)
  ).multiplyScalar(1 / nodes.length);
  const root = nodes.find((node) => node.id === 0) ?? null;
  const rootCenter = root ? targetVectorForNode(root) : center;
  const radius = nodes.reduce((maxRadius, node) => {
    const distance = center.distanceTo(new Vector3(node.x, node.y, node.z));
    return Math.max(maxRadius, distance);
  }, 0);
  const rootRadius = nodes.reduce((maxRadius, node) => {
    const distance = rootCenter.distanceTo(new Vector3(node.x, node.y, node.z));
    return Math.max(maxRadius, distance);
  }, 0);
  return {
    center,
    distance: MathUtils.clamp(18 + radius * 2.35 + Math.sqrt(nodes.length) * 1.45, 26, 72),
    rootDistance: MathUtils.clamp(20 + rootRadius * 2.55 + Math.sqrt(nodes.length) * 1.65, 30, 90)
  };
}

function layoutGraph(nodes: GraphNode[], edges: GraphEdge[]) {
  const layoutNodes = nodes.map((node) => ({
    ...node,
    x: Math.sin(node.id * 1.21) * 3.2,
    y: -node.depth * 5.2 + Math.sin(node.id) * 1.2,
    z: Math.cos(node.id * 1.7) * 3
  }));
  const links = edges.map((edge) => ({ source: edge.source, target: edge.target }));
  const simulation = forceSimulation(layoutNodes, 3)
    .force("charge", forceManyBody().strength(-42))
    .force("link", forceLink(links).id((node: LayoutNode) => node.id).distance(4.4).strength(0.8))
    .force("center", forceCenter(0, 0, 0))
    .force("z", forceZ(0).strength(0.04))
    .stop();
  for (let i = 0; i < 80; i += 1) simulation.tick();
  const nodeMap = new Map(layoutNodes.map((node) => [node.id, node]));
  const edgeMap = new Map(
    edges.flatMap((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      return source && target ? [[edge.id, { source, target }] as const] : [];
    })
  );
  return { nodes: layoutNodes, edges, nodeMap, edgeMap };
}

function colorForNode(node: GraphNode, selected: boolean): string {
  if (selected) return "#ff3b6b";
  if (node.status === "evaluated") return valueColor(node.value);
  if (node.status === "expanded") return "#6ee7ff";
  if (node.status === "selected") return "#d7264f";
  return "#b83256";
}

function valueColor(value: number): string {
  if (value >= 0.5) return "#6ee7ff";
  if (value >= 0) return "#d7264f";
  return "#ff6b8c";
}

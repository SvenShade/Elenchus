import { Eye } from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import {
  cancelSessionBeacon,
  createSession,
  eventsUrl,
  getSession,
  planP1,
  sendP2,
  type SessionSummary,
  type TranscriptTurn
} from "./api";
import { GraphView } from "./GraphView";
import { actionLabel } from "./graphMetrics";
import { graphReducer, initialGraphState, type MCTSEvent } from "./graphState";

export function App() {
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, dispatch] = useReducer(graphReducer, initialGraphState);
  const [simulations, setSimulations] = useState(8);
  const [maxRolloutDepth, setMaxRolloutDepth] = useState(2);
  const [graphDetail, setGraphDetail] = useState(100);
  const [reflexion, setReflexion] = useState(false);
  const [p2Draft, setP2Draft] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    createSession()
      .then(async ({ session_id }) => {
        if (cancelled) return;
        setSessionId(session_id);
        sessionIdRef.current = session_id;
        const summary = await getSession(session_id);
        if (cancelled) return;
        setSession(summary);
        setSimulations(summary.slider.default);
        setMaxRolloutDepth(summary.depth_slider.default);
        setReflexion(summary.reflexion.default);
        dispatch({ type: "transcript_updated", transcript: summary.transcript });
      })
      .catch((error) => {
        dispatch({ type: "planning_failed", error: error.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    const cancelOnClose = () => {
      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) cancelSessionBeacon(currentSessionId);
    };
    window.addEventListener("pagehide", cancelOnClose);
    window.addEventListener("beforeunload", cancelOnClose);
    return () => {
      window.removeEventListener("pagehide", cancelOnClose);
      window.removeEventListener("beforeunload", cancelOnClose);
    };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let active = true;
    const socket = new WebSocket(eventsUrl(sessionId));
    socket.onmessage = (message) => {
      if (!active) return;
      const event = JSON.parse(message.data) as MCTSEvent;
      dispatch(event);
    };
    socket.onerror = () => {
      if (active) dispatch({ type: "planning_failed", error: "WebSocket connection failed" });
    };
    return () => {
      active = false;
      socket.close();
    };
  }, [sessionId]);

  const transcript = state.transcript.length > 0 ? state.transcript : session?.transcript ?? [];
  const nodes = useMemo(() => Object.values(state.nodes), [state.nodes]);
  const edges = useMemo(() => Object.values(state.edges), [state.edges]);
  const activeSelectedNodeId = useMemo(() => {
    if (selectedNodeId !== null && nodes.some((node) => node.id === selectedNodeId)) {
      return selectedNodeId;
    }
    return state.selectedNodeId;
  }, [nodes, selectedNodeId, state.selectedNodeId]);
  const slider = session?.slider ?? { min: 1, max: 128, default: 8 };
  const depthSlider = session?.depth_slider ?? { min: 1, max: 5, default: 2 };
  const reflexionControl = session?.reflexion ?? { available: false, default: false };
  const pendingP2Turn = transcript.at(-1)?.speaker === "P2";
  const canUseComposer = Boolean(sessionId) && !state.planning && (p2Draft.trim().length > 0 || pendingP2Turn);

  const handleComposerSubmit = useCallback(async () => {
    if (!sessionId || state.planning || (p2Draft.trim().length === 0 && !pendingP2Turn)) return;
    const content = p2Draft.trim();
    setP2Draft("");
    try {
      if (content.length > 0) {
        const summary = await sendP2(sessionId, content);
        setSession(summary);
      }
      await planP1(sessionId, simulations, maxRolloutDepth, reflexion && reflexionControl.available);
    } catch (error) {
      dispatch({
        type: "planning_failed",
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }, [
    sessionId,
    p2Draft,
    pendingP2Turn,
    simulations,
    maxRolloutDepth,
    reflexion,
    reflexionControl.available,
    state.planning
  ]);

  return (
    <main className="app-shell">
      <section className="chat-pane" aria-label="Conversation">
        <header className="topbar">
          <div>
            <h1>{session?.config.name ?? "llm-mcts"}</h1>
            <p>{session?.config.description ?? "Loading cooperative planning session"}</p>
          </div>
        </header>

        <div className="panel-stack">
          <details className="fold-panel algorithm-panel" open>
            <summary>Purpose</summary>
            <div className="fold-body">
              <p>
                Underneath every conversation lies an unspoken labyrinth. Elenchus enters these
                labyrinths as a lantern, testing doors, gathering keys, and following passages into
                chambers of simulated dialogue. It observes where candor sharpens or retreats, and
                returns with words carefully chosen.
              </p>
            </div>
          </details>

          <details className="fold-panel controls-panel">
            <summary>Controls</summary>
            <div className="fold-body">
              <div className="slider-stack">
                <label className="budget-control">
                  <span>Compute</span>
                  <input
                    aria-label="Compute per-move simulations"
                    type="range"
                    min={slider.min}
                    max={slider.max}
                    value={simulations}
                    onChange={(event) => setSimulations(Number(event.target.value))}
                  />
                  <strong>{simulations}</strong>
                </label>
                <label className="budget-control">
                  <span>MCTS depth</span>
                  <input
                    aria-label="MCTS rollout depth"
                    type="range"
                    min={depthSlider.min}
                    max={depthSlider.max}
                    value={maxRolloutDepth}
                    onChange={(event) => setMaxRolloutDepth(Number(event.target.value))}
                  />
                  <strong>{maxRolloutDepth}</strong>
                </label>
                <label className="budget-control">
                  <span>Graph detail</span>
                  <input
                    aria-label="Graph detail"
                    type="range"
                    min={0}
                    max={100}
                    value={graphDetail}
                    onChange={(event) => setGraphDetail(Number(event.target.value))}
                  />
                  <strong>{graphDetail}%</strong>
                </label>
                <label className={`toggle-control ${reflexionControl.available ? "" : "disabled"}`}>
                  <span>Reflexion</span>
                  <input
                    aria-label="Reflexion"
                    type="checkbox"
                    checked={reflexion && reflexionControl.available}
                    disabled={!reflexionControl.available || state.planning}
                    onChange={(event) => setReflexion(event.target.checked)}
                  />
                  <strong>{reflexionControl.available ? (reflexion ? "on" : "off") : "n/a"}</strong>
                </label>
              </div>
            </div>
          </details>
        </div>

        <div className="status-strip">
          <Eye size={18} />
          <span>{displayStatus(state.status)}</span>
          {state.error && <strong>{state.error}</strong>}
        </div>

        <div className="messages">
          {transcript.map((turn, index) => (
            <MessageBubble key={`${turn.speaker}-${index}`} turn={turn} />
          ))}
        </div>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void handleComposerSubmit();
          }}
        >
          <input
            aria-label="You manual message"
            value={p2Draft}
            onChange={(event) => setP2Draft(event.target.value)}
            placeholder="..."
            disabled={state.planning}
          />
          <button
            type="submit"
            disabled={!canUseComposer}
            title={p2Draft.trim().length > 0 ? "Send message and summon Elenchus" : "Summon Elenchus"}
            aria-label={p2Draft.trim().length > 0 ? "Send message and summon Elenchus" : "Summon Elenchus"}
          >
            <span className="psi-icon" aria-hidden="true">Ψ</span>
          </button>
        </form>
      </section>

      <section className="visual-pane">
        <GraphView
          nodes={nodes}
          edges={edges}
          selectedNodeId={activeSelectedNodeId}
          chosenNodeId={state.chosenNodeId}
          chosenPath={state.chosenPath}
          graphDetail={graphDetail}
          planning={state.planning}
          explorationSummary={state.explorationSummary}
          rolloutReflection={state.rolloutReflection}
          coalescence={state.coalescence}
          singularity={state.singularity}
          visualEffects={state.visualEffects}
          candidatePoolsByNode={state.candidatePoolsByNode}
          candidateRejectionsByNode={state.candidateRejectionsByNode}
          candidateOpeningsByCandidate={state.candidateOpeningsByCandidate}
          cognitiveStatesByNode={state.cognitiveStatesByNode}
          selectedCandidateId={state.selectedCandidateId}
          abstractStats={state.abstractStats}
          onSelectNode={(nodeId) => {
            setSelectedNodeId(nodeId);
            dispatch({ type: "scene_selected", node_id: nodeId });
          }}
          onSelectCandidate={(candidateId) => {
            dispatch({ type: "candidate_selected", candidate_id: candidateId });
          }}
        />
      </section>
    </main>
  );
}

function displayStatus(status: string): string {
  if (status.startsWith("Planning ")) return "Entering the labyrinth";
  if (status.startsWith("Found ") && status.endsWith(" Doors")) return status;
  const mapped: Record<string, string> = {
    "Session ready": "Elenchus is listening",
    "Expanding search tree": "Revealing Chambers",
    "Expanding legal moves": "Unlocking Doors",
    "Opening new passages": "Unlocking Doors",
    "Opening new passage": "Unlocking a Door",
    "Unlocking a Door": "Unlocking a Door",
    "Reading the belief graph": "Reading the Floorplan",
    "Gathering possible scenes": "Finding Doors",
    "Finding Doors": "Finding Doors",
    "Widening the inner theatre": "Adding Keys to the Keyring",
    "Discarding unsafe passage": "Refusing a Door",
    "Traversing promising branch": "Tracing Passages",
    "Judging rollout": "Reading Omens",
    "Backing up utility": "Carrying Omens to the Threshold",
    "Reflecting on rollouts": "Drawing Cartography",
    "Rollout reflection ready": "Cartography drawn",
    "Synthesizing reflected P1 move": "Summoning the final utterance",
    "Reflected P1 move ready": "Thought materialised",
    "Reflexion pass complete": "Cartography sealed",
    "Building reflected tree": "Re-entering the labyrinth",
    "Cancelling thought": "Leaving the labyrinth",
    "Thought dissolved": "Thought dissolved",
    "Exploration summary ready": "Cartography drawn",
    "P1 message appended": "Thought materialised",
    "P1 move selected": "Thought materialised"
  };
  return mapped[status] ?? status;
}

function MessageBubble({ turn }: { turn: TranscriptTurn }) {
  const speaker = turn.speaker === "P1" ? "Elenchus" : "You";
  return (
    <article className={`message ${turn.speaker === "P1" ? "p1" : "p2"}`}>
      <div className="message-meta">
        <span>{speaker}</span>
        {turn.action_id && <span>{actionLabel(turn.action_id)}</span>}
      </div>
      <p>{turn.content}</p>
    </article>
  );
}

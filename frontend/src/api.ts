export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8787";

export type TranscriptTurn = {
  speaker: string;
  content: string;
  action_id?: string | null;
  imagined?: boolean;
};

export type SessionSummary = {
  id: string;
  planning: boolean;
  current_run_id: string | null;
  transcript: TranscriptTurn[];
  config: {
    name: string;
    description: string;
    actions: { id: string; description: string }[];
  };
  slider: {
    min: number;
    max: number;
    default: number;
  };
  depth_slider: {
    min: number;
    max: number;
    default: number;
  };
  labyrinth_depth_slider: {
    min: number;
    max: number;
    default: number;
  };
  lantern_range_slider: {
    min: number;
    max: number;
    default: number;
  };
  reflexion: {
    available: boolean;
    default: boolean;
  };
};

export type SessionCreated = {
  session_id: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return response.json() as Promise<T>;
}

export function createSession(): Promise<SessionCreated> {
  return request<SessionCreated>("/api/sessions", { method: "POST" });
}

export function getSession(sessionId: string): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${sessionId}`);
}

export function planP1(
  sessionId: string,
  simulations: number,
  labyrinthDepth: number,
  lanternRange: number,
  reflexion: boolean
): Promise<{ status: string; run_id: string }> {
  return request(`/api/sessions/${sessionId}/plan`, {
    method: "POST",
    body: JSON.stringify({
      simulations,
      labyrinth_depth: labyrinthDepth,
      lantern_range: lanternRange,
      reflexion
    })
  });
}

export function sendP2(sessionId: string, content: string): Promise<SessionSummary> {
  return request<SessionSummary>(`/api/sessions/${sessionId}/p2`, {
    method: "POST",
    body: JSON.stringify({ content })
  });
}

export function cancelSession(sessionId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/sessions/${sessionId}/cancel`, {
    method: "POST"
  });
}

export function cancelSessionBeacon(sessionId: string): void {
  const url = `${API_BASE}/api/sessions/${sessionId}/cancel`;
  if (navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob(["{}"], { type: "application/json" }));
    return;
  }
  void fetch(url, {
    method: "POST",
    keepalive: true,
    headers: { "Content-Type": "application/json" },
    body: "{}"
  });
}

export function eventsUrl(sessionId: string): string {
  const url = new URL(API_BASE);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `/api/sessions/${sessionId}/events`;
  return url.toString();
}

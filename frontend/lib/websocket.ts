// lib/websocket.ts – WebSocket hook for streaming frames
import { useEffect, useRef, useState } from "react";

export type AttendanceResult = {
  face_detected: boolean;
  bbox: [number, number, number, number] | null;
  liveness: { label: string; score: number; raw_score?: number } | null;
  emotion: { label: string; emoji: string; score: number } | null;
  recognition: { matched: boolean; employee_id: string | null; name: string | null; similarity: number } | null;
  head_pose: { yaw: number; pitch: number; roll: number; angle: string | null } | null;
  attendance_logged?: boolean;
};

export type RegisterResult = {
  face_detected: boolean;
  bbox: [number, number, number, number] | null;
  liveness: { label: string; score: number; raw_score?: number } | null;
  head_pose: { yaw: number; pitch: number; roll: number; angle: string | null } | null;
  face_preview: string | null;
  ready_to_capture: boolean;
  already_registered?: boolean;
  recognition?: { matched: boolean; employee_id: string | null; name: string | null; similarity: number } | null;
};


// 127.0.0.1 instead of localhost: on Windows "localhost" may resolve to ::1 first,
// while uvicorn only listens on IPv4.
export const API_BASE = "http://127.0.0.1:8000";
const WS_BASE = "ws://127.0.0.1:8000";

const RECONNECT_DELAY_MS = 1000;
// If the backend never answers a frame, stop waiting after this long.
const FRAME_TIMEOUT_MS = 3000;

// Time each socket's in-flight frame was sent (absent = idle).
const inFlight = new WeakMap<WebSocket, number>();

export function createWebSocket(path: string): WebSocket {
  return new WebSocket(`${WS_BASE}${path}`);
}

/** True when the socket is open and not still waiting on a previous frame. */
export function canSendFrame(ws: WebSocket | null): ws is WebSocket {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  const sentAt = inFlight.get(ws);
  return sentAt === undefined || performance.now() - sentAt > FRAME_TIMEOUT_MS;
}

/**
 * Send one frame. Only one frame is in flight at a time: on slower machines the
 * backend takes longer than the capture interval, and queuing frames makes the
 * results (and the face box) lag further and further behind the camera.
 */
export function sendFrame(ws: WebSocket, canvas: HTMLCanvasElement, quality = 0.95) {
  if (!canSendFrame(ws)) return;
  const b64 = canvas.toDataURL("image/jpeg", quality);
  inFlight.set(ws, performance.now());
  ws.send(JSON.stringify({ frame: b64 }));
}

/**
 * Open a frame-streaming WebSocket that reconnects automatically (e.g. while the
 * backend is still starting or reloading). Error replies are logged, not passed on.
 */
export function useFrameSocket<T>(path: string, onResult: (data: T) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const onResultRef = useRef(onResult);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    onResultRef.current = onResult;
  });

  useEffect(() => {
    let closed = false;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const ws = createWebSocket(path);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (wsRef.current === ws) wsRef.current = null;
        if (!closed) retryTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      ws.onmessage = (e) => {
        inFlight.delete(ws);
        const data = JSON.parse(e.data);
        if (data.error) {
          console.warn(`[${path}] backend error:`, data.error);
          return;
        }
        onResultRef.current(data as T);
      };
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, [path]);

  return { wsRef, connected };
}

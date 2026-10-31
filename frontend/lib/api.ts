// lib/api.ts – REST calls to the FastAPI backend, plus a small polling hook
import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/websocket";

export type Employee = {
  id: string;
  name: string;
  created_at: string;
  angle_count: number;
};

export type AttendanceRecord = {
  id: number;
  employee_id: string;
  name: string;
  timestamp: string;
  emotion: string | null;
  liveness_score: number | null;
  similarity: number | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getEmployees = () => request<Employee[]>("/api/employees");
export const deleteEmployee = (id: string) =>
  request<{ deleted: string }>(`/api/employees/${encodeURIComponent(id)}`, { method: "DELETE" });
export const getToday = () => request<AttendanceRecord[]>("/api/attendance/today");
export const getHistory = (limit: number) => request<AttendanceRecord[]>(`/api/attendance/history?limit=${limit}`);

export type Resource<T> =
  | { status: "loading"; data: null; error: null }
  | { status: "error"; data: T | null; error: string }
  | { status: "ready"; data: T; error: null };

/**
 * Fetch once, then every `intervalMs` if given. Pass a stable fetcher (a module
 * function or a useCallback), since a new one triggers a fetch. A failed refresh keeps the last
 * good data on screen; only a failure with nothing loaded shows the error state.
 */
export function useResource<T>(fetcher: () => Promise<T>, intervalMs?: number) {
  const [state, setState] = useState<Resource<T>>({ status: "loading", data: null, error: null });
  const fetcherRef = useRef(fetcher);

  useEffect(() => {
    fetcherRef.current = fetcher;
  });

  const load = useCallback(async () => {
    try {
      const data = await fetcherRef.current();
      setState({ status: "ready", data, error: null });
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      setState((prev) => (prev.data !== null
        ? { status: "ready", data: prev.data as T, error: null }
        : { status: "error", data: null, error }));
    }
  }, []);

  const reload = useCallback(() => {
    setState((prev) => (prev.data !== null ? prev : { status: "loading", data: null, error: null }));
    load();
  }, [load]);

  // A new fetcher (e.g. a different history limit) loads again at once.
  useEffect(() => {
    load();
    if (!intervalMs) return;
    const timer = setInterval(load, intervalMs);
    return () => clearInterval(timer);
  }, [load, intervalMs, fetcher]);

  return { ...state, reload };
}

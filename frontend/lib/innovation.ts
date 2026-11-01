// lib/innovation.ts – feature flag and typed client for the innovation API
import { API_BASE } from "@/lib/websocket";

export { INNOVATION } from "@/lib/flags";

export type InnovationConfig = {
  research_mode: boolean;
  challenge: { digits: number; window_s: number; lang: string; max_retries: number };
  asr: { model: string; max_distance: number };
  fusion: { rule: string };
  thresholds: Record<string, number>;
  media: { audio_rate: number; min_fps: number };
};

export type MediaProbe = {
  video: { frames: number; fps: number; width: number; height: number; duration_s: number };
  audio: { sample_rate: number; samples: number; duration_s: number; rms_dbfs: number | null };
  decode_ms: number;
  checks: { fps_ok: boolean; audio_ok: boolean };
  content_type: string | null;
  bytes: number;
};

export type Attempt = {
  id: number;
  challenge_id: string | null;
  candidate_id: string | null;
  decision: "accept" | "reject" | "retry";
  reason_codes: string[];
  s_face: number | null;
  p_live: number | null;
  s_voice: number | null;
  asr_text: string | null;
  asr_dist: number | null;
  s_av: number | null;
  t_response: number | null;
  latency_ms: number | null;
  attack_label: string | null;
  session_tag: string | null;
  created_at: string;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { const b = await res.json(); if (b?.detail) detail = String(b.detail); } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const getInnovationConfig = () => call<InnovationConfig>("/api/innovation/config");
export const getAttempts = (limit = 100) => call<Attempt[]>(`/api/attempts?limit=${limit}`);

export function probeMedia(clip: Blob): Promise<MediaProbe> {
  const form = new FormData();
  const ext = clip.type.includes("mp4") ? "mp4" : "webm";
  form.append("file", clip, `probe.${ext}`);
  return call<MediaProbe>("/api/innovation/media/probe", { method: "POST", body: form });
}

// ── Voice enrolment ─────────────────────────────────────────────────────────

export type VoiceIssue = "TOO_SHORT" | "TOO_QUIET" | "CLIPPING" | "NOISY";

export type VoiceQuality = {
  duration_s: number;
  speech_s: number;
  rms_dbfs: number | null;
  snr_db: number | null;
  clip_fraction: number;
  issues: VoiceIssue[];
};

export type EnrollStart = {
  employee_id: string;
  prompts: { challenge_id: string; digits: string }[];
  samples_needed: number;
  n_samples: number;
};

export type EnrollResult = {
  ok: boolean;
  quality: VoiceQuality;
  n_samples: number;
  enrolled?: boolean;
  embed_ms?: number;
};

export type VoiceStatusAll = { samples_needed: number; counts: Record<string, number> };

/** What to tell the person for each quality issue. */
export const VOICE_ISSUE_TEXT: Record<VoiceIssue, string> = {
  TOO_SHORT: "Less than 1.5 seconds of speech was heard. Read all six digits at a normal pace.",
  TOO_QUIET: "Too quiet. Move closer to the microphone or speak up.",
  CLIPPING: "Too loud, the sound clipped. Move back a little or speak more softly.",
  NOISY: "Too much background noise. Find a quieter spot or move closer to the microphone.",
};

export const startVoiceEnrollment = (employeeId: string, reset = false) =>
  call<EnrollStart>("/api/voice/enroll/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ employee_id: employeeId, reset }),
  });

export function submitVoiceSample(employeeId: string, challengeId: string, clip: Blob, keepAudio = false) {
  const form = new FormData();
  form.append("employee_id", employeeId);
  form.append("challenge_id", challengeId);
  form.append("keep_audio", keepAudio ? "true" : "false");
  form.append("file", clip, "voice.webm");
  return call<EnrollResult>("/api/voice/enroll/submit", { method: "POST", body: form });
}

export const getVoiceStatusAll = () => call<VoiceStatusAll>("/api/voice/status");

// ── Frame capture (research mode) ───────────────────────────────────────────

export function saveFrame(person: string, session: string, index: number, frame: Blob, clientMs: number) {
  const form = new FormData();
  form.append("person", person);
  form.append("session", session);
  form.append("index", String(index));
  form.append("client_ms", String(clientMs));
  form.append("file", frame, `${index}.jpg`);
  return call<{ saved: string }>("/api/innovation/frames", { method: "POST", body: form });
}

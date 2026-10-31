// lib/checks.ts – turn one backend frame result into the rows of the check ladder
import type { AttendanceResult } from "@/lib/websocket";
import { emotionLabel } from "@/lib/emotion";
import { clamp01, formatScore } from "@/lib/format";

export type CheckState = "pass" | "fail" | "wait";

export type Check = {
  key: string;
  name: string;
  state: CheckState;
  detail: string;          // the state in words, never colour alone
  reading?: string;        // what the instrument measured, e.g. "0.93"
  value?: number;          // 0–1, drawn as the meter
  icon?: string;           // emotion label, when the row shows one
};

/** Liveness labels the backend sends, mapped to a check state and plain words. */
export function livenessCheck(label: string | undefined): { state: CheckState; detail: string; face: "real" | "fake" | "pending" } {
  switch (label) {
    case "Real":        return { state: "pass", detail: "Live person", face: "real" };
    case "Fake":        return { state: "fail", detail: "Spoof suspected", face: "fake" };
    case "Checking":    return { state: "wait", detail: "Checking", face: "pending" };
    case "Move closer": return { state: "wait", detail: "Move closer to the camera", face: "pending" };
    case "Move back":   return { state: "wait", detail: "Move back from the camera", face: "pending" };
    default:            return { state: "wait", detail: label || "Waiting for a face", face: "pending" };
  }
}

const AFTER_LIVENESS = "Runs once liveness passes";

export function buildChecks(result: AttendanceResult | null): Check[] {
  const face = !!result?.face_detected;
  const live = result?.liveness ?? null;
  const rec = result?.recognition ?? null;
  const emo = result?.emotion ?? null;
  const l = livenessCheck(face ? live?.label : undefined);

  return [
    {
      key: "face",
      name: "Face found",
      state: face ? "pass" : "wait",
      detail: face ? "One face in view" : "Step in front of the camera",
    },
    {
      key: "liveness",
      name: "Liveness",
      state: face ? l.state : "wait",
      detail: face ? l.detail : "Waiting for a face",
      reading: face && live ? formatScore(live.score) : undefined,
      value: face && live ? clamp01(live.score) : undefined,
    },
    {
      key: "identity",
      name: "Identity",
      state: rec ? (rec.matched ? "pass" : "fail") : "wait",
      detail: rec ? (rec.matched ? "Matches a registered person" : "No registered match") : AFTER_LIVENESS,
      reading: rec ? formatScore(rec.similarity) : undefined,
      value: rec ? clamp01(rec.similarity) : undefined,
    },
    {
      key: "emotion",
      name: "Emotion",
      state: emo ? "pass" : "wait",
      detail: emo ? (emotionLabel(emo.label) ?? emo.label) : AFTER_LIVENESS,
      reading: emo ? formatScore(emo.score) : undefined,
      value: emo ? clamp01(emo.score) : undefined,
      icon: emo?.label,
    },
  ];
}

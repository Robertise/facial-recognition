// lib/format.ts – display formatting shared by the screens

/**
 * SQLite's CURRENT_TIMESTAMP is UTC without a zone ("2026-10-31 08:02:11").
 * Parse it as UTC so the browser shows local time.
 */
export function parseDbTime(ts: string): Date {
  const iso = ts.includes("T") ? ts : ts.replace(" ", "T");
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
}

const timeFmt = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });
const longDateFmt = new Intl.DateTimeFormat("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });

export const formatTime = (d: Date) => timeFmt.format(d);
export const formatDate = (d: Date) => dateFmt.format(d);
export const formatLongDate = (d: Date) => longDateFmt.format(d);

/** A 0–1 score with two decimals, or an empty string when absent. */
export function formatScore(v: number | null | undefined): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "";
}

export function clamp01(v: number | null | undefined): number {
  if (typeof v !== "number" || !Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}

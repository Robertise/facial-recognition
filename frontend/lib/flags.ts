// lib/flags.ts – build-time feature flags, safe to import from server and client components

const ON = new Set(["1", "true", "yes", "on"]);

/**
 * NEXT_PUBLIC_INNOVATION shows the innovation screens. Any of 1/true/yes/on
 * turns it on. It is read when `next dev` or `next build` starts, so restart
 * the dev server after changing frontend/.env.local.
 */
export const INNOVATION = ON.has((process.env.NEXT_PUBLIC_INNOVATION ?? "").trim().toLowerCase());

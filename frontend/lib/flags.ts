// lib/flags.ts – build-time feature flags, safe to import from server and client components

/** NEXT_PUBLIC_INNOVATION=1 shows the innovation screens (inlined at build time). */
export const INNOVATION = process.env.NEXT_PUBLIC_INNOVATION === "1";

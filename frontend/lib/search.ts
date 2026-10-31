// lib/search.ts – filters kept in the URL, and accent-insensitive matching
import { useCallback } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

/** Lowercase and strip diacritics, so "nguyen" finds "Nguyễn". */
export function fold(s: string): string {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();
}

export function matches(query: string, ...fields: (string | null | undefined)[]): boolean {
  const q = fold(query.trim());
  return !q || fields.some((f) => f && fold(f).includes(q));
}

/** Read and write one search param, replacing history so Back leaves the page. */
export function useSearchParam(key: string, fallback = ""): [string, (value: string) => void] {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const value = params.get(key) ?? fallback;
  const set = useCallback((next: string) => {
    const p = new URLSearchParams(params.toString());
    if (next && next !== fallback) p.set(key, next); else p.delete(key);
    const qs = p.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [params, router, pathname, key, fallback]);
  return [value, set];
}

"use client";
import { RotateCcw } from "lucide-react";

/** Skeleton rows in the shape of the table that is loading. */
export function TableSkeleton({ columns, rows = 4 }: { columns: string[]; rows?: number }) {
  return (
    <div className="table-wrap" aria-busy="true">
      <span className="visually-hidden" role="status">Loading</span>
      <table className="data" aria-hidden="true">
        <thead><tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>
          {Array.from({ length: rows }, (_, r) => (
            <tr key={r}>
              {columns.map((c, i) => (
                <td key={c} data-label={c}><span className="skeleton" style={{ "--w": `${40 + ((r * 7 + i * 13) % 45)}%` } as React.CSSProperties} /></td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="state-block">
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export function ErrorState({ what, error, onRetry }: { what: string; error: string; onRetry: () => void }) {
  return (
    <div className="state-block" role="alert">
      <strong>Could not load {what}.</strong>
      <p>The backend at 127.0.0.1:8000 did not answer ({error}). Check that it is running, then try again.</p>
      <button type="button" className="btn" onClick={onRetry}>
        <RotateCcw size={16} aria-hidden="true" /> Try again
      </button>
    </div>
  );
}

"use client";
import { useCallback, useState } from "react";
import Link from "next/link";
import AttendanceTable, { HISTORY_COLUMNS } from "@/components/AttendanceTable";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/DataState";
import { getHistory, useResource } from "@/lib/api";
import { matches, useSearchParam } from "@/lib/search";

const LIMITS = ["50", "100", "500"];

export default function HistoryView() {
  const [urlQ, setUrlQ] = useSearchParam("q");
  const [q, setText] = useState(urlQ);
  const setQ = (v: string) => { setText(v); setUrlQ(v); };
  const [limitParam, setLimit] = useSearchParam("limit", "100");
  const limit = LIMITS.includes(limitParam) ? Number(limitParam) : 100;

  const fetchHistory = useCallback(() => getHistory(limit), [limit]);
  const history = useResource(fetchHistory, 15000);

  const all = history.data ?? [];
  const shown = all.filter((r) => matches(q, r.employee_id, r.name));

  return (
    <>
      <div className="toolbar">
        <div className="field">
          <label htmlFor="history-search">Search by name or ID</label>
          <input id="history-search" name="q" type="search" className="input" value={q}
            onChange={(e) => setQ(e.target.value)} placeholder="nguyen" autoComplete="off" />
        </div>
        <div className="field field--narrow">
          <label htmlFor="history-limit">Show</label>
          <select id="history-limit" name="limit" className="input" value={String(limit)} onChange={(e) => setLimit(e.target.value)}>
            {LIMITS.map((l) => <option key={l} value={l}>Last {l}</option>)}
          </select>
        </div>
        {history.status === "ready" && (
          <p className="helper" role="status">
            {q ? `${shown.length} of ${all.length} records` : `${all.length} ${all.length === 1 ? "record" : "records"}`}
          </p>
        )}
      </div>

      {history.status === "loading" && <TableSkeleton columns={HISTORY_COLUMNS} rows={6} />}
      {history.status === "error" && <ErrorState what="the history" error={history.error} onRetry={history.reload} />}
      {history.status === "ready" && (all.length === 0 ? (
        <EmptyState title="No check-ins recorded yet.">
          <p>Each successful check-in is kept here with its liveness and similarity scores.</p>
          <Link href="/attendance" className="btn">Open check-in</Link>
        </EmptyState>
      ) : shown.length === 0 ? (
        <EmptyState title={`No check-ins match “${q}”.`}>
          <p>Search looks at names and IDs within the records shown.</p>
          <button type="button" className="btn" onClick={() => setQ("")}>Clear search</button>
        </EmptyState>
      ) : (
        <AttendanceTable records={shown} withDate caption="Check-in history" />
      ))}
    </>
  );
}

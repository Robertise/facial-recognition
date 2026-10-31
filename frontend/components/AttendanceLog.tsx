"use client";
import Link from "next/link";
import { getToday, useResource } from "@/lib/api";
import AttendanceTable, { ATTENDANCE_COLUMNS } from "@/components/AttendanceTable";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/DataState";

/** Today's check-ins, refreshed every 5 seconds. */
export default function AttendanceLog() {
  const today = useResource(getToday, 5000);
  const count = today.data?.length ?? 0;

  return (
    <section aria-labelledby="today-heading">
      <div className="section-head">
        <h2 id="today-heading">Check-ins today</h2>
        {today.status === "ready" && count > 0 && (
          <span className="measured" aria-label={`${count} records`}>{count} {count === 1 ? "record" : "records"}</span>
        )}
      </div>
      {today.status === "loading" && <TableSkeleton columns={ATTENDANCE_COLUMNS} />}
      {today.status === "error" && <ErrorState what="today's check-ins" error={today.error} onRetry={today.reload} />}
      {today.status === "ready" && (count === 0 ? (
        <EmptyState title="No one has checked in today.">
          <p>Each check-in appears here as soon as a registered face passes the checks.</p>
          <Link href="/attendance" className="btn">Open check-in</Link>
        </EmptyState>
      ) : (
        <AttendanceTable records={today.data} caption="Check-ins today" />
      ))}
    </section>
  );
}

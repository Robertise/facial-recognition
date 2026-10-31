"use client";
import { useSyncExternalStore } from "react";
import Link from "next/link";
import { Camera, UserPlus } from "lucide-react";
import AttendanceLog from "@/components/AttendanceLog";
import { ErrorState } from "@/components/DataState";
import { getEmployees, getToday, useResource } from "@/lib/api";
import { formatLongDate } from "@/lib/format";

const noSubscribe = () => () => {};

export default function Overview() {
  // The date is the visitor's, so it is read in the browser, never at build time.
  const date = useSyncExternalStore(noSubscribe, () => formatLongDate(new Date()), () => null);

  const today = useResource(getToday, 5000);
  const people = useResource(getEmployees, 30000);
  const checkedIn = today.data ? new Set(today.data.map((r) => r.employee_id)).size : null;
  const registered = people.data ? people.data.length : null;
  const failed = today.status === "error" ? today : people.status === "error" ? people : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Today</h1>
          <p className="lead measured" aria-live="polite">{date ?? " "}</p>
        </div>
        <div className="actions">
          <Link href="/attendance" className="btn btn--primary"><Camera size={16} aria-hidden="true" />Start check-in</Link>
          <Link href="/register" className="btn"><UserPlus size={16} aria-hidden="true" />Register a person</Link>
        </div>
      </div>

      {failed ? (
        <ErrorState what="today's figures" error={failed.error ?? ""} onRetry={() => { today.reload(); people.reload(); }} />
      ) : (
        <dl className="readings">
          <div>
            <dt>People checked in today</dt>
            <dd className="reading">{checkedIn ?? <span className="skeleton" style={{ "--w": "2ch" } as React.CSSProperties} />}</dd>
          </div>
          <div>
            <dt>Registered people</dt>
            <dd className="reading">{registered ?? <span className="skeleton" style={{ "--w": "2ch" } as React.CSSProperties} />}</dd>
          </div>
        </dl>
      )}

      <AttendanceLog />
    </>
  );
}

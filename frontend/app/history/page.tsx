import type { Metadata } from "next";
import { Suspense } from "react";
import { TableSkeleton } from "@/components/DataState";
import { HISTORY_COLUMNS } from "@/components/AttendanceTable";
import HistoryView from "./HistoryView";

export const metadata: Metadata = { title: "History" };

export default function HistoryPage() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p className="lead">Every check-in, newest first, with the scores it passed on.</p>
        </div>
      </div>
      <Suspense fallback={<TableSkeleton columns={HISTORY_COLUMNS} />}>
        <HistoryView />
      </Suspense>
    </>
  );
}

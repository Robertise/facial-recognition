"use client";
import Link from "next/link";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/DataState";
import { useResource } from "@/lib/api";
import { formatDate, formatScore, formatTime, parseDbTime } from "@/lib/format";
import { getAttempts } from "@/lib/innovation";

const fetchAttempts = () => getAttempts(200);
const COLUMNS = ["Time", "Candidate", "Decision", "Reasons", "Face", "Liveness", "Voice", "AV"];

/** Verification attempts, newest first. */
export default function SecurityView() {
  const attempts = useResource(fetchAttempts, 10000);

  if (attempts.status === "loading") return <TableSkeleton columns={COLUMNS} />;
  if (attempts.status === "error") return <ErrorState what="verification attempts" error={attempts.error} onRetry={attempts.reload} />;
  if (attempts.data.length === 0) {
    return (
      <EmptyState title="No verification attempts yet.">
        <p>Secure check-ins and lab runs are listed here once they record an attempt.</p>
        <Link href="/lab" className="btn">Open the lab</Link>
      </EmptyState>
    );
  }

  return (
    <div className="table-wrap">
      <table className="data">
        <caption className="visually-hidden">Verification attempts</caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Candidate</th>
            <th scope="col">Decision</th>
            <th scope="col">Reasons</th>
            <th scope="col" className="num">Face</th>
            <th scope="col" className="num">Liveness</th>
            <th scope="col" className="num">Voice</th>
            <th scope="col" className="num">AV</th>
          </tr>
        </thead>
        <tbody>
          {attempts.data.map((a) => {
            const at = parseDbTime(a.created_at);
            const tone = a.decision === "accept" ? "ok" : a.decision === "reject" ? "bad" : undefined;
            return (
              <tr key={a.id}>
                <td className="measured" data-label="Time"><time dateTime={at.toISOString()}>{formatDate(at)} {formatTime(at)}</time></td>
                <td className="measured measured--cool" data-label="Candidate">{a.candidate_id ?? ""}</td>
                <td data-label="Decision"><span className="chip" data-tone={tone}>{a.decision}</span></td>
                <td data-label="Reasons">{a.reason_codes.join(", ")}</td>
                <td className="num" data-label="Face">{formatScore(a.s_face)}</td>
                <td className="num" data-label="Liveness">{formatScore(a.p_live)}</td>
                <td className="num" data-label="Voice">{formatScore(a.s_voice)}</td>
                <td className="num" data-label="AV">{formatScore(a.s_av)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

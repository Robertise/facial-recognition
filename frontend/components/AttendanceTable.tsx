"use client";
import type { AttendanceRecord } from "@/lib/api";
import { emotionIcon, emotionLabel } from "@/lib/emotion";
import { formatDate, formatScore, formatTime, parseDbTime } from "@/lib/format";

export const ATTENDANCE_COLUMNS = ["Time", "ID", "Name", "Emotion", "Liveness", "Similarity"];
export const HISTORY_COLUMNS = ["Date", ...ATTENDANCE_COLUMNS];

/** Check-in records as a table; `withDate` adds a date column for multi-day views. */
export default function AttendanceTable({ records, withDate = false, caption }: {
  records: AttendanceRecord[];
  withDate?: boolean;
  caption: string;
}) {
  return (
    <div className="table-wrap">
      <table className="data">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            {withDate && <th scope="col">Date</th>}
            <th scope="col">Time</th>
            <th scope="col">ID</th>
            <th scope="col">Name</th>
            <th scope="col">Emotion</th>
            <th scope="col" className="num">Liveness</th>
            <th scope="col" className="num">Similarity</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => {
            const at = parseDbTime(r.timestamp);
            const emotion = emotionLabel(r.emotion);
            const Icon = emotionIcon(r.emotion);
            return (
              <tr key={r.id}>
                {withDate && <td className="measured" data-label="Date">{formatDate(at)}</td>}
                <td className="measured" data-label="Time"><time dateTime={at.toISOString()}>{formatTime(at)}</time></td>
                <td className="measured measured--cool" data-label="ID">{r.employee_id}</td>
                <td data-label="Name">{r.name}</td>
                <td data-label="Emotion">
                  {emotion
                    ? <span className="emotion"><Icon size={16} aria-hidden="true" />{emotion}</span>
                    : <span className="visually-hidden">Not recorded</span>}
                </td>
                <td className="num" data-label="Liveness">{formatScore(r.liveness_score)}</td>
                <td className="num" data-label="Similarity">{formatScore(r.similarity)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

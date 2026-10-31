"use client";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/websocket";
import { emotionEmoji } from "@/lib/emotion";

type Record = {
  id: number;
  employee_id: string;
  name: string;
  timestamp: string;
  emotion: string;
  liveness_score: number;
  similarity: number;
};

export default function AttendanceLog() {
  const [records, setRecords] = useState<Record[]>([]);

  const fetchToday = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/attendance/today`);
      setRecords(await res.json());
    } catch {}
  };

  useEffect(() => {
    fetchToday();
    const t = setInterval(fetchToday, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="w-full mt-6">
      <h2 className="text-lg font-semibold text-white mb-3">📋 Điểm danh hôm nay</h2>
      {records.length === 0 ? (
        <p className="text-gray-400 text-sm">Chưa có ai điểm danh hôm nay.</p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-gray-700">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-800 text-gray-300">
              <tr>
                <th className="px-4 py-2">Mã NV</th>
                <th className="px-4 py-2">Tên</th>
                <th className="px-4 py-2">Thời gian</th>
                <th className="px-4 py-2">Cảm xúc</th>
                <th className="px-4 py-2">Độ tương đồng</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.id} className="border-t border-gray-700 hover:bg-gray-800/50">
                  <td className="px-4 py-2 font-mono text-blue-400">{r.employee_id}</td>
                  <td className="px-4 py-2 text-white">{r.name}</td>
                  <td className="px-4 py-2 text-gray-300">{new Date(r.timestamp).toLocaleTimeString("vi-VN")}</td>
                  <td className="px-4 py-2 text-yellow-300">
                    {emotionEmoji(r.emotion)
                      ? `${emotionEmoji(r.emotion)} ${r.emotion}`
                      : r.emotion || "—"}
                  </td>
                  <td className="px-4 py-2 text-green-400">{(r.similarity * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

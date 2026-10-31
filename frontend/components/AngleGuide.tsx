"use client";

export const ANGLES = ["front", "left", "right", "up", "down"] as const;
export type Angle = typeof ANGLES[number];

const ANGLE_META: Record<Angle, { label: string; icon: string }> = {
  front: { label: "Nhìn thẳng",    icon: "😐" },
  left:  { label: "Quay trái",     icon: "⬅️" },
  right: { label: "Quay phải",     icon: "➡️" },
  up:    { label: "Ngẩng lên",     icon: "⬆️" },
  down:  { label: "Cúi xuống",     icon: "⬇️" },
};

interface Props {
  captured: Set<Angle>;
  current: Angle | null;
}

export default function AngleGuide({ captured, current }: Props) {
  return (
    <div className="w-full grid grid-cols-5 gap-2 my-4">
      {ANGLES.map((angle) => {
        const meta = ANGLE_META[angle];
        const done = captured.has(angle);
        const active = current === angle && !done;
        return (
          <div key={angle}
            className={`flex flex-col items-center p-3 rounded-xl border-2 transition-all
              ${done   ? "border-green-500 bg-green-900/30 text-green-400"
              : active ? "border-yellow-400 bg-yellow-900/30 text-yellow-300 animate-pulse"
              :          "border-gray-600 bg-gray-800/50 text-gray-400"}`}
          >
            <span className="text-2xl">{done ? "✅" : meta.icon}</span>
            <span className="text-xs mt-1 text-center leading-tight">{meta.label}</span>
          </div>
        );
      })}
    </div>
  );
}

"use client";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Check, ScanFace, type LucideIcon } from "lucide-react";

export const ANGLES = ["front", "left", "right", "up", "down"] as const;
export type Angle = typeof ANGLES[number];

export const ANGLE_META: Record<Angle, { label: string; instruction: string; icon: LucideIcon }> = {
  front: { label: "Front", instruction: "Look straight at the camera", icon: ScanFace },
  left:  { label: "Left",  instruction: "Turn your head to the left",  icon: ArrowLeft },
  right: { label: "Right", instruction: "Turn your head to the right", icon: ArrowRight },
  up:    { label: "Up",    instruction: "Tilt your head up",           icon: ArrowUp },
  down:  { label: "Down",  instruction: "Tilt your head down",         icon: ArrowDown },
};

interface Props {
  /** Preview image (data URL) for each angle already captured. */
  previews: Partial<Record<Angle, string>>;
  current: Angle | null;
}

/** The five head angles: a captured one shows its photo, the next one is marked. */
export default function AngleGuide({ previews, current }: Props) {
  return (
    <ol className="angles" aria-label="Face angles">
      {ANGLES.map((angle) => {
        const meta = ANGLE_META[angle];
        const preview = previews[angle];
        const state = preview ? "done" : current === angle ? "current" : "todo";
        const Icon = meta.icon;
        return (
          <li key={angle} className="angle" data-state={state}
            aria-label={`${meta.label}: ${state === "done" ? "captured" : state === "current" ? "next" : "not yet"}`}>
            {preview ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element -- a data URL from the camera */}
                <img src={preview} alt="" width={96} height={96} />
                <span className="angle-done"><Check size={14} aria-hidden="true" /></span>
              </>
            ) : (
              <>
                <Icon size={20} aria-hidden="true" />
                <span>{meta.label}</span>
              </>
            )}
          </li>
        );
      })}
    </ol>
  );
}

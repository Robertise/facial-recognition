"use client";
import { Check as CheckIcon, Minus, X } from "lucide-react";
import type { Check } from "@/lib/checks";
import { emotionIcon } from "@/lib/emotion";

const STATE_ICON = { pass: CheckIcon, fail: X, wait: Minus } as const;
const STATE_WORD = { pass: "Passed", fail: "Failed", wait: "Waiting" } as const;

/**
 * The checks a frame went through, in order. Each row says its state in words,
 * shows what was measured and draws the 0–1 reading as a meter.
 */
export default function CheckLadder({ checks }: { checks: Check[] }) {
  return (
    <ol className="ladder" aria-label="Checks for the face in view">
      {checks.map((c) => {
        const Icon = c.icon ? emotionIcon(c.icon) : STATE_ICON[c.state];
        return (
          <li key={c.key} data-state={c.state}>
            <Icon className="ladder-icon" size={18} aria-hidden="true" />
            <div>
              <span className="ladder-name">{c.name}</span>
              <span className="ladder-state">
                <span className="visually-hidden">{STATE_WORD[c.state]}: </span>
                {c.detail}
              </span>
            </div>
            <span className="ladder-reading measured measured--cool">{c.reading ?? ""}</span>
            {c.value !== undefined && (
              <div className="meter" aria-hidden="true">
                <span style={{ transform: `scaleX(${c.value})` }} />
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

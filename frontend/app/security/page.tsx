import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { INNOVATION } from "@/lib/flags";
import SecurityView from "./SecurityView";

export const metadata: Metadata = { title: "Security" };

export default function SecurityPage() {
  if (!INNOVATION) notFound();
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Security</h1>
          <p className="lead">Every verification attempt, with the score each channel gave it.</p>
        </div>
      </div>
      <SecurityView />
    </>
  );
}

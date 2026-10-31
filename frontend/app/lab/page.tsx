import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { INNOVATION } from "@/lib/flags";
import LabView from "./LabView";

export const metadata: Metadata = { title: "Lab" };

export default function LabPage() {
  if (!INNOVATION) notFound();
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Lab</h1>
          <p className="lead">Check that the camera and microphone record cleanly before any verification runs.</p>
        </div>
      </div>
      <LabView />
    </>
  );
}

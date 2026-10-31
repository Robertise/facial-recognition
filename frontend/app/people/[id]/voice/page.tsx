import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { INNOVATION } from "@/lib/flags";
import VoiceEnrollView from "./VoiceEnrollView";

export const metadata: Metadata = { title: "Enrol voice" };

export default async function VoicePage({ params }: { params: Promise<{ id: string }> }) {
  if (!INNOVATION) notFound();
  const { id } = await params;
  return <VoiceEnrollView employeeId={decodeURIComponent(id)} />;
}

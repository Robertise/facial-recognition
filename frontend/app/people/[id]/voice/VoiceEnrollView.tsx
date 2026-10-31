"use client";
import Link from "next/link";
import { EmptyState, ErrorState } from "@/components/DataState";
import VoiceEnrollment from "@/components/VoiceEnrollment";
import { getEmployees, useResource } from "@/lib/api";

/** Voice enrolment for someone already registered by face. */
export default function VoiceEnrollView({ employeeId }: { employeeId: string }) {
  const people = useResource(getEmployees);
  const person = people.data?.find((e) => e.id === employeeId);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Enrol voice</h1>
          <p className="lead">
            {person ? <>For {person.name} (ID <span className="measured">{person.id}</span>).</> : <>For ID <span className="measured">{employeeId}</span>.</>}
          </p>
        </div>
        <Link href="/people" className="btn">Back to people</Link>
      </div>

      {people.status === "loading" && <p className="helper" role="status">Loading</p>}
      {people.status === "error" && <ErrorState what="people" error={people.error} onRetry={people.reload} />}
      {people.status === "ready" && (person ? (
        <VoiceEnrollment employeeId={person.id} name={person.name} />
      ) : (
        <EmptyState title={`No registered person has the ID ${employeeId}.`}>
          <p>Register their face first, then enrol their voice.</p>
          <Link href="/register" className="btn">Register a person</Link>
        </EmptyState>
      ))}
    </>
  );
}

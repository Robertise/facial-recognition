import type { Metadata } from "next";
import { Suspense } from "react";
import Link from "next/link";
import { UserPlus } from "lucide-react";
import { TableSkeleton } from "@/components/DataState";
import PeopleView, { PEOPLE_COLUMNS } from "./PeopleView";

export const metadata: Metadata = { title: "People" };

export default function PeoplePage() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1>People</h1>
          <p className="lead">Everyone registered to check in, with their saved face angles.</p>
        </div>
        <Link href="/register" className="btn btn--primary"><UserPlus size={16} aria-hidden="true" />Register a person</Link>
      </div>
      <Suspense fallback={<TableSkeleton columns={PEOPLE_COLUMNS} />}>
        <PeopleView />
      </Suspense>
    </>
  );
}

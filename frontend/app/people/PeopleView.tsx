"use client";
import { useRef, useState } from "react";
import Link from "next/link";
import { LoaderCircle, Trash2 } from "lucide-react";
import { EmptyState, ErrorState, TableSkeleton } from "@/components/DataState";
import { deleteEmployee, getEmployees, useResource, type Employee } from "@/lib/api";
import { formatDate, parseDbTime } from "@/lib/format";
import { matches, useSearchParam } from "@/lib/search";

export const PEOPLE_COLUMNS = ["ID", "Name", "Face angles", "Registered", ""];

export default function PeopleView() {
  const people = useResource(getEmployees);
  const [urlQ, setUrlQ] = useSearchParam("q");
  const [q, setText] = useState(urlQ);
  const setQ = (v: string) => { setText(v); setUrlQ(v); };
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [target, setTarget] = useState<Employee | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const askDelete = (e: Employee) => {
    setTarget(e);
    setDeleteError(null);
    dialogRef.current?.showModal();
  };

  const confirmDelete = async () => {
    if (!target) return;
    setBusy(true);
    setDeleteError(null);
    try {
      await deleteEmployee(target.id);
      dialogRef.current?.close();
      people.reload();
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      setDeleteError(`The backend refused (${reason}). Someone who already has check-in records cannot be deleted yet.`);
    } finally {
      setBusy(false);
    }
  };

  const all = people.data ?? [];
  const shown = all.filter((e) => matches(q, e.id, e.name));

  return (
    <>
      <div className="toolbar">
        <div className="field">
          <label htmlFor="people-search">Search by name or ID</label>
          <input id="people-search" name="q" type="search" className="input" value={q}
            onChange={(e) => setQ(e.target.value)} placeholder="nguyen" autoComplete="off" />
        </div>
        {people.status === "ready" && (
          <p className="helper" role="status">
            {q ? `${shown.length} of ${all.length} people` : `${all.length} ${all.length === 1 ? "person" : "people"}`}
          </p>
        )}
      </div>

      {people.status === "loading" && <TableSkeleton columns={PEOPLE_COLUMNS} />}
      {people.status === "error" && <ErrorState what="people" error={people.error} onRetry={people.reload} />}
      {people.status === "ready" && (all.length === 0 ? (
        <EmptyState title="No one is registered yet.">
          <p>Registered people are recognised at check-in.</p>
          <Link href="/register" className="btn">Register a person</Link>
        </EmptyState>
      ) : shown.length === 0 ? (
        <EmptyState title={`No one matches “${q}”.`}>
          <p>Search looks at names and IDs, ignoring accents.</p>
          <button type="button" className="btn" onClick={() => setQ("")}>Clear search</button>
        </EmptyState>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <caption className="visually-hidden">Registered people</caption>
            <thead>
              <tr>
                <th scope="col">ID</th>
                <th scope="col">Name</th>
                <th scope="col" className="num">Face angles</th>
                <th scope="col">Registered</th>
                <th scope="col" className="cell-action"><span className="visually-hidden">Actions</span></th>
              </tr>
            </thead>
            <tbody>
              {shown.map((e) => {
                const at = parseDbTime(e.created_at);
                return (
                  <tr key={e.id}>
                    <td className="measured measured--cool" data-label="ID">{e.id}</td>
                    <td data-label="Name">{e.name}</td>
                    <td className="num" data-label="Face angles">{e.angle_count} of 5</td>
                    <td className="measured" data-label="Registered"><time dateTime={at.toISOString()}>{formatDate(at)}</time></td>
                    <td className="cell-action" data-label="">
                      <button type="button" className="btn btn--danger" onClick={() => askDelete(e)} aria-label={`Delete ${e.name}`}>
                        <Trash2 size={16} aria-hidden="true" /> Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}

      <dialog ref={dialogRef} className="dialog" aria-labelledby="delete-title" onClose={() => setTarget(null)}>
        {target && (
          <>
            <h2 id="delete-title">Delete {target.name}?</h2>
            <p>This removes {target.name} (ID <span className="measured">{target.id}</span>) and their saved face angles. It cannot be undone.</p>
            {deleteError && <p className="notice" data-tone="bad" role="alert">{deleteError}</p>}
            <div className="actions">
              <button type="button" className="btn btn--quiet" onClick={() => dialogRef.current?.close()}>Cancel</button>
              <button type="button" className="btn btn--danger" onClick={confirmDelete} disabled={busy} aria-busy={busy || undefined}>
                {busy ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <Trash2 size={16} aria-hidden="true" />}
                Delete
              </button>
            </div>
          </>
        )}
      </dialog>
    </>
  );
}

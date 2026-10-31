"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ScanFace } from "lucide-react";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/attendance", label: "Check-in" },
  { href: "/register", label: "Register" },
  { href: "/people", label: "People" },
  { href: "/history", label: "History" },
];

export default function TopBar() {
  const pathname = usePathname();
  return (
    <header className="topbar">
      <div className="container topbar-inner">
        <Link href="/" className="wordmark" aria-label="Face Attendance, overview">
          <ScanFace size={20} aria-hidden="true" />
          Face Attendance
        </Link>
        <nav className="nav" aria-label="Main">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} className="nav-link"
              aria-current={pathname === l.href ? "page" : undefined}>
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}

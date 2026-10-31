import type { Metadata } from "next";
import { Atkinson_Hyperlegible_Mono, Be_Vietnam_Pro, Host_Grotesk } from "next/font/google";
import TopBar from "@/components/TopBar";
import "./globals.css";

const hostGrotesk = Host_Grotesk({ subsets: ["latin", "latin-ext"], weight: ["400", "500"], variable: "--font-host-grotesk", display: "swap" });
// Host Grotesk has no Vietnamese glyphs; this face fills in only those (names such as "Nguyễn").
const beVietnam = Be_Vietnam_Pro({ subsets: ["vietnamese"], weight: ["400", "500"], variable: "--font-be-vietnam", display: "swap", preload: false });
const atkinsonMono = Atkinson_Hyperlegible_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-atkinson-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Face Attendance", template: "%s | Face Attendance" },
  description: "Check people in by face, with liveness and emotion checks on every frame.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${hostGrotesk.variable} ${beVietnam.variable} ${atkinsonMono.variable}`}>
      <body>
        <a className="skip" href="#main">Skip to content</a>
        <TopBar />
        <main id="main" className="container page">
          {children}
        </main>
      </body>
    </html>
  );
}

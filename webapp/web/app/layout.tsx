import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ThaiSLM — ภาษามือไทย → เสียงพูด",
  description: "Thai Sign Language to Thai text and speech, in the browser",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th">
      <body>{children}</body>
    </html>
  );
}

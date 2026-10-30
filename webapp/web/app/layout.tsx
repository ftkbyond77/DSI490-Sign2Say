import type { Metadata, Viewport } from "next";
import { IBM_Plex_Sans_Thai, Prompt } from "next/font/google";
import "./globals.css";

// the two typefaces of the Figma file: Prompt (headings), IBM Plex Sans Thai (body) — downloaded at build time and served from our origin
const prompt = Prompt({ subsets: ["thai", "latin"], weight: ["500", "600", "700"], variable: "--font-prompt", display: "swap" });
const plex = IBM_Plex_Sans_Thai({ subsets: ["thai", "latin"], weight: ["400", "500", "600"], variable: "--font-plex", display: "swap" });

export const metadata: Metadata = {
  title: "ThaiSLM — แปลภาษามือไทย",
  description: "Thai Sign Language ⇄ Thai text and speech: sign to the camera, or type / speak and the avatar signs",
};
export const viewport: Viewport = { themeColor: "#0f766e", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th" className={`${prompt.variable} ${plex.variable}`} suppressHydrationWarning>
      <head>
        {/* the folded side menu is applied before the first paint, so pages do not animate it open → closed on every visit */}
        <script dangerouslySetInnerHTML={{ __html: `try{if(localStorage.getItem("thaislm.sidebar")==="folded")document.documentElement.dataset.sidebar="folded"}catch(e){}` }} />
      </head>
      <body>{children}</body>
    </html>
  );
}

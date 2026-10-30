"use client";
// Renders dialogs / menus at the end of <body>: inside an animated card (transform) a position: fixed layer would be trapped in the card.
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

export default function Portal({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  useEffect(() => setReady(true), []);
  return ready ? createPortal(children, document.body) : null;
}

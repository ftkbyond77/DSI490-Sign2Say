"use client";
// WebGL is decoration on the landing page: if a canvas cannot start (no WebGL, lost context, driver bug) the page must keep working.
import { Component, type ReactNode } from "react";

export function webglOk(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch { return false; }
}

export default class GlBoundary extends Component<{ fallback?: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(e: unknown) { console.warn("[landing] WebGL layer disabled:", e); }
  render() { return this.state.failed ? (this.props.fallback ?? null) : this.props.children; }
}

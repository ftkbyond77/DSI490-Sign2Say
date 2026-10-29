export type CameraDiagnostics = {
  mode: "wide";
  requested: MediaStreamConstraints;
  settings: MediaTrackSettings;
  capabilities: MediaTrackCapabilities | null;
  appliedConstraints: MediaTrackConstraints | null;
  capabilitiesAvailable: boolean;
  capabilitiesError: string | null;
  resizeModeCapabilityPresent: boolean;
  resizeModeNoneSupported: boolean | null;
  resizeModeConstraintRecognized: boolean | null;
  resizeModeNoneApplied: boolean | null;
  capturedAt: string;
};
const wideRequests = new WeakMap<MediaStream, MediaStreamConstraints>();

/** Report missing capability separately from unsupported; never infer full sensor FOV. */
export function cameraDiagnostics(stream: MediaStream): CameraDiagnostics | undefined {
  const requested = wideRequests.get(stream);
  if (!requested) return undefined;
  const tr = stream.getVideoTracks()[0];
  if (!tr) return undefined;
  let caps: MediaTrackCapabilities | null = null, capabilitiesError: string | null = null;
  try { caps = tr.getCapabilities ? tr.getCapabilities() : null; }
  catch (e) { capabilitiesError = String(e); }
  let appliedConstraints: MediaTrackConstraints | null = null;
  try { appliedConstraints = tr.getConstraints?.() ?? null; } catch { /* optional diagnostic */ }
  const settings = tr.getSettings();
  const modes = (caps as (MediaTrackCapabilities & { resizeMode?: string[] }) | null)?.resizeMode;
  const actual = (settings as MediaTrackSettings & { resizeMode?: string }).resizeMode;
  const recognized = navigator.mediaDevices.getSupportedConstraints?.() as (MediaTrackSupportedConstraints & { resizeMode?: boolean }) | undefined;
  return { mode: "wide", requested, settings, capabilities: caps, appliedConstraints,
    capabilitiesAvailable: caps !== null, capabilitiesError,
    resizeModeCapabilityPresent: Array.isArray(modes),
    resizeModeNoneSupported: Array.isArray(modes) ? modes.includes("none") : null,
    resizeModeConstraintRecognized: recognized?.resizeMode ?? null,
    resizeModeNoneApplied: actual === undefined ? null : actual === "none",
    capturedAt: new Date().toISOString() };
}

/** Open the webcam with its widest and tallest view: the full sensor (native aspect, often 4:3), not a 16:9 crop of it.
 *  The native aspect is the one of the largest mode; a native-aspect mode shows the whole sensor at any size, so we take the smallest one
 *  from 720 lines up that runs at ≥ 24 fps (more pixels add no view, only time: MediaPipe downsamples anyway). Zoom goes to its minimum. */
export async function openWidestCamera(): Promise<MediaStream> {
  if (new URLSearchParams(window.location.search).get("cam") === "wide") {
    const requested: MediaStreamConstraints = { video: {
      facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, resizeMode: "none",
    } as MediaTrackConstraints, audio: false };
    const stream = await navigator.mediaDevices.getUserMedia(requested);
    wideRequests.set(stream, requested);
    return stream; // No resolution loop, zoom override, or silent fallback in this mode.
  }
  const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  const tr = s.getVideoTracks()[0];
  const caps = (tr.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { zoom?: { min: number } };
  const maxW = caps.width?.max, maxH = caps.height?.max;
  if (!maxW || !maxH) return s;
  const a = maxW / maxH;
  // pitty: ?cam=480 (or 360/540/…) asks for that many lines first — a lighter frame for slow machines (A/B test; default unchanged)
  const want = Number(new URLSearchParams(window.location.search).get("cam")) || 0;
  const lines = want ? [want, 720] : [720, 768, 900, 960, 1080];
  const tries = lines.filter((h) => h <= maxH).map((h) => ({ w: Math.min(maxW, Math.round((h * a) / 16) * 16), h }));
  for (const t of tries) {
    try {
      await tr.applyConstraints({ width: { exact: t.w }, height: { exact: t.h }, frameRate: { ideal: 30 }, resizeMode: "none" } as MediaTrackConstraints);
      if ((tr.getSettings().frameRate ?? 30) >= 24) break;
    } catch { /* the camera has no such mode: try the next */ }
  }
  if (caps.zoom) try { await tr.applyConstraints({ advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet] }); } catch { /* fixed zoom */ }
  return s;
}

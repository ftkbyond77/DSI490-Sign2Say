/** Open the webcam with its widest and tallest view: the full sensor (native aspect, often 4:3), not a 16:9 crop of it.
 *  The native aspect is the one of the largest mode; a native-aspect mode shows the whole sensor at any size, so we take the smallest one
 *  from 720 lines up that runs at ≥ 24 fps (more pixels add no view, only time: MediaPipe downsamples anyway). Zoom goes to its minimum. */
export async function openWidestCamera(): Promise<MediaStream> {
  const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  const tr = s.getVideoTracks()[0];
  const caps = (tr.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { zoom?: { min: number } };
  const maxW = caps.width?.max, maxH = caps.height?.max;
  if (!maxW || !maxH) return s;
  const a = maxW / maxH;
  const tries = [720, 768, 900, 960, 1080].filter((h) => h <= maxH).map((h) => ({ w: Math.min(maxW, Math.round((h * a) / 16) * 16), h }));
  for (const t of tries) {
    try {
      await tr.applyConstraints({ width: { exact: t.w }, height: { exact: t.h }, frameRate: { ideal: 30 }, resizeMode: "none" } as MediaTrackConstraints);
      if ((tr.getSettings().frameRate ?? 30) >= 24) break;
    } catch { /* the camera has no such mode: try the next */ }
  }
  if (caps.zoom) try { await tr.applyConstraints({ advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet] }); } catch { /* fixed zoom */ }
  return s;
}

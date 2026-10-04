// Speech playback that browsers do not block: one AudioContext is resumed during a click (a user gesture), after which any
// later WAV — e.g. the result of live mode, which arrives without a click — can be played through it.
let ctx: AudioContext | null = null;

/** Call from a click handler (open camera, process, replay) so later playback is allowed. */
export function unlockAudio() {
  try {
    if (!ctx) ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    if (ctx.state === "suspended") void ctx.resume();
    const b = ctx.createBuffer(1, 1, 22050), s = ctx.createBufferSource();   // a silent sample completes the unlock on Safari
    s.buffer = b; s.connect(ctx.destination); s.start(0);
  } catch { /* no Web Audio: playWav falls back to <audio> */ }
}

export function audioReady() {
  return !!ctx && ctx.state === "running";
}

/** Play WAV bytes; resolves when playback ends. Throws when the browser blocks audio. */
export async function playWav(bytes: ArrayBuffer): Promise<void> {
  if (ctx) {
    if (ctx.state === "suspended") await ctx.resume();
    const buf = await ctx.decodeAudioData(bytes.slice(0));
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    await new Promise<void>((done) => { src.onended = () => done(); src.start(0); });
    return;
  }
  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  const a = new Audio(url);
  await a.play();
  await new Promise<void>((done) => { a.onended = () => { URL.revokeObjectURL(url); done(); }; });
}

export function b64ToBytes(b64: string): ArrayBuffer {
  const s = atob(b64), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out.buffer;
}

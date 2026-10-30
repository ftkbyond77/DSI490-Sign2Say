"use client";
// The signing avatar of the reverse part (text / speech → Thai Sign Language), as a React hook: the three-vrm stage, the character
// catalogue, the idle pose, the render loop (breathing + natural blinks, from reverse_text2sign/service/web/src/main.ts) and playback.
// Used by /text (full stage), /chat (tile) and /sign/live (picture-in-picture).
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type AvatarEntry, type IR, type Result } from "./api";
import type { Avatar as AvatarT, Pose } from "./avatar";
import type { Motion as MotionT, Player as PlayerT } from "./player";

const FALLBACK_AVATAR: AvatarEntry = { id: null, label: "Default", file: "avatar.vrm", title: "", author: "", vrm_version: "0",
  url: "/avatar.vrm", thumb: null, bytes: 0, scale: 1, optimised: true };

export type Framing = "half" | "close";

export function useAvatar(hostRef: React.RefObject<HTMLDivElement | null>) {
  const avatarRef = useRef<AvatarT | null>(null);
  const playerRef = useRef<PlayerT | null>(null);
  const libRef = useRef<{ Motion: typeof MotionT; sample: typeof import("./player").sample } | null>(null);
  const restRef = useRef<Pose | null>(null);
  const currentRef = useRef<Result | null>(null);
  const headRef = useRef<HTMLDivElement | null>(null);          // timeline play head: moved every frame without a React render
  const fillRef = useRef<HTMLDivElement | null>(null);
  const unitRef = useRef(-2);
  const switching = useRef(false);

  const [ready, setReady] = useState(false);
  const [loadText, setLoadText] = useState("กำลังโหลดตัวละคร…");
  const [failed, setFailed] = useState("");
  const [cast, setCast] = useState<AvatarEntry[]>([]);
  const [avatarId, setAvatarId] = useState<string | null>(null);
  const [switchingId, setSwitchingId] = useState<string | null | undefined>(undefined);
  const [current, setCurrent] = useState<Result | null>(null);
  const [unit, setUnit] = useState(-1);                          // index into current.ir.units of the sign on screen (-1: none)
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [loop, setLoopState] = useState(false);
  const [framing, setFramingState] = useState<Framing>("half");
  const [fps, setFps] = useState(0);

  const setRest = useCallback((rp: MotionT) => {
    const av = avatarRef.current!, lib = libRef.current!, pl = playerRef.current!;
    av.bind(rp.bones, rp.morphs);
    const pose = { q: new Float32Array(rp.B * 4), m: new Float32Array(rp.M) };
    lib.sample(rp, 0, pose);
    restRef.current = pose;
    if (pl.motion) av.bind(pl.motion.bones, pl.motion.morphs);
  }, []);

  // boot: three.js only in the browser (dynamic import keeps it out of the server bundle)
  useEffect(() => {
    let alive = true, raf = 0;
    (async () => {
      const host = hostRef.current;
      if (!host) return;
      const [{ Avatar }, { Motion, Player, sample }] = await Promise.all([import("./avatar"), import("./player")]);
      if (!alive) return;
      libRef.current = { Motion, sample };
      const av = new Avatar(host);
      const pl = new Player();
      pl.onEnd = () => setPlaying(false);
      avatarRef.current = av; playerRef.current = pl;
      try {
        let list: AvatarEntry[] = [];
        try { list = (await api.avatars()).avatars; } catch { list = []; }
        if (!list.length) list = [FALLBACK_AVATAR];                   // API without a catalogue: the bundled library avatar
        let saved: string | null = null;
        try { saved = localStorage.getItem("t2s-avatar"); } catch { /* storage blocked */ }
        const first = list.find((c) => c.id === saved) ?? list[0];
        const [vrm, rest] = await Promise.all([
          av.fetch(first.url, (f) => alive && setLoadText(`กำลังโหลดตัวละคร… ${Math.round(f * 100)}%`)),
          api.rest(first.id),
        ]);
        if (!alive) return;
        av.show(vrm);
        setAvatarId(first.id); setCast(list);
        setRest(new Motion(rest));
        setReady(true);
        // characters whose .vrm carries no thumbnail: render a headshot once in the background and give it to the API (kept for the
        // next visitor) — from the reverse part's front end; a missing thumbnail never disturbs the page
        (async () => {
          const { headshot } = await import("./avatar");
          const { VRMUtils } = await import("@pixiv/three-vrm");
          for (const c of list.filter((x) => !x.thumb && x.id)) {
            if (!alive) return;
            try {
              const v = await av.fetch(c.url);
              const blob = await headshot(v);
              VRMUtils.deepDispose(v.scene);
              const r = await api.avatarThumbnail(c.id!, blob);
              setCast((cs) => cs.map((x) => (x.id === c.id ? { ...x, thumb: r.thumb } : x)));
            } catch { /* keep the initial */ }
          }
        })();
      } catch (e) {
        if (alive) setFailed("โหลดตัวละครไม่สำเร็จ: " + (e as Error).message);
        return;
      }
      // render loop: life (breathing on the chest, blinks when the motion's own face is not blinking), playback, timeline
      let last = performance.now(), fpsAcc = 0, fpsN = 0, blinkAt = performance.now() + 2500;
      const idle: Pose = { q: new Float32Array(0), m: new Float32Array(0) };
      const frame = (now: number) => {
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        fpsAcc += dt; fpsN++;
        if (fpsAcc > 1) { setFps(Math.round(fpsN / fpsAcc)); fpsAcc = 0; fpsN = 0; }
        let pose: Pose | null = null;
        const f = pl.tick(dt);
        const restPose = restRef.current;
        if (pl.motion && (pl.playing || pl.t > 0)) pose = pl.pose;
        else if (restPose) {
          if (idle.q.length !== restPose.q.length) { idle.q = new Float32Array(restPose.q); idle.m = new Float32Array(restPose.m); }
          idle.q.set(restPose.q); idle.m.set(restPose.m);
          pose = idle;
        }
        if (pose) {
          const br = Math.sin(now / 1000 * 2 * Math.PI / 4.2) * 0.006;
          const ci = 1 * 4; // "chest" is bone #1 in rig order (spine, chest, …): a ~0.7° breathing sway about its X axis
          const qx = pose.q[ci], qw = pose.q[ci + 3];
          pose.q[ci] = qx + br * qw; pose.q[ci + 3] = qw - br * qx;
          av.apply(pose);
          av.update(dt);
          const m = pose.m.slice();
          if (now > blinkAt) {
            const u = (now - blinkAt) / 140;
            const w = u < 1 ? Math.sin(u * Math.PI) : 0;
            if (u >= 1) blinkAt = now + 2500 + Math.random() * 3500;
            const il = pl.motion?.morphs.indexOf("Fcl_EYE_Close_L") ?? 3, ir = pl.motion?.morphs.indexOf("Fcl_EYE_Close_R") ?? 4;
            if (il >= 0) m[il] = Math.max(m[il], w);
            if (ir >= 0) m[ir] = Math.max(m[ir], w);
          }
          av.applyMorphs(m);
        } else av.update(dt);
        av.render();
        const mo = pl.motion;
        if (mo && f >= 0) {
          const p = `${(f / Math.max(mo.frames - 1, 1)) * 100}%`;
          if (headRef.current) headRef.current.style.left = p;
          if (fillRef.current) fillRef.current.style.width = p;
          const seg = mo.segments.find((s) => s.kind !== "transition" && f >= s.start && f < s.end);
          const ui = seg ? seg.unit : -1;
          if (ui !== unitRef.current) { unitRef.current = ui; setUnit(ui); }
        }
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);
    })();
    return () => { alive = false; cancelAnimationFrame(raf); avatarRef.current?.dispose(); avatarRef.current?.renderer.domElement.remove(); avatarRef.current = null; };
  }, [hostRef, setRest]);

  /** Load a translation result on stage (and play it). */
  const show = useCallback((r: Result, autoplay = true) => {
    const av = avatarRef.current, pl = playerRef.current, lib = libRef.current;
    if (!av || !pl || !lib) return;
    currentRef.current = r; setCurrent(r);
    const mo = new lib.Motion(r.motion);
    av.bind(mo.bones, mo.morphs);
    pl.load(mo);
    pl.speed = speedRef.current; pl.loop = loopRef.current;
    unitRef.current = -2; setUnit(-1);
    if (autoplay) { pl.play(); setPlaying(true); } else setPlaying(false);
  }, []);

  const speedRef = useRef(1), loopRef = useRef(false);
  const setSpeed = useCallback((s: number) => { speedRef.current = s; setSpeedState(s); if (playerRef.current) playerRef.current.speed = s; }, []);
  const setLoop = useCallback((l: boolean) => { loopRef.current = l; setLoopState(l); if (playerRef.current) playerRef.current.loop = l; }, []);
  const setFraming = useCallback((f: Framing) => { setFramingState(f); const av = avatarRef.current; if (av) { av.framing = f; av.frame(); } }, []);

  const toggle = useCallback(() => {
    const pl = playerRef.current;
    if (!pl?.motion) return false;
    if (pl.playing) { pl.pause(); setPlaying(false); } else { pl.play(); setPlaying(true); }
    return true;
  }, []);
  const replay = useCallback(() => { const pl = playerRef.current; if (pl?.motion) { pl.seek(0); pl.play(); setPlaying(true); } }, []);
  const seek = useCallback((t: number, play = true) => {
    const pl = playerRef.current; if (!pl?.motion) return;
    pl.seek(t); if (play) { pl.play(); setPlaying(true); }
  }, []);
  /** jump to a unit (gloss) of the current sentence */
  const seekUnit = useCallback((i: number) => {
    const u = currentRef.current?.ir.units[i];
    if (u?.timing) seek(u.timing.start_s);
  }, [seek]);
  const duration = () => playerRef.current?.motion?.duration ?? 0;

  /** switch character: character, idle pose and the current sentence (fitted to the new body) arrive together, then swap */
  const switchAvatar = useCallback(async (c: AvatarEntry) => {
    const av = avatarRef.current, pl = playerRef.current, lib = libRef.current;
    if (!av || !pl || !lib || switching.current || c.id === avatarId) return;
    switching.current = true; setSwitchingId(c.id);
    try {
      const cur = currentRef.current;
      const [vrm, rest, r] = await Promise.all([av.fetch(c.url), api.rest(c.id), cur ? api.render(cur.ir, 1.0, c.id) : Promise.resolve(null)]);
      av.show(vrm);
      setAvatarId(c.id);
      if (r && cur) {
        const t = pl.t, wasPlaying = pl.playing;
        const next = { ir: cur.ir, motion: r.motion };
        currentRef.current = next; setCurrent(next);
        pl.load(new lib.Motion(r.motion));
        pl.speed = speedRef.current; pl.loop = loopRef.current;
        pl.seek(t);
        if (wasPlaying) pl.play();
      }
      setRest(new lib.Motion(rest));
      try { localStorage.setItem("t2s-avatar", c.id ?? ""); } catch { /* storage blocked */ }
    } finally { switching.current = false; setSwitchingId(undefined); }
  }, [avatarId, setRest]);

  /** re-render an edited IR (sign order editor, variant picker) on the current character */
  const render = useCallback(async (ir: IR) => { show(await api.render(ir, 1.0, avatarId), true); }, [avatarId, show]);

  return { ready, loadText, failed, cast, avatarId, switchingId, switchAvatar, current, unit, playing, speed, setSpeed, loop, setLoop,
    framing, setFraming, fps, show, toggle, replay, seek, seekUnit, duration, render, headRef, fillRef, avatarRef };
}
export type AvatarCtl = ReturnType<typeof useAvatar>;

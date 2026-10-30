"use client";
// v2-E — ข้อความ → ภาษามือ: type or speak (Thai / English), see the sign order (TSL grammar or Thai order), watch the avatar sign it.
// The logic is the reverse part's front end (reverse_text2sign/service/web/src/main.ts) on this app's design.
import { useCallback, useEffect, useRef, useState } from "react";
import AvatarStage, { KIND_TH } from "@/components/avatar/AvatarStage";
import AppShell from "@/components/AppShell";
import { ProgressBar } from "@/components/ui";
import { api, type AsrResult, type IR, type Unit } from "@/lib/t2s/api";
import { useAsr } from "@/lib/t2s/useAsr";
import { useAvatar } from "@/lib/t2s/useAvatar";

const ROLE_TH: Record<string, string> = { time: "เวลา", subject: "ผู้ทำ", verb: "กริยา", object: "กรรม", noun: "คำนาม", wh: "คำถาม", neg: "ปฏิเสธ",
  adjective: "คุณศัพท์", number: "จำนวน", greeting: "ทักทาย", aspect: "การณ์", phrase: "ประโยค" };
const TYPE_TH: Record<string, string> = { wh_question: "คำถาม (ใคร/อะไร/ที่ไหน…)", yn_question: "คำถาม ใช่/ไม่ใช่", negative: "ปฏิเสธ", statement: "บอกเล่า", greeting: "ทักทาย" };
const MATCH_TH: Record<string, string> = { alias: "ชื่อเรียกอื่น", group: "คำพ้องความหมาย", ttrs_synonym: "คำพ้อง (TTRS)", pronoun: "สรรพนาม → ท่าชี้",
  compound: "แยกคำประสม", aspect: "คำบอกการณ์", fingerspell: "สะกดนิ้ว", none: "ไม่มีในคลัง", english: "อังกฤษ", curated: "คำพ้อง (คัดเลือก)",
  phrase: "ประโยคจริง", sense: "เลือกความหมายจากบริบท" };
const ORIGIN_TH: Record<string, string> = { recorded_clip: "คลิปจริง", recorded_phrase: "ประโยคบันทึกจริง", composed: "ประกอบจากท่าจริง",
  fingerspelling: "สะกดนิ้ว (สำรอง)", text: "ไม่มีท่า (ข้อความ)" };
const EXAMPLES = ["สวัสดี สบายดีไหม", "ขอบคุณมาก", "ห้องน้ำอยู่ที่ไหน"];
const pct = (x?: number) => (x === undefined || x === null ? "–" : `${Math.round(x * 100)}%`);

/** the small line under a gloss: its grammar role (design: เวลา · ผู้ทำ · กริยา · คำถาม · คิ้ว), or how it is made */
function sub(u: Unit, now: boolean) {
  if (now) return "กำลังแสดง";
  const brows = u.nmm && (u.nmm as Record<string, string>).brows;
  if (u.kind !== "sign") return KIND_TH[u.kind] ?? "";
  const role = u.role ? ROLE_TH[u.role] ?? "" : "";
  return [role, brows && brows !== "neutral" ? "คิ้ว" : ""].filter(Boolean).join(" · ") || MATCH_TH[u.match] || "";
}

export default function TextPage() {
  const hostRef = useRef<HTMLDivElement>(null);
  const a = useAvatar(hostRef);
  const [text, setText] = useState("");
  const [lang, setLang] = useState<"auto" | "th" | "en">("auto");
  const [grammar, setGrammar] = useState<"tsl51" | "thai">("tsl51");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [examples, setExamples] = useState<string[]>(EXAMPLES);
  const [health, setHealth] = useState<{ ok: boolean; text: string; device: string } | null>(null);
  const [asr, setAsr] = useState<AsrResult | null>(null);
  const asrMeta = useRef<Record<string, unknown> | null>(null);
  const [edit, setEdit] = useState<Unit[] | null>(null);
  const [showIR, setShowIR] = useState(false);
  const pending = useRef<string | null>(null);

  useEffect(() => {
    api.health().then((h) => setHealth({ ok: h.ok, device: h.speech?.device ?? "cpu",
      text: `คลังท่า ${h.signs.toLocaleString()} คำ${h.english ? " · EN" : ""}${h.speech?.th_model ? ` · เสียง ${h.speech.th_model.replace("whisper-th-", "Whisper-th ")}` : ""}` }))
      .catch(() => setHealth({ ok: false, device: "cpu", text: "เชื่อมต่อเซิร์ฟเวอร์แปลข้อความไม่ได้" }));
    api.examples().then(({ examples }) => examples.length && setExamples(examples.slice(0, 6))).catch(() => {});
    const q = new URLSearchParams(window.location.search).get("q");         // /text?q=คำ (from คลังคำ): sign it right away
    if (q) { setText(q); pending.current = q; }
  }, []);

  const go = useCallback(async (source = "text", t = text, g = grammar) => {
    const s = t.trim();
    if (!s || busy) return;
    setBusy(true); setError("");
    try {
      const r = await api.translate(s, lang, 1.0, source, a.avatarId, g, source === "speech" ? asrMeta.current : null);
      a.show(r, true);
      setEdit(null);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }, [a, busy, grammar, lang, text]);

  useEffect(() => { if (a.ready && pending.current) { const q = pending.current; pending.current = null; go("text", q); } }, [a.ready, go]);

  const mic = useAsr((r) => {
    setText(r.text);
    asrMeta.current = { raw: r.raw, model: r.model, device: r.device, reliable: r.reliable, flags: r.flags, suggestions: r.suggestions, ms: r.ms };
    setAsr(r);
    if (r.text.trim() && r.reliable) go("speech", r.text);
  }, health?.device);

  const ir = a.current?.ir;
  const segs = a.current?.motion.segments.filter((s) => s.kind !== "transition") ?? [];
  const frames = a.current?.motion.frames ?? 1;
  const types = ir ? [...new Set(ir.clauses.map((c) => c.type))].filter((t) => t !== "statement") : [];

  const rerender = async (units: Unit[]) => {
    if (!ir) return;
    const next: IR = { ...ir, units, clauses: [{ text: ir.thai, type: ir.clauses[0]?.type ?? "statement", negated: false, moves: ["แก้ไขโดยผู้ใช้"], units: units.map((_, i) => i) }] };
    try { await a.render(next); setEdit(null); } catch (e) { setError((e as Error).message); }
  };
  const exportVrma = async () => {
    if (!ir) return;
    try {
      const blob = await api.vrma(ir, 1.0, a.avatarId);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob); link.download = "signing.vrma"; link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 4000);
    } catch (e) { setError((e as Error).message); }
  };

  return (
    <AppShell tone="rose">
      <div className="title-row rise">
        <div><div className="hello rose">ข้อความหรือเสียง → อวาตาร์ทำภาษามือ</div><h1>แปลข้อความเป็นท่า</h1></div>
        <div className="options">
          <span className="opt-label">ภาษาที่พิมพ์</span>
          <div className="seg" role="radiogroup" aria-label="ภาษาที่พิมพ์">
            {(["auto", "th", "en"] as const).map((l) => <button key={l} className={lang === l ? "on" : ""} onClick={() => setLang(l)} role="radio" aria-checked={lang === l}>
              {l === "auto" ? "อัตโนมัติ" : l === "th" ? "ไทย" : "EN"}</button>)}
          </div>
        </div>
      </div>

      <div className="rev-main">
        <div className="rev-col">
          <section className="card">
            <h2>พิมพ์หรือพูด</h2>
            <textarea className="textarea" value={text} maxLength={400} rows={3} placeholder="เช่น พรุ่งนี้คุณไปกินข้าวที่ไหน"
              onChange={(e) => { setText(e.target.value); if (asrMeta.current) asrMeta.current = { ...asrMeta.current, edited: true }; }}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); }} aria-label="ข้อความที่ต้องการแปล" />
            {mic.state === "recording" && (
              <div className="live-box">
                <div className="h"><span className="dot rec" />{mic.committed || mic.tentative ? "กำลังฟัง — ข้อความชั่วคราว" : "กำลังฟัง… พูดได้เลย"}</div>
                <span>{mic.committed}</span> <span className="t">{mic.tentative}</span>
              </div>
            )}
            {mic.progress && <ProgressBar rose progress={{ p: mic.progress.p / 100 }} label={`กำลังถอดเสียงเป็นข้อความ… ${Math.floor(mic.progress.p)}% · ${mic.progress.elapsed.toFixed(1)} / ~${mic.progress.expect.toFixed(1)} วิ`} />}
            {asr && mic.state === "idle" && (
              <div className="asrbox">
                <span>ได้ยินว่า: “{asr.raw}” · {asr.ms} ms</span>
                {!asr.reliable && <div className="warnbox">ระบบไม่แน่ใจข้อความนี้ — ยังไม่แปล: {(asr.flags ?? []).join(" · ")}
                  <button className="mini-btn" onClick={() => mic.start(lang === "en" ? "en" : "th")}>พูดใหม่</button>
                  <button className="mini-btn" onClick={() => { setAsr(null); go("speech"); }}>ใช้ข้อความนี้</button></div>}
                {(asr.suggestions ?? []).map((sg) => (
                  <div className="sugg" key={sg.original + sg.corrected}>แนะนำแก้: “{sg.original}” → “{sg.corrected}”
                    <button className="mini-btn" onClick={() => {
                      const t = text.replace(sg.original, sg.corrected); setText(t);
                      asrMeta.current = { ...(asrMeta.current ?? {}), applied: [...((asrMeta.current?.applied as unknown[]) ?? []), sg] };
                      setAsr({ ...asr, suggestions: asr.suggestions.filter((x) => x !== sg) }); go("speech", t);
                    }}>ใช้</button></div>
                ))}
              </div>
            )}
            <div className="in-actions">
              <button className={`mic ${mic.state === "recording" ? "on" : mic.state === "final" ? "wait" : ""}`} onClick={() => mic.start(lang === "en" ? "en" : "th")}
                disabled={mic.state === "final"} aria-label={mic.state === "recording" ? "หยุดพูด" : "กดเพื่อพูด"} title={mic.state === "recording" ? `หยุด (${mic.seconds.toFixed(0)} วิ)` : "กดเพื่อพูด กดอีกครั้งเพื่อหยุด"}>
                <img src="/ui/mic-coral-24.svg" width={24} height={24} alt="" />
              </button>
              <button className="btn coral wide" onClick={() => go()} disabled={busy || !text.trim()}>{busy ? "กำลังแปล…" : "▶ แสดงภาษามือ"}</button>
            </div>
            <div className="examples">
              <span>ลองประโยค:</span>
              {examples.map((x) => <button key={x} onClick={() => { setText(x); asrMeta.current = null; go("text", x); }}>{x}</button>)}
            </div>
            <div className="meta-row"><span>{text.length}/400 · Ctrl+Enter เพื่อแปล</span>{health && <span><span className={`dot ${health.ok ? "" : "off"}`} style={{ display: "inline-block", width: 8, height: 8, marginRight: 6 }} />{health.text}</span>}</div>
            {(error || mic.error) && <p className="err">เกิดข้อผิดพลาด: {error || mic.error}</p>}
          </section>

          <section className="card" style={{ animationDelay: ".08s" }}>
            <div className="head">
              <h2>ลำดับภาษามือ</h2>
              <div className="seg soft" role="radiogroup" aria-label="รูปแบบไวยากรณ์">
                <button className={grammar === "tsl51" ? "on" : ""} onClick={() => { setGrammar("tsl51"); if (ir) go(ir.input.source || "text", text, "tsl51"); }}
                  title="เรียงตามกฎที่สรุปจากชุดข้อมูล TSL51">ไวยากรณ์ TSL</button>
                <button className={grammar === "thai" ? "on" : ""} onClick={() => { setGrammar("thai"); if (ir) go(ir.input.source || "text", text, "thai"); }}
                  title="ท่าเดียวกัน แต่เรียงตามลำดับคำภาษาไทย">ลำดับไทย</button>
              </div>
            </div>
            {ir ? (
              <>
                <div className="glosses" key={a.current?.motion.q.length}>
                  {ir.units.map((u, i) => (
                    <button key={i} className={`gloss k-${u.kind} ${u.role === "wh" ? "wh" : ""} ${a.unit === i ? "now" : ""}`} onClick={() => a.seekUnit(i)}
                      style={{ animationDelay: `${i * 0.04}s` }} title={`${KIND_TH[u.kind]} · แปล ${pct(u.confidence.translation)} · ท่าทาง ${pct(u.confidence.motion)}${u.note ? "\n" + u.note : ""}`}>
                      <b>{u.gloss}</b><small>{sub(u, a.unit === i)}</small>
                    </button>
                  ))}
                </div>
                <p className="order-thai">ไทย: {ir.thai} {types.length > 0 && <span className="stype">{types.map((t) => TYPE_TH[t] ?? t).join(" · ")}</span>}</p>
              </>
            ) : <p className="note">ลำดับท่าจะแสดงที่นี่ — ภาษามือไทยเรียงคำต่างจากภาษาไทย เช่น เวลา → ผู้ทำ → กรรม → กริยา → คำถาม</p>}
          </section>

          {ir && (
            <details className="more">
              <summary>รายละเอียดการแปล · ความมั่นใจ · ที่มาของท่า</summary>
              <div className="more-body">
                <div className="conf">
                  <label>ความมั่นใจการแปล <span>{pct(ir.confidence.translation)}</span></label>
                  <div className="bar"><i className={ir.confidence.translation < .5 ? "lo" : ir.confidence.translation < .8 ? "mid" : ""} style={{ width: pct(ir.confidence.translation) }} /></div>
                </div>
                <div className="conf">
                  <label>ความแม่นยำท่าทาง (เทียบคลิปต้นฉบับ) <span>{pct(ir.confidence.motion)}</span></label>
                  <div className="bar"><i className={ir.confidence.motion < .5 ? "lo" : ir.confidence.motion < .8 ? "mid" : ""} style={{ width: pct(ir.confidence.motion) }} /></div>
                </div>
                <p className="note">มีท่าจริงในคลัง {ir.confidence.signs}/{ir.confidence.units} หน่วย ({pct(ir.confidence.coverage)}) · ยาว {ir.confidence.duration_s ?? "–"} วิ{ir.confidence.t_total_ms !== undefined ? ` · ประมวลผล ${ir.confidence.t_total_ms} ms` : ""}</p>
                {ir.warnings.length > 0 && <ul className="warn-list">{ir.warnings.map((w) => <li key={w}>{w}</li>)}</ul>}
                <ul className="warn-list" style={{ opacity: .85 }}>
                  {ir.clauses.flatMap((c) => (c.moves.length ? c.moves : ["ลำดับเดิมถูกต้องแล้ว"]).map((m) => <li key={c.text + m} style={{ background: "var(--rose-50)", color: "var(--rose-ink)" }}>“{c.text}”: {m}</li>))}
                </ul>
                <div className="row">
                  <button className="mini-btn" onClick={() => setEdit(edit ? null : ir.units.map((u) => ({ ...u })))}>✎ แก้ลำดับ/คำ</button>
                  <button className="mini-btn" onClick={exportVrma} title="VRM Animation — เปิดใน Unity (UniVRM) หรือ three-vrm">⬇ .vrma</button>
                  <button className="mini-btn" onClick={() => setShowIR(!showIR)}>IR (JSON)</button>
                </div>
                {edit && <Editor units={edit} setUnits={setEdit} onRender={rerender} />}
                {showIR && <pre className="ir">{JSON.stringify(ir, null, 1)}</pre>}
                {ir.units.map((u, i) => {
                  const p = u.provenance ?? {};
                  return (
                    <div key={i} className={`prov-row o-${p.origin ?? "x"}`}>
                      <span><b>{u.gloss}</b> · {ORIGIN_TH[p.origin ?? ""] ?? p.origin ?? "–"}{u.sense?.ambiguous ? " · ความหมายกำกวม" : ""}</span>
                      {p.dataset && <span>ชุดข้อมูล {p.dataset} · คลิป {p.clip_id} · ผู้แสดง {p.signer}{p.mirrored ? " · กลับซ้าย-ขวา" : ""}</span>}
                      {p.licence && <span>สิทธิ์การใช้: {p.licence}</span>}
                    </div>
                  );
                })}
              </div>
            </details>
          )}
        </div>

        <div className="rev-col avatar-col">
          <AvatarStage a={a} hostRef={hostRef} />
          {segs.length ? (
            <div className="timeline" role="group" aria-label="ลำดับท่า" onClick={(e) => {
              if ((e.target as HTMLElement).classList.contains("seg-b")) return;
              const r = e.currentTarget.getBoundingClientRect(); a.seek(((e.clientX - r.left) / r.width) * a.duration());
            }}>
              {segs.map((s, i) => (
                <button key={i} className={`seg-b k-${s.kind} ${a.unit === s.unit ? "on" : ""}`} style={{ flexGrow: Math.max(1, s.end - s.start) }}
                  onClick={() => a.seek(s.start / (a.current?.motion.fps ?? 25))}>{s.label}</button>
              ))}
              <div className="head" ref={a.headRef} style={{ left: `${(0 / frames) * 100}%` }} />
            </div>
          ) : <div className="timeline empty">เส้นเวลาของท่าจะแสดงเมื่อแปลแล้ว</div>}
          <div className="player">
            <button className="play" onClick={() => { if (!a.toggle()) go(); }} aria-label={a.playing ? "หยุดชั่วคราว" : "เล่น"}>
              {a.playing ? <img src="/ui/pause-white.svg" width={22} height={22} alt="" />
                : <svg viewBox="0 0 24 24" fill="#fff"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" /></svg>}
            </button>
            <button className="replay" onClick={a.replay} aria-label="เล่นซ้ำ"><img src="/ui/replay-coral.svg" width={22} height={22} alt="" /></button>
            <div className="seg" role="radiogroup" aria-label="ความเร็ว">
              {[0.5, 0.75, 1, 1.25].map((s) => <button key={s} className={a.speed === s ? "on" : ""} onClick={() => a.setSpeed(s)} role="radio" aria-checked={a.speed === s}>{s}×</button>)}
            </div>
            <div className="right">
              <label><input type="checkbox" checked={a.loop} onChange={(e) => a.setLoop(e.target.checked)} /> วนซ้ำ</label>
              <div className="seg soft">
                <button className={a.framing === "half" ? "on" : ""} onClick={() => a.setFraming("half")}>ครึ่งตัว</button>
                <button className={a.framing === "close" ? "on" : ""} onClick={() => a.setFraming("close")}>ใกล้</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

/** reorder / remove / add signs, then render again on the avatar */
function Editor({ units, setUnits, onRender }: { units: Unit[]; setUnits: (u: Unit[]) => void; onRender: (u: Unit[]) => void }) {
  const [add, setAdd] = useState("");
  const [sugg, setSugg] = useState<string[]>([]);
  const timer = useRef(0);
  const move = (i: number, d: number) => { const j = i + d; if (j < 0 || j >= units.length) return; const u = [...units]; [u[i], u[j]] = [u[j], u[i]]; setUnits(u); };
  return (
    <div className="more-body">
      <div className="edit-list">
        {units.map((u, i) => (
          <span key={i}><button onClick={() => move(i, -1)} aria-label="ย้ายไปซ้าย">◀</button>{u.gloss}<button onClick={() => move(i, 1)} aria-label="ย้ายไปขวา">▶</button>
            <button onClick={() => setUnits(units.filter((_, k) => k !== i))} aria-label="ลบ">×</button></span>
        ))}
      </div>
      <div className="row">
        <input value={add} list="t2s-sugg" placeholder="เพิ่มคำ (ค้นหาในคลังท่า)" onChange={(e) => {
          const q = e.target.value; setAdd(q); clearTimeout(timer.current);
          timer.current = window.setTimeout(async () => { if (q.trim()) setSugg((await api.lookup(q.trim())).items.map((x) => x.text)); }, 180);
        }} />
        <datalist id="t2s-sugg">{sugg.map((s) => <option key={s} value={s} />)}</datalist>
        <button className="mini-btn" onClick={() => {
          const w = add.trim(); if (!w) return;
          setUnits([...units, { gloss: w, kind: "sign", concept: null, source_words: [w], match: "lemma", role: null, letters: [], clip: null, manual: null, nmm: {},
            confidence: { translation: 1 }, timing: null, note: "added by user" }]);
          setAdd("");
        }}>+ เพิ่ม</button>
        <button className="btn coral sm" onClick={() => onRender(units)}>แสดงใหม่</button>
      </div>
    </div>
  );
}

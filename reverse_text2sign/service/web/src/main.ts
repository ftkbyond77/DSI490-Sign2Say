// ThaiSLM Reverse — text / speech → Thai Sign Language on a 3D avatar.
// Left: input (text or microphone), the TSL structure (IR) with its confidences and fallbacks, gloss editor. Right: the avatar.
import "./style.css";
import { api, type AvatarEntry, type IR, type Result, type Unit } from "./api";
import { Avatar, type Pose } from "./avatar";
import { Motion, Player, sample } from "./player";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const h = (tag: string, cls = "", text = "") => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
};

document.querySelector("#app")!.innerHTML = `
<header class="top">
  <div class="brand"><span class="logo">🤟</span><div><b>ThaiSLM Reverse</b><small>ข้อความ / เสียง → ภาษามือไทย</small></div></div>
  <button id="theme" class="theme-toggle" role="switch" aria-checked="false" title="โหมดกลางวัน / กลางคืน" aria-label="โหมดกลางคืน">
    <svg class="ic sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    <svg class="ic moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
    <span class="knob"></span>
  </button>
</header>
<main class="layout">
  <section class="panel input-panel">
    <div class="card">
      <div class="row between">
        <h2>ข้อความที่ต้องการแปล</h2>
        <div class="seg" id="lang">
          <button data-v="auto" class="on">อัตโนมัติ</button><button data-v="th">ไทย</button><button data-v="en">English</button>
        </div>
      </div>
      <textarea id="text" rows="3" maxlength="400" placeholder="พิมพ์ภาษาไทยหรืออังกฤษ เช่น “คุณชื่ออะไร” แล้วกด แปลเป็นภาษามือ"></textarea>
      <div class="row between">
        <div class="row gap">
          <button id="mic" class="btn ghost" title="กดเพื่อพูด กดอีกครั้งเพื่อหยุด"><span class="mic-ic">🎤</span><span id="mic-l">พูด</span></button>
          <span id="count" class="muted">0/400</span>
        </div>
        <button id="go" class="btn primary">แปลเป็นภาษามือ ▶</button>
      </div>
      <div id="examples" class="chips small"></div>
    </div>

    <div class="card" id="result" hidden>
      <div class="row between"><h2>โครงสร้างภาษามือ (TSL)</h2><span id="stype" class="badge"></span></div>
      <div id="glosses" class="glosses"></div>
      <div class="order"><span class="muted">ไทย:</span> <span id="thai"></span></div>
      <details id="why"><summary>ทำไมเรียงแบบนี้ (กฎไวยากรณ์ที่ใช้)</summary><ul id="moves"></ul></details>
      <div class="conf">
        <div><label>ความมั่นใจการแปล <span id="c-tr"></span></label><div class="bar"><i id="b-tr"></i></div></div>
        <div><label>ความแม่นยำท่าทาง (เทียบคลิปต้นฉบับ) <span id="c-mo"></span></label><div class="bar"><i id="b-mo"></i></div></div>
        <div class="muted small" id="cov"></div>
      </div>
      <ul id="warn" class="warn"></ul>
      <div class="row between actions">
        <div class="row gap">
          <button id="edit" class="btn ghost small">✎ แก้ลำดับ/คำ</button>
          <button id="vrma" class="btn ghost small" title="VRM Animation — เปิดใน Unity (UniVRM) หรือ three-vrm">⬇ .vrma (Unity)</button>
          <button id="irbtn" class="btn ghost small">IR (JSON)</button>
        </div>
        <span id="timing" class="muted small"></span>
      </div>
      <div id="editor" hidden>
        <div id="edit-list" class="chips"></div>
        <div class="row gap"><input id="add" placeholder="เพิ่มคำ (ค้นหาในคลังท่า)" autocomplete="off" list="sugg"/><datalist id="sugg"></datalist>
          <button id="add-b" class="btn ghost small">+ เพิ่ม</button><button id="rerender" class="btn primary small">แสดงใหม่</button></div>
      </div>
      <pre id="irjson" hidden></pre>
    </div>
    <div class="statusbar"><span id="st-dot" class="dot"></span><span id="st-text">กำลังเชื่อมต่อ…</span>
      <a class="fwd" href="http://localhost:8080" target="_blank" rel="noopener">ภาษามือ → ข้อความ (8080) ↗</a></div>
  </section>

  <section class="panel stage-panel">
    <div class="stage" id="stage">
      <div class="caption"><div id="cur" class="cur"></div><div id="cur-sub" class="cur-sub"></div></div>
      <div id="loading" class="loading"><div class="spin"></div><span id="load-t">กำลังโหลดตัวละคร…</span></div>
      <aside class="cast" id="cast" hidden>
        <button class="cast-head" id="cast-toggle" aria-expanded="true" aria-controls="cast-list" title="ย่อ / ขยาย">
          <span>ตัวละคร</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
            stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg></button>
        <div class="cast-list" id="cast-list" role="radiogroup" aria-label="เลือกตัวละคร"></div>
      </aside>
    </div>
    <div class="timeline" id="timeline"><div class="segs" id="segs"></div><div class="head" id="head"></div></div>
    <div class="controls">
      <button id="play" class="btn round" title="เล่น / หยุด">▶</button>
      <button id="replay" class="btn ghost" title="เล่นซ้ำ">⟲</button>
      <label class="row gap small">ความเร็ว
        <select id="speed"><option value="0.5">0.5×</option><option value="0.75">0.75×</option><option value="1" selected>1×</option>
          <option value="1.25">1.25×</option><option value="1.5">1.5×</option></select></label>
      <label class="row gap small"><input type="checkbox" id="loop"/> วนซ้ำ</label>
      <div class="seg small" id="framing"><button data-v="half" class="on">ครึ่งตัว</button><button data-v="close">ใกล้</button></div>
      <span id="fps" class="muted small"></span>
    </div>
  </section>
</main>`;

// ------------------------------------------------------------------------------------------------ day / night theme
// light is the default; the choice is remembered in this browser only (index.html applies it before the first paint)
function applyTheme(t: "light" | "dark") {
  document.documentElement.dataset.theme = t;
  const b = $("#theme");
  b.setAttribute("aria-checked", String(t === "dark"));
  b.setAttribute("aria-label", t === "dark" ? "โหมดกลางวัน" : "โหมดกลางคืน");
  try { localStorage.setItem("t2s-theme", t); } catch { /* storage blocked: the switch still works for this visit */ }
}
applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light");
$("#theme").onclick = () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

// ------------------------------------------------------------------------------------------------ state
const avatar = new Avatar($("#stage"));
const player = new Player();
let restPose: Pose | null = null;
let current: Result | null = null;
let lang = "auto";
let busy = false;
const KIND_TH: Record<string, string> = { sign: "ท่ามือ", fingerspell: "สะกดนิ้ว", text: "ไม่มีท่า (แสดงข้อความ)",
  phrase: "ประโยคบันทึกจากผู้ใช้ภาษามือจริง" };
const TYPE_TH: Record<string, string> = { wh_question: "คำถาม (ใคร/อะไร/ที่ไหน…)", yn_question: "คำถาม ใช่/ไม่ใช่", negative: "ปฏิเสธ",
  statement: "บอกเล่า", greeting: "ทักทาย" };
const LOC_TH: Record<string, string> = { head: "ระดับศีรษะ", face: "ระดับใบหน้า", chest: "ระดับอก", low: "ระดับเอว" };
const MATCH_TH: Record<string, string> = { lemma: "ตรงคำ", gloss: "ตรงคำ", alias: "ชื่อเรียกอื่น", group: "คำพ้องความหมาย (ตรวจแล้ว)",
  ttrs_synonym: "คำพ้อง (พจนานุกรม TTRS)", pronoun: "สรรพนาม → ท่าชี้", compound: "แยกคำประสม", aspect: "คำบอกการณ์ (แล้ว)",
  fingerspell: "สะกดนิ้ว", none: "ไม่มีในคลัง", english: "อังกฤษ", curated: "คำพ้อง (คัดเลือก)", phrase: "ประโยคจริง" };

// ------------------------------------------------------------------------------------------------ characters (vrm/ catalogue)
// every *.vrm in reverse_text2sign/vrm/ is listed by the API (default.vrm = "Default", the others "Model 1…n"); the motion is
// fitted to the chosen body server-side, so switching re-requests the current sentence for the new character
const FALLBACK_AVATAR: AvatarEntry = { id: null, label: "Default", file: "avatar.vrm", title: "", author: "", vrm_version: "0",
  url: "/avatar.vrm", thumb: null, bytes: 0, scale: 1, optimised: true };
let cast: AvatarEntry[] = [];
let avatarId: string | null = null;
let switching = false;

function setRest(rp: Motion) {
  avatar.bind(rp.bones, rp.morphs);
  restPose = { q: new Float32Array(rp.B * 4), m: new Float32Array(rp.M) };
  sample(rp, 0, restPose);
  if (player.motion) avatar.bind(player.motion.bones, player.motion.morphs);
}

function buildCast() {
  const box = $("#cast"), list = $("#cast-list");
  list.innerHTML = "";
  if (cast.length < 2) { box.hidden = true; return; }
  for (const c of cast) {
    const b = h("button", "cast-item");
    b.setAttribute("role", "radio");
    b.dataset.id = c.id ?? "";
    b.title = [c.label, c.title, c.author && `โดย ${c.author}`].filter(Boolean).join(" · ");
    const th = h("span", "cast-thumb");
    if (c.thumb) {
      const img = document.createElement("img");
      img.src = c.thumb; img.alt = ""; img.loading = "lazy"; img.decoding = "async";
      th.appendChild(img);
    } else th.appendChild(h("span", "cast-initial", c.label === "Default" ? "D" : c.label.replace(/\D+/g, "") || "?"));
    th.appendChild(h("span", "cast-spin"));
    b.append(th, h("span", "cast-name", c.label));
    b.onclick = () => switchAvatar(c);
    list.appendChild(b);
  }
  markCast();
  // a phone shows only the chosen character until the box is opened (re-decided when the window crosses the phone width)
  setCastOpen(!isPhone());
  box.hidden = false;
}
const PHONE = window.matchMedia("(max-width: 640px)");
const isPhone = () => PHONE.matches && window.innerWidth > 0;
PHONE.addEventListener("change", () => setCastOpen(!isPhone()));

function markCast(loadingId?: string | null) {
  document.querySelectorAll<HTMLButtonElement>("#cast-list .cast-item").forEach((b) => {
    const on = (b.dataset.id || null) === avatarId;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
    b.classList.toggle("loading", loadingId !== undefined && (b.dataset.id || null) === loadingId);
  });
}

function setCastOpen(open: boolean) {
  $("#cast").classList.toggle("collapsed", !open);
  $("#cast-toggle").setAttribute("aria-expanded", String(open));
}
$("#cast-toggle").onclick = () => setCastOpen($("#cast").classList.contains("collapsed"));

async function switchAvatar(c: AvatarEntry) {
  if (switching) return;
  const phone = isPhone();
  if (c.id === avatarId) {                      // the chosen one: opens a collapsed box (and closes it again on a phone)
    if ($("#cast").classList.contains("collapsed")) setCastOpen(true);
    else if (phone) setCastOpen(false);
    return;
  }
  switching = true;
  markCast(c.id);
  try {
    // character, idle pose and the current sentence (fitted to the new body) arrive together, then swap in one frame
    const [vrm, rest, r] = await Promise.all([
      avatar.fetch(c.url), api.rest(c.id), current ? api.render(current.ir, 1.0, c.id) : Promise.resolve(null),
    ]);
    avatar.show(vrm);
    avatarId = c.id;
    if (r && current) {
      const t = player.t, playing = player.playing;
      current = { ir: current.ir, motion: r.motion };
      player.load(new Motion(r.motion));
      player.seek(t);
      if (playing) player.play();
    }
    setRest(new Motion(rest));
    try { localStorage.setItem("t2s-avatar", c.id ?? ""); } catch { /* storage blocked */ }
    if (phone) setCastOpen(false);
  } catch (e) {
    showError("เปลี่ยนตัวละครไม่สำเร็จ: " + (e as Error).message);
  } finally {
    switching = false;
    markCast();
  }
}

function setStatus(ok: boolean, text: string) {
  $("#st-dot").className = "dot " + (ok ? "ok" : "bad");
  $("#st-text").textContent = text;
}

// ------------------------------------------------------------------------------------------------ boot
(async () => {
  try {
    const hl = await api.health();
    setStatus(hl.ok, `พร้อม · คลังท่า ${hl.signs.toLocaleString()} คำ${hl.english ? " · EN" : ""}`);
  } catch { setStatus(false, "API ไม่ตอบสนอง"); }
  api.examples().then(({ examples }) => {
    const box = $("#examples");
    for (const ex of examples) {
      const b = h("button", "chip", ex);
      b.onclick = () => { ($("#text") as HTMLTextAreaElement).value = ex; updateCount(); go(); };
      box.appendChild(b);
    }
  }).catch(() => {});
  try {
    try { cast = (await api.avatars()).avatars; } catch { cast = []; }
    if (!cast.length) cast = [FALLBACK_AVATAR];                   // API without a catalogue: the bundled library avatar
    let saved: string | null = null;
    try { saved = localStorage.getItem("t2s-avatar"); } catch { /* storage blocked */ }
    const first = cast.find((c) => c.id === saved) ?? cast[0];
    const [vrm, rest] = await Promise.all([
      avatar.fetch(first.url, (f) => ($("#load-t").textContent = `กำลังโหลดตัวละคร… ${Math.round(f * 100)}%`)),
      api.rest(first.id),
    ]);
    avatar.show(vrm);
    avatarId = first.id;
    setRest(new Motion(rest));
    buildCast();
  } catch (e) {
    $("#load-t").textContent = "โหลดตัวละครไม่สำเร็จ: " + (e as Error).message;
    return;
  }
  $("#loading").hidden = true;
  requestAnimationFrame(loop);
})();

// ------------------------------------------------------------------------------------------------ render loop
let last = performance.now(), fpsAcc = 0, fpsN = 0, blinkAt = performance.now() + 2500;
const idle: Pose = { q: new Float32Array(0), m: new Float32Array(0) };
function loop(now: number) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 1) { $("#fps").textContent = `${Math.round(fpsN / fpsAcc)} fps`; fpsAcc = 0; fpsN = 0; }
  let pose: Pose | null = null;
  const f = player.tick(dt);
  if (player.motion && (player.playing || player.t > 0)) pose = player.pose;
  else if (restPose) {
    if (idle.q.length !== restPose.q.length) { idle.q = new Float32Array(restPose.q); idle.m = new Float32Array(restPose.m); }
    idle.q.set(restPose.q); idle.m.set(restPose.m);
    pose = idle;
  }
  if (pose) {
    // life: breathing on the chest, natural blinks when the motion's own face is not blinking
    const br = Math.sin(now / 1000 * 2 * Math.PI / 4.2) * 0.006;
    const ci = 1 * 4; // "chest" is bone #1 in rig order (spine, chest, …): a ~0.7° breathing sway about its X axis
    const qx = pose.q[ci], qw = pose.q[ci + 3];
    pose.q[ci] = qx + br * qw; pose.q[ci + 3] = qw - br * qx;
    avatar.apply(pose);
    avatar.update(dt);
    const m = pose.m.slice();
    if (now > blinkAt) {
      const u = (now - blinkAt) / 140;
      const w = u < 1 ? Math.sin(u * Math.PI) : 0;
      if (u >= 1) blinkAt = now + 2500 + Math.random() * 3500;
      const il = player.motion?.morphs.indexOf("Fcl_EYE_Close_L") ?? 3, ir = player.motion?.morphs.indexOf("Fcl_EYE_Close_R") ?? 4;
      if (il >= 0) m[il] = Math.max(m[il], w);
      if (ir >= 0) m[ir] = Math.max(m[ir], w);
    }
    avatar.applyMorphs(m);
  } else avatar.update(dt);
  avatar.render();
  if (player.motion && f >= 0) updateTimeline(f);
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------------------------------------ translate
const textEl = $<HTMLTextAreaElement>("#text");
const updateCount = () => ($("#count").textContent = `${textEl.value.length}/400`);
textEl.addEventListener("input", updateCount);
textEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); });
$("#go").onclick = () => go();
document.querySelectorAll<HTMLButtonElement>("#lang button").forEach((b) => (b.onclick = () => {
  lang = b.dataset.v!;
  document.querySelectorAll("#lang button").forEach((x) => x.classList.toggle("on", x === b));
}));

const speedEl = $<HTMLSelectElement>("#speed");
async function go(source = "text") {
  const text = textEl.value.trim();
  if (!text || busy) return;
  busy = true;
  const btn = $<HTMLButtonElement>("#go");
  btn.disabled = true; btn.textContent = "กำลังแปล…";
  try {
    const r = await api.translate(text, lang, 1.0, source, avatarId);
    show(r, true);
  } catch (e) {
    showError((e as Error).message);
  } finally {
    busy = false; btn.disabled = false; btn.textContent = "แปลเป็นภาษามือ ▶";
  }
}

function showError(msg: string) {
  $("#result").hidden = false;
  $("#warn").innerHTML = "";
  const li = h("li", "", "เกิดข้อผิดพลาด: " + msg);
  $("#warn").appendChild(li);
}

function show(r: Result, autoplay: boolean) {
  current = r;
  const mo = new Motion(r.motion);
  avatar.bind(mo.bones, mo.morphs);
  player.load(mo);
  player.speed = parseFloat(speedEl.value);
  player.loop = $<HTMLInputElement>("#loop").checked;
  renderIR(r.ir);
  buildTimeline();
  if (autoplay) { player.play(); $("#play").textContent = "⏸"; }
  if (window.innerWidth < 980) $("#stage").scrollIntoView({ behavior: "smooth", block: "start" });   // phone: the avatar is above
}

function pct(x?: number) { return x === undefined || x === null ? "–" : `${Math.round(x * 100)}%`; }

function unitTip(u: Unit) {
  const lines = [`${KIND_TH[u.kind]} · ${MATCH_TH[u.match] ?? u.match}`];
  if (u.source_words?.length) lines.push(`จากคำ: ${u.source_words.join(" ")}`);
  if (u.clip) lines.push(`คลิป: ${u.clip.dataset} · ผู้แสดง ${u.clip.signer} · ระดับ ${u.clip.tier}${u.clip.mirrored ? " · กลับซ้าย-ขวา (มือถนัดขวา)" : ""}` +
    (u.clip.n_signers > 1 ? ` · มี ${u.clip.n_signers} ผู้แสดง${u.clip.consensus !== null ? ` (ฉันทามติ ${pct(u.clip.consensus)})` : ""}` : ""));
  if (u.manual) lines.push(`มือ: ${u.manual.two_handed ? "สองมือ" : "มือเดียว"} · ตำแหน่ง ${LOC_TH[u.manual.location] ?? u.manual.location} · ${u.manual.movement_kind}`);
  if (u.letters?.length) lines.push(`ตัวอักษร: ${u.letters.join(" · ")}`);
  if (u.nmm && (u.nmm as Record<string, string>).sentence) lines.push(`สีหน้า: คิ้ว ${(u.nmm as Record<string, string>).brows} (${(u.nmm as Record<string, string>).source})`);
  lines.push(`แปล ${pct(u.confidence.translation)} · ท่าทาง ${pct(u.confidence.motion)}`);
  if (u.note) lines.push(u.note);
  return lines.join("\n");
}

function renderIR(ir: IR) {
  $("#result").hidden = false;
  const g = $("#glosses");
  g.innerHTML = "";
  ir.units.forEach((u, i) => {
    const c = h("button", `gloss k-${u.kind}`);
    c.dataset.i = String(i);
    c.innerHTML = `<b></b><small></small>`;
    c.querySelector("b")!.textContent = u.gloss;
    c.querySelector("small")!.textContent = u.kind === "sign" ? (u.match === "lemma" || u.match === "gloss" ? "" : MATCH_TH[u.match] ?? "")
      : u.kind === "phrase" ? "ประโยคจริง" : KIND_TH[u.kind];
    c.title = unitTip(u);
    c.onclick = () => { if (u.timing) { player.seek(u.timing.start_s); player.play(); $("#play").textContent = "⏸"; } };
    g.appendChild(c);
  });
  $("#thai").textContent = `${ir.thai}  →  ${ir.units.map((u) => u.gloss).join(" · ")}`;
  const types = [...new Set(ir.clauses.map((c) => c.type))].filter((t) => t !== "statement");
  $("#stype").textContent = types.map((t) => TYPE_TH[t] ?? t).join(" · ") || "บอกเล่า";
  const mv = $("#moves");
  mv.innerHTML = "";
  for (const c of ir.clauses) for (const m of c.moves.length ? c.moves : ["ลำดับเดิมถูกต้องแล้ว"]) mv.appendChild(h("li", "", `“${c.text}”: ${m}`));
  for (const c of ir.clauses) if (c.nmm) mv.appendChild(h("li", "", `สีหน้า: ${c.nmm.brows === "raise" ? "เลิกคิ้ว" : c.nmm.brows} — ${c.nmm.source === "data" ? "จากข้อมูล" : "กฎ (รอผู้เชี่ยวชาญยืนยัน)"}: ${c.nmm.ref}`));
  const C = ir.confidence;
  $("#c-tr").textContent = pct(C.translation); ($("#b-tr") as HTMLElement).style.width = pct(C.translation);
  $("#c-mo").textContent = pct(C.motion); ($("#b-mo") as HTMLElement).style.width = pct(C.motion);
  ($("#b-tr") as HTMLElement).className = C.translation < 0.5 ? "lo" : C.translation < 0.8 ? "mid" : "";
  ($("#b-mo") as HTMLElement).className = C.motion < 0.5 ? "lo" : C.motion < 0.8 ? "mid" : "";
  $("#cov").textContent = `มีท่าจริงในคลัง ${C.signs}/${C.units} หน่วย (${pct(C.coverage)} ของคำเนื้อหา) · ความยาว ${C.duration_s ?? "–"} วินาที`;
  const w = $("#warn");
  w.innerHTML = "";
  for (const x of ir.warnings) w.appendChild(h("li", "", x));
  $("#timing").textContent = C.t_total_ms !== undefined ? `ประมวลผล ${C.t_total_ms} ms` : "";
  $("#irjson").textContent = JSON.stringify(ir, null, 1);
  renderEditor(ir);
}

// ------------------------------------------------------------------------------------------------ timeline / caption
function buildTimeline() {
  const mo = player.motion!;
  const box = $("#segs");
  box.innerHTML = "";
  for (const s of mo.segments) {
    if (s.kind === "transition") continue;
    const d = h("div", `seg-b k-${s.kind}`, s.label);
    d.style.left = `${(s.start / mo.frames) * 100}%`;
    d.style.width = `${((s.end - s.start) / mo.frames) * 100}%`;
    d.onclick = () => { player.seek(s.start / mo.fps); player.play(); $("#play").textContent = "⏸"; };
    box.appendChild(d);
  }
}

let lastUnit = -2;
function updateTimeline(f: number) {
  const mo = player.motion!;
  ($("#head") as HTMLElement).style.left = `${(f / Math.max(mo.frames - 1, 1)) * 100}%`;
  const seg = mo.segments.find((s) => s.kind !== "transition" && f >= s.start && f < s.end);
  const ui = seg ? seg.unit : -1;
  if (ui !== lastUnit) {
    lastUnit = ui;
    const u = ui >= 0 && current ? current.ir.units[ui] : null;
    $("#cur").textContent = u ? u.gloss : "";
    $("#cur-sub").textContent = u ? (u.kind === "sign" ? "" : KIND_TH[u.kind]) : "";
    $("#cur").className = "cur" + (u ? ` k-${u.kind}` : "");
    document.querySelectorAll(".gloss").forEach((g) => g.classList.toggle("now", (g as HTMLElement).dataset.i === String(ui)));
  }
  if (!player.playing) $("#play").textContent = "▶";
}

$("#play").onclick = () => {
  if (!player.motion) return go();
  if (player.playing) { player.pause(); $("#play").textContent = "▶"; } else { player.play(); $("#play").textContent = "⏸"; }
};
$("#replay").onclick = () => { if (player.motion) { player.seek(0); player.play(); $("#play").textContent = "⏸"; } };
speedEl.onchange = () => (player.speed = parseFloat(speedEl.value));
$<HTMLInputElement>("#loop").onchange = (e) => (player.loop = (e.target as HTMLInputElement).checked);
document.querySelectorAll<HTMLButtonElement>("#framing button").forEach((b) => (b.onclick = () => {
  avatar.framing = b.dataset.v as "half" | "close";
  avatar.frame();
  document.querySelectorAll("#framing button").forEach((x) => x.classList.toggle("on", x === b));
}));
$("#timeline").addEventListener("click", (e) => {
  if ((e.target as HTMLElement).classList.contains("seg-b") || !player.motion) return;
  const r = ($("#timeline") as HTMLElement).getBoundingClientRect();
  player.seek(((e.clientX - r.left) / r.width) * player.motion.duration);
});

// ------------------------------------------------------------------------------------------------ IR tools
$("#irbtn").onclick = () => { const p = $("#irjson"); p.hidden = !p.hidden; };
$("#vrma").onclick = async () => {
  if (!current) return;
  try {
    const blob = await api.vrma(current.ir, 1.0, avatarId);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "signing.vrma";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  } catch (e) { showError((e as Error).message); }
};

let editUnits: Unit[] = [];
function renderEditor(ir: IR) {
  editUnits = ir.units.map((u) => ({ ...u }));
  drawEditor();
}
function drawEditor() {
  const box = $("#edit-list");
  box.innerHTML = "";
  editUnits.forEach((u, i) => {
    const c = h("span", `chip edit k-${u.kind}`);
    const l = h("button", "mini", "◀"), r = h("button", "mini", "▶"), x = h("button", "mini", "×");
    l.onclick = () => { if (i > 0) { [editUnits[i - 1], editUnits[i]] = [editUnits[i], editUnits[i - 1]]; drawEditor(); } };
    r.onclick = () => { if (i < editUnits.length - 1) { [editUnits[i + 1], editUnits[i]] = [editUnits[i], editUnits[i + 1]]; drawEditor(); } };
    x.onclick = () => { editUnits.splice(i, 1); drawEditor(); };
    c.append(l, h("b", "", u.gloss), r, x);
    box.appendChild(c);
  });
}
$("#edit").onclick = () => { const e = $("#editor"); e.hidden = !e.hidden; };
const addEl = $<HTMLInputElement>("#add");
let lookT = 0;
addEl.addEventListener("input", () => {
  clearTimeout(lookT);
  lookT = window.setTimeout(async () => {
    const q = addEl.value.trim();
    if (!q) return;
    const { items } = await api.lookup(q);
    const dl = $("#sugg");
    dl.innerHTML = "";
    for (const it of items) { const o = document.createElement("option"); o.value = it.text; dl.appendChild(o); }
  }, 180);
});
$("#add-b").onclick = () => {
  const w = addEl.value.trim();
  if (!w) return;
  editUnits.push({ gloss: w, kind: "sign", concept: null, source_words: [w], match: "lemma", role: null, letters: [], clip: null,
    manual: null, nmm: {}, confidence: { translation: 1 }, timing: null, note: "added by user" });
  addEl.value = "";
  drawEditor();
};
$("#rerender").onclick = async () => {
  if (!current) return;
  const ir: IR = { ...current.ir, units: editUnits, clauses: [{ text: current.ir.thai, type: current.ir.clauses[0]?.type ?? "statement",
    negated: false, moves: ["แก้ไขโดยผู้ใช้"], units: editUnits.map((_, i) => i) }] };
  try { show(await api.render(ir, 1.0, avatarId), true); } catch (e) { showError((e as Error).message); }
};

// ------------------------------------------------------------------------------------------------ microphone → ASR
let rec: MediaRecorder | null = null, chunks: Blob[] = [], recT0 = 0, recTimer = 0;
$("#mic").onclick = async () => {
  if (rec && rec.state === "recording") { rec.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, noiseSuppression: true, echoCancellation: true } });
    rec = new MediaRecorder(stream, MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? { mimeType: "audio/webm;codecs=opus" } : undefined);
    chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recTimer);
      $("#mic").classList.remove("rec");
      $("#mic-l").textContent = "กำลังถอดเสียง…";
      try {
        const asrLang = lang === "en" ? "en" : "th";
        const r = await api.asr(new Blob(chunks, { type: rec!.mimeType || "audio/webm" }), asrLang);
        textEl.value = r.text; updateCount();
        $("#mic-l").textContent = "พูด";
        if (r.text.trim()) go("speech");
      } catch (e) { $("#mic-l").textContent = "พูด"; showError("ถอดเสียงไม่สำเร็จ: " + (e as Error).message); }
    };
    rec.start();
    recT0 = performance.now();
    $("#mic").classList.add("rec");
    recTimer = window.setInterval(() => {
      const s = (performance.now() - recT0) / 1000;
      $("#mic-l").textContent = `หยุด (${s.toFixed(0)} วิ)`;
      if (s > 30) rec!.stop();
    }, 250);
  } catch (e) { showError("เปิดไมโครโฟนไม่ได้: " + (e as Error).message); }
};

// ------------------------------------------------------------------------------------------------ automation hook (evaluation renders, UI tests)
(window as unknown as { __t2s: unknown }).__t2s = {
  ready: () => !!restPose,
  load: (r: Result) => show(r, false),
  duration: () => player.motion?.duration ?? 0,
  /** per-frame JavaScript cost (timeline sample + bone / morph writes + VRM update), GPU excluded: ms per frame */
  bench: (n = 1000) => {
    if (!player.motion) return -1;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) {
      player.seek((i / n) * player.motion.duration);
      player.tick(0);
      avatar.apply(player.pose);
      avatar.update(1 / 60);
      avatar.applyMorphs(player.pose.m);
    }
    return (performance.now() - t0) / n;
  },
  /** render the pose at time t (s) now and return a JPEG data URL of the stage */
  capture: (t: number, q = 0.85) => {
    player.pause();
    player.seek(t);
    player.tick(0);
    avatar.apply(player.pose);
    avatar.update(1 / 60);
    avatar.applyMorphs(player.pose.m);
    avatar.renderer.setClearColor(0xe4e9f2, 1);           // opaque for the snapshot (the live stage is transparent over CSS)
    avatar.render();
    const url = avatar.renderer.domElement.toDataURL("image/jpeg", q);
    avatar.renderer.setClearColor(0x000000, 0);
    return url;
  },
};

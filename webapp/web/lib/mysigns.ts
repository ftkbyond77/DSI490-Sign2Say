// Personal sign memory ("ท่าของฉัน"): when the user confirms or corrects a word on a result card, that segment's embedding is kept
// here (this browser only) and sent with every request; the server averages a word's examples into one prototype and adds it to the
// bank for this user's requests only. The shared model and bank never change.
export type MySign = { word: string; z: string; t: number };

const KEY = "thaislm.mysigns";
const PER_WORD = 3;
const MAX = 600;

export function loadSigns(): MySign[] {
  try { return JSON.parse(localStorage.getItem(KEY) || "[]") as MySign[]; } catch { return []; }
}

function store(list: MySign[]) {
  try { localStorage.setItem(KEY, JSON.stringify(list.slice(-MAX))); } catch { /* storage full / private mode */ }
}

/** Save one example of `word` (keeps the newest PER_WORD examples per word). */
export function saveSign(word: string, z: string) {
  const list = loadSigns().filter((s) => !(s.word === word && s.z === z));
  list.push({ word, z, t: Date.now() });
  const same = list.filter((s) => s.word === word);
  const drop = new Set(same.slice(0, Math.max(0, same.length - PER_WORD)));
  store(list.filter((s) => !drop.has(s)));
}

export function clearSigns() { store([]); }

export function signWords(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of loadSigns()) out[s.word] = (out[s.word] || 0) + 1;
  return out;
}

export function userBank() {
  return loadSigns().map(({ word, z }) => ({ word, z }));
}

"""Context-aware correction of Thai speech recognition output (v2, result_v2.md §7.3) — conservative by construction.

The recogniser's own text is always kept (`raw`); a correction replaces a span of 1–3 recognised words only when ALL of:
  1. sound   the replacement has the same Thai soundex code (lk82) as the span (or one symbol apart) AND is spelled almost
             the same: ≤ 30 % of the characters (≥ 1, ≤ 2) differ — a span of several recognised words only when ONE character
             differs (ร้อง|ให้ → ร้องไห้); the recogniser heard something that sounds AND reads like it;
  2. domain  the replacement is a word the sign translator knows (a dictionary sign, a curated colloquial form, a time / day word,
             a place name) — the correction never invents vocabulary outside what can be signed;
  3. context the Thai National Corpus bigrams make the corrected word sequence (with its neighbours) at least `MARGIN` nats
             more likely than the recognised one (≈ 50×), + MERGE_COST per word merged (fewer words ⇒ fewer bigram factors,
             which would otherwise always favour merging);
  4. doubt   if the recogniser reports token probabilities, the span's weakest character had probability < `DOUBT`.
Each applied change is returned with its evidence (codes, likelihood gain, probability), so the UI can show it and the user can
undo it. It never reorders, adds or deletes words, and never touches names after ชื่อ (they are spelled, not looked up).
Thresholds were set on the development audio only (v1's 96 sentences) and evaluated on held-out audio (result_v2.md §7.3).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

MARGIN = 4.0
MERGE_COST = 4.0
DOUBT = 0.85
MAX_SPAN = 3
NAME_TRIGGER = ("ชื่อ", "นามสกุล", "ชื่อเล่น")


def build_lm(out=None):
    """TNC unigrams + bigrams → compact arrays (word list, sorted int64 pair keys, counts) in models/asr_lm.npz (~25 MB);
    the API loads only this file (pythainlp keeps its full corpus cached in memory: ~0.6 GB per worker)."""
    import numpy as np
    from pythainlp.corpus import tnc
    from .. import paths
    uni = dict(tnc.word_freqs())
    bi = tnc.bigram_word_freqs()
    ids = {w: i for i, w in enumerate(uni)}
    for a, b in bi:
        for w in (a, b):
            if w not in ids:
                ids[w] = len(ids)
    n = len(ids)
    keys = np.fromiter((ids[a] * n + ids[b] for a, b in bi), np.int64, len(bi))
    cnt = np.fromiter(bi.values(), np.int64, len(bi))
    order = np.argsort(keys)
    unic = np.zeros(n, np.int64)
    for w, c in uni.items():
        unic[ids[w]] = c
    out = out or (paths.MODELS / "asr_lm.npz")
    np.savez_compressed(out, words=np.array(list(ids), dtype=object).astype(str), keys=keys[order], cnt=cnt[order].astype(np.int32),
                        uni=unic.astype(np.int32))
    return out


@lru_cache(maxsize=1)
def _lm():
    import numpy as np
    from .. import paths
    f = paths.MODELS / "asr_lm.npz"
    if not f.exists():
        build_lm(f)
    z = np.load(f)
    words = z["words"]
    ids = {w: i for i, w in enumerate(words.tolist())}
    n = len(words)
    keys, cnt = z["keys"], z["cnt"].astype(np.int64)
    left = np.bincount(keys // n, weights=cnt, minlength=n)
    unic = z["uni"].astype(np.float64)
    return ids, n, keys, cnt, left, unic, float(unic.sum())


def _logp(a, b):
    """stupid-backoff log P(b | a) on the TNC (unknown words get a small floor)."""
    import numpy as np
    ids, n, keys, cnt, left, unic, total = _lm()
    ia, ib = ids.get(a), ids.get(b)
    if ia is not None and ib is not None and left[ia] > 0:
        k = ia * n + ib
        j = int(np.searchsorted(keys, k))
        if j < len(keys) and keys[j] == k:
            return math.log(cnt[j] / left[ia])
    return math.log(0.4) + math.log(((unic[ib] if ib is not None else 0) + 0.5) / total)


def _seq_logp(words):
    w = ["<s/>"] + [x for x in words if x.strip()] + ["<s/>"]
    return sum(_logp(a, b) for a, b in zip(w[:-1], w[1:]))


def _code(w):
    from pythainlp.soundex import soundex
    try:
        return soundex(w, engine="lk82")
    except Exception:  # noqa: BLE001
        return None


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def _code_dist(a, b):
    if a is None or b is None or a[0] != b[0] or len(a) != len(b):
        return 9
    return sum(x != y for x, y in zip(a, b))


class Corrector:
    def __init__(self, vocabulary):
        """vocabulary: the words the sign translator can use (dictionary surfaces, curated forms, time / place words)."""
        self.vocab = {w for w in vocabulary if 1 < len(w) <= 14 and re.fullmatch(r"[฀-๿]+", w)}
        self.by_code, self.by_first = {}, {}
        for w in self.vocab:
            self.by_code.setdefault(_code(w), set()).add(w)
        for c in self.by_code:
            self.by_first.setdefault(c[:1] if c else "", []).append(c)

    def _candidates(self, span, n_words):
        c0 = _code(span)
        lim = 1 if n_words > 1 else max(1, min(2, int(0.3 * len(span))))
        out = set()
        for code in self.by_first.get(c0[:1] if c0 else "", ()):
            if _code_dist(code, c0) <= 1:
                out |= {w for w in self.by_code[code] if w != span and abs(len(w) - len(span)) <= lim and _lev(w, span) <= lim}
        return out

    def correct(self, text, char_probs=None):
        """→ (corrected text, [change]); char_probs: per-character recogniser probability of `text` (optional)."""
        from pythainlp.tokenize import word_tokenize
        toks = [t for t in word_tokenize(text, engine="newmm", keep_whitespace=True)]
        pos, offs = 0, []
        for t in toks:
            offs.append(pos); pos += len(t)
        words = list(toks)
        changes = []
        i = 0
        while i < len(words):
            if not words[i].strip() or (i > 0 and any(words[k].strip() in NAME_TRIGGER for k in range(max(0, i - 2), i))):
                i += 1; continue
            best = None
            for L in range(1, MAX_SPAN + 1):
                if i + L > len(words) or any(not w.strip() for w in words[i:i + L]):
                    break
                span = "".join(words[i:i + L])
                if (span in self.vocab and L == 1) or not re.fullmatch(r"[฀-๿]+", span):
                    continue                                     # a known word is not "corrected" into another; punctuation /
                                                                 # Latin / digits are left alone
                doubt = None
                if char_probs is not None:
                    a = offs[i]; b = a + len(span)
                    doubt = min(char_probs[a:b]) if b <= len(char_probs) and b > a else None
                    if doubt is not None and doubt >= DOUBT:
                        continue
                ctx_l = [w for w in words[max(0, i - 2):i] if w.strip()]
                ctx_r = [w for w in words[i + L:i + L + 2] if w.strip()]
                base = _seq_logp(ctx_l + [span] + ctx_r) if L == 1 else _seq_logp(ctx_l + [w for w in words[i:i + L]] + ctx_r)
                for c in self._candidates(span, L):
                    gain = _seq_logp(ctx_l + [c] + ctx_r) - base - MERGE_COST * (L - 1)
                    if gain >= MARGIN and (best is None or gain > best[2]):
                        best = (L, c, gain, span, doubt)
            if best:
                L, c, gain, span, doubt = best
                changes.append(dict(original=span, corrected=c, gain_nats=round(gain, 1), sound=[_code(span), _code(c)],
                                    probability=None if doubt is None else round(float(doubt), 3)))
                words[i:i + L] = [c]
                offs[i:i + L] = [offs[i]]
            i += 1
        return "".join(words), changes

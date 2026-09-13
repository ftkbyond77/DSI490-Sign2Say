"""Export what the v4 Sign model knows → vocab/corpus/{vocab.json, sentence.json}.

  python scripts/export_vocab.py

vocab.json/.csv    every entry of the deployed vocabulary bank (artifacts/v4/bank.npz) = the only words the model can output
sentence.json/.csv the model does not store sentences as units; this file lists (1) multi-word phrase entries that are retrievable
              as a single sign, (2) the YouTube lesson/sentence videos words were learned from (weak supervision),
              (3) evaluation-only sentences (never trained) with their vocabulary coverage.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modules.utils import ART, read_json

OUT = ROOT / "vocab" / "corpus"
V4, REP = ART / "v4", ART / "reports"
POS_EN = {"คำนาม": "noun", "คำกริยา": "verb", "คำวิเศษณ์": "adverb/adjective", "คำสรรพนาม": "pronoun", "คำบุพบท": "preposition",
          "คำสันธาน": "conjunction", "คำอุทาน": "interjection"}
TEST_REFS = {"ฉันปลอบเพื่อนร้องไห้": ["ฉัน", "ปลอบใจ", "เพื่อน", "ร้องไห้"], "ไปทานข้าวด้วยกันมั้ย": ["ไป", "กิน", "ข้าว", "ด้วยกัน", "ไหม"]}
SYN = {"ทาน": "กิน", "รับประทาน": "กิน", "มั้ย": "ไหม", "ปลอบ": "ปลอบใจ", "ร่วมกัน": "ด้วยกัน"}


def tokens(text):
    from pythainlp.tokenize import word_tokenize
    return [t for t in word_tokenize(text, engine="newmm") if t.strip() and t.strip() not in {"/", ",", "(", ")", "-", "."}]


def kind_of(word, toks, categories=()):
    if any("ประโยค" in c for c in categories) and len(toks) >= 2:
        return "sentence"
    if len(toks) <= 1:
        return "word"
    if len(toks) == 2 and " " not in word:
        return "compound"
    return "phrase"


def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else v


def main():
    b = np.load(V4 / "bank.npz")
    bank = pd.DataFrame(dict(word=b["words"], source=b["source"], signer=b["signer"], clip_id=b["clip"]))
    cs = pd.read_parquet(ART / "manifests" / "clips_signers.parquet").set_index("clip_id")
    cv4 = pd.read_parquet(ART / "manifests_v4" / "clips_v4.parquet").set_index("clip_id")
    mined = pd.read_parquet(V4 / "mined.parquet")
    speech = pd.read_parquet(ART / "manifests_v4" / "speech_words.parquet")
    title_df = pd.read_parquet(V4 / "emb_title.parquet")
    bank["src_row"] = bank.groupby("source").cumcount()  # row inside its source table (bank = ttrs ‖ title ‖ mined, in order)
    assert (bank[bank.source == "mined"].word.values == mined.word.values).all() and (bank[bank.source == "title"].word.values == title_df.word.values).all()
    ev = read_json(REP / "v4_eval.json"); sp = read_json(V4 / "spotting.json"); cal = read_json(V4 / "calibration.json")
    cfg = read_json(V4 / "config.json")

    # ------------------------------------------------------------------ vocab.json
    entries = []
    for word, g in bank.groupby("word", sort=True):
        toks = tokens(word)
        ttrs_ids = [c for c in g.clip_id if c in cs.index and cs.loc[c, "source"] == "ttrs"]
        meta = cs.loc[ttrs_ids] if ttrs_ids else None
        pos_th = sorted({p for p in (meta.pos if meta is not None else []) if _clean(p)})
        syn = sorted({s.strip() for v in (meta.synonyms if meta is not None else []) if _clean(v) for s in str(v).replace("/", ",").replace(";", ",").split(",") if s.strip() and s.strip() != word})
        clips = []
        for r in g.itertuples():
            item = dict(clip_id=r.clip_id, source=r.source, signer=r.signer)
            if r.source == "mined":
                mr = mined.iloc[r.src_row]
                item.update(t0=round(float(mr.t0), 2), t1=round(float(mr.t1), 2), encoder_rank=int(mr["rank"]),
                            title=_clean(cv4.loc[r.clip_id, "title"]) if r.clip_id in cv4.index else None)
            elif r.source == "title" and r.clip_id in cv4.index:
                item["title"] = _clean(cv4.loc[r.clip_id, "title"])
            clips.append(item)
        n_sig = int(g.signer.nunique())
        cats = sorted({c for c in (meta.category if meta is not None else []) if _clean(c)})
        entries.append(dict(
            word=word, kind=kind_of(word, toks, cats), tokens=toks,
            pos_th=pos_th, pos=[POS_EN.get(p, p) for p in pos_th],
            categories=cats,
            synonyms=syn,
            ttrs_variants=dict(Counter(v for v in (meta.variant_tag if meta is not None else []) if _clean(v))),
            n_instances=int(len(g)), n_signers=n_sig, sources=dict(Counter(g.source)),
            reliability="multi_signer" if n_sig >= 2 else "single_signer",
            clips=clips))
    kinds = Counter(e["kind"] for e in entries)
    vocab = dict(
        meta=dict(
            generated_at=dt.datetime.now().isoformat(timespec="seconds"),
            description="Every entry the ThaiSLM v4 Sign model can recognise. Recognition = retrieval: a signed segment is embedded "
                        "and compared with the instances of each entry; an entry's score is the mean of its top-2 instance similarities. "
                        "A word that is not listed here can never be output (it is reported as [?] ≈ nearest entry).",
            model=dict(pose="RTMW-X whole-body 133 keypoints @12.5 fps (upsampled 2x)",
                       encoder="Uni-Sign pose encoder, WLASL ISLR checkpoint, mT5 contextual states, mean-pooled (zero-shot)",
                       adapter_used=bool(cfg.get("use_adapter", False))),
            bank="artifacts/v4/bank.npz",
            n_entries=len(entries), n_instances=int(len(bank)), instances_by_source=dict(Counter(bank.source)),
            entries_by_kind=dict(kinds), entries_multi_signer=sum(e["reliability"] == "multi_signer" for e in entries),
            sources_explained=dict(ttrs="Thai sign dictionary (TTRS) clip, label = gloss",
                                   title="YouTube single-sign lesson, label parsed from the video title/filename",
                                   mined="instance mined from a YouTube lesson: the word was spoken (Thai captions, word timestamps) and the "
                                         "encoder ranked it top-3 of the whole vocabulary in a nearby window"),
            kinds_explained=dict(word="1 Thai token", compound="2 tokens, no space (e.g. ขับรถยนต์)", phrase="3+ tokens or contains a space (names, noun phrases)",
                                 sentence="TTRS category 'ประโยค': a whole sentence signed and stored as ONE entry"),
            reliability_explained="multi_signer = instances from >= 2 signers (more robust to a new signer); single_signer = one signer only",
            status_rule=dict(accepted=cal["accept"], uncertain=cal["uncertain"], unknown="otherwise → [?] ≈ nearest entry"),
            accuracy=dict(E1_new_signer_full_vocab_test=ev.get("E1|test|ttrs_train+youtube"), E5_youtube_title_clips=ev.get("E5|youtube_title_clips_vs_full_TTRS")),
            spotting=dict(lam=sp["lam"], gap_cost=sp["gap_cost"], min_len_frames=sp["min_len"], candidate_aggregation=sp.get("agg")),
        ),
        words=entries)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ sentence.json
    vocab_set = {e["word"] for e in entries}
    by_word = {e["word"]: e for e in entries}
    def _row(e):
        return dict(text=e["word"], tokens=e["tokens"], n_tokens=len(e["tokens"]), categories=e["categories"], pos_th=e["pos_th"],
                    n_instances=e["n_instances"], n_signers=e["n_signers"], sources=e["sources"], clip_ids=[c["clip_id"] for c in e["clips"]])
    sentences_in_vocab = [_row(e) for e in entries if e["kind"] == "sentence"]
    phrases = [_row(e) for e in entries if e["kind"] == "phrase"]

    lessons = []
    m_by_clip = mined.groupby("clip_id")
    s_by_clip = speech.groupby("clip_id")
    used = cv4[cv4.usable_for == "weak_word"]
    for cid, r in used.iterrows():
        mw = m_by_clip.get_group(cid) if cid in m_by_clip.groups else mined.iloc[:0]
        n_hits = int(len(s_by_clip.get_group(cid))) if cid in s_by_clip.groups else 0
        lessons.append(dict(clip_id=cid, title=_clean(r.title), level=_clean(cs.loc[cid, "level"]) if cid in cs.index else None, subset=r.subset,
                            duration_s=round(float(r.duration_s), 1), spoken_dictionary_word_hits=n_hits,
                            learned_instances=[dict(word=x.word, t0=round(float(x.t0), 2), t1=round(float(x.t1), 2), encoder_rank=int(x["rank"]),
                                                    score=round(float(x.score), 3)) for _, x in mw.sort_values("t0").iterrows()]))
    lessons.sort(key=lambda x: -len(x["learned_instances"]))
    sent_clips = cv4[cv4.subset == "youtube_sentence"]
    sentence_videos = [dict(clip_id=cid, title=_clean(r.title), usable_for=_clean(r.usable_for) or "excluded", exclude_reason=_clean(r.exclude_reason),
                            label=_clean(r.label), label_source=_clean(r.label_source)) for cid, r in sent_clips.iterrows()]

    def coverage(glosses):
        rows = []
        for g in glosses:
            c = SYN.get(g, g)
            hit = c if c in vocab_set else (g if g in vocab_set else None)
            rows.append(dict(gloss=g, in_vocab=hit is not None, entry=hit, n_signers=by_word[hit]["n_signers"] if hit else 0))
        return rows

    lm = read_json(V4 / "lm_eval_sentences.json")
    lm_res = {tuple(x["ref"]): x for x in read_json(REP / "v4_lm_eval.json")["sentences"]}
    eval_sents = [dict(thai=s.get("thai"), glosses=s["glosses"], coverage=coverage(s["glosses"]),
                       result=dict(top1=lm_res.get(tuple(s["glosses"]), {}).get("top1"), llm_chosen=lm_res.get(tuple(s["glosses"]), {}).get("llm")))
                  for s in lm]
    inf = read_json(REP / "v4_infer_final.json")
    test_sents = [dict(thai=k, glosses=v, coverage=coverage(v), segments_found=inf.get(k, {}).get("n_segments"),
                       evidence_gloss=inf.get(k, {}).get("evidence_gloss"), sentence_output=inf.get(k, {}).get("sentence") or None,
                       best_rank_of_reference_words=inf.get(k, {}).get("best_rank"))
                  for k, v in TEST_REFS.items()]
    sentence = dict(
        meta=dict(
            generated_at=dt.datetime.now().isoformat(timespec="seconds"),
            important="The v4 model does NOT learn sentences as units. A sentence is recognised as a sequence of vocabulary entries "
                      "(spotting → one entry per segment) and then composed by the LLM under an evidence guard. Only the phrase "
                      "entries (sentence_entries_in_vocab, phrase_entries_in_vocab) can be matched as a whole — only when signed like the stored clip.",
            counts=dict(sentence_entries=len(sentences_in_vocab), phrase_entries=len(phrases), lesson_videos_used=len(lessons),
                        lesson_videos_with_learned_instances=sum(bool(x["learned_instances"]) for x in lessons),
                        learned_instances_from_lessons=int(len(mined)), sentence_level_videos=len(sentence_videos),
                        evaluation_sentences=len(eval_sents), test_sentences=len(test_sents)),
        ),
        sentence_entries_in_vocab=sentences_in_vocab,
        phrase_entries_in_vocab=phrases,
        lessons_words_were_learned_from=lessons,
        sentence_level_youtube_videos=sentence_videos,
        evaluation_only_sentences=dict(
            note="GPT-written sentences built from vocabulary glosses signed by >= 2 signers; each gloss performed with a real TTRS clip and "
                 "matched against a bank without that signer. Used only to test decoding — never trained on.",
            sentences=eval_sents),
        test_only_sentences=dict(note="data_test/ references; never used for training or tuning", sentences=test_sents),
    )
    (OUT / "sentence.json").write_text(json.dumps(sentence, ensure_ascii=False, indent=1), encoding="utf-8")
    # ------------------------------------------------------------------ simple CSV checklists (utf-8-sig so Excel shows Thai)
    KIND_TH = {"word": "คำ", "compound": "คำประสม", "phrase": "วลี", "sentence": "ประโยค"}
    pd.DataFrame([dict(no=i + 1, word=e["word"], kind=KIND_TH[e["kind"]], category=" / ".join(e["categories"]),
                       examples=e["n_instances"], signers=e["n_signers"],
                       sources=" + ".join(f"{k}:{v}" for k, v in e["sources"].items()),
                       multi_signer="yes" if e["reliability"] == "multi_signer" else "no", checked="")
                  for i, e in enumerate(entries)]).to_csv(OUT / "vocab.csv", index=False, encoding="utf-8-sig")
    rows = [dict(text=x["text"], type="ประโยคในคลังคำ (จับได้ทั้งประโยค)", in_vocab="yes", signers=x["n_signers"], missing_words="") for x in sentences_in_vocab]
    for group, items in (("ประโยคทดสอบ (ไม่ได้ train)", eval_sents), ("data_test (ไม่ได้ train)", test_sents)):
        for x in items:
            miss = [c["gloss"] for c in x["coverage"] if not c["in_vocab"]]
            rows.append(dict(text=x["thai"] or " ".join(x["glosses"]), type=group, in_vocab="all words" if not miss else "partly",
                             signers="", missing_words=", ".join(miss), glosses=" · ".join(x["glosses"])))
    pd.DataFrame(rows, columns=["text", "glosses", "type", "in_vocab", "missing_words", "signers"]).assign(checked="").to_csv(
        OUT / "sentence.csv", index=False, encoding="utf-8-sig")
    print("vocab.json:", len(entries), "entries", dict(kinds), "| multi-signer", vocab["meta"]["entries_multi_signer"])
    print("sentence.json:", sentence["meta"]["counts"])


if __name__ == "__main__":
    main()

"""English input → Thai, then the Thai → TSL pipeline (TSL is not derived from English word order; the lexicon is Thai).
The translation model (NLLB-200 distilled 600M, local, CPU int8) runs in the reverse-speech service (POST /mt), so the motion API
stays small. The MT output is kept in the IR (`input.mt`) so a reviewer can see what was actually translated.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

SPEECH_URL = os.environ.get("T2S_SPEECH_URL", "http://speech:8002")


# Thai place names NLLB transliterates badly ("Chiang Mai" → "ชิงแมย์"): substituted before translation (the Thai name survives
# translation unchanged), longest first.
GAZETTEER = {"Bangkok": "กรุงเทพ", "Chiang Mai": "เชียงใหม่", "Chiang Rai": "เชียงราย", "Phuket": "ภูเก็ต", "Pattaya": "พัทยา",
             "Khon Kaen": "ขอนแก่น", "Hat Yai": "หาดใหญ่", "Ayutthaya": "อยุธยา", "Nakhon Ratchasima": "นครราชสีมา", "Korat": "โคราช",
             "Udon Thani": "อุดรธานี", "Krabi": "กระบี่", "Hua Hin": "หัวหิน", "Sukhothai": "สุโขทัย", "Lampang": "ลำปาง",
             "Nonthaburi": "นนทบุรี", "Pathum Thani": "ปทุมธานี", "Samut Prakan": "สมุทรปราการ", "Chonburi": "ชลบุรี",
             "Ubon Ratchathani": "อุบลราชธานี", "Nakhon Pathom": "นครปฐม", "Songkhla": "สงขลา", "Surat Thani": "สุราษฎร์ธานี",
             "Thailand": "ประเทศไทย", "Siriraj": "ศิริราช", "Suvarnabhumi": "สุวรรณภูมิ", "Don Mueang": "ดอนเมือง"}


def protect_names(text):
    import re
    for en in sorted(GAZETTEER, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(en)}\b", GAZETTEER[en], text, flags=re.I)
    return text


class LocalNLLB:
    """the same model in-process (offline evaluation / no speech container): T2S_MT_DIR = converted CT2 NLLB directory."""

    def __init__(self, d):
        import ctranslate2
        import sentencepiece as spm
        self.tr = ctranslate2.Translator(str(d), device="cpu", compute_type="int8", intra_threads=4)
        self.sp = spm.SentencePieceProcessor(model_file=str(Path(d) / "sentencepiece.bpe.model"))

    def to_thai(self, text):
        src = ["eng_Latn"] + self.sp.encode(protect_names(text.strip()), out_type=str) + ["</s>"]
        r = self.tr.translate_batch([src], target_prefix=[["tha_Thai"]], beam_size=2, max_decoding_length=200, return_scores=True)
        hyp = r[0].hypotheses[0]
        th = self.sp.decode(hyp[1:] if hyp and hyp[0] == "tha_Thai" else hyp)
        return th, dict(model="nllb-200-distilled-600M (local)", thai=th, score=round(float(r[0].scores[0]), 3))


class Translator:
    def __init__(self, url=SPEECH_URL, timeout=20):
        self.url, self.timeout = url.rstrip("/"), timeout

    @classmethod
    def load(cls):
        d = os.environ.get("T2S_MT_DIR")
        if d and Path(d).exists():
            return LocalNLLB(d)
        return cls()

    def to_thai(self, text):
        req = urllib.request.Request(self.url + "/mt", data=json.dumps({"text": protect_names(text)}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"English → Thai translation unavailable ({type(e).__name__})") from e
        return d["text"], dict(model=d.get("model"), thai=d["text"], score=d.get("score"), ms=d.get("ms"))

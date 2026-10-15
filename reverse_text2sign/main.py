"""ThaiSLM reverse — Thai / English text or speech → Thai Sign Language on a 3D avatar. The only CLI entry point.

  python main.py audit                 Phase 1: quality of every clip that could become avatar motion (data/audit/)
  python main.py sheets                contact sheets per failure reason (look at the noise)
  python main.py select                exemplar per concept (consensus medoid, dictionaries first, eye-checked fill-ins)
  python main.py library               retarget every exemplar → models/motion/ (q.bin, m.bin, index.json) + stroke annotations
  python main.py add-clips             append exemplars missing from the library (e.g. a new harvest) without a full rebuild
  python main.py lexicon               written forms → signable concepts → models/lexicon.json (+ models/asr_lm.npz, TNC bigrams)
  python main.py timing                transition / tempo model from recorded signing → models/timing.json
  python main.py avatar                web-optimised avatar → models/avatar.vrm
  python main.py harvest WORD [WORD…]  targeted YouTube lessons for missing words (download, MediaPipe, QC → cloud_s3/agent_data/reverse_harvest/)
  python main.py translate "TEXT" [--thai]   text → IR (+ timeline length); --thai = Normal grammar mode (Thai word order)
  python main.py vrma "TEXT" out.vrma  export a sentence as VRM Animation (Unity / three-vrm)
  python main.py eval [all|tsl51|conversation|backtranslate|renders]   (v1 report blocks)
  python main.py eval2 [language|motion|nmm|asr|compare]             v2 evaluations (result_v2.md)
  python main.py test                  pytest suite
Production: `docker compose up --build` at the repository root → http://localhost:8081 (this part) and :8080 (forward part).
"""
from __future__ import annotations

import json
import sys

from t2s import paths


def main(argv):
    if not argv:
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "audit":
        from t2s.data import audit
        audit.run()
    elif cmd == "sheets":
        from t2s.data import audit
        print(len(audit.sheets()), "sheets in", audit.OUT / "sheets")
    elif cmd == "select":
        from t2s.data import select
        select.run()
    elif cmd == "library":
        import pandas as pd
        from t2s.motion import library
        from t2s.motion.compose import annotate
        s = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet")
        recs = [dict(key=r.concept, **{k: v for k, v in r._asdict().items() if k not in ("Index", "concept")}) for r in s.itertuples()]
        library.build(recs)
        print("annotated", annotate())
    elif cmd == "add-clips":
        import pandas as pd
        from t2s.motion import library
        from t2s.motion.compose import annotate
        s = pd.read_parquet(paths.WORK / "select" / "exemplars.parquet")
        have = set(library.Library().clips)
        recs = [dict(key=r.concept, **{k: v for k, v in r._asdict().items() if k not in ("Index", "concept")})
                for r in s.itertuples() if r.concept not in have]
        if recs:
            keys, err = library.add(recs)
            print(f"added {len(keys)} clips, {len(err)} errors; annotated", annotate(only_missing=True))
        else:
            print("nothing to add")
    elif cmd == "lexicon":
        from t2s.text import lexicon
        from t2s.text.asr_correct import build_lm
        lexicon.build()
        print("ASR correction language model:", build_lm())
    elif cmd == "timing":
        from t2s.eval import timing
        timing.run()
    elif cmd == "avatar":
        from t2s import avatar
        a, b = avatar.optimise()
        print(f"{a // 1024} KB → {b // 1024} KB: {paths.AVATAR}")
    elif cmd == "harvest":
        from t2s.data import harvest
        c = harvest.candidates(args)
        d = harvest.download(c)
        harvest.extract(d)
        print(harvest.qc(d)[["word", "id", "tier"]].to_string())
    elif cmd == "translate":
        from t2s.pipeline import Engine
        e = Engine()
        thai = "--thai" in args
        ir, mo = e.run(" ".join(a for a in args if a != "--thai"), grammar="thai" if thai else "tsl51")
        print(json.dumps(ir.to_dict(), ensure_ascii=False, indent=1))
        print(f"{mo['frames']} frames = {mo['frames'] / mo['fps']:.1f} s")
    elif cmd == "vrma":
        from t2s.motion.vrma import to_vrma
        from t2s.pipeline import Engine
        ir, mo = Engine().run(args[0])
        open(args[1] if len(args) > 1 else "signing.vrma", "wb").write(to_vrma(mo))
    elif cmd == "eval":
        from t2s.eval import report_data
        report_data.run(args[0] if args else "all")
    elif cmd == "eval2":
        what = args[0] if args else "language"
        if what == "language":
            from t2s.eval import language
            from t2s.pipeline import Engine
            e = Engine()
            d, s = language.evaluate(lambda t, lang, mode, phrasebook=True: e.translate(t, lang=lang, grammar=mode,
                                                                                        phrasebook=phrasebook).to_dict())
            print(s.to_string())
        elif what == "motion":
            from t2s.eval import motion_quality
            from t2s.motion.rig import get_rig
            from t2s.pipeline import Engine
            print(motion_quality.run(Engine(english=False), get_rig())[1])
        elif what == "nmm":
            from t2s.eval import nmm_data
            print(json.dumps(nmm_data.run(), ensure_ascii=False, indent=1))
        elif what == "asr":
            from t2s.eval import asr
            S = asr.sets()
            for k in ("dev", "test_a"):
                print(k, asr.via_api(S[k])[1])
            print("test_b streamed", asr.via_api(S["test_b"], stream=True)[1])
        elif what == "compare":
            from t2s.eval import compare
            print(json.dumps(compare.run(), ensure_ascii=False, indent=1))
    elif cmd == "test":
        import pytest
        return pytest.main(["-q", str(paths.HERE / "tests")])
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

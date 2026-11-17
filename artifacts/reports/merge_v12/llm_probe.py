import json, random, time, sys
sys.path.insert(0, "/app")
from modules.language import compose_sentence, _order
from modules.utils import read_json
from pathlib import Path
M = Path("/app/models")
roles = read_json(M / "roles_language.json") if (M / "roles_language.json").exists() else read_json(M / "roles.json")
grammar = read_json(M / "grammar.json") if (M / "grammar.json").exists() else {}
vocab = [w for w in roles]
pass
vocab = [v for v in vocab if v]
rng = random.Random(0); out = []
tries = 0
while len(out) < 25 and tries < 5000:
    tries += 1
    k = rng.choice([3, 3, 4, 5]); w = rng.sample(vocab, k)
    if _order(w, roles) == w: continue
    segs = [dict(status="accepted", p_correct=0.9, t0=i, t1=i + 1, nearest=x, candidates=[dict(word=x, score=0.9)] + [dict(word=y, score=0.5) for y in rng.sample(vocab, 2)]) for i, x in enumerate(w)]
    t = time.perf_counter(); r = compose_sentence(segs, {}, grammar, roles, use_llm=True); ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter(); r0 = compose_sentence(segs, {}, grammar, roles, use_llm=False); ms0 = (time.perf_counter() - t) * 1000
    out.append(dict(words=w, llm_ms=ms, provider=r.get("provider"), notes=r.get("notes"), sentence=r.get("sentence_th"), rules_ms=ms0, rules=r0.get("sentence_th")))
    print(json.dumps(out[-1], ensure_ascii=False), flush=True)

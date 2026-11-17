"""v11 component tests (prompt-opt-v11 §4C): bank scoring v11, three-way decision, conformal sets, open-set head, input-v11 features,
ONNX parity of a v11 encoder, Path-B boundary labels / decoding."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ------------------------------------------------------------------ bank scoring v11
def test_formbank_matches_rule():
    from modules.runtime import FormBank
    rng = np.random.RandomState(0)
    Z = _unit(rng.randn(12, 8)).astype(np.float32)
    c = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]); sg = np.array(list("aabbccddeeff")); form = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    q = _unit(rng.randn(2, 8)).astype(np.float32)
    for lam in (0.0, 0.05):
        fb = FormBank(Z, c, 3, sg, form, lam=lam)
        S = fb.scores(q)
        for w in range(3):                                     # brute force: max over forms of 0.5·proto + 0.5·top-2 signer-best − λ/√n
            best = -9
            for f in np.unique(form[c == w]):
                m = (c == w) & (form == f)
                means = [_unit(Z[m & (sg == s)].mean(0)) for s in np.unique(sg[m])]
                proto = _unit(np.mean(means, 0))
                per = sorted([max(q @ Z[m & (sg == s)].T, key=lambda v: 0) for s in np.unique(sg[m])], key=lambda v: 0)
                cell = np.array([(q @ Z[m & (sg == s)].T).max(1) for s in np.unique(sg[m])]).T          # [Q, signers]
                top = np.sort(cell, 1)[:, ::-1][:, :2].mean(1) if cell.shape[1] >= 2 else cell[:, 0]
                v = 0.5 * (q @ proto) + 0.5 * top - lam / np.sqrt(len(means))
                best = np.maximum(best, v)
            np.testing.assert_allclose(S[:, w], best, atol=1e-5)


def test_formbank_unknown_concept_is_minus_two():
    from modules.runtime import FormBank
    Z = _unit(np.eye(4)[:3].astype(np.float32))
    fb = FormBank(Z, np.array([0, 0, 1]), 5, np.array(["a", "b", "a"]))
    S = fb.scores(Z[:1])
    assert S.shape == (1, 5) and (S[0, 2:] == -2).all()


# ------------------------------------------------------------------ decision layer v11
def _seg(word, pc, probs, f0=0, f1=20, **kw):
    cands = [dict(word=w, score=0.5, support=0.5, prob=p) for w, p in probs]
    return dict(f0=f0, f1=f1, nearest=word, candidates=cands, p_correct=pc, prob_margin=probs[0][1] - probs[1][1], status="uncertain", **kw)


def _cfg(bound=0.0, tau=0.8, q=0.9):
    from modules.openset import OPEN_FEATS, Logit
    head = Logit(np.zeros(len(OPEN_FEATS)), 10.0, np.zeros(len(OPEN_FEATS)), np.ones(len(OPEN_FEATS)))     # always P(known) ≈ 1
    return dict(open_head=head.to_dict(), known_bound=bound, q={}, q_strict=q, tau={}, tau_strict=tau, tiers={"a": "core", "b": "core"}, groups=[["a", "b"]])


def test_three_way_sure_ambiguous_unknown():
    from modules.openset import three_way
    sure = _seg("a", 0.95, [("a", 0.7), ("b", 0.2), ("c", 0.1)])
    amb = _seg("a", 0.5, [("a", 0.5), ("b", 0.45), ("c", 0.05)])
    segs = three_way([sure, amb], _cfg(), [["a", "b"]])
    assert segs[0]["decision"] == "KNOWN-SURE" and segs[0]["status"] == "accepted"
    assert segs[1]["decision"] == "KNOWN-AMBIGUOUS" and segs[1]["status"] == "uncertain" and set(segs[1]["alternatives"]) >= {"a", "b"}
    blocked = three_way([_seg("a", 0.99, [("a", 0.9), ("b", 0.05), ("c", 0.05)])], _cfg(bound=1.1), [["a", "b"]])
    assert blocked[0]["decision"] == "UNKNOWN" and blocked[0]["status"] == "unknown"


def test_three_way_keeps_transitions_and_old_status_field():
    from modules.openset import three_way
    t = _seg("a", 0.99, [("a", 0.9), ("b", 0.1)]); t["status"] = "transition"
    out = three_way([t], _cfg(), [])
    assert out[0]["status"] == "transition" and "decision" not in out[0]


def test_conformal_set_lac():
    from modules.openset import conformal_set, word_probs
    s = _seg("a", 0.6, [("a", 0.5), ("b", 0.3), ("c", 0.2)])
    wp = word_probs(s)
    assert abs(sum(wp.values()) - 1.0) < 1e-6 and wp["a"] == pytest.approx(0.6)
    assert conformal_set(s, 0.5) == ["a"]                       # threshold 1 − q = 0.5
    assert conformal_set(s, 0.9)[:2] == ["a", "b"]              # threshold 0.1


def test_open_head_logit_roundtrip():
    from modules.openset import Logit
    rng = np.random.RandomState(0)
    X = rng.randn(400, 3); y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(float)
    m = Logit().fit(X, y)
    m2 = Logit.from_dict(m.to_dict())
    np.testing.assert_allclose(m.predict(X), m2.predict(X))
    assert ((m.predict(X) > 0.5) == y).mean() > 0.9


# ------------------------------------------------------------------ input v11 + ONNX parity
def test_features_v11_extends_v10_unchanged():
    from modules.lmenc import N_FEAT, N_FEAT_V11, features, features_v11
    rng = np.random.RandomState(0)
    kp = rng.rand(30, 133, 2).astype(np.float32); sc = (rng.rand(30, 133) > 0.2).astype(np.float32)
    f10, f11 = features(kp, sc), features_v11(kp, sc)
    assert f11.shape == (30, N_FEAT_V11) and f10.shape == (30, N_FEAT)
    np.testing.assert_array_equal(f11[:, :N_FEAT], f10)
    sc2 = sc.copy(); sc2[:, 112:133] = 0                        # right hand not observed → its distances are zero
    f = features_v11(kp, sc2)
    assert np.isfinite(f).all()


def test_v10_model_state_unchanged_by_v11_options():
    torch = pytest.importorskip("torch")
    from modules.lmenc import build_model
    m = build_model(d=32, emb=16, depth=1, heads=2)
    assert "fv" not in m.cfg and not any(k.startswith("miss") for k in m.state_dict())


def test_onnx_parity_v11_encoder(tmp_path):
    torch = pytest.importorskip("torch"); pytest.importorskip("onnxruntime")
    from modules.lmenc import N_FEAT_V11, build_model, export_onnx
    m = build_model(n_feat=N_FEAT_V11, d=32, emb=16, depth=1, heads=2, fv=11)
    ck = tmp_path / "enc.pt"
    torch.save(dict(cfg=m.cfg, state=m.state_dict()), ck)
    assert export_onnx(ck, tmp_path / "enc.onnx") <= 1e-3


# ------------------------------------------------------------------ Path B
def test_boundary_labels_and_decode():
    from modules.boundary import as_tagger_prob, boundary_f1, labels_from_bounds, segment_f1, segments_from_prob
    y, w = labels_from_bounds(40, [(5, 12), (20, 35)])
    assert y[5] == 1 and y[11] == 3 and y[6] == 2 and y[0] == 0 and w[6] == 3.0 and w[25] == 1.0
    p = np.zeros((40, 4)); p[np.arange(40), y] = 1.0
    segs = segments_from_prob(p)
    assert boundary_f1(segs, [(5, 12), (20, 35)]) == 1.0 and segment_f1(segs, [(5, 12), (20, 35)]) == 1.0
    assert np.allclose(as_tagger_prob(p).sum(1), 1.0)

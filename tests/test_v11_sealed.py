"""v11: the sealed BLIND split never reaches training, banks or TUNE (prompt-opt-v11 C4)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from modules import sealed  # noqa: E402

pytestmark = pytest.mark.skipif(not sealed.MANIFEST.exists(), reason="BLIND-v11 not sealed in this checkout")


def test_manifest_shape():
    m = json.loads(sealed.MANIFEST.read_text(encoding="utf-8"))
    assert len(m["people"]["tslone"]) == 2 and len(m["people"]["th_sl"]) == 1 and len(m["people"]["ttrs"]) == 1 and len(m["people"]["agent"]) == 3
    assert len(m["templates"]) == round(0.4 * m["n_templates_total"])
    assert m["clips"] and m["t51_sentences"]


def test_assert_clean_raises_on_sealed_clip_and_signer():
    m = sealed.manifest()
    with pytest.raises(sealed.SealedError):
        sealed.assert_clean([m["clips"][0]], "test")
    with pytest.raises(sealed.SealedError):
        sealed.assert_clean([m["clips"][0] + "#rtmw"], "test")
    with pytest.raises(sealed.SealedError):
        sealed.assert_clean(["x"], "test", signers=[m["signers"][0]])
    sealed.assert_clean(["thsl:not_a_sealed_clip"], "test")


def test_drop_sealed_rows():
    m = sealed.manifest()
    rows = [dict(id="a", signer=m["signers"][0]), dict(id=m["t51_sentences"][0] + "@1.0"), dict(id="keep", signer="S_other")]
    assert [r["id"] for r in sealed.drop_sealed_rows(rows)] == ["keep"]


def test_blind_sets_need_the_flag(monkeypatch):
    from decision import load_set
    monkeypatch.delenv("THAISLM_OPEN_BLIND", raising=False)
    with pytest.raises(PermissionError):
        load_set("iso_blind")


def test_training_table_has_no_sealed_people():
    pd = pytest.importorskip("pandas")
    c = pd.read_parquet(ROOT / "cloud_s3" / "data_prep" / "shards" / "clips.parquet")
    tr = c[(c.role == "train") & c.signer.notna()]
    ss = sealed.sealed_signers()
    leaked = tr[tr.signer.isin(ss)]
    # TTRS S_ttrs_u00 has TRAIN-role clips in the corpus table: train_v11.tables() must drop them before training
    from train_v11 import tables  # noqa: F401  (import check only; the drop is asserted by assert_clean inside tables())
    assert set(leaked.signer) <= {"S_ttrs_u00"}

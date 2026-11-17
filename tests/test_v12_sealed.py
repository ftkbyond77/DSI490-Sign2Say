"""v12: BLIND-v12 (whole new-source people + held-out sentences) never reaches training, banks or TUNE (prompt-opt-v12 C4 / 1G)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from modules import sealed  # noqa: E402

pytestmark = pytest.mark.skipif(not sealed.MANIFEST_V12.exists(), reason="BLIND-v12 not sealed in this checkout")


def test_manifest_v12_shape():
    m = json.loads(sealed.MANIFEST_V12.read_text(encoding="utf-8"))
    assert m["seed"] == 12 and m["clips"] and "S_rid_00" in m["signers"]
    assert isinstance(m.get("openings"), list)


def test_v12_people_and_clips_are_sealed_with_v11_kept():
    m12 = sealed.manifest_v12()
    assert sealed.is_sealed(signer="S_rid_00") and sealed.is_sealed(signer="S_rid_02")
    assert sealed.is_sealed(m12["clips"][0])
    with pytest.raises(sealed.SealedError):
        sealed.assert_clean([m12["clips"][0]], "test")
    with pytest.raises(sealed.SealedError):
        sealed.assert_clean(["x"], "test", signers=["S_rid_00"])
    if sealed.MANIFEST.exists():                          # BLIND-v11 still enforced (union)
        m11 = json.loads(sealed.MANIFEST.read_text(encoding="utf-8"))
        assert sealed.is_sealed(m11["clips"][0])
    assert not sealed.is_sealed(signer="S_etsl_00")       # the TRAIN new signer is not sealed


def test_new_rows_roles_match_manifest():
    pd = pytest.importorskip("pandas")
    f = ROOT / "cloud_s3" / "data_prep" / "manifest" / "clips_v12_new.parquet"
    if not f.exists():
        pytest.skip("no v12 table")
    N = pd.read_parquet(f)
    tr = N[N.role.isin(["train", "newsig_dev"])]
    assert not tr.clip_id.map(sealed.is_sealed).any()
    assert not tr.signer.isin(sealed.manifest_v12()["signers"]).any()


def test_blind12_needs_its_own_flag(monkeypatch):
    monkeypatch.delenv("THAISLM_OPEN_BLIND12", raising=False)
    assert not sealed.blind12_open()
    monkeypatch.setenv("THAISLM_OPEN_BLIND12", "1")
    assert sealed.blind12_open()

"""API contract (FastAPI TestClient, in-process; the speech service is not needed)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "service" / "api" / "app.py"
    spec = importlib.util.spec_from_file_location("t2s_api", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t2s_api"] = mod
    spec.loader.exec_module(mod)
    with TestClient(mod.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health").json()
    assert r["ok"] and r["signs"] > 9000


def test_translate_and_render(client):
    r = client.post("/api/translate", json={"text": "วันนี้คุณไปไหน"})
    assert r.status_code == 200
    d = r.json()
    assert [u["gloss"] for u in d["ir"]["units"]] == ["วันนี้", "คุณ", "ไป", "ที่ไหน"]
    assert d["motion"]["frames"] > 25
    # an expert edits the gloss order → re-render
    ir = d["ir"]
    ir["units"] = list(reversed(ir["units"]))
    r2 = client.post("/api/render", json={"ir": ir})
    assert r2.status_code == 200 and r2.json()["ir"]["units"][0]["gloss"] == "ที่ไหน"


def test_errors(client):
    assert client.post("/api/translate", json={"text": "  "}).status_code == 400
    assert client.post("/api/translate", json={"text": "ก" * 500}).status_code == 413
    assert client.get("/api/sign/ไม่มีคำนี้แน่นอน").status_code == 404


def test_lookup_and_sign(client):
    items = client.get("/api/lookup", params={"q": "ห้องน"}).json()["items"]
    assert any(i["text"] == "ห้องน้ำ" for i in items)
    assert client.get("/api/sign/ห้องน้ำ").json()["motion"]["frames"] > 10


def test_vrma(client):
    r = client.post("/api/export/vrma", json={"text": "ขอบคุณ"})
    assert r.status_code == 200 and r.content[:4] == b"glTF"


def test_avatars(client):
    r = client.get("/api/avatars")
    assert r.status_code == 200
    av = r.json()["avatars"]
    assert av and av[0]["label"] == "Default"
    assert [a["label"] for a in av[1:]] == [f"Model {k}" for k in range(1, len(av))]
    for a in av:
        assert a["url"].startswith("/avatars/")
        rest = client.get("/api/rest", params={"avatar": a["id"]}).json()
        assert rest["frames"] == 1
    if len(av) > 1:
        a, b = (client.post("/api/translate", json=dict(text="คุณชื่ออะไร", avatar=x["id"])).json() for x in av[:2])
        assert a["ir"]["units"] == b["ir"]["units"] and a["motion"]["frames"] == b["motion"]["frames"]
        assert a["motion"]["q"] != b["motion"]["q"]            # same signs, arms fitted to the other body

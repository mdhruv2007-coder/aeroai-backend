import math
import pytest
from fastapi.testclient import TestClient

from config import settings

if not (settings.model_dir / settings.onnx_model).exists():
    pytest.skip("model files not present", allow_module_level=True)

from main import app

H = {"X-API-Key": settings.api_key_admin}
BODY = {"m": 2, "p": 4, "t": 12, "aoa": 4, "re": 1_000_000}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs the lifespan (loads the model)
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_auth_required(client):
    assert client.post("/v1/predict", json=BODY).status_code == 401
    assert client.post("/v1/predict", json=BODY, headers={"X-API-Key": "nope"}).status_code == 403


def test_predict_is_finite_and_uncertainty_is_honest(client):
    r = client.post("/v1/predict", json=BODY, headers=H)
    assert r.status_code == 200
    d = r.json()
    assert all(math.isfinite(d[k]) for k in ("cl", "cd", "cm", "stall_margin", "ld_ratio"))
    assert d["cd"] > 0
    assert d["confidence"] is None and d["cl_std"] is None and d["confidence_tier"] == "unavailable"


def test_naca0012_symmetry_at_zero_aoa(client):
    d = client.post("/v1/predict", headers=H, json={"m": 0, "p": 0, "t": 12, "aoa": 0, "re": 1_000_000}).json()
    assert abs(d["cl"]) < 0.05 and abs(d["cm"]) < 0.02


def test_cl_increases_with_aoa_below_stall(client):
    cl = [client.post("/v1/predict", headers=H, json={**BODY, "aoa": a}).json()["cl"] for a in range(-2, 9, 2)]
    assert all(b > a for a, b in zip(cl, cl[1:])), cl


def test_envelope_warning(client):
    d = client.post("/v1/predict", headers=H, json={**BODY, "p": 0.5}).json()
    assert any("outside 2-7" in w for w in d["warnings"])

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from ml_project.predictor import Predictor,DEFAULT,read
from lab08.app import create_app
from .checks import ROOT,C,drills

@pytest.fixture
def client():
    with TestClient(create_app()) as c: yield c

def test_golden(client):
    for row in read(ROOT/"golden.json"):
        r=client.post("/predict",json={"schema_version":"1.0","features":row["features"]}).json()
        for k in ("model_version","schema_version","predicted_class"): assert r[k]==row["expected"][k]
        assert abs(r["probability"]-row["expected"]["probability"])<=C["tolerance"]

def test_partition_order(client):
    records=[x["features"] for x in read(ROOT/"golden.json")]
    def batch(rs):
        return client.post("/predict-batch",json=dict(schema_version="1.0",records=rs)).json()["predictions"]
    full=batch(records)
    split=batch(records[:3])+batch(records[3:])
    for i,r in enumerate(records):
        single=client.post("/predict",json=dict(schema_version="1.0",features=dict(reversed(list(r.items()))))).json()
        assert full[i]["probability"]==pytest.approx(single["probability"],abs=1e-10)
        assert full[i]["probability"]==pytest.approx(split[i]["probability"],abs=1e-10)
def test_model_lifecycle(client):
    p=client.app.state.predictor
    with patch.object(p.model,"fit",side_effect=AssertionError("unexpected fit")),patch.object(Predictor,"__init__",side_effect=AssertionError("unexpected reload")):
        for _ in range(2):assert client.post("/predict",json=dict(schema_version="1.0",features=read(DEFAULT.parent/"request.json"))).status_code==200
def test_internal_error(client):
    with patch.object(client.app.state.predictor,"predict_batch",side_effect=RuntimeError("private diagnosis")):
        r=client.post("/predict",json=dict(schema_version="1.0",features=read(DEFAULT.parent/"request.json")))
    assert r.status_code==500 and r.json()["error"]=="inference_failed" and "private" not in r.text and r.headers["X-Request-ID"]==r.json()["request_id"]
def test_warnings(client):
    row=read(DEFAULT.parent/"request.json"); row["alcohol"]=40
    client.post("/predict",json=dict(schema_version="1.0",features=row))
    assert client.get("/metrics").json()["warnings"]==1
def test_drills(): assert (drills().ready==503).all()

import pytest
from fastapi.testclient import TestClient
from ml_project.predictor import DEFAULT,read
from .app import create_app

@pytest.fixture
def client():
    with TestClient(create_app()) as c: yield c

def payload(): return dict(schema_version="1.0",features=read(DEFAULT.parent/"request.json"))

@pytest.mark.parametrize("path",["/health","/ready","/metadata","/metrics","/openapi.json"])
def test_endpoints(client,path): assert client.get(path).status_code==200

def test_control(client):
    r=client.post("/predict",json=payload())
    assert r.json()["probability"]==pytest.approx(read(DEFAULT.parent/"inference_example.json")["probability"],abs=1e-10)
    assert r.json()["request_id"]==r.headers["X-Request-ID"]

@pytest.mark.parametrize("case",["empty","missing","extra","text","envelope","version","nan","infinity"])
def test_bad(client,case):
    p=payload()
    if case in ("nan","infinity"):
        import json
        p["features"]["alcohol"]=float(case)
        r=client.post("/predict",content=json.dumps(p),headers={"content-type":"application/json"})
    else:
        if case=="empty": p={}
        elif case=="missing": del p["features"]["alcohol"]
        elif case=="extra": p["features"]["quality"]=8
        elif case=="text": p["features"]["alcohol"]="1"
        elif case=="envelope": p["unknown"]=1
        else: p["schema_version"]="2"
        r=client.post("/predict",json=p)
    assert r.status_code==422 and r.json()["error"]=="invalid_request"

@pytest.mark.parametrize("size",[0,65])
def test_batch_limits(client,size):
    assert client.post("/predict-batch",json=dict(schema_version="1.0",records=[payload()["features"]]*size)).status_code==422

def test_unready(tmp_path):
    with TestClient(create_app(tmp_path/"absent")) as c:
        assert c.get("/health").status_code==200 and c.get("/ready").status_code==503

def test_batch(client):
    r=client.post("/predict-batch",json=dict(schema_version="1.0",records=[payload()["features"]]*3))
    assert r.status_code==200 and r.json()["count"]==3

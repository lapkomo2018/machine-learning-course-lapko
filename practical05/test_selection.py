import joblib
import pandas as pd
import pytest
from src.experiment_io import read_json
from src.titanic_workflow import load_titanic,split_features_target
from .main import ROOT,A
from .selection import check_constraints,normalize,rank_models,validate

def cfg(): return read_json(ROOT/"configs"/"interactive.json")
def sample():
    return pd.DataFrame([dict(model=n,f1=.85,recall=.85,p95_ms=3,size_bytes=5000) for n in ("z","a")])
def test_gain(): assert normalize(.92,cfg()["criteria"]["f1"])==1
def test_cost(): assert normalize(1,cfg()["criteria"]["p95_ms"])==1
def test_clip(): assert normalize(100,cfg()["criteria"]["p95_ms"])==0
def test_boundary(): assert check_constraints({"x":2},{"x":{"max":2,"min":2}})==(True,[])
def test_reasons(): assert len(check_constraints({"a":1,"b":2},{"a":{"min":3},"b":{"max":0}})[1])==2
def test_reject():
    x=sample(); x.loc[0,"p95_ms"]=101
    r,b=rank_models(x,cfg()); assert list(r.model)==["a"] and list(b.model)==["z"]
def test_empty():
    x=sample().assign(recall=0); assert rank_models(x,cfg())[0].empty
def test_weights():
    c=cfg(); c["criteria"]["f1"]["weight"]=0
    with pytest.raises(ValueError): validate(c)
def test_missing():
    with pytest.raises(KeyError): rank_models(sample().drop(columns="recall"),cfg())
def test_tie(): assert list(rank_models(sample(),cfg())[0].model)==["a","z"]
def test_serialized():
    X,_=split_features_target(load_titanic())
    for p in A.glob("*.joblib"): assert joblib.load(p).predict_proba(X.head()).shape==(5,2)

import joblib
import numpy as np
import pandas as pd
import pytest
from .main import ROOT,H,R,data,metrics,validate
from .search import candidates
from src.experiment_io import read_json,digest

def test_no_leak():
    c=read_json(ROOT/"config.json"); c["test_used_for_search"]=True
    with pytest.raises(ValueError): validate(c)
def test_manual_budget(): assert len(candidates(71))==12
def test_score(): assert metrics([0,1],[.1,.9])["ap"]==1
def test_ids():
    tr,te=[pd.read_csv(H/f"{k}_ids.csv").id for k in ("train","test")]
    X,_=data()
    assert not set(tr)&set(te) and len(tr)+len(te)==len(X) and not X.duplicated().any()
def test_oof():
    p=pd.read_csv(R/"oof.csv")
    assert p.id.is_unique and set(p.id)==set(pd.read_csv(H/"train_ids.csv").id) and p[["manual","automl"]].notna().all().all()
def test_trials():
    d=pd.read_csv(R/"budget.csv"); assert (d.trials==12).all() and (d.inner_fits==36).all() and len(d)==16
def test_decision():
    assert (H/"decision.sha256").read_text().strip()==digest(H/"selection_decision.json") and not read_json(H/"selection_decision.json")["test_metrics_present"]
def test_example():
    e=read_json(H/"inference_example.json");p=joblib.load(H/"model.joblib").predict_proba(pd.DataFrame([e["features"]]))[0,1]
    assert np.isclose(p,e["probability"],rtol=0,atol=1e-10)
def test_manifest(): assert all(digest(H/k)==v for k,v in read_json(H/"manifest.json").items())

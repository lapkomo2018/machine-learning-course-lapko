import numpy as np
import pandas as pd
import pytest
from ml_project.predictor import Predictor,DEFAULT,read
from .main import R,quality,audited_predict

def test_true_metrics():assert quality([0,1],np.array([.2,.8]))["ap"]==1
def test_split():assert read(R/"data_audit.json")["feature_overlap"]==0
def test_claims():assert pd.read_csv(R/"claims.csv").confirmed.all()
def test_cases():assert pd.read_csv(R/"cases.csv").id.nunique()>=10
def test_verdict():assert read(R/"verdict.json")["fixed"]==["F1","F2","F3"]
def test_control():
    ref=read(DEFAULT.parent/"inference_example.json")
    assert audited_predict(Predictor(),ref["features"])["probability"]==pytest.approx(ref["probability"],abs=1e-10)
@pytest.mark.parametrize("value",[True,"12",-1,float("nan"),float("inf")])
def test_guard(value):
    row=read(DEFAULT.parent/"request.json");row["alcohol"]=value
    with pytest.raises(ValueError):audited_predict(Predictor(),row)
def test_evidence():assert len(read(R/"file_manifest.json"))==9

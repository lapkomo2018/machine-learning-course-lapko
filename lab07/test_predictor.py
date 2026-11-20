from pathlib import Path
import os
import subprocess
import sys
import pytest
from ml_project.predictor import DEFAULT,Predictor,local,read

@pytest.fixture(scope="module")
def pred(): return Predictor()
@pytest.fixture
def row(): return read(DEFAULT.parent/"request.json")
@pytest.mark.parametrize("change",["missing","extra","str","bool","nan","inf"])
def test_invalid(pred,row,change):
    k=next(iter(row))
    if change=="missing": del row[k]
    elif change=="extra": row["quality"]=8
    else: row[k]={"str":"2","bool":True,"nan":float("nan"),"inf":float("inf")}[change]
    with pytest.raises(ValueError): pred.predict_record(row)
def test_warning(pred,row):
    row["alcohol"]=50
    assert "outside_train_range:alcohol" in pred.predict_record(row)["warnings"]
def test_order(pred,row): assert pred.predict_record(row)==pred.predict_record(dict(reversed(list(row.items()))))
def test_batch(pred,row):
    assert pred.predict_record(row)["probability"]==pytest.approx(pred.predict_batch([row]*64)[0]["probability"],abs=1e-10)
    for batch in ([],[row]*65):
        with pytest.raises(ValueError): pred.predict_batch(batch)
def test_control(pred,row): assert pred.predict_record(row)["probability"]==pytest.approx(read(DEFAULT.parent/"inference_example.json")["probability"],abs=1e-10)
def test_path(tmp_path):
    with pytest.raises(ValueError):local(tmp_path,"../escape")
def test_metadata(pred): assert pred.describe()["feature_count"]==11
def test_cli(tmp_path):
    cmd=[sys.executable,"-m","ml_project.cli"]
    assert subprocess.run(cmd+["metadata"],capture_output=True).returncode==0
    bad=tmp_path/"bad.json";bad.write_text("{}")
    r=subprocess.run(cmd+["predict",str(bad)],capture_output=True)
    assert r.returncode==2 and r.stderr and not r.stdout

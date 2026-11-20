import joblib
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_digits

from .main import A,R,select_q,align,inject,scores
from src.experiment_io import digest,read_json

def test_dimension(): assert select_q([.6,.9,1],.9)==2
def test_last_component(): assert select_q([.6,.9999999999],1)==2
def test_bad_dimension():
    with pytest.raises(ValueError): select_q([1],0)
def test_bad_fraction():
    with pytest.raises(ValueError): inject(np.zeros((10,64)),np.arange(10),2,42)
def test_split():
    ids=np.load(A/"split.npz")
    assert not set(ids["train"]) & set(ids["test"])
def test_alignment():
    assert np.array_equal(align([0,1,1],[1,0,0]),[0,1,1])
def test_ari(): assert adjusted_rand_score([1,1,0],[0,0,1])==1
def test_injection():
    x=load_digits().data[:30]
    before=x.copy()
    a,m=inject(x,np.arange(30),.3,2)
    b,n=inject(x,np.arange(30),.3,2)
    assert np.array_equal(a,b) and m.equals(n) and np.array_equal(x,before)
def test_pipeline():
    b=joblib.load(A/"bundle.joblib")
    x=load_digits().data[:4]
    z=b["pca"].transform(b["scaler"].transform(x))
    assert b["pca"].inverse_transform(z).shape==x.shape and len(b["km_pca"].predict(z))==4
    assert scores(b,x).combined.between(0,1).all()
def test_manifest():
    assert all(digest(A/k)==v for k,v in read_json(A/"manifest.json").items())
    assert all((R/f).exists() for f in ("external.csv","stability.csv","detectors.csv","audit.csv","dimensions.csv"))

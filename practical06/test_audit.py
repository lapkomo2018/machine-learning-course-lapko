from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from .main import validate,verify,boundaries,representatives,jaccard,policy,align,profile,R,F,ROOT,SOURCE
from src.experiment_io import read_json,write_json

def sample():
    return pd.DataFrame(dict(id=["a","b"],combined=[.4,.6],if_full=[.2,.3],if_pca=[.2,.4],reconstruction=[.1,.3]))
def test_missing(tmp_path):
    with pytest.raises(FileNotFoundError): verify(tmp_path)
def test_corrupt(tmp_path):
    (tmp_path/"x").write_text("bad");write_json(tmp_path/"manifest.json",{"x":"bad"})
    with pytest.raises(ValueError): verify(tmp_path)
def test_duplicate():
    with pytest.raises(ValueError): validate(sample().assign(id="x"),[1,2],2,[0,1])
def test_shape():
    with pytest.raises(ValueError): validate(sample(),[1],2,[0,1])
def test_align(): assert align([0,1,0],[1,0,1]).tolist()==[0,1,0]
def test_margin():
    d=boundaries([[1]],[[0],[4]]).iloc[0]; assert d.distance==1 and d.margin==2
def test_roles():
    sizes=pd.read_csv(R/"object_stability.csv").groupby("cluster").size().clip(upper=3)
    actual=pd.read_csv(R/"representatives.csv").groupby(["cluster","role"]).size().unstack()
    assert list(actual.columns)==["boundary","distant","typical"] and actual.eq(sizes,axis=0).all().all()
def test_jaccard(): assert jaccard([1,2],[2,3])==1/3
def test_immutable():
    x=np.array([[0.,1],[1,2]]);original=x.copy();profile(x,np.array([0,1]));assert np.array_equal(x,original)
def test_priority():
    c=read_json(ROOT/"config.json")
    r=SimpleNamespace(combined=.99,if_pca_flag=True,reconstruction_flag=True,margin=.01)
    assert policy(r,c)[0]=="rescan"
def test_cards():
    d=pd.read_csv(R/"cards.csv"); assert len(d)==15 and d.id.is_unique and d.hypothesis.notna().all()
def test_contract(): verify(SOURCE);assert (F/"components.png").is_file() and (R/"profiles.csv").is_file()

"""Build the package from the trusted LR6 handoff and independently repeat its frozen fit."""
import platform
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile,ZIP_DEFLATED

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.experiment_io import digest,read_json,write_json

ROOT=Path(__file__).resolve().parent
H=ROOT.parent/"lab06/handoff"
P=ROOT/"src/ml_project"; A=P/"assets"
TOL=1e-10


def rebuild(decision):
    raw=pd.read_csv(ROOT.parent/"data/winequality-red.csv",sep=";")
    ids=pd.read_csv(H/"train_ids.csv").id
    train=raw.loc[ids]; features=[f["name"] for f in read_json(H/"data_schema.json")["features"]]
    estimators={cls.__name__:cls for cls in (LogisticRegression,RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier)}
    estimator=estimators[decision["family"]](**decision["params"])
    rebuilt=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),estimator)
    rebuilt.fit(train[features],(train.quality>=7).astype(int))
    return rebuilt,raw,features


def build():
    A.mkdir(exist_ok=True); (ROOT/"results").mkdir(exist_ok=True)
    manifest=read_json(H/"manifest.json")
    required={"model.joblib","selection_decision.json","experiment_config.json","data_schema.json","train_ids.csv","test_ids.csv","test_metrics.json","cv_results.csv","inference_example.json","MODEL_CARD.md","requirements.txt","decision.sha256"}
    if required-set(manifest) or any(digest(H/k)!=v for k,v in manifest.items()): raise ValueError("Invalid LR6 handoff")
    for name in required: copyfile(H/name,A/name)
    dec=read_json(A/"selection_decision.json")
    cfg=dict(model="model.joblib",schema="data_schema.json",metadata="metadata.json",decision="selection_decision.json",
             model_version="1.0.0",schema_version="1.0",threshold=dec["threshold"],positive_class=dec["positive_class"],max_batch_size=64)
    write_json(A/"project_config.json",cfg)
    write_json(A/"metadata.json",dict(model_version=cfg["model_version"],schema_version=cfg["schema_version"],family=dec["family"],
               threshold=cfg["threshold"],positive_class=cfg["positive_class"],model_sha256=digest(A/"model.joblib"),predictor_sha256=digest(P/"predictor.py"),
               feature_count=len(read_json(A/"data_schema.json")["features"]),python=platform.python_version(),sklearn=sklearn.__version__,
               joblib=joblib.__version__,intended_use="educational ranking of red wine samples"))
    rebuilt,raw,features=rebuild(dec)
    ids=pd.read_csv(H/"test_ids.csv").id
    frame=raw.loc[ids,features]
    original=joblib.load(H/"model.joblib")
    old,new=[model.predict_proba(frame)[:,1] for model in (original,rebuilt)]
    diff=float(max(abs(new-old)))
    ap=average_precision_score(raw.loc[ids,"quality"]>=7,new)
    ref=read_json(H/"test_metrics.json")["ap"]
    if diff>TOL or abs(ap-ref)>TOL: raise ValueError("Frozen rebuild differs")
    write_json(ROOT/"results/reproducibility.json",dict(max_abs_probability_difference=diff,n=len(ids),tolerance=TOL,ap=ap,reference_ap=ref,model=digest(H/"model.joblib")))
    record=read_json(A/"inference_example.json")["features"]
    write_json(A/"request.json",record);write_json(A/"batch.json",[record,record])
    write_json(A/"manifest.json",{p.name:digest(p) for p in sorted(A.iterdir()) if p.name!="manifest.json"})
    with ZipFile(ROOT/"handoff.zip","w",ZIP_DEFLATED) as archive:
        for path in sorted([*P.rglob("*.py"),*A.iterdir(),ROOT/"pyproject.toml",ROOT/"test_predictor.py",ROOT/"README.md"]):
            archive.write(path,path.relative_to(ROOT))
    with ZipFile(ROOT/"handoff.zip") as archive,TemporaryDirectory() as tmp:
        if archive.testzip(): raise ValueError("Damaged archive")
        archive.extractall(tmp)
        extracted=Path(tmp)/"src/ml_project/assets"
        assert all(digest(extracted/k)==v for k,v in read_json(extracted/"manifest.json").items())


if __name__=="__main__":
    with threadpool_limits(1): build()

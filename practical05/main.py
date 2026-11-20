import argparse
from copy import deepcopy
import io
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, average_precision_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from src.titanic_workflow import load_titanic, split_features_target, stratified_split, make_preprocessor, make_cv
from src.experiment_io import read_json, write_json, save_plot
from .selection import rank_models

ROOT = Path(__file__).resolve().parent
R, A, F = [ROOT/d for d in ("results","artifacts","figures")]


def quality(y,p):
    return dict(f1=f1_score(y,p>=.5),recall=recall_score(y,p>=.5),ap=average_precision_score(y,p))


def benchmark(X,y):
    specs = {"logistic":LogisticRegression(C=1,max_iter=500,random_state=42),
             "tree":DecisionTreeClassifier(max_depth=3,random_state=42),
             "forest":RandomForestClassifier(n_estimators=80,max_depth=6,n_jobs=1,random_state=42),
             "boost":HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,random_state=42)}
    folds = list(make_cv().split(X,y))
    rows,fold_rows,times,models = [],[],[],{}
    for name,estimator in specs.items():
        pipe = make_pipeline(make_preprocessor(),estimator)
        metrics = []
        for i,(train,val) in enumerate(folds):
            pipe.fit(X.iloc[train],y.iloc[train])
            row = quality(y.iloc[val],pipe.predict_proba(X.iloc[val])[:,1])
            metrics.append(row)
            fold_rows.append(dict(model=name,fold=i,**row))
        start = perf_counter()
        pipe.fit(X,y)
        fit_s = perf_counter()-start
        batch = X.iloc[:32]
        for _ in range(5):
            pipe.predict_proba(batch)
        latency = []
        for rep in range(40):
            start = perf_counter()
            pipe.predict_proba(batch)
            latency.append(1000*(perf_counter()-start))
            times.append((name,rep,latency[-1]))
        buf = io.BytesIO()
        joblib.dump(pipe,buf,compress=3)
        models[name]=buf.getvalue()
        folds_frame = pd.DataFrame(metrics)
        rows.append(dict(model=name,**folds_frame.mean(),**folds_frame.std().add_suffix("_sd"),
                         fit_s=fit_s,p50_ms=np.median(latency),p95_ms=np.percentile(latency,95),
                         rows_per_s=32000/np.mean(latency),size_bytes=len(models[name])))
    matrix=pd.DataFrame(rows)
    matrix.to_csv(R/"benchmark_results.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(R/"folds.csv",index=False)
    timing=pd.DataFrame(times,columns=["model","repeat","ms"])
    timing.to_csv(R/"latency.csv",index=False)
    timing.boxplot(column="ms",by="model"); plt.suptitle("")
    save_plot(F/"latency.png","40 вимірювань після прогріву","Модель","мс / 32 записи")
    plt.scatter(matrix.p95_ms,matrix.f1)
    for row in matrix.itertuples():
        plt.annotate(row.model,(row.p95_ms,row.f1))
    save_plot(F/"tradeoff.png","Якість і затримка","p95, мс / пакет","F1")
    return matrix,models


def select(matrix,models,cfg):
    name=cfg["name"]
    rank,reject=rank_models(matrix,cfg)
    rank.to_csv(R/f"ranking_{name}.csv",index=False)
    reject.to_csv(R/f"rejected_{name}.csv",index=False)
    if rank.empty:
        write_json(R/f"selection_{name}.json",dict(scenario=name,status="no_eligible_model",rejected=reject.to_dict("records")))
        return None
    winner=rank.iloc[0].model
    backup=rank.iloc[1].model if len(rank)>1 else None
    row=matrix.set_index("model").loc[winner]
    selected=dict(scenario=name,version=cfg["version"],config=cfg,winner=winner,backup=backup,score=rank.iloc[0].score,
                  expected_f1=[max(0,row.f1-row.f1_sd),min(1,row.f1+row.f1_sd)],rejected=reject.to_dict("records"),
                  risks=["hardware-dependent latency","historical population"],reselect="new hardware, schema or recall below floor")
    write_json(R/f"selection_{name}.json",selected)
    (A/f"{name}_primary.joblib").write_bytes(models[winner])
    if backup:
        (A/f"{name}_backup.joblib").write_bytes(models[backup])
    (R/f"{name}.md").write_text(f"# {name}\n\nОсновна модель: {winner}. Резерв: {backup}. Бал: {selected['score']:.4f}.\n")
    trials=[]
    for weight in np.linspace(.05,.95,19):
        alt=deepcopy(cfg)
        current=cfg["criteria"]["f1"]["weight"]
        for k,v in alt["criteria"].items():
            v["weight"]=weight if k=="f1" else v["weight"]*(1-weight)/(1-current)
        r,_=rank_models(matrix,alt)
        trials.append(r.assign(f1_weight=weight))
    sweep=pd.concat(trials)
    sweep.to_csv(R/f"sensitivity_{name}.csv",index=False)
    for model,g in sweep.groupby("model"):
        plt.plot(g.f1_weight,g.score,label=model)
    plt.legend()
    save_plot(F/f"sensitivity_{name}.png",f"Чутливість: {name}","Вага F1","Бал")
    return winner


def run(configs):
    for d in (R,A,F):
        d.mkdir(exist_ok=True)
    X,T,y,t=stratified_split(*split_features_target(load_titanic()))
    matrix,models=benchmark(X,y)
    selected={}
    for path in configs:
        cfg=read_json(path)
        selected[cfg["name"]]=select(matrix,models,cfg)
    # Both scenarios are locked before any test evaluation, each unique winner scored once.
    test={m:quality(t,joblib.load(io.BytesIO(models[m])).predict_proba(T)[:,1]) for m in set(selected.values()) if m}
    write_json(R/"test.json",test)
    write_json(R/"environment.json",dict(python=platform.python_version(),sklearn=sklearn.__version__,machine=platform.machine(),system=platform.platform(),batch=32,warmup=5,repeats=40,threads=1))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,nargs="+",default=sorted((ROOT/"configs").glob("*.json")))
    with threadpool_limits(1):
        run(parser.parse_args().config)

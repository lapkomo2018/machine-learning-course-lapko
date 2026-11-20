import io
from itertools import combinations
from pathlib import Path
from shutil import copyfile
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as m
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits

from src.experiment_io import digest,read_json,save_plot,write_json
from .search import automl_search,manual_search

ROOT=Path(__file__).resolve().parent
R,F,H= [ROOT/k for k in ("results","figures","handoff")]
DATA=ROOT.parent/"data/winequality-red.csv"


def data():
    df=pd.read_csv(DATA,sep=";")
    X=df.drop(columns="quality")
    keep=~X.duplicated()
    return X.loc[keep],(df.loc[keep,"quality"]>=7).astype(int)


def metrics(y,p,threshold=.5):
    p=np.asarray(p)
    return dict(ap=m.average_precision_score(y,p),roc=m.roc_auc_score(y,p),f1=m.f1_score(y,p>=threshold,zero_division=0),
                brier=m.brier_score_loss(y,p),recall=m.recall_score(y,p>=threshold,zero_division=0))


def validate(c):
    if c["test_used_for_search"] or c["test_used_for_selection"] or c["budget"]!=12 or c["inner"]<3 or c["outer"]<5:
        raise ValueError("Invalid search protocol")


def latency(model,X):
    for _ in range(5): model.predict_proba(X)
    samples=[]
    for _ in range(35):
        start=perf_counter(); model.predict_proba(X); samples.append((perf_counter()-start)*1000)
    return float(np.percentile(samples,95))


def run():
    for d in (R,F,H,R/"logs"):
        d.mkdir(exist_ok=True)
    cfg=read_json(ROOT/"config.json"); validate(cfg)
    seed=cfg["seed"]
    X,y=data()
    tr,te=train_test_split(X.index,test_size=cfg["test_size"],stratify=y,random_state=seed)
    for label,ids in (("train",tr),("test",te)):
        pd.DataFrame({"id":ids}).to_csv(H/f"{label}_ids.csv",index=False)
    train=X.loc[tr]; target=y.loc[tr]
    history=[]; oof=[]; fold_rows=[]; importance=[]; resources=[]; stability=[]; probabilities={}; finalists={}
    outer=StratifiedKFold(cfg["outer"],shuffle=True,random_state=seed)
    jobs=[(f"fold{i}",seed,a,b) for i,(a,b) in enumerate(outer.split(train,target))]
    jobs += [(f"final{s}",s,np.arange(len(tr)),None) for s in cfg["stability_seeds"]]
    for stage,s,a,b in jobs:
        xx,yy=train.iloc[a],target.iloc[a]
        splitter=StratifiedKFold(cfg["inner"],shuffle=True,random_state=s)
        splits=list(splitter.split(xx,yy))
        for branch in ("manual","automl"):
            start=perf_counter()
            if branch=="manual": model,rows=manual_search(xx,yy,splits,s)
            else: model,rows=automl_search(xx,yy,splitter,cfg,s,R/"logs"/f"{stage}.jsonl")
            search_s=perf_counter()-start
            history.extend(dict(stage=stage,branch=branch,**row) for row in rows)
            start=perf_counter();model.fit(xx,yy);fit_s=perf_counter()-start
            fam=type(model.steps[-1][1]).__name__
            if b is not None:
                xv,yv=train.iloc[b],target.iloc[b]
                p=model.predict_proba(xv)[:,1]
                fold_rows.append(dict(stage=stage,branch=branch,baseline=yy.mean(),baseline_ap=yv.mean(),**metrics(yv,p)))
                oof.extend(dict(id=i,truth=v,stage=stage,branch=branch,p=prob) for i,v,prob in zip(xv.index,yv,p))
                imp=permutation_importance(model,xv,yv,scoring="average_precision",n_repeats=2,random_state=s)
                importance.extend(dict(stage=stage,branch=branch,feature=k,importance=v) for k,v in zip(X.columns,imp.importances_mean))
            else:
                probabilities[branch,s]=model.predict_proba(train)[:,1]
                stability.append(dict(branch=branch,seed=s,family=fam,best_inner_ap=max(row["ap"] for row in rows)))
                if s==seed:
                    buf=io.BytesIO();joblib.dump(model,buf,compress=3)
                    (R/f"{branch}.joblib").write_bytes(buf.getvalue())
                    finalists[branch]=model
                    resources.append(dict(branch=branch,search_s=search_s,fit_s=fit_s,bytes=len(buf.getvalue()),
                                          single_p95_ms=latency(model,train.iloc[:1]),batch32_p95_ms=latency(model,train.iloc[:32])))
    pd.DataFrame(history).to_csv(R/"candidate_history.csv",index=False)
    h=pd.DataFrame(history)
    budget=h.groupby(["stage","branch"]).agg(trials=("trial","size"),trial_seconds=("seconds","sum")).reset_index()
    budget["inner_fits"]=budget.trials*cfg["inner"];budget.to_csv(R/"budget.csv",index=False)
    pd.DataFrame(stability).to_csv(R/"stability.csv",index=False)
    changes=[dict(branch=branch,seed_a=a,seed_b=b,disagreement=np.mean((probabilities[branch,a]>=.5)!=(probabilities[branch,b]>=.5))) for branch in finalists for a,b in combinations(cfg["stability_seeds"],2)]
    pd.DataFrame(changes).to_csv(R/"seed_disagreement.csv",index=False)
    eng=pd.DataFrame(resources).set_index("branch");eng.to_csv(R/"resources.csv")
    folds=pd.DataFrame(fold_rows); folds.to_csv(H/"cv_results.csv",index=False)
    wide=folds.pivot(index="stage",columns="branch",values="ap")
    diff=wide.automl-wide.manual
    means=folds.groupby("branch").mean(numeric_only=True)
    means.to_csv(R/"means.csv")
    frame=pd.DataFrame(oof).pivot(index=["id","truth","stage"],columns="branch",values="p").reset_index()
    frame["delta"]=frame.automl-frame.manual
    frame["decision_pair"]=np.char.add(np.where(frame.manual>=.5,"1","0"),np.where(frame.automl>=.5,"1","0"))
    frame.to_csv(R/"oof.csv",index=False)
    frame.iloc[np.argsort(-frame.delta.abs())[:10]].merge(X,left_on="id",right_index=True).to_csv(R/"disagreements.csv",index=False)
    importance=pd.DataFrame(importance)
    importance.to_csv(R/"importance.csv",index=False)
    for branch in finalists:
        pd.DataFrame(m.confusion_matrix(frame.truth,frame[branch]>=.5)).to_csv(R/f"confusion_{branch}.csv",index=False)
        frac,pred=calibration_curve(frame.truth,frame[branch],n_bins=8,strategy="quantile")
        plt.plot(pred,frac,"o-",label=branch)
    plt.plot([0,1],[0,1],"k--");plt.legend()
    save_plot(F/"calibration.png","Калібрування OOF","Середній прогноз","Частка позитивних")
    for _,row in wide.iterrows(): plt.plot(["Manual","FLAML"],[row.manual,row.automl],"o-")
    save_plot(F/"paired.png","Парна перевірка на зовнішніх folds","Гілка","Average precision")
    wide.boxplot();save_plot(F/"box.png","Розкид AP на п’яти folds","Гілка","Average precision")
    interval=np.quantile(np.random.default_rng(seed).choice(diff,(cfg["bootstrap"],len(diff)),replace=True).mean(axis=1),[.025,.975])
    equivalent=abs(diff.mean())<=cfg["epsilon"]
    winner=eng.bytes.idxmin() if equivalent else means.ap.idxmax()
    model=finalists[winner];est=model.steps[-1][1]
    decision=dict(branch=winner,family=type(est).__name__,params=est.get_params(),threshold=cfg["threshold"],positive_class=1,
                  test_metrics_present=False,mean_ap=means.ap.to_dict(),delta_automl_minus_manual=diff.mean(),interval=interval,
                  reason="smallest serialized size within epsilon" if equivalent else "higher outer AP beyond epsilon",
                  epsilon=cfg["epsilon"],scope="red Vinho Verde; human review only")
    write_json(H/"selection_decision.json",decision)
    (H/"decision.sha256").write_text(digest(H/"selection_decision.json")+"\n")
    copyfile(R/f"{winner}.joblib",H/"model.joblib")
    write_json(H/"test_metrics.json",metrics(y.loc[te],model.predict_proba(X.loc[te])[:,1]))
    write_json(H/"data_schema.json",dict(version="1.0",features=[dict(name=k,type="number",min=train[k].min(),max=train[k].max()) for k in X]))
    row=train.iloc[[0]]
    write_json(H/"inference_example.json",dict(features=row.iloc[0].to_dict(),probability=model.predict_proba(row)[0,1]))
    copyfile(ROOT/"config.json",H/"experiment_config.json")
    (H/"requirements.txt").write_text("scikit-learn==1.8.0\npandas==2.2.3\nnumpy==2.3.5\nscipy==1.17.0\njoblib==1.5.3\nthreadpoolctl==3.6.0\n")
    (H/"MODEL_CARD.md").write_text(f"# Пріоритизація лабораторних зразків вина\n\n{decision['family']}; поріг 0,5. Позитивний клас — сенсорна оцінка ≥ 7. Для навчального сортування зразків експертом. Інші регіони, роки врожаю й білі вина потребують зовнішньої валідації. Прогноз не сертифікує якість.\n\nCV AP {means.loc[winner,'ap']:.4f}; test AP {read_json(H/'test_metrics.json')['ap']:.4f}.\n")
    write_json(H/"manifest.json",{p.name:digest(p) for p in sorted(H.iterdir()) if p.name!="manifest.json"})


if __name__=="__main__":
    with threadpool_limits(1): run()

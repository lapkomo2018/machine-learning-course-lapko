from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ml_project.predictor import Predictor
from src.experiment_io import digest,read_json,write_json,save_plot

ROOT=Path(__file__).resolve().parent; H=ROOT.parent/"lab06/handoff"; R=ROOT/"results"; F=ROOT/"figures"


def audited_predict(predictor,record):
    """Strict schema/order plus the non-negative physical measurement domain."""
    predictor.validate(record)
    if any(v<0 for v in record.values()): raise ValueError("Physical measurement below zero")
    return predictor.predict_record(record)


def quality(y,p):
    return dict(ap=average_precision_score(y,p),roc=roc_auc_score(y,p),f1=f1_score(y,p>=.5,zero_division=0))


def run():
    R.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
    cfg=read_json(ROOT/"config.json"); model_path=H/"model.joblib"
    if digest(model_path)!=cfg["source_model_sha256"]: raise ValueError("Wrong audit target")
    model=joblib.load(model_path);predictor=Predictor()
    raw=pd.read_csv(ROOT.parent/"data/winequality-red.csv",sep=";")
    X=raw.drop(columns="quality");y=(raw.quality>=7).astype(int)
    tr,te=[pd.read_csv(H/f"{k}_ids.csv").id for k in ("train","test")]
    audit=dict(rows=len(X),features=X.shape[1],missing=int(X.isna().sum().sum()),duplicates=int(X.duplicated().sum()),
               id_overlap=len(set(tr)&set(te)),feature_overlap=len(set(map(tuple,X.loc[tr].values))&set(map(tuple,X.loc[te].values))),
               test_positive=int(y.loc[te].sum()),train_positive=int(y.loc[tr].sum()),data_sha256=digest(ROOT.parent/"data/winequality-red.csv"))
    write_json(R/"data_audit.json",audit)
    X.loc[tr].corrwith(y.loc[tr]).rename("correlation").to_csv(R/"correlations.csv")
    p=model.predict_proba(X.loc[te])[:,1]
    post=np.array([audited_predict(predictor,r)["probability"] for r in X.loc[te].to_dict("records")])
    rows=[dict(version=k,**quality(y.loc[te],v)) for k,v in (("original",p),("guarded",post),("constant",np.full(len(te),y.loc[tr].mean())))]
    pd.DataFrame(rows).to_csv(R/"independent_metrics.csv",index=False)
    pd.DataFrame(confusion_matrix(y.loc[te],p>=.5)).to_csv(R/"confusion.csv",index=False)
    e=X.loc[te].copy().assign(truth=y.loc[te],p=p,predicted=(p>=.5).astype(int))
    e["distance_to_threshold"]=(e.p-.5).abs()
    errors=e[e.predicted!=e.truth].nlargest(5,"distance_to_threshold")
    near=e.drop(errors.index).nsmallest(5,"distance_to_threshold")
    pd.concat([errors.assign(reason="confident_error"),near.assign(reason="near_threshold")]).to_csv(R/"cases.csv",index_label="id")
    cv=[]; ranks=[]; pred={}
    for seed in cfg["seeds"]:
        oof=pd.Series(index=tr,dtype=float)
        for fold,(a,b) in enumerate(StratifiedKFold(cfg["folds"],shuffle=True,random_state=seed).split(X.loc[tr],y.loc[tr])):
            xa,xb=tr.iloc[a],tr.iloc[b]
            trained=clone(model).fit(X.loc[xa],y.loc[xa])
            vals=trained.predict_proba(X.loc[xb])[:,1]
            oof.loc[xb]=vals
            cv.append(dict(seed=seed,fold=fold,**quality(y.loc[xb],vals)))
            imp=permutation_importance(trained,X.loc[xb],y.loc[xb],scoring="average_precision",random_state=seed,n_repeats=2)
            ranks.extend(dict(seed=seed,fold=fold,feature=k,importance=v) for k,v in zip(X.columns,imp.importances_mean))
        pred[seed]=oof
    cv=pd.DataFrame(cv);cv.to_csv(R/"repeat_cv.csv",index=False)
    pd.DataFrame(ranks).to_csv(R/"importance.csv",index=False)
    pd.DataFrame([dict(a=a,b=b,mean_probability_change=(pred[a]-pred[b]).abs().mean(),
                       decision_change=((pred[a]>=.5)!=(pred[b]>=.5)).mean()) for a,b in combinations(cfg["seeds"],2)]).to_csv(R/"repeat_disagreement.csv",index=False)
    ci=np.quantile(np.random.default_rng(7).choice(cv.ap,(2000,len(cv)),replace=True).mean(axis=1),[.025,.975])
    write_json(R/"uncertainty.json",dict(mean=cv.ap.mean(),sd=cv.ap.std(),min=cv.ap.min(),max=cv.ap.max(),descriptive_interval=ci))
    cv.boxplot(column="ap",by="seed");plt.suptitle("");save_plot(F/"repeats.png","Розкид AP замороженої конфігурації","Seed","AP")
    evidence=read_json(H/"test_metrics.json")
    stages=pd.read_csv(H.parent/"results/budget.csv")
    oof=pd.read_csv(H.parent/"results/oof.csv")
    claims=[("C1","Протокол ЛР6","ID і вектори не повторюються між частинами","data_audit.json",audit["id_overlap"]==audit["feature_overlap"]==0),
            ("C2","Таблиця test","AP, ROC-AUC, F1 відтворюються з допуском 1e-10","independent_metrics.csv",all(abs(rows[0][k]-evidence[k])<=cfg["tolerance"] for k in ("ap","roc","f1"))),
            ("C3","Пошуковий журнал","Кожен пошук має 12 trials і 36 inner fits","../lab06/results/budget.csv",len(stages)==16 and (stages.inner_fits==36).all()),
            ("C4","Таблиця OOF","Один прогноз кожної гілки на train ID","../lab06/results/oof.csv",oof.id.is_unique and set(oof.id)==set(tr)),
            ("C5","Модельна картка","Сортування краще сталого baseline на test","independent_metrics.csv",rows[0]["ap"]>rows[2]["ap"]),
            ("C6","Гіпотеза аудиту","Схема зберігає коректні прогнози","independent_metrics.csv",np.allclose(p,post,atol=1e-10,rtol=0))]
    pd.DataFrame(claims,columns=["id","source","criterion","evidence","confirmed"]).to_csv(R/"claims.csv",index=False)
    files=[model_path,H/"data_schema.json",H/"selection_decision.json",H/"requirements.txt",H/"test_metrics.json",H.parent/"README.md",H.parent/"AI_USAGE.md",H.parent/"main.py",ROOT.parent/"data/winequality-red.csv"]
    pd.DataFrame([dict(file=str(f.relative_to(ROOT.parent)),bytes=f.stat().st_size,sha256=digest(f)) for f in files]).to_csv(R/"inventory.csv",index=False)
    findings=[("F1","high","Числовий рядок проходив через sklearn","negative_cases.csv","Сувора схема до inference","fixed"),
              ("F2","medium","Зміна порядку полів давала помилку sklearn","negative_cases.csv","Зібрати поля за схемою","fixed"),
              ("F3","high","Від’ємний відсоток алкоголю отримував прогноз","negative_cases.csv","Перевірити фізичну область","fixed"),
              ("F4","medium","Test AP нижча за вкладену CV","independent_metrics.csv","Перевірити нову партію, лишити рішення експерту","open"),
              ("F5","medium","Немає ID виробника та партії","data_audit.json","Зібрати такі ID для групової валідації","open")]
    pd.DataFrame(findings,columns=["id","severity","problem","evidence","mitigation","status"]).to_csv(R/"findings.csv",index=False)
    row=X.loc[te].iloc[0].to_dict()
    variants={"str":dict(row,alcohol=str(row["alcohol"])),"negative":dict(row,alcohol=-1),"order":dict(reversed(list(row.items()))),
              "nan":dict(row,alcohol=np.nan),"inf":dict(row,alcohol=np.inf),"extra":dict(row,quality=8),
              "missing":{k:v for k,v in row.items() if k!="alcohol"},"out_of_range":dict(row,alcohol=25)}
    checks=[]
    for case,record in variants.items():
        for version in ("original","guarded"):
            try:
                if version=="original":model.predict_proba(pd.DataFrame([record]));out="predicted"
                else:out="warning" if audited_predict(predictor,record)["warnings"] else "predicted"
            except (ValueError,TypeError):out="rejected"
            checks.append((case,version,out))
    pd.DataFrame(checks,columns=["case","version","outcome"]).to_csv(R/"negative_cases.csv",index=False)
    write_json(R/"verdict.json",dict(verdict="прийнято з обмеженнями",claims=[c[0] for c in claims if c[-1]],
               fixed=["F1","F2","F3"],open=["F4","F5"],max_probability_change=np.max(abs(p-post)),model_sha256=digest(model_path)))
    write_json(R/"file_manifest.json",{str(f.relative_to(ROOT.parent)):digest(f) for f in files})


if __name__=="__main__":
    with threadpool_limits(1): run()

from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from src.experiment_io import read_json,write_json,digest,save_plot
from lab05.main import A as SOURCE, R as SOURCE_RESULTS,align

ROOT=Path(__file__).resolve().parent
R,F=ROOT/"results",ROOT/"figures"


def verify(directory):
    for k,h in read_json(directory/"manifest.json").items():
        if digest(directory/k)!=h: raise ValueError(f"checksum mismatch: {k}")


def validate(frame,images,k,clusters):
    if frame.id.duplicated().any() or len(frame)!=len(images) or len(clusters)!=len(images): raise ValueError("IDs/length")
    if not np.isfinite(frame[["combined","if_full","if_pca","reconstruction"]]).all().all(): raise ValueError("non-finite")
    if not frame.combined.between(0,1).all() or min(clusters)<0 or max(clusters)>=k: raise ValueError("range")


def boundaries(z,centers):
    distances=cdist(z,centers)
    ordered=np.sort(distances,axis=1)
    return pd.DataFrame(dict(cluster=distances.argmin(axis=1),distance=ordered[:,0],margin=ordered[:,1]-ordered[:,0]))


def profile(raw,labels):
    sd=np.where(raw.std(axis=0)>0,raw.std(axis=0),1)
    rows=[]
    for k in np.unique(labels):
        group=raw[labels==k]
        mean=group.mean(axis=0); med=np.median(group,axis=0)
        z=(mean-raw.mean(axis=0))/sd
        rows.extend((k,f,len(group),len(group)/len(raw),mean[f],med[f],z[f],group[:,f].std()) for f in range(raw.shape[1]))
    return pd.DataFrame(rows,columns=["cluster","pixel","n","fraction","mean","median","z","sd"])


def representatives(frame,n):
    rows=[]
    for k,g in frame.groupby("cluster"):
        for name,subset in (("typical",g.nsmallest(n,"distance")),("distant",g.nlargest(n,"distance")),("boundary",g.nsmallest(n,"margin"))):
            rows.append(subset.assign(role=name))
    return pd.concat(rows)


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 1.


def policy(row,c):
    enabled={"rescan":row.combined>=c["combined_review"] and row.if_pca_flag and row.reconstruction_flag,
             "review":row.if_pca_flag or row.reconstruction_flag,
             "inspect_boundary":row.margin<c["margin_review"],"no_action":True}
    triggered=[key for key in c["priority"] if enabled[key] and key!="no_action"]
    return (triggered[0] if triggered else "no_action"),",".join(triggered)


def run():
    R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
    cfg=read_json(ROOT/"config.json");verify(SOURCE)
    b=joblib.load(SOURCE/"bundle.joblib")
    km=b["km_pca"]; centers=km.cluster_centers_
    ids=np.load(SOURCE/"split.npz");tr,te=ids["train"],ids["test"]
    if set(tr)&set(te): raise ValueError("ID overlap")
    raw=load_digits().data
    z=b["pca"].transform(b["scaler"].transform(raw[tr]))
    train=boundaries(z,centers).assign(id=tr)
    labels=train.cluster.to_numpy()
    runs=np.load(SOURCE/"runs_pca.npz")["labels"]
    train["stability"]=(runs==labels).mean(axis=0)
    train.sort_values("stability").to_csv(R/"object_stability.csv",index=False)
    representatives(train,cfg["representatives"]).to_csv(R/"representatives.csv",index=False)
    prof=profile(raw[tr],labels);prof.to_csv(R/"profiles.csv",index=False)
    center_dist=cdist(centers,centers);np.fill_diagonal(center_dist,np.inf)
    summaries=[]
    for k,g in prof.groupby("cluster"):
        fmt=lambda df:"; ".join(f"p{r.pixel} {r.z:+.2f}" for r in df.itertuples())
        summaries.append(dict(cluster=k,n=int(g.n.iloc[0]),positive=fmt(g.nlargest(5,"z")),negative=fmt(g.nsmallest(5,"z")),
                              rms_radius=np.sqrt(np.mean(train.loc[labels==k,"distance"]**2)),nearest=int(center_dist[k].argmin())))
    pd.DataFrame(summaries).to_csv(R/"descriptions.csv",index=False)
    maps=[profile(raw[tr],r).pivot(index="cluster",columns="pixel",values="z") for r in runs]
    stability=[]
    for i,j in combinations(range(len(runs)),2):
        for k in maps[0].index:
            a,c=maps[i].loc[k],maps[j].loc[k]
            stability.append(dict(run_a=i,run_b=j,cluster=k,jaccard=jaccard(a.abs().nlargest(5).index,c.abs().nlargest(5).index),
                                  max_abs_diff=(a-c).abs().max(),top_a=str(a.abs().nlargest(5).index.tolist()),top_b=str(c.abs().nlargest(5).index.tolist())))
    pd.DataFrame(stability).to_csv(R/"profile_stability.csv",index=False)
    plt.figure(figsize=(10,4));plt.imshow(maps[0],aspect="auto",cmap="coolwarm",vmin=-2,vmax=2);plt.colorbar(label="z")
    save_plot(F/"profiles.png","Відхилення середньої яскравості від загального профілю","Піксель (рядковий індекс)","Кластер")
    pcs=[];fig,axes=plt.subplots(3,3,figsize=(6,6))
    for p in range(3):
        load=b["pca"].components_[p];neg=load.argsort()[:5];pos=load.argsort()[-5:][::-1]
        for tag,i in (("min",z[:,p].argmin()),("max",z[:,p].argmax())):
            pcs.append(dict(pc=p+1,extreme=tag,id=tr[i],score=z[i,p],positive=pos.tolist(),negative=neg.tolist(),
                            positive_mean=raw[tr[i],pos].mean(),negative_mean=raw[tr[i],neg].mean()))
        for ax,img,title in zip(axes[p],[raw[tr[z[:,p].argmin()]],load,raw[tr[z[:,p].argmax()]]],["min",f"PC{p+1}","max"]):
            ax.imshow(img.reshape(8,8),cmap="coolwarm" if title.startswith("PC") else "gray");ax.set_title(title);ax.axis("off")
    fig.tight_layout();fig.savefig(F/"components.png");plt.close(fig)
    pd.DataFrame(pcs).to_csv(R/"components.csv",index=False)
    audit=pd.read_csv(SOURCE_RESULTS/"audit.csv");images=np.load(SOURCE/"audit.npz")["images"]
    za=b["pca"].transform(b["scaler"].transform(images));diag=boundaries(za,centers)
    validate(audit,images,len(centers),diag.cluster)
    # Test stability uses the five frozen train partitions and their aligned centers.
    votes=[cdist(za,np.vstack([z[r==k].mean(axis=0) for k in range(len(centers))])).argmin(axis=1) for r in runs]
    cases=pd.concat([audit,diag],axis=1)
    cases["stability"]=(np.stack(votes)==diag.cluster.to_numpy()).mean(axis=0)
    actions=[policy(row,cfg) for row in cases.itertuples()]
    cases["action"],cases["rules"]=zip(*actions)
    cases.to_csv(R/"actions.csv",index=False)
    cards=[]
    for category,subset in (("highest",cases.nlargest(5,"combined")),("disagreement",cases[cases.if_pca_flag!=cases.reconstruction_flag]),
                             ("false_positive",cases[(cases.truth==0)&(cases.combined_flag)])):
        used=[x["id"] for x in cards]
        for row in subset[~subset.id.isin(used)].head(5).to_dict("records"):
            cards.append(dict(**row,category=category,hypothesis="Підозра на дефект штриха" if row["reconstruction_flag"] else "Нетипова форма цифри",
                              limitation="Сигнал не доводить дефект; потрібне порівняння з оригіналом"))
    pd.DataFrame(cards).to_csv(R/"cards.csv",index=False)
    pd.crosstab(cases.action,cases.kind).to_csv(R/"action_composition.csv")
    write_json(R/"input_hashes.json",{str(f.relative_to(ROOT.parent)):digest(f) for f in (SOURCE/"manifest.json",SOURCE_RESULTS/"audit.csv",ROOT/"config.json")})


if __name__=="__main__": run()

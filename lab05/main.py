"""Run from repository root with python -m lab05.main."""
from itertools import combinations
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as mt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.experiment_io import digest, read_json, save_plot, write_json

ROOT = Path(__file__).resolve().parent
R, A, F = (ROOT / d for d in ("results", "artifacts", "figures"))


def select_q(cumulative, target):
    if not 0 < target <= 1:
        raise ValueError("variance threshold must be in (0, 1]")
    return min(len(cumulative), int(np.searchsorted(cumulative, target)) + 1)


def align(ref, labels):
    cross = pd.crosstab(ref, labels)
    rows, cols = linear_sum_assignment(-cross.to_numpy())
    mapping = dict(zip(cross.columns[cols], cross.index[rows]))
    return np.array([mapping[v] for v in labels])


def inject(X, ids, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("invalid corruption fraction")
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(X), round(len(X)*fraction), replace=False)
    images = X[subset].copy().reshape(-1, 8, 8)
    records = []
    for i, (img, idx) in enumerate(zip(images, subset)):
        offset = int(rng.integers(1, 6))
        kind = ("dead_rows", "bright_column", "horizontal_shift")[i % 3]
        if kind == "dead_rows":
            img[offset:offset+2] = 0
        elif kind == "bright_column":
            img[:, offset] = 16
        else:
            img[:, 2:] = img[:, :-2].copy()
            img[:, :2] = 0
            offset = 2
        records.append((int(ids[idx]), kind, offset))
    return images.reshape(-1, 64), pd.DataFrame(records, columns=["source_id","kind","offset"])


def cluster_grid(spaces, config):
    rows, assignments = [], {}
    for name, data in spaces.items():
        for k in config["k_values"]:
            for seed in config["seeds"]:
                start = perf_counter()
                model = KMeans(k, n_init=1, random_state=seed).fit(data)
                elapsed = perf_counter() - start
                labels = model.labels_
                rows.append((name,k,seed,model.inertia_,mt.silhouette_score(data,labels),
                             mt.davies_bouldin_score(data,labels),mt.calinski_harabasz_score(data,labels),elapsed))
                assignments[name,k,seed] = labels
    return pd.DataFrame(rows,columns=["space","k","seed","inertia","silhouette","DB","CH","seconds"]), assignments


def scores(bundle, raw):
    x = bundle["scaler"].transform(raw)
    z = bundle["pca"].transform(x)
    s = pd.DataFrame({"if_full":-bundle["if_full"].score_samples(x),
                      "if_pca":-bundle["if_pca"].score_samples(z),
                      "reconstruction":((x-bundle["pca"].inverse_transform(z))**2).mean(axis=1)})
    if "reference" in bundle:
        s["combined"] = np.mean([np.searchsorted(v,s[key],side="right") / len(v)
                                 for key,v in bundle["reference"].items()],axis=0)
    return s


def run():
    for path in (R,A,F):
        path.mkdir(exist_ok=True)
    cfg = read_json(ROOT/"config.json")
    if cfg["scaler"] != "standard" or not 0 < cfg["quantile"] < 1 or min(cfg["k_values"]) < 2:
        raise ValueError("Invalid scaler, quantile or cluster count")
    dataset = load_digits()
    raw = dataset.data
    tr, te = train_test_split(np.arange(len(raw)),test_size=cfg["test_size"],random_state=cfg["seed"])
    np.savez_compressed(A/"split.npz",train=tr,test=te)
    scaler = StandardScaler().fit(raw[tr])
    x = scaler.transform(raw[tr])
    full = PCA(svd_solver="full").fit(x)
    ev = full.explained_variance_ratio_
    pd.DataFrame({"component":np.arange(1,len(ev)+1),"explained_variance_ratio":ev,"cumulative_variance":ev.cumsum()}).to_csv(R/"variance.csv",index=False)
    plt.plot(np.arange(1,len(ev)+1),ev,label="Окрема компонента")
    plt.plot(np.arange(1,len(ev)+1),ev.cumsum(),label="Сумарно")
    plt.legend()
    save_plot(F/"variance.png","Пояснена дисперсія","Компонента","Частка")
    dims = []
    for target in cfg["thresholds"]:
        start = perf_counter()
        q = select_q(ev.cumsum(),target)
        trial = PCA(q,svd_solver="full").fit(x)
        z = trial.transform(x)
        labels = KMeans(8,n_init=1,random_state=cfg["seed"]).fit_predict(z)
        elapsed = perf_counter()-start
        dims.append((target,q,ev[:q].sum(),np.mean((x-trial.inverse_transform(z))**2),
                     mt.silhouette_score(z,labels),elapsed))
    pd.DataFrame(dims,columns=["threshold","q","variance","mse","silhouette_k8","seconds"]).to_csv(R/"dimensions.csv",index=False)
    pca = PCA(select_q(ev.cumsum(),cfg["variance"]),svd_solver="full").fit(x)
    spaces = {"full":x,"pca":pca.transform(x)}
    grid, runs = cluster_grid(spaces,cfg)
    grid.to_csv(R/"cluster_runs.csv",index=False)
    summary = grid.groupby(["space","k"]).mean(numeric_only=True).reset_index()
    summary.to_csv(R/"cluster_summary.csv",index=False)
    b = {"scaler":scaler,"pca":pca}
    choices, stable = {}, []
    for name, matrix in spaces.items():
        k = int(summary.query("space == @name").nlargest(1,"silhouette").iloc[0].k)
        choices[name] = k
        seeds = cfg["seeds"]
        groups = [runs[name,k,s] for s in seeds]
        aligned = np.stack([align(groups[0],g) for g in groups])
        np.savez_compressed(A/f"runs_{name}.npz",labels=aligned)
        stability = (aligned == aligned[0]).mean(axis=0)
        pd.DataFrame({"id":tr,"cluster":groups[0],"stability":stability}).sort_values("stability").to_csv(R/f"stability_{name}.csv",index=False)
        stable.append((name,k,*[np.mean([metric(a,c) for a,c in combinations(groups,2)])
                               for metric in (mt.adjusted_rand_score,mt.normalized_mutual_info_score)]))
        b["km_"+name] = KMeans(k,n_init=1,random_state=seeds[0]).fit(matrix)
        b["if_"+name] = IsolationForest(n_estimators=cfg["trees"],random_state=cfg["seed"],n_jobs=1).fit(matrix)
    pd.DataFrame(stable,columns=["space","k","ARI","NMI"]).to_csv(R/"stability.csv",index=False)
    write_json(R/"selection.json",{"q":pca.n_components_,"k":choices,"rule":"maximum mean train silhouette; smaller K on ties"})
    # External labels enter only after both K choices and q are frozen.
    y = dataset.target[te]
    test_scaled = scaler.transform(raw[te])
    external = []
    for name, matrix in (("full",test_scaled),("pca",pca.transform(test_scaled))):
        labels = b["km_"+name].predict(matrix)
        pd.crosstab(labels,y).to_csv(R/f"composition_{name}.csv")
        external.append((name,*[metric(y,labels) for metric in (mt.adjusted_rand_score,mt.normalized_mutual_info_score,mt.homogeneity_score,mt.completeness_score)]))
    pd.DataFrame(external,columns=["space","ARI","NMI","homogeneity","completeness"]).to_csv(R/"external.csv",index=False)
    z = spaces["pca"]
    plt.scatter(z[:,0],z[:,1],s=5,c=b["km_pca"].labels_,cmap="tab20")
    save_plot(F/"projection.png","Навчальні кластери","PC1","PC2")
    fig, axes = plt.subplots(1,3,figsize=(7,2.5))
    for i,ax in enumerate(axes):
        ax.imshow(pca.components_[i].reshape(8,8),cmap="coolwarm")
        ax.set_title(f"PC{i+1}")
        ax.axis("off")
    fig.savefig(F/"loadings.png",bbox_inches="tight"); plt.close(fig)
    train_scores = scores(b,raw[tr])
    b["reference"] = {key:np.sort(train_scores[key]) for key in ("if_pca","reconstruction")}
    train_scores = scores(b,raw[tr])
    b["thresholds"] = train_scores.quantile(cfg["quantile"]).to_dict()
    train_scores.assign(id=tr).to_csv(R/"train_scores.csv",index=False)
    damaged, records = inject(raw[te],te,cfg["corruption_fraction"],cfg["seed"])
    records.to_csv(R/"corruptions.csv",index=False)
    audit_raw = np.vstack([raw[te],damaged])
    np.savez_compressed(A/"audit.npz",images=audit_raw)
    s = scores(b,audit_raw)
    truth = np.r_[np.zeros(len(te),int),np.ones(len(damaged),int)]
    meta = pd.DataFrame({"id":[f"clean-{i}" for i in te]+[f"damaged-{i}" for i in records.source_id],
                        "kind":["clean"]*len(te)+records.kind.tolist(),"truth":truth})
    audit = pd.concat([meta,s],axis=1)
    rows = []
    for key in s:
        hit = s[key] >= b["thresholds"][key]
        audit[key+"_flag"] = hit
        cm = mt.confusion_matrix(truth,hit).ravel()
        top = s[key].nlargest(len(damaged)).index
        rows.append((key,b["thresholds"][key],sum(train_scores[key]>=b["thresholds"][key]),
                     mt.roc_auc_score(truth,s[key]),mt.average_precision_score(truth,s[key]),
                     mt.precision_score(truth,hit,zero_division=0),mt.recall_score(truth,hit),mt.f1_score(truth,hit),
                     truth[top].mean(),*cm))
    audit.to_csv(R/"audit.csv",index=False)
    audit.groupby("kind")[[k+"_flag" for k in s]].mean().to_csv(R/"recall_by_kind.csv")
    top1,top2 = [set(s[k].nlargest(len(damaged)).index) for k in ("if_full","if_pca")]
    write_json(R/"agreement.json",{"top_k":len(damaged),"jaccard":len(top1&top2)/len(top1|top2)})
    pd.crosstab(audit.if_full_flag,audit.if_pca_flag).to_csv(R/"disagreements.csv")
    pd.DataFrame(rows,columns=["detector","threshold","train_flags","roc_auc","AP","precision","recall","F1","precision_at_k","TN","FP","FN","TP"]).to_csv(R/"detectors.csv",index=False)
    for label,series in (("Train",train_scores.reconstruction),("Test",s.reconstruction[truth==0]),("Спотворені",s.reconstruction[truth==1])):
        plt.hist(np.log1p(series),bins=30,alpha=.5,density=True,label=label)
    plt.legend()
    save_plot(F/"error.png","Похибка відновлення","log(1 + MSE)","Густина")
    joblib.dump(b,A/"bundle.joblib",compress=3)
    write_json(A/"manifest.json",{p.name:digest(p) for p in sorted(A.iterdir()) if p.name!="manifest.json"})


if __name__ == "__main__":
    with threadpool_limits(1):
        run()

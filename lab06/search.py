from pathlib import Path
from time import perf_counter
import json
import warnings

from flaml import AutoML
from flaml.automl.model import LRL2Classifier, RandomForestEstimator, ExtraTreesEstimator
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),estimator)


class FoldPipeline:
    """FLAML creates this sklearn pipeline anew on each inner training fold."""
    def _fit(self,X_train,y_train,**kwargs):
        start=perf_counter()
        self._model=pipeline(self.estimator_class(**self.params))
        self._model.fit(X_train,y_train)
        return perf_counter()-start


class ScaledLR(FoldPipeline,LRL2Classifier): pass
class ScaledRF(FoldPipeline,RandomForestEstimator): pass
class ScaledET(FoldPipeline,ExtraTreesEstimator): pass
LEARNERS={"lrl2":ScaledLR,"rf":ScaledRF,"extra_tree":ScaledET}


def candidates(seed):
    models=[LogisticRegression(C=c,max_iter=400,random_state=seed) for c in (.03,.3,1,8)]
    models.extend(RandomForestClassifier(n_estimators=48,max_depth=d,min_samples_leaf=l,n_jobs=1,random_state=seed)
                  for d,l in ((4,2),(6,2),(8,1),(None,2)))
    models.extend(HistGradientBoostingClassifier(max_iter=55,learning_rate=lr,max_leaf_nodes=k,random_state=seed)
                  for lr,k in ((.05,7),(.1,7),(.05,15),(.1,15)))
    return [pipeline(m) for m in models]


def manual_search(X,y,splits,seed):
    history=[]
    options=candidates(seed)
    for i,pipe in enumerate(options):
        start=perf_counter()
        ap=[]
        for a,b in splits:
            pipe.fit(X.iloc[a],y.iloc[a])
            ap.append(average_precision_score(y.iloc[b],pipe.predict_proba(X.iloc[b])[:,1]))
        est=pipe.steps[-1][1]
        history.append(dict(trial=i,family=type(est).__name__,params=json.dumps(est.get_params()),ap=np.mean(ap),seconds=perf_counter()-start))
    best=int(np.argmax([r["ap"] for r in history]))
    return clone(options[best]),history


def automl_search(X,y,splitter,cfg,seed,log):
    search=AutoML()
    for name in cfg["automl"]:
        search.add_learner(name,LEARNERS[name])
    # Fixed data seed + supplied splitter. No FLAML preprocessing outside folds.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",FutureWarning)
        search.fit(X_train=X,y_train=y,task="classification",metric="ap",eval_method="cv",
                   split_type=splitter,n_splits=cfg["inner"],max_iter=cfg["budget"],time_budget=-1,
                   n_jobs=1,seed=seed,estimator_list=cfg["automl"],log_file_name=str(log),
                   log_type="all",model_history=False,verbose=0,skip_transform=True,
                   sample=False,ensemble=False,retrain_full=True,keep_search_state=True)
    raw=[json.loads(line) for line in Path(log).read_text().splitlines()]
    history=[dict(trial=r["record_id"],family=r["learner"],params=json.dumps(r["config"]),
                  ap=1-r["validation_loss"],seconds=r["trial_time"]) for r in raw if "record_id" in r]
    if len(history)!=cfg["budget"] or not np.isfinite([r["ap"] for r in history]).all():
        raise RuntimeError(f"Incomplete FLAML budget: {len(history)}")
    return clone(search.model.estimator),history

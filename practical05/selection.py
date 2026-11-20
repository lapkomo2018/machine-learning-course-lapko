from math import isclose, isfinite

import pandas as pd


def validate(cfg):
    criteria = cfg["criteria"]
    if not criteria or not isclose(sum(v["weight"] for v in criteria.values()),1):
        raise ValueError("criterion weights must sum to one")
    for v in criteria.values():
        if (v["direction"] not in ("maximize","minimize") or
            (v["best"]-v["worst"])*(1 if v["direction"]=="maximize" else -1)<=0 or
            not 0 <= v["weight"] <= 1):
            raise ValueError("invalid anchors, direction or weight")
    if any(op not in ("min","max") for limits in cfg["constraints"].values() for op in limits):
        raise ValueError("unknown constraint")


def normalize(value, criterion):
    if not isfinite(value):
        raise ValueError("metric must be finite")
    return min(1,max(0,(value-criterion["worst"])/(criterion["best"]-criterion["worst"])))


def check_constraints(row, constraints):
    reasons = []
    for metric, limits in constraints.items():
        value = row[metric]
        if not isfinite(value):
            raise ValueError(f"non-finite {metric}")
        for op, bound in limits.items():
            if (op=="min" and value<bound) or (op=="max" and value>bound):
                reasons.append(f"{metric}={value:.4g} ({op} {bound})")
    return not reasons,reasons


def rank_models(matrix, cfg):
    validate(cfg)
    if matrix.model.duplicated().any():
        raise ValueError("duplicate candidate")
    accepted, rejected = [], []
    for row in matrix.to_dict("records"):
        ok, reasons = check_constraints(row,cfg["constraints"])
        if not ok:
            rejected.append({"model":row["model"],"violations":" | ".join(reasons)})
            continue
        parts = {k:normalize(row[k],v) for k,v in cfg["criteria"].items()}
        score = sum(parts[k]*v["weight"] for k,v in cfg["criteria"].items())
        accepted.append(dict(model=row["model"],score=score,**parts))
    ranked = pd.DataFrame(accepted,columns=["model","score",*cfg["criteria"]]).sort_values(["score","model"],ascending=[False,True])
    ranked.insert(0,"rank",range(1,len(ranked)+1))
    return ranked,pd.DataFrame(rejected,columns=["model","violations"])

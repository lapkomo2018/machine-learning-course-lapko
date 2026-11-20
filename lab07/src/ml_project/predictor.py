"""Validated inference over the trusted model bundle installed with the package."""
import hashlib
import json
import math
from pathlib import Path

import joblib
import pandas as pd

DEFAULT=Path(__file__).parent/"assets/project_config.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root,name):
    if not isinstance(name,str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Invalid artifact path")
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("Artifact outside package")
    return p


def config(path):
    cfg=read(path)
    for key in ("model","schema","metadata","decision","model_version","schema_version"):
        if not isinstance(cfg.get(key),str) or not cfg[key]: raise ValueError("Invalid configuration: "+key)
    if (type(cfg.get("max_batch_size")) is not int or cfg["max_batch_size"]<1 or
        type(cfg.get("positive_class")) is not int or
        type(cfg.get("threshold")) not in (int,float) or not 0<cfg["threshold"]<1):
        raise ValueError("Invalid prediction limits")
    return cfg


class Predictor:
    def __init__(self,path=DEFAULT):
        path=Path(path)
        self.cfg=config(path)
        base=path.parent
        manifest=read(base/"manifest.json")
        required={path.name,*(self.cfg[k] for k in ("metadata","model","schema","decision"))}
        if not required.issubset(manifest): raise ValueError("Incomplete manifest")
        for name,expected in manifest.items():
            actual=sha(local(base,name))
            if actual!=expected: raise ValueError("Artifact checksum mismatch")
        self.metadata=read(local(base,self.cfg["metadata"]))
        if self.metadata["predictor_sha256"]!=sha(Path(__file__)): raise ValueError("Predictor checksum mismatch")
        self.schema=read(local(base,self.cfg["schema"]))
        decision=read(local(base,self.cfg["decision"]))
        if (decision["threshold"]!=self.cfg["threshold"] or self.schema["version"]!=self.cfg["schema_version"] or
            decision["positive_class"]!=self.cfg["positive_class"] or self.metadata["model_version"]!=self.cfg["model_version"]):
            raise ValueError("Inconsistent frozen contract")
        self.model=joblib.load(local(base,self.cfg["model"]))
        self.features=[f["name"] for f in self.schema["features"]]
        if self.features!=list(self.model.feature_names_in_) or type(self.model.steps[-1][1]).__name__!=decision["family"]:
            raise ValueError("Model/schema mismatch")
        self.positive_index=list(self.model.classes_).index(self.cfg["positive_class"])

    def describe(self):
        return {k:self.metadata[k] for k in ("model_version","schema_version","family","model_sha256","feature_count","threshold","positive_class")}

    def validate(self,record):
        if not isinstance(record,dict) or set(record)!=set(self.features):
            raise ValueError("Expected exactly the schema fields")
        values,warnings=[],[]
        for f in self.schema["features"]:
            key=f["name"]; value=record[key]
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError("Expected a finite number for "+key)
            values.append(value)
            if not f["min"]<=value<=f["max"]: warnings.append("outside_train_range:"+key)
        return values,warnings

    def predict_batch(self,records):
        if not isinstance(records,list) or not 1<=len(records)<=self.cfg["max_batch_size"]:
            raise ValueError("Invalid batch size")
        values,warnings=zip(*(self.validate(r) for r in records))
        frame=pd.DataFrame(list(values),columns=self.features)
        probs=self.model.predict_proba(frame)[:,self.positive_index]
        return [dict(probability=float(p),predicted_class=int(p>=self.cfg["threshold"]),threshold=self.cfg["threshold"],
                     positive_class=self.cfg["positive_class"],model_version=self.cfg["model_version"],
                     schema_version=self.schema["version"],warnings=w) for p,w in zip(probs,warnings)]

    def predict_record(self,record):
        return self.predict_batch([record])[0]

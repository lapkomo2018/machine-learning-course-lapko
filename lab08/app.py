import logging
from collections import Counter, deque
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from ml_project.predictor import Predictor, DEFAULT

class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str

class Single(Envelope):
    features: dict

class Batch(Envelope):
    records: list[dict]

class Prediction(BaseModel):
    probability: float
    predicted_class: int
    threshold: float
    positive_class: int
    model_version: str
    schema_version: str
    warnings: list[str]

class SingleResult(Prediction):
    request_id: str

class BatchResult(BaseModel):
    request_id: str
    count: int
    predictions: list[Prediction]

class Unavailable(Exception): pass

def create_app(config_path=DEFAULT):
    @asynccontextmanager
    async def lifespan(app):
        try: app.state.predictor=Predictor(config_path)
        except (OSError,ValueError,KeyError,TypeError): app.state.predictor=None
        yield
    app=FastAPI(title="Wine Quality",version="1.0.0",lifespan=lifespan)
    counts=Counter()
    durations=deque(maxlen=2048)
    warnings=0
    def error(req,status,code):
        return JSONResponse({"error":code,"request_id":req.state.request_id},status_code=status)
    @app.middleware("http")
    async def observe(req,call_next):
        req.state.request_id=uuid4().hex
        req.state.batch_size=0
        start=perf_counter()
        try: resp=await call_next(req)
        except Exception: resp=error(req,500,"inference_failed")
        elapsed=(perf_counter()-start)*1000
        resp.headers["X-Request-ID"]=req.state.request_id
        counts[resp.status_code]+=1
        durations.append(elapsed)
        logging.getLogger("wine.api").info("id=%s route=%s status=%d ms=%.2f n=%d",req.state.request_id,req.scope.get("route").path if req.scope.get("route") else "unmatched",resp.status_code,elapsed,req.state.batch_size)
        return resp
    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValueError)
    async def invalid(req,exc): return error(req,422,"invalid_request")
    @app.exception_handler(Unavailable)
    async def unavailable(req,exc): return error(req,503,"model_unavailable")
    def predictor():
        if app.state.predictor is None: raise Unavailable()
        return app.state.predictor
    @app.get("/health")
    async def health(): return {"alive":True}
    @app.get("/ready")
    async def ready(): predictor(); return {"ready":True}
    @app.get("/metadata")
    async def metadata(): return predictor().describe()
    @app.get("/metrics")
    async def metrics():
        total=sum(counts.values());errors=sum(n for k,n in counts.items() if k>=400)
        return dict(requests=total,errors=errors,error_rate=errors/max(1,total),status_counts=counts,warnings=warnings,
                    mean_ms=float(np.mean(durations)) if durations else 0,percentiles_ms=np.percentile(durations,[50,95,99]).tolist() if durations else [0,0,0])
    def infer(req,version,records):
        nonlocal warnings
        p=predictor()
        if version!=p.schema["version"]: raise ValueError("schema")
        req.state.batch_size=len(records)
        result=p.predict_batch(records)
        warnings+=sum(len(r["warnings"]) for r in result)
        return result
    responses={422:{"description":"Invalid request"},503:{"description":"Model unavailable"},500:{"description":"Internal error"}}
    @app.post("/predict",response_model=SingleResult,responses=responses)
    async def single(payload:Single,req:Request):
        return dict(request_id=req.state.request_id,**infer(req,payload.schema_version,[payload.features])[0])
    @app.post("/predict-batch",response_model=BatchResult,responses=responses)
    async def batch(payload:Batch,req:Request):
        rows=infer(req,payload.schema_version,payload.records)
        return dict(request_id=req.state.request_id,count=len(rows),predictions=rows)
    return app

app=create_app()

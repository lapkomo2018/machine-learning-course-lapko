from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform
from shutil import copytree
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter,sleep

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from lab08.app import create_app
from ml_project.predictor import DEFAULT,Predictor,read
from src.experiment_io import write_json

ROOT=Path(__file__).resolve().parent
R=ROOT/"results"
LAB=ROOT.parent/"lab08"
C=read(ROOT/"config.json")


def drills():
    rows=[]
    for case,name in [("missing","model.joblib"),("damaged","model.joblib"),("schema","data_schema.json"),("config","project_config.json")]:
        with TemporaryDirectory() as t:
            dest=Path(t)/"assets";copytree(DEFAULT.parent,dest)
            file=dest/name
            original=file.read_bytes()
            if case=="missing":file.unlink()
            else:file.write_bytes(b"{}")
            with TestClient(create_app(dest/DEFAULT.name)) as client:
                live=client.get("/health").status_code;ready=client.get("/ready").status_code
                error=client.get("/metadata").json()["error"]
            file.write_bytes(original)
            with TestClient(create_app(dest/DEFAULT.name)) as client: recovered=client.get("/ready").status_code
            rows.append(dict(drill=case,artifact=name,health=live,ready=ready,signal=error,recovered=recovered))
    return pd.DataFrame(rows)


def benchmark(url):
    raw=[]
    row=read(DEFAULT.parent/"request.json")
    with httpx.Client(base_url=url,trust_env=False,timeout=15) as client:
        for _ in range(C["warmup"]):
            client.post("/predict",json=dict(schema_version="1.0",features=row)).raise_for_status()
        for mode in ("single","batch","burst"):
            size=C["batch_size"] if mode=="batch" else 1
            endpoint="/predict-batch" if size>1 else "/predict"
            body=dict(schema_version="1.0",**({"features":row} if size==1 else {"records":[row]*size}))
            def request(i):
                start=perf_counter()
                r=client.post(endpoint,json=body)
                return dict(mode=mode,repeat=i,status=r.status_code,ms=1000*(perf_counter()-start),records=size)
            start=perf_counter()
            with ThreadPoolExecutor(max_workers=C["concurrency"] if mode=="burst" else 1) as pool:
                rows=list(pool.map(request,range(C[mode])))
            elapsed=perf_counter()-start
            raw.extend(dict(**v,wall_seconds=elapsed) for v in rows)
        write_json(R/"metrics.json",client.get("/metrics").json())
    df=pd.DataFrame(raw)
    df.to_csv(R/"benchmark_raw.csv",index=False)
    groups=[]
    for mode,d in df.groupby("mode"):
        groups.append(dict(mode=mode,requests=len(d),p50=d.ms.median(),p95=d.ms.quantile(.95),p99=d.ms.quantile(.99),
                        rows_per_s=d.records.sum()/d.wall_seconds.iloc[0],success=(d.status==200).mean(),errors500=(d.status==500).sum()))
    return pd.DataFrame(groups).set_index("mode")


def run():
    R.mkdir(exist_ok=True);(LAB/"results").mkdir(exist_ok=True)
    app=create_app(); write_json(LAB/"openapi.json",app.openapi())
    p=Predictor()
    golden=ROOT/"golden.json"
    if not golden.exists():
        frame=pd.read_csv(ROOT.parent/"data/winequality-red.csv",sep=";")
        ids=pd.read_csv(DEFAULT.parent/"train_ids.csv").id.iloc[:8]
        records=frame.loc[ids,p.features].to_dict("records")
        write_json(golden,[dict(id=int(i),features=r,expected=p.predict_record(r)) for i,r in zip(ids,records)])
    subprocess.run([sys.executable,"-m","pytest","-q","lab07","lab08","practical08"],cwd=ROOT.parent,check=True)
    outcomes=drills();outcomes.to_csv(R/"drills.csv",index=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    url=f"http://127.0.0.1:{port}"
    proc=subprocess.Popen([sys.executable,"-m","uvicorn","lab08.app:app","--host","127.0.0.1","--port",str(port),"--no-access-log"],cwd=ROOT.parent)
    try:
        with httpx.Client(trust_env=False) as c:
            for _ in range(120):
                try:
                    if c.get(url+"/ready").status_code==200: break
                except httpx.RequestError: pass
                sleep(.1)
            else: raise RuntimeError("server did not become ready")
        stats=benchmark(url)
    finally:
        proc.terminate();proc.wait(timeout=10)
    stats.to_csv(R/"benchmark.csv");stats.to_csv(LAB/"results/benchmark.csv")
    actual=dict(single_p95_ms=stats.loc["single","p95"],batch_p95_ms=stats.loc["batch","p95"],success_rate=stats.success.min(),unexpected_500=int(stats.errors500.sum()))
    passed={k:(v>=C["slo"][k] if k=="success_rate" else v<=C["slo"][k]) for k,v in actual.items()}
    ok=all(passed.values()) and all(outcomes.ready==503) and all(outcomes.recovered==200)
    write_json(R/"verdict.json",dict(actual=actual,targets=C["slo"],passed=passed,ready=bool(ok),python=platform.python_version(),platform=platform.platform(),scope="локальний навчальний прототип"))
    if not ok: raise SystemExit(1)


if __name__=="__main__": run()

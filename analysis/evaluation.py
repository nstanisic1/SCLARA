"""Finite conditional-estimator evaluation on the current Swedish design.

Synthetic labels test locality mixtures, source multiplicity and conditional
intervals. The observed archaeological design supplies covariates, not a
historical truth. All conditions and adverse outcomes remain in the result.
"""
from pathlib import Path
import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import sqlite3
import time
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from analysis.infer import arrays, encode, primary_designs, read, save, sha
from analysis.model import LabelMixture


def prepare(data, measurements, specification, extension, output):
    if (output/"fixture.json").exists():
        raise FileExistsError("A frozen evaluation fixture exists")
    output.mkdir(parents=True,exist_ok=True)
    spec,extension_data=read(specification),read(extension)
    contract=extension_data["method_benchmark"]
    design=primary_designs(measurements,spec)["linear_0"]
    cohort=arrays(measurements/"common_cohort.npz")
    original=read(data/"paternal.json")
    units={u["unit_id"]:u for u in original["analysis_units"]}
    multiplicity=np.array([len(units[u]["record_ids"]) for u in cohort["unit_ids"]])
    blocks,block_index=np.unique(cohort["blocks"],return_inverse=True)
    k=len(design["categories"]);p=design["x"].shape[1]
    rng=np.random.default_rng(contract["seed_base"])
    beta=np.zeros((k,p))
    proportions=(np.bincount(design["y"],minlength=k)+1)/(len(design["y"])+k)
    beta[:,0]=np.log(proportions/proportions[0])
    beta[1:,1:-12]=rng.normal(0,.08,(k-1,p-13))
    contrast=np.asarray(spec["common"]["design"]["lineage_contrasts"]["I1_vs_R1a_Z284"])
    direction=contrast/(contrast@contrast)
    direction-=direction[0]
    effect=np.zeros_like(beta)
    periods=spec["common"]["design"]["periods"]
    for period,sign in (("Late_Neolithic",.5),("Roman_Iron_Age",-.5)):
        column=p-12+periods.index(period)*3+2
        effect[:,column]=sign*direction
    vectors=np.array([design["contrasts"][name] for name in contract["focal_estimands"]])
    expected=vectors@effect[1:].ravel()
    if not np.allclose(expected,[.5,1],rtol=0,atol=1e-12):
        raise ValueError("Synthetic generating contrasts do not match their declaration")
    np.savez_compressed(
        output/"fixture.npz",x=design["x"],counts=design["counts"],scales=design["scales"],
        vectors=vectors,nuisance=beta,effect_direction=effect,multiplicity=multiplicity,
        block_index=block_index,
    )
    save(output/"fixture.json",dict(
        contract=contract,categories=design["categories"],blocks=len(blocks),
        units=len(design["y"]),locality_rows=len(design["x"]),parameters=(k-1)*p,
        nuisance_rule="Laplace-smoothed empirical category proportions for intercepts; seeded N(0,0.08^2) nonarchaeological nonreference coefficients. No fitted archaeological slopes enter the generator.",
        evaluation="Conditional modes and inverse-Hessian intervals; no MCMC or historical-selection calibration.",
        heldout_rule="An independently drawn label at each fixed design unit, conditional on the same simulated block effects. This is not geographical extrapolation.",
        input_hashes={str(q):sha(q) for q in (data/"paternal.json",specification,extension,measurements/"common.npz",measurements/"common_cohort.npz")},
        fixture_sha256=sha(output/"fixture.npz"),preparation_sha256=sha(Path(__file__)),
    ))
    print("Frozen",len(contract["conditions"]),"conditions;",contract["maximum_method_fits"],"method records",flush=True)


def initialize(directory):
    global _CONTEXT, _THREAD_LIMIT
    _THREAD_LIMIT=threadpool_limits(limits=1)
    directory=Path(directory)
    meta=read(directory/"fixture.json")
    if sha(directory/"fixture.npz")!=meta["fixture_sha256"]:
        raise ValueError("Changed evaluation fixture")
    fixture=arrays(directory/"fixture.npz")
    fixture["starts"]=np.r_[0,np.cumsum(fixture["counts"])[:-1]]
    fixture["mean_x"]=np.add.reduceat(fixture["x"],fixture["starts"],axis=0)/fixture["counts"][:,None]
    _CONTEXT=dict(meta=meta,fixture=fixture)


def generated(condition_index,replicate):
    meta,fixture=_CONTEXT["meta"],_CONTEXT["fixture"]
    contract=meta["contract"];condition=contract["conditions"][condition_index]
    rng=np.random.default_rng(np.random.SeedSequence([contract["seed_base"],condition_index,replicate]))
    x=fixture["mean_x"] if condition["localities"]=="unit_mean" else fixture["x"]
    counts=np.ones(meta["units"],dtype=int) if condition["localities"]=="unit_mean" else fixture["counts"]
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    beta=fixture["nuisance"]+condition["effect"]*fixture["effect_direction"]
    logits=x@beta.T
    if condition["block_sd"]:
        block=rng.normal(0,condition["block_sd"],(meta["blocks"],len(meta["categories"])))
        logits+=block[np.repeat(fixture["block_index"],counts)]
    local=softmax(logits,axis=1)
    truth=np.add.reduceat(local,starts,axis=0)/counts[:,None]
    cumulative=np.cumsum(truth,axis=1)
    y=(rng.random(meta["units"])[:,None]>cumulative).sum(1)
    heldout=(rng.random(meta["units"])[:,None]>cumulative).sum(1)
    multiplicity=fixture["multiplicity"] if condition["multiplicity"]=="observed_source_records" else np.ones(meta["units"])
    return x,counts,starts,y,heldout,truth,multiplicity,beta[1:].ravel()


def evaluate(task):
    ci,replicate=task
    meta,fixture=_CONTEXT["meta"],_CONTEXT["fixture"]
    cfg=meta["contract"];condition=cfg["conditions"][ci]
    x,counts,starts,y,testy,truth,multiplicity,beta=generated(ci,replicate)
    target=fixture["vectors"]@beta
    mean_x=np.add.reduceat(x,starts,axis=0)/counts[:,None]
    results=[];cache={}
    for method in cfg["methods"]:
        if method=="unit_mean_locality":
            xx=mean_x;cc=np.ones(len(y),dtype=int);weights=np.ones(len(y))
        else:
            xx=x;cc=counts
            weights=multiplicity if method=="source_record_weighted_mixture" else np.ones(len(y))
        signature=hashlib.sha256(xx.tobytes()+cc.tobytes()+weights.tobytes()).hexdigest()
        if signature in cache:
            previous=cache[signature]
            detail=dict(previous["detail"],reused_from=previous["method"],optimization_computed=False)
            results.append((condition["name"],method,replicate,previous["status"],detail,previous["blob"]))
            continue
        started=time.monotonic()
        model=LabelMixture(xx,y,cc,fixture["scales"],len(meta["categories"]),weights)
        try:
            fit=model.optimize()
        except (ValueError,np.linalg.LinAlgError) as error:
            detail=dict(error=str(error),optimization_computed=True,seconds=time.monotonic()-started)
            results.append((condition["name"],method,replicate,"failed",detail,None))
            cache[signature]=dict(method=method,status="failed",detail=detail,blob=None)
            continue
        estimate=fixture["vectors"]@fit["mode"]
        variance=np.einsum("ij,ij->i",fixture["vectors"]@fit["covariance"],fixture["vectors"])
        if np.any(variance<0):
            raise ValueError("Negative fitted contrast variance")
        sd=np.sqrt(variance);lower=estimate-1.959963984540054*sd;upper=estimate+1.959963984540054*sd
        logits=np.column_stack([np.zeros(len(xx)),xx@fit["mode"].reshape(len(meta["categories"])-1,-1).T])
        local=softmax(logits,axis=1)
        ss=np.r_[0,np.cumsum(cc)[:-1]]
        predicted=np.add.reduceat(local,ss,axis=0)/cc[:,None]
        logp=np.log(np.maximum(predicted,np.finfo(float).tiny))
        onehot=np.eye(len(meta["categories"]))[testy]
        detail=dict(
            estimates=estimate,standard_errors=sd,truth=target,lower=lower,upper=upper,
            covered=(lower<=target)&(target<=upper),
            interval_excludes_zero=(lower>0)|(upper<0),
            test_log=float(logp[np.arange(len(y)),testy].mean()),
            test_brier=float(((predicted-onehot)**2).sum(1).mean()),
            expected_log=float((truth*logp).sum(1).mean()),
            expected_brier=float((np.sum(predicted**2,axis=1)-2*np.sum(predicted*truth,axis=1)+1).mean()),
            gradient_max=fit["gradient"],minimum_curvature=fit["curvature"],
            iterations=fit["iterations"],seconds=time.monotonic()-started,
            optimization_computed=True,
        )
        stream=io.BytesIO();np.savez_compressed(stream,mode=fit["mode"])
        blob=stream.getvalue()
        results.append((condition["name"],method,replicate,"passed",detail,blob))
        cache[signature]=dict(method=method,status="passed",detail=detail,blob=blob)
    return results


def summaries(db,meta):
    cfg=meta["contract"]
    def mean_se(values):
        a=np.asarray(values,dtype=float)
        return dict(mean=a.mean(axis=0),mcse=a.std(axis=0,ddof=1)/np.sqrt(len(a)))
    db.execute("create table if not exists summaries(condition text,method text,details text,primary key(condition,method))")
    for condition in cfg["conditions"]:
        name=condition["name"];by_method={}
        for method in cfg["methods"]:
            records={int(rep):json.loads(raw) for rep,raw in db.execute(
                "select replicate,details from fits where condition=? and method=? and status='passed'",(name,method)
            )}
            failures=db.execute("select count(*) from fits where condition=? and method=? and status!='passed'",(name,method)).fetchone()[0]
            by_method[method]=records
            rows=list(records.values())
            if not rows:
                result=dict(successful=0,failed=failures)
            else:
                error=np.array([r["estimates"] for r in rows])-np.array([r["truth"] for r in rows])
                result=dict(
                    successful=len(rows),failed=failures,estimands=cfg["focal_estimands"],
                    bias=mean_se(error),rmse=np.sqrt(np.mean(error**2,axis=0)),
                    coverage=mean_se([r["covered"] for r in rows]),
                    interval_exclusion=mean_se([r["interval_excludes_zero"] for r in rows]),
                    interval_width=mean_se([np.array(r["upper"])-r["lower"] for r in rows]),
                    **{key:mean_se([r[key] for r in rows]) for key in ("test_log","test_brier","expected_log","expected_brier")},
                )
            db.execute("insert or replace into summaries values(?,?,?)",(name,method,encode(result)))
        for comparator in cfg["methods"][1:]:
            ids=sorted(set(by_method[cfg["methods"][0]]) & set(by_method[comparator]))
            if not ids:continue
            a,b=by_method[cfg["methods"][0]],by_method[comparator]
            paired={
                key:mean_se([(a[i][key]-b[i][key])*(-1 if "brier" in key else 1) for i in ids])
                for key in ("test_log","test_brier","expected_log","expected_brier")
            }
            db.execute("insert or replace into summaries values(?,?,?)",(
                name,"paired_vs_"+comparator,encode(dict(replicates=len(ids),positive_favours="unit_locality_mixture",**paired))
            ))
    db.commit()


def run(directory,workers):
    initialize(directory);meta=_CONTEXT["meta"];cfg=meta["contract"]
    identity=dict(fixture=sha(directory/"fixture.json"),data=sha(directory/"fixture.npz"),
                  model=sha(Path(__file__).with_name("model.py")),implementation=sha(Path(__file__)))
    db=sqlite3.connect(directory/"evaluation.sqlite")
    db.execute("create table if not exists metadata(key text primary key,value text not null)")
    db.execute("create table if not exists fits(condition text,method text,replicate integer,status text,details text,arrays blob,sha256 text,primary key(condition,method,replicate))")
    old=db.execute("select value from metadata where key='input_hashes'").fetchone()
    if old and json.loads(old[0])!=identity:raise ValueError("Changed evaluation inputs or implementation")
    db.execute("insert or ignore into metadata values(?,?)",("input_hashes",encode(identity)));db.commit()
    done={(condition,int(rep)):n for condition,rep,n in db.execute("select condition,replicate,count(*) from fits group by condition,replicate")}
    if any(n!=len(cfg["methods"]) for n in done.values()):raise ValueError("Incomplete within-replicate method family")
    pending=[(i,j) for i,c in enumerate(cfg["conditions"]) for j in range(cfg["replicates_per_condition"])
             if (c["name"],j) not in done]
    print("Evaluation:",len(pending),"pending replicates;",len(cfg["methods"]),"methods each",flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(directory),)) as pool:
        futures=[pool.submit(evaluate,task) for task in pending]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            for condition,method,rep,status,detail,blob in future.result():
                db.execute("insert into fits values(?,?,?,?,?,?,?)",(
                    condition,method,rep,status,encode(detail),blob,hashlib.sha256(blob).hexdigest() if blob else None
                ))
            db.commit()
            if i%25==0 or i==len(pending):print("Evaluation replicates",i,"/",len(pending),flush=True)
    if db.execute("select count(*) from fits").fetchone()[0]!=cfg["maximum_method_fits"]:
        raise ValueError("Incomplete evaluation family")
    summaries(db,meta)
    failures=db.execute("select count(*) from fits where status!='passed'").fetchone()[0]
    print("Evaluation complete; failed method records:",failures,flush=True)
    db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","run"])
    p.add_argument("--data",type=Path,default=Path("data"))
    p.add_argument("--measurements",type=Path,default=Path("build/measurements"))
    p.add_argument("--specification",type=Path,default=Path("analysis/specification.json"))
    p.add_argument("--extension",type=Path,default=Path("analysis/extension_specification.json"))
    p.add_argument("--output",type=Path,default=Path("build/evaluation"))
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.action=="prepare":prepare(a.data,a.measurements,a.specification,a.extension,a.output)
        else:run(a.output,a.workers)


if __name__=="__main__":
    main()

"""Fixed-design Swedish estimator benchmarks.

These evaluations retain their original designs and outcome-informed conditions.
They evaluate conditional estimation, temporal attribution and prediction; they
neither validate historical selection nor identify lineage arrival dates.
"""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import io
import json
import sqlite3
import time
import numpy as np
from scipy.special import softmax, ndtr, expit, logit
from threadpoolctl import threadpool_limits
from analysis.model import LabelMixture


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(
        value,
        default=serial,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def blob(value):
    stream = io.BytesIO()
    np.savez_compressed(stream, **value)
    return stream.getvalue()


def benchmark_probabilities(a, beta, curved):
    eta = a["X"] @ (a["B"] + beta * a["signal"]).T
    if curved:
        eta = eta + a["field"]
    p = softmax(np.column_stack([np.zeros(len(eta)), eta]), axis=1)
    return np.add.reduceat(p, a["starts"], axis=0) / a["counts"][:, None]


def first_one(job, context):
    scenario, rep = job
    a = context["first"]
    s = context["specification"]["simulation"]
    start = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([s["seed"], rep]))
    zunit = rng.normal(size=len(a["counts"]))
    zblock = rng.normal(size=int(a["blocks"].max() + 1))
    rho = scenario["rho"]
    u = ndtr(np.sqrt(rho) * zblock[a["blocks"]] + np.sqrt(1 - rho) * zunit)
    p = benchmark_probabilities(a, scenario["beta"], scenario["curvature"])
    y = np.minimum((u[:, None] > np.cumsum(p, axis=1)).sum(axis=1), 7)
    output = []
    for method in ["full", "simple"]:
        cols = np.arange(a["X"].shape[1]) if method == "full" else a["simple_cols"]
        vector = a["vector"] if method == "full" else a["simple_vector"]
        X = a["X"][:, cols]
        scales = a["scales"][cols]
        model = LabelMixture(X, y, a["counts"], scales, 8)
        v = (a["B"] + scenario["beta"] * a["signal"])[:, cols].ravel()
        fail = []
        fit = None
        t = time.time()
        for retry in range(2):
            try:
                fit = model.optimize(v if retry == 0 else None)
                break
            except Exception as e:
                fail.append(str(e))
        result = {
            "beta": scenario["beta"],
            "seconds": time.time() - t,
            "retry_messages": fail,
            "success": fit is not None,
            "counts": np.bincount(y, minlength=8).tolist(),
        }
        if fit is not None:
            mean = float(vector @ fit["mode"])
            se = float(np.sqrt(vector @ fit["covariance"] @ vector))
            result.update(
                mean=mean,
                sd=se,
                lower=mean - 1.96 * se,
                upper=mean + 1.96 * se,
                gradient=fit["gradient"],
                curvature=fit["curvature"],
                iterations=fit["iterations"],
            )
        output.append((scenario["id"], rep, method, result))
    return (
        "first",
        scenario["id"],
        rep,
        [
            (method, "passed" if result["success"] else "failed", result, {})
            for _, _, method, result in output
        ],
    )


def benchmark_wilson(k, n):
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {
        "count": int(k),
        "n": n,
        "rate": p,
        "mcse": float(np.sqrt(p * (1 - p) / n)),
        "lower": float(c - r),
        "upper": float(c + r),
    }


def unit_probabilities(X, mode, counts):
    logits = np.column_stack([np.zeros(len(X)), X @ mode.reshape(7, X.shape[1]).T])
    return (
        np.add.reduceat(
            softmax(logits, axis=1), np.r_[0, np.cumsum(counts)[:-1]], axis=0
        )
        / counts[:, None]
    )


def outcome(prob, rng):
    return (rng.random(len(prob))[:, None] > np.cumsum(prob, axis=1)).sum(1)


def performance_scores(prob, trueprob, testy):
    clipped = np.clip(prob, 1e-300, 1)
    onehot = np.eye(8)[testy]
    bins = []
    for j in range(10):
        mask = (prob >= j / 10) & (prob < ((j + 1) / 10) if j < 9 else prob <= 1)
        bins.append(
            dict(
                n=int(mask.sum()),
                predicted_sum=float(prob[mask].sum()),
                true_sum=float(trueprob[mask].sum()),
            )
        )
    return dict(
        test_log=float(np.log(clipped[np.arange(len(testy)), testy]).mean()),
        test_brier=float(((prob - onehot) ** 2).sum(1).mean()),
        expected_log=float((trueprob * np.log(clipped)).sum(1).mean()),
        expected_brier=float(
            (np.sum(prob**2, axis=1) - 2 * np.sum(prob * trueprob, axis=1) + 1).mean()
        ),
        calibration_bins=bins,
    )


def temporal_inference(fit, truth, support, ar):
    V = ar["V"]
    mu = V @ fit["mode"]
    cov = V @ fit["covariance"] @ V.T
    T = ar["T"]
    allmu = T @ mu
    allcov = T @ cov @ T.T
    se = np.sqrt(np.diag(allcov))
    sim = (ar["gauss"] @ np.linalg.cholesky(cov).T @ T.T) / se
    critical = float(np.quantile(np.abs(sim).max(1), 0.95))
    lower = allmu - critical * se
    upper = allmu + critical * se
    positive = lower[:4] > 0
    nominees = []
    for i in range(4):
        differences = []
        for j in range(4):
            if j == i:
                continue
            v = np.eye(4)[i] - np.eye(4)[j]
            differences.append(float(v @ mu - critical * np.sqrt(v @ cov @ v)))
        if positive[i] and min(differences) > 0:
            nominees.append(i)
    assert len(nominees) <= 1
    supported = min(support) >= 5
    nomination = nominees[0] if nominees and supported else None
    status = (
        "insufficient_archaeological_support"
        if not supported
        else (
            "unique_period"
            if nomination is not None
            else (
                "positive_chronology_unresolved"
                if positive.any()
                else "no_positive_relationship_detected"
            )
        )
    )
    details = dict(
        critical=critical,
        nomination=nomination,
        unrestricted_nomination=nominees[0] if nominees else None,
        status=status,
        support=support,
        any_positive=bool(positive.any()),
        any_two_sided=bool(((lower[:4] > 0) | (upper[:4] < 0)).any()),
    )
    if truth is not None:
        t = np.asarray(truth)
        tt = T @ t
        details.update(
            marginal_coverage=(
                (mu - 1.959963984540054 * np.sqrt(np.diag(cov)) <= t)
                & (mu + 1.959963984540054 * np.sqrt(np.diag(cov)) >= t)
            ).tolist(),
            joint_coverage=bool(((lower <= tt) & (upper >= tt)).all()),
        )
        top = np.flatnonzero(t == t.max())
        unique = int(top[0]) if len(top) == 1 and t.max() > 0 else None
        details.update(
            correct_nomination=nomination is not None and nomination == unique,
            wrong_nomination=nomination is not None and nomination != unique,
        )
    return details, dict(
        period_mean=mu,
        period_covariance=cov,
        all_mean=allmu,
        simultaneous_lower=lower,
        simultaneous_upper=upper,
    )


def chronology_one(job, context):
    index, scenario, rep = job
    ar = context["arr"]
    cfg = context["specification"]["chronology"]
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], index, rep]))
    test_rng = np.random.default_rng(
        np.random.SeedSequence([cfg["test_seed"], index, rep])
    )
    X = ar["X"]
    pooled = ar["pooled"]
    genX = X
    support = context["meta"]["support"]
    retention_report = None
    if scenario.get("recording_stress"):
        from analysis.chronology import panel_sets, panel_measurement, designs

        pool = context["pool"]
        u = dict(
            zip(pool["retention_groups"], rng.random(len(pool["retention_groups"])))
        )
        bases = np.array(
            [
                [81 / 148, 91 / 134],
                [101 / 247, 39 / 120],
                [63 / 262, 103 / 469],
                [41 / 1098, 72 / 1481],
            ]
        )
        tally = collections.Counter()

        def retention(r, period):
            f = r["feature_id"]
            base = (
                bases[period, int(f == "hearth")]
                if f in ["posthole", "hearth"]
                else bases[period].mean()
            )
            passed = u[r["retention_group"]] < expit(
                logit(base) + 0.7 * pool["northing"][r["retention_group"]]
            )
            tally[str(period) + ":" + f + ":eligible"] += 1
            tally[str(period) + ":" + f + ":retained"] += int(passed)
            return passed

        sets = panel_sets(pool, context["meta"]["latent"], retention)
        meas = panel_measurement(sets, context["spatial"])
        d = designs(context["cohort"], meas, context["specification"])["0"]
        X = d["X"]
        support = meas["support"]
        raw = (meas["pooled"][:, 0] - meas["pooled"][:, 1]) / context["specification"][
            "normalization"
        ]["common_exposure_sd"]
        pooled = (
            raw[ar["local"]]
            - ar["pooled_global_center"]
            - ar["pooled_regional_center"][ar["cell_inverse"]]
        )
        genX = ar["latent_X"]
        retention_report = dict(tally)
    pn = genX.shape[1] - 4
    B = np.zeros((7, genX.shape[1]))
    B[:, :pn] = ar["nuisance_mode"].reshape(7, pn)
    if scenario["id"] == "pooled":
        generating = np.column_stack([genX[:, :pn], ar["pooled"]])
        gb = np.column_stack([B[:, :pn], ar["direction"] * 0.4])
        truth = None
    else:
        truth = scenario["truth"]
        B[:, -4:] = ar["direction"][:, None] * np.asarray(truth)[None, :]
        generating = genX
        gb = B
    trueprob = unit_probabilities(generating, gb.ravel(), ar["counts"])
    y = outcome(trueprob, rng)
    testy = outcome(trueprob, test_rng)
    methods = [
        ("geography", X[:, :pn], ar["scales"][:pn]),
        (
            "pooled",
            np.column_stack([X[:, :pn], pooled]),
            np.r_[ar["scales"][:pn], ar["scales"][-1]],
        ),
        ("periods", X, ar["scales"]),
    ]
    results = []
    for method, xx, scales in methods:
        start = np.zeros((7, xx.shape[1]))
        start[:, :pn] = ar["nuisance_mode"].reshape(7, pn)
        errors = []
        fit = None
        for initial in [start.ravel(), None]:
            try:
                fit = LabelMixture(xx, y, ar["counts"], scales, 8).optimize(initial)
                break
            except (ValueError, np.linalg.LinAlgError) as e:
                errors.append(str(e))
        details = dict(
            scenario=scenario["id"],
            replicate=rep,
            seed=[cfg["seed"], index, rep],
            test_seed=[cfg["test_seed"], index, rep],
            truth=truth,
            retries=errors,
            retention=retention_report,
            gradient_implementation="matrix product; same analytic score",
        )
        if fit is None:
            results.append((method, "failed", details, {}))
            continue
        details.update(
            {k: v for k, v in fit.items() if k not in ["mode", "covariance"]}
        )
        details.update(
            performance_scores(
                unit_probabilities(xx, fit["mode"], ar["counts"]), trueprob, testy
            )
        )
        arrays = dict(mode=fit["mode"])
        if method == "periods":
            infer, contrasts = temporal_inference(fit, truth, support, ar)
            details.update(infer)
            arrays.update(contrasts)
        results.append((method, "passed", details, arrays))
    return "chronology", scenario["id"], rep, results


def first_summaries(con, cfg):
    for scenario in cfg["scenarios"]:
        paired = {}
        for method in ["full", "simple"]:
            rows = [
                (i, json.loads(r))
                for i, r in con.execute(
                    "select replicate,details from fits where family='first' and scenario=? and method=? order by replicate",
                    (scenario["id"], method),
                )
            ]
            assert len(rows) == cfg["replicates"]
            good = [(i, r) for i, r in rows if r["success"]]
            assert good
            values = np.array([r["mean"] for i, r in good])
            ses = np.array([r["sd"] for i, r in good])
            truth = scenario["beta"]
            err = values - truth
            n = len(good)
            paired[method] = {i: (r["mean"] - truth) ** 2 for i, r in good}
            result = dict(
                scenario=scenario["id"],
                method=method,
                truth=truth,
                successful=n,
                failed=len(rows) - n,
                mean=float(values.mean()),
                bias=float(err.mean()),
                bias_mcse=float(values.std(ddof=1) / np.sqrt(n)),
                rmse=float(np.sqrt(np.mean(err**2))),
                empirical_sd=float(values.std(ddof=1)),
                mean_reported_sd=float(ses.mean()),
                coverage=benchmark_wilson(
                    sum(r["lower"] <= truth <= r["upper"] for i, r in good), n
                ),
                zero_exclusion=benchmark_wilson(
                    sum(r["lower"] > 0 or r["upper"] < 0 for i, r in good), n
                ),
                positive_resolution=benchmark_wilson(
                    sum(r["lower"] > 0 for i, r in good), n
                ),
                max_gradient=max(r["gradient"] for i, r in good),
                minimum_curvature=min(r["curvature"] for i, r in good),
            )
            con.execute(
                "insert or replace into summaries values(?,?,?,?)",
                ("first", scenario["id"], method, encode(result)),
            )
        keys = sorted(set(paired["full"]) & set(paired["simple"]))
        diff = np.array([paired["full"][i] - paired["simple"][i] for i in keys])
        con.execute(
            "insert or replace into summaries values(?,?,?,?)",
            (
                "first",
                scenario["id"],
                "paired",
                encode(
                    dict(
                        paired_n=len(keys),
                        full_minus_simple_mse=float(diff.mean()),
                        mcse=float(diff.std(ddof=1) / np.sqrt(len(keys))),
                    )
                ),
            ),
        )
    con.commit()


def chronology_summaries(db, specification):

    def rate(values, n):
        k = int(np.sum(values))
        r = benchmark_wilson(k, n)
        r["estimate"] = r.pop("rate")
        return r

    def average(values):
        v = np.asarray(values, dtype=float)
        return dict(
            n=len(v),
            mean=float(v.mean()) if len(v) else None,
            mcse=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
        )

    cfg = specification["chronology"]
    totals = collections.Counter()
    for scenario in cfg["scenarios"]:
        name = scenario["id"]
        n = scenario["replicates"]
        by_method = {}
        for method in ["geography", "pooled", "periods"]:
            rows = db.execute(
                "select replicate,status,details,arrays from fits where family='chronology' and scenario=? and method=? order by replicate",
                (name, method),
            ).fetchall()
            assert len(rows) == n, (name, method, len(rows), n)
            rr = [
                (rep, json.loads(details), dict(np.load(io.BytesIO(arrays))))
                for rep, status, details, arrays in rows
                if status == "passed"
            ]
            totals[method] += len(rows)
            out = dict(
                total=n,
                passed=len(rr),
                failed=n - len(rr),
                performance={
                    k: average([d[k] for _, d, _ in rr])
                    for k in [
                        "test_log",
                        "test_brier",
                        "expected_log",
                        "expected_brier",
                    ]
                },
            )
            by_method[method] = {rep: d for rep, d, _ in rr}
            out["calibration_bins"] = []
            for j in range(10):
                bins = [d["calibration_bins"][j] for _, d, _ in rr]
                nn = sum(b["n"] for b in bins)
                differences = [
                    (b["predicted_sum"] - b["true_sum"]) / b["n"]
                    for b in bins
                    if b["n"]
                ]
                out["calibration_bins"].append(
                    dict(
                        lower=j / 10,
                        upper=(j + 1) / 10,
                        category_predictions=nn,
                        mean_predicted=(
                            sum(b["predicted_sum"] for b in bins) / nn if nn else None
                        ),
                        mean_true=sum(b["true_sum"] for b in bins) / nn if nn else None,
                        replicate_error=average(differences) if differences else None,
                    )
                )
            if method == "periods":
                out["status_counts"] = dict(
                    collections.Counter(d["status"] for _, d, _ in rr)
                )
                out["nomination_counts"] = dict(
                    collections.Counter(str(d["nomination"]) for _, d, _ in rr)
                )
                for key in [
                    "any_positive",
                    "any_two_sided",
                    "correct_nomination",
                    "wrong_nomination",
                    "joint_coverage",
                ]:
                    if all(key in d for _, d, _ in rr):
                        out[key] = rate([d[key] for _, d, _ in rr], n)
                out["insufficient_support"] = rate(
                    [
                        d["status"] == "insufficient_archaeological_support"
                        for _, d, _ in rr
                    ],
                    n,
                )
                out["positive_unresolved"] = rate(
                    [d["status"] == "positive_chronology_unresolved" for _, d, _ in rr],
                    n,
                )
                out["no_positive_detected"] = rate(
                    [
                        d["status"] == "no_positive_relationship_detected"
                        for _, d, _ in rr
                    ],
                    n,
                )
                if "truth" in scenario and rr:
                    truth = np.array(scenario["truth"])
                    m = np.array([ar["period_mean"] for _, _, ar in rr])
                    cv = np.array([np.diag(ar["period_covariance"]) for _, _, ar in rr])
                    out["periods"] = []
                    for j, p in enumerate(specification["periods"]):
                        errors = m[:, j] - truth[j]
                        out["periods"].append(
                            dict(
                                period=p,
                                truth=float(truth[j]),
                                estimate=average(m[:, j]),
                                bias=float(errors.mean()),
                                bias_mcse=float(
                                    errors.std(ddof=1) / np.sqrt(len(errors))
                                ),
                                rmse=float(np.sqrt(np.mean(errors**2))),
                                mean_standard_error=float(np.sqrt(cv[:, j]).mean()),
                                coverage=rate(
                                    [d["marginal_coverage"][j] for _, d, _ in rr], n
                                ),
                            )
                        )
                if scenario.get("recording_stress"):
                    keys = sorted(set(k for _, d, _ in rr for k in d["retention"]))
                    out["recording_counts"] = {
                        k: average([d["retention"].get(k, 0) for _, d, _ in rr])
                        for k in keys
                    }
            db.execute(
                "insert or replace into summaries values(?,?,?,?)",
                ("chronology", name, method, encode(out)),
            )
        pairs = {}
        for comparator in ["geography", "pooled"]:
            shared = sorted(set(by_method["periods"]) & set(by_method[comparator]))
            pairs[comparator] = {
                key: average(
                    [
                        by_method["periods"][r][key] - by_method[comparator][r][key]
                        for r in shared
                    ]
                )
                for key in ["test_log", "test_brier", "expected_log", "expected_brier"]
            }
        db.execute(
            "insert or replace into summaries values(?,?,?,?)",
            ("chronology", name, "paired", encode(pairs)),
        )
        db.commit()
    if sum(totals.values()) != cfg["fits"]:
        raise ValueError("Chronology fit count differs from its finite contract")
    db.commit()


def initialize(paths):
    """Load immutable inputs once in each process; use one BLAS thread."""
    global _context
    threadpool_limits(limits=1)
    paths = {key: Path(value) for key, value in paths.items()}
    specification = read(paths["specification"])
    specification = specification.get("legacy", specification)
    context = dict(specification=specification)
    for name, key in [("first_design.npz", "first"), ("chronology_design.npz", "arr")]:
        path = paths["inputs"] / name
        expected = specification["fixtures"][name]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Benchmark fixture checksum differs: " + name)
        context[key] = arrays(path)
    if "chronology" in paths:
        from analysis.chronology import load_spatial

        context["pool"] = json.loads(
            gzip.decompress(
                (paths["chronology"] / "material_pool.json.gz").read_bytes()
            )
        )
        positions = read(paths["data"] / "paternal.json")["spatial_positions"]
        context["cohort"] = arrays(paths["measurements"] / "common_cohort.npz")
        context["spatial"] = load_spatial(
            context["pool"], positions, paths["chronology"]
        )
        context["meta"] = read(paths["chronology"] / "preparation.json")
    _context = context


def task(job):
    family, payload = job
    if family == "first":
        return first_one(payload, _context)
    return chronology_one(payload, _context)


def open_results(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS fits(
            family TEXT, scenario TEXT, replicate INTEGER, method TEXT,
            status TEXT, details TEXT, arrays BLOB,
            PRIMARY KEY(family,scenario,replicate,method));
        CREATE TABLE IF NOT EXISTS summaries(
            family TEXT, scenario TEXT, method TEXT, details TEXT,
            PRIMARY KEY(family,scenario,method));
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
    """
    )
    return connection


def input_manifest(paths):
    files = [
        Path(paths["specification"]),
        Path(__file__),
        Path(__file__).with_name("model.py"),
    ]
    files += [
        Path(paths["inputs"]) / name
        for name in ["first_design.npz", "chronology_design.npz"]
    ]
    if "chronology" in paths:
        files += [
            Path(paths["chronology"]) / name
            for name in [
                "material_pool.json.gz",
                "material_proximity.npy",
                "spatial.json",
                "preparation.json",
            ]
        ]
        files += [
            Path(paths["measurements"]) / "common_cohort.npz",
            Path(paths["data"]) / "paternal.json",
            Path(__file__).with_name("chronology.py"),
        ]
    result = {}
    for path in files:
        with path.open("rb") as source:
            result[path.name] = dict(
                sha256=hashlib.file_digest(source, "sha256").hexdigest(),
                bytes=path.stat().st_size,
            )
    return result


def run(paths, output, family="all", workers=2, limit=None):
    """Resume only compatible finite batches; record every attempted fit."""
    if not 1 <= workers <= 8:
        raise ValueError("Use between one and eight workers")
    specification = read(paths["specification"])
    specification = specification.get("legacy", specification)
    jobs = []
    if family in ["first", "all"]:
        jobs += [
            ("first", (scenario, replicate))
            for scenario in specification["simulation"]["scenarios"]
            for replicate in range(specification["simulation"]["replicates"])
        ]
    if family in ["chronology", "all"]:
        jobs += [
            ("chronology", (index, scenario, replicate))
            for index, scenario in enumerate(specification["chronology"]["scenarios"])
            for replicate in range(scenario["replicates"])
        ]
    manifest = input_manifest(paths)
    db = open_results(output / "benchmarks.sqlite")
    previous = db.execute("select value from metadata where key='inputs'").fetchone()
    if previous and json.loads(previous[0]) != manifest:
        raise ValueError(
            "Existing results use different inputs or code; use a new output directory"
        )
    db.execute("insert or ignore into metadata values('inputs',?)", (encode(manifest),))
    db.commit()
    statuses = collections.defaultdict(dict)
    for fam, scenario, replicate, method, status in db.execute(
        "select family,scenario,replicate,method,status from fits"
    ):
        statuses[fam, scenario, replicate][method] = status
    pending = []
    for job in jobs:
        fam, payload = job
        scenario, replicate = payload[-2:]
        previous = statuses[fam, scenario["id"], replicate]
        expected = (
            {"full", "simple"} if fam == "first" else {"geography", "pooled", "periods"}
        )
        if set(previous) - expected:
            raise ValueError("Unexpected method in the retained result table")
        if previous and any(status != "passed" for status in previous.values()):
            raise ValueError(
                "A failed fit is retained; review its diagnostics before any rerun"
            )
        if set(previous) != expected:
            pending.append(job)
    scheduled = pending if limit is None else pending[:limit]
    if workers == 1:
        initialize(paths)
        results = map(task, scheduled)
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=initialize, initargs=(paths,)
        )
        results = executor.map(task, scheduled, chunksize=1)
    completed = 0
    try:
        for fam, scenario, replicate, fits in results:
            for method, status, details, values in fits:
                db.execute(
                    "insert or ignore into fits values(?,?,?,?,?,?,?)",
                    (
                        fam,
                        scenario,
                        replicate,
                        method,
                        status,
                        encode(details),
                        blob(values),
                    ),
                )
            db.commit()
            completed += 1
            if completed % 50 == 0 or completed == len(scheduled):
                print("Completed datasets", completed, "of", len(scheduled), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    complete = len(scheduled) == len(pending)
    if complete:
        if family in ["first", "all"]:
            first_summaries(db, specification["simulation"])
        if family in ["chronology", "all"]:
            chronology_summaries(db, specification)
    summary = dict(
        family=family,
        complete=complete,
        declared_datasets=len(jobs),
        newly_computed_datasets=completed,
        remaining_datasets=len(pending) - completed,
        statuses=[
            dict(family=f, status=s, fits=n)
            for f, s, n in db.execute(
                "select family,status,count(*) from fits group by family,status"
            )
        ],
    )
    db.execute(
        "insert or replace into metadata values(?,?)",
        ("completion:" + family, encode(summary)),
    )
    db.commit()
    db.close()
    print(encode(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--specification", type=Path, default=Path("analysis/specification.json")
    )
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument("--chronology", type=Path, default=Path("build/chronology"))
    parser.add_argument("--output", type=Path, default=Path("build/benchmarks"))
    parser.add_argument(
        "--family", choices=["first", "chronology", "all"], default="all"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        help="Compute at most this many pending datasets; a partial run is never marked complete",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    paths = {
        key: str(getattr(args, key).resolve())
        for key in ["inputs", "data", "specification", "measurements"]
    }
    if args.family != "first":
        paths["chronology"] = str(args.chronology.resolve())
    run(paths, args.output, args.family, args.workers, args.limit)


if __name__ == "__main__":
    main()

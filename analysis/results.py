"""Read and verify retained numerical results without refitting models.

Coefficient draws, derived contrasts and conditional sensitivity fits remain
separate. This check verifies numerical representation; a fresh inference run
is a separate reproduction step.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import io
import json
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.draws import read_draws

GROUPS = ("current", "appraisal", "branches")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    Path(path).write_bytes(
        gzip.compress(raw, mtime=0) if Path(path).suffix == ".gz" else raw
    )


def posterior_summaries(directory):
    result = {}
    for p in sorted(Path(directory).glob("*_contrasts.csv")):
        job = p.name.removesuffix("_contrasts.csv")
        receipt = read(p.with_name(job + ".json"))
        if (
            receipt["contrasts_sha256"] != sha(p)
            or not receipt["diagnostics"]["passed"]
        ):
            raise ValueError((job, "Undiagnosed or altered posterior summary"))
        with p.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        parsed = {}
        for row in rows:
            target = row["target"]
            if target in parsed:
                raise ValueError((job, "Duplicate contrast", target))
            parsed[target] = {
                k: (
                    v
                    if k in ["scope", "comparator"]
                    else (
                        v.upper() == "TRUE"
                        if k == "all_draws_same_sign"
                        else None if v in ["NA", ""] else float(v)
                    )
                )
                for k, v in row.items()
                if k != "target"
            }
        result[job] = parsed
    return result


def conditional_summary(path):
    grouped = defaultdict(list)
    fits = []
    names = {}
    with connect(path) as db:
        for job, family, key, status, raw, blob, digest in db.execute(
            "SELECT * FROM fits ORDER BY job,family,replicate"
        ):
            if status != "passed":
                raise ValueError((job, family, key, status))
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError((job, family, key, "Changed numerical blob"))
            detail = json.loads(raw)
            n = detail["contrast_names"]
            if job in names and names[job] != n:
                raise ValueError((job, "Inconsistent contrast order"))
            names[job] = n
            with np.load(io.BytesIO(blob), allow_pickle=False) as a:
                means = a["contrast_mean"]
                sd = a["contrast_sd"]
                if (
                    means.shape != sd.shape
                    or means.shape != (len(n),)
                    or not np.isfinite(means).all()
                    or not np.isfinite(sd).all()
                    or (sd < 0).any()
                ):
                    raise ValueError((job, family, key, "Invalid contrasts"))
                means = means.tolist()
                sd = sd.tolist()
            grouped[job, family].append(means)
            fits.append(dict(job=job, family=family, replicate=key, means=means, sd=sd))
    aggregates = []
    for (job, family), values in sorted(grouped.items()):
        a = np.asarray(values)
        q = np.quantile(a, [0, 0.025, 0.5, 0.975, 1], axis=0)
        aggregates.append(
            dict(
                job=job,
                family=family,
                replicates=len(a),
                **{
                    k: q[i].tolist()
                    for i, k in enumerate(
                        ["minimum", "q025", "median", "q975", "maximum"]
                    )
                },
                fraction_positive=np.mean(a > 0, axis=0).tolist(),
            )
        )
    return dict(
        contrast_names=names,
        fits=fits,
        aggregates=aggregates,
        uncertainty="Conditional mode and inverse-Hessian standard error. Quantiles across sensitivities are not posterior intervals.",
    )


def summary(results):
    results = Path(results)
    return dict(
        schema="sclara-results-summary-1",
        posterior={g: posterior_summaries(results / g) for g in GROUPS},
        conditional=conditional_summary(results / "current/conditional.sqlite"),
    )


def verify_posterior(directory, design_path):
    jobs = read(design_path)["jobs"]
    summaries = posterior_summaries(directory)
    if set(jobs) != set(summaries):
        raise ValueError(("Incomplete posterior family", set(jobs) ^ set(summaries)))
    checked = []
    for job, d in jobs.items():
        p = Path(directory) / (job + "_draws.npz")
        if not p.exists():
            p = Path(directory) / (job + "_draws.csv.gz")
        receipt = read(p.with_name(job + ".json"))
        if sha(p) != receipt["draws_sha256"]:
            raise ValueError((job, "Changed posterior draws"))
        header, a = read_draws(p)
        storage = receipt.get("draw_storage")
        if storage and hashlib.sha256(a.astype("<f8").tobytes()).hexdigest() != storage["float64_values_sha256"]:
            raise ValueError((job, "Changed coefficient values"))
        if not np.isfinite(a).all() or a.shape[0] != receipt["draws"]:
            raise ValueError((job, "Invalid draws"))
        per_chain = receipt["attempts"][-1]["retained_per_chain"]
        expected = np.column_stack(
            [
                np.repeat(np.arange(1, 5), per_chain),
                np.tile(np.arange(1, per_chain + 1), 4),
            ]
        )
        if not np.array_equal(a[:, :2], expected):
            raise ValueError((job, "Chain/iteration ordering changed"))
        width = len(d["x"][0])
        classes = len(d["categories"]) - 1
        beta_names = [
            f"beta[{i},{j}]" for j in range(1, classes + 1) for i in range(1, width + 1)
        ]
        if header[2 : 2 + width * classes] != beta_names:
            raise ValueError((job, "Coefficient ordering changed"))
        stored_contrasts = header[2 + width * classes :]
        if stored_contrasts and stored_contrasts != list(d["contrasts"]):
            raise ValueError((job, "Contrast ordering changed"))
        vectors = np.asarray(list(d["contrasts"].values()))
        derived = a[:, 2 : 2 + width * classes] @ vectors.T
        error = None
        if stored_contrasts:
            error = float(np.max(np.abs(derived - a[:, 2 + width * classes :])))
            if error > 1e-10:
                raise ValueError((job, "Contrast reconstruction", error))
        q = np.quantile(derived, [0.025, 0.5, 0.975], axis=0)
        means = derived.mean(0)
        sds = derived.std(0, ddof=1)
        positive = (derived > 0).mean(0)
        maximum = 0.0
        for i, name in enumerate(d["contrasts"]):
            row = summaries[job][name]
            for k, value in dict(
                mean=means[i],
                sd=sds[i],
                lower=q[0, i],
                median=q[1, i],
                upper=q[2, i],
                probability_positive=positive[i],
            ).items():
                difference = abs(float(value) - row[k])
                maximum = max(maximum, difference)
                if difference > 1e-10:
                    raise ValueError((job, name, k, difference))
        dg = receipt["diagnostics"]
        if not (
            dg["max_rhat"] <= 1.01
            and dg["contrast_max_rhat"] <= 1.01
            and min(
                dg[k]
                for k in [
                    "min_bulk_ess",
                    "min_tail_ess",
                    "contrast_min_bulk_ess",
                    "contrast_min_tail_ess",
                ]
            )
            >= 400
            and dg["divergences"] == dg["treedepth_hits"] == 0
        ):
            raise ValueError((job, "Failed retained diagnostics"))
        checked.append(
            dict(
                job=job,
                draws=len(a),
                coefficients=width * classes,
                contrasts=len(vectors),
                stored_contrast_columns=bool(stored_contrasts),
                maximum_contrast_error=error,
                maximum_summary_error=maximum,
                draws_sha256=sha(p),
                retained_diagnostics_pass=True,
            )
        )
    return checked


def verify_conditionals(path, design_path, expected_counts=None):
    designs = read(design_path)["jobs"]
    counts = defaultdict(int)
    maximum = 0.0
    with connect(path) as db:
        for job, family, key, status, raw, blob, digest in db.execute(
            "SELECT * FROM fits"
        ):
            if status != "passed" or hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError((job, family, key, "Invalid conditional fit"))
            detail = json.loads(raw)
            d = designs[job]
            if detail["contrast_names"] != list(d["contrasts"]):
                raise ValueError((job, family, key, "Incorrect contrast order"))
            with np.load(io.BytesIO(blob), allow_pickle=False) as a:
                if not all(np.isfinite(a[k]).all() for k in a.files):
                    raise ValueError((job, family, key, "Invalid arrays"))
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(list(d["contrasts"].values())) @ a["mode"]
                            - a["contrast_mean"]
                        )
                    )
                )
                maximum = max(maximum, error)
                if error > 1e-10:
                    raise ValueError((job, family, key, error))
            if detail["gradient_max"] >= 1e-5 or detail["minimum_curvature"] <= 0:
                raise ValueError((job, family, key, "Failed optimization checks"))
            counts[job, family] += 1
    if expected_counts is None:
        expected = {}
        for job in ["linear_0", "linear_100000"]:
            expected.update(
                {
                    (job, f): n
                    for f, n in dict(
                        allocation=64,
                        crossed_dependence=399,
                        phase_omission=1,
                        omit_block=5,
                        omit_cell=3,
                        region=6,
                        omit_site=5,
                        omit_pair=3,
                        paternal_location=2,
                    ).items()
                }
            )
        for job in [
            "viking_primary",
            "viking_companion",
            "medieval_primary",
            "medieval_companion",
        ]:
            expected.update(
                {
                    (job, f): n
                    for f, n in dict(
                        crossed_dependence=399,
                        omit_block=8,
                        omit_cell=4,
                        paternal_location=2,
                    ).items()
                }
            )
    else:
        expected = expected_counts
    if dict(counts) != expected:
        raise ValueError("Incomplete conditional family")
    return dict(
        fits=sum(counts.values()),
        maximum_contrast_error=maximum,
        families=[
            dict(job=j, family=f, fits=n) for (j, f), n in sorted(counts.items())
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("build/results_summary.json.gz"))
    p.add_argument("--verify", action="store_true")
    p.add_argument(
        "--current-designs", type=Path, default=Path("build/inference/designs.json")
    )
    p.add_argument(
        "--appraisal-designs", type=Path, default=Path("build/appraisal/designs.json")
    )
    p.add_argument(
        "--branch-designs", type=Path, default=Path("build/branches/designs.json")
    )
    a = p.parse_args()
    with threadpool_limits(limits=1):
        result = summary(a.results)
        if a.verify:
            checks = {
                g: verify_posterior(a.results / g, d)
                for g, d in zip(
                    GROUPS, [a.current_designs, a.appraisal_designs, a.branch_designs]
                )
            }
            checks["conditional"] = verify_conditionals(
                a.results / "current/conditional.sqlite", a.current_designs
            )
            result["verification"] = dict(
                complete=True,
                numerical_tolerance=1e-10,
                checks=checks,
                scope="Draw/summary representation and retained diagnostics; fresh inference is a separate check.",
            )
        save(a.output, result)
    print(
        "Saved",
        a.output,
        "with",
        sum(len(v) for v in result["posterior"].values()),
        "posterior jobs and",
        len(result["conditional"]["fits"]),
        "conditional fits",
    )


if __name__ == "__main__":
    main()

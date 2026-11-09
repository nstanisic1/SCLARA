"""Collect and verify the finite representation, support and estimator extensions."""

from pathlib import Path
import argparse
import hashlib
import io
import json
import shutil
import sqlite3
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.draws import compact_directory
from analysis.results import (
    read,
    save,
    sha,
    connect,
    verify_conditionals,
    verify_posterior,
    posterior_summaries,
    conditional_summary,
)


def conditional_expectations(build):
    branch = read(build / "representation/designs.json")["jobs"]
    support = read(build / "representation/common_support/designs.json")["jobs"]
    if len(branch) != 32 or len(support) != 4:
        raise ValueError("Changed extension family sizes")
    a = {(name, "conditional"): 1 for name in branch}
    a.update(
        {
            (name, "crossed_dependence"): 400
            for name in branch
            if "_full_cut_" in name
            and name.endswith(("_linear_0", "_viking_primary", "_medieval_primary"))
        }
    )
    b = {(name, "conditional"): 1 for name in support}
    b.update(
        {(name, "crossed_dependence"): 399 for name in support if "_linear_" in name}
    )
    return a, b


def verify_evaluation(path, fixture_path, metadata_path):
    metadata = read(metadata_path)
    contract = metadata["contract"]
    if sha(fixture_path) != metadata["fixture_sha256"]:
        raise ValueError("Changed evaluation fixture")
    with np.load(fixture_path, allow_pickle=False) as z:
        vectors = z["vectors"]
        width = z["nuisance"].shape
        truths = {
            c["name"]: vectors
            @ (z["nuisance"] + c["effect"] * z["effect_direction"])[1:].ravel()
            for c in contract["conditions"]
        }
    expected = {
        (c["name"], m) for c in contract["conditions"] for m in contract["methods"]
    }
    counts = {}
    indices = {}
    failed = []
    records = {key: {} for key in expected}
    maximum = 0.0
    with connect(path) as db:
        for condition, method, replicate, status, raw, blob, digest in db.execute(
            "select * from fits"
        ):
            key = (condition, method)
            counts[key] = counts.get(key, 0) + 1
            indices.setdefault(key, set()).add(int(replicate))
            detail = json.loads(raw)
            if status != "passed":
                failed.append(
                    dict(
                        condition=condition,
                        method=method,
                        replicate=replicate,
                        detail=detail,
                    )
                )
                continue
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError("Changed evaluation coefficients")
            with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                mode = z["mode"]
            if (
                mode.shape != ((width[0] - 1) * width[1],)
                or not np.isfinite(mode).all()
            ):
                raise ValueError("Invalid evaluation coefficient array")
            error = float(
                np.max(np.abs(vectors @ mode - np.asarray(detail["estimates"])))
            )
            maximum = max(maximum, error)
            if (
                error > 1e-10
                or detail["gradient_max"] >= 1e-5
                or detail["minimum_curvature"] <= 0
            ):
                raise ValueError("Evaluation reconstruction or numerical check failed")
            estimate = np.asarray(detail["estimates"])
            sd = np.asarray(detail["standard_errors"])
            truth = truths[condition]
            lower, upper = (
                estimate - 1.959963984540054 * sd,
                estimate + 1.959963984540054 * sd,
            )
            if not np.isfinite(sd).all() or np.any(sd <= 0):
                raise ValueError("Invalid evaluation standard errors")
            for key, value in [("truth", truth), ("lower", lower), ("upper", upper)]:
                if not np.allclose(detail[key], value, rtol=0, atol=1e-10):
                    raise ValueError(
                        (
                            condition,
                            method,
                            replicate,
                            "Changed interval or generating truth",
                            key,
                        )
                    )
            if not np.array_equal(
                detail["covered"], (lower <= truth) & (truth <= upper)
            ) or not np.array_equal(
                detail["interval_excludes_zero"], (lower > 0) | (upper < 0)
            ):
                raise ValueError("Changed coverage or interval decision")
            records[condition, method][int(replicate)] = detail
        if set(counts) != expected or any(
            n != contract["replicates_per_condition"] for n in counts.values()
        ):
            raise ValueError("Incomplete finite evaluation")
        if any(
            v != set(range(contract["replicates_per_condition"]))
            for v in indices.values()
        ):
            raise ValueError("Changed evaluation replicate identities")
        summaries = [
            dict(condition=c, method=m, **json.loads(raw))
            for c, m, raw in db.execute("select * from summaries")
        ]
        if len(summaries) != len(contract["conditions"]) * (
            2 * len(contract["methods"]) - 1
        ):
            raise ValueError("Incomplete evaluation summaries")
    summary_map = {(r["condition"], r["method"]): r for r in summaries}
    maximum_summary_error = 0.0

    def check_value(actual, expected_value):
        nonlocal maximum_summary_error
        error = float(
            np.max(
                np.abs(
                    np.asarray(actual, dtype=float)
                    - np.asarray(expected_value, dtype=float)
                )
            )
        )
        maximum_summary_error = max(maximum_summary_error, error)
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(("Evaluation summary reconstruction", error))

    def check_mean_se(actual, values):
        values = np.asarray(values, dtype=float)
        check_value(actual["mean"], values.mean(axis=0))
        check_value(actual["mcse"], values.std(axis=0, ddof=1) / np.sqrt(len(values)))

    for condition in truths:
        for method in contract["methods"]:
            rows = list(records[condition, method].values())
            summary = summary_map[condition, method]
            if summary["successful"] != len(rows) or summary["failed"] != contract[
                "replicates_per_condition"
            ] - len(rows):
                raise ValueError("Changed successful/failed evaluation counts")
            if not rows:
                continue
            error = np.asarray([r["estimates"] for r in rows]) - truths[condition]
            check_mean_se(summary["bias"], error)
            check_value(summary["rmse"], np.sqrt(np.mean(error**2, axis=0)))
            check_mean_se(summary["coverage"], [r["covered"] for r in rows])
            check_mean_se(
                summary["interval_exclusion"],
                [r["interval_excludes_zero"] for r in rows],
            )
            check_mean_se(
                summary["interval_width"],
                [np.asarray(r["upper"]) - r["lower"] for r in rows],
            )
            for key in ["test_log", "test_brier", "expected_log", "expected_brier"]:
                check_mean_se(summary[key], [r[key] for r in rows])
        for comparator in contract["methods"][1:]:
            primary = records[condition, contract["methods"][0]]
            other = records[condition, comparator]
            ids = sorted(set(primary) & set(other))
            if not ids:
                continue
            summary = summary_map[condition, "paired_vs_" + comparator]
            if summary["replicates"] != len(ids):
                raise ValueError("Changed paired evaluation count")
            for key in ["test_log", "test_brier", "expected_log", "expected_brier"]:
                sign = -1 if "brier" in key else 1
                check_mean_se(
                    summary[key],
                    [sign * (primary[i][key] - other[i][key]) for i in ids],
                )
    return dict(
        method_records=sum(counts.values()),
        failed_records=failed,
        maximum_contrast_reconstruction_error=maximum,
        maximum_performance_summary_error=maximum_summary_error,
        summaries=summaries,
        scope="All coefficient checksums, focal contrasts, generating truths, interval decisions and complete performance summaries; fresh reruns separately assess computational reproduction and prediction from regenerated outcomes.",
    )


def verify(build, results):
    a, b = conditional_expectations(build)
    post = verify_posterior(
        results / "representation", build / "representation/posterior/designs.json"
    )
    if len(post) != 8:
        raise ValueError("Incomplete current-profile branch posteriors")
    return dict(
        posterior=post,
        representation=verify_conditionals(
            results / "representation/conditional.sqlite",
            build / "representation/designs.json",
            a,
        ),
        common_support=verify_conditionals(
            results / "common_support/conditional.sqlite",
            build / "representation/common_support/designs.json",
            b,
        ),
        evaluation=verify_evaluation(
            results / "evaluation/evaluation.sqlite",
            results / "evaluation/fixture.npz",
            results / "evaluation/fixture.json",
        ),
    )


def verify_source_batches(build):
    a, b = conditional_expectations(build)
    rep = build / "representation"
    ev = build / "evaluation"
    # Verify complete source batches before writing a collection.
    checks = dict(
        posterior=verify_posterior(
            rep / "posterior/results", rep / "posterior/designs.json"
        ),
        representation=verify_conditionals(
            rep / "conditional.sqlite", rep / "designs.json", a
        ),
        common_support=verify_conditionals(
            rep / "common_support/conditional.sqlite",
            rep / "common_support/designs.json",
            b,
        ),
        evaluation=verify_evaluation(
            ev / "evaluation.sqlite", ev / "fixture.npz", ev / "fixture.json"
        ),
    )
    if len(checks["posterior"]) != 8:
        raise ValueError("Incomplete posterior extension")
    frozen = read(rep / "posterior/design_freeze.json")
    for name, key in [
        ("designs.json", "designs_sha256"),
        ("mixture.stan", "stan_sha256"),
        ("specification.json", "specification_sha256"),
    ]:
        if sha(rep / "posterior" / name) != frozen[key]:
            raise ValueError(("Changed extension freeze", name))
    for job in read(rep / "posterior/designs.json")["jobs"]:
        receipt = read(rep / "posterior/results" / (job + ".json"))
        if (
            receipt["design_sha256"] != frozen["designs_sha256"]
            or receipt["model_sha256"] != frozen["stan_sha256"]
        ):
            raise ValueError((job, "Extension receipt does not match frozen design"))
    return checks


def collect(build, output, checks=None):
    checks = verify_source_batches(build) if checks is None else checks
    rep, ev = build / "representation", build / "evaluation"
    if any(
        (output / n).exists()
        for n in ["representation", "common_support", "evaluation"]
    ):
        raise FileExistsError("Use a collection without existing extensions")
    for name in ["representation", "common_support", "evaluation"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    for p in (rep / "posterior/results").iterdir():
        if p.is_file() and (
            p.name.endswith(
                (
                    ".json",
                    "_draws.csv.gz",
                    "_contrasts.csv",
                    "_parameter_diagnostics.csv",
                )
            )
        ):
            shutil.copyfile(p, output / "representation" / p.name)
    for src, dst in [
        (rep / "conditional.sqlite", output / "representation/conditional.sqlite"),
        (rep / "selection.json", output / "representation/selection.json"),
        (
            rep / "common_support/conditional.sqlite",
            output / "common_support/conditional.sqlite",
        ),
        (rep / "common_support/support.json", output / "common_support/support.json"),
        (ev / "evaluation.sqlite", output / "evaluation/evaluation.sqlite"),
        (ev / "fixture.npz", output / "evaluation/fixture.npz"),
    ]:
        shutil.copyfile(src, dst)
    meta = read(ev / "fixture.json")
    meta["run_manifest_sha256"] = sha(ev / "fixture.json")
    # Collection provenance names scientific input roles, not a staging directory.
    meta["input_hashes"] = {Path(k).name: v for k, v in meta["input_hashes"].items()}
    save(output / "evaluation/fixture.json", meta)
    conversion = compact_directory(output / "representation")
    checks["draw_storage_conversion"] = conversion
    files = [
        dict(
            path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha(p)
        )
        for name in ["representation", "common_support", "evaluation"]
        for p in sorted((output / name).rglob("*"))
        if p.is_file()
    ]
    save(
        output / "extension_provenance.json",
        dict(schema="sclara-extension-results-1", checks=checks, files=files),
    )
    return dict(
        files=len(files),
        bytes=sum(r["bytes"] for r in files),
        posterior_jobs=8,
        conditional_fits=3234,
        evaluation_method_records=4800,
    )


def summary(results):
    with connect(results / "evaluation/evaluation.sqlite") as db:
        evaluation = [
            dict(condition=c, method=m, **json.loads(raw))
            for c, m, raw in db.execute("select * from summaries")
        ]
    return dict(
        posterior=posterior_summaries(results / "representation"),
        representation=conditional_summary(
            results / "representation/conditional.sqlite"
        ),
        common_support=conditional_summary(
            results / "common_support/conditional.sqlite"
        ),
        selection=read(results / "representation/selection.json"),
        support=read(results / "common_support/support.json"),
        evaluation=evaluation,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["collect", "verify", "summary"])
    p.add_argument("--build", type=Path, default=Path("build"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("build/extension_results"))
    a = p.parse_args()
    with threadpool_limits(limits=1):
        if a.action == "collect":
            print(json.dumps(collect(a.build, a.output), indent=2))
        else:
            value = (
                verify(a.build, a.results)
                if a.action == "verify"
                else summary(a.results)
            )
            save(a.output / "extension_summary.json.gz", value)
            print("Saved", a.output / "extension_summary.json.gz")


if __name__ == "__main__":
    main()

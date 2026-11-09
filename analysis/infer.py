"""Build and fit the fixed Swedish lineage comparisons.

The export command recomputes conditional modes and density checks, then writes
portable inputs for sample.R. Conditional sensitivities use the same likelihood,
priors and frozen transforms; their ranges are not posterior intervals.
"""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import sqlite3
import time
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.model import LabelMixture, make_designs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def save(path, value):
    Path(path).write_text(encode(value) + "\n", encoding="utf-8")


def primary_designs(measurements, specification):
    """Reconstruct all four common-period and four later-profile designs."""
    result = {}
    common = arrays(measurements / "common_cohort.npz")
    data = arrays(measurements / "common.npz")
    config = specification["common"]["design"]
    for quadratic in [False, True]:
        for origin, d in make_designs(
            common, data, config, quadratic=quadratic
        ).items():
            name = ("quadratic" if quadratic else "linear") + "_" + origin
            result[name] = dict(
                scope="common",
                job_id=name,
                seed=specification["common"]["inference"]["posterior"]["seed_base"]
                + len(result),
                x=d["X"],
                y=d["y"],
                counts=d["counts"],
                starts=common["starts"],
                scales=d["scales"],
                categories=config["categories"],
                contrasts=d["vectors"],
            )
    later = arrays(measurements / "later_cohort.npz")
    data = arrays(measurements / "later_0km.npz")
    config = specification["later"]["design"]
    for i, name in enumerate(config["jobs"]):
        x = np.column_stack([data["controls"][i], data["scores"][:, i]])[
            later["local_index"]
        ]
        scales = np.array([2] + [0.5] * (x.shape[1] - 2) + [1])
        contrasts = {}
        for key, weights in config["lineage_contrasts"].items():
            v = np.zeros((len(config["categories"]) - 1, x.shape[1]))
            v[:, -1] = weights[1:]
            contrasts[key] = v.ravel()
        result[name] = dict(
            scope="later",
            job_id=name,
            seed=specification["later"]["inference"]["posterior"]["seed_base"] + i,
            x=x,
            y=later["category"],
            counts=later["counts"],
            starts=np.r_[0, np.cumsum(later["counts"])[:-1]],
            scales=scales,
            categories=config["categories"],
            contrasts=contrasts,
        )
    return result


def fit_design(d, initial=None, weights=None):
    model = LabelMixture(
        d["x"], d["y"], d["counts"], d["scales"], len(d["categories"]), weights
    )
    fit = model.optimize(initial)
    vectors = np.array(list(d["contrasts"].values()))
    variance = np.einsum("ij,ij->i", vectors @ fit["covariance"], vectors)
    if variance.min() < -1e-10:
        raise ValueError("Negative contrast variance")
    return (
        model,
        fit,
        dict(
            mode=fit["mode"],
            contrast_mean=vectors @ fit["mode"],
            contrast_sd=np.sqrt(np.maximum(variance, 0)),
        ),
    )


def export(measurements, specification_path, output):
    if (output / "design_freeze.json").exists():
        raise FileExistsError(
            "Frozen export already exists; verify it or use a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    spec = read(specification_path)
    designs = primary_designs(measurements, spec)
    jobs, modes, checks = {}, {}, []
    for name, d in designs.items():
        started = time.monotonic()
        model, fit, result = fit_design(d)
        rng = np.random.default_rng(d["seed"])
        density = []
        for v in [
            fit["mode"],
            fit["mode"] + rng.normal(0, 0.02, len(fit["mode"])),
            fit["mode"] + rng.normal(0, 0.1, len(fit["mode"])),
        ]:
            value, gradient = model.evaluate(v)
            density.append(dict(v=v, log_density=-value, gradient=-gradient))
        direction = rng.normal(size=len(fit["mode"]))
        direction /= np.linalg.norm(direction)
        h = 1e-5
        _, gradient, hessian = model.evaluate(fit["mode"], True)
        plus, gp = model.evaluate(fit["mode"] + h * direction)
        minus, gm = model.evaluate(fit["mode"] - h * direction)
        errors = dict(
            directional_gradient=abs((plus - minus) / (2 * h) - gradient @ direction),
            directional_hessian=float(
                abs((gp - gm) / (2 * h) - hessian @ direction).max()
            ),
            inverse_hessian=float(
                abs(hessian @ fit["covariance"] - np.eye(len(gradient))).max()
            ),
        )
        if (
            errors["directional_gradient"] >= 1e-5
            or errors["directional_hessian"] >= 1e-5
            or errors["inverse_hessian"] >= 1e-8
        ):
            raise ValueError((name, errors))
        vectors = np.array(list(d["contrasts"].values()))
        for key, value in result.items():
            modes[name + "__" + key] = value
        modes[name + "__covariance"] = fit["covariance"]
        modes[name + "__contrast_covariance"] = vectors @ fit["covariance"] @ vectors.T
        scope = (
            "conditional exploratory joint-period comparison"
            if d["scope"] == "common"
            else "conditional exploratory source-corrected later-period profile"
        )
        jobs[name] = dict(d, scope=scope, density_checks=density)
        checks.append(
            dict(
                job=name,
                design_sha256=hashlib.sha256(d["x"].tobytes()).hexdigest(),
                gradient_max=fit["gradient"],
                minimum_curvature=fit["curvature"],
                errors=errors,
                seconds=time.monotonic() - started,
            )
        )
        print("Checked", name, "gradient", fit["gradient"], flush=True)
    np.savez_compressed(output / "primary_modes.npz", **modes)
    save(output / "primary_checks.json", checks)
    save(output / "designs.json", dict(jobs=jobs))
    shutil.copyfile(specification_path, output / "specification.json")
    shutil.copyfile(Path(__file__).with_name("mixture.stan"), output / "mixture.stan")
    save(
        output / "design_freeze.json",
        dict(
            specification_sha256=sha(output / "specification.json"),
            designs_sha256=sha(output / "designs.json"),
            stan_sha256=sha(output / "mixture.stan"),
            sampler_sha256=sha(Path(__file__).with_name("sample.R")),
        ),
    )


def subset(d, keep):
    return dict(
        d,
        x=d["x"][np.repeat(keep, d["counts"])],
        y=d["y"][keep],
        counts=d["counts"][keep],
    )


class Sensitivities:
    """Construct the prespecified conditional designs and whole-unit weights."""

    def __init__(self, measurements, specification, inference):
        self.path = Path(measurements)
        self.spec = specification
        self.designs = primary_designs(self.path, specification)
        self.common = arrays(self.path / "common_cohort.npz")
        self.measurement = arrays(self.path / "common.npz")
        self.phase = arrays(self.path / "common_phase_omission.npz")
        self.later = arrays(self.path / "later_cohort.npz")
        self.modes = arrays(Path(inference) / "primary_modes.npz")
        self.common_designs = make_designs(
            self.common, self.measurement, specification["common"]["design"]
        )
        dep = specification["common"]["inference"]["sensitivity"]["crossed_dependence"]
        blocks, self.block_index = np.unique(self.common["blocks"], return_inverse=True)
        self.block_weights = np.random.default_rng(dep["block_seed"]).exponential(
            size=(dep["replicates_per_origin"], len(blocks))
        )
        self.cell_index, self.cell_weights = {}, {}
        for origin, d in self.common_designs.items():
            cells, self.cell_index[origin] = np.unique(
                d["centroid_labels"], return_inverse=True
            )
            self.cell_weights[origin] = np.random.default_rng(
                dep["cell_seeds"][origin]
            ).exponential(size=(dep["replicates_per_origin"], len(cells)))
        assignments = read(self.path / "common_regions.json")
        per_position = np.array([r["region"] or "unassigned" for r in assignments])
        local_regions = per_position[self.common["local"]]
        self.regions = np.array(
            [
                next(iter(v)) if len(v) == 1 else "mixed"
                for v in [
                    set(local_regions[s : s + n])
                    for s, n in zip(self.common["starts"], self.common["counts"])
                ]
            ]
        )
        xy = self.common["xy"][self.later["local_index"]]
        starts = np.r_[0, np.cumsum(self.later["counts"])[:-1]]
        centroids = np.add.reduceat(xy, starts) / self.later["counts"][:, None]
        self.later_cells = {
            str(origin): np.array(
                [
                    str(x) + "," + str(y)
                    for x, y in np.floor((centroids - origin) / 200000).astype(int)
                ]
            )
            for origin in [0, 100000]
        }
        dep = specification["later"]["inference"]["conditional_sensitivities"][
            "crossed_dependence"
        ]
        blocks, self.later_block_index = np.unique(
            self.later["blocks"], return_inverse=True
        )
        cells, self.later_cell_index = np.unique(
            self.later_cells["0"], return_inverse=True
        )
        self.later_block_weights = np.random.default_rng(dep["block_seed"]).exponential(
            size=(dep["replicates_per_axis"], len(blocks))
        )
        self.later_cell_weights = np.random.default_rng(dep["cell_seed"]).exponential(
            size=(dep["replicates_per_axis"], len(cells))
        )
        self.position = {
            radius: arrays(self.path / f"later_{radius}km.npz") for radius in [25, 50]
        }

    def jobs(self):
        common = self.spec["common"]["inference"]["sensitivity"]
        jobs = []
        for origin in ["0", "100000"]:
            name = "linear_" + origin
            jobs += [
                (name, "allocation", str(i))
                for i in range(common["measurement_allocations"]["draws"])
            ]
            jobs += [
                (name, "crossed_dependence", str(i))
                for i in range(common["crossed_dependence"]["replicates_per_origin"])
            ]
            jobs += [(name, "phase_omission", "reviewed")]
            jobs += [
                (name, "omit_block", r["block"])
                for r in common["genetic_omissions"]["blocks"]
            ]
            jobs += [
                (name, "omit_cell", r["cell"])
                for r in common["genetic_omissions"]["cells"][origin]
            ]
            jobs += [(name, "region", r) for r in common["regions"]["blocks"]]
        for scenario in read(self.path / "common_omissions.json")["scenarios"]:
            for origin in ["0", "100000"]:
                jobs.append(("linear_" + origin, scenario["family"], scenario["key"]))
        later = self.spec["later"]["inference"]["conditional_sensitivities"]
        for name in self.spec["later"]["design"]["jobs"]:
            jobs += [
                (name, "paternal_location", str(r)) for r in later["paternal_radius_km"]
            ]
            jobs += [
                (name, "crossed_dependence", str(i))
                for i in range(later["crossed_dependence"]["replicates_per_axis"])
            ]
            jobs += [
                (name, "omit_block", key)
                for key in sorted({r["block"] for r in later["omit_blocks"]})
            ]
            jobs += [
                (name, "omit_cell", str(r["origin"]) + "|" + r["cell"])
                for r in later["omit_cells"]
            ]
        assert len(jobs) == 976 + 1652 and len(set(jobs)) == len(jobs)
        return jobs

    def build(self, job):
        name, family, key = job
        d = self.designs[name]
        weights = None
        keep = np.ones(len(d["y"]), bool)
        if d["scope"] == "common":
            origin = name.split("_")[-1]
            if family in ["allocation", "phase_omission"]:
                measurement = (
                    self.phase if family == "phase_omission" else self.measurement
                )
                changed = make_designs(
                    self.common,
                    measurement,
                    self.spec["common"]["design"],
                    **({"draw": int(key)} if family == "allocation" else {}),
                )[origin]
                d = dict(d, x=changed["X"])
            elif family == "crossed_dependence":
                weights = (
                    self.block_weights[int(key), self.block_index]
                    * self.cell_weights[origin][int(key), self.cell_index[origin]]
                )
            elif family == "omit_block":
                keep = self.common["blocks"] != key
            elif family == "omit_cell":
                keep = self.common_designs[origin]["centroid_labels"] != key
            elif family == "region":
                keep = self.regions == key
            elif family in ["omit_site", "omit_pair", "paternal_location"]:
                scenario = next(
                    r
                    for r in read(self.path / "common_omissions.json")["scenarios"]
                    if r["family"] == family and r["key"] == key
                )
                measurement = arrays(self.path / scenario["file"])
                changed = make_designs(
                    self.common, measurement, self.spec["common"]["design"]
                )[origin]
                d = dict(d, x=changed["X"])
            else:
                raise ValueError(family)
        else:
            if family == "crossed_dependence":
                weights = (
                    self.later_block_weights[int(key), self.later_block_index]
                    * self.later_cell_weights[int(key), self.later_cell_index]
                )
            elif family == "paternal_location":
                i = self.spec["later"]["design"]["jobs"].index(name)
                data = self.position[int(key)]
                d = dict(
                    d,
                    x=np.column_stack([data["controls"][i], data["scores"][:, i]])[
                        self.later["local_index"]
                    ],
                )
            elif family == "omit_block":
                keep = self.later["blocks"] != key
            elif family == "omit_cell":
                origin, cell = key.split("|")
                keep = self.later_cells[origin] != cell
            else:
                raise ValueError(family)
        if weights is not None:
            weights *= len(weights) / weights.sum()
        if not keep.all():
            d = subset(d, keep)
        return d, weights, keep

    def fit(self, job):
        name, family, key = job
        started = time.monotonic()
        d, weights, keep = self.build(job)
        detail = dict(
            scope=d["scope"],
            units=len(d["y"]),
            localities=len(d["x"]),
            category_counts=np.bincount(d["y"], minlength=len(d["categories"])),
            omitted_units=int((~keep).sum()),
            design_sha256=hashlib.sha256(d["x"].tobytes()).hexdigest(),
            weights_sha256=(
                hashlib.sha256(weights.tobytes()).hexdigest()
                if weights is not None
                else None
            ),
            contrast_names=list(d["contrasts"]),
            uncertainty="Conditional mode and inverse-Hessian standard error; sensitivity distributions are not posterior intervals.",
        )
        cohort = self.common if d["scope"] == "common" else self.later
        detail["dependence_blocks"] = len(set(cohort["blocks"][keep]))
        weights_by_category = self.spec[d["scope"]]["design"]["lineage_contrasts"]
        detail["lineage_contrast_has_all_categories"] = {
            key: bool(np.all(detail["category_counts"][np.asarray(value) != 0] > 0))
            for key, value in weights_by_category.items()
        }
        if family == "phase_omission":
            detail["primary_support_passed"] = False
            detail["source_limit"] = (
                "Late Neolithic clearance has eight qualifying sites; targeted sensitivity only."
            )
        if family == "region" and (
            len(set(d["y"])) < 2 or np.max(np.std(d["x"][:, -12:], axis=0)) < 1e-8
        ):
            return (
                job,
                "not_estimable",
                dict(
                    detail,
                    reason="Insufficient regional category or exposure variation",
                ),
                None,
            )
        errors = []
        for initial in [self.modes[name + "__mode"], None]:
            try:
                _, fit, result = fit_design(d, initial, weights)
                break
            except (ValueError, np.linalg.LinAlgError) as error:
                errors.append(str(error))
        else:
            return job, "failed", dict(detail, numerical_attempt_errors=errors), None
        detail.update(
            gradient_max=fit["gradient"],
            minimum_curvature=fit["curvature"],
            iterations=fit["iterations"],
            numerical_attempt_errors=errors,
            seconds=time.monotonic() - started,
        )
        stream = io.BytesIO()
        np.savez_compressed(stream, **result)
        return job, "passed", detail, stream.getvalue()


def initialize_worker(measurements, specification, output):
    global _ENGINE, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _ENGINE = Sensitivities(Path(measurements), read(specification), Path(output))


def fit_worker(job):
    return _ENGINE.fit(job)


def sensitivity_run(measurements, specification, output, workers=2):
    """Resume the finite batch, refusing incompatible frozen inputs."""
    if workers < 1:
        raise ValueError("At least one worker is required")
    engine = Sensitivities(measurements, read(specification), output)
    identity = dict(
        specification=sha(specification),
        designs=sha(output / "designs.json"),
        modes=sha(output / "primary_modes.npz"),
        measurements={
            p.name: sha(p)
            for p in sorted(Path(measurements).iterdir())
            if p.suffix in [".npz", ".json"]
        },
    )
    with sqlite3.connect(output / "conditional.sqlite") as db:
        db.execute(
            "create table if not exists metadata(key text primary key,value text not null)"
        )
        db.execute(
            "create table if not exists fits(job text,family text,replicate text,status text,details text,arrays blob,sha256 text,primary key(job,family,replicate))"
        )
        previous = db.execute(
            "select value from metadata where key='input_hashes'"
        ).fetchone()
        if previous and json.loads(previous[0]) != identity:
            raise ValueError("Existing sensitivity output has different frozen inputs")
        failed = list(
            db.execute("select job,family,replicate from fits where status='failed'")
        )
        if failed:
            raise RuntimeError(
                ("Existing failed numerical fits require review", failed)
            )
        db.execute(
            "insert or ignore into metadata values(?,?)",
            ("input_hashes", encode(identity)),
        )
        db.commit()
        done = {tuple(r) for r in db.execute("select job,family,replicate from fits")}
        jobs = engine.jobs()
        pending = [job for job in jobs if job not in done]
        print(
            "Conditional batch",
            len(jobs),
            "total;",
            len(pending),
            "pending",
            flush=True,
        )

        def store(result):
            job, status, detail, blob = result
            db.execute(
                "insert into fits values(?,?,?,?,?,?,?)",
                (
                    *job,
                    status,
                    encode(detail),
                    blob,
                    hashlib.sha256(blob).hexdigest() if blob else None,
                ),
            )
            db.commit()
            if status == "failed":
                raise RuntimeError((job, status, detail))

        if workers == 1:
            for i, job in enumerate(pending, 1):
                store(engine.fit(job))
                if i % 25 == 0 or i == len(pending):
                    print("Conditional fits", i, "/", len(pending), flush=True)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=initialize_worker,
                initargs=(str(measurements), str(specification), str(output)),
            ) as pool:
                futures = [pool.submit(fit_worker, job) for job in pending]
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    store(future.result())
                    if i % 25 == 0 or i == len(pending):
                        print("Conditional fits", i, "/", len(pending), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "sensitivities"])
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "export":
            export(args.measurements, args.specification, args.output)
        else:
            sensitivity_run(
                args.measurements, args.specification, args.output, args.workers
            )


if __name__ == "__main__":
    main()

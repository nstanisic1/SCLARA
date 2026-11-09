"""Retained original-window coverage, geography and archaeological omissions."""

from pathlib import Path
import argparse
import collections
import hashlib
import json
import sqlite3
import numpy as np
from analysis.landscapes import distance, position_draws
from analysis.chronology import read, read_database, source_member, designs, encode


def archaeological_sources(data, spec):
    c = read_database(data / "archaeology.sqlite")
    site_map = {
        r[0]: r[1]
        for r in c.execute(
            "select distinct original_site_id,corrected_site_id from source_site_corrections"
        )
    }

    def corrected(row):
        r = dict(row)
        r["site_id"] = site_map.get(r["site_id"], r["site_id"])
        return r

    sites = {
        r["site_id"]: dict(r)
        for r in c.execute(
            "select * from sites where site_id in(select site_id from period_recording_sites) order by site_id"
        )
    }
    geometry = [
        dict(r)
        for r in c.execute(
            "select * from positions where site_id in(select site_id from period_recording_sites) order by position_id"
        )
    ]
    feature = [
        set(
            r[0]
            for r in c.execute(
                "select site_id from comparison_sites where period_id=? and feature_id=?",
                (p, f),
            )
        )
        for p in spec["periods"]
        for f in ["posthole", "hearth"]
    ]
    recording = [
        set(
            r[0]
            for r in c.execute(
                "select site_id from comparison_recording_sites where period_id=?", (p,)
            )
        )
        for p in spec["periods"]
    ]
    assertions = [
        corrected(r)
        for r in c.execute(
            'select a.*,q.quality,q.eligible,q.decision_json from assertions a join comparison_assertions q using(assertion_id) where q.period_id="Migration"'
        )
    ]
    reviews = {
        r[0]: json.loads(r[1])
        for r in c.execute("select * from comparison_source_reviews")
    }
    quality = [
        set(
            r["site_id"]
            for r in assertions
            if r["feature_id"] == f and r["eligible"] and r["quality"] in ["Q1", "Q2"]
        )
        for f in ["posthole", "hearth"]
    ]
    corrections = {
        r[0]: json.loads(r[1])
        for r in c.execute("select * from comparison_date_corrections")
    }
    rules = json.loads(source_member(data, "comparison_decisions.json"))[
        "function_and_quality_rules"
    ]
    eligible = []
    for row in c.execute(
        "select * from assertions where disposition='eligible_reported_context' and start_year>=400 and end_year<550 order by assertion_id"
    ):
        r = corrected(row)
        aid = r["assertion_id"]
        if aid in corrections:
            lo, hi = corrections[aid]["calendar_envelope"]
            if not (lo >= 400 and hi < 550):
                continue
        r.update(
            {
                k: v
                for k, v in rules["decisions"].get(aid, {}).items()
                if k in ["short_lived", "material", "species"]
            }
        )
        eligible.append(r)
    assert set(r["site_id"] for r in eligible) == recording[3]
    qrec = {
        r["site_id"]
        for r in eligible
        if r["short_lived"] or r["assertion_id"] in rules["q2_qualified_ids"]
    }
    all_assertions = [
        corrected(r)
        for r in c.execute(
            "select * from assertions where disposition='eligible_reported_context'"
        )
    ]
    groups = collections.defaultdict(set)
    group_refs = {}
    used = set.union(*recording)
    for r in all_assertions:
        if r["site_id"] not in used:
            continue
        src = json.loads(r["source_row_json"])
        keys = []
        investigation = src.get("uppdragsnummer", "").strip()
        if investigation and investigation.lower() not in [
            "none",
            "nan",
            "null",
            "0",
            "-",
        ]:
            keys.append("investigation:" + investigation)
        if all(
            src.get(k, "").strip()
            for k in ["author", "publication_year", "title", "journal"]
        ):
            ref = " | ".join(
                " ".join(src[k].split()).casefold()
                for k in ["author", "publication_year", "title", "journal"]
            )
            keys.append("publication:" + hashlib.sha256(ref.encode()).hexdigest()[:16])
            group_refs[keys[-1]] = ref
        for key in keys:
            groups[key].add(r["site_id"])
    qualified = [(k, v) for k, v in groups.items() if len(v & recording[3]) >= 2]
    qualified.sort(key=lambda kv: (-len(kv[1] & recording[3]), kv[0]))
    seen = set()
    selected = []
    for key, ids in qualified:
        identity = tuple(sorted(ids))
        if identity not in seen:
            selected.append(
                dict(
                    key=key,
                    sites=list(identity),
                    focal_sites=len(ids & recording[3]),
                    reference=group_refs.get(key),
                )
            )
            seen.add(identity)
    c.close()
    return dict(
        sites=sites,
        geometry=geometry,
        feature=feature,
        recording=recording,
        assertions=assertions,
        reviews=reviews,
        quality=quality,
        qrec=qrec,
        eligible=eligible,
        groups=selected[:20],
        groups_available=len(selected),
    )


def archaeological_spatial_arrays(cohort, source, current):
    geometry = source["geometry"]
    sites = source["sites"]
    union = set.union(*source["feature"])
    selected = [g for g in geometry if g["site_id"] in union]
    ids = sorted(union)
    index = {s: i for i, s in enumerate(ids)}
    radii = {}
    for g in geometry:
        key = (g["latitude"], g["longitude"])
        radii[key] = max(
            radii.get(key, 0), sites[g["site_id"]]["source_uncertainty_m"] or 0
        )
    unique = sorted({(g["latitude"], g["longitude"]) for g in selected})
    draws = archived_archaeological_draws(geometry, sites, current)
    gd = np.array([draws[g["latitude"], g["longitude"]] for g in selected]).transpose(
        1, 0, 2
    )
    positions = cohort["positions"]
    nominal = np.array([[p["lat"], p["lon"]] for p in positions])
    values = np.zeros((64, len(positions), len(ids)))
    gindex = np.array([index[g["site_id"]] for g in selected])
    for draw in range(64):
        prox = np.exp(-distance(nominal, gd[draw]) / 50)
        for j in range(len(ids)):
            values[draw, :, j] = prox[:, gindex == j].max(1)
    groups = {}
    for s in sorted(set.union(*source["recording"])):
        groups[sites[s]["position_group"]] = sites[s]
    group_ids = sorted(groups)
    coordinates = np.array(
        [[groups[g]["latitude"], groups[g]["longitude"]] for g in group_ids]
    )
    recorded = distance(nominal, coordinates) <= 75
    return dict(
        values=values, ids=ids, index=index, recorded=recorded, group_ids=group_ids
    )


def archaeological_measurement(source, spatial, omit=(), quality=False):
    omit = set(omit)
    feature = [s - omit for s in source["feature"]]
    recording = [s - omit for s in source["recording"]]
    if quality:
        feature[-2:] = source["quality"]
        recording[-1] = source["qrec"]
    proximity = np.column_stack(
        [
            spatial["values"][:, :, [spatial["index"][s] for s in sorted(ids)]]
            .max(2)
            .mean(0)
            for ids in feature
        ]
    )
    rec = []
    for ids in recording:
        groups = {source["sites"][s]["position_group"] for s in ids}
        cols = [i for i, g in enumerate(spatial["group_ids"]) if g in groups]
        rec.append(np.log1p(spatial["recorded"][:, cols].sum(1)))
    return dict(proximity=proximity, recording=np.column_stack(rec))


def restrict_units(d, keep):
    starts = np.r_[0, np.cumsum(d["counts"])[:-1]]
    rows = np.concatenate(
        [np.arange(s, s + n) for s, n in zip(starts[keep], d["counts"][keep])]
    )
    return dict(d, X=d["X"][rows], y=d["y"][keep], counts=d["counts"][keep])


def archaeological_support(cohort, measurement, source, spec, quality=False):
    local = cohort["local"]
    counts = cohort["counts"]
    starts = cohort["starts"]
    near = (measurement["proximity"][local, -2:] >= np.exp(-2)).all(1)
    keep = np.minimum.reduceat(near, starts).astype(bool)
    I = np.isin(cohort["y"], [0, 1, 2])
    Z = cohort["y"] == 3
    out = {
        "units": int(keep.sum()),
        "I1_units": int((keep & I).sum()),
        "Z284_units": int((keep & Z).sum()),
        "I1_blocks": len(set(cohort["blocks"][keep & I])),
        "Z284_blocks": len(set(cohort["blocks"][keep & Z])),
        "lineage_counts": {
            c: int((keep & (cohort["y"] == i)).sum())
            for i, c in enumerate(spec["categories"])
        },
        "grids": {},
    }
    weights = np.repeat(1 / counts, counts)
    E = measurement["proximity"][local, -2] - measurement["proximity"][local, -1]
    out["raw_exposure_sd"] = float(
        np.sqrt(np.average((E - np.average(E, weights=weights)) ** 2, weights=weights))
    )
    out["raw_exposure_sd_population"] = (
        "Full reference cohort; inherited gate definition"
    )
    covered = np.repeat(keep, counts)
    cw = weights[covered]
    ce = E[covered]
    out["covered_raw_exposure_sd"] = float(
        np.sqrt(np.average((ce - np.average(ce, weights=cw)) ** 2, weights=cw))
    )
    for origin, d in designs(cohort, measurement, spec).items():
        ds = restrict_units(d, keep)
        w = np.repeat(1 / ds["counts"], ds["counts"])
        x = ds["X"][:, -1]
        nuis = ds["X"][:, :-1]
        residual = (
            x
            - nuis
            @ np.linalg.lstsq(nuis * np.sqrt(w[:, None]), x * np.sqrt(w), rcond=None)[0]
        )
        sd = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
        ratio = float(np.sqrt(np.average(residual**2, weights=w)) / sd) if sd else 0
        cells = []
        unitE = np.add.reduceat(E, starts) / counts
        for cell in sorted(set(d["centroid_labels"])):
            mask = (d["centroid_labels"] == cell) & keep
            ni = int((mask & I).sum())
            nz = int((mask & Z).sum())
            ranges = {}
            for name, codes in [("I1", [0, 1, 2]), ("Z284", [3]), ("U106", [4])]:
                vals = unitE[mask & np.isin(cohort["y"], codes)]
                ranges[name] = (
                    np.quantile(vals, [0.1, 0.5, 0.9]).tolist() if len(vals) else None
                )
            overlap = None
            if ni and nz:
                overlap = float(
                    min(ranges["I1"][2], ranges["Z284"][2])
                    - max(ranges["I1"][0], ranges["Z284"][0])
                )
            cells.append(
                dict(
                    cell=cell,
                    I1=ni,
                    Z284=nz,
                    units=int(mask.sum()),
                    ranges=ranges,
                    central_exposure_overlap=overlap,
                    shared=ni >= 5 and nz >= 3 and overlap is not None and overlap > 0,
                )
            )
        out["grids"][origin] = dict(
            cells=cells,
            shared_cells=sum(c["shared"] for c in cells),
            residual_sd_ratio=ratio,
        )
    levels = source["quality"] if quality else source["feature"][-2:]
    out["features"] = {
        f: dict(
            sites=len(ids), counties=len({source["sites"][s]["county"] for s in ids})
        )
        for f, ids in zip(["posthole", "hearth"], levels)
    }
    g = spec["appraisal"]["quality_gate"]
    checks = {
        k: out[k] >= g["minimum_" + k]
        for k in [
            "I1_units",
            "Z284_units",
            "I1_blocks",
            "Z284_blocks",
            "raw_exposure_sd",
        ]
    }
    checks["features"] = all(
        x["sites"] >= g["minimum_sites_per_feature"]
        and x["counties"] >= g["minimum_counties_per_feature"]
        for x in out["features"].values()
    )
    checks["regional_overlap"] = all(
        x["shared_cells"] >= g["minimum_shared_cells_each_origin"]
        for x in out["grids"].values()
    )
    checks["residual_variation"] = all(
        x["residual_sd_ratio"] >= g["minimum_residual_sd_ratio_each_origin"]
        for x in out["grids"].values()
    )
    out["checks"] = checks
    out["gate_pass"] = all(checks.values())
    return out, keep


def quadratic_coordinates(cohort, d):
    # Once-only moments use equal unit weight distributed over alternative localities.
    xy = cohort["xy"][cohort["local"]] / 100000
    w = np.repeat(1 / cohort["counts"], cohort["counts"])
    center = np.average(xy, axis=0, weights=w)
    sd = np.sqrt(np.average((xy - center) ** 2, axis=0, weights=w))
    z = (xy - center) / sd
    raw = np.column_stack([z[:, 1] ** 2, z[:, 1] * z[:, 0], z[:, 0] ** 2])
    mu = np.average(raw, axis=0, weights=w)
    scale = np.sqrt(np.average((raw - mu) ** 2, axis=0, weights=w))
    q = (raw - mu) / scale
    X = np.column_stack([d["X"][:, :-4], q, d["X"][:, -4:]])
    vectors = {
        k: np.column_stack(
            [v.reshape(7, -1)[:, :-4], np.zeros((7, 3)), v.reshape(7, -1)[:, -4:]]
        ).ravel()
        for k, v in d["vectors"].items()
    }
    receipt = dict(
        coordinate_order=["easting", "northing"],
        xy_unit="100km",
        weights="1/locality_count within each unit",
        xy_center=center,
        xy_sd=sd,
        term_order=["north_squared", "north_east", "east_squared"],
        term_center=mu,
        term_sd=scale,
        prior_scale=0.5,
    )
    return (
        dict(
            d,
            X=X,
            scales=list(d["scales"][:-4]) + [0.5] * 3 + list(d["scales"][-4:]),
            vectors=vectors,
        ),
        receipt,
    )


def archived_archaeological_draws(geometry, sites, current):
    allowed = set(current["legacy"]["archaeological_position_stream_ids"])
    streams = current["common"]["position_streams"]
    groups = collections.defaultdict(list)
    for row in geometry:
        if row["position_id"] not in allowed:
            raise ValueError("Unfrozen archaeological position identity")
        groups[tuple(streams[row["position_id"]])].append(row)
    result = {}
    for (latitude, longitude, radius, seed), rows in groups.items():
        if (
            max(sites[r["site_id"]]["source_uncertainty_m"] or 0 for r in rows)
            != radius
        ):
            raise ValueError("Source positional radius differs")
        draws = position_draws(latitude, longitude, radius, seed, 64)
        for row in rows:
            if (
                abs(row["latitude"] - latitude) >= 1e-9
                or abs(row["longitude"] - longitude) >= 1e-9
            ):
                raise ValueError("Original source coordinate differs")
            result[row["latitude"], row["longitude"]] = draws
    return result


def build_designs(data, measurements, current):
    spec = current["legacy"]
    with np.load(measurements / "common_cohort.npz", allow_pickle=False) as source:
        cohort = dict(source)
    cohort["positions"] = read(data / "paternal.json")["spatial_positions"]
    source = archaeological_sources(data, spec)
    spatial = archaeological_spatial_arrays(cohort, source, current)
    measured = archaeological_measurement(source, spatial)
    nominal = designs(cohort, measured, spec)
    broad, keep = archaeological_support(cohort, measured, source, spec)
    quality, _ = archaeological_support(
        cohort,
        archaeological_measurement(source, spatial, quality=True),
        source,
        spec,
        True,
    )
    if quality["gate_pass"]:
        raise ValueError(
            "The withheld material restriction now passes: a changed scientific specification requires review"
        )
    constructed, quadratic = {}, {}
    for origin, d in nominal.items():
        constructed["baseline_" + origin] = d
        constructed["coverage_" + origin] = restrict_units(d, keep)
        constructed["quadratic_" + origin], quadratic[origin] = quadratic_coordinates(
            cohort, d
        )
    for name, sites in [
        ("kvarnbacken", ["L2008:2604", "L2008:575"]),
        ("bobygden", ["L1948:7082"]),
    ]:
        for origin, d in designs(
            cohort, archaeological_measurement(source, spatial, sites), spec
        ).items():
            constructed[name + "_omit_" + origin] = d
    jobs = {}
    for name, d in constructed.items():
        jobs[name] = dict(
            job_id=name,
            scope="Original-window archaeological appraisal",
            seed=spec["appraisal"]["seeds"][name],
            unit_ids=(
                cohort["unit_ids"][keep]
                if name.startswith("coverage_")
                else cohort["unit_ids"]
            ),
            x=d["X"],
            y=d["y"],
            counts=d["counts"],
            starts=np.r_[0, np.cumsum(d["counts"])[:-1]],
            scales=d["scales"],
            categories=spec["categories"],
            contrasts=d["vectors"],
        )
    singles = {}
    for site in ["L2008:2604", "L2008:575"]:
        for origin, d in designs(
            cohort, archaeological_measurement(source, spatial, [site]), spec
        ).items():
            singles[site + "__" + origin] = d
    return (
        jobs,
        dict(
            broad_support=broad, quality_support=quality, quadratic_definition=quadratic
        ),
        singles,
    )


def export(data, measurements, specification, output):
    import shutil
    from analysis.infer import fit_design
    from analysis.model import LabelMixture

    if (output / "design_freeze.json").exists():
        raise FileExistsError(
            "Frozen appraisal export exists; use a fresh output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    jobs, support, singles = build_designs(data, measurements, read(specification))
    modes, diagnostics = {}, {}
    for name, d in jobs.items():
        model, fit, estimates = fit_design(d)
        rng = np.random.default_rng(d["seed"])
        checks = []
        for v in [
            fit["mode"],
            fit["mode"] + rng.normal(0, 0.02, len(fit["mode"])),
            fit["mode"] + rng.normal(0, 0.1, len(fit["mode"])),
        ]:
            value, gradient = model.evaluate(v)
            checks.append(dict(v=v, log_density=-value, gradient=-gradient))
        d["density_checks"] = checks
        modes[name] = estimates
        diagnostics[name] = {
            k: v for k, v in fit.items() if k not in ["mode", "covariance"]
        }
    omissions = {}
    for name, d in singles.items():
        fit = LabelMixture(d["X"], d["y"], d["counts"], d["scales"], 8).optimize()
        vectors = np.array(list(d["vectors"].values()))
        omissions[name] = dict(
            contrast_names=list(d["vectors"]),
            mode=fit["mode"],
            contrast_mean=vectors @ fit["mode"],
            contrast_covariance=vectors @ fit["covariance"] @ vectors.T,
            diagnostics={
                k: v for k, v in fit.items() if k not in ["mode", "covariance"]
            },
        )
    for filename, value in [
        ("designs.json", dict(jobs=jobs)),
        ("support.json", support),
        ("modes.json", modes),
        ("numerical_checks.json", diagnostics),
        ("single_site_omissions.json", omissions),
    ]:
        (output / filename).write_text(encode(value) + "\n", encoding="utf-8")
    shutil.copyfile(specification, output / "specification.json")
    shutil.copyfile(Path(__file__).with_name("mixture.stan"), output / "mixture.stan")
    frozen = {
        key: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for key, name in [
            ("specification_sha256", "specification.json"),
            ("designs_sha256", "designs.json"),
            ("stan_sha256", "mixture.stan"),
        ]
    }
    (output / "design_freeze.json").write_text(encode(frozen) + "\n", encoding="utf-8")
    print("Exported ten retained archaeological-appraisal posterior jobs", flush=True)


def main():
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument(
        "--specification", type=Path, default=Path("analysis/specification.json")
    )
    parser.add_argument("--output", type=Path, default=Path("build/appraisal"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        export(args.data, args.measurements, args.specification, args.output)


if __name__ == "__main__":
    main()

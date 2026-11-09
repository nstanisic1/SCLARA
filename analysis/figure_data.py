"""Assemble the principal and supplementary figures from portable scientific inputs.

No profiles are selected or fitted here. The plotted comparisons and ancient
source layers are fixed by the study specification and source qualifications.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import sqlite3
import zipfile
import numpy as np
from pyproj import Transformer
from analysis.results import conditional_summary, posterior_summaries


def read(path):
    b = Path(path).read_bytes()
    return json.loads(gzip.decompress(b) if Path(path).suffix == ".gz" else b)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def conn(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def assemble(data, measurements, results, ancient, specification, output):
    proj = Transformer.from_crs(4326, 3006, always_xy=True)
    inputs = [
        specification,
        data / "paternal.json",
        data / "sources/paternal.zip",
        measurements / "common.npz",
        measurements / "common_cohort.npz",
        measurements / "common_support.json",
        measurements / "common_pool.json.gz",
        results / "current/conditional.sqlite",
        results / "chronology.sqlite",
        results / "benchmarks.sqlite",
    ] + [
        ancient / n
        for n in [
            "identity_inventory.json",
            "branch_crosswalk.json",
            "dated_observations.json",
            "eligibility.json",
        ]
    ]
    e = {
        "schema": "sclara-figure-evidence-1",
        "inputs": {p.name: sha(p) for p in inputs},
        "scope": "Fixed current measurements and complete specified comparisons. Figures are descriptive and do not add effect fits.",
    }
    spec = read(specification)
    e["basis"] = spec["common"]["basis"]
    e["families"] = spec["common"]["families"]
    e["periods"] = [p[0] for p in spec["common"]["periods"]]
    e["period_labels"] = [p.replace("_", " ") for p in e["periods"]]
    e["period_bounds"] = [p[1:] for p in spec["common"]["periods"]]
    e["support"] = [
        x
        for x in read(measurements / "common_support.json")["support"]
        if x["period"] in e["periods"]
    ]
    assert len(e["support"]) == 28
    positions = read(data / "paternal.json")["spatial_positions"]
    position_ids = [p["position_id"] for p in positions]
    e["categories"] = spec["common"]["design"]["categories"]
    ll = np.array([[p["lon"], p["lat"]] for p in positions])
    xy = np.column_stack(proj.transform(ll[:, 0], ll[:, 1])) / 1000
    with np.load(measurements / "common_cohort.npz") as c:
        local = c["local"]
        category = np.repeat(c["y"], c["counts"])
        e["cohort"] = dict(
            units=len(c["y"]),
            localities=len(local),
            blocks=len(set(c["blocks"])),
            positions=len(positions),
            category_counts=np.bincount(
                c["y"], minlength=len(e["categories"])
            ).tolist(),
            fitted_positions=len(np.unique(local)),
        )
    with np.load(measurements / "later_0km.npz") as later:
        later_scores = later["scores"]
    e["later_jobs"] = spec["later"]["design"]["jobs"]
    e["inputs"]["later_0km.npz"] = sha(measurements / "later_0km.npz")
    with np.load(measurements / "common.npz") as a:
        scores = a["scores"]
        proximities = a["proximity_draws"].mean(0)
    # Country outlines are the same frozen source used by the measurement pipeline.
    archive = data / "sources/paternal.zip"
    with zipfile.ZipFile(archive) as z:
        raw = z.read("references/country_boundaries.geojson")
        bound = json.loads(raw)
    e["boundary_member_sha256"] = hashlib.sha256(raw).hexdigest()
    e["boundaries"] = []
    for f in bound["features"]:
        if f["properties"]["ADMIN"] not in [
            "Sweden",
            "Norway",
            "Finland",
            "Denmark",
            "Estonia",
            "Latvia",
            "Lithuania",
            "Poland",
            "Germany",
        ]:
            continue
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            a = np.array(poly[0])
            x, y = proj.transform(a[:, 0], a[:, 1])
            e["boundaries"].append(
                {
                    "country": f["properties"]["ADMIN"],
                    "xy_km": np.column_stack([x, y]).tolist(),
                }
            )
    # Scale the coordinates once after transformation, preserving the source polygons.
    for g in e["boundaries"]:
        g["xy_km"] = (np.array(g["xy_km"]) / 1000).tolist()
    pool = read(measurements / "common_pool.json.gz")
    e["archaeology_maps"] = []
    for period in e["periods"]:
        pi = [p[0] for p in pool["periods"]].index(period)
        ids = sorted(
            {x["site_id"] for x in pool["rows"] if x["period_probability"][pi] >= 0.1}
        )
        sites = [pool["sites"][i] for i in ids]
        x, y = proj.transform(
            [s["longitude"] for s in sites], [s["latitude"] for s in sites]
        )
        e["archaeology_maps"].append(
            {
                "period": period,
                "site_ids": ids,
                "xy_km": (np.column_stack([x, y]) / 1000).tolist(),
            }
        )
    e["map_limits_km"] = [240, 980, 6100, 7740]
    # Retain all contrasts and the full fixed sensitivity family in the plotting inputs.
    posterior = posterior_summaries(results / "current")
    conditional = conditional_summary(results / "current/conditional.sqlite")
    for p in sorted((results / "current").glob("*_contrasts.csv")):
        e["inputs"][p.name] = sha(p)
    e["posterior"] = {
        job: {
            n: {k: v[k] for k in ["mean", "lower", "upper"]}
            for n, v in posterior[job].items()
        }
        for job in ["linear_0", "linear_100000", "quadratic_0", "quadratic_100000"]
    }
    e["sensitivity"] = {}
    e["later_sensitivity"] = {}
    for f in conditional["aggregates"]:
        names = conditional["contrast_names"][f["job"]]
        values = {
            n: {
                k: float(f[k][i])
                for k in ["median", "q025", "q975", "minimum", "maximum"]
            }
            for i, n in enumerate(names)
        }
        if f["job"].startswith("linear_"):
            e["sensitivity"][f["family"] + "_" + f["job"].split("_")[-1]] = values
        elif f["family"] == "crossed_dependence":
            e["later_sensitivity"][f["job"]] = {
                n: dict(v, contrast=n, fraction_positive=f["fraction_positive"][i])
                for i, (n, v) in enumerate(values.items())
            }
    e["block_omission"] = {}
    e["phase_omission"] = {}
    for f in conditional["fits"]:
        if not f["job"].startswith("linear_"):
            continue
        if (f["family"] == "omit_block" and f["replicate"] == "B-26258") or f[
            "family"
        ] == "phase_omission":
            target = (
                e["block_omission"]
                if f["family"] == "omit_block"
                else e["phase_omission"]
            )
            target[f["job"].split("_")[-1]] = {
                n: {
                    "mean": f["means"][i],
                    "lower": f["means"][i] - 1.959963984540054 * f["sd"][i],
                    "upper": f["means"][i] + 1.959963984540054 * f["sd"][i],
                }
                for i, n in enumerate(conditional["contrast_names"][f["job"]])
            }
    # Regional restrictions describe six groups under two grids, not twelve regions.
    with np.load(measurements / "common_cohort.npz") as c:
        cohort = {key: c[key] for key in c.files}
    assignments = read(measurements / "common_regions.json")
    location_regions = np.array([r["region"] or "unassigned" for r in assignments])[
        cohort["local"]
    ]
    unit_regions = np.array(
        [
            next(iter(v)) if len(v) == 1 else "mixed"
            for v in [
                set(location_regions[start : start + count])
                for start, count in zip(cohort["starts"], cohort["counts"])
            ]
        ]
    )
    e["regional"] = []
    for f in conditional["fits"]:
        if f["family"] != "region":
            continue
        mask = unit_regions == f["replicate"]
        counts = np.bincount(cohort["y"][mask], minlength=len(e["categories"]))
        blocks = [
            len(set(cohort["blocks"][mask & (cohort["y"] == k)]))
            for k in range(len(e["categories"]))
        ]
        names = conditional["contrast_names"][f["job"]]
        values = {}
        for name in [
            "Late_Neolithic__PC3__I1_vs_R1a_Z284",
            "Late_Neolithic_minus_Roman_Iron_Age__PC3__I1_vs_R1a_Z284",
        ]:
            i = names.index(name)
            values[name] = dict(
                mode=f["means"][i],
                lower=f["means"][i] - 1.959963984540054 * f["sd"][i],
                upper=f["means"][i] + 1.959963984540054 * f["sd"][i],
            )
        e["regional"].append(
            dict(
                region=f["replicate"],
                job=f["job"],
                units=int(mask.sum()),
                category_units=counts.tolist(),
                category_blocks=blocks,
                total_blocks=len(set(cohort["blocks"][mask])),
                focal_display_support=bool(
                    np.all(counts[:4] >= 10) and np.all(np.array(blocks[:4]) >= 3)
                ),
                display_rule="At least ten units and three dependence blocks in each constituent category of I1–Z284; display qualification only, not a refit or retrospective sampling rule.",
                values=values,
            )
        )
    e["regional_shapes"] = []
    for feature in read(data / "sources/regions.geojson")["features"]:
        properties = feature["properties"]
        if properties["CNTR_CODE"] != "SE":
            continue
        region = properties["NUTS_ID"]
        region = "SE3" if region.startswith("SE3") else region
        if region not in {r["region"] for r in e["regional"]}:
            continue
        g = feature["geometry"]
        polygons = (
            g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        )
        for polygon in polygons:
            points = np.asarray(polygon[0])
            px, py = proj.transform(points[:, 0], points[:, 1])
            e["regional_shapes"].append(
                dict(region=region, xy_km=(np.column_stack([px, py]) / 1000).tolist())
            )
    chosen = cohort["blocks"] == "B-26258"
    indices = np.unique(cohort["local"][np.repeat(chosen, cohort["counts"])])
    e["influential_block"] = dict(
        block="B-26258",
        units=int(chosen.sum()),
        locality_rows=int(cohort["counts"][chosen].sum()),
        distinct_positions=len(indices),
        lon_lat=ll[indices].tolist(),
        xy_km=xy[indices].tolist(),
    )
    e["context_examples"] = read(data / "reports/reviews.json")[
        "measurement_context_examples"
    ]
    e["inputs"].update(
        {
            "common_regions.json": sha(measurements / "common_regions.json"),
            "regions.geojson": sha(data / "sources/regions.geojson"),
            "context_reviews.json": sha(data / "reports/reviews.json"),
        }
    )
    e["later_posterior"] = {
        job: {
            n: {k: v[k] for k in ["mean", "lower", "upper"]}
            for n, v in posterior[job].items()
        }
        for job in spec["later"]["design"]["jobs"]
    }
    # A geographical example is chosen without consulting lineage labels.
    mann = pool["sites"]["L1996:7017"]
    row = next(x for x in pool["rows"] if x["assertion_id"] == "C14:23457")
    p = np.deg2rad(ll)
    a, b = np.deg2rad([mann["longitude"], mann["latitude"]])
    dist = (
        2
        * 6371.0088
        * np.arcsin(
            np.sqrt(
                np.sin((p[:, 1] - b) / 2) ** 2
                + np.cos(b) * np.cos(p[:, 1]) * np.sin((p[:, 0] - a) / 2) ** 2
            )
        )
    )
    ix = int(np.argmin(dist))
    e["worked_example"] = {
        "site_id": mann["site_id"],
        "assertion": row,
        "site": mann,
        "position_id": position_ids[ix],
        "position_lon_lat": ll[ix].tolist(),
        "selection": "Closest represented nominal origin to the site representative point, using spherical distance without lineage labels. This isolated nominal distance is illustrative, not the uncertainty-averaged family measurement.",
        "distance_km": float(dist[ix]),
        "nominal_site_proximity": float(np.exp(-dist[ix] / 50)),
        "full_mean_family_proximities": proximities[ix, 0, :].tolist(),
        "full_Late_Neolithic_PC3": float(scores[ix, 0, 2]),
    }
    # Person-level assignments and source qualifications are reconciled upstream.
    anc = read(ancient / "eligibility.json")["observations"]
    for a in anc:
        if a["latitude"] is not None and a["longitude"] is not None:
            a["xy_km"] = (
                np.array(proj.transform(a["longitude"], a["latitude"])) / 1000
            ).tolist()
    e["ancient_observations"] = anc
    # Published source nomenclature is explicit; fields never upgrade a broad label.
    e["ancient_limits"] = [
        "Source-selected collection, not a frequency sample or exhaustive census.",
        "Source-specific dates are displayed without assigning new date likelihoods.",
        "Contextual and direct dates remain distinct; a plotted range can overlap more than one window.",
        "The contemporary Migration subset has three retained broad-I1 individuals at two coordinates and no qualified U106/Z284 comparison.",
        "Low-coverage and tentative early I1 calls remain flagged; no modern descent or arrival date is inferred.",
    ]
    # Original discontinuous windows remain separate from the common-period measurement.
    with conn(results / "chronology.sqlite") as c:
        e["paired"] = {}
        for scenario, origin, contrast, raw in c.execute(
            "SELECT scenario,origin,contrast,details FROM paired_summary"
        ):
            e["paired"].setdefault(scenario, {}).setdefault(origin, {})[contrast] = (
                json.loads(raw)
            )
    with conn(results / "benchmarks.sqlite") as c:
        e["benchmarks"] = {}
        e["first_benchmark"] = {}
        for family, scenario, method, raw in c.execute(
            "SELECT family,scenario,method,details FROM summaries"
        ):
            if family == "chronology":
                e["benchmarks"].setdefault(scenario, {})[method] = json.loads(raw)
            elif family == "first" and method != "paired":
                e["first_benchmark"][scenario + "__" + method] = json.loads(raw)
    example = e["worked_example"]
    q = np.asarray(example["full_mean_family_proximities"])
    basis = e["basis"]
    terms = (
        ((q - q.mean() - np.asarray(basis["center"])) / np.asarray(basis["scale"]))
        * np.asarray(basis["loadings"])[:, 2]
        / np.sqrt(basis["eigenvalues"][2])
    )
    if abs(terms.sum() - example["full_Late_Neolithic_PC3"]) > 1e-12:
        raise ValueError(
            "Worked-example feature contributions do not reconstruct its score"
        )
    example["PC3_feature_contributions"] = dict(zip(e["families"], terms.tolist()))
    support_path = results / "common_support/support.json"
    if support_path.exists():
        support = read(support_path)
        e["inputs"]["shared_recording_support.json"] = sha(support_path)
        kept_ids = set(support["kept_unit_ids"])
        with np.load(measurements / "common_cohort.npz") as cohort:
            unit_keep = np.array([u in kept_ids for u in cohort["unit_ids"]])
            locality_keep = np.repeat(unit_keep, cohort["counts"])
            kept_positions = np.unique(cohort["local"][locality_keep])
            original_positions = np.unique(cohort["local"])
            category_blocks = {
                category: len(set(cohort["blocks"][cohort["y"] == i]))
                for i, category in enumerate(e["categories"])
            }
        e["shared_support_map"] = dict(
            retained_position_indices=kept_positions.tolist(),
            other_position_indices=np.setdiff1d(
                original_positions, kept_positions
            ).tolist(),
            units=support["units"],
            omitted_units=support["omitted_units"],
            categories=support["categories"],
            original_category_blocks=category_blocks,
            rule=support["rule"],
            interpretation=support["interpretation"],
        )
    evaluation_path = results / "evaluation/evaluation.sqlite"
    if evaluation_path.exists():
        e["inputs"]["current_evaluation.sqlite"] = sha(evaluation_path)
        e["current_evaluation"] = {}
        with conn(evaluation_path) as c:
            for condition, method, raw in c.execute("SELECT * FROM summaries"):
                e["current_evaluation"].setdefault(condition, {})[method] = json.loads(
                    raw
                )
        fixture = results / "evaluation/fixture.json"
        e["inputs"]["current_evaluation_fixture.json"] = sha(fixture)
        e["current_evaluation_contract"] = read(fixture)["contract"]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "measurements.npz",
        origin_lon_lat=ll,
        origin_xy_km=xy,
        local_indices=local,
        local_categories=category,
        common_scores=scores,
        common_proximity=proximities,
        later_scores=later_scores,
    )
    save(output / "evidence.json", e)
    print(
        "Saved",
        len(anc),
        "ancient observations;",
        sum(len(v) for v in e["posterior"].values()),
        "common and",
        sum(len(v) for v in e["later_posterior"].values()),
        "later contrasts",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key, value in dict(
        data="data",
        measurements="build/measurements",
        results="results",
        ancient="build/ancient",
        specification="analysis/specification.json",
        output="build/figure_data",
    ).items():
        p.add_argument("--" + key, type=Path, default=Path(value))
    a = p.parse_args()
    assemble(**vars(a))


if __name__ == "__main__":
    main()

"""Prespecified upward branch-cut sensitivities on the original chronology.

The two caps alter the represented source labels before non-nested pruning and
family grouping. They do not create independent individuals, date lineages or
assign an upstream record to an unobserved descendant.
"""

from pathlib import Path
import argparse
import collections
import copy
import functools
import hashlib
import json
import shutil
import sqlite3
from itertools import combinations
import numpy as np
from scipy.linalg import helmert
from pyproj import Transformer
from threadpoolctl import threadpool_limits
from analysis.units import construct_units
from analysis.measure import (
    lineage_category,
    terrain_controls,
    registry_cells,
    source_member,
)
from analysis.landscapes import distance, position_draws
from analysis.model import LabelMixture
from analysis.chronology import read, encode


def flexible_cohort(genetic, archive, current, scope="legacy"):
    """Same frozen transformations and whole-unit terrain rule for a declared alternate cohort."""
    spec = current[scope]
    categories = current["categories"]
    category = lambda path: lineage_category(path, current["ancestral_nodes"])
    pos = genetic["spatial_positions"]
    ix = {p["position_id"]: i for i, p in enumerate(pos)}
    terrain = terrain_controls(pos, archive)
    units = []
    excluded = []
    for u in genetic["analysis_units"]:
        labels = {category(genetic["lineage_paths"][k]) for k in u["branch_node_ids"]}
        if len(labels) != 1:
            excluded.append((u["unit_id"], "mixed_category"))
            continue
        if not all(
            np.isfinite(terrain[ix[l["position_id"]]]).all() for l in u["localities"]
        ):
            excluded.append((u["unit_id"], "terrain"))
            continue
        units.append(dict(u, category=labels.pop()))
    local = np.array([ix[l["position_id"]] for u in units for l in u["localities"]])
    counts = np.array([len(u["localities"]) for u in units])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    y = np.array([categories.index(u["category"]) for u in units])
    lat = np.array([p["lat"] for p in pos])
    lon = np.array([p["lon"] for p in pos])
    reg = [
        r
        for r in genetic["records"]
        if r["reconciled_screen_candidate"] and r["source_lat"] is not None
    ]
    density = collections.Counter(
        registry_cells(
            np.array([r["source_lat"] for r in reg]),
            np.array([r["source_lon"] for r in reg]),
        )
    )
    dc = np.array([np.log(max(1, density[k])) for k in registry_cells(lat, lon)])
    raw = np.column_stack(
        [
            (lat - 62) * 111.195 / 100,
            (lon - 16) * 111.32 * np.cos(np.radians(62)) / 100,
            dc,
            terrain,
        ]
    )
    norm = spec["normalization"]
    controls = (raw - norm["nonarchaeological_center"]) / norm["nonarchaeological_sd"]
    tf = Transformer.from_crs(4326, 3006, always_xy=True)
    xy = np.array([tf.transform(p["lon"], p["lat"]) for p in pos])
    return dict(
        genetic=genetic,
        positions=pos,
        units=units,
        local=local,
        counts=counts,
        starts=starts,
        y=y,
        controls=controls,
        xy=xy,
        blocks=np.array([u["dependence_block"] for u in units]),
        unit_ids=np.array([u["unit_id"] for u in units]),
        excluded=excluded,
    )


def flexible_designs(cohort, measurement, spec, additional_centers=None):
    """Extend regional basis only when the declared source-screened reference adds cells."""
    local = cohort["local"]
    norm = spec["normalization"]
    prox = measurement["proximity"][local]
    rec = measurement["recording"][local]
    E = (prox[:, ::2] - prox[:, 1::2] - norm["exposure_centers"]) / norm[
        "common_exposure_sd"
    ]
    opp = prox[:, ::2] + prox[:, 1::2]
    raw = np.column_stack([rec[:, 1:], opp[:, 1:], rec[:, 0], opp[:, 0]])
    controls = np.column_stack(
        [
            np.ones(len(local)),
            cohort["controls"][local],
            (raw - norm["recording_opportunity_centers"])
            / norm["recording_opportunity_sd"],
        ]
    )
    out = {}
    for origin in spec["geography"]["origins_m"]:
        frozen = spec["geography"]["cells"][str(origin)]
        cells, inv = np.unique(
            np.floor((cohort["xy"][local] - origin) / 200000).astype(int),
            axis=0,
            return_inverse=True,
        )
        centers = {
            tuple(k): v
            for k, v in zip(frozen["coordinates"], frozen["exposure_centers"])
        }
        if additional_centers:
            centers = {**additional_centers[str(origin)], **centers}
        cc = np.array([centers[tuple(k)] for k in cells])
        regional = helmert(len(cells), full=False).T[inv]
        X = np.column_stack([controls, regional, E - cc[inv]])
        original_scales = frozen["prior_scales"]
        assert len(set(original_scales[14:-4])) == 1
        scales = (
            original_scales[:14]
            + [original_scales[14]] * (len(cells) - 1)
            + original_scales[-4:]
        )
        vectors = {}
        rows = [(p, [(i, 1)]) for i, p in enumerate(spec["periods"])]
        rows += [
            (spec["periods"][i] + "_minus_" + spec["periods"][j], [(i, 1), (j, -1)])
            for i, j in combinations(range(4), 2)
        ]
        for label, terms in rows:
            for contrast, w in spec["lineage_contrasts"].items():
                v = np.zeros((7, X.shape[1]))
                for col, sgn in terms:
                    v[:, -4 + col] = sgn * np.array(w)[1:]
                vectors[label + "__" + contrast] = v.ravel()
        out[str(origin)] = dict(
            X=X,
            y=cohort["y"],
            counts=cohort["counts"],
            scales=scales,
            vectors=vectors,
            cells=cells,
            centers=cc,
        )
    return out


def measure_positions(data, positions, current):
    """Measure the original strict panels at the source-screened coordinates."""
    spec = current["legacy"]
    db = sqlite3.connect(
        (data / "archaeology.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    db.row_factory = sqlite3.Row
    geometry = [
        dict(r)
        for r in db.execute(
            "select * from positions where site_id in (select site_id from period_recording_sites) order by position_id"
        )
    ]
    sites = {
        r["site_id"]: dict(r)
        for r in db.execute(
            "select * from sites where site_id in (select site_id from period_recording_sites) order by site_id"
        )
    }
    sets = [
        {
            r[0]
            for r in db.execute(
                "select site_id from comparison_sites where period_id=? and feature_id=?",
                (period, feature),
            )
        }
        for period in spec["periods"]
        for feature in ["posthole", "hearth"]
    ]
    allowed = set(spec["archaeological_position_stream_ids"])
    streams = current["common"]["position_streams"]
    groups = collections.defaultdict(list)
    for g in geometry:
        if g["position_id"] not in allowed:
            raise ValueError("Unfrozen archaeological position identity")
        groups[tuple(streams[g["position_id"]])].append(g)
    draws = {}
    for (latitude, longitude, radius, seed), rows in groups.items():
        if (
            max(sites[r["site_id"]]["source_uncertainty_m"] or 0 for r in rows)
            != radius
        ):
            raise ValueError("Original archaeological position radius differs")
        values = position_draws(latitude, longitude, radius, seed, 64)
        for row in rows:
            if (
                abs(row["latitude"] - latitude) >= 1e-9
                or abs(row["longitude"] - longitude) >= 1e-9
            ):
                raise ValueError("Original source coordinate differs")
            draws[row["latitude"], row["longitude"]] = values
    used = set.union(*sets)
    geometry = [r for r in geometry if r["site_id"] in used]
    coordinates = np.array(
        [draws[g["latitude"], g["longitude"]] for g in geometry]
    ).transpose(1, 0, 2)
    masks = [np.array([g["site_id"] in sites for g in geometry]) for sites in sets]
    recording = []
    for period in spec["periods"]:
        ids = [
            r[0]
            for r in db.execute(
                "select site_id from comparison_recording_sites where period_id=? order by site_id",
                (period,),
            )
        ]
        groups = {sites[k]["position_group"]: sites[k] for k in ids}
        recording.append(
            np.array([[v["latitude"], v["longitude"]] for v in groups.values()])
        )
    db.close()
    nominal = np.array([[p["lat"], p["lon"]] for p in positions])
    proximity, counts = np.zeros((len(positions), 8)), np.zeros((len(positions), 4))
    for draw in range(64):
        distances = distance(nominal, coordinates[draw])
        for j, mask in enumerate(masks):
            proximity[:, j] += np.exp(-distances[:, mask].min(1) / 50) / 64
        for j, coords in enumerate(recording):
            counts[:, j] += np.log1p((distance(nominal, coords) <= 75).sum(1)) / 64
    return dict(proximity=proximity, recording=counts)


def coarsen_records(original, tree, release, release_hash, source_name, current, cap):
    """Apply a category-preserving upward cut before the existing grouping rules."""
    nodes = {str(k): v for k, v in tree["allNodes"].items()}
    for node in tree["roots"]:
        nodes.setdefault(str(node["haplogroupId"]), node)

    @functools.lru_cache(None)
    def path(key):
        result = [key]
        while str(nodes[key].get("parentId")) in nodes:
            key = str(nodes[key]["parentId"])
            if key in result:
                raise ValueError("Cycle in source tree")
            result.append(key)
        return result

    category = lambda p: lineage_category(p, current["ancestral_nodes"])
    screened = [r for r in original["records"] if r["reconciled_screen_candidate"]]
    represented = {r["branch_node_id"] for r in screened}
    nested = set()
    for key in represented:
        nested.update(set(path(key)[1:]) & represented)
    support = collections.Counter(n for key in represented - nested for n in path(key))
    mapping = {
        key: next(
            (n for n in reversed(path(key))
             if support[n] <= cap and category(path(n)) == category(path(key))),
            key,
        )
        for key in represented
    }
    genetic = copy.deepcopy(original)
    for row in genetic["records"]:
        if row["reconciled_screen_candidate"]:
            row["original_branch_node_id"] = row["branch_node_id"]
            row["branch_node_id"] = mapping[row["branch_node_id"]]
    construct_units(genetic, tree, release, release_hash, source_name)
    return genetic, mapping


def build_designs(data, current, baseline_ids):
    """Reconstruct both prespecified cuts without selecting among their effects."""
    archive = data / "sources/paternal.zip"
    archaeological_archive = data / "sources/archaeology.zip"
    original = read(data / "paternal.json")
    tree = json.loads(source_member(archive, "references/ftdna_tree.json"))
    release_bytes = source_member(archive, "decisions/broad_release_specification.json")
    release = json.loads(release_bytes)
    if (
        hashlib.sha256(source_member(archive, "observations.zip")).hexdigest()
        != release["raw_sha256"]
    ):
        raise ValueError("Paternal source differs from the grouping specification")
    spec = current["legacy"]
    screened = [r for r in original["records"] if r["reconciled_screen_candidate"]]
    allcoords = sorted({(r["source_lat"], r["source_lon"]) for r in screened})
    positions = [
        dict(position_id="REF-" + str(i), lat=p[0], lon=p[1])
        for i, p in enumerate(allcoords)
    ]
    measured = measure_positions(data, positions, current)
    coordinate_index = {p: i for i, p in enumerate(allcoords)}
    ri = np.array(
        [coordinate_index[r["source_lat"], r["source_lon"]] for r in screened]
    )
    transform = Transformer.from_crs(4326, 3006, always_xy=True)
    xy = np.array([transform.transform(lon, lat) for lat, lon in allcoords])
    norm = spec["normalization"]
    exposures = (
        measured["proximity"][:, ::2]
        - measured["proximity"][:, 1::2]
        - norm["exposure_centers"]
    ) / norm["common_exposure_sd"]
    newcenters, reference = {}, {}
    for origin in spec["geography"]["origins_m"]:
        cells, index = np.unique(
            np.floor((xy[ri] - origin) / 200000).astype(int),
            axis=0,
            return_inverse=True,
        )
        centers = np.array(
            [exposures[ri][index == j].mean(0) for j in range(len(cells))]
        )
        newcenters[str(origin)] = {tuple(k): v for k, v in zip(cells, centers)}
        reference[str(origin)] = dict(
            cells=cells,
            centers=centers,
            weight="one per source-screened record locality; used only for newly represented cells",
        )
    baseline_records = {
        record
        for u in original["analysis_units"]
        if u["unit_id"] in baseline_ids
        for record in u["record_ids"]
    }
    jobs, reports = {}, {}
    for cap in spec["branch_resolution"]["coarsening"]:
        genetic, mapping = coarsen_records(
            original, tree, release, hashlib.sha256(release_bytes).hexdigest(),
            archive.name, current, cap,
        )
        cohort = flexible_cohort(genetic, archaeological_archive, current)
        indices = np.array(
            [coordinate_index[p["lat"], p["lon"]] for p in cohort["positions"]]
        )
        measurement = {
            key: measured[key][indices] for key in ["proximity", "recording"]
        }
        fitted = flexible_designs(cohort, measurement, spec, newcenters)
        retained = {record for u in cohort["units"] for record in u["record_ids"]}
        locations = {
            p["position_id"]: [p["lat"], p["lon"]] for p in cohort["positions"]
        }
        spans = [
            float(distance(coords, coords).max())
            for u in cohort["units"]
            for coords in [
                np.array([locations[l["position_id"]] for l in u["localities"]])
            ]
        ]
        reports[str(cap)] = dict(
            cap=cap,
            screened=len(screened),
            retained_records=len(retained),
            recovered_vs_baseline=len(retained - baseline_records),
            lost_vs_baseline=len(baseline_records - retained),
            units=len(cohort["units"]),
            positions=len(cohort["positions"]),
            localities=len(cohort["local"]),
            blocks=len(set(cohort["blocks"])),
            category_counts=dict(
                collections.Counter(current["categories"][i] for i in cohort["y"])
            ),
            excluded=cohort["excluded"],
            counts=genetic["counts"],
            spans_km=spans,
            mapping=mapping,
            retained_ids=sorted(retained),
            withheld_ids=sorted({r["record_id"] for r in screened} - retained),
            analysis_units=cohort["units"],
        )
        for origin, design in fitted.items():
            name = "branch_" + str(cap) + "_" + origin
            jobs[name] = dict(
                job_id=name,
                scope="Upward branch-cut sensitivity on the original four-window measurement",
                seed=spec["branch_resolution"]["seed"] + cap + int(origin),
                x=design["X"],
                y=design["y"],
                counts=design["counts"],
                starts=cohort["starts"],
                scales=design["scales"],
                categories=current["categories"],
                contrasts=design["vectors"],
                cells=design["cells"],
                centers=design["centers"],
            )
        print(
            "Branch cut",
            cap,
            "units",
            len(cohort["units"]),
            "retained source records",
            len(retained),
            flush=True,
        )
    return jobs, reports, reference


def export(data, measurements, specification, output):
    from analysis.infer import fit_design

    if (output / "design_freeze.json").exists():
        raise FileExistsError("Frozen branch export exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(measurements / "common_cohort.npz", allow_pickle=False) as c:
        baseline_ids = set(c["unit_ids"])
    jobs, reports, reference = build_designs(data, read(specification), baseline_ids)
    modes, checks = {}, {}
    for name, d in jobs.items():
        model, fit, estimates = fit_design(d)
        rng = np.random.default_rng(d["seed"])
        density = []
        for v in [
            fit["mode"],
            fit["mode"] + rng.normal(0, 0.02, len(fit["mode"])),
            fit["mode"] + rng.normal(0, 0.1, len(fit["mode"])),
        ]:
            value, gradient = model.evaluate(v)
            density.append(dict(v=v, log_density=-value, gradient=-gradient))
        d["density_checks"] = density
        modes[name] = estimates
        checks[name] = {k: v for k, v in fit.items() if k not in ["mode", "covariance"]}
    for name, value in [
        ("designs.json", dict(jobs=jobs)),
        ("selection.json", dict(cuts=reports, reference=reference)),
        ("modes.json", modes),
        ("numerical_checks.json", checks),
    ]:
        (output / name).write_text(encode(value) + "\n", encoding="utf-8")
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
    print("Exported four fixed branch-cut posterior jobs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument(
        "--specification", type=Path, default=Path("analysis/specification.json")
    )
    parser.add_argument("--output", type=Path, default=Path("build/branches"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        export(args.data, args.measurements, args.specification, args.output)


if __name__ == "__main__":
    main()

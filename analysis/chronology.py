"""Original four-window chronology comparisons and recording-stress inputs.

The historical posthole/hearth comparison uses its original windows, source
qualification and nested date/position averaging. It is distinct from the
current seven-family, common-basis comparison.
"""

from pathlib import Path
import argparse
import collections
import functools
import gzip
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from itertools import combinations
import numpy as np
from scipy.linalg import helmert
from analysis.landscapes import distance, position_draws


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_member(data, name):
    with zipfile.ZipFile(data / "sources" / "archaeology.zip") as source:
        manifest = json.loads(source.read("manifest.json"))
        entry = next(r for r in manifest["members"] if r["path"] == name)
        value = source.read(name)
    if hashlib.sha256(value).hexdigest() != entry["sha256"]:
        raise ValueError("Source checksum differs: " + name)
    return value


def read_database(path):
    result = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    result.row_factory = sqlite3.Row
    return result


def designs(cohort, measurement, specification):

    from itertools import combinations

    local = cohort["local"]
    counts = cohort["counts"]
    starts = cohort["starts"]
    norm = specification["normalization"]

    prox = measurement["proximity"][local]
    recording = measurement["recording"][local]

    E = (prox[:, ::2] - prox[:, 1::2] - norm["exposure_centers"]) / norm[
        "common_exposure_sd"
    ]

    opp = prox[:, ::2] + prox[:, 1::2]

    raw = np.column_stack([recording[:, 1:], opp[:, 1:], recording[:, 0], opp[:, 0]])

    controls = np.column_stack(
        [
            np.ones(len(local)),
            cohort["controls"][local],
            (raw - norm["recording_opportunity_centers"])
            / norm["recording_opportunity_sd"],
        ]
    )

    assert controls.shape[1] == 14

    output = {}
    periods = specification["periods"]

    for origin in specification["geography"]["origins_m"]:

        cells, inverse = np.unique(
            np.floor((cohort["xy"][local] - origin) / 200000).astype(int),
            axis=0,
            return_inverse=True,
        )

        frozen = specification["geography"]["cells"][str(origin)]

        assert np.array_equal(cells, np.array(frozen["coordinates"]))

        regional = helmert(len(cells), full=False).T[inverse]

        centered = E - np.array(frozen["exposure_centers"])[inverse]

        X = np.column_stack([controls, regional, centered])
        vectors = {}

        rows = [(p, [(i, 1)]) for i, p in enumerate(periods)]

        rows += [
            (periods[i] + "_minus_" + periods[j], [(i, 1), (j, -1)])
            for i, j in combinations(range(4), 2)
        ]

        for label, terms in rows:

            for contrast, weights in specification["lineage_contrasts"].items():

                v = np.zeros((len(specification["categories"]) - 1, X.shape[1]))

                for col, sign in terms:
                    v[:, -4 + col] = sign * np.array(weights)[1:]

                vectors[label + "__" + contrast] = v.ravel()

        unit_xy = np.add.reduceat(cohort["xy"][local], starts, axis=0) / counts[:, None]

        centroids = np.floor((unit_xy - origin) / 200000).astype(int)

        output[str(origin)] = dict(
            X=X,
            y=cohort["y"],
            counts=counts,
            scales=frozen["prior_scales"],
            vectors=vectors,
            centroid_labels=np.array([str(x) + "," + str(y) for x, y in centroids]),
        )

    return output


def material_pool(data, specification):
    """All eligible reported contexts, before full-envelope period containment."""
    db = read_database(data / "archaeology.sqlite")
    WINDOWS = np.array(specification["period_bounds"])
    rules = json.loads(source_member(data, "comparison_decisions.json"))
    corrections = {
        r[0]: json.loads(r[1])
        for r in db.execute("select * from comparison_date_corrections")
    }
    adjud = {
        r[0]: json.loads(r[1]) for r in db.execute("select * from source_adjudications")
    }
    site_map = {
        r[0]: r[1]
        for r in db.execute(
            "select distinct original_site_id,corrected_site_id from source_site_corrections"
        )
    }
    curve = np.loadtxt(
        io.BytesIO(source_member(data, "intcal20.14c")), delimiter=",", comments="#"
    )
    curve = curve[np.argsort(curve[:, 0])]
    grid = np.arange(55001)
    mu = np.interp(grid, curve[:, 0], curve[:, 1])
    sd = np.interp(grid, curve[:, 0], curve[:, 2])
    calendar = 1950 - grid

    @functools.lru_cache(None)
    def calibration(age, error):
        variance = sd**2 + error**2
        mask = np.abs(age - mu) <= 8 * np.sqrt(variance)
        lp = -0.5 * (age - mu[mask]) ** 2 / variance[mask] - 0.5 * np.log(
            variance[mask]
        )
        pr = np.exp(lp - lp.max())
        pr /= pr.sum()
        years = calendar[mask]
        order = np.argsort(pr)[::-1]
        selected = order[: np.searchsorted(np.cumsum(pr[order]), 0.954) + 1]
        lo, hi = int(years[selected].min()), int(years[selected].max())
        pm = np.array([pr[(years >= s) & (years < e)].sum() for s, e in WINDOWS])
        return years, pr, lo, hi, pm

    excluded = []
    rows = []
    lab_parameters = collections.defaultdict(set)
    for rr in db.execute(
        "select * from assertions where disposition='eligible_reported_context' order by assertion_id"
    ):
        r = dict(rr)
        aid = r["assertion_id"]
        src = json.loads(r["source_row_json"])
        patch = {
            **rules["older_rules"].get(aid, {}),
            **rules["function_and_quality_rules"]["decisions"].get(aid, {}),
        }
        if (
            patch.get("eligible") is False
            or patch.get("comparison_eligible") is False
            or aid in ["C14:14586", "C14:14587"]
        ):
            excluded.append(
                dict(
                    assertion_id=aid,
                    reason=patch.get(
                        "reason", "unresolved original laboratory-age contradiction"
                    ),
                )
            )
            continue
        age = float(src["c14_age_bp"])
        error = float(src["c14_error"])
        if adjud.get(aid, {}).get("corrected_measurement_bp_error"):
            age, error = map(float, adjud[aid]["corrected_measurement_bp_error"])
        if aid in corrections:
            age = float(corrections[aid]["corrected_age_bp"])
            error = float(corrections[aid]["corrected_error_bp"])
        assert error > 0
        r["site_id"] = site_map.get(r["site_id"], r["site_id"])
        r["lab_id"] = patch.get("lab_id", r["lab_id"])
        lab = re.sub("[^a-z0-9]", "", str(r["lab_id"]).casefold())
        if lab in ["", "none", "null", "na"]:
            lab = "fallback:" + r["evidence_group"] + ":" + str(age) + ":" + str(error)
        key = "lab:" + lab
        lab_parameters[key].add((age, error))
        r.update(age=age, error=error, calibration_key=key)
        rows.append(r)
    conflicts = {k: sorted(v) for k, v in lab_parameters.items() if len(v) > 1}
    keep = []
    cal = {}
    for r in rows:
        if r["calibration_key"] in conflicts:
            excluded.append(
                dict(
                    assertion_id=r["assertion_id"],
                    reason="conflicting parameters for normalized laboratory identifier",
                    calibration_key=r["calibration_key"],
                )
            )
            continue
        years, prob, lo, hi, pm = calibration(r["age"], r["error"])
        if pm.sum() == 0:
            continue
        cal[r["calibration_key"]] = dict(
            years=years,
            probability=prob,
            period_probability=pm,
            envelope=[lo, hi],
            age=r["age"],
            error=r["error"],
        )
        intersects = np.any((lo < WINDOWS[:, 1]) & (hi >= WINDOWS[:, 0]))
        r["latent_period"] = int(np.argmax(pm)) if intersects else -1
        keep.append(r)
    sites = {r["site_id"]: dict(r) for r in db.execute("select * from sites")}
    used = {r["site_id"] for r in keep}
    geometry = [
        dict(r)
        for r in db.execute("select * from positions order by position_id")
        if r["site_id"] in used
    ]
    db.close()
    # Publication-level recording dependence and shared laboratory identifiers form connected groups.
    parent = {r["evidence_group"]: r["evidence_group"] for r in keep}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    labs = collections.defaultdict(list)
    for r in keep:
        labs[r["calibration_key"]].append(r["evidence_group"])
    for group in labs.values():
        for k in group[1:]:
            x, y = sorted([find(group[0]), find(k)])
            parent[y] = x
    for r in keep:
        r["retention_group"] = find(r["evidence_group"])
    groups = sorted({r["retention_group"] for r in keep})
    north = []
    for g in groups:
        ids = {r["site_id"] for r in keep if r["retention_group"] == g}
        north.append(np.mean([sites[s]["northing"] for s in sorted(ids)]))
    north = np.array(north)
    z = (north - north.mean()) / north.std()
    report = dict(
        eligible_before_periods=len(rows),
        eligible_with_probability=len(keep),
        calibrations=len(cal),
        sites=len(used),
        feature_sites=len(
            {r["site_id"] for r in keep if r["feature_id"] in ["posthole", "hearth"]}
        ),
        excluded=excluded,
        conflicts=conflicts,
        retention_groups=len(groups),
        retention_northing_center=float(north.mean()),
        retention_northing_sd=float(north.std()),
        standardization="Unweighted mean and population SD across connected evidence groups; each group location is the mean of its distinct source-site northings.",
        unresolved_material_event_link="Material dates are not event dates; chronology draws do not resolve inbuilt age or residuality.",
    )
    # The complete source disposition is returned alongside the compact pool.
    return report, dict(
        rows=keep,
        calibration=cal,
        sites=sites,
        geometry=geometry,
        retention_groups=groups,
        northing=dict(zip(groups, z)),
    )


def compact_material_pool(pool):
    """Retain source identities and sufficient calibration inputs, without duplicate curve evaluations."""
    out = dict(pool)
    out["calibration"] = {
        k: {n: v for n, v in item.items() if n not in ["years", "probability"]}
        for k, item in pool["calibration"].items()
    }
    fields = [
        "assertion_id",
        "site_id",
        "lab_id",
        "context_id",
        "evidence_group",
        "feature_id",
        "age",
        "error",
        "calibration_key",
        "latent_period",
        "retention_group",
    ]
    out["rows"] = [{k: r[k] for k in fields} for r in pool["rows"]]
    out["storage"] = (
        "Source identifiers, calibration parameters and period probabilities; annual densities are regenerated from archived IntCal20."
    )
    return out


def expand_material_pool(pool, data):
    if all("years" in c and "probability" in c for c in pool["calibration"].values()):
        return pool
    curve = np.loadtxt(
        io.BytesIO(source_member(data, "intcal20.14c")), delimiter=",", comments="#"
    )
    curve = curve[np.argsort(curve[:, 0])]
    grid = np.arange(55001)
    mu = np.interp(grid, curve[:, 0], curve[:, 1])
    sd = np.interp(grid, curve[:, 0], curve[:, 2])
    cache = {}
    for item in pool["calibration"].values():
        age, error = item["age"], item["error"]
        if (age, error) not in cache:
            variance = sd**2 + error**2
            mask = np.abs(age - mu) <= 8 * np.sqrt(variance)
            lp = -0.5 * (age - mu[mask]) ** 2 / variance[mask] - 0.5 * np.log(
                variance[mask]
            )
            p = np.exp(lp - lp.max())
            p /= p.sum()
            cache[age, error] = (1950 - grid[mask], p)
        item["years"], item["probability"] = cache[age, error]
    return pool


def panel_sets(pool, period_for_key, retention=None):
    feature = [set() for _ in range(8)]
    recording = [set() for _ in range(4)]
    for r in pool["rows"]:
        period = period_for_key[r["calibration_key"]]
        if period < 0:
            continue
        if retention is not None and not retention(r, period):
            continue
        recording[period].add(r["site_id"])
        if r["feature_id"] in ["posthole", "hearth"]:
            feature[2 * period + int(r["feature_id"] == "hearth")].add(r["site_id"])
    return dict(feature=feature, recording=recording, sites=pool["sites"])


def panel_measurement(source, spatial):
    n = spatial["values"].shape[1]
    prox = []
    for ss in source["feature"]:
        indices = [spatial["index"][s] for s in sorted(ss)]
        # Index one draw at a time to avoid a second large 64-draw array.
        value = np.zeros(n)
        if indices:
            for draw in range(64):
                value += spatial["values"][draw][:, indices].max(1) / 64
        prox.append(value)
    rec = []
    for ss in source["recording"]:
        groups = {source["sites"][s]["position_group"]: s for s in sorted(ss)}
        indices = [spatial["recording_index"][s] for s in groups.values()]
        rec.append(np.log1p(spatial["recorded"][:, indices].sum(1)))
    union = []
    for f in range(2):
        ids = set.union(*source["feature"][f::2])
        indices = [spatial["index"][s] for s in sorted(ids)]
        v = np.zeros(n)
        if indices:
            for draw in range(64):
                v += spatial["values"][draw][:, indices].max(1) / 64
        union.append(v)
    return dict(
        proximity=np.column_stack(prox),
        recording=np.column_stack(rec),
        pooled=np.column_stack(union),
        support=[len(s) for s in source["feature"]],
    )


def strict_panels(data, specification):
    """Read the original full-envelope feature and recording panels."""
    db = read_database(data / "archaeology.sqlite")
    try:
        sites = {r["site_id"]: dict(r) for r in db.execute("select * from sites")}
        feature = [
            set(
                r[0]
                for r in db.execute(
                    "select site_id from comparison_sites where period_id=? and feature_id=?",
                    (p, f),
                )
            )
            for p in specification["periods"]
            for f in ["posthole", "hearth"]
        ]
        recording = [
            set(
                r[0]
                for r in db.execute(
                    "select site_id from comparison_recording_sites where period_id=?",
                    (p,),
                )
            )
            for p in specification["periods"]
        ]
    finally:
        db.close()
    return dict(sites=sites, feature=feature, recording=recording)


def spatial_metadata(pool, specification, current, positions, strict):
    """Recover inherited and new coordinate streams without a stored distance cube."""
    allowed = set(specification["archaeological_position_stream_ids"])
    old = {
        k: v for k, v in current["common"]["position_streams"].items() if k in allowed
    }
    if set(old) != allowed:
        raise ValueError("An inherited archaeological position stream is missing")
    groups = collections.defaultdict(list)
    for g in pool["geometry"]:
        groups[g["latitude"], g["longitude"]].append(g)
    draws, new = {}, []
    for coord, rows in groups.items():
        archived = [old[r["position_id"]] for r in rows if r["position_id"] in old]
        if archived:
            chosen = archived[0]
            if not all(g == chosen for g in archived):
                raise ValueError("Inconsistent streams at a shared coordinate")
            latitude, longitude, radius, seed = chosen
        else:
            latitude, longitude = coord
            radius = max(
                pool["sites"][r["site_id"]]["source_uncertainty_m"] or 0 for r in rows
            )
            seed = "SCLARA-MATERIAL-1:" + ",".join(
                sorted(r["position_id"] for r in rows)
            )
            new.append(
                dict(
                    position_ids=sorted(r["position_id"] for r in rows),
                    latitude=latitude,
                    longitude=longitude,
                    radius_m=radius,
                    seed=seed,
                )
            )
        values = position_draws(latitude, longitude, radius, seed, 64)
        for r in rows:
            draws[r["position_id"]] = values
    ids = sorted(
        {
            r["site_id"]
            for r in pool["rows"]
            if r["feature_id"] in ["posthole", "hearth"]
        }
    )
    recording = sorted(
        {r["site_id"] for r in pool["rows"]} | set.union(*strict["recording"])
    )
    return draws, dict(
        new_groups=new,
        feature_site_ids=ids,
        recording_site_ids=recording,
        shape=[64, len(positions), len(ids)],
    )


def make_spatial(pool, specification, current, positions, strict, output):
    """Regenerate the 64-draw site-proximity cache and its source identities."""
    draws, metadata = spatial_metadata(pool, specification, current, positions, strict)
    ids = metadata["feature_site_ids"]
    index = {site: i for i, site in enumerate(ids)}
    geometry = [g for g in pool["geometry"] if g["site_id"] in index]
    draw_array = np.array([draws[g["position_id"]] for g in geometry]).transpose(
        1, 0, 2
    )
    nominal = np.array([[p["lat"], p["lon"]] for p in positions])
    geometry_index = np.array([index[g["site_id"]] for g in geometry])
    output.mkdir(parents=True, exist_ok=True)
    target = output / "material_proximity.npy"
    values = np.lib.format.open_memmap(
        target, mode="w+", dtype="float64", shape=tuple(metadata["shape"])
    )
    for draw in range(64):
        proximity = np.exp(-distance(nominal, draw_array[draw]) / 50)
        for j in range(len(ids)):
            values[draw, :, j] = proximity[:, geometry_index == j].max(1)
        if draw % 16 == 0:
            print("Material spatial draws", draw + 1, "of 64", flush=True)
    values.flush()
    del values
    with target.open("rb") as source:
        metadata["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    (output / "spatial.json").write_text(
        json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return load_spatial(pool, positions, output), metadata


def load_spatial(pool, positions, output):
    metadata = read(output / "spatial.json")
    ids = metadata["feature_site_ids"]
    recording = metadata["recording_site_ids"]
    nominal = np.array([[p["lat"], p["lon"]] for p in positions])
    sites = pool["sites"]
    values = np.load(
        output / "material_proximity.npy", mmap_mode="r", allow_pickle=False
    )
    if list(values.shape) != metadata["shape"]:
        raise ValueError("Spatial cache dimensions differ from the manifest")
    return dict(
        values=values,
        ids=ids,
        index={s: i for i, s in enumerate(ids)},
        recorded=distance(
            nominal,
            np.array(
                [[sites[s]["latitude"], sites[s]["longitude"]] for s in recording]
            ),
        )
        <= 75,
        recording_ids=recording,
        recording_index={s: i for i, s in enumerate(recording)},
    )


def paired_memberships(data, specification, pool):
    """Keep the original source universe separate from expanded date allocation.

    A reviewed hearth-pit excluded from the focal hearth class is restored as
    recording opportunity only. Its independent date stream preserves all
    original calibration draws.
    """
    db = read_database(data / "archaeology.sqlite")
    windows = [
        (p, *bounds)
        for p, bounds in zip(specification["periods"], specification["period_bounds"])
    ]
    rules = json.loads(source_member(data, "comparison_decisions.json"))
    original = {}
    corrected = {
        r[0]: json.loads(r[1])
        for r in db.execute("select * from comparison_date_corrections")
    }
    site_map = {
        r[0]: r[1]
        for r in db.execute(
            "select distinct original_site_id,corrected_site_id from source_site_corrections"
        )
    }
    focal = {
        r[0]
        for r in db.execute(
            "select assertion_id from comparison_assertions where eligible=1"
        )
    }
    for period, start, end in windows:
        kept = []
        for row in db.execute(
            "select * from assertions where disposition='eligible_reported_context' and start_year>=? and end_year<?",
            (start, end),
        ):
            item = dict(row)
            aid = item["assertion_id"]
            if (
                period == "Pre2800"
                and rules["older_rules"].get(aid, {}).get("eligible") is False
            ):
                continue
            if period == "Early_Bronze" and aid in ["C14:14586", "C14:14587"]:
                continue
            if aid in corrected:
                lo, hi = corrected[aid]["calendar_envelope"]
                if not lo >= start or not hi < end:
                    continue
            item["site_id"] = site_map.get(item["site_id"], item["site_id"])
            original[aid] = item
            kept.append(item)
        observed = {
            row[0]
            for row in db.execute(
                "select site_id from comparison_recording_sites where period_id=?",
                (period,),
            )
        }
        if {row["site_id"] for row in kept} != observed:
            raise ValueError("Original recording universe differs: " + period)
    db.close()
    missing = set(original) - {r["assertion_id"] for r in pool["rows"]}
    if missing != {"C14:22950"}:
        raise ValueError("The recording-only restoration no longer matches its source")
    row = original["C14:22950"]
    raw = json.loads(row["source_row_json"])
    extra = dict(
        assertion_id=row["assertion_id"],
        site_id=row["site_id"],
        lab_id=row["lab_id"],
        feature_id=None,
        calibration_key="lab:" + re.sub("[^a-z0-9]", "", row["lab_id"].lower()),
        age=float(raw["c14_age_bp"]),
        error=float(raw["c14_error"]),
        scope="recording only; source-reviewed hearth-pit, excluded from generic hearth exposure",
    )
    if extra["calibration_key"] in pool["calibration"]:
        raise ValueError(
            "The recording-only assay unexpectedly duplicates an existing key"
        )
    parameters = dict(pool["calibration"])
    parameters[extra["calibration_key"]] = dict(age=extra["age"], error=extra["error"])
    cal = expand_material_pool(dict(calibration=parameters), data)["calibration"]
    cdfs = {
        key: (item["years"], np.cumsum(item["probability"]))
        for key, item in cal.items()
    }
    strict = strict_panels(data, specification)
    sets = {}
    for i, period in enumerate(specification["periods"]):
        sets["frozen_containment", -1, period, "recording"] = strict["recording"][i]
        for j, feature in enumerate(["posthole", "hearth"]):
            sets["frozen_containment", -1, period, feature] = strict["feature"][
                2 * i + j
            ]
    legacy_keys = sorted(pool["calibration"])
    expanded = pool["rows"] + [extra]
    kept = [row for row in expanded if row["assertion_id"] in original]
    summaries = collections.defaultdict(list)
    for replicate in range(64):
        rng = np.random.default_rng(np.random.SeedSequence([2026110620, replicate]))
        assignment = {}
        for key in legacy_keys:
            years, cdf = cdfs[key]
            assignment[key] = int(years[np.searchsorted(cdf, rng.random())])
        extra_rng = np.random.default_rng(
            np.random.SeedSequence([2026110620, replicate, 22950])
        )
        years, cdf = cdfs[extra["calibration_key"]]
        assignment[extra["calibration_key"]] = int(
            years[np.searchsorted(cdf, extra_rng.random())]
        )
        for scenario, rows in [
            ("fixed_original_pool", kept),
            ("expanded_pool", expanded),
            ("package_A_pool", pool["rows"]),
        ]:
            selected = {
                (period, feature): set()
                for period, _, _ in windows
                for feature in ["posthole", "hearth", "recording"]
            }
            for row in rows:
                year = assignment[row["calibration_key"]]
                for period, start, end in windows:
                    if not start <= year < end:
                        continue
                    selected[period, "recording"].add(row["site_id"])
                    if row["feature_id"] in ["posthole", "hearth"] and (
                        scenario != "fixed_original_pool"
                        or row["assertion_id"] in focal
                    ):
                        selected[period, row["feature_id"]].add(row["site_id"])
            for (period, feature), sites in selected.items():
                sets[scenario, replicate, period, feature] = sites
                summaries[scenario, period, feature].append(len(sites))
    report = dict(
        original_assertions=len(original),
        expanded_assertions=len(expanded),
        recording_only_restore=extra,
        seed=2026110620,
        replicates=64,
        summaries=[
            dict(
                scenario=s,
                period=p,
                feature=f,
                minimum=min(v),
                median=float(np.median(v)),
                maximum=max(v),
            )
            for (s, p, f), v in summaries.items()
        ],
    )
    return sets, report


def paired_source(memberships, scenario, replicate, periods, sites):
    return dict(
        feature=[
            memberships[scenario, replicate, p, f]
            for p in periods
            for f in ["posthole", "hearth"]
        ],
        recording=[memberships[scenario, replicate, p, "recording"] for p in periods],
        sites=sites,
    )


def paired_spatial(pool, positions, spatial, memberships):
    """Add the restored recording-only site's nominal proximity when required."""
    recording = sorted(
        set(spatial["recording_ids"])
        | set.union(*(v for k, v in memberships.items() if k[3] == "recording"))
    )
    nominal = np.array([[p["lat"], p["lon"]] for p in positions])
    sites = pool["sites"]
    return dict(
        spatial,
        recording_ids=recording,
        recording_index={s: i for i, s in enumerate(recording)},
        recorded=distance(
            nominal,
            np.array(
                [[sites[s]["latitude"], sites[s]["longitude"]] for s in recording]
            ),
        )
        <= 75,
    )


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


def read_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def prepare(data, measurements, specification_path, inputs, output):
    """Rebuild source memberships and validate the frozen synthetic design."""
    from scipy.special import ndtri
    from scipy.stats import qmc
    from threadpoolctl import threadpool_limits
    from analysis.model import LabelMixture

    threadpool_limits(limits=1)
    if (output / "preparation.json").exists():
        raise FileExistsError(
            "A prepared chronology already exists; use a fresh output directory"
        )
    current = read(specification_path)
    specification = current["legacy"]
    cohort = read_arrays(measurements / "common_cohort.npz")
    positions = read(data / "paternal.json")["spatial_positions"]
    report, full_pool = material_pool(data, specification)
    pool = compact_material_pool(full_pool)
    strict = strict_panels(data, specification)
    spatial, metadata = make_spatial(
        pool, specification, current, positions, strict, output
    )
    measured = panel_measurement(strict, spatial)
    design = designs(cohort, measured, specification)["0"]
    fixture = inputs / "chronology_design.npz"
    if (
        hashlib.sha256(fixture.read_bytes()).hexdigest()
        != specification["fixtures"][fixture.name]["sha256"]
    ):
        raise ValueError("The chronological benchmark fixture changed")
    expected = read_arrays(fixture)
    checks = {}

    def verify(key, value, tolerance=1e-12):
        value = np.asarray(value)
        if value.shape != expected[key].shape:
            raise ValueError("Fixture dimensions differ: " + key)
        error = float(np.max(np.abs(value - expected[key])))
        if error > tolerance:
            raise ValueError(
                "Source-derived fixture differs: " + key + " (" + str(error) + ")"
            )
        checks[key] = dict(maximum_error=error, tolerance=tolerance)

    for key in ["X", "counts", "y", "scales"]:
        verify(key, design[key])
    verify("local", cohort["local"])
    weights = np.repeat(1 / cohort["counts"], cohort["counts"])
    local = cohort["local"]
    raw = (measured["pooled"][:, 0] - measured["pooled"][:, 1]) / specification[
        "normalization"
    ]["common_exposure_sd"]
    global_center = np.average(raw[local], weights=weights)
    cells, index = np.unique(
        np.floor(cohort["xy"][local] / 200000).astype(int), axis=0, return_inverse=True
    )
    regional_center = np.array(
        [
            np.average(
                raw[local][index == j] - global_center, weights=weights[index == j]
            )
            for j in range(len(cells))
        ]
    )
    verify("cell_inverse", index)
    verify("pooled_global_center", global_center)
    verify("pooled_regional_center", regional_center)
    verify("pooled", raw[local] - global_center - regional_center[index])
    verify(
        "V",
        np.array(
            [
                design["vectors"][period + "__I1_vs_R1a_Z284"]
                for period in specification["periods"]
            ]
        ),
    )
    direction = np.array([0.5, 0.5, 0.5, -0.5, 0, 0, 0, 0])
    verify("direction", direction[1:] - direction[0])
    transform = np.vstack(
        [
            np.eye(4),
            np.array(
                [np.eye(4)[i] - np.eye(4)[j] for i, j in combinations(range(4), 2)]
            ),
        ]
    )
    verify("T", transform)
    gauss = ndtri(
        qmc.Sobol(
            4,
            scramble=True,
            seed=specification["chronology"]["gaussian_integration_seed"],
        ).random_base2(14)
    )
    verify("gauss", gauss)
    latent = {r["calibration_key"]: r["latent_period"] for r in pool["rows"]}
    latent_measurement = panel_measurement(panel_sets(pool, latent), spatial)
    verify("latent_X", designs(cohort, latent_measurement, specification)["0"]["X"])
    fit = LabelMixture(
        design["X"][:, :-4], design["y"], design["counts"], design["scales"][:-4], 8
    ).optimize()
    verify("nuisance_mode", fit["mode"], 1e-5)
    preparation = dict(
        support=measured["support"],
        latent_support=latent_measurement["support"],
        latent=latent,
        shape=list(design["X"].shape),
        fixture_checks=checks,
        nuisance_diagnostics={
            k: v for k, v in fit.items() if k not in ["mode", "covariance"]
        },
        benchmark_parameters="Use the immutable generating fixture; its source-derived nuisance fit is checked within 1e-5.",
    )
    (output / "material_pool.json.gz").write_bytes(
        gzip.compress(encode(pool).encode(), mtime=0)
    )
    (output / "material_report.json").write_text(
        encode(report) + "\n", encoding="utf-8"
    )
    (output / "preparation.json").write_text(
        encode(preparation) + "\n", encoding="utf-8"
    )
    print(
        "Prepared original chronology and recording stress",
        len(pool["rows"]),
        "assertions",
        flush=True,
    )


def paired_summary(output, specification):
    """Keep every contrast, including paired changes and measurement diagnostics."""
    import numpy as np

    z = 1.959963984540054
    with sqlite3.connect(output / "paired.sqlite") as c:
        rows = c.execute(
            "SELECT scenario,replicate,origin,status,details,arrays FROM paired_fit"
        ).fetchall()
        assert len(rows) == 258, (len(rows), "incomplete paired batch")
        fits = {
            (s, r, o): (st, json.loads(d), dict(np.load(io.BytesIO(b))))
            for s, r, o, st, d, b in rows
        }
        c.execute(
            "CREATE TABLE IF NOT EXISTS paired_summary(scenario TEXT,origin TEXT,contrast TEXT,details TEXT,PRIMARY KEY(scenario,origin,contrast))"
        )
        displayed = []
        for origin in ["0", "100000"]:
            base = fits["frozen_containment", -1, origin]
            assert base[0] == "passed"
            names = base[1]["contrast_names"]
            reference = base[2]["contrast_mean"]
            for scenario in ["fixed_original_pool", "expanded_pool"]:
                rr = [(r, fits[scenario, r, origin]) for r in range(64)]
                passed = [(r, v) for r, v in rr if v[0] == "passed"]
                for i, name in enumerate(names):
                    means = np.array([v[2]["contrast_mean"][i] for _, v in passed])
                    se = np.array(
                        [np.sqrt(v[2]["contrast_covariance"][i, i]) for _, v in passed]
                    )
                    pair = np.array(
                        [
                            v[2]["contrast_mean"][i]
                            - fits["fixed_original_pool", r, origin][2][
                                "contrast_mean"
                            ][i]
                            for r, v in passed
                            if fits["fixed_original_pool", r, origin][0] == "passed"
                        ]
                    )
                    summary = dict(
                        n=len(passed),
                        failed=[r for r, v in rr if v[0] != "passed"],
                        strict_mode=float(reference[i]),
                        strict_conditional_interval=(
                            reference[i]
                            + np.array([-1, 1])
                            * z
                            * np.sqrt(base[2]["contrast_covariance"][i, i])
                        ).tolist(),
                        mode_quantiles=np.quantile(
                            means, [0, 0.025, 0.5, 0.975, 1]
                        ).tolist(),
                        positive_modes=int(sum(means > 0)),
                        positive_lower_intervals=int(sum(means - z * se > 0)),
                        negative_upper_intervals=int(sum(means + z * se < 0)),
                        mean_conditional_se=float(se.mean()),
                        delta_from_strict_quantiles=np.quantile(
                            means - reference[i], [0, 0.025, 0.5, 0.975, 1]
                        ).tolist(),
                        delta_from_fixed_quantiles=np.quantile(
                            pair, [0, 0.025, 0.5, 0.975, 1]
                        ).tolist(),
                        scope="Conditional estimates across paired archaeological allocations. Empirical quantiles and paired differences are not posterior intervals for a pooled or difference effect; no independence between allocations is assumed.",
                    )
                    c.execute(
                        "INSERT OR REPLACE INTO paired_summary VALUES(?,?,?,?)",
                        (scenario, origin, name, encode(summary)),
                    )
                    if name.split("__")[0] in specification[
                        "periods"
                    ] and name.endswith("__I1_vs_R1a_Z284"):
                        displayed.append(
                            dict(
                                scenario=scenario,
                                origin=origin,
                                contrast=name,
                                strict=round(reference[i], 6),
                                median=round(float(np.median(means)), 6),
                                range=[
                                    round(float(means.min()), 6),
                                    round(float(means.max()), 6),
                                ],
                                positive_lower=int(sum(means - z * se > 0)),
                            )
                        )
        source = {}
        measurements = c.execute(
            "SELECT scenario,replicate,details FROM paired_measurement"
        ).fetchall()
        for sc in ["frozen_containment", "fixed_original_pool", "expanded_pool"]:
            rr = [json.loads(d) for s, _, d in measurements if s == sc]
            source[sc] = dict(
                n=len(rr),
                support_quantiles=np.quantile(
                    [r["support"] for r in rr], [0, 0.5, 1], axis=0
                ),
                recording_quantiles=np.quantile(
                    [r["recording_support"] for r in rr], [0, 0.5, 1], axis=0
                ),
                proximity_mean_quantiles=np.quantile(
                    [r["proximity_mean"] for r in rr], [0, 0.5, 1], axis=0
                ),
                residual_sd_quantiles={
                    o: np.quantile(
                        [r["diagnostics"][o]["residual_sd"] for r in rr],
                        [0, 0.5, 1],
                        axis=0,
                    )
                    for o in ["0", "100000"]
                },
            )
        counts = dict(collections.Counter(r[3] for r in rows))
        receipt = dict(
            fits=counts,
            measurement_summary=source,
            summary_rows=c.execute("SELECT count(*) FROM paired_summary").fetchone()[0],
            scope="Paired source/chronology measurement appraisal, not selection calibration or independent confirmation.",
        )
        c.execute(
            "insert or replace into metadata values(?,?)",
            ("paired_results_complete", encode(receipt)),
        )
        print("Paired contrast summaries", receipt["summary_rows"])
        print("Fit statuses", counts)


def paired_initialize(paths):
    global _paired_context
    from threadpoolctl import threadpool_limits
    from analysis.model import LabelMixture

    threadpool_limits(limits=1)
    paths = {k: Path(v) for k, v in paths.items()}
    spec = read(paths["specification"])["legacy"]
    pool = json.loads(
        gzip.decompress((paths["chronology"] / "material_pool.json.gz").read_bytes())
    )
    memberships, report = paired_memberships(paths["data"], spec, pool)
    positions = read(paths["data"] / "paternal.json")["spatial_positions"]
    spatial = paired_spatial(
        pool, positions, load_spatial(pool, positions, paths["chronology"]), memberships
    )
    cohort = read_arrays(paths["measurements"] / "common_cohort.npz")
    strict = panel_measurement(
        paired_source(
            memberships, "frozen_containment", -1, spec["periods"], pool["sites"]
        ),
        spatial,
    )
    modes = {}
    for origin, d in designs(cohort, strict, spec).items():
        modes[origin] = LabelMixture(
            d["X"], d["y"], d["counts"], d["scales"], 8
        ).optimize()["mode"]
    _paired_context = dict(
        specification=spec,
        pool=pool,
        memberships=memberships,
        report=report,
        spatial=spatial,
        cohort=cohort,
        modes=modes,
    )


def paired_task(job):
    from analysis.model import LabelMixture

    scenario, replicate = job
    context = _paired_context
    cohort, spec = context["cohort"], context["specification"]
    source = paired_source(
        context["memberships"],
        scenario,
        replicate,
        spec["periods"],
        context["pool"]["sites"],
    )
    measured = panel_measurement(source, context["spatial"])
    fitted_designs = designs(cohort, measured, spec)
    local = cohort["local"]
    weights = np.repeat(1 / cohort["counts"], cohort["counts"])
    diagnostics, results = {}, []
    for origin, design in fitted_designs.items():
        x = design["X"]
        exposures, nuisance = x[:, -4:], x[:, :-4]
        sw = np.sqrt(weights)
        residual = (
            exposures
            - nuisance
            @ np.linalg.lstsq(
                nuisance * sw[:, None], exposures * sw[:, None], rcond=None
            )[0]
        )
        covariance = residual.T @ (weights[:, None] * residual) / weights.sum()
        sd = np.sqrt(np.diag(covariance))
        design_hash = hashlib.sha256(x.tobytes()).hexdigest()
        diagnostics[origin] = dict(
            residual_sd=sd,
            residual_correlation=covariance / np.outer(sd, sd),
            exposure_sd=np.sqrt(
                np.average(
                    (exposures - np.average(exposures, axis=0, weights=weights)) ** 2,
                    axis=0,
                    weights=weights,
                )
            ),
            design_sha256=design_hash,
        )
        details = dict(
            scenario=scenario,
            replicate=replicate,
            origin=origin,
            contrast_names=list(design["vectors"]),
            design_sha256=design_hash,
        )
        fit, errors = None, []
        for start in [context["modes"][origin], None]:
            try:
                fit = LabelMixture(
                    x, design["y"], design["counts"], design["scales"], 8
                ).optimize(start)
                break
            except (ValueError, np.linalg.LinAlgError) as error:
                errors.append(str(error))
        details["numerical_errors"] = errors
        values, status = {}, "failed"
        if fit is not None:
            vectors = np.array(list(design["vectors"].values()))
            values = dict(
                mode=fit["mode"],
                contrast_mean=vectors @ fit["mode"],
                contrast_covariance=vectors @ fit["covariance"] @ vectors.T,
            )
            details.update(
                {k: v for k, v in fit.items() if k not in ["mode", "covariance"]}
            )
            status = "passed"
        results.append((origin, status, details, values))
    summary = dict(
        support=measured["support"],
        recording_support=[len(s) for s in source["recording"]],
        proximity_mean=np.average(
            measured["proximity"][local], axis=0, weights=weights
        ),
        diagnostics=diagnostics,
    )
    return (
        scenario,
        replicate,
        summary,
        {k: v for k, v in measured.items() if k != "support"},
        results,
    )


def paired_run(paths, output, workers=2, limit=None):
    import concurrent.futures
    from analysis.benchmarks import blob

    if not 1 <= workers <= 8:
        raise ValueError("Use between one and eight workers")
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "paired.sqlite", timeout=60)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS paired_fit(scenario TEXT,replicate INTEGER,origin TEXT,status TEXT,details TEXT,arrays BLOB,PRIMARY KEY(scenario,replicate,origin));
        CREATE TABLE IF NOT EXISTS paired_measurement(scenario TEXT,replicate INTEGER,details TEXT,arrays BLOB,PRIMARY KEY(scenario,replicate));
        CREATE TABLE IF NOT EXISTS chronology_membership(scenario TEXT,replicate INTEGER,period TEXT,feature TEXT,site_ids TEXT,PRIMARY KEY(scenario,replicate,period,feature));
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
    """
    )
    inputs = [
        Path(paths["specification"]),
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(paths["measurements"]) / "common_cohort.npz",
        Path(paths["data"]) / "archaeology.sqlite",
        Path(paths["data"]) / "paternal.json",
        Path(paths["data"]) / "sources/archaeology.zip",
    ]
    inputs += [
        Path(paths["chronology"]) / name
        for name in ["material_pool.json.gz", "spatial.json", "material_proximity.npy"]
    ]
    hashes = {}
    for path in inputs:
        with path.open("rb") as source:
            hashes[path.name] = hashlib.file_digest(source, "sha256").hexdigest()
    previous = db.execute("select value from metadata where key='inputs'").fetchone()
    if previous and json.loads(previous[0]) != hashes:
        raise ValueError(
            "Existing paired fits use different inputs or code; use a fresh output directory"
        )
    db.execute("insert or ignore into metadata values('inputs',?)", (encode(hashes),))
    db.commit()
    paired_initialize(paths)
    for (scenario, replicate, period, feature), sites in _paired_context[
        "memberships"
    ].items():
        db.execute(
            "insert or ignore into chronology_membership values(?,?,?,?,?)",
            (scenario, replicate, period, feature, encode(sorted(sites))),
        )
    db.execute(
        "insert or replace into metadata values('source_membership',?)",
        (encode(_paired_context["report"]),),
    )
    db.commit()
    jobs = [("frozen_containment", -1)] + [
        (s, r) for r in range(64) for s in ["fixed_original_pool", "expanded_pool"]
    ]
    pending = []
    for job in jobs:
        existing = dict(
            db.execute(
                "select origin,status from paired_fit where scenario=? and replicate=?",
                job,
            )
        )
        if any(s != "passed" for s in existing.values()):
            raise ValueError(
                "A failed paired fit is retained; inspect its diagnostics before rerunning"
            )
        if set(existing) != {"0", "100000"}:
            pending.append(job)
    scheduled = pending if limit is None else pending[:limit]
    executor = None
    if workers == 1:
        results = map(paired_task, scheduled)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=paired_initialize, initargs=(paths,)
        )
        results = executor.map(paired_task, scheduled, chunksize=1)
    completed = 0
    try:
        for scenario, replicate, summary, measurement, fits in results:
            db.execute(
                "insert or ignore into paired_measurement values(?,?,?,?)",
                (scenario, replicate, encode(summary), blob(measurement)),
            )
            for origin, status, details, values in fits:
                db.execute(
                    "insert or ignore into paired_fit values(?,?,?,?,?,?)",
                    (
                        scenario,
                        replicate,
                        origin,
                        status,
                        encode(details),
                        blob(values),
                    ),
                )
            db.commit()
            completed += 1
            if completed % 10 == 0 or completed == len(scheduled):
                print("Paired allocations", completed, "of", len(scheduled), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        db.close()
    if len(scheduled) == len(pending):
        paired_summary(output, _paired_context["specification"])
    else:
        print(
            "Partial paired run; remaining allocations",
            len(pending) - completed,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "paired"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument(
        "--specification", type=Path, default=Path("analysis/specification.json")
    )
    parser.add_argument("--inputs", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--chronology", type=Path, default=Path("build/chronology"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        help="At most this many pending paired allocations; never mark a partial batch complete",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "prepare":
        prepare(
            args.data,
            args.measurements,
            args.specification,
            args.inputs,
            args.output or args.chronology,
        )
    else:
        paths = {
            key: str(getattr(args, key).resolve())
            for key in ["data", "measurements", "specification", "chronology"]
        }
        paired_run(paths, args.output or Path("build/paired"), args.workers, args.limit)


if __name__ == "__main__":
    main()

"""Construct the frozen Swedish archaeological measurements and model cohorts.

Source decisions precede measurement. Common-period scores use one frozen basis;
the four later profiles retain their distinct feature meanings and transforms.
All paths are explicit. No source downloads or lineage-model fitting occur here.
"""

from pathlib import Path
import argparse
import collections
import copy
import functools
import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
import time
import zipfile
import numpy as np
from PIL import Image
from pyproj import Transformer
from analysis.landscapes import distance, position_draws, measure_landscapes


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_pool(path, value):
    raw = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    Path(path).write_bytes(gzip.compress(raw, mtime=0))


def source_member(archive, name):
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read("manifest.json"))
        record = next(r for r in manifest["members"] if r["path"] == name)
        data = source.read(name)
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Source member checksum differs: " + name)
    return data


def lineage_category(path, nodes):
    path = set(path)
    if nodes["I1"] in path:
        return (
            "I1_L22"
            if nodes["I1_L22"] in path
            else ("I1_Z58" if nodes["I1_Z58"] in path else "I1_other")
        )
    return next(
        (c for c in ["R1a_Z284", "R1b_U106", "R1b_P312", "N"] if nodes[c] in path),
        "Other",
    )


def read_db(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    return db


class Calibration:
    def __init__(self, source):
        with zipfile.ZipFile(source) as z:
            curve = np.loadtxt(
                io.BytesIO(z.read("intcal20.14c")), delimiter=",", comments="#"
            )
        curve = curve[np.argsort(curve[:, 0])]
        self.grid = np.arange(55001)
        self.mu = np.interp(self.grid, curve[:, 0], curve[:, 1])
        self.sd = np.interp(self.grid, curve[:, 0], curve[:, 2])

    @functools.lru_cache(None)
    def distribution(self, age, error):
        variance = self.sd**2 + error**2
        mask = abs(age - self.mu) <= 8 * np.sqrt(variance)
        lp = -0.5 * (age - self.mu[mask]) ** 2 / variance[mask] - 0.5 * np.log(
            variance[mask]
        )
        prob = np.exp(lp - lp.max())
        prob /= prob.sum()
        return 1950 - self.grid[mask], prob

    def probabilities(self, age, error, periods):
        years, prob = self.distribution(age, error)
        return [
            float(prob[(years >= lo) & (years < hi)].sum()) for _, lo, hi in periods
        ]


def registry_geometries(adjudications):
    sites, geometries = {}, []
    tf = Transformer.from_crs(3006, 4326, always_xy=True)
    for source in adjudications["registry_sources"]:
        row = source["response"]["results"][0]
        sid = row["lamningsnummer"]
        assert row["geometri"]["crs"]["properties"]["name"] == "EPSG:3006"
        parts = []
        for feature in row["geometri"]["features"]:
            # Both retrieved sources contain simple single-ring polygons. Use
            # their area centroid, retaining the predecessor's coordinate rule.
            geometry = feature["geometry"]
            assert geometry["type"] == "Polygon" and len(geometry["coordinates"]) == 1
            ring = np.asarray(geometry["coordinates"][0], dtype=float)
            assert ring.shape[1] == 2 and np.array_equal(ring[0], ring[-1])
            shifted = ring - ring[0]
            cross = shifted[:-1, 0] * shifted[1:, 1] - shifted[1:, 0] * shifted[:-1, 1]
            area_twice = cross.sum()
            assert abs(area_twice) > 1e-6
            x, y = ring[0] + ((shifted[:-1] + shifted[1:]) * cross[:, None]).sum(0) / (
                3 * area_twice
            )
            bounds = [
                ring[:, 0].min(),
                ring[:, 1].min(),
                ring[:, 0].max(),
                ring[:, 1].max(),
            ]
            assert bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
            lon, lat = tf.transform(x, y)
            radius = max(
                math.hypot(xx - x, yy - y)
                for xx in [bounds[0], bounds[2]]
                for yy in [bounds[1], bounds[3]]
            )
            parts.append((x, y, radius))
            geometries.append(
                dict(
                    position_id="registry:" + feature["id"],
                    site_id=sid,
                    longitude=lon,
                    latitude=lat,
                    easting=x,
                    northing=y,
                    geometry_radius_m=radius,
                    within_site_weight=1 / len(row["geometri"]["features"]),
                    source_table="registry",
                    source_fid=None,
                )
            )
        x, y = np.mean(np.array(parts)[:, :2], axis=0)
        lon, lat = tf.transform(x, y)
        radius = max(math.hypot(xx - x, yy - y) + r for xx, yy, r in parts)
        division = row["geografisk_indelning"]
        municipality = division["kommun"][0]
        sites[sid] = dict(
            site_id=sid,
            uuid=row["lamning_id"],
            old_id=row["raa_nummer"],
            site_type=row["lamningstyp_namn"],
            name=None,
            county=municipality["lan_namn"],
            municipality=municipality["kommun_namn"],
            parish=division["socken"][0]["socken_namn"],
            status=row["antikvariskbedomningtyp_namn"],
            investigation_status=None,
            longitude=lon,
            latitude=lat,
            easting=float(x),
            northing=float(y),
            geometry_radius_m=radius,
            source_uncertainty_m=None,
            position_group=f"{x:.3f},{y:.3f}",
        )
    return sites, geometries


def material_dating_properties(material, species):
    """Classify verified material age risk without assigning an activity date."""
    material = (material or "").strip().casefold()
    species = (species or "").strip().casefold()
    short = int(
        material in ("förkolnat frö", "frö (ej förkolnat)", "sädeskorn", "hasselnötsskal")
        or material == "växtdelar"
        and any(t in species for t in ("korn", "vete", "havre", "råg", "hasselnöt"))
    )
    risk = (
        "short_lived_terrestrial" if short else
        "possible_inbuilt_age" if material in (
            "träkol", "trä", "bark", "kol i järn", "träkol och kottefjäll"
        ) else "reservoir_or_material_review_needed"
    )
    return short, risk


def source_pool(database, archive, rules, periods, cal, adjudications=None, omitted=()):
    with read_db(database) as c:
        raw = [
            dict(r) for r in c.execute("select * from assertions order by assertion_id")
        ]
        sites = {r["site_id"]: dict(r) for r in c.execute("select * from sites")}
        geometry = [
            dict(r) for r in c.execute("select * from positions order by position_id")
        ]
        updates = {
            r[0]: json.loads(r[1])
            for r in c.execute("select * from comparison_date_corrections")
        }
        prior = {
            r[0]: json.loads(r[1])
            for r in c.execute("select * from source_adjudications")
        }
        site_map = {
            r[0]: r[1]
            for r in c.execute(
                "select distinct original_site_id,corrected_site_id from source_site_corrections"
            )
        }
    with zipfile.ZipFile(archive) as z:
        decisions = json.loads(z.read("comparison_decisions.json"))
    family = {f: g for g, ff in rules["families"].items() for f in ff}
    repairs, eligibility, identities = {}, {}, {}
    if adjudications:
        repairs = {r["assertion_id"]: r for r in adjudications["records"]}
        eligibility = {
            r["assertion_id"]: r for r in adjudications["eligibility_adjudications"]
        }
        identities = {
            aid: r["corrected_site_id"]
            for r in adjudications["site_identity_adjudications"]
            for aid in r["assertion_ids"]
        }
        new_sites, new_geom = registry_geometries(adjudications)
        sites.update(new_sites)
        geometry.extend(new_geom)
    rows, excluded, parameters = [], [], collections.defaultdict(set)
    for r in raw:
        aid = r["assertion_id"]
        source = json.loads(r["source_row_json"])
        action = eligibility.get(aid, {}).get("action")
        if (
            r["disposition"] != "eligible_reported_context"
            and action != "admit_reported_context"
        ):
            continue
        decision = prior.get(aid, {})
        if (
            aid in rules["exclusions"]
            or decision.get("hold")
            or action in ["hold", "retain_hold"]
            or aid in omitted
        ):
            excluded.append(
                dict(
                    assertion_id=aid,
                    reason=rules["exclusions"].get(aid, decision.get("rationale"))
                    or eligibility.get(aid, {}).get("reason")
                    or "Frozen reviewed-phase omission",
                )
            )
            continue
        age, error = float(source["c14_age_bp"]), float(source["c14_error"])
        if decision.get("corrected_measurement_bp_error"):
            age, error = map(float, decision["corrected_measurement_bp_error"])
        if aid in updates:
            age, error = float(updates[aid]["corrected_age_bp"]), float(
                updates[aid]["corrected_error_bp"]
            )
        patch = {
            **decisions["older_rules"].get(aid, {}),
            **decisions["function_and_quality_rules"]["decisions"].get(aid, {}),
        }
        r["lab_id"] = patch.get("lab_id", r["lab_id"])
        r["original_feature_id"] = r["feature_id"]
        r["feature_id"] = (
            rules["feature_updates"].get(aid, {}).get("feature", r["feature_id"])
        )
        r["site_id"] = site_map.get(r["site_id"], r["site_id"])
        if aid in repairs:
            verified = repairs[aid]["verified"]
            for key in [
                "lab_id",
                "site_id",
                "context_id",
                "feature_id",
                "material",
                "species",
            ]:
                if key in verified:
                    r[key] = verified[key]
            age = float(verified.get("c14_age_bp", age))
            error = float(verified.get("c14_error", error))
            if "sample_id" in verified:
                r["sample_id"] = verified["sample_id"]
            r["corrected_source_id"] = repairs[aid]["source_id"]
            if "material" in verified or "species" in verified:
                r["original_dating_risk"] = r["dating_risk"]
                r["original_short_lived"] = r["short_lived"]
                r["short_lived"], r["dating_risk"] = material_dating_properties(
                    r["material"], r["species"]
                )
            r["disposition"] = "eligible_reported_context"
            r["reasons"] = "[]"
        if aid in identities:
            r["site_id"] = identities[aid]
        lab = re.sub("[^a-z0-9]", "", str(r["lab_id"]).casefold())
        key = (
            "lab:" + lab if lab not in ["", "none", "null", "na"] else "fallback:" + aid
        )
        assert error > 0
        r.update(
            age=age,
            error=error,
            calibration_key=key,
            family=family.get(r["feature_id"]),
        )
        parameters[key].add((age, error))
        rows.append(r)
    conflicts = {k: sorted(v) for k, v in parameters.items() if len(v) > 1}
    keep = []
    for r in rows:
        if r["calibration_key"] in conflicts:
            excluded.append(
                dict(
                    assertion_id=r["assertion_id"],
                    reason="Conflicting normalized laboratory parameters",
                )
            )
            continue
        pr = cal.probabilities(r["age"], r["error"], periods)
        if sum(pr) <= 0:
            continue
        r["period_probability"] = pr
        r.pop("source_row_json")
        keep.append(r)
    used = {r["site_id"] for r in keep}
    return dict(
        periods=periods,
        rows=keep,
        sites={sid: sites[sid] for sid in sorted(used)},
        geometry=[g for g in geometry if g["site_id"] in used],
        exclusions=excluded,
        conflicts=conflicts,
    )


def support(pool, families, config):
    out = []
    gate = config["support_gate"]
    for i, (name, lo, hi) in enumerate(pool["periods"]):
        for fam in families:
            rows = [
                r
                for r in pool["rows"]
                if r["family"] == fam
                and r["period_probability"][i] >= gate["probability_threshold"]
            ]
            ids = sorted({r["site_id"] for r in rows})
            counties = sorted({pool["sites"][sid]["county"] for sid in ids})
            out.append(
                dict(
                    period=name,
                    family=fam,
                    assertions=len(rows),
                    sites=len(ids),
                    counties=counties,
                    site_ids=ids,
                    passed=len(ids) >= gate["minimum_sites"]
                    and len(counties) >= gate["minimum_counties"],
                )
            )
    return out


def registry_cells(lat, lon):
    """Inherited 25 km registry-count cells in a spherical azimuthal equidistant projection."""

    lat = np.radians(lat)
    lon = np.radians(lon)
    la = np.radians(62)
    lo = np.radians(16)
    dl = lon - lo

    angle = np.arccos(
        np.clip(np.sin(la) * np.sin(lat) + np.cos(la) * np.cos(lat) * np.cos(dl), -1, 1)
    )

    scale = np.ones_like(angle)
    nz = angle > 1e-12
    scale[nz] = angle[nz] / np.sin(angle[nz])

    x = 6371.0088 * scale * np.cos(lat) * np.sin(dl)

    y = (
        6371.0088
        * scale
        * (np.cos(la) * np.sin(lat) - np.sin(la) * np.cos(lat) * np.cos(dl))
    )

    return list(zip(np.floor(x / 25).astype(int), np.floor(y / 25).astype(int)))


def terrain_controls(positions, archive):
    """Read the retained, unrotated EPSG4326 GeoTIFF without resampling."""

    with Image.open(io.BytesIO(source_member(archive, "terrain_sweden.tif"))) as image:

        dem = np.asarray(image).copy()

        dx, dy, _ = image.tag_v2[33550]

        i, j, _, x0, y0, _ = image.tag_v2[33922]

        nodata = float(image.tag_v2[42113])

        keys = image.tag_v2[34735]

    assert i == j == 0 and dx > 0 and dy > 0 and 4326 in keys

    result = np.full((len(positions), 2), np.nan)

    for k, p in enumerate(positions):

        # Affine inverse ordering matches the original raster reader.

        col = int(np.floor(p["lon"] / dx - x0 / dx))

        row = int(np.floor(-p["lat"] / dy + y0 / dy))

        if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
            continue

        window = dem[
            max(0, row - 3) : min(dem.shape[0], row + 4),
            max(0, col - 3) : min(dem.shape[1], col + 4),
        ]

        valid = window[window != nodata]

        if dem[row, col] != nodata and len(valid):

            result[k] = [
                float(dem[row, col]) / 500,
                float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1)),
            ]

    return result


def prepare_cohort(genetic, archive, specification):

    SPEC = specification["common"]
    CATS = specification["categories"]
    category = lambda path: lineage_category(path, specification["ancestral_nodes"])

    positions = genetic["spatial_positions"]
    index = {p["position_id"]: i for i, p in enumerate(positions)}

    units = []

    for unit in genetic["analysis_units"]:

        labels = {
            category(genetic["lineage_paths"][k]) for k in unit["branch_node_ids"]
        }

        if len(labels) != 1:
            continue

        assert all(
            abs(r["weight"] - 1 / len(unit["localities"])) < 1e-12
            for r in unit["localities"]
        )

        units.append(dict(unit, category=labels.pop()))

    terrain = terrain_controls(positions, archive)

    keep = [
        u
        for u in units
        if all(
            np.isfinite(terrain[index[l["position_id"]]]).all() for l in u["localities"]
        )
    ]

    excluded = sorted(
        set(u["unit_id"] for u in units) - set(u["unit_id"] for u in keep)
    )

    assert excluded == sorted(SPEC["cohort"]["terrain_excluded_units"])

    units = keep

    local = np.array(
        [index[l["position_id"]] for u in units for l in u["localities"]], dtype=int
    )

    counts = np.array([len(u["localities"]) for u in units], dtype=int)

    starts = np.r_[0, np.cumsum(counts)[:-1]]

    y = np.array([CATS.index(u["category"]) for u in units], dtype=int)

    lat = np.array([p["lat"] for p in positions])
    lon = np.array([p["lon"] for p in positions])

    registry = [
        r
        for r in genetic["records"]
        if r["reconciled_screen_candidate"] and r["source_lat"] is not None
    ]

    density_counts = collections.Counter(
        registry_cells(
            np.array([r["source_lat"] for r in registry]),
            np.array([r["source_lon"] for r in registry]),
        )
    )

    density = np.array(
        [np.log(max(1, density_counts[c])) for c in registry_cells(lat, lon)]
    )

    raw = np.column_stack(
        [
            (lat - 62) * 111.195 / 100,
            (lon - 16) * 111.32 * np.cos(np.radians(62)) / 100,
            density,
            terrain,
        ]
    )

    norm = SPEC["normalization"]
    controls = (raw - norm["nonarchaeological_center"]) / norm["nonarchaeological_sd"]

    tf = Transformer.from_crs(4326, 3006, always_xy=True)

    xy = np.array([tf.transform(p["lon"], p["lat"]) for p in positions])

    assert (
        len(units) == SPEC["cohort"]["units"]
        and len(local) == SPEC["cohort"]["locality_rows"]
    )

    assert dict(collections.Counter(CATS[k] for k in y)) == SPEC["cohort"]["categories"]

    return dict(
        genetic=genetic,
        positions=positions,
        units=units,
        local=local,
        counts=counts,
        starts=starts,
        y=y,
        controls=controls,
        xy=xy,
        blocks=np.array([u["dependence_block"] for u in units]),
        unit_ids=np.array([u["unit_id"] for u in units]),
    )


def feature_definitions(spec):
    definitions = []
    for period, _, _ in spec["periods"]:
        axes = [a for a in spec["axes"] if a["period"] == period]
        fs = axes[0]["feature_ids"]
        assert all(a["feature_ids"] == fs for a in axes)
        definitions.extend((period, f) for f in fs)
    return definitions


def load_position_streams(data, specification, archive):
    """Validate the frozen positional input and its upstream archaeological source."""
    entry = specification["later"]["position_stream_registry"]
    path = data / entry["path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Position-stream registry checksum differs from specification")
    registry = json.loads(raw)
    expected = specification["source_archives"]["archaeology.zip"]
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected or registry["archaeological_source_archive_sha256"] != expected:
        raise ValueError("Position-stream registry belongs to a different archaeological source")
    return registry


def registered_position_draws(geometry, radii, registry):
    """Preserve original draws by stable ID across negligible projection roundoff."""
    if registry.get("schema") != "sclara-preserved-later-position-streams-1" or registry.get("draws") != 64:
        raise ValueError("Unsupported position-stream registry")
    positions = registry["positions"]
    ids = [g["position_id"] for g in geometry]
    if len(ids) != len(set(ids)) or set(ids) != set(positions):
        raise ValueError("Position-stream membership differs from the frozen input")
    shared, streams, draws = {}, {}, []
    for g in geometry:
        row = positions[g["position_id"]]
        centre = (row["latitude"], row["longitude"])
        current = (g["latitude"], g["longitude"])
        radius = row["radius_m"]
        if not all(math.isfinite(x) for x in (*centre, *current, radius)) or radius < 0:
            raise ValueError("Invalid position-stream geometry")
        if not (-90 <= centre[0] <= 90 and -180 <= centre[1] <= 180):
            raise ValueError("Invalid position-stream coordinate range")
        if max(abs(a-b) for a,b in zip(centre,current)) > 1e-9:
            raise ValueError((g["position_id"], "Position-stream centre has changed"))
        if radii.get(current, 0) != radius:
            raise ValueError((g["position_id"], "Position-stream radius has changed"))
        seed = row["seed"]
        if not isinstance(seed, str) or not seed:
            raise ValueError("Missing position-stream seed")
        identity = (*centre, radius, seed)
        if centre in shared and shared[centre] != identity:
            raise ValueError("Inconsistent shared positional stream")
        shared[centre] = identity
        if identity not in streams:
            streams[identity] = position_draws(*centre, radius, seed, 64)
        draws.append(streams[identity])
    return np.array(draws).transpose(1, 0, 2)


def construct(pool, definitions, radii, registry):
    sites = pool["sites"]
    site_ids = sorted(sites)
    si = {s: i for i, s in enumerate(site_ids)}
    geometry = sorted(pool["geometry"], key=lambda g: g["position_id"])
    assert len({g["position_id"] for g in geometry}) == len(geometry)
    gsi = np.array([si[g["site_id"]] for g in geometry])
    xy = np.array([[g["latitude"], g["longitude"]] for g in geometry])
    periods = list(dict.fromkeys(p for p, f in definitions))
    recording = {
        p: sorted({r["site_id"] for r in pool["rows"] if r["period"] == p})
        for p in periods
    }
    pg = {
        p: np.flatnonzero(np.isin(gsi, [si[s] for s in recording[p]])) for p in periods
    }
    features = []
    support = []
    for p, f in definitions:
        rr = [r for r in pool["rows"] if r["period"] == p and r["feature_id"] == f]
        target = sorted({r["site_id"] for r in rr})
        tg = np.flatnonzero(np.isin(gsi, [si[s] for s in target]))
        coords = {tuple(xy[k]) for k in tg}
        other = set(recording[p]) - set(target)
        cg = np.array(
            [
                i
                for i, g in enumerate(geometry)
                if g["site_id"] in other and tuple(xy[i]) not in coords
            ],
            int,
        )
        counties = collections.Counter(sites[s]["county"] for s in target)
        positions = len({sites[s]["position_group"] for s in target})
        groups = len({r["evidence_group"] for r in rr})
        support.append(
            dict(
                period_id=p,
                feature_id=f,
                sites=len(target),
                positions=positions,
                source_groups=groups,
                counties=len(counties),
                largest_county_share=(
                    max(counties.values()) / len(target) if target else None
                ),
                short_lived_sites=len({r["site_id"] for r in rr if r["short_lived"]}),
                minimum_gate_pass=int(positions >= 5 and groups >= 3),
                comparator_sites=len({geometry[k]["site_id"] for k in cg}),
                comparator_positions=len({tuple(xy[k]) for k in cg}),
                target_sites=target,
                assertion_ids=sorted(r["assertion_id"] for r in rr),
            )
        )
        mask = np.zeros(len(geometry), bool)
        mask[cg] = True
        features.append(dict(period=p, tg=tg, cg=cg, comparator_mask=mask))
    gd = registered_position_draws(geometry, radii, registry)
    new = []
    seen = set()
    for g in geometry:
        key = (g["latitude"], g["longitude"])
        if key not in radii and key not in seen:
            new.append(dict(coordinate=list(key),
                            source_uncertainty_m=sites[g["site_id"]]["source_uncertainty_m"],
                            assumed_displacement_m=0))
        seen.add(key)
    recording_indices = {
        p: [
            si[s]
            for s in {sites[s]["position_group"]: s for s in recording[p]}.values()
        ]
        for p in periods
    }
    return dict(
        geometry=geometry,
        draws=gd,
        features=features,
        period_geometry=pg,
        recording_indices=recording_indices,
        site_xy=np.array(
            [[sites[s]["latitude"], sites[s]["longitude"]] for s in site_ids]
        ),
        periods=periods,
        support=support,
        new_streams=new,
    )


def measure_later(d, positions, radius, deadline=float("inf")):
    n = len(positions)
    f = len(d["features"])
    out = {
        k + "_" + v: np.zeros((n, f))
        for k in ["exponential", "gaussian"]
        for v in ["relative", "opportunity"]
    }
    recording = np.zeros((n, len(d["periods"])))
    half = {}
    pd = np.array(
        [
            position_draws(p["lat"], p["lon"], radius * 1000, p["draw_seed"], 64)
            for p in positions
        ]
    ).transpose(1, 0, 2)
    for draw in range(64):
        assert time.monotonic() < deadline, "Frozen measurement time limit reached"
        distances = distance(pd[draw], d["draws"][draw])
        nearest = {}
        for p, pg in d["period_geometry"].items():
            dd = distances[:, pg]
            take = min(32, len(pg))
            ii = np.argpartition(dd, take - 1, axis=1)[:, :take]
            nearest[p] = (pg[ii], np.take_along_axis(dd, ii, axis=1))
        for k, f in enumerate(d["features"]):
            assert len(f["tg"]) and len(f["cg"]), "Unsupported empty target/comparator"
            da = distances[:, f["tg"]].min(1)
            nn, nd = nearest[f["period"]]
            db = np.where(f["comparator_mask"][nn], nd, np.inf).min(1)
            missing = ~np.isfinite(db)
            if missing.any():
                db[missing] = distances[np.ix_(np.flatnonzero(missing), f["cg"])].min(1)
            if draw == 0:
                assert np.max(abs(db[:10] - distances[:10, f["cg"]].min(1))) < 1e-12
            for kernel, a, b in [
                ("exponential", np.exp(-da / 50), np.exp(-db / 50)),
                (
                    "gaussian",
                    sum(np.exp(-da * da / (2 * h * h)) for h in [50, 75, 100]) / 3,
                    sum(np.exp(-db * db / (2 * h * h)) for h in [50, 75, 100]) / 3,
                ),
            ]:
                out[kernel + "_relative"][:, k] += (a - b) / 64
                out[kernel + "_opportunity"][:, k] += (a + b) / 64
        sd = distance(pd[draw], d["site_xy"])
        for k, p in enumerate(d["periods"]):
            recording[:, k] += (
                np.log1p((sd[:, d["recording_indices"][p]] <= 75).sum(1)) / 64
            )
        if draw == 31:
            half = {"half_" + k: v.copy() * 2 for k, v in out.items()}
            half["half_recording"] = recording.copy() * 2
        if draw % 16 == 15:
            print("Measurement", radius, "km", draw + 1, "/64", flush=True)
    out.update(recording=recording, **half)
    return out


def project_later(data, axes, definitions, env):
    scores = []
    controls = []
    periods = list(dict.fromkeys(p for p, f in definitions))
    for a in axes:
        tr = a["transform"]
        indices = [definitions.index((a["period"], f)) for f in a["feature_ids"]]
        raw = data[a["kernel"] + "_relative"][:, indices]
        scores.append(
            (
                (raw - np.array(tr["center"]))
                / np.array(tr["sd"])
                / np.array(tr["balance"])
            )
            @ np.array(tr["coefficient"])
        )
        cc = np.column_stack(
            [
                env,
                data["recording"][:, periods.index(a["period"])],
                data[a["kernel"] + "_opportunity"][:, indices].mean(1),
            ]
        )
        ct = tr["controls"]
        keep = np.array(ct["keep"], bool)
        cc = (cc[:, keep] - np.array(ct["center"])[keep]) / np.array(ct["sd"])[keep]
        controls.append(np.column_stack([np.ones(len(env)), cc]))
    return np.column_stack(scores), np.array(controls)


def later_pool(database, archive, spec, ledger, calibration):
    """Retain strict date envelopes; recompute only verified laboratory repairs."""
    with read_db(database) as db:
        original = {
            r["assertion_id"]: dict(r)
            for r in db.execute("select * from assertions order by assertion_id")
        }
        sites = {r["site_id"]: dict(r) for r in db.execute("select * from sites")}
        geometry = [
            dict(r) for r in db.execute("select * from positions order by position_id")
        ]
        recording_ids = {
            r[0]
            for r in db.execute("select distinct site_id from period_recording_sites")
        }
        decisions = {
            r[0]: json.loads(r[1])
            for r in db.execute("select * from source_adjudications")
        }
    omitted_coordinates = {
        tuple(v) for v in spec["later"]["excluded_reference_stream_coordinates"]
    }
    radii = {}
    for g in geometry:
        key = (g["latitude"], g["longitude"])
        if g["site_id"] in recording_ids and key not in omitted_coordinates:
            radii[key] = max(
                radii.get(key, 0), sites[g["site_id"]]["source_uncertainty_m"] or 0
            )
    pool = source_pool(
        database,
        archive,
        spec["common"]["rules"],
        [["all", -55000, 2000]],
        calibration,
        ledger,
    )
    rows = []
    for row in pool["rows"]:
        before = original[row["assertion_id"]]
        source = json.loads(before["source_row_json"])
        pair = decisions.get(row["assertion_id"], {}).get(
            "corrected_measurement_bp_error"
        ) or [float(source["c14_age_bp"]), float(source["c14_error"])]
        row["original_start_year"], row["original_end_year"] = (
            before["start_year"],
            before["end_year"],
        )
        row["original_age_error"] = pair
        if tuple(map(float, pair)) != (row["age"], row["error"]):
            years, probability = calibration.distribution(row["age"], row["error"])
            order = np.argsort(probability)[::-1]
            selected = order[
                : np.searchsorted(np.cumsum(probability[order]), 0.954) + 1
            ]
            row["start_year"], row["end_year"] = int(years[selected].min()), int(
                years[selected].max()
            )
            row["calendar_rule"] = (
                "verified changed laboratory pair; annual IntCal20 95.4% HPD outer envelope"
            )
        else:
            assert (row["start_year"], row["end_year"]) == (
                before["start_year"],
                before["end_year"],
            )
            row["calendar_rule"] = "unchanged original conservative calendar envelope"
        row.pop("period_probability")
        for period, lo, hi in spec["later"]["periods"]:
            if lo <= row["start_year"] and row["end_year"] < hi:
                rows.append(dict(row, period=period))
    used = {r["site_id"] for r in rows}
    return (
        dict(
            periods=spec["later"]["periods"],
            rows=rows,
            sites={s: pool["sites"][s] for s in sorted(used)},
            geometry=[g for g in pool["geometry"] if g["site_id"] in used],
            exclusions=pool["exclusions"],
            conflicts=pool["conflicts"],
        ),
        radii,
    )


def later_environment(positions, archaeology_archive, paternal_archive, density_reference=None):
    """Reproduce the ten original later-profile controls from shared sources."""
    from rasterio.io import MemoryFile
    from shapely import contains_xy, from_geojson

    world = json.loads(
        source_member(paternal_archive, "references/country_boundaries.geojson")
    )
    sweden = from_geojson(
        json.dumps(
            next(
                f["geometry"]
                for f in world["features"]
                if f["properties"]["ADMIN"] == "Sweden"
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    xy = np.array([[p["lat"], p["lon"]] for p in positions])
    if density_reference is None:
        density = np.log1p((distance(xy, xy) <= 75).sum(1) - 1)
    else:
        reference = np.array([[p["lat"], p["lon"]] for p in density_reference])
        reference_coordinates = set(map(tuple, reference))
        self_match = np.array([tuple(p) in reference_coordinates for p in xy], dtype=int)
        density = np.log1p((distance(xy, reference) <= 75).sum(1) - self_match)
    values = []
    with MemoryFile(
        source_member(archaeology_archive, "terrain_sweden.tif")
    ) as memory, memory.open() as raster:
        for i, p in enumerate(positions):
            points = [[p["lat"], p["lon"]]] + position_draws(
                p["lat"], p["lon"], 25000, "terrain:" + p["position_id"], 16
            ).tolist()
            heights = np.array(
                [v[0] for v in raster.sample([(v[1], v[0]) for v in points])]
            )
            valid = (heights != raster.nodata) & np.isfinite(heights)
            mean = float(heights[valid].mean()) if valid.any() else 0.0
            sd = float(heights[valid].std()) if valid.any() else 0.0
            disk = position_draws(
                p["lat"], p["lon"], 100000, "boundary:" + p["position_id"], 64
            )
            fraction = float(contains_xy(sweden, disk[:, 1], disk[:, 0]).mean())
            lat, lon = p["lat"] - 62, p["lon"] - 16
            values.append(
                [
                    lat,
                    lon,
                    lat * lat,
                    lon * lon,
                    lat * lon,
                    mean,
                    sd,
                    1 - valid.mean(),
                    fraction,
                    density[i],
                ]
            )
    return np.array(values)


def later_cohort(genetic, spec):
    units = []
    for unit in genetic["analysis_units"]:
        labels = {
            lineage_category(genetic["lineage_paths"][node], spec["ancestral_nodes"])
            for node in unit["branch_node_ids"]
        }
        if len(labels) == 1:
            assert all(
                abs(p["weight"] - 1 / len(unit["localities"])) < 1e-12
                for p in unit["localities"]
            )
            units.append(dict(unit, category=labels.pop()))
    positions = genetic["spatial_positions"]
    index = {p["position_id"]: i for i, p in enumerate(positions)}
    counts = np.array([len(u["localities"]) for u in units])
    local = np.array([index[p["position_id"]] for u in units for p in u["localities"]])
    assert len(units) == 1339
    return dict(
        counts=counts,
        local_index=local,
        category=np.array([spec["categories"].index(u["category"]) for u in units]),
        blocks=np.array([u["dependence_block"] for u in units]),
        unit_ids=np.array([u["unit_id"] for u in units]),
    )


def regional_assignments(positions, xy, source, expected_sha256):
    """Assign modern NUTS2 regions using the retained 5-km shoreline rule."""
    from shapely.geometry import Polygon, Point
    from shapely.ops import transform

    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Regional boundary checksum differs")
    tf = Transformer.from_crs(4326, 3006, always_xy=True)
    polygons = {}
    for feature in json.loads(raw)["features"]:
        geometry = feature["geometry"]
        parts = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        polygons[feature["properties"]["NUTS_ID"]] = [
            transform(tf.transform, Polygon(part[0], part[1:])) for part in parts
        ]
    assignments = []
    for position, coordinate in zip(positions, xy):
        point = Point(coordinate)
        inside = sorted(
            k for k, parts in polygons.items() if any(g.covers(point) for g in parts)
        )
        distances = sorted(
            (min(g.distance(point) for g in parts), k) for k, parts in polygons.items()
        )
        if len(inside) == 1:
            key, rule = inside[0], "inside"
        elif not inside and distances[0][0] <= 5000:
            key, rule = distances[0][1], "nearest_within_5km"
        else:
            key, rule = "", ("ambiguous_boundary" if inside else "unassigned_over_5km")
        region = ("SE3" if key.startswith("SE3") else key) if key else ""
        assignments.append(
            dict(
                position_id=position["position_id"],
                nuts=key,
                region=region,
                rule=rule,
                distance_m=0 if inside else distances[0][0],
            )
        )
    return assignments


def neighboring_pairs(pool, maximum_km):
    """Enumerate the frozen nearest-geographical-neighbour candidate pairs."""
    site_ids = sorted(pool["sites"])
    if len(site_ids) < 2:
        return {}
    centers = np.array(
        [[pool["sites"][s]["latitude"], pool["sites"][s]["longitude"]] for s in site_ids]
    )
    pairs = {}
    for start in range(0, len(site_ids), 256):
        stop = min(start + 256, len(site_ids))
        distances = distance(centers[start:stop], centers)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = distances.argmin(1)
        for i, j in enumerate(nearest, start):
            separation = float(distances[i - start, j])
            if separation <= maximum_km:
                pairs[tuple(sorted((site_ids[i], site_ids[j])))] = separation
    return dict(sorted(pairs.items()))


def common_sensitivity_measurements(
    data, specification, ledger, calibration, pool, cohort, positions, detail, output
):
    """Recompute the fixed source, site and paternal-location sensitivities."""
    common = specification["common"]
    contract = common["inference"]["sensitivity"]["archaeological_omissions"]
    contribution = detail["nearest_gap_contribution"]
    ranked = sorted(contribution, key=lambda key: (-contribution[key], key))
    singles = [key for key in ranked if contribution[key] > 0][
        : contract["single_sites"]
    ]
    candidates = neighboring_pairs(pool, contract["pair_maximum_km"])
    joint = {
        tuple(row["sites"]): row["contribution"]
        for row in detail["joint_gap_contribution"]
    }
    if set(candidates) != set(joint):
        raise ValueError("Joint deletion contributions do not cover the frozen candidate pairs")
    pairs = {
        pair: (joint[pair], gap)
        for pair, gap in candidates.items() if joint[pair] > 0
    }
    selected, used = [], set()
    for pair, (value, gap) in sorted(
        pairs.items(), key=lambda item: (-item[1][0], item[0])
    ):
        if used.intersection(pair):
            continue
        selected.append(
            dict(sites=list(pair), combined_contribution=value, distance_km=gap)
        )
        used.update(pair)
        if len(selected) == contract["disjoint_pairs"]:
            break
    assert len(singles) == 5 and len(selected) == 3
    scenarios = [("omit_site", s, [s], 0) for s in singles]
    scenarios += [("omit_pair", "+".join(r["sites"]), r["sites"], 0) for r in selected]
    scenarios += [
        ("paternal_location", str(radius), [], radius)
        for radius in common["inference"]["sensitivity"]["paternal_location"][
            "radii_km"
        ]
    ]
    manifest = dict(
        site_contributions=contribution,
        single_sites=[dict(site_id=s, contribution=contribution[s]) for s in singles],
        neighbor_pairs=selected,
        candidate_pairs=[
            dict(sites=list(pair), joint_contribution=joint[pair], distance_km=gap)
            for pair, gap in candidates.items()
        ],
        selection="Unit-locality weighted exact deletion losses, averaged over the fixed draws, periods and families. Pair losses use the nearest surviving site after joint deletion. Nearest-geographical-neighbour candidates retain the frozen distance threshold. No lineage labels or coefficients enter nomination.",
        scenarios=[],
    )
    arguments = (
        common["periods"],
        common["families"],
        calibration,
        common["rules"]["seed"],
        common["position_streams"],
        positions,
        common["basis"],
    )
    phase = source_pool(
        data / "archaeology.sqlite",
        data / "sources/archaeology.zip",
        common["rules"],
        common["pool_periods"],
        calibration,
        ledger,
        common["phase_omission"]["assertion_ids"],
    )
    arrays, metadata = measure_landscapes(phase, *arguments)
    np.savez_compressed(output / "common_phase_omission.npz", **arrays)
    save_pool(output / "common_phase_omission_pool.json.gz", phase)
    save_json(
        output / "common_phase_omission_support.json",
        dict(support=support(phase, common["families"], common), measurement=metadata),
    )
    for i, (family, key, omitted, radius) in enumerate(scenarios):
        draws = (
            None
            if radius == 0
            else np.array(
                [
                    position_draws(
                        p["lat"], p["lon"], radius * 1000, p["draw_seed"], 64
                    )
                    for p in positions
                ]
            ).transpose(1, 0, 2)
        )
        arrays, metadata = measure_landscapes(
            pool, *arguments, paternal_draws=draws, omitted_sites=omitted
        )
        name = f"common_sensitivity_{i}.npz"
        np.savez_compressed(output / name, **arrays)
        manifest["scenarios"].append(
            dict(
                family=family,
                key=key,
                omitted_sites=omitted,
                paternal_radius_km=radius,
                file=name,
                sha256=hashlib.sha256((output / name).read_bytes()).hexdigest(),
                measurement=metadata,
            )
        )
        print(
            "Measured sensitivity", i + 1, "/", len(scenarios), family, key, flush=True
        )
    save_json(output / "common_omissions.json", manifest)


def common_transport_diagnostics(arrays, cohort, common):
    """Summarize the fixed basis on the represented common-period cohort.

    Each unit contributes equal total weight, shared among its origins. Variance
    is computed across draw-mean relative features after within-period centring;
    projection uses orthonormal loadings, without eigenvalue score scaling.
    """
    proximity = arrays["proximity_draws"].mean(0)
    local = np.asarray(cohort["local"])
    counts = np.asarray(cohort["counts"])
    if counts.ndim != 1 or not len(counts) or np.any(counts <= 0):
        raise ValueError("Transport diagnostics require nonempty units")
    if counts.sum() != len(local) or np.any(local < 0) or np.any(local >= len(proximity)):
        raise ValueError("Unit origins do not match the measurement positions")
    weight = np.bincount(
        local, weights=np.repeat(1 / counts, counts), minlength=len(proximity)
    ) / len(counts)
    basis = common["basis"]
    loadings = np.asarray(basis["loadings"])
    if not np.allclose(loadings.T @ loadings, np.eye(loadings.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("Variance projection requires orthonormal basis loadings")
    relative = proximity - proximity.mean(-1, keepdims=True)
    standardized = (relative - np.asarray(basis["center"])) / np.asarray(basis["scale"])
    centered = standardized - np.sum(weight[:, None, None] * standardized, axis=0)
    total = np.sum(weight[:, None, None] * centered**2, axis=(0, 2))
    projected = np.sum(weight[:, None, None] * (centered @ loadings)**2, axis=(0, 2))
    coverage = np.sum(weight[:, None, None] * (proximity >= np.exp(-2)), axis=0)
    return dict(
        units=len(counts),
        origin_rows=len(local),
        represented_positions=int(np.count_nonzero(weight)),
        weighting="Equal total weight per unit, divided equally among its retained origins",
        variance_definition="Within-period variance of draw-mean relative features on the frozen reference scales",
        coverage_definition="Weight with draw-mean proximity at least exp(-2), equivalent to 100 km under the 50 km kernel",
        families=common["families"],
        periods=[
            dict(
                period=period[0],
                retained_variance_fraction=float(projected[i] / total[i]) if total[i] > 0 else None,
                proximity_equivalent_coverage_fraction=coverage[i].tolist(),
            )
            for i, period in enumerate(common["periods"])
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=["common", "later", "all"], default="all")
    parser.add_argument("--phase-omission", action="store_true")
    parser.add_argument("--radius", type=int, choices=[0, 25, 50], action="append")
    parser.add_argument("--sensitivities", action="store_true")
    args = parser.parse_args()
    if args.phase_omission and args.sensitivities:
        parser.error(
            "Use --sensitivities for the full primary-based batch, or --phase-omission for that single scenario"
        )
    spec = read_json(args.specification)
    for name, expected in spec["source_archives"].items():
        if (
            hashlib.sha256((args.data / "sources" / name).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("Source archive checksum differs: " + name)
    archive = args.data / "sources/archaeology.zip"
    database = args.data / "archaeology.sqlite"
    genetic = read_json(args.data / "paternal.json")
    ledger = read_json(args.data / "adjudications.json")
    calibration = Calibration(archive)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.profile in ["common", "all"]:
        cohort = prepare_cohort(genetic, archive, spec)
        np.savez_compressed(
            args.output / "common_cohort.npz",
            **{
                k: cohort[k]
                for k in [
                    "local",
                    "counts",
                    "starts",
                    "y",
                    "controls",
                    "xy",
                    "blocks",
                    "unit_ids",
                ]
            },
        )
        common = spec["common"]
        regions = common["regions"]
        save_json(
            args.output / "common_regions.json",
            regional_assignments(
                genetic["spatial_positions"],
                cohort["xy"],
                args.data / regions["source_path"],
                regions["source_sha256"],
            ),
        )
        omitted = (
            common["phase_omission"]["assertion_ids"] if args.phase_omission else []
        )
        pool = source_pool(
            database,
            archive,
            common["rules"],
            common["pool_periods"],
            calibration,
            ledger,
            omitted,
        )
        arrays, detail = measure_landscapes(
            pool,
            common["periods"],
            common["families"],
            calibration,
            common["rules"]["seed"],
            common["position_streams"],
            genetic["spatial_positions"],
            common["basis"],
            candidate_pairs=(
                list(neighboring_pairs(
                    pool, common["inference"]["sensitivity"]["archaeological_omissions"]["pair_maximum_km"]
                )) if args.sensitivities else ()
            ),
            origin_weights=(
                np.bincount(
                    cohort["local"],
                    weights=np.repeat(1 / cohort["counts"], cohort["counts"]),
                    minlength=len(genetic["spatial_positions"]),
                )
                if args.sensitivities
                else None
            ),
        )
        name = "common_phase_omission" if args.phase_omission else "common"
        np.savez_compressed(args.output / (name + ".npz"), **arrays)
        transport = common_transport_diagnostics(arrays, cohort, common)
        transport["measurement_sha256"] = hashlib.sha256(
            (args.output / (name + ".npz")).read_bytes()
        ).hexdigest()
        transport["specification_sha256"] = hashlib.sha256(
            args.specification.read_bytes()
        ).hexdigest()
        save_json(args.output / (name + "_transport.json"), transport)
        save_pool(args.output / (name + "_pool.json.gz"), pool)
        save_json(
            args.output / (name + "_support.json"),
            dict(support=support(pool, common["families"], common), measurement=detail),
        )
        print(
            name,
            len(pool["rows"]),
            "assertions;",
            len(pool["sites"]),
            "sites",
            flush=True,
        )
        if args.sensitivities:
            common_sensitivity_measurements(
                args.data,
                spec,
                ledger,
                calibration,
                pool,
                cohort,
                genetic["spatial_positions"],
                detail,
                args.output,
            )
    if args.profile in ["later", "all"]:
        pool, radii = later_pool(database, archive, spec, ledger, calibration)
        definitions = feature_definitions(spec["later"])
        registry = load_position_streams(args.data, spec, archive)
        configuration = construct(pool, definitions, radii, registry)
        environment = later_environment(
            genetic["spatial_positions"], archive, args.data / "sources/paternal.zip"
        )
        cohort = later_cohort(genetic, spec)
        np.savez_compressed(args.output / "later_cohort.npz", **cohort)
        save_pool(args.output / "later_pool.json.gz", pool)
        save_json(
            args.output / "later_support.json",
            dict(
                support=configuration["support"],
                new_streams=configuration["new_streams"],
            ),
        )
        for radius in (
            args.radius
            if args.radius is not None
            else spec["later"]["paternal_radii_km"]
        ):
            arrays = measure_later(configuration, genetic["spatial_positions"], radius)
            scores, controls = project_later(
                arrays, spec["later"]["axes"], definitions, environment
            )
            np.savez_compressed(
                args.output / f"later_{radius}km.npz",
                **arrays,
                scores=scores,
                controls=controls,
                environment=environment,
            )
        print(
            "later",
            len(pool["rows"]),
            "assertions;",
            len(pool["sites"]),
            "sites",
            flush=True,
        )


if __name__ == "__main__":
    main()

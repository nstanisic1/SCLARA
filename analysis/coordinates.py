"""Export source-linked coordinates, their geographical meanings and DMS displays."""

from collections import Counter
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import numpy as np


FIELDS = (
    "entity_type", "entity_id", "parent_id", "latitude_decimal",
    "longitude_decimal", "latitude_dms", "longitude_dms",
    "geographical_meaning", "reliability", "source_locator", "source_uri",
    "reported_uncertainty_m", "common_period_fit", "later_profile_fit",
)
REGISTER_URL = "https://www.raa.se/hitta-information/oppna-data/oppna-data-portal/"


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dms(value, latitude):
    """Round to 0.1 arcsecond, carrying rounded seconds into minutes/degrees."""
    value = float(value)
    degree, remainder = divmod(round(abs(value) * 36000), 36000)
    minute, second = divmod(remainder, 600)
    hemisphere = ("N" if value >= 0 else "S") if latitude else ("E" if value >= 0 else "W")
    return f'{degree}\u00b0{minute:02d}\'{second / 10:04.1f}"{hemisphere}'


def ancient_display_name(row):
    """Restore two verified display spellings while retaining raw source identifiers."""
    name = row["parent_id"]
    locator = row["source_locator"]
    if "S01_tables.xlsx" in locator and "S5.1. Clustered Ancient Samples" in locator:
        return {"Alb√§cksbacken Maglarp": "Albäcksbacken Maglarp",
                "Landbog√•rden": "Landbogården"}.get(name, name)
    return name


def export(data, measurements, figure_data, output):
    paths = {
        "paternal": data / "paternal.json",
        "cohort": measurements / "common_cohort.npz",
        "pool": measurements / "common_pool.json.gz",
        "later_pool": measurements / "later_pool.json.gz",
        "archaeology": data / "archaeology.sqlite",
        "reviews": data / "reports/reviews.json",
        "figure_evidence": figure_data / "evidence.json",
        "ancient_sources": data / "ancient/sources.json",
    }
    rows = []

    def add(kind, identifier, parent, lat, lon, meaning, reliability, source,
            uri="", radius="", common="", later=""):
        if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
            raise ValueError((kind, identifier, "Invalid coordinates"))
        values = (
            kind, identifier, parent, float(lat), float(lon), dms(lat, True),
            dms(lon, False), meaning, reliability, source, uri, radius, common, later,
        )
        rows.append(dict(zip(FIELDS, values, strict=True)))

    genetic = read(paths["paternal"])
    positions = genetic["spatial_positions"]
    with np.load(paths["cohort"], allow_pickle=False) as cohort:
        used = {positions[i]["position_id"] for i in np.unique(cohort["local"])}
    by_position = {
        p["position_id"]: {"units": set(), "records": set(), "evidence": set()}
        for p in positions
    }
    record_lookup = {p["record_id"]: p for p in genetic["records"]}
    for unit in genetic["analysis_units"]:
        for locality in unit["localities"]:
            entry = by_position[locality["position_id"]]
            entry["units"].add(unit["unit_id"])
            entry["records"].update(locality["record_ids"])
            for record in locality["record_ids"]:
                entry["evidence"].update(record_lookup[record]["evidence_ids"])
    for p in positions:
        linkage = by_position[p["position_id"]]
        if not all(linkage.values()):
            raise ValueError((p["position_id"], "Incomplete paternal provenance"))
        locator = (
            "data/paternal.json / " + p["position_id"]
            + " / units: " + ",".join(sorted(linkage["units"]))
            + " / source records: " + ",".join(sorted(linkage["records"]))
            + " / evidence: " + ",".join(sorted(linkage["evidence"]))
        )
        add(
            "paternal_origin", p["position_id"], "", p["lat"], p["lon"],
            "Retained reported paternal-origin point; not a verified ancient residence or universally dated birth/residence location.",
            "Empirical positional error is unknown. The 25/50 km disks are assumed sensitivity scenarios, not measured accuracy.",
            locator, common=int(p["position_id"] in used), later=1,
        )
    pool = read(paths["pool"])
    later_pool = read(paths["later_pool"])
    sites = dict(pool["sites"])
    geometry = {p["position_id"]: p for p in pool["geometry"]}
    for key, p in later_pool["sites"].items():
        if key in sites and p != sites[key]:
            raise ValueError((key, "Conflicting site representatives"))
        sites[key] = p
    for p in later_pool["geometry"]:
        key = p["position_id"]
        if key in geometry and p != geometry[key]:
            raise ValueError((key, "Conflicting geometry representatives"))
        geometry[key] = p
    reviews = read(paths["reviews"])
    review_sources = {}
    for case in reviews["clearance_cases"]:
        review_sources.setdefault(case["site_id"], set()).update(case.get("source_ids", []))
    with sqlite3.connect(paths["archaeology"].resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for case in reviews["later_cases"]:
            for assertion in case["assertions"]:
                row = connection.execute("SELECT site_id FROM assertions WHERE assertion_id=?", (assertion,)).fetchone()
                if row is None:
                    raise ValueError((assertion, "Unresolved reviewed location"))
                review_sources.setdefault(row["site_id"], set()).add(case["source_id"])
        for key in sorted(review_sources):
            if key not in sites:
                row = connection.execute("SELECT * FROM sites WHERE site_id=?", (key,)).fetchone()
                if row is None:
                    raise ValueError((key, "Missing reviewed site"))
                sites[key] = dict(row)
    # Include later and reviewed contexts without changing the common-pool coordinates.
    for key, p in sorted(sites.items()):
        add(
            "archaeological_site", key, p.get("uuid", ""), p["latitude"], p["longitude"],
            "Representative site coordinate in the reviewed measurement pool; not a surveyed radiocarbon sample position." if key in pool["sites"] else "Representative site coordinate in the later measurement pool or original-context review; not a surveyed radiocarbon sample position.",
            "Geometry extent, source precision and context-to-site linkage remain distinct. Coordinate display precision is not accuracy.",
            f"Reviewed archaeological sites / {key} / KMR {p.get('uuid', '')}",
            REGISTER_URL,
            p.get("source_uncertainty_m") if p.get("source_uncertainty_m") is not None else "",
        )
    for p in sorted(geometry.values(), key=lambda p: p["position_id"]):
        add(
            "archaeological_geometry", p["position_id"], p["site_id"],
            p["latitude"], p["longitude"],
            "Retained geometry representative point used for nearest-distance measurement; polygons/lines are represented by projected centroids.",
            "Not the nearest polygon boundary or necessarily the sample location. Extent radius and declared source uncertainty are separate.",
            f"{p.get('source_table', '')} / {p.get('source_fid', '')} / {p['position_id']}",
            REGISTER_URL,
        )
    sources = {x["source_id"]: x for x in read(paths["ancient_sources"])}
    for p in read(paths["figure_evidence"])["ancient_observations"]:
        source = p["source_row"]
        add(
            "ancient_observation", p["individual_id"], p.get("site", ""),
            p["latitude"], p["longitude"],
            "Published archaeological location of an ancient individual; not demonstrated birthplace or lifetime residence.",
            "Source rounding and context/identity qualifications retained. No measured coordinate error or independent-lineage status is inferred.",
            f"{source['source_id']} / {source['sheet']} / row {source['row']}",
            sources[source["source_id"]].get("url", ""),
        )
    if len({(r["entity_type"], r["entity_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate entity identifiers")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with output.open(encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream, delimiter="\t"))
    if len(restored) != len(rows):
        raise ValueError("Coordinate inventory did not round-trip")
    for row in restored:
        for field in ("latitude", "longitude"):
            match = re.fullmatch(r'(\d+)\u00b0(\d+)\'(\d+\.\d)"([NSEW])', row[field + "_dms"])
            if match is None:
                raise ValueError((row["entity_id"], "Invalid DMS display"))
            value = int(match[1]) + int(match[2]) / 60 + float(match[3]) / 3600
            if match[4] in "SW":
                value = -value
            if abs(value - float(row[field + "_decimal"])) >= .051 / 3600:
                raise ValueError((row["entity_id"], "DMS rounding mismatch"))
    # Public archaeological locations contain no modern paternal-origin rows or links.
    public_rows = []
    for row in restored:
        if row["entity_type"] == "paternal_origin":
            continue
        row = dict(row)
        key = row["entity_id"] if row["entity_type"] == "archaeological_site" else row["parent_id"]
        site = sites.get(key, {})
        row["site_name"] = "; ".join(str(x) for x in [site.get("name"), site.get("old_id"), ", ".join(sorted(review_sources.get(key, [])))] if x)
        if row["entity_type"] == "ancient_observation":
            row["site_name"] = ancient_display_name(row)
        public_rows.append(row)
    public = output.with_name("archaeological_coordinates.tsv")
    with public.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, "site_name"), delimiter="\t")
        writer.writeheader()
        writer.writerows(public_rows)
    receipt = {
        "rows": len(rows),
        "public_rows": len(public_rows),
        "public_sha256": sha(public),
        "public_entity_counts": dict(Counter(r["entity_type"] for r in public_rows)),
        "public_excludes_modern_paternal_origins": True,
        "reviewed_site_ids": sorted(review_sources),
        "common_pool_sites": len(pool["sites"]),
        "later_pool_sites": len(later_pool["sites"]),
        "entity_counts": dict(Counter(r["entity_type"] for r in rows)),
        "common_fitted_origin_positions": sum(r["common_period_fit"] == 1 for r in rows),
        "DMS_rounding_arcseconds": .1,
        "interpretation": "DMS is a presentation conversion only. Reported decimal values and source-specific geographical meanings are preserved. No extra positional accuracy is asserted.",
        "sha256": sha(output),
        "input_hashes": {name: sha(path) for name, path in paths.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument("--figure-data", type=Path, default=Path("build/figure_data"))
    parser.add_argument("--output", type=Path, default=Path("build/paper/coordinates.tsv"))
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.measurements, args.figure_data, args.output), indent=2))


if __name__ == "__main__":
    main()
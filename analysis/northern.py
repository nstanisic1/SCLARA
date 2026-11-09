"""Reconstruct the source-qualified northern trapping candidate.

This is a source inventory and support appraisal, with no fitted lineage effect.
Material dates, pit construction, pit use and registered objects remain distinct.
"""

from pathlib import Path
from datetime import time, timedelta
import argparse
import collections
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
import numpy as np
import openpyxl
from lxml import html


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def html_table(raw):
    table = html.fromstring(raw).xpath("//table")[0]
    return [
        [
            " ".join(t.strip() for t in cell.itertext() if t.strip())
            for cell in row.xpath("./th|./td")
        ]
        for row in table.xpath(".//tr")
    ]


def registry_sites(path):
    """Read native register rows with the inherited geometry convention."""
    import math
    from pyproj import Transformer
    from shapely import wkb

    source = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    transform = Transformer.from_crs(3006, 4326, always_xy=True)
    sites = {}
    uuidmap = {}
    geometries = collections.defaultdict(list)
    for row in source.execute("SELECT *FROM lamning ORDER BY uuid"):
        sid = row["lamningsnummer"]
        uuidmap[row["uuid"]] = sid
        sites[sid] = dict(
            site_id=sid,
            old_id=row["raa_nummer"],
            site_type=row["lamningstyp"],
            county=row["lan"],
            municipality=row["kommun"],
            parish=row["socken"],
            longitude=None,
            latitude=None,
            easting=None,
            northing=None,
            geometry_radius_m=None,
            source_uncertainty_m=None,
        )
    for table in ["point", "linestring", "polygon"]:
        for row in source.execute("SELECT *FROM " + table + " ORDER BY fid"):
            sid = uuidmap[row["lamning_uuid"]]
            blob = row["geometri"]
            if blob[:2] != b"GP":
                raise ValueError("Invalid GeoPackage geometry")
            envelope = (blob[3] >> 1) & 7
            g = wkb.loads(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope] :])
            if g.is_empty:
                raise ValueError((sid, "Empty source geometry"))
            pt = g.centroid
            x, y = pt.x, pt.y
            lon, lat = transform.transform(x, y)
            if not (9 <= lon <= 25 and 54 <= lat <= 70):
                raise ValueError((sid, "Source geometry outside retained frame"))
            bounds = g.bounds
            radius = max(
                math.hypot(xx - x, yy - y)
                for xx in [bounds[0], bounds[2]]
                for yy in [bounds[1], bounds[3]]
            )
            geometries[sid].append((x, y, radius))
    for sid, gs in geometries.items():
        x = sum(g[0] for g in gs) / len(gs)
        y = sum(g[1] for g in gs) / len(gs)
        radius = max(math.hypot(g[0] - x, g[1] - y) + g[2] for g in gs)
        lon, lat = transform.transform(x, y)
        sites[sid].update(
            longitude=lon, latitude=lat, easting=x, northing=y, geometry_radius_m=radius
        )
    for (table,) in source.execute(
        "SELECT table_name FROM gpkg_contents ORDER BY rowid"
    ):
        quoted = '"' + table.replace('"', '""') + '"'
        if "lagesosakerhet_i_meter" not in [
            x[1] for x in source.execute("PRAGMA table_info(" + quoted + ")")
        ]:
            continue
        for uid, uncertainty in source.execute(
            "SELECT lamning_uuid,max(lagesosakerhet_i_meter)FROM "
            + quoted
            + " GROUP BY lamning_uuid"
        ):
            if uncertainty is not None:
                sites[uuidmap[uid]]["source_uncertainty_m"] = uncertainty
    source.close()
    return sites


def qualify(data, output):
    catalog = read(data / "northern/sources.json")
    for source in catalog["sources"]:
        p = data / "northern/sources" / source["file"]
        if p.resolve().parent != (data / "northern/sources").resolve():
            raise ValueError("Source path leaves input directory")
        if (
            p.stat().st_size != source["bytes"]
            or digest(p.read_bytes()) != source["sha256"]
        ):
            raise ValueError(source["file"])
    inputs = {}
    recovered = []
    nonrecords = []

    def plain(value):
        if isinstance(value, (time, timedelta)):
            return str(value)
        return value

    def norm(value):
        return re.sub(
            r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip()
        ).casefold()

    def lab(value):
        s = re.sub(r"[\s\-(),.]", "", str(value or "")).upper()
        m = re.fullmatch(r"([A-Z]+)(\d+)([A-Z]?)", s)
        if not m or m[1] not in {
            "UA",
            "ST",
            "BETA",
            "LU",
            "LUA",
            "T",
            "TA",
            "TRH",
            "TRA",
            "KIA",
            "AAR",
            "HELA",
            "GRN",
            "GRM",
            "UBA",
        }:
            return None
        return m[1] + "-" + str(int(m[2])) + m[3]

    def legacy(parish, identifier):
        p = norm(parish)
        if not p:
            return None, None, None
        decoded = False
        if isinstance(identifier, time):
            value = "%d:%d" % (identifier.hour, identifier.minute)
            decoded = True
        elif isinstance(identifier, timedelta):
            seconds = round(identifier.total_seconds())
            value = "%d:%d" % (seconds // 3600, (seconds % 3600) // 60)
            decoded = True
        else:
            value = norm(identifier).replace("_", ":")
        value = re.sub(r"^raä\s*", "", value)
        m = re.fullmatch(r"(\d+)(?::(\d+))?", value)
        if not m:
            return None, None, None
        base = p + " " + str(int(m[1]))
        full = base + ":" + str(int(m[2])) if m[2] else base
        return full, base, ("decoded_excel_time" if decoded else "explicit_locator")

    def member(name):
        b = (data / "northern" / "sources" / name).read_bytes()
        inputs[name] = digest(b)
        return b

    wb = openpyxl.load_workbook(
        io.BytesIO(member("A24_Hennius_DATASET05.xlsx")), read_only=True, data_only=True
    )
    for sheet in wb:
        for row_no, values in enumerate(sheet.iter_rows(values_only=True), 1):
            v = list(values)
            if not any(x is not None for x in v):
                continue
            if row_no == 1 or v[0] in ("Lokal", "Land"):
                nonrecords.append(
                    dict(
                        sheet=sheet.title,
                        row=row_no,
                        reason="header",
                        raw=[plain(x) for x in v],
                    )
                )
                continue
            if all(x is None for x in v[:-1]):
                nonrecords.append(
                    dict(
                        sheet=sheet.title,
                        row=row_no,
                        reason="citation_note",
                        raw=[plain(x) for x in v],
                    )
                )
                continue
            sw = sheet.title == "Sverige"
            r = dict(
                record_id="H2020:%s:%d" % (sheet.title, row_no),
                source="Hennius_deposited_compilation",
                country="Sweden" if sw else "Norway",
                source_locator=dict(
                    archive_member="A24_Hennius_DATASET05.xlsx",
                    sheet=sheet.title,
                    row=row_no,
                ),
                raw_cells=[plain(x) for x in v],
                locality_raw=v[0] if sw else v[1],
                parish_raw=v[1] if sw else None,
                legacy_id_raw=plain(v[2]) if sw else None,
                region_raw=v[3] if sw else v[2],
                feature_raw=v[4] if sw else v[3],
                date_text_raw=v[5] if sw else None,
                lab_raw=v[6] if sw else v[5],
                bp=number(v[7] if sw else v[6]),
                sd=number(v[8] if sw else v[7]),
                reference_raw=v[9] if sw else v[8],
                context_raw=None if sw else v[4],
                event_semantics="sample_to_pit_construction_or_use_not_established_by_deposit_row",
                period_membership_approved=False,
                flags=[],
            )
            r["lab_normalized"] = lab(r["lab_raw"])
            if r["lab_raw"] and r["lab_normalized"] is None:
                r["flags"].append("unrecognized_or_placeholder_laboratory_identifier")
            if r["bp"] is None or r["sd"] is None:
                r["flags"].append("no_numeric_conventional_age_and_error")
            if sw:
                full, base, kind = legacy(v[1], v[2])
                r.update(
                    legacy_full_key=full, legacy_base_key=base, locator_encoding=kind
                )
                if kind == "decoded_excel_time":
                    r["flags"].append(
                        "legacy_id_stored_as_excel_time_decoding_is_provisional"
                    )
            recovered.append(r)
    wb.close()
    table_rows = html_table(member("A24_Rimpi_2025_table4.html"))
    assert len(table_rows) == 62 and all(len(r) == 9 for r in table_rows)
    for row_no, v in enumerate(table_rows[1:], 2):
        m = re.fullmatch(r"(L\d{4}:\d+):(\d+)", v[2])
        assert m, v[2]
        r = dict(
            record_id="R2025:T4:%d" % row_no,
            source="Rimpi_2025_table4",
            country="Sweden",
            source_locator=dict(
                archive_member="A24_Rimpi_2025_table4.html", table=4, row=row_no
            ),
            raw_cells=v,
            lab_raw=v[0],
            lab_normalized=lab(v[0]),
            sample_raw=v[1],
            feature_raw=v[2],
            kmr_system_raw=m[1],
            system_raw=v[3],
            method_raw=v[4],
            bp=number(v[6].replace(",", "")),
            sd=number(v[7].replace(",", "")),
            date_text_raw=v[8],
            interval_probability=0.683,
            event_semantics="buried_ground_charcoal_construction_terminus_post_quem_not_use_date",
            period_membership_approved=False,
            flags=[],
        )
        if v[0] == "Ua-80506":
            r["flags"].append(
                "reported_106.1_under_BP_header_requires_unit_and_calendar_reconciliation"
            )
        recovered.append(r)
    system_table = html_table(member("A24_Rimpi_2025_table3.html"))

    assert collections.Counter(r["country"] for r in recovered) == {
        "Sweden": 353,
        "Norway": 86,
    }
    by_lab = collections.defaultdict(list)
    for r in recovered:
        if r["lab_normalized"]:
            by_lab[r["lab_normalized"]].append(r)
    duplicate_labs = []
    for laboratory_id, group in by_lab.items():
        if len(group) > 1:
            pairs = sorted(set((r["bp"], r["sd"]) for r in group), key=str)
            duplicate_labs.append(
                dict(
                    lab=laboratory_id,
                    record_ids=[r["record_id"] for r in group],
                    measurement_pairs=pairs,
                    values_agree=len(pairs) == 1,
                )
            )
            for r in group:
                r["flags"].append(
                    "repeated_laboratory_identifier_not_an_independent_determination"
                )

    db = data / "archaeology.sqlite"
    con = sqlite3.connect("file:" + db.as_posix() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    registry = registry_sites(data / "northern/sources/registry.sqlite")
    full_index, base_index = collections.defaultdict(set), collections.defaultdict(set)
    wanted_full = {r.get("legacy_full_key") for r in recovered} - {None}
    wanted_base = {r.get("legacy_base_key") for r in recovered} - {None}
    for s in registry.values():
        old = norm(s["old_id"])
        # Do not conflate neighbouring numbers: colon suffixes alone are removed.
        m = re.fullmatch(r"(.*?\s\d+)(?::\d+)?", old)
        base = m[1] if m else old
        if old in wanted_full:
            full_index[old].add(s["site_id"])
        if base in wanted_base:
            base_index[base].add(s["site_id"])
    for r in recovered:
        if r["country"] != "Sweden":
            continue
        if r.get("kmr_system_raw"):
            candidates = {r["kmr_system_raw"]}
            kind = "explicit_KMR_system"
        else:
            f, b = r.get("legacy_full_key"), r.get("legacy_base_key")
            if f in full_index:
                candidates, kind = full_index[f], "exact_legacy_key"
            elif f and ":" not in f and b in base_index:
                candidates, kind = base_index[b], "legacy_base_number"
            else:
                candidates, kind = set(), "unresolved"
            if r.get("locator_encoding") == "decoded_excel_time":
                kind += "_decoded_excel_time_provisional"
        r["candidate_site_ids"] = sorted(candidates)
        r["site_match_rule"] = kind
        r["candidate_site_id"] = (
            next(iter(candidates)) if len(candidates) == 1 else None
        )
        if len(candidates) > 1:
            r["flags"].append("legacy_locator_matches_multiple_KMR_objects")

    canonical = collections.defaultdict(list)
    for a in con.execute(
        "SELECT assertion_id,site_id,lab_id,context_raw,feature_id,material,species,"
        "start_year,end_year,date_relation,dating_risk,short_lived,disposition,source_row_json FROM assertions"
    ):
        identifier = lab(a["lab_id"])
        if identifier not in by_lab:
            continue
        entry = dict(a)
        raw = json.loads(entry.pop("source_row_json"))
        entry.update(
            bp=number(raw.get("c14_age_bp")),
            sd=number(raw.get("c14_error")),
            source_reference={
                k: raw.get(k)
                for k in ["author", "publication_year", "title", "journal"]
            },
            calendar_basis=raw.get("calibration_method"),
        )
        canonical[identifier].append(entry)
    for r in recovered:
        if r["country"] != "Sweden":
            continue
        matches = canonical.get(r["lab_normalized"], [])
        r["canonical_lab_matches"] = []
        for a in matches:
            q = dict(a)
            q["bp_sd_agree"] = (
                r["bp"] is not None
                and r["sd"] is not None
                and r["bp"] == a["bp"]
                and r["sd"] == a["sd"]
            )
            q["site_agrees"] = (
                a["site_id"] in r["candidate_site_ids"]
                if r["candidate_site_ids"]
                else None
            )
            r["canonical_lab_matches"].append(q)
            if a["bp"] is None or a["sd"] is None:
                r["flags"].append("canonical_numeric_age_or_error_missing")
            elif r["bp"] is not None and r["sd"] is not None and not q["bp_sd_agree"]:
                r["flags"].append("canonical_age_or_error_conflict")
            if q["site_agrees"] is False:
                r["flags"].append("canonical_lab_site_conflict")
        if (
            not r["candidate_site_ids"]
            and len({a["site_id"] for a in matches if a["site_id"]}) == 1
        ):
            r["candidate_site_id"] = matches[0]["site_id"]
            r["candidate_site_ids"] = [r["candidate_site_id"]]
            r["site_match_rule"] = (
                "unique_laboratory_site_candidate_not_independent_locator_confirmation"
            )
        elif len(r["candidate_site_ids"]) > 1:
            compatible = {
                a["site_id"] for a in matches if a["site_id"] in r["candidate_site_ids"]
            }
            if len(compatible) == 1:
                r["candidate_site_id"] = next(iter(compatible))
                r["site_match_rule"] = (
                    "exact_laboratory_match_resolves_legacy_base_candidate"
                )
    wanted_sites = {r.get("candidate_site_id") for r in recovered} - {None}
    sites = {}
    for site_id in sorted(wanted_sites):
        s = registry.get(site_id)
        if s:
            sites[site_id] = dict(s)
    periods = [dict(p) for p in con.execute("SELECT * FROM periods")]
    # Only inherited, agreed radiocarbon assertions support this descriptive census.
    # It does not turn charcoal age into a newly validated pit-use date.
    inherited = {}
    for r in recovered:
        for a in r.get("canonical_lab_matches", []):
            if (
                a["bp_sd_agree"]
                and a["site_agrees"] is not False
                and a["feature_id"] == "trapping_pit"
                and a["disposition"] == "eligible_reported_context"
            ):
                inherited[a["assertion_id"]] = a
    phase_census = {}
    assertion_periods = collections.defaultdict(set)
    for assertion_id, period_id in con.execute(
        "SELECT assertion_id,period_id FROM assertion_periods"
    ):
        if assertion_id in inherited:
            assertion_periods[period_id].add(assertion_id)
    for period in periods:
        name = period["period_id"]
        # Reuse existing panel membership, rather than invent a new dating rule.
        existing = {
            s[0]
            for s in con.execute(
                "SELECT site_id FROM panel_units WHERE period_id=? AND feature_id=?",
                (name, "trapping_pit"),
            )
        }
        phase_assertions = [
            inherited[a]
            for a in assertion_periods[name]
            if inherited[a]["site_id"] in existing
        ]
        agreed_sites = {a["site_id"] for a in phase_assertions}
        county_counts = collections.Counter(
            sites[s]["county"] for s in agreed_sites if s in sites
        )
        phase_census[name] = dict(
            inherited_panel_agreed_sites=len(agreed_sites),
            agreed_assertion_ids=sorted(a["assertion_id"] for a in phase_assertions),
            site_ids=sorted(agreed_sites),
            counties=dict(county_counts),
            entire_existing_trapping_panel_sites=len(existing),
        )
    comparator_support = {
        feature: {
            period: count
            for period, count in con.execute(
                "SELECT p.period_id,count(distinct p.site_id) FROM panel_units p "
                "JOIN sites s USING(site_id) WHERE p.feature_id=? AND s.latitude>60 "
                "GROUP BY p.period_id",
                (feature,),
            )
        }
        for feature in ["posthole", "hearth"]
    }
    con.close()

    m = read(data / "paternal.json")
    positions = {p["position_id"]: p for p in m["spatial_positions"]}
    known_xy = [(s["latitude"], s["longitude"]) for s in sites.values()]
    rimpi_ids = {
        r["kmr_system_raw"] for r in recovered if r["source"] == "Rimpi_2025_table4"
    }
    rimpi_xy = [
        (sites[s]["latitude"], sites[s]["longitude"]) for s in rimpi_ids if s in sites
    ]

    def distance_km(lat, lon, xy):
        if not xy:
            return None
        ll = np.radians(np.asarray(xy))
        a, b = np.radians([lat, lon])
        x = (
            np.sin((ll[:, 0] - a) / 2) ** 2
            + np.cos(a) * np.cos(ll[:, 0]) * np.sin((ll[:, 1] - b) / 2) ** 2
        )
        return float(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(x, 0, 1))).min())

    support = {}
    for label, node in [("N_VL29_related", "2621"), ("N_Z1936", "21237")]:
        groups = collections.defaultdict(list)
        for name in [
            "all",
            "any_origin_north_of_60",
            "all_origins_north_of_60",
            "all_candidate_sites_any_origin_within_50km",
            "all_candidate_sites_all_origins_within_50km",
            "Rimpi_six_systems_any_origin_within_50km",
            "Rimpi_six_systems_all_origins_within_50km",
        ]:
            groups[name] = []
        for u in m["analysis_units"]:
            if not all(
                node in m["lineage_paths"].get(str(b), []) for b in u["branch_node_ids"]
            ):
                continue
            ps = [positions[x["position_id"]] for x in u["localities"]]
            entry = dict(unit_id=u["unit_id"], dependence_block=u["dependence_block"])
            groups["all"].append(entry)
            if any(p["lat"] >= 60 for p in ps):
                groups["any_origin_north_of_60"].append(entry)
            if all(p["lat"] >= 60 for p in ps):
                groups["all_origins_north_of_60"].append(entry)
            for source_name, xy in [
                ("all_candidate_sites", known_xy),
                ("Rimpi_six_systems", rimpi_xy),
            ]:
                distances = [distance_km(p["lat"], p["lon"], xy) for p in ps]
                if any(t is not None and t <= 50 for t in distances):
                    groups[source_name + "_any_origin_within_50km"].append(entry)
                if all(t is not None and t <= 50 for t in distances):
                    groups[source_name + "_all_origins_within_50km"].append(entry)
        support[label] = {
            k: dict(
                units=len(v),
                dependence_blocks=len({x["dependence_block"] for x in v}),
                unit_ids=[x["unit_id"] for x in v],
            )
            for k, v in groups.items()
        }

    sw = [r for r in recovered if r["country"] == "Sweden"]
    hen = [r for r in sw if r["source"] == "Hennius_deposited_compilation"]
    rim = [r for r in sw if r["source"] == "Rimpi_2025_table4"]
    result = dict(
        schema="sclara-northern-qualification-1",
        source_sha256=inputs,
        records=recovered,
        excluded_nonrecords=nonrecords,
        duplicate_laboratory_groups=duplicate_labs,
        Rimpi_reported_system_table=system_table,
        candidate_sites=sites,
        inherited_phase_census=phase_census,
        northern_genetic_support=support,
        counts=dict(
            total_source_rows=len(recovered),
            Swedish_source_assertions=len(sw),
            Hennius_Sweden_rows=len(hen),
            Hennius_Norway_rows=len(recovered) - len(sw),
            Hennius_Sweden_numeric_BP_SD=sum(
                r["bp"] is not None and r["sd"] is not None for r in hen
            ),
            Rimpi_determinations=len(rim),
            Rimpi_unique_feature_ids=len({r["feature_raw"] for r in rim}),
            Rimpi_systems=len(rimpi_ids),
            Swedish_recognized_unique_lab_ids=len(
                {r["lab_normalized"] for r in sw if r["lab_normalized"]}
            ),
            Swedish_with_unique_KMR_candidate=sum(
                r.get("candidate_site_id") in sites for r in sw
            ),
            distinct_registered_KMR_candidates=len(sites),
            matched_canonical_laboratory_rows=sum(
                bool(r.get("canonical_lab_matches")) for r in sw
            ),
            existing_agreed_eligible_trapping_assertions=len(inherited),
        ),
        flags=dict(collections.Counter(f for r in recovered for f in set(r["flags"]))),
        geographic_scope=dict(
            counties=dict(collections.Counter(s["county"] for s in sites.values())),
            site_types=dict(
                collections.Counter(s["site_type"] for s in sites.values())
            ),
        ),
        effect_model=None,
        interpretation_limits=[
            "Rows, laboratory determinations, individual pits and registered sites are distinct units. Registered KMR candidates are not certified independent hunting systems.",
            "Excel-time site identifiers are mechanically decoded candidates, not independently confirmed corrections. Multi-object locators remain unresolved.",
            "Rimpi buried-ground charcoal supplies construction termini; the source 68.3% calendar limits are not equivalent to the frozen 95.4% use-period rule.",
            "Hennius deposited rows lack individual sample-to-construction/use links. Canonical exact-lab matches retain existing assumptions, not new original-report validation.",
            "Any 50 km counts are descriptive location-scenario support, not power estimates, geographic matching, independent lineages, fitted associations or validated date panels.",
            "The later-settlement/contemporary land-use comparator remains necessary; the source limitation does not establish national absence.",
        ],
    )
    result["appraisal"] = read(data / "northern/appraisal.json")
    if comparator_support != result["appraisal"]["northern_existing_comparator_counts"]:
        raise ValueError("Retained northern comparator counts changed")
    result["comparator_support"] = comparator_support
    result["canonical_inputs"] = {
        name: digest((data / name).read_bytes())
        for name in [
            "paternal.json",
            "archaeology.sqlite",
            "northern/sources/registry.sqlite",
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["counts"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("build/northern/evidence.json"))
    a = p.parse_args()
    qualify(a.data, a.output)


if __name__ == "__main__":
    main()

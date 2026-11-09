"""Reconstruct the bounded ancient-source inventory, identities and branch calls.

Published dates, source exclusions and relatedness remain distinct assertions.
This adapter does not recall genotypes or infer lineage arrival times.
"""

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import collections
import csv
import hashlib
import json
import math
import re
import zipfile
import openpyxl
from analysis.dates import overlaps_period


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def headers(row):
    counts = Counter()
    result = []
    for i, h in enumerate(row):
        h = str(h) if h is not None else "unnamed_" + str(i)
        counts[h] += 1
        result.append(h if counts[h] == 1 else h + "__" + str(counts[h]))
    return result


def clean(x):
    x = str(x)
    for prefix in ["S01:", "ancient:"]:
        if x.startswith(prefix):
            x = x[len(prefix) :]
    return x


def broad(label):
    label = str(label or "").strip()
    if label.upper() in {"", "NA", "N/A", "NONE", "NAN", "NULL", "-", ".."}:
        return None
    for k in ["I1", "I2", "R1a", "R1b", "N"]:
        if label.startswith(k):
            return k
    return None


def locator(row):
    return {k: row[k] for k in ["source_id", "sheet", "row"]}


def rowkey(row):
    return tuple(row[k] for k in ["source_id", "sheet", "row"])


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def population_analysis_status(value, source_id, field):
    """Normalize the published inclusion field without treating missing as passed."""
    text = str(value).strip().lower() if value is not None else ""
    if field == "used in population genetic analyses" and source_id == "A19_Cell2023_TableS1.xlsx":
        token = re.match(r"^(true|false)(?:\s|$)", text)
        if token:
            return "included" if token.group(1) == "true" else "excluded"
    if field in {"used in population genetic analyses", "Imputed & passing all filters"}:
        if text in {"true", "yes", "1", "1.0", "included"}:
            return "included"
        if text in {"false", "no", "0", "0.0", "excluded"}:
            return "excluded"
    return "unknown"


def source_flag_categories(value, source_id):
    """Categories for the actual retained source vocabularies; keep raw flags."""
    common = {"low_cov": "low_coverage", "low_gpave_cov": "genotype_probability",
              "high_ibd_sharing": "kinship", "score_tie": "call_ambiguity",
              "unstable_downstream": "call_ambiguity"}
    allentoft = {"lowcov": "low_coverage", "lowgpavg": "genotype_probability",
                "contmt_5pct": "mitochondrial_contamination"}
    categories = set()
    for token in str(value or "").lower().split(";"):
        token = token.strip()
        if source_id == "S01_tables.xlsx":
            if token in common: categories.add(common[token])
            if token.startswith("merged_"): categories.add("sample_identity")
        elif source_id in {"S03_41586_2023_6862_MOESM5_ESM.xlsx", "S29_MOESM6.xlsx"}:
            if token in allentoft: categories.add(allentoft[token])
            if re.match(r"^[12]d_rel_", token): categories.add("kinship")
    return sorted(categories)


def validated_alias_edges(raw, aliases):
    """Resolve explicitly reviewed cross-study identities against their exact rows."""
    edges = []
    for alias in aliases:
        keys = [tuple(alias[k]) for k in ("left_source", "right_source")]
        if any(k not in raw for k in keys):
            raise ValueError("Reviewed alias source row is absent")
        if str(raw[keys[0]].get("ID")) != alias["left"] or str(raw[keys[1]].get("popId")) != alias["right"]:
            raise ValueError("Reviewed alias does not match its source identifiers")
        if not alias.get("primary_citation") or not alias.get("primary_url"):
            raise ValueError("Reviewed alias lacks primary-source credit")
        edges.append((alias["left"], alias["right"], alias["basis"], [list(k) for k in keys]))
    return edges


def canonical_identity_ids(groups, aliases):
    """Keep declared mapped identifiers stable without heuristic cross-study merges."""
    def preference(x):
        return (x.startswith("CGG_2_"), "_" in x, len(x), x)

    canonical = {}
    for ids in groups.values():
        chosen = set()
        for alias in aliases:
            if alias["left"] in ids or alias["right"] in ids:
                if not {alias["left"], alias["right"], alias["canonical_id"]}.issubset(ids):
                    raise ValueError("Declared canonical identity is outside its alias component")
                chosen.add(alias["canonical_id"])
        if len(chosen) > 1:
            raise ValueError("Conflicting canonical identities in one alias component")
        key = next(iter(chosen)) if chosen else min(ids, key=preference)
        canonical.update({i:key for i in ids})
    return canonical


def build_identity(data, output, selection):
    OUT = output
    verified = read(OUT / "verified_assertions.json")
    qualification = selection["qualification"]
    parent = {}
    edges = []

    def find(x):
        x = clean(x)
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b, basis, evidence):
        a, b = clean(a), clean(b)
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[max(aa, bb)] = min(aa, bb)
        edges.append(dict(left=a, right=b, basis=basis, source_rows=evidence))

    def source_key(r):
        return r["source_id"], r["sheet"], r["row"]

    raw = {source_key(r): r["raw"] for r in verified["assertions"]}
    links = defaultdict(set)
    for r in verified["identity_links"]:
        links[source_key(r)].add(clean(r["retained_identity"]))
        find(r["retained_identity"])
    for k, ids in links.items():
        ids = sorted(ids)
        for i in ids[1:]:
            union(ids[0], i, "same verified source row", [list(k)])
    for e in qualification["entities"]:
        if len(e["source_ids"]) <= 1:
            continue
        keys = [
            k
            for k, r in raw.items()
            if k[0] == "S01_tables.xlsx"
            and (
                str(r.get("popId")) in e["source_ids"]
                or any(i in str(r.get("flag", "")) for i in e["source_ids"])
            )
        ]
        tokens = set()
        for k in keys:
            rr = raw[k]
            text = " ".join(str(rr.get(v, "")) for v in ["popId", "flag"])
            tokens.update(re.findall(r"CGG\d+", text))
            tokens.add(str(rr.get("popId")))
        assert all(i in tokens for i in e["source_ids"]), (
            e["entity_id"],
            "missing published merge identifiers",
        )
        for i in e["source_ids"]:
            union(
                e["entity_id"],
                i,
                "published composite sample or merged flag",
                [list(k) for k in keys],
            )
    for alias in qualification["cross_study"]["S29_explicit_aliases"]:
        left, right = alias["left"], alias["right"]
        evidence = []
        for k, r in raw.items():
            if (
                k[0] == "S29_MOESM4.xlsx"
                and str(r.get("unnamed_0", "")).rstrip("*") == left
                and r.get("CGG sample ID") == right
            ):
                evidence.append(list(k))
            if (
                k[0] == "S29_MOESM3.xlsx"
                and str(r.get("Sample ID", "")).rstrip("*") == left
                and right in str(r.get("Relatedness and notes"))
                and "same individual" in str(r.get("Relatedness and notes")).lower()
            ):
                evidence.append(list(k))
        assert evidence, (alias, "alias lacks matching published row")
        union(left, right, alias["basis"], evidence)
    reviewed_aliases = qualification["cross_study"].get("reviewed_aliases", [])
    for edge in validated_alias_edges(raw, reviewed_aliases):
        union(*edge)
    groups = defaultdict(list)
    for i in sorted(parent):
        groups[find(i)].append(i)

    canonical = canonical_identity_ids(groups, reviewed_aliases)
    individuals = []
    for ids in groups.values():
        key = canonical[ids[0]]
        sources = [
            list(k) for k, v in links.items() if any(canonical[i] == key for i in v)
        ]
        individuals.append(
            dict(
                individual_id=key,
                aliases=ids,
                source_rows=sorted(sources),
                independent_paternal_lineage=False,
                independence_status="Not established by source identity or shared ancestry cluster.",
            )
        )
    for key, ids in links.items():
        assert len({canonical[i] for i in ids}) == 1
    observations = []
    for k, r in raw.items():
        if k[0] != "S01_tables.xlsx" or k[1] != "S5.1. Clustered Ancient Samples":
            continue
        pid = str(r["popId"])
        assert pid in canonical
        lo, hi = r.get("ageHigh"), r.get("ageLow")
        bounds = (
            [1950 - lo, 1950 - hi]
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float))
            else None
        )
        if bounds:
            assert bounds[0] <= bounds[1]
        observations.append(
            dict(
                individual_id=canonical[pid],
                source_id=k[0],
                sheet=k[1],
                row=k[2],
                sample_id=r["sampleId"],
                published_id=pid,
                site=r["site"],
                country=r["country"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                sex=r["sex"],
                mean_year=(
                    1950 - r["ageAverage"]
                    if isinstance(r["ageAverage"], (int, float))
                    else None
                ),
                calendar_bounds=bounds,
                date_type=r["datingSource"],
                raw_radiocarbon=r["age14C"],
                raw_date=r["ageRaw"],
                broad_Y=r["hgYMajor"],
                reported_Y=r["hgYMinor"],
                ancestry_cluster=r["clusterIBDGlobal"],
                ancestry_cluster2=r["clusterIBDGlobal2"],
                ancestry_cluster_fine=r["clusterIBDFine"],
                data_source=r["dataSource"],
                genome_coverage=r["dpAvg_all"],
                status="Source assertion; original dating, branch and kinship qualifications retained separately.",
            )
        )
    workbook = openpyxl.load_workbook(
        data / "sources/S01_tables.xlsx", read_only=True, data_only=True
    )
    it = workbook["S5.2. Reclustering (2800 BP +)"].iter_rows(values_only=True)
    head = headers(next(it))
    recluster = {}
    for n, values in enumerate(it, 2):
        r = dict(zip(head, values))
        sid = r.get("sampleId")
        if sid in recluster:
            assert recluster[sid][1] == r
        else:
            recluster[sid] = (n, r)
    for o in observations:
        if o["sample_id"] in recluster:
            n, r = recluster[o["sample_id"]]
            o["pre2800BP_cluster"] = r["clusterIBDGlobal_2.8k"]
            o["pre2800BP_source_row"] = n
        else:
            o["pre2800BP_cluster"] = None
            o["pre2800BP_source_row"] = None
    workbook.close()
    windows = {
        "Late_Neolithic": [-2300, -1700],
        "Bronze_Age": [-1700, -500],
        "Roman_Iron_Age": [1, 400],
        "Migration": [400, 550],
        "Vendel": [550, 750],
        "Viking": [750, 1050],
        "Medieval": [1050, 1520],
    }
    for o in observations:
        year = o["mean_year"]
        o["nominal_period"] = next(
            (
                p
                for p, (lo, hi) in windows.items()
                if year is not None and lo <= year < hi
            ),
            None,
        )
        b = o["calendar_bounds"]
        o["overlapping_periods"] = [
            p for p, (lo, hi) in windows.items() if overlaps_period(b, lo, hi)
        ]
        o["date_semantics"] = (
            "Source calBP bounds are converted using 1950 as the reference year; the bounds are not treated as uniform probability distributions. The nominal period follows the published average."
        )
    early = [
        o
        for o in observations
        if o["nominal_period"] in ["Late_Neolithic", "Bronze_Age"]
    ]
    selected = [
        o
        for o in early
        if o["pre2800BP_cluster"] in ["pre2.8k_0_1_3_2-EScan", "pre2.8k_0_1_3_1-SScan"]
    ]
    assert len(early) == 17 and len(selected) == 16
    result = dict(
        schema="ancient-source-inventory-1",
        input_sha256={
            "verified_assertions.json": sha(OUT / "verified_assertions.json"),
            "sources.json": sha(data / "sources.json"),
            "retained_qualification": sha(data / "selection.json"),
        },
        script_sha256=sha(Path(__file__)),
        identity_rule="Exact retained identifiers, the same original source row, published composite/merged samples and explicit same-individual aliases only. A pairwise genetic-relative edge does not merge people.",
        identity_edges=edges,
        individuals=sorted(individuals, key=lambda x: x["individual_id"]),
        McColl_observations=observations,
        period_windows=windows,
        counts={
            "verified_source_rows": len(raw),
            "documented_identity_groups": len(individuals),
            "McColl_clustered_Swedish_observations": len(observations),
            "McColl_LN_BA_nominal_candidates": len(early),
            "McColl_LN_BA_two_cluster_candidates": len(selected),
        },
        scope="Finite retained source inventory, not an exhaustive census or a population-frequency sample. The 135 McColl observations are a source-selected mapping layer; other verified source rows remain available and are not silently treated as absent.",
    )
    save(OUT / "identity_inventory.json", result)
    print(json.dumps(result["counts"], indent=2))
    print("Alias edges", len(edges))


def build_branches(data, output, selection):
    OUT = output
    assert all(broad(v) is None for v in [None, "NA", "N/A", "None", "-", "..", ""])
    assert broad("N1a1a") == "N" and broad("N-M231") == "N" and broad("I1a1") == "I1"
    inventory = read(OUT / "identity_inventory.json")
    aliases = {
        a: r["individual_id"] for r in inventory["individuals"] for a in r["aliases"]
    }
    raw = read(OUT / "verified_assertions.json")["assertions"]
    with zipfile.ZipFile(data / "sources/YFull_tree_12.00.0.zip") as z:
        tree = json.loads(z.read("tree_12.00.0.json"))
    paths = {}
    markers = defaultdict(set)

    def walk(n, path):
        paths[n["id"]] = path + [n["id"]]
        for snp in re.split(r"[,/ ]+", n.get("snps", "")):
            if snp:
                markers[snp].add(n["id"])
        for c in n.get("children", []):
            walk(c, paths[n["id"]])

    walk(tree, [])

    def from_label(label):
        if label in paths:
            return paths[label]
        hits = markers.get(str(label).split("-", 1)[-1], set())
        return paths[next(iter(hits))] if len(hits) == 1 else []

    def qualified(path):
        return [
            label
            for node, label in [
                ("I1", "I1"),
                ("I-L22", "I1_L22"),
                ("I-Z58", "I1_Z58"),
                ("R-Z284", "R1a_Z284"),
                ("R-U106", "R1b_U106"),
                ("R-P312", "R1b_P312"),
            ]
            if node in path
        ]

    def source(r):
        return dict(source_id=r["source_id"], sheet=r["sheet"], row=r["row"])

    dictionary = defaultdict(list)
    for n, line in enumerate(
        (data / "sources/Yleaf_2_1_hg19.txt").read_text().splitlines(), 1
    ):
        f = line.split("\t")
        if len(f) == 7:
            dictionary[f[2]].append(
                dict(row=n, marker=f[1], hg19_position=f[3], change=f[4])
            )
    with (data / "sources/S02_scoreties_summary.yplace").open() as f:
        ties = list(csv.DictReader(f, delimiter="\t"))
    with (data / "sources/S02_unstabledownstream.yplace").open() as f:
        unstable = list(csv.DictReader(f, delimiter="\t"))
    new = defaultdict(list)
    viking_y = {}
    calls = []
    for rr in raw:
        r = rr["raw"]
        if rr["source_id"] == "S01_tables.xlsx" and rr["sheet"].startswith("S1.1"):
            new[str(r["popId"])].append(rr)
        if rr["source_id"].startswith("S06") and "haplogroups" in rr["sheet"]:
            viking_y[r["SampleID"]] = rr
    # Preserve one record per published call, including broad calls not mapped to a terminal marker.
    for rr in raw:
        r = rr["raw"]
        path = []
        proof = []
        dictionary_rows = []
        flags = {}
        label = None
        pid = None
        agrees = None
        if rr["source_id"] == "S01_tables.xlsx" and rr["sheet"].startswith("S5.1"):
            pid = str(r["popId"])
            label = r["hgYMinor"]
            path = from_label(label)
            basis = "Literal published terminal marker uniquely mapped to frozen YFull12 tree"
            proof = [source(rr)]
            reported_broad = broad(r["hgYMajor"]) or broad(label)
            n = new.get(pid, [])
            if n:
                tt = [t for t in ties if t["individual"] == pid]
                flags = {
                    "published_metadata": [
                        {
                            k: x["raw"].get(k)
                            for k in [
                                "flag",
                                "flag__2",
                                "chrY_DoC",
                                "C14 Lab #",
                                "Museum inv #",
                                "Context",
                            ]
                        }
                        for x in n
                    ],
                    "deposited_ties": tt,
                    "deposited_unstable": [
                        t for t in unstable if t["individual"] == pid
                    ],
                }
                reported = [
                    str(x["raw"].get("root_to_tip_path") or "").split(">") for x in n
                ]
                reported = [p for p in reported if p != [""]]
                if reported:
                    assert all(p == reported[0] for p in reported)
                    path = reported[0]
                    basis = "Published root-to-tip path with deposited tie restriction"
                    proof.extend(source(x) for x in n)
                    tt = [t for t in ties if t["individual"] == pid]
                    if tt:
                        common = set.intersection(
                            *(
                                set(paths.get(t["most_recent_common_parent"], []))
                                for t in tt
                            )
                        )
                        path = [v for v in path if v in common]
        elif (
            rr["source_id"].startswith("S06")
            and rr["sheet"] == "TableS1_New samples"
            and r["Sex"] == "Male"
        ):
            pid = r["Sample__2"]
            label = str(r["Y_haplogroup"] or "").strip()
            h = viking_y[pid]
            agrees = label == str(h["raw"]["Y chromosomal haplogroups"] or "").strip()
            assert agrees
            dictionary_rows = dictionary.get(label, [])
            matches = {
                next(iter(markers[x["marker"]]))
                for x in dictionary_rows
                if len(markers.get(x["marker"], [])) == 1
            }
            common = (
                set.intersection(*(set(paths[n]) for n in matches))
                if matches
                else set()
            )
            node = max(common, key=lambda n: len(paths[n])) if common else None
            path = paths[node] if node else []
            basis = "Source-version Yleaf2.1 dictionary: common ancestor of unique marker-name matches; not the byte-identical original caller database"
            proof = [source(rr), source(h)]
            reported_broad = broad(label)
            flags = {
                k: r.get(k)
                for k in [
                    "Relatedness",
                    "Contam_mtDNA",
                    "Contam_X (%)",
                    "Damage (%)",
                    "AvgDepth(X)",
                ]
            }
        elif rr["source_id"] == "A19_Cell2023_TableS1.xlsx":
            pid = r["id"]
            label = r["chrY_hapGrp_isogg2019"]
            reported_broad = broad(label)
            basis = "Published broad parent only; no terminal SNP assigned from historical alphanumeric label"
            proof = [source(rr)]
            flags = {
                k: r.get(k)
                for k in [
                    "used in population genetic analyses",
                    "FiltmapQ30_DoC_AUTO",
                    "FiltmapQ30_DoC_chrY",
                    "xContam_meth1_prop",
                    "xContam_meth2_prop",
                ]
            }
        elif rr["source_id"] == "S29_MOESM3.xlsx":
            pid = str(r["Sample ID"]).rstrip("*")
            label = r["Y-chr haplogroup"]
            reported_broad = broad(label)
            basis = (
                "Published broad parent only; original source-era nomenclature retained"
            )
            proof = [source(rr)]
            flags = {
                k: r.get(k)
                for k in [
                    "Genomic coverage, X",
                    "Imputed & passing all filters",
                    "Relatedness and notes",
                ]
            }
        else:
            continue
        assert pid in aliases, pid
        calls.append(
            dict(
                individual_id=aliases[pid],
                reported_id=pid,
                source_rows=proof,
                reported_Y=label,
                reported_broad_parent=reported_broad,
                qualified_path=path,
                qualified_targets=qualified(path),
                basis=basis,
                dictionary_rows=dictionary_rows,
                original_Viking_tables_agree=agrees,
                quality_and_relatedness=flags,
            )
        )
    # The retained source-qualified crosswalk is a verification target, not a computational input.
    old = selection["reference_viking_calls"]
    current = {
        r["reported_id"]: r
        for r in calls
        if r["source_rows"][0]["source_id"].startswith("S06")
    }
    assert len(current) == len(old) == 82
    for r in old:
        assert (
            current[r["id"]]["qualified_path"] == r["qualified_path"]
            and current[r["id"]]["reported_Y"] == r["exact_source_label"]
        ), r["id"]
    grouped = defaultdict(list)
    for r in calls:
        grouped[r["individual_id"]].append(r)
    people = []
    for pid, rr in sorted(grouped.items()):
        pp = [r["qualified_path"] for r in rr if r["qualified_path"]]
        conflict = any(p[-1] not in q and q[-1] not in p for p in pp for q in pp)
        path = max(pp, key=len) if pp and not conflict else []
        parents = sorted(
            {r["reported_broad_parent"] for r in rr if r["reported_broad_parent"]}
        )
        parent_conflict = len(parents) > 1
        if parent_conflict:
            path = []
        people.append(
            dict(
                individual_id=pid,
                reported_broad_parents=parents,
                qualified_targets=qualified(path),
                qualified_path=path,
                path_conflict=conflict,
                broad_parent_conflict=parent_conflict,
                call_count=len(rr),
                interpretation="Compatible finer source calls refine coarse calls; unresolved labels do not imply absence. Quality and relatedness remain separate qualifications.",
            )
        )
    result = dict(
        schema="ancient-branch-crosswalk-1",
        input_sha256={
            name: hashlib.sha256(
                (data / name if name == "sources.json" else OUT / name).read_bytes()
            ).hexdigest()
            for name in [
                "verified_assertions.json",
                "identity_inventory.json",
                "sources.json",
            ]
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calls=calls,
        individuals=people,
        counts={
            "source_call_rows_including_missing": len(calls),
            "identities_with_call_rows": len(people),
            "identities_with_reported_parent_or_mapped_path": sum(
                bool(r["reported_broad_parents"] or r["qualified_path"]) for r in people
            ),
            "Viking_original_male_records": len(current),
            "Viking_crosswalk_exact_matches": len(old),
            "Viking_mapped_paths": sum(
                bool(r["qualified_path"]) for r in current.values()
            ),
            "incompatible_qualified_paths": sum(r["path_conflict"] for r in people),
            "conflicting_broad_parents": sum(
                r["broad_parent_conflict"] for r in people
            ),
        },
        limits=[
            "Published assignments and source-version nomenclature reconciliation, not independent genotype validation.",
            "Qualified targets are nested: I1_L22 is also I1. Use mutually exclusive groups for any counts.",
            "R1b-L51, L11 or M269 are not automatically U106 or P312; upstream R1a is not automatically Z284.",
            "Cross-study agreement is not independent replication; documented relatives and merged identities remain explicit.",
        ],
    )
    (OUT / "branch_crosswalk.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["counts"], indent=2))


def build_dates(data, output):
    OUT = output
    verified = read(OUT / "verified_assertions.json")
    identity = read(OUT / "identity_inventory.json")
    branches = read(OUT / "branch_crosswalk.json")
    sources = {x["source_id"]: x for x in read(data / "sources.json")}
    rows = verified["assertions"]
    bykey = {rowkey(x): x for x in rows}
    owner = {}
    aliases = collections.defaultdict(set)
    for x in identity["individuals"]:
        for key in x["source_rows"]:
            key = tuple(key)
            assert key not in owner or owner[key] == x["individual_id"]
            owner[key] = x["individual_id"]
        for alias in x["aliases"]:
            aliases[alias.casefold()].add(x["individual_id"])

    def identify(value):
        matches = aliases.get(str(value).casefold(), set())
        return next(iter(matches)) if len(matches) == 1 else None

    calls = collections.defaultdict(list)
    for c in branches["calls"]:
        calls[c["individual_id"]].append(c)
    periods = identity["period_windows"]
    relationships = []
    for row in rows:
        if row["source_id"] == "S29_MOESM5.xlsx":
            keys = ("sample1", "sample2")
        elif row["source_id"] == "A19_Cell2023_TableS2.xlsx":
            keys = ("Ind1", "Ind2")
        else:
            continue
        raw = row["raw"]
        ends = [str(raw[k]) for k in keys]
        relationships.append(
            dict(
                source_row=locator(row),
                reported_individuals=ends,
                individual_ids=[identify(x) for x in ends],
                identity_rule="Unique case-insensitive match to already documented published identifiers; unmatched relatives remain external, never merged.",
                raw=raw,
                interpretation="Published relatedness assertion; not same-person identity and not evidence for independent paternal lineages.",
            )
        )

    def temporal(lo, hi, mean):
        return dict(
            overlapping_periods=[
                p for p, (a, b) in periods.items() if overlaps_period([lo, hi], a, b)
            ],
            fully_contained_periods=[
                p for p, (a, b) in periods.items() if lo >= a and hi < b
            ],
            nominal_period=next(
                (p for p, (a, b) in periods.items() if number(mean) and a <= mean < b),
                None,
            ),
            rule="Period overlap/containment uses reported calendar bounds; nominal period uses the source midpoint. No uniform date probabilities or reconstructed likelihoods.",
        )

    observations = []
    samples = {
        x["raw"]["Sample ID"]: x for x in rows if x["source_id"] == "S29_MOESM3.xlsx"
    }
    for row in rows:
        if row["source_id"] != "S29_MOESM4.xlsx":
            continue
        raw = row["raw"]
        sample = raw["unnamed_0"]
        individual = owner[rowkey(row)]
        meta = samples[sample]
        assert owner[rowkey(meta)] == individual
        variants = []
        for name, start, end, mid in [
            (
                "original_calibrated",
                "BC cal from (95.4%)",
                "BC cal to (95.4%)",
                "BC cal midpoint",
            ),
            (
                "published_reservoir_corrected",
                "BC cal rescorr from (95.4%)",
                "BC cal rescorr to (95,4%)",
                "BC cal midpoint rescorr",
            ),
        ]:
            lo, hi, mean = raw.get(start), raw.get(end), raw.get(mid)
            if number(lo) and number(hi):
                assert lo <= hi
                variants.append(
                    dict(
                        name=name,
                        calendar_bounds=[lo, hi],
                        source_midpoint=mean,
                        source_columns=[start, end, mid],
                        confidence_level=0.954,
                        **temporal(lo, hi, mean)
                    )
                )
        related = [r for r in relationships if individual in r["individual_ids"]]
        quality = [
            dict(
                source_row=locator(x),
                flags={
                    k: v
                    for k, v in x["raw"].items()
                    if k
                    in [
                        "Coverage",
                        "Average GP",
                        "Flag",
                        "Genomic coverage, X",
                        "Imputed & passing all filters",
                        "Relatedness and notes",
                    ]
                },
            )
            for x in rows
            if owner.get(rowkey(x)) == individual
            and x["source_id"] in ["S29_MOESM3.xlsx", "S29_MOESM6.xlsx"]
        ]
        observations.append(
            dict(
                individual_id=individual,
                reported_id=sample,
                source_row=locator(row),
                source_sha256=sources[row["source_id"]]["sha256"],
                site=raw["Site"],
                site_type=raw["Site type"],
                country=raw["Country"],
                latitude=raw["Lat"],
                longitude=raw["Long"],
                sample_description=raw["Sample description"],
                sex=meta["raw"]["DNA sex ID"],
                laboratory_id=raw["14C Lab ID"],
                uncalibrated_bp=raw["BP uncal"],
                laboratory_sd_bp=raw["1s"],
                date_variants=variants,
                date_basis="Published sample radiocarbon calibration where a laboratory determination is supplied; preserve both reported variants.",
                source_context_period=raw["Site Period"],
                source_sample_date=raw["Sample date"],
                sample_metadata_source=locator(meta),
                published_quality=quality,
                branch_calls=calls[individual],
                relatedness=related,
                independent_lineage=False,
            )
        )
    for row in rows:
        if row["source_id"] != "A19_Cell2023_TableS1.xlsx":
            continue
        raw = row["raw"]
        individual = owner[rowkey(row)]
        lo, hi, mean = [raw[k] for k in ["age_early", "age_late", "age_mid"]]
        assert all(number(v) for v in [lo, hi, mean]) and lo <= hi
        direct = raw["Directly dated C14 *"]
        assert direct is False
        flag = raw["used in population genetic analyses"]
        status = population_analysis_status(flag, row["source_id"], "used in population genetic analyses")
        retained = None if status == "unknown" else status == "included"
        observations.append(
            dict(
                individual_id=individual,
                reported_id=raw["id"],
                source_row=locator(row),
                source_sha256=sources[row["source_id"]]["sha256"],
                site=raw["site_minor"],
                country=raw["country"],
                latitude=raw["lat"],
                longitude=raw["long"],
                sex=raw["rYsexKaryotype"],
                date_variants=[
                    dict(
                        name="published_context_range",
                        calendar_bounds=[lo, hi],
                        source_midpoint=mean,
                        confidence_level=None,
                        source_columns=["age_early", "age_late", "age_mid"],
                        **temporal(lo, hi, mean)
                    )
                ],
                date_basis="Published contextual range; explicitly not directly radiocarbon dated.",
                source_population_analysis_retained=retained,
                source_population_analysis_flag=flag,
                branch_calls=calls[individual],
                relatedness=[
                    r for r in relationships if individual in r["individual_ids"]
                ],
                independent_lineage=False,
            )
        )
    assert (
        sum(x["source_row"]["source_id"] == "S29_MOESM4.xlsx" for x in observations)
        == 21
    )
    contemporary = [
        x
        for x in observations
        if x["source_row"]["source_id"] == "A19_Cell2023_TableS1.xlsx"
    ]
    assert (
        len(contemporary) == 9
        and sum(x["source_population_analysis_retained"] for x in contemporary) == 7
    )
    original_i1 = [
        x
        for x in contemporary
        if any(
            c["reported_broad_parent"] == "I1"
            and c["source_rows"][0]["source_id"] == "A19_Cell2023_TableS1.xlsx"
            for c in x["branch_calls"]
        )
    ]
    assert len(original_i1) == 4
    retained_i1 = [x for x in original_i1 if x["source_population_analysis_retained"]]
    assert (
        len(retained_i1) == 3
        and len({(x["latitude"], x["longitude"]) for x in retained_i1}) == 2
    )
    neo220 = next(x for x in observations if x["reported_id"] == "NEO220")
    neo223 = next(x for x in observations if x["reported_id"] == "NEO223")
    assert neo220["date_variants"][0]["calendar_bounds"] == [-2137, -1940] and neo220[
        "date_variants"
    ][1]["calendar_bounds"] == [-2140, -1932]
    assert neo223["date_variants"][0]["calendar_bounds"] == [-2134, -1916] and neo223[
        "date_variants"
    ][1]["calendar_bounds"] == [-2135, -1900]
    assert any(
        "lowcov" in str(q["flags"].get("Flag")) for q in neo223["published_quality"]
    )
    # All normalized scientific date/coordinate/quality values point to verified
    # original source rows. Other studies remain in their separate dated layer.
    result = dict(
        schema="ancient-original-date-qualification-1",
        inputs={
            n: sha(data / n if n == "sources.json" else OUT / n)
            for n in [
                "verified_assertions.json",
                "identity_inventory.json",
                "branch_crosswalk.json",
                "sources.json",
            ]
        },
        script_sha256=sha(Path(__file__)),
        calendar="Signed published calendar years; negative values denote BCE. Original workbook fields retained in verified_assertions.json. No recalibration or reinterpretation of the anomalously signed source BP midpoint field.",
        observations=observations,
        relatedness_assertions=relationships,
        counts=dict(
            original_context_observations=21,
            contemporary_context_observations=9,
            contemporary_source_retained=7,
            contemporary_reported_I1=4,
            contemporary_source_retained_I1=3,
            contemporary_source_retained_I1_coordinates=2,
            relatedness_assertions=len(relationships),
        ),
        scope="Finite original-date and kinship reconciliation for the retained source-selected observations. The separate 135-row McColl layer retains its own published dates and ancestry clusters. Neither observation counts nor source study exclusions certify independent paternal lineages or population frequencies.",
        limits=[
            "NEO223 retains the published low-coverage and low-GP flags; its source qualifications differ from those of NEO220.",
            "The nine contemporary Migration candidates have contextual dates; none becomes a direct radiocarbon anchor.",
            "Reported broad I1 and upstream R1b/R1a labels are not upgraded to terminal modern branches.",
            "Absence of a recorded pairwise relationship is not evidence of unrelatedness.",
            "Shared published individual identity across studies is one observation with multiple source assertions, not independent replication.",
        ],
    )
    save(OUT / "dated_observations.json", result)
    print(json.dumps(result["counts"], indent=2))


def extract_assertions(data, output, selection):
    """Read the declared original rows; never infer missing source values."""
    catalog = read(data / "sources.json")
    entries = {entry["source_id"]: entry for entry in catalog}
    if len(entries) != len(catalog):
        raise ValueError("Duplicate ancient source identifiers")
    paths = {}
    for entry in catalog:
        path = (data / entry["path"]).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError("Source path leaves the declared input directory")
        if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Ancient source checksum differs: " + entry["source_id"])
        paths[entry["source_id"]] = path
    wanted = defaultdict(set)
    for item in selection["retained_assertions"]:
        wanted[item["source_id"], item["sheet"]].add(item["row"])
    header_rows = {
        (item["source_id"], item["sheet"]): item["header_row"]
        for item in selection["headers"]
    }
    if set(wanted) != set(header_rows):
        raise ValueError("Selected sheets differ from their declared header inventory")
    records, checks = {}, []
    books = {}
    try:
        for (source, sheet), rows in sorted(wanted.items()):
            if source not in books:
                books[source] = openpyxl.load_workbook(
                    paths[source], read_only=True, data_only=True
                )
            worksheet = books[source][sheet]
            first = header_rows[source, sheet]
            if min(rows) <= first or (
                worksheet.max_row is not None and max(rows) > worksheet.max_row
            ):
                raise ValueError("Selected row lies outside the source data range")
            head = None
            for number, values in enumerate(
                worksheet.iter_rows(min_row=first, max_row=max(rows), values_only=True),
                first,
            ):
                if number == first:
                    head = headers(values)
                elif number in rows:
                    if not any(value is not None for value in values):
                        raise ValueError("Selected original row is empty")
                    records[source, sheet, number] = {
                        key: values[i] if i < len(values) else None
                        for i, key in enumerate(head)
                    }
            checks.append(
                dict(
                    source_id=source,
                    sheet=sheet,
                    header_row=first,
                    unique_selected_rows=len(rows),
                )
            )
    finally:
        for workbook in books.values():
            workbook.close()
    if len(records) != sum(map(len, wanted.values())):
        raise ValueError("A selected original row was not reconstructed")
    links = {
        (item["source_id"], item["sheet"], item["row"], item["individual"])
        for item in selection["retained_assertions"]
        if item["individual"]
    }
    assertions = [
        dict(source_id=source, sheet=sheet, row=row, raw=value)
        for (source, sheet, row), value in sorted(records.items())
    ]
    verified = dict(
        assertions=assertions,
        identity_links=[
            dict(source_id=source, sheet=sheet, row=row, retained_identity=individual)
            for source, sheet, row, individual in sorted(links)
        ],
        identity_status="Retained exact identifiers and published aliases; identity-link agreement and final deduplication are checked separately. No independent-lineage count.",
    )
    save(output / "verified_assertions.json", verified)
    receipt = dict(
        source_count=len(catalog),
        retained_assertions=len(selection["retained_assertions"]),
        unique_source_rows=len(assertions),
        identity_links=len(links),
        headers=checks,
        selection_sha256=sha(data / "selection.json"),
        catalog_sha256=sha(data / "sources.json"),
        script_sha256=sha(Path(__file__)),
        reader="Published workbook values with cached formula results. Frozen header rows; duplicate headings receive numbered suffixes and unheaded columns retain positional names. Missing source values remain missing.",
    )
    save(output / "source_verification.json", receipt)
    print(
        "Ancient sources",
        len(catalog),
        "selected assertions",
        len(selection["retained_assertions"]),
        "unique rows",
        len(assertions),
        flush=True,
    )



def reconciled_plot_branch(person):
    """A plotted call follows the reconciled person, never a union of conflicts."""
    if not person or person["path_conflict"] or person["broad_parent_conflict"]:
        return "other_or_unresolved"
    targets=set(person["qualified_targets"])
    return next((t for t in ["R1b_U106","R1b_P312","R1a_Z284","I1_L22","I1_Z58","I1"]
                 if t in targets),
                "I1" if person["reported_broad_parents"]==["I1"] else "other_or_unresolved")


def build_eligibility(data, output, selection):
    """Eligibility-only census of the retained inventory; no landscape outcomes."""
    identity=read(output/"identity_inventory.json")
    branches=read(output/"branch_crosswalk.json")
    dates=read(output/"dated_observations.json")
    raw={rowkey(r):r["raw"] for r in read(output/"verified_assertions.json")["assertions"]}
    people={r["individual_id"]:r for r in branches["individuals"]}
    calls=defaultdict(list)
    for r in branches["calls"]:calls[r["individual_id"]].append(r)
    appraisals=defaultdict(list)
    for r in selection["qualification"].get("source_appraisals",[]):
        appraisals[r["individual_id"]].append(r)
    observed={}
    for source in identity["McColl_observations"]:
        r=dict(source)
        r.update(source_row=locator(source),branch_calls=calls[r["individual_id"]],
                 date_variants=[dict(name="McColl_published",calendar_bounds=r["calendar_bounds"],source_midpoint=r["mean_year"])],
                 date_basis=r["date_type"],source_layer="McColl clustered",
                 source_population_analysis_retained=True)
        observed[r["individual_id"]]=r
    for source in dates["observations"]:
        iid=source["individual_id"]
        if iid in observed:
            observed[iid]["date_variants"].extend(source["date_variants"])
            observed[iid]["original_context"]=source
        else:
            r=dict(source);v=r["date_variants"][0] if r["date_variants"] else {}
            r.update(calendar_bounds=v.get("calendar_bounds"),mean_year=v.get("source_midpoint"),
                     published_id=r["reported_id"],source_layer="Original study",ancestry_cluster=None)
            observed[iid]=r
    records=[]
    for person in identity["individuals"]:
        iid=person["individual_id"];r=observed.get(iid)
        source_quality=[];source_status=[];kinship=[];flags=[];identity_flags=[];metadata_labels=[]
        for key in person["source_rows"]:
            rr=raw[tuple(key)]
            quality={k:v for k,v in rr.items() if any(t in k.lower() for t in
                     ["coverage","doc","contam","damage","average gp","gpavg","flag"])}
            if quality:source_quality.append(dict(source_row=list(key),fields=quality))
            for k in ["flag","flag__2","Flag"]:
                value=str(rr.get(k) or "").strip()
                if value.lower() not in {"","0","0.0","...","na","nan","none"}:
                    categories=source_flag_categories(value, key[0])
                    entry=dict(source_row=list(key),field=k,value=value,categories=categories)
                    if set(categories) & {"low_coverage", "genotype_probability", "mitochondrial_contamination", "call_ambiguity"}:flags.append(entry)
                    if "kinship" in categories:kinship.append(entry)
                    if "sample_identity" in categories:identity_flags.append(entry)
            for k in ["Imputed & passing all filters","used in population genetic analyses"]:
                if k in rr:
                    status=population_analysis_status(rr[k], key[0], k)
                    source_status.append(dict(source_row=list(key),field=k,value=rr[k],status=status,
                        excluded=None if status == "unknown" else status == "excluded"))
                    if key[0] == "A19_Cell2023_TableS1.xlsx" and re.search(r"\brelated\b", str(rr[k]), re.I):
                        kinship.append(dict(source_row=list(key),field=k,value=rr[k],categories=["kinship"]))
            for k in ["Relatedness","Relatedness and notes"]:
                if rr.get(k):kinship.append(dict(source_row=list(key),field=k,value=rr[k]))
            if str(key[1]).startswith("S1.1"):
                metadata_labels.append(dict(source_row=list(key),reported_Y=rr.get("hgYMinor"),
                    published_path=rr.get("root_to_tip_path"),disposition="Retained metadata assertion; not promoted by this plotting adapter to an independently qualified call."))
        date_flags=[]
        if r:
            for v in r["date_variants"]:
                b=v.get("calendar_bounds");m=v.get("source_midpoint")
                if b and all(number(z) for z in b) and number(m) and not b[0]<=m<=b[1]:
                    date_flags.append(dict(variant=v["name"],code="midpoint_outside_reported_bounds",bounds=b,midpoint=m,
                        disposition="Source fields are retained; period overlap follows the reported bounds rather than the inconsistent midpoint."))
        branch=reconciled_plot_branch(people.get(iid))
        quality_caveat=bool(flags or appraisals[iid])
        excluded=any(x['excluded'] for x in source_status)
        questions={}
        for period,bounds in identity["period_windows"].items():
            reasons=[]
            if r is None:reasons.append("No observation in the declared dated plotting adapters")
            elif not all(number(r.get(k)) for k in ["latitude","longitude"]):reasons.append("No qualified coordinate pair")
            if not r or not any(overlaps_period(v.get("calendar_bounds"),*bounds) for v in r["date_variants"]):
                reasons.append("No retained date variant overlaps this period")
            descriptive=not reasons
            if branch=="other_or_unresolved":reasons.append("No reconciled focal paternal branch")
            if quality_caveat:reasons.append("Published quality or original-call qualification requires adjudication")
            if date_flags:reasons.append("Unresolved source-date inconsistency")
            if excluded:reasons.append("Explicit exclusion in at least one source population-analysis field")
            if r and "context" in str(r.get("date_basis","")).lower():reasons.append("Context date rather than direct sample date")
            questions[period]=dict(descriptive_plot_eligible=descriptive,
                direct_dated_focal_screen_eligible=not reasons,reasons=reasons,
                independent_validation_eligible=False,
                independence_reason="A distinct identity or coordinate is not an independently verified paternal lineage or test sample. Source selection and site/kinship dependence require an external sampling design.")
        item=dict(individual_id=iid,aliases=person["aliases"],plot_branch=branch,
                  source_quality=source_quality,source_population_status=source_status,
                  reported_relatedness=kinship,source_identity_flags=identity_flags,quality_flags=flags,source_appraisals=appraisals[iid],
                  date_flags=date_flags,metadata_published_labels=metadata_labels,
                  quality_caveat=quality_caveat,any_source_population_exclusion=excluded,
                  questions=questions,has_plotted_observation=r is not None)
        records.append(item)
        if r:
            r.update(plot_branch=branch,independent_paternal_lineage=False,
                     structured_qualification=item,
                     quality_note="Published assignments with source-specific quality, date, population-analysis and relatedness qualifications; no new read-level validation.")
    census=[]
    for period in identity["period_windows"]:
        for scope in ["descriptive_plot_eligible","direct_dated_focal_screen_eligible"]:
            included=[r for r in records if r['questions'][period][scope]]
            bybranch=defaultdict(list)
            for item in included:bybranch[item['plot_branch']].append(observed[item['individual_id']])
            census.append(dict(period=period,scope=scope,identities=len(included),branches={k:dict(
                identities=len(v),distinct_coordinate_pairs=len({(a['longitude'],a['latitude']) for a in v}),
                individual_ids=[a['individual_id'] for a in v]) for k,v in sorted(bybranch.items())}))
    save(output/"eligibility.json",dict(schema="ancient-question-eligibility-2",
        inputs={n:sha(output/n) for n in ["identity_inventory.json","branch_crosswalk.json","dated_observations.json","verified_assertions.json"]},
        selection_sha256=sha(data/"selection.json"),script_sha256=sha(Path(__file__)),
        policy="No universal coverage cutoff. Published flags, original-call appraisals, source population exclusions, date inconsistencies and kinship are separate. Missing status is unknown, not passed. The strict direct-dated screen is an eligibility census, not independent validation. No archaeological landscape scores were used.",
        scope="All retained identities; a source-selected collection, not an exhaustive ancient census. Metadata-only identities remain visible with reasons. Period windows can reuse the same people.",
        individuals=records,observations=list(observed.values()),census=census))
    print("Eligibility census",len(records),"identities;",len(observed),"plotted observations")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ancient"))
    parser.add_argument("--output", type=Path, default=Path("build/ancient"))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.data.resolve()):
        parser.error("Write derived evidence outside the source directory")
    if (args.output / "dated_observations.json").exists():
        parser.error(
            "A completed ancient inventory exists; use a fresh output directory"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    selection = read(args.data / "selection.json")
    extract_assertions(args.data, args.output, selection)
    build_identity(args.data, args.output, selection)
    build_branches(args.data, args.output, selection)
    build_dates(args.data, args.output)
    build_eligibility(args.data, args.output, selection)


if __name__ == "__main__":
    main()

"""Sample-defined paternal label units and relatedness blocks."""

import collections
import functools
import hashlib


def sha(value):
    return hashlib.sha256(value).hexdigest()


def construct_units(out, tree, spec, specification_sha256, source_name):
    """Group non-nested source labels, documented families and alternative localities.

    Mutates the supplied assertion object. The caller provides the frozen tree,
    grouping specification and its verified source identity. Relatedness blocks
    are sample-defined safeguards, not independently verified genealogies.
    """
    records = out["records"]
    nodes = {str(k): v for k, v in tree["allNodes"].items()}
    for n in tree["roots"]:
        nodes.setdefault(str(n["haplogroupId"]), n)

    @functools.lru_cache(None)
    def path(k):
        a = [k]
        while str(nodes[k].get("parentId")) in nodes:
            k = str(nodes[k]["parentId"])
            assert k not in a
            a.append(k)
        return a

    candidates = [r for r in records if r["reconciled_screen_candidate"]]
    represented = {r["branch_node_id"] for r in candidates}
    nested = set()
    for k in represented:
        nested.update(set(path(k)[1:]) & represented)
    leaves = represented - nested
    selected = [r for r in candidates if r["branch_node_id"] in leaves]
    support = collections.Counter(a for k in leaves for a in path(k))
    block = {k: next(a for a in reversed(path(k)) if support[a] <= 20) for k in leaves}

    class Union:
        def __init__(self, keys):
            self.p = {k: k for k in keys}

        def find(self, k):
            while self.p[k] != k:
                self.p[k] = self.p[self.p[k]]
                k = self.p[k]
            return k

        def join(self, a, b):
            x, y = sorted([self.find(a), self.find(b)])
            self.p[y] = x

    units_union = Union(leaves)
    blocks_union = Union(set(block.values()))
    families = collections.defaultdict(set)
    proxies = collections.defaultdict(set)
    for r in selected:
        if r["documented_family_cluster"]:
            families[r["documented_family_cluster"]].add(r["branch_node_id"])
        if r["family_proxy_id"]:
            proxies[r["family_proxy_id"]].add(r["branch_node_id"])
    for group in families.values():
        ks = sorted(group)
        for k in ks[1:]:
            units_union.join(ks[0], k)
    for group in list(families.values()) + list(proxies.values()):
        ks = sorted(group)
        for k in ks[1:]:
            blocks_union.join(block[ks[0]], block[k])
    positions = {}
    groups = collections.defaultdict(list)
    for r in selected:
        groups[units_union.find(r["branch_node_id"])].append(r)
    units = []
    for key, rr in sorted(groups.items()):
        pp = collections.defaultdict(list)
        for r in rr:
            pp[(r["source_lat"], r["source_lon"])].append(r["record_id"])
        localities = []
        for (lat, lon), ids in sorted(pp.items()):
            pid = (
                "POS-"
                + sha((format(lat, ".12g") + "," + format(lon, ".12g")).encode())[:16]
            )
            seed = spec["spatial"]["seed"] + ":" + pid
            positions[pid] = dict(
                position_id=pid,
                lat=lat,
                lon=lon,
                empirical_uncertainty_km=None,
                scenario_radii_km=[0, 25, 50],
                draw_seed=seed,
                draws=64,
                distribution="uniform_area_geodesic_disk",
                boundary_truncation=False,
            )
            localities.append(
                dict(position_id=pid, weight=1 / len(pp), record_ids=sorted(ids))
            )
        ks = sorted({r["branch_node_id"] for r in rr})
        uid = "U-" + key
        units.append(
            dict(
                unit_id=uid,
                branch_node_ids=ks,
                canonical_branches=[nodes[k]["name"] for k in ks],
                dependence_block="B-" + blocks_union.find(block[key]),
                record_ids=sorted(r["record_id"] for r in rr),
                localities=localities,
                total_likelihood_weight=1,
                curated_record_ids=sorted(
                    r["record_id"] for r in rr if r["curated_exact_kit_coverage"]
                ),
                outcome_rule="All member branches must map to one declared comparison category; otherwise withhold unit for that contrast",
            )
        )
        for r in rr:
            r["analysis_unit_id"] = uid
    selected_ids = {r["record_id"] for r in selected}
    for r in records:
        r["modelling_ready"] = r["record_id"] in selected_ids
        r["genetic_release_disposition"] = (
            "approved_conditional_regional"
            if r["modelling_ready"]
            else (
                "retained_upstream_resolution"
                if r["reconciled_screen_candidate"]
                else "withheld_source_screen"
            )
        )
        r["empirical_coordinate_uncertainty_km"] = r["coordinate_uncertainty_km"]
    out["schema_version"] = "sclara-sweden-ydna-6"
    out["release_status"] = (
        "REGIONAL_GENETIC_INPUTS_APPROVED_CONDITIONAL_ON_SPECIFICATION"
    )
    out["release_specification"] = dict(
        version=spec["version"],
        sha256=specification_sha256,
        path=source_name + ":decisions/broad_release_specification.json",
        scope=spec["approved_scope"],
    )
    out["counts"].update(
        approved_source_records=len(selected),
        primary_non_nested_labels=len(leaves),
        primary_analysis_units=len(units),
        dependence_blocks=len({u["dependence_block"] for u in units}),
        positions=len(positions),
        upstream_records_retained_without_primary_weight=len(candidates)
        - len(selected),
        curated_primary_records=sum(r["curated_exact_kit_coverage"] for r in selected),
        modelling_ready=len(selected),
    )
    out.pop("candidate_label_units", None)
    out["analysis_units"] = units
    out["spatial_positions"] = list(positions.values())
    out["modelling_records"] = sorted(selected_ids)
    out["lineage_paths"] = {k: path(k) for k in sorted(leaves)}
    out["limitations"] = [
        "The inputs meet the preparation requirements for the specified regional analyses; this assessment does not certify coordinate accuracy or historical residence.",
        "Non-nested branches remain related; the dependence blocks and tree structure therefore matter for inference.",
        "Branch selection and dependence blocks are defined by the frozen sample; they do not identify lineages that were independent at a particular historical date.",
        "Regional spatial sensitivity analyses and checks that account for dependence blocks form part of model assessment.",
    ]
    assert len({r for u in units for r in u["record_ids"]}) == len(selected_ids)
    assert sum(len(u["record_ids"]) for u in units) == len(selected_ids)
    assert all(
        abs(sum(x["weight"] for x in u["localities"]) - 1) < 1e-12 for u in units
    )
    assert all(not (set(path(k)[1:]) & leaves) for k in leaves)
    out["schema_version"] = "sclara-sweden-genetics-1"
    out["limitations"] += [
        "Source birth/death dates do not date the map location.",
        "Exact-kit source overlap is not independent corroboration.",
        "Curated source coverage is geographically selected and does not validate every pedigree.",
        "Analysis units and dependence blocks are not certified independent genealogies.",
    ]
    return out

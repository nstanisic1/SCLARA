"""Normalize the retained Swedish paternal and archaeological sources.

Source archives carry original observations, documentary decisions and member
checksums. This preparation step preserves excluded assertions and alternative
localities. It neither fits lineage models nor changes archaeological profiles.
"""

from pathlib import Path
import argparse
import tempfile


def prepare_paternal(source_path, output_path, work_dir=None):
    """Reproduce the frozen paternal source normalization."""
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    if source_path == output_path or not source_path.is_file():
        raise ValueError(
            "A retained source archive and distinct output path are required"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SOURCE = source_path
    import collections, functools, hashlib, io, json, math, random, re, sys, unicodedata, zipfile

    from lxml import html

    from shapely.geometry import Polygon, Point

    SUPPORT = Path(".")

    REF = Path("references")

    YEAR = r"(?:1[0-9]{3}|20[0-2][0-9])"

    def resource(path):
        name = Path(path).as_posix()
        with zipfile.ZipFile(SOURCE) as z:
            manifest = json.loads(z.read("manifest.json"))
            item = next(v for v in manifest["members"] if v["path"] == name)
            value = z.read(name)
        assert hashlib.sha256(value).hexdigest() == item["sha256"], name
        return value

    def read(path):
        return json.loads(resource(path))

    def source_bytes(name):
        with zipfile.ZipFile(io.BytesIO(resource("observations.zip"))) as z:
            manifest = json.loads(z.read("manifest.json"))
            expected = next(
                v["sha256"] for v in manifest["sources"] if v["path"] == name
            )
            data = z.read(name)
        assert hashlib.sha256(data).hexdigest() == expected, name
        return data

    def norm(x):
        s = unicodedata.normalize("NFKD", str(x or "")).casefold()
        return " ".join(
            re.sub(
                r"[^a-z0-9]+",
                " ",
                "".join(c for c in s if not unicodedata.combining(c)),
            ).split()
        )

    def sha(b):
        return hashlib.sha256(b).hexdigest()

    def distance(a, b, c, d):
        p1, p2 = math.radians(a), math.radians(c)
        v = (
            math.sin((p2 - p1) / 2) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
        )
        return 6371.0088 * 2 * math.asin(min(1, math.sqrt(v)))

    def dates(s, max_lifespan=110):
        s = unicodedata.normalize("NFKC", s or "")
        patterns = {
            "birth": r"(?<!\w)(?:b(?:orn)?|f|född|fodd|föddes)\s*[.:]?\s*(?:(?:ca|cirka|c|abt|about|omkring)\.?\s*)?("
            + YEAR
            + r")(?!\d)",
            "death": r"(?<!\w)(?:d(?:ied)?|död|dod|avliden)\s*[.:]?\s*(?:(?:ca|cirka|c|abt|about)\.?\s*)?("
            + YEAR
            + r")(?!\d)",
        }
        out = {
            k: sorted(set(int(m.group(1)) for m in re.finditer(v, s, re.I)))
            for k, v in patterns.items()
        }
        out["year_tokens"] = sorted(
            set(int(x) for x in re.findall(r"(?<!\d)(" + YEAR + r")(?!\d)", s))
        )
        out["approximate_or_uncertain"] = bool(
            re.search(
                r"\?|\b(?:ca|cirka|abt|about|omkring|before|after|före|efter)\b",
                s,
                re.I,
            )
        )
        out["approximate_or_uncertain"] |= bool(
            re.search(
                r"(?<![a-z])(?:ca|cirka|c|abt|about|omkring|est|circa)\.?\s*(?="
                + YEAR
                + r")",
                s,
                re.I,
            )
        )
        out["approximate_or_uncertain"] |= bool(
            re.search(
                r"(?<!\w)(?:b(?:orn)?|f|född|fodd|d(?:ied)?|död)\s*[.:]?\s*(?:ca|cirka|c|abt|about|omkring|est|circa)\.?\s*"
                + YEAR,
                s,
                re.I,
            )
        )
        out["unlabelled_range"] = re.findall(
            "(" + YEAR + r")\s*[-–—]\s*(" + YEAR + ")", s
        )
        out["explicit_birth_year"] = out["birth"][0] if len(out["birth"]) == 1 else None
        out["explicit_death_year"] = out["death"][0] if len(out["death"]) == 1 else None
        # A labelled range is not an exact event year; ISO YYYY-MM-DD remains a date.
        out["labelled_range_expressions"] = {}
        for role, pattern in patterns.items():
            ranges = []
            for match in re.finditer(pattern, s, re.I):
                tail = s[match.end() :]
                suffix = re.match(r"\s*[-–—]\s*(\d{4}|\d{2})(?!\d)", tail)
                iso = re.match(r"-\d{2}-\d{2}(?!\d)", tail)
                if suffix and not iso:
                    ranges.append(s[match.start() : match.end() + suffix.end()])
            if ranges:
                out["labelled_range_expressions"][role] = ranges
                out["explicit_" + role + "_year"] = None
                out["approximate_or_uncertain"] = True

        out["chronology_conflict"] = len(out["birth"]) > 1 or len(out["death"]) > 1
        if out["explicit_birth_year"] and out["explicit_death_year"]:
            age = out["explicit_death_year"] - out["explicit_birth_year"]
            out["chronology_conflict"] |= not 0 <= age <= max_lifespan
        out["residence_date"] = None
        out["evidence_level"] = "public_source_assertion_not_document_verified"
        return out

    def validate_adjudications(manual, manifest, record_ids, raw_hash):
        evidence = {e["id"]: e for e in manifest if "sha256" in e}
        for e in evidence.values():
            if sha(resource(e["path"])) != e["sha256"]:
                raise ValueError("Evidence file missing or changed: " + e["id"])
        by_record = collections.defaultdict(list)
        seen = set()
        for a in manual["decisions"]:
            required = [
                "decision_id",
                "record_ids",
                "decision",
                "reason",
                "evidence_ids",
                "evidence_locator",
                "reviewed_on",
                "raw_snapshot_sha256",
            ]
            if any(not a.get(k) for k in required):
                raise ValueError("Incomplete adjudication")
            if a["decision_id"] in seen:
                raise ValueError("Duplicate adjudication ID")
            seen.add(a["decision_id"])
            if a["raw_snapshot_sha256"] != raw_hash:
                raise ValueError("Adjudication belongs to another source snapshot")
            if a["decision"] not in [
                "withhold",
                "annotate",
                "retain_provisional",
                "select_ancestor_event",
            ]:
                raise ValueError(
                    "Unsupported decision; release approval requires a separately validated country specification"
                )
            if any(
                e != "raw_snapshot" and e not in evidence for e in a["evidence_ids"]
            ):
                raise ValueError("Unknown evidence ID")
            if not set(a["record_ids"]) <= record_ids:
                raise ValueError("Unknown record in adjudication")
            if a["decision"] == "select_ancestor_event":
                if (
                    not isinstance(a.get("selected_birth_year"), int)
                    or not a.get("ancestor_name_key")
                    or a["evidence_ids"] == ["raw_snapshot"]
                ):
                    raise ValueError(
                        "Event selection needs external evidence, focal ancestor and year"
                    )
            for k in a["record_ids"]:
                by_record[k].append(a)
        for k, aa in by_record.items():
            if sum(a["decision"] == "select_ancestor_event" for a in aa) > 1:
                raise ValueError("Multiple event selections for " + k)
        return by_record

    def release_outcome(gates, outside_scope=False):
        primary = [
            "primary_pedigree",
            "primary_origin_event",
            "spatial_uncertainty",
            "branch",
            "dependence_representation",
        ]
        sensitivity = [
            "sensitivity_pedigree",
            "sensitivity_origin_event",
            "spatial_uncertainty",
            "branch",
            "dependence_representation",
        ]
        if outside_scope:
            return "excluded"
        if all(gates[k]["status"] == "pass" for k in primary):
            return "approved"
        if all(gates[k]["status"] == "pass" for k in sensitivity):
            return "sensitivity_only"
        return "unresolved"

    def parse_ftdna():
        rules = read(SUPPORT / "decisions/cleaning_rules.json")
        manual = read(SUPPORT / "decisions/manual_decisions.json")
        date_rules = rules["date_rules"]
        place_rules = rules["place_rules"]
        parse_dates = functools.partial(
            dates, max_lifespan=date_rules["max_plausible_lifespan"]
        )

        with zipfile.ZipFile(io.BytesIO(source_bytes("ftdna_sweden.zip"))) as z:
            manifest = json.loads(z.read("provenance.json"))
            for m in manifest["members"]:
                assert sha(z.read(m["path"])) == m["sha256"]
            maps = json.loads(z.read("raw/map_data.json"))
            # This large HTML document is a hash-verified frozen source, parsed offline.
            parser = html.HTMLParser(huge_tree=True, no_network=True)
            doc = html.fromstring(
                z.read("raw/overview_all.html").decode("utf8"), parser=parser
            )
            resource_errors = [
                e
                for e in parser.error_log
                if e.level_name == "FATAL" or "Resource limit" in e.message
            ]
            if resource_errors:
                raise ValueError(
                    "Incomplete frozen HTML parse: " + str(resource_errors)
                )
        table = []
        for tr in doc.xpath("//tr"):
            cells = tr.xpath("./td")
            v = [" ".join(c.text_content().split()) for c in cells]
            if len(v) >= 6 and v[0].isdigit():
                table.append(
                    dict(
                        kit=v[1],
                        ancestor=v[2],
                        country=v[3],
                        branch=v[4],
                        test=v[5],
                        haplo_css=";".join(cells[4].xpath(".//span/@class")),
                    )
                )
        mi = {str(x["kit"]): x for x in maps}
        ti = {str(x["kit"]): x for x in table}
        assert len(mi) == len(maps) == 5246 and len(ti) == len(table) == 4623
        evidence_manifest = read(SUPPORT / "decisions/evidence_manifest.json")
        adjudications = validate_adjudications(
            manual,
            evidence_manifest,
            set(mi) | set(ti),
            sha(source_bytes("ftdna_sweden.zip")),
        )
        tree = read(REF / "ftdna_tree.json")
        nodes = {str(k): v for k, v in tree["allNodes"].items()}
        for n in tree["roots"]:
            nodes.setdefault(str(n["haplogroupId"]), n)
        exact = collections.defaultdict(set)
        aliases = collections.defaultdict(set)
        for k, n in nodes.items():
            exact[n["name"].upper()].add(k)
            for v in n.get("variants", []):
                aliases[(n.get("root", "") + "-" + v["variant"]).upper()].add(k)

        @functools.lru_cache(None)
        def ancestors(k):
            out = []
            seen = {k}
            cur = nodes[k]
            while str(cur.get("parentId")) in nodes:
                par = str(cur["parentId"])
                if par in seen:
                    raise ValueError("Cycle in tree")
                seen.add(par)
                out.append(par)
                cur = nodes[par]
            return tuple(out)

        def branch(s):
            s = (s or "").strip().upper()
            hit = exact.get(s, set())
            how = "exact_node_name"
            if not hit:
                hit = aliases.get(s, set())
                how = "unique_equivalent_variant"
            if len(hit) != 1:
                return {
                    "source_label": s,
                    "status": (
                        "missing" if s in ["", "-"] else "unresolved_or_ambiguous"
                    ),
                    "node_id": None,
                }
            k = next(iter(hit))
            n = nodes[k]
            return dict(
                source_label=s,
                status="resolved",
                match_method=how,
                node_id=k,
                canonical_name=n["name"],
                parent_node_id=n.get("parentId"),
                tree_leaf=not n.get("children"),
                ancestral_node_ids=list(ancestors(k)),
                root=n.get("root"),
            )

        parishes = read(REF / "isof_parishes.json")
        provinces = {str(x["id"]): x for x in read(REF / "isof_provinces.json")}
        placeindex = collections.defaultdict(list)
        for x in parishes:
            if x["country_code_2"] != "SE" or not x["latitude"] or not x["longitude"]:
                continue
            base = re.sub(r"\s+(?:sn|stad|kn|fs|socken)$", "", norm(x["name"]))
            names = {base}
            if base.endswith("s"):
                names.add(base[:-1])
            record = dict(
                id="isof:" + str(x["id"]),
                name=x["name"],
                lat=x["latitude"],
                lon=x["longitude"],
                kind="parish_1951_reference",
                authority="Isof",
                province=provinces.get(str(x["province_id"]), {}).get("name"),
                county_id=x["county_id"],
            )
            for name in names:
                placeindex[name].append(record)
        with zipfile.ZipFile(io.BytesIO(resource(REF / "geonames_SE.zip"))) as z:
            for line in z.read("SE.txt").decode("utf8").splitlines():
                f = line.split("\t")
                if len(f) < 19 or f[6] != "P":
                    continue
                rec = dict(
                    id="geonames:" + f[0],
                    name=f[1],
                    lat=float(f[4]),
                    lon=float(f[5]),
                    kind=f[7],
                    authority="GeoNames",
                    province=None,
                    county_id=None,
                )
                for name in set(norm(x) for x in [f[1], f[2]] + f[3].split(",") if x):
                    placeindex[name].append(rec)
        boundaries = read(REF / "country_boundaries.geojson")
        sw = next(
            x for x in boundaries["features"] if x["properties"]["ADMIN"] == "Sweden"
        )
        polys = [Polygon(c[0], c[1:]) for c in sw["geometry"]["coordinates"]]

        def locate(m):
            if not m:
                return {"status": "no_map_record"}
            lat, lon = m.get("lat"), m.get("lng")
            text = m.get("savedLoc") or ""
            if (
                not isinstance(lat, (int, float))
                or not isinstance(lon, (int, float))
                or not -90 <= lat <= 90
                or not -180 <= lon <= 180
            ):
                return {"status": "invalid_coordinate"}
            inside = any(g.covers(Point(lon, lat)) for g in polys)
            first = text.split(",")[0]
            first = re.sub(r"^\s*\d{3}\s*\d{2}\s*", "", first)
            key = norm(first)
            key = re.sub(r"\s+(?:parish|socken|sn)$", "", key)
            base = dict(
                source_text=text,
                source_lat=lat,
                source_lon=lon,
                inside_generalized_sweden=inside,
                lookup_key=key,
                coordinate_uncertainty_km=None,
                historical_boundary_validated=False,
            )
            if not inside:
                return dict(base, status="outside_generalized_boundary_review")
            if (
                not key
                or key in ["sweden", "sverige", "suecia", "suede", "schweden"]
                or re.search(r"\b(?:county|lan|municipality|kommun|region)\b", key)
            ):
                return dict(base, status="missing_or_administratively_broad")
            hits = []
            seen = set()
            for g in placeindex.get(key, []):
                if g["id"] in seen:
                    continue
                seen.add(g["id"])
                dist = distance(lat, lon, g["lat"], g["lon"])
                if dist <= place_rules["max_pin_distance_km"]:
                    hits.append(dict(g, pin_distance_km=round(dist, 4)))
            hits.sort(key=lambda x: (x["pin_distance_km"], x["id"]))
            if not hits:
                return dict(base, status="name_pin_not_corroborated")
            distinct = [
                h
                for h in hits
                if distance(hits[0]["lat"], hits[0]["lon"], h["lat"], h["lon"])
                > place_rules["colocated_match_distance_km"]
            ]
            if distinct:
                return dict(base, status="ambiguous_nearby_matches", matches=hits)
            preferred = next((h for h in hits if h["authority"] == "Isof"), hits[0])
            return dict(
                base,
                status="place_pin_corroborated",
                matched_place=preferred,
                matches=hits,
                evidence_limit="The reference identifies the place and approximate agreement with the map point; it does not establish that this ancestor lived there. The 10 km tolerance is not measured uncertainty.",
            )

        records = []
        for kit in sorted(set(mi) | set(ti)):
            m, t = mi.get(kit), ti.get(kit)
            anc = (m or {}).get("ancestor") or (t or {}).get("ancestor") or ""
            dm = parse_dates((m or {}).get("ancestor", ""))
            dt = parse_dates((t or {}).get("ancestor", ""))
            d = parse_dates(anc)
            original_date = dict(d)
            aa = adjudications.get(kit, [])
            selections = [a for a in aa if a["decision"] == "select_ancestor_event"]
            if selections:
                choice = selections[0]
                year = choice["selected_birth_year"]
                if year not in d["birth"]:
                    raise ValueError(
                        "Selected year absent from source birth assertions"
                    )
                d = dict(
                    d,
                    birth=[year],
                    explicit_birth_year=year,
                    chronology_conflict=False,
                    evidence_level="published_kit_linked_focal_ancestor_event",
                    event_selection_decision_id=choice["decision_id"],
                )
            if any(
                a.get("birth_precision") in ["approximate", "approximate_upper_bound"]
                for a in aa
            ):
                d = dict(
                    d,
                    approximate_or_uncertain=True,
                    adjudicated_birth_precision=next(
                        a["birth_precision"]
                        for a in aa
                        if a.get("birth_precision")
                        in ["approximate", "approximate_upper_bound"]
                    ),
                )
            flags = []
            if m is None:
                flags.append("table_only")
            if t is None:
                flags.append("map_only")
            if m and t and m.get("isConfirmed") and t["haplo_css"] == "haplo1":
                flags.append("confirmation_conflict")
            bm = branch((m or {}).get("deephaplo"))
            bt = branch((t or {}).get("branch"))
            b = bm if m and (m.get("deephaplo") or "") not in ["", "-"] else bt
            if bm["node_id"] and bt["node_id"] and bm["node_id"] != bt["node_id"]:
                flags.append("branch_disagreement_between_views")
            if (
                m
                and t
                and dm["explicit_birth_year"]
                and dt["explicit_birth_year"]
                and dm["explicit_birth_year"] != dt["explicit_birth_year"]
            ):
                flags.append("birth_disagreement_between_views")
            if (
                m
                and t
                and dm["explicit_death_year"]
                and dt["explicit_death_year"]
                and dm["explicit_death_year"] != dt["explicit_death_year"]
            ):
                flags.append("death_disagreement_between_views")
            if d["chronology_conflict"]:
                flags.append("implausible_or_conflicting_dates")
            geo = locate(m)
            if t and t["country"] not in ["Sweden", "", "Unknown Origin"]:
                flags.append("table_country_not_sweden")
            if geo["status"] == "place_pin_corroborated" and geo["matched_place"].get(
                "province"
            ):
                mentions = [
                    x["name"]
                    for x in provinces.values()
                    if norm(x["name"])
                    and re.search(r"\b" + re.escape(norm(x["name"])) + r"\b", norm(anc))
                ]
                if mentions and all(
                    norm(v) != norm(geo["matched_place"]["province"]) for v in mentions
                ):
                    flags.append("ancestor_province_name_requires_event_review")
            birth = d["explicit_birth_year"]
            dateok = (
                birth is not None
                and date_rules["primary_birth_year_min"]
                <= birth
                <= date_rules["primary_birth_year_max"]
                and not d["approximate_or_uncertain"]
                and not d["chronology_conflict"]
            )
            confirmed = bool(m and m.get("isConfirmed") is True)
            hard = {
                "confirmation_conflict",
                "branch_disagreement_between_views",
                "birth_disagreement_between_views",
                "death_disagreement_between_views",
                "implausible_or_conflicting_dates",
                "table_country_not_sweden",
                "ancestor_province_pin_conflict",
            }
            reasons = []
            if geo["status"] != "place_pin_corroborated":
                reasons.append("place:" + geo["status"])
            if not dateok:
                reasons.append("explicit_precise_birth_1500_1850_not_established")
            if b["status"] != "resolved":
                reasons.append("branch_unresolved")
            if not confirmed:
                reasons.append("map_branch_not_confirmed")
            reasons.extend(sorted(set(flags) & hard))
            # Name only is a conservative collision key, never proof of the person's identity.
            namepart = re.split(
                r"(?<!\w)(?:b(?:orn)?|f|född|fodd|d(?:ied)?|död)\s*[.:]?\s*(?=\d|ca\b|cirka\b|c\b|abt\b|about\b|omkring\b)|(?<!\d)"
                + YEAR,
                anc,
                flags=re.I,
            )[0]
            name = norm(re.sub(r"^(?:Mr\.?|Herr)\s+", "", namepart, flags=re.I))
            if len(name.split()) < 2:
                reasons.append("ancestor_name_insufficient")
            if selections:
                name = selections[0]["ancestor_name_key"]
            for a in aa:
                if a["decision"] == "withhold":
                    reasons.append("adjudication:" + a["decision_id"])
            records.append(
                dict(
                    record_id=kit,
                    adjudication_ids=[a["decision_id"] for a in aa],
                    documentary_evidence_status=next(
                        (
                            a["assessment"]
                            for a in aa
                            if a.get("assessment")
                            in [
                                "published_kit_linked_family",
                                "historical_person_event_corroborated",
                            ]
                        ),
                        "source_assertion_only",
                    ),
                    documented_family_cluster=next(
                        (
                            a["documented_cluster"]
                            for a in aa
                            if a.get("documented_cluster")
                        ),
                        None,
                    ),
                    original_date_assessment=original_date,
                    source_map=m,
                    source_table=t,
                    ancestor_name_key=name,
                    source_asserted_ancestor=anc,
                    date_assessment=d,
                    table_date_assessment=dt,
                    origin_assessment=geo,
                    branch_assessment=b,
                    table_branch_assessment=bt,
                    review_flags=flags,
                    candidate_eligible=not reasons,
                    candidate_withholding_reasons=reasons,
                    documentary_pedigree_verified=False,
                    related_lineage_independence_verified=False,
                    modelling_ready=False,
                )
            )
        # Distinct reported tree nodes form a dependency hierarchy, not independent people.
        represented = {
            r["branch_assessment"]["node_id"]
            for r in records
            if r["branch_assessment"]["node_id"]
        }
        ancestor_with_descendants = set()
        for k in represented:
            ancestor_with_descendants.update(set(ancestors(k)) & represented)
        for r in records:
            b = r["branch_assessment"]
            b["represented_descendants_exist"] = (
                b["node_id"] in ancestor_with_descendants
            )
        # Family collision review: require a birth year and corroborated place before comparing.
        families = collections.defaultdict(list)
        for r in records:
            d = r["date_assessment"]
            g = r["origin_assessment"]
            if (
                d["explicit_birth_year"]
                and g["status"] == "place_pin_corroborated"
                and len(r["ancestor_name_key"].split()) >= 2
            ):
                key = (
                    r["ancestor_name_key"],
                    d["explicit_birth_year"],
                    g["matched_place"]["id"],
                )
                families[key].append(r)
        duplicate_groups = []
        for key, rr in sorted(families.items()):
            ks = {
                r["branch_assessment"]["node_id"]
                for r in rr
                if r["branch_assessment"]["node_id"]
            }
            complete = all(r["branch_assessment"]["node_id"] for r in rr)
            major_roots = {nodes[k].get("root") for k in ks}
            linear = all(
                a == b or a in ancestors(b) or b in ancestors(a) for a in ks for b in ks
            )
            conflict = len(major_roots) > 1
            compatible = (linear if complete else None) if not conflict else False
            relation = (
                "cross_major_haplogroup_conflict"
                if conflict
                else (
                    "unresolved_branch_comparison"
                    if not complete
                    else (
                        "same_or_nested_nodes"
                        if linear
                        else "divergent_descendants_require_genealogy"
                    )
                )
            )
            group = "F-" + sha(json.dumps(key).encode())[:14]
            for r in rr:
                r["family_proxy_id"] = group
            if len(rr) > 1:
                duplicate_groups.append(
                    dict(
                        family_proxy_id=group,
                        key=key,
                        record_ids=[r["record_id"] for r in rr],
                        canonical_nodes=sorted(ks),
                        branch_compatible=compatible,
                        branch_relationship=relation,
                        status=(
                            "hard_branch_conflict"
                            if conflict
                            else "provisional_family_review"
                        ),
                        confirmed_genealogical_duplicate=False,
                    )
                )
                if conflict:
                    for r in rr:
                        r["candidate_eligible"] = False
                        r["candidate_withholding_reasons"].append(
                            "family_proxy_branch_conflict"
                        )
        for r in records:
            r.setdefault("family_proxy_id", "K-" + r["record_id"])
        assessment = read("decisions/pedigree_assessment.json")
        for r in records:
            r["genetic_release_outcome"] = "not_assessed"
        index = {r["record_id"]: r for r in records}
        for decision in assessment["decisions"]:
            rr = [index[k] for k in decision["record_ids"]]
            g = dict(decision["gates"])
            branches = [r["branch_assessment"] for r in rr]
            ok = all(
                b["status"] == "resolved"
                and (r["source_map"] or {}).get("isConfirmed") is True
                and not set(r["review_flags"])
                & {"confirmation_conflict", "branch_disagreement_between_views"}
                for r, b in zip(rr, branches)
            )
            ok &= len({b.get("root") for b in branches}) == 1
            g["branch"] = {"status": "pass" if ok else "fail"}
            outcome = release_outcome(g, bool(decision["outside_scope_evidence_ids"]))
            for r in rr:
                r["genetic_release_outcome"] = outcome
        assert len(records) == 6337 and len(index) == 6337
        return records

    def parse_visual(prior_records):
        from lxml import etree
        from shapely.ops import nearest_points
        from shapely.prepared import prep

        def dist(a, b, c, d):
            return distance(a, b, c, d)

        with zipfile.ZipFile(io.BytesIO(source_bytes("visual_dna.zip"))) as z:
            raw = z.read("map_response.xml")
            manifest = json.loads(z.read("manifest.json"))
        assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
        source = [dict(x.attrib) for x in etree.fromstring(raw).findall(".//marker")]
        assert len(source) == manifest["markers"] and len(
            {x["Kit"] for x in source}
        ) == len(source)
        prior = {x["record_id"]: x for x in prior_records}
        boundaries = read(REF / "country_boundaries.geojson")
        geom = next(
            x["geometry"]
            for x in boundaries["features"]
            if x["properties"]["ADMIN"] == "Sweden"
        )
        polys = [Polygon(c[0], c[1:]) for c in geom["coordinates"]]
        prepared = [prep(p) for p in polys]
        tree = read(REF / "ftdna_tree.json")
        nodes = {str(k): v for k, v in tree["allNodes"].items()}
        for n in tree["roots"]:
            nodes.setdefault(str(n["haplogroupId"]), n)
        exact = collections.defaultdict(set)
        aliases = collections.defaultdict(set)
        for k, n in nodes.items():
            exact[n["name"].upper()].add(k)
            for v in n.get("variants", []):
                aliases[(n.get("root", "") + "-" + v["variant"]).upper()].add(k)

        @functools.lru_cache(None)
        def ancestors(k):
            result = []
            seen = {k}
            while str(nodes[k].get("parentId")) in nodes:
                k = str(nodes[k]["parentId"])
                assert k not in seen
                seen.add(k)
                result.append(k)
            return tuple(result)

        def branch(label):
            label = label.strip().upper()
            hit = exact.get(label, set())
            method = "exact"
            if not hit:
                hit = aliases.get(label, set())
                method = "unique_alias"
            if len(hit) != 1:
                return (
                    None,
                    None,
                    "missing" if label in ["", "-"] else "unresolved_in_frozen_tree",
                )
            k = next(iter(hit))
            return k, nodes[k]["name"], method

        sw_words = r"\b(?:sweden|sverige|ruotsi|schweden|suecia|suede|suède)\b|(?:^|,)\s*SE\s*$"
        foreign = r"\b(?:finland|suomi|norway|norge|denmark|danmark|estonia|latvia|russia|germany)\s*$"
        result = []
        for x in source:
            kit = x["Kit"]
            old = prior.get(kit)
            flags = []
            try:
                lat, lon = float(x["Lat"]), float(x["Lng"])
                valid = (
                    math.isfinite(lat)
                    and math.isfinite(lon)
                    and -90 <= lat <= 90
                    and -180 <= lon <= 180
                )
            except (ValueError, TypeError):
                lat = lon = None
                valid = False
            inside = (
                valid
                and 10 <= lon <= 25
                and 55 <= lat <= 70
                and any(p.covers(Point(lon, lat)) for p in prepared)
            )
            try:
                age = float(x["Ancestor"])
                modern = age == 0
                obj = (
                    "modern_coded"
                    if modern
                    else "ancient_or_other_dated" if age > 0 else "unknown"
                )
            except ValueError:
                modern = False
                obj = "unknown"
            swtext = bool(re.search(sw_words, x["Savedloc"], re.I))
            foreigntext = bool(re.search(foreign, x["Savedloc"], re.I))
            boundary_distance = None
            if modern and swtext and valid and not inside:
                pt = Point(lon, lat)
                poly = min(polys, key=lambda p: p.distance(pt))
                q = nearest_points(poly, pt)[0]
                boundary_distance = round(dist(lat, lon, q.y, q.x), 3)
            node, canonical, method = branch(x["YDNA_Deephaplo"])
            confirmed = x["Isconfirmed"] == "1"
            if node and x["YDNA_Basehaplo"] not in ["", "-", nodes[node].get("root")]:
                flags.append("root_disagreement")
            standard_id = bool(re.fullmatch(r"(?:\d+|[A-Z]{1,3}\d+)", kit))
            if not standard_id:
                flags.append("nonstandard_or_surrogate_identifier")
            if inside and foreigntext:
                flags.append("foreign_country_text_at_swedish_pin")
            first = x["Savedloc"].split(",")[0].strip()
            broad = (
                not first
                or bool(re.fullmatch(sw_words, first, re.I))
                or bool(
                    re.search(
                        r"\b(?:county|län|lan|kommun|municipality|region)\b",
                        first,
                        re.I,
                    )
                )
            )
            if broad:
                flags.append("missing_or_admin_broad_locality_text")
            relation = "no_snapshot_record"
            shift = None
            oldinside = False
            if old:
                oldinside = old["origin_assessment"].get(
                    "inside_generalized_sweden", False
                )
                oldnode = old["branch_assessment"].get("node_id")
                if node and oldnode:
                    relation = (
                        "same"
                        if node == oldnode
                        else (
                            "map_more_resolved"
                            if oldnode in ancestors(node)
                            else (
                                "snapshot_more_resolved"
                                if node in ancestors(oldnode)
                                else "divergent"
                            )
                        )
                    )
                else:
                    relation = "unresolved_comparison"
                m = old.get("source_map") or {}
                if (
                    valid
                    and isinstance(m.get("lat"), (int, float))
                    and isinstance(m.get("lng"), (int, float))
                ):
                    shift = round(dist(lat, lon, m["lat"], m["lng"]), 6)
            if relation == "divergent":
                flags.append("branch_divergence_from_snapshot")
            if shift is not None and shift > 10:
                flags.append("coordinate_shift_over_10km")
            broad_candidate = bool(
                modern
                and inside
                and confirmed
                and node
                and "root_disagreement" not in flags
            )
            low_conflict = bool(
                broad_candidate
                and standard_id
                and not foreigntext
                and relation != "divergent"
                and not (shift is not None and shift > 10)
                and not broad
            )
            result.append(
                dict(
                    source=x,
                    record_id=kit,
                    object_class=obj,
                    valid_coordinate=valid,
                    inside_generalized_sweden=inside,
                    swedish_country_text=swtext,
                    foreign_country_text=foreigntext,
                    outside_distance_to_boundary_km=boundary_distance,
                    canonical_node=node,
                    canonical_branch=canonical,
                    branch_resolution=method,
                    confirmed=confirmed,
                    source_id_looks_like_kit=standard_id,
                    snapshot_overlap=old is not None,
                    snapshot_inside_sweden=oldinside,
                    branch_comparison=relation,
                    coordinate_shift_km=shift,
                    prior_origin_status=(
                        old["origin_assessment"]["status"] if old else None
                    ),
                    source_reported_birth_year=(
                        old["original_date_assessment"].get("explicit_birth_year")
                        if old
                        else None
                    ),
                    known_documented_family=(
                        old.get("documented_family_cluster") if old else None
                    ),
                    prior_family_proxy=old.get("family_proxy_id") if old else None,
                    broad_candidate=broad_candidate,
                    low_conflict_candidate=low_conflict,
                    flags=flags,
                )
            )
        return result

    def reconcile(prior_records, visual):
        prior = {r["record_id"]: r for r in prior_records}
        legacy = visual
        lm = {r["record_id"]: r for r in legacy if r["object_class"] == "modern_coded"}
        tree = read(REF / "ftdna_tree.json")
        nodes = {str(k): v for k, v in tree["allNodes"].items()}
        for n in tree["roots"]:
            nodes.setdefault(str(n["haplogroupId"]), n)
        exact = collections.defaultdict(set)
        alias = collections.defaultdict(set)
        for k, n in nodes.items():
            exact[n["name"].upper()].add(k)
            for v in n.get("variants", []):
                alias[(n.get("root", "") + "-" + v["variant"]).upper()].add(k)

        def resolve(label):
            label = re.sub(r"^([A-Z])[0-9a-z]+-", r"\1-", label.strip()).upper()
            hits = exact.get(label) or alias.get(label, set())
            return next(iter(hits)) if len(hits) == 1 else None

        @functools.lru_cache(None)
        def ancestry(k):
            out = []
            while str(nodes[k].get("parentId")) in nodes:
                k = str(nodes[k]["parentId"])
                assert k not in out
                out.append(k)
            return out

        def relation(a, b):
            if not a or not b:
                return "unresolved"
            return (
                "same"
                if a == b
                else (
                    "primary_more_resolved"
                    if b in ancestry(a)
                    else "other_more_resolved" if a in ancestry(b) else "divergent"
                )
            )

        sources = []
        curated = collections.defaultdict(list)
        kit_pattern = r"\b(?:IN\d{3,}|[A-Z]{1,2}\d{4,}|\d{5,7})\b"
        with zipfile.ZipFile(io.BytesIO(source_bytes("bostrom.zip"))) as z:
            manifest = {m["path"]: m for m in json.loads(z.read("manifest.json"))}
            for filename in sorted(
                n for n in z.namelist() if re.fullmatch(r"bostrom_sheet00[2-7]\.htm", n)
            ):
                raw = z.read(filename)
                assert sha(raw) == manifest[filename]["sha256"]
                doc = html.fromstring(raw)
                rows = doc.xpath("//tr")
                grid = []
                # Track physical spreadsheet columns, including colspan/rowspan. Only cells
                # starting at the same column are eligible for the bounded branch adjacency rule.
                occupied = {}
                for ri, tr in enumerate(rows):
                    cells = {}
                    col = 0
                    for td in tr.xpath("./td|./th"):
                        while occupied.get(col, 0) > ri:
                            col += 1
                        width = int(td.get("colspan", "1"))
                        height = int(td.get("rowspan", "1"))
                        cells[col] = " ".join(td.text_content().split())
                        for j in range(col, col + width):
                            occupied[j] = ri + height
                        col += width
                    grid.append(cells)
                for ri, cells in enumerate(grid):
                    for col, txt in cells.items():
                        if not re.search(r"\bFT(?:DNA)?\b", txt):
                            continue
                        kits = re.findall(kit_pattern, txt)
                        for kit in kits:
                            branch_text = None
                            branch_row = None
                            for nxt in range(ri + 1, min(ri + 4, len(grid))):
                                t = grid[nxt].get(col, "")
                                if not t:
                                    continue
                                if re.fullmatch(
                                    r"[A-Z][0-9a-z]*-[A-Z0-9]+(?:>[A-Z0-9]+)*", t
                                ):
                                    branch_text = t
                                    branch_row = nxt + 1
                                break
                            label = None
                            if branch_text:
                                label = (
                                    branch_text[0]
                                    + "-"
                                    + branch_text.split(">")[-1].split("-")[-1]
                                )
                            key = f"BOSTROM:{filename}:r{ri+1}:c{col+1}:{kit}"
                            assertion = dict(
                                evidence_id=key,
                                source="Bostrom_compilation",
                                source_url=manifest[filename]["url"],
                                source_sha256=sha(raw),
                                sheet=filename,
                                row=ri + 1,
                                column=col + 1,
                                kit=kit,
                                kit_cell=txt,
                                branch_text=branch_text,
                                branch_row=branch_row,
                                branch_node=resolve(label) if label else None,
                                linkage="exact_explicit_kit",
                                branch_linkage=(
                                    "same_spreadsheet_column_first_nonempty_within_three_rows"
                                    if branch_text
                                    else None
                                ),
                                context_rows=[
                                    dict(
                                        row=j + 1,
                                        cells={
                                            str(c + 1): v
                                            for c, v in grid[j].items()
                                            if v
                                        },
                                    )
                                    for j in range(
                                        max(0, ri - 2), min(len(grid), ri + 4)
                                    )
                                ],
                                pedigree_verified=False,
                                ancestor_place_or_date_assigned=False,
                            )
                            curated[kit].append(assertion)
                sources.append(dict(source_id=filename, **manifest[filename]))
        assert (
            len(curated) == 144 and sum(map(len, curated.values())) == 145
        ), "Curated extraction changed: review layout"
        selected = (
            set(prior)
            | set(curated)
            | {
                r["record_id"]
                for r in legacy
                if r["object_class"] == "modern_coded"
                and (r["inside_generalized_sweden"] or r["swedish_country_text"])
            }
        )
        audit = []
        clean = []
        pairs = collections.defaultdict(list)
        for kit in sorted(selected):
            old = prior.get(kit)
            m = lm.get(kit)
            cs = curated.get(kit, [])
            branch = (
                m["canonical_node"]
                if m
                else old["branch_assessment"].get("node_id") if old else None
            )
            assertions = []
            flags = []
            comparisons = []
            if m:
                assertions.append(
                    dict(
                        evidence_id="VISUAL:" + kit,
                        source="Visual_DNA_legacy",
                        raw=m["source"],
                        origin_type="source_reported_paternal_locality_event_unspecified",
                        origin_date=None,
                    )
                )
                flags.extend(m["flags"])
            if old:
                for field, source in [
                    ("source_map", "FTDNA_Sweden_map"),
                    ("source_table", "FTDNA_Sweden_table"),
                ]:
                    if old.get(field):
                        assertions.append(
                            dict(
                                evidence_id=source + ":" + kit,
                                source=source,
                                raw=old[field],
                                event_assertions=dates(old[field].get("ancestor", "")),
                            )
                        )
                flags.extend(old["review_flags"])
                if (
                    "family_proxy_branch_conflict"
                    in old["candidate_withholding_reasons"]
                ):
                    flags.append("known_family_proxy_branch_conflict")
                for field in ["branch_assessment", "table_branch_assessment"]:
                    b = old.get(field, {})
                    if b.get("node_id"):
                        rel = relation(branch, b["node_id"])
                        comparisons.append(
                            dict(source=field, node=b["node_id"], relationship=rel)
                        )
                        if rel == "divergent":
                            flags.append("source_branch_conflict")
                # Keep adjudicated events separate from raw source assertions; never relocate pins.
                dt = old["date_assessment"]
                original = old["original_date_assessment"]
                if dt.get("event_selection_decision_id"):
                    flags.append("focal_event_selected_by_prior_adjudication")
                if old["documentary_evidence_status"] != "source_assertion_only":
                    flags.append("prior_documentary_evidence_available")
            else:
                dt = dates("")
                original = dt
            for c in cs:
                rel = relation(branch, c["branch_node"])
                comparisons.append(
                    dict(
                        source=c["evidence_id"], node=c["branch_node"], relationship=rel
                    )
                )
                if rel == "divergent":
                    flags.append("curated_branch_conflict_requires_review")
            broad = bool(m and m["broad_candidate"])
            blocking = {
                "source_branch_conflict",
                "known_family_proxy_branch_conflict",
                "confirmation_conflict",
                "curated_branch_conflict_requires_review",
            }
            screened = bool(
                m and m["low_conflict_candidate"] and not (set(flags) & blocking)
            )
            lat = (
                float(m["source"]["Lat"])
                if m
                else (old.get("source_map") or {}).get("lat") if old else None
            )
            lon = (
                float(m["source"]["Lng"])
                if m
                else (old.get("source_map") or {}).get("lng") if old else None
            )
            place = (
                m["source"]["Savedloc"]
                if m
                else (old.get("source_map") or {}).get("savedLoc") if old else None
            )
            date_flags = (
                dt["chronology_conflict"]
                or dt["approximate_or_uncertain"]
                or bool(
                    old
                    and any(
                        "date" in x or "chronology" in x
                        for x in old["candidate_withholding_reasons"]
                    )
                )
            )
            record = dict(
                record_id=kit,
                ancestor_name_key=old["ancestor_name_key"] if old else None,
                event_evidence_level=dt["evidence_level"],
                canonical_branch=nodes[branch]["name"] if branch else None,
                branch_node_id=branch,
                source_lat=lat,
                source_lon=lon,
                origin_text=place,
                origin_type=(
                    "source_reported_paternal_locality_event_unspecified"
                    if place
                    else "unlocated"
                ),
                origin_date=None,
                source_reported_birth_year=dt["explicit_birth_year"],
                source_reported_death_year=dt["explicit_death_year"],
                date_uncertain_or_conflicted=bool(date_flags),
                birth_year_dates_map_location=False,
                coordinate_uncertainty_km=None,
                place_id=(
                    old["origin_assessment"].get("matched_place", {}).get("id")
                    if old
                    else None
                ),
                place_reference_status=(
                    old["origin_assessment"]["status"] if old else "not_crosschecked"
                ),
                broad_primary_candidate=broad,
                reconciled_screen_candidate=screened,
                curated_exact_kit_coverage=bool(cs),
                curated_robustness_candidate=bool(cs and screened),
                documented_family_cluster=(
                    old.get("documented_family_cluster") if old else None
                ),
                family_proxy_id=old.get("family_proxy_id") if old else None,
                evidence_ids=[a["evidence_id"] for a in assertions]
                + [c["evidence_id"] for c in cs]
                + (old["adjudication_ids"] if old else []),
                conflict_flags=sorted(
                    set(flags)
                    & (
                        blocking
                        | {
                            "implausible_or_conflicting_dates",
                            "foreign_country_text_at_swedish_pin",
                            "branch_divergence_from_snapshot",
                        }
                    )
                ),
                strict_pilot_outcome=(
                    old["genetic_release_outcome"] if old else "not_assessed"
                ),
                modelling_ready=False,
            )
            clean.append(record)
            audit.append(
                dict(
                    record_id=kit,
                    assertions=assertions,
                    curated_assertions=cs,
                    branch_comparisons=comparisons,
                    flags=sorted(set(flags)),
                    selected_branch_policy="legacy_map_when_present_otherwise_previous_source; curated_not_auto_promoted",
                    selected_event=dt,
                    original_event=original,
                    prior_adjudication_ids=old["adjudication_ids"] if old else [],
                    date_location_link_established=False,
                )
            )
            if screened:
                pairs[(branch, lat, lon)].append(kit)
        labels = collections.defaultdict(list)
        for (branch, lat, lon), kits in sorted(pairs.items()):
            labels[branch].append(
                dict(lat=lat, lon=lon, record_ids=kits, coordinate_uncertainty_km=None)
            )
        units = [
            dict(
                branch_node_id=k,
                canonical_branch=nodes[k]["name"],
                localities=v,
                locality_weight=1 / len(v),
                unit_role="one_label_contribution_locality_mixture; broader_dependence_pending",
            )
            for k, v in sorted(labels.items())
        ]
        counts = dict(
            retained_union_records=len(clean),
            broad_primary_candidates=sum(r["broad_primary_candidate"] for r in clean),
            reconciled_screen_candidates=sum(
                r["reconciled_screen_candidate"] for r in clean
            ),
            reconciled_labels=len(labels),
            reconciled_label_coordinate_pairs=len(pairs),
            curated_explicit_identifiers=len(curated),
            curated_mentions=sum(map(len, curated.values())),
            curated_ids_matching_prior_union=len(set(curated) & set(prior)),
            curated_ids_matching_modern_map=len(set(curated) & set(lm)),
            curated_robustness_candidates=sum(
                r["curated_robustness_candidate"] for r in clean
            ),
            screened_with_source_birth_year=sum(
                r["reconciled_screen_candidate"]
                and r["source_reported_birth_year"] is not None
                for r in clean
            ),
            screened_with_uncertain_or_conflicted_dates=sum(
                r["reconciled_screen_candidate"] and r["date_uncertain_or_conflicted"]
                for r in clean
            ),
            conflict_flags=dict(
                collections.Counter(f for r in clean for f in r["conflict_flags"])
            ),
            branch_comparisons=dict(
                collections.Counter(
                    c["relationship"] for r in audit for c in r["branch_comparisons"]
                )
            ),
            modelling_ready=0,
        )
        out = dict(
            schema_version="sclara-sweden-genetics-1",
            primary_design="The primary analysis uses reported paternal localities without requiring a documented pedigree or a dated ancestor for every record.",
            counts=counts,
            records=clean,
            analysis_units=[],
            spatial_positions=[],
            lineage_paths={},
            source_assertions=audit,
        )
        assert len({r["record_id"] for r in clean}) == len(clean)
        assert (
            sum(len(x["record_ids"]) for u in units for x in u["localities"])
            == counts["reconciled_screen_candidates"]
        )
        return out

    def spatial_scenario_draws(lat, lon, radius_km, seed, n=64):
        """Assumed location perturbation, not an empirically estimated error bound."""
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            delta = radius_km * math.sqrt(rng.random()) / 6371.0088
            theta = 2 * math.pi * rng.random()
            p, l = math.radians(lat), math.radians(lon)
            q = math.asin(
                math.sin(p) * math.cos(delta)
                + math.cos(p) * math.sin(delta) * math.cos(theta)
            )
            z = l + math.atan2(
                math.sin(theta) * math.sin(delta) * math.cos(p),
                math.cos(delta) - math.sin(p) * math.sin(q),
            )
            out.append([math.degrees(q), (math.degrees(z) + 180) % 360 - 180])
        return out

    def construct_units(out):
        from analysis.units import construct_units as group_labels

        specpath = SUPPORT / "decisions/broad_release_specification.json"
        spec = read(specpath)
        assert spec["raw_sha256"] == sha(resource("observations.zip"))
        tree = read(REF / "ftdna_tree.json")
        return group_labels(out, tree, spec, sha(resource(specpath)), source_path.name)

    records = parse_ftdna()
    out = construct_units(reconcile(records, parse_visual(records)))
    out["provenance"] = {
        "source_archive": source_path.name,
        "sha256": sha(SOURCE.read_bytes()),
        "record_key": "Original source kit identifier; exact-identifier joins only.",
    }
    expected = {
        "retained_union_records": 6785,
        "primary_analysis_units": 1339,
        "approved_source_records": 1647,
        "dependence_blocks": 246,
        "positions": 1404,
    }
    if any(out["counts"][key] != value for key, value in expected.items()):
        raise ValueError("Paternal source counts differ from the frozen cohort")
    output_path.write_bytes(
        (json.dumps(out, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return out["counts"]


def prepare_archaeology(source_path, output_path, work_dir=None):
    """Reproduce the frozen archaeology source normalization."""
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    if source_path == output_path or not source_path.is_file():
        raise ValueError(
            "A retained source archive and distinct output path are required"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    SOURCE = source_path
    OUT = output_path.parent
    DOC = None
    import argparse, collections, csv, hashlib, io, json, math, re, shutil, sqlite3, sys, tempfile, zipfile

    def resource(name):
        with zipfile.ZipFile(SOURCE) as z:
            manifest = json.loads(z.read("manifest.json"))
            expected = next(
                v["sha256"] for v in manifest["members"] if v["path"] == name
            )
            value = z.read(name)
        assert hashlib.sha256(value).hexdigest() == expected, name
        return value

    def strucke():
        with zipfile.ZipFile(io.BytesIO(resource("StruckeC14_Sweden_v2.zip"))) as z:
            return list(
                csv.DictReader(
                    io.StringIO(
                        z.read("StruckeC14_Sweden.csv").decode("utf-8-sig"), newline=""
                    ),
                    delimiter=";",
                )
            )

    def upgrade_rules():
        return json.loads(resource("dictionary_decisions.json"))

    PERIODS = [
        ("Mesolithic", -10000, -4000, "fine"),
        ("Early_Middle_Neolithic", -4000, -2300, "fine"),
        ("Late_Neolithic", -2300, -1700, "fine"),
        ("Early_Bronze", -1700, -1100, "fine"),
        ("Late_Bronze", -1100, -500, "fine"),
        ("Pre_Roman_Iron", -500, 0, "fine"),
        ("Roman_Iron", 0, 400, "fine"),
        ("Migration", 400, 550, "fine"),
        ("Vendel", 550, 750, "fine"),
        ("Viking", 750, 1050, "fine"),
        ("Medieval", 1050, 1520, "fine"),
        ("Bronze", -1700, -500, "broad"),
        ("Iron", -500, 1050, "broad"),
    ]

    CONTEXTS = {
        "härd": ("hearth", "fire_features"),
        "härdgrop": ("hearth_pit", "fire_features"),
        "kokgrop": ("cooking_pit", "fire_features"),
        "stolphål": ("posthole", "structures"),
        "grophus": ("pit_house", "structures"),
        "hus": ("house", "structures"),
        "golvlager": ("floor_layer", "structures"),
        "stolpfundament": ("post_foundation", "structures"),
        "påle": ("stake", "structures"),
        "grop": ("pit", "excavated_features"),
        "gropsystem": ("pit_system", "excavated_features"),
        "nedgrävning": ("cut", "excavated_features"),
        "ränna": ("trench", "excavated_features"),
        "dike": ("ditch", "excavated_features"),
        "brunn": ("well", "water"),
        "vattenhål": ("waterhole", "water"),
        "lager": ("layer", "deposits"),
        "kulturlager": ("cultural_layer", "deposits"),
        "brandlager": ("burnt_layer", "deposits"),
        "skärvstenslager": ("fire_cracked_stone_layer", "deposits"),
        "avfallsgrop": ("refuse_pit", "deposits"),
        "skärvstenshög": ("fire_cracked_stone_mound", "stone_features"),
        "skärvstensflak": ("fire_cracked_stone_spread", "stone_features"),
        "röse": ("cairn", "stone_features"),
        "hög": ("mound", "stone_features"),
        "stenpackning": ("stone_packing", "stone_features"),
        "stensträng": ("stone_wall_line", "land_use"),
        "röjningsröse": ("clearance_cairn", "land_use"),
        "odlingsröse": ("clearance_cairn", "land_use"),
        "odlingslager": ("cultivation_layer", "land_use"),
        "fångstgrop": ("trapping_pit", "hunting"),
        "ugn": ("oven_unspecified", "thermal_installations"),
        "blästugn": ("bloomery_furnace", "metal_production"),
        "slaggvarp": ("slag_heap", "metal_production"),
        "kolningsgrop": ("charcoal_pit", "charcoal_production"),
        "kolbotten": ("charcoal_platform", "charcoal_production"),
        "skelettgrav": ("inhumation_grave", "burial"),
        "brandgrav": ("cremation_grave", "burial"),
        "brandgrop": ("cremation_pit", "burial"),
        "urnebrandgrop": ("urn_cremation_pit", "burial"),
        "grav": ("grave_unspecified", "burial"),
        "stensättning": ("stone_setting", "burial"),
        "urnegrop": ("urn_pit", "burial"),
        "urnegrav": ("urn_grave", "burial"),
        "flatmarksgrav": ("flat_grave", "burial"),
        "blockgrav": ("boulder_grave", "burial"),
        "hällkista": ("stone_cist", "burial"),
        "bengrop": ("bone_pit", "deposits"),
        "benlager": ("bone_layer", "deposits"),
        "kollager": ("charcoal_layer", "deposits"),
        "sotlager": ("soot_layer", "deposits"),
        "utkastlager": ("discard_layer", "deposits"),
        "ässja": ("forge_hearth", "metal_production"),
        "blästerugn": ("bloomery_furnace", "metal_production"),
        "slagglager": ("slag_layer", "metal_production"),
        "slaggrop": ("slag_pit", "metal_production"),
        "väggränna": ("wall_trench", "structures"),
        "husgrund": ("building_foundation", "structures"),
        "terrasskant": ("terrace_edge", "land_use"),
        "terrass": ("terrace", "land_use"),
        "tjärdal": ("tar_kiln", "tar_production"),
        "tjärgrop": ("tar_pit", "tar_production"),
        "liggmila": ("horizontal_charcoal_stack", "charcoal_production"),
        "eldstad": ("fireplace", "fire_features"),
        "grophärd": ("hearth_pit", "fire_features"),
        "skärvstenspackning": ("fire_cracked_stone_packing", "stone_features"),
        "skärvstenskoncentration": (
            "fire_cracked_stone_concentration",
            "stone_features",
        ),
        "skärvstenssamling": ("fire_cracked_stone_concentration", "stone_features"),
        "stenläggning": ("stone_paving", "structures"),
        "rest sten": ("standing_stone", "stone_features"),
        "lertäktsgrop": ("clay_extraction_pit", "extraction"),
        "boplatsgrop": ("settlement_pit", "excavated_features"),
        "stockbåt": ("logboat", "water_transport"),
        "pålspärr": ("pile_barrier", "water_structures"),
    }

    def context_feature(raw):
        text = (raw or "").strip().lower()
        if text.startswith("stensättning,"):
            return CONTEXTS["stensättning"]
        return CONTEXTS.get(text)

    def file_hash(p):
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(b)
        return h.hexdigest()

    def digest(b):
        return hashlib.sha256(b).hexdigest()

    def dump(p, x):
        p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding="utf-8")

    def valid(x):
        return x is not None and str(x).strip().lower() not in ("", "null", "na", "n/a")

    def num(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    def verify_comparison_dates(q, db):
        """Verify documented laboratory corrections without rewriting source assertions."""
        import numpy as np

        curve = np.loadtxt(DOC / "intcal20.14c", delimiter=",", comments="#")
        curve = curve[np.argsort(curve[:, 0])]
        grid = np.arange(0, 55001)
        mu = np.interp(grid, curve[:, 0], curve[:, 1])
        sigma = np.interp(grid, curve[:, 0], curve[:, 2])
        checked = {}
        for aid, r in q.get("comparison_date_corrections", {}).items():
            raw = db.execute(
                "select source_row_json from assertions where assertion_id=?", (aid,)
            ).fetchone()[0]
            assert digest(raw.encode()) == r["expected_source_row_sha256"], aid
            original = json.loads(raw)
            assert (
                float(original["c14_age_bp"]) == r["source_age_bp"]
                and float(original["c14_error"]) == r["source_error_bp"]
            )
            age = r["corrected_age_bp"]
            var = sigma * sigma + r["corrected_error_bp"] ** 2
            mask = np.abs(age - mu) <= 8 * np.sqrt(var)
            gg = grid[mask]
            vv = var[mask]
            mm = mu[mask]
            logp = -0.5 * (age - mm) ** 2 / vv - 0.5 * np.log(vv)
            prob = np.exp(logp - logp.max())
            prob /= prob.sum()
            order = np.argsort(prob)[::-1]
            selected = order[: np.searchsorted(np.cumsum(prob[order]), 0.954) + 1]
            bounds = [int(1950 - gg[selected].max()), int(1950 - gg[selected].min())]
            assert bounds == r["expected_calendar_envelope"], (aid, bounds)
            checked[aid] = dict(r, calendar_envelope=bounds)
        return checked

    def calibration_check(rows):
        import numpy as np

        a = np.loadtxt(DOC / "intcal20.14c", delimiter=",", comments="#")
        a = a[np.argsort(a[:, 0])]
        grid = np.arange(0, 55001)
        mu = np.interp(grid, a[:, 0], a[:, 1])
        sd = np.interp(grid, a[:, 0], a[:, 2])
        checks = []
        eligible = [
            r
            for r in rows
            if r["calibration_status"] == "calibrated"
            and num(r["c14_age_bp"]) is not None
            and num(r["c14_error"]) is not None
            and num(r["c14_error"]) > 0
        ]
        cache = {}
        for r in eligible:
            key = (num(r["c14_age_bp"]), num(r["c14_error"]))
            if key not in cache:
                var = sd * sd + key[1] ** 2
                mask = np.abs(key[0] - mu) <= 8 * np.sqrt(var)
                gg = grid[mask]
                vv = var[mask]
                mm = mu[mask]
                logp = -0.5 * (key[0] - mm) ** 2 / vv - 0.5 * np.log(vv)
                p = np.exp(logp - logp.max())
                p /= p.sum()
                order = np.argsort(p)[::-1]
                sel = order[: np.searchsorted(np.cumsum(p[order]), 0.954) + 1]
                cache[key] = (float(1950 - gg[sel].max()), float(1950 - gg[sel].min()))
            lo, hi = cache[key]
            source = sorted([num(r["cal_95_min"]), num(r["cal_95_max"])])
            checks.append(
                {
                    "fid": r["fid"],
                    "source_raw_min": r["cal_95_min"],
                    "source_raw_max": r["cal_95_max"],
                    "normalized_source_bounds": source,
                    "independent_hpd_envelope": [lo, hi],
                    "maximum_endpoint_difference_years": max(
                        abs(lo - source[0]), abs(hi - source[1])
                    ),
                }
            )
        report = {
            "method": "Independent Gaussian radiocarbon likelihood including curve sigma, linear IntCal20 interpolation on annual 0..55000 calBP grid; negligible likelihood beyond eight joint standard deviations omitted; flat calendar prior; 95.4% HPD outer envelope. Source bounds retained. Endpoint differences above five years withheld, not labelled proven source errors. For agreeing dates use conservative union of both outer envelopes. No probability weights derived from envelope widths.",
            "checks": checks,
            "unique_age_error_pairs": len(cache),
            "endpoint_disagreement_over_5_years": sum(
                x["maximum_endpoint_difference_years"] > 5 for x in checks
            ),
        }
        dump(DOC / "calibration_verification.json", report)
        assert len(checks) > 29000
        return {x["fid"]: x for x in checks}

    def gpkg_geometry(blob):
        from shapely import wkb

        assert blob[:2] == b"GP"
        envelope = (blob[3] >> 1) & 7
        return wkb.loads(blob[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope] :])

    def apply_upgrade(db):
        """Apply frozen exact mappings/adjudications; preserve original row JSON and every other gate."""
        v = upgrade_rules()
        db.row_factory = sqlite3.Row
        before = [
            dict(x)
            for x in db.execute("select * from assertions order by assertion_id")
        ]
        decisions = {x["assertion_id"]: x for x in v["corrections"]}
        mapping = v["mapping_rules"]
        assert set(decisions) <= set(x["assertion_id"] for x in before)
        definitions = collections.defaultdict(list)
        for label, x in mapping.items():
            definitions[(x["feature_id"], x["functional_block"])].append(label)
        for (fid, block), labels in definitions.items():
            old = db.execute(
                "select functional_block,source_meaning from features where feature_id=?",
                (fid,),
            ).fetchone()
            if old:
                assert old[0] == block
                labels = old[1].split("; ") + labels
            db.execute(
                "insert or replace into features values(?,?,?)",
                (fid, block, "; ".join(sorted(set(labels)))),
            )
        db.execute(
            "create table if not exists source_adjudications(assertion_id text primary key,decision_json text)"
        )
        changes = []
        inventory = collections.Counter()
        mapped_inventory = collections.Counter()
        for original in before:
            row = dict(original)
            why = json.loads(row["reasons"])
            decision = decisions.get(row["assertion_id"])
            if decision:
                assert (
                    digest(row["source_row_json"].encode())
                    == decision["source_row_sha256"]
                )
                for k, expected in decision["expected"].items():
                    assert row[k] == expected, (row["assertion_id"], k)
                assert set(decision["updates"]) <= {
                    "context_id",
                    "context_raw",
                    "material",
                    "species",
                    "short_lived",
                    "dating_risk",
                    "date_relation",
                }
                row.update(decision["updates"])
                if decision["hold"]:
                    why.append("original_report_link_unresolved")
                db.execute(
                    "insert or replace into source_adjudications values(?,?)",
                    (row["assertion_id"], json.dumps(decision, ensure_ascii=False)),
                )
            label = (row["context_raw"] or "").strip().lower()
            rule = mapping.get(label)
            feature = context_feature(row["context_raw"])
            if rule:
                feature = (rule["feature_id"], rule["functional_block"])
            if original["feature_id"] is None:
                inventory[original["context_raw"]] += 1
            row["feature_id"] = feature[0] if feature else None
            if feature:
                why = [x for x in why if x != "unmapped_context_type"]
                if original["feature_id"] is None:
                    mapped_inventory[original["context_raw"]] += 1
            elif "unmapped_context_type" not in why:
                why.append("unmapped_context_type")
            # Preserve the exclusions for source comments, reservoir risk, geometry and calibration conflicts.
            assert set(json.loads(original["reasons"])) - set(why) <= {
                "unmapped_context_type"
            }
            row["reasons"] = json.dumps(sorted(set(why)))
            row["disposition"] = "withheld" if why else "eligible_reported_context"
            fields = [
                "context_id",
                "context_raw",
                "feature_id",
                "material",
                "species",
                "short_lived",
                "dating_risk",
                "date_relation",
                "reasons",
                "disposition",
            ]
            meaningful = [
                f
                for f in fields
                if (
                    sorted(json.loads(row[f])) != sorted(json.loads(original[f]))
                    if f == "reasons"
                    else row[f] != original[f]
                )
            ]
            if meaningful:
                db.execute(
                    "update assertions set "
                    + ",".join(f + "=?" for f in fields)
                    + " where assertion_id=?",
                    [row[f] for f in fields] + [row["assertion_id"]],
                )
                changes.append(
                    {
                        "assertion_id": row["assertion_id"],
                        "cause": "source_adjudication" if decision else "dictionary",
                        "changed_fields": meaningful,
                        "before": {f: original[f] for f in fields},
                        "after": {f: row[f] for f in fields},
                    }
                )
        # Reconstruct panels with exactly the inherited containment and support rules.
        db.executescript(
            "delete from assertion_periods; delete from panel_units; delete from panel_support;"
        )
        for r in db.execute(
            "select * from assertions where disposition='eligible_reported_context'"
        ).fetchall():
            for pid, start, end, family in PERIODS:
                if start <= r["start_year"] and r["end_year"] < end:
                    db.execute(
                        "insert into assertion_periods values(?,?)",
                        (r["assertion_id"], pid),
                    )
                    db.execute(
                        "insert into panel_units values(?,?,?,?,?) on conflict(period_id,feature_id,site_id) do update set short_lived=max(short_lived,excluded.short_lived)",
                        (
                            pid,
                            r["feature_id"],
                            r["site_id"],
                            r["evidence_group"],
                            r["short_lived"],
                        ),
                    )
        for pid, f in db.execute(
            "select distinct period_id,feature_id from panel_units"
        ).fetchall():
            units = db.execute(
                "select u.site_id,s.position_group,s.county,u.short_lived from panel_units u join sites s using(site_id) where period_id=? and feature_id=?",
                (pid, f),
            ).fetchall()
            counties = collections.Counter(x[2] for x in units)
            positions = len({x[1] for x in units})
            short = sum(x[3] for x in units)
            groups = db.execute(
                "select count(distinct a.evidence_group) from assertions a join assertion_periods ap using(assertion_id) where ap.period_id=? and a.feature_id=?",
                (pid, f),
            ).fetchone()[0]
            db.execute(
                "insert into panel_support values(?,?,?,?,?,?,?,?,?)",
                (
                    pid,
                    f,
                    len(units),
                    positions,
                    groups,
                    len(counties),
                    max(counties.values()) / len(units),
                    short,
                    int(positions >= 5 and groups >= 3),
                ),
            )
        # Public source rows, dates and location metadata are invariant, including withheld records.
        for r, old in zip(
            db.execute(
                "select assertion_id,source_row_json,start_year,end_year,site_id from assertions order by assertion_id"
            ),
            before,
        ):
            assert tuple(r) == tuple(
                old[k]
                for k in [
                    "assertion_id",
                    "source_row_json",
                    "start_year",
                    "end_year",
                    "site_id",
                ]
            )
        db.commit()

    def apply_corrections(db):
        """Apply the finite v3 ledger; raw source measurements remain immutable provenance."""
        import numpy as np

        v = json.loads((DOC / "chronology_decisions.json").read_text(encoding="utf-8"))
        db.row_factory = sqlite3.Row
        before = {
            r["assertion_id"]: dict(r) for r in db.execute("select * from assertions")
        }
        changes = []
        with zipfile.ZipFile(SOURCE) as z:
            curve = np.loadtxt(
                io.BytesIO(z.read("intcal20.14c")), delimiter=",", comments="#"
            )
        curve = curve[np.argsort(curve[:, 0])]
        grid = np.arange(55001)
        mu = np.interp(grid, curve[:, 0], curve[:, 1])
        sd = np.interp(grid, curve[:, 0], curve[:, 2])
        for item in v["decisions"]:
            aid = item["assertion_id"]
            old = before[aid]
            row = dict(old)
            assert digest(row["source_row_json"].encode()) == item["source_row_sha256"]
            for k, value in item["expected"].items():
                assert row[k] == value, (aid, k)
            assert set(item["updates"]) <= {
                "context_raw",
                "feature_id",
                "date_relation",
            }
            row.update(item["updates"])
            decision = dict(item)
            if item["corrected_measurement_bp_error"]:
                age, error = item["corrected_measurement_bp_error"]
                variance = sd**2 + error**2
                mask = np.abs(age - mu) <= 8 * np.sqrt(variance)
                g = grid[mask]
                vvar = variance[mask]
                logp = -0.5 * (age - mu[mask]) ** 2 / vvar - 0.5 * np.log(vvar)
                prob = np.exp(logp - logp.max())
                prob /= prob.sum()
                order = np.argsort(prob)[::-1]
                sel = order[: np.searchsorted(np.cumsum(prob[order]), 0.954) + 1]
                row["start_year"] = float(1950 - g[sel].max())
                row["end_year"] = float(1950 - g[sel].min())
                decision["corrected_intcal20_hpd_envelope"] = [
                    row["start_year"],
                    row["end_year"],
                ]
                decision["calibration_policy"] = (
                    "Same annual IntCal20 95.4% HPD method; superseded wrong-age source envelope is not unioned with corrected calibration. Raw source JSON retained."
                )
            why = json.loads(row["reasons"])
            if item["hold_reason"]:
                why.append(item["hold_reason"])
            row["reasons"] = json.dumps(sorted(set(why)))
            row["disposition"] = "withheld" if why else "eligible_reported_context"
            fields = [
                "context_raw",
                "feature_id",
                "date_relation",
                "start_year",
                "end_year",
                "reasons",
                "disposition",
            ]
            db.execute(
                "update assertions set "
                + ",".join(k + "=?" for k in fields)
                + " where assertion_id=?",
                [row[k] for k in fields] + [aid],
            )
            db.execute(
                "insert or replace into source_adjudications values(?,?)",
                (aid, json.dumps(decision, ensure_ascii=False)),
            )
            changes.append(
                {
                    "assertion_id": aid,
                    "before": {k: old[k] for k in fields},
                    "after": {k: row[k] for k in fields},
                    "decision": decision,
                }
            )
        # Review notes qualify interpretation without globally excluding every unreviewed source record.
        for aid, review in v["review_annotations"].items():
            if aid not in {x["assertion_id"] for x in v["decisions"]}:
                db.execute(
                    "insert or replace into source_adjudications values(?,?)",
                    (
                        aid,
                        json.dumps(
                            {
                                "action": "retain_reported_context_with_review_limits",
                                "review": review,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
        db.executescript(
            "delete from assertion_periods; delete from panel_units; delete from panel_support;"
        )
        for r in db.execute(
            "select * from assertions where disposition='eligible_reported_context'"
        ).fetchall():
            for pid, start, end, family in PERIODS:
                if start <= r["start_year"] and r["end_year"] < end:
                    db.execute(
                        "insert into assertion_periods values(?,?)",
                        (r["assertion_id"], pid),
                    )
                    db.execute(
                        "insert into panel_units values(?,?,?,?,?) on conflict(period_id,feature_id,site_id) do update set short_lived=max(short_lived,excluded.short_lived)",
                        (
                            pid,
                            r["feature_id"],
                            r["site_id"],
                            r["evidence_group"],
                            r["short_lived"],
                        ),
                    )
        for pid, f in db.execute(
            "select distinct period_id,feature_id from panel_units"
        ).fetchall():
            units = db.execute(
                "select u.site_id,s.position_group,s.county,u.short_lived from panel_units u join sites s using(site_id) where period_id=? and feature_id=?",
                (pid, f),
            ).fetchall()
            counties = collections.Counter(x[2] for x in units)
            positions = len({x[1] for x in units})
            short = sum(x[3] for x in units)
            groups = db.execute(
                "select count(distinct a.evidence_group) from assertions a join assertion_periods ap using(assertion_id) where ap.period_id=? and a.feature_id=?",
                (pid, f),
            ).fetchone()[0]
            db.execute(
                "insert into panel_support values(?,?,?,?,?,?,?,?,?)",
                (
                    pid,
                    f,
                    len(units),
                    positions,
                    groups,
                    len(counties),
                    max(counties.values()) / len(units),
                    short,
                    int(positions >= 5 and groups >= 3),
                ),
            )

        for row in db.execute(
            "select assertion_id,source_row_json,site_id from assertions"
        ):
            old = before[row["assertion_id"]]
            assert (
                row["source_row_json"] == old["source_row_json"]
                and row["site_id"] == old["site_id"]
            )
        db.commit()

    def build():
        import numpy as np
        from pyproj import Transformer
        from shapely.geometry import Point

        kmrfile = DOC / "kmr_dated_sites.sqlite"
        source = sqlite3.connect(kmrfile)
        source.row_factory = sqlite3.Row
        assert source.execute("pragma quick_check").fetchone()[0] == "ok"
        rows = strucke()
        assert len(rows) == 30296
        calibration = calibration_check(rows)
        temp = DOC / "build_in_progress.sqlite"
        if temp.exists():
            temp.unlink()
        db = sqlite3.connect(temp)
        db.execute("pragma journal_mode=OFF")
        db.execute("pragma synchronous=OFF")
        db.executescript(
            """
     create table metadata(key text primary key,value text);
     create table sites(site_id text primary key,uuid text unique,old_id text,site_type text,name text,county text,municipality text,parish text,status text,investigation_status text,longitude real,latitude real,easting real,northing real,geometry_radius_m real,source_uncertainty_m real,position_group text);
     create table positions(position_id text primary key,site_id text,longitude real,latitude real,easting real,northing real,geometry_radius_m real,within_site_weight real,source_table text,source_fid integer);
     create table components(component_id integer,site_id text,feature_raw text,reported_count integer);
     create table properties(property_id integer,site_id text,component_id integer,property_type text,value text);
     create table periods(period_id text primary key,start_year integer,end_year_exclusive integer,family text);
     create table features(feature_id text primary key,functional_block text,source_meaning text);
     create table assertions(assertion_id text primary key,site_id text,lab_id text,context_id text,evidence_group text,feature_id text,context_raw text,material text,species text,start_year real,end_year real,date_relation text,dating_risk text,short_lived integer,source_longitude real,source_latitude real,kmr_geometry_distance_m real,disposition text,reasons text,source_row_json text);
     create table assertion_periods(assertion_id text,period_id text,primary key(assertion_id,period_id));
     create table panel_units(period_id text,feature_id text,site_id text,evidence_group text,short_lived integer,primary key(period_id,feature_id,site_id));
     create table panel_support(period_id text,feature_id text,sites integer,positions integer,source_groups integer,counties integer,largest_county_share real,short_lived_sites integer,minimum_gate_pass integer,primary key(period_id,feature_id));
     """
        )
        db.executemany("insert into periods values(?,?,?,?)", PERIODS)
        fdefs = {v[0]: (v[1], []) for v in CONTEXTS.values()}
        for k, (f, b) in CONTEXTS.items():
            fdefs[f][1].append(k)
        db.executemany(
            "insert into features values(?,?,?)",
            [(f, b, "; ".join(labels)) for f, (b, labels) in fdefs.items()],
        )
        transform = Transformer.from_crs(3006, 4326, always_xy=True)
        uuidmap = {}
        siteinfo = {}
        for s in source.execute("select * from lamning order by uuid"):
            sid = s["lamningsnummer"]
            uid = s["uuid"]
            assert sid not in siteinfo, "Duplicate KMR site ID"
            uuidmap[uid] = sid
            siteinfo[sid] = {
                "uuid": uid,
                "status": s["aktualitetstatus"],
                "county": s["lan"],
                "geom": [],
            }
            db.execute(
                "insert into sites values(?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                (
                    sid,
                    uid,
                    s["raa_nummer"],
                    s["lamningstyp"],
                    s["lamningsnamn"],
                    s["lan"],
                    s["kommun"],
                    s["socken"],
                    s["aktualitetstatus"],
                    s["undersokningsstatus"],
                ),
            )
        print("Radiocarbon-linked sites", len(siteinfo), flush=True)
        bad_geometry = []
        for table in ["point", "linestring", "polygon"]:
            for r in source.execute("select * from " + table + " order by fid"):
                sid = uuidmap.get(r["lamning_uuid"])
                if not sid:
                    continue
                try:
                    g = gpkg_geometry(r["geometri"])
                    if g.is_empty:
                        raise ValueError("empty")
                    pt = g.centroid
                    x, y = pt.x, pt.y
                    lon, lat = transform.transform(x, y)
                    if not (9 <= lon <= 25 and 54 <= lat <= 70):
                        raise ValueError("outside_broad_swedish_coordinate_bounds")
                    bounds = g.bounds
                    radius = max(
                        math.hypot(xx - x, yy - y)
                        for xx in [bounds[0], bounds[2]]
                        for yy in [bounds[1], bounds[3]]
                    )
                    siteinfo[sid]["geom"].append((x, y, radius, table, r["fid"]))
                    db.execute(
                        "insert into positions values(?,?,?,?,?,?,?,?,?,?)",
                        (
                            table + ":" + str(r["fid"]),
                            sid,
                            lon,
                            lat,
                            x,
                            y,
                            radius,
                            0,
                            table,
                            r["fid"],
                        ),
                    )
                except Exception as e:
                    bad_geometry.append(
                        {
                            "site_id": sid,
                            "table": table,
                            "fid": r["fid"],
                            "reason": str(e),
                        }
                    )
            print("Geometry", table, "done", flush=True)
        # Per-site equally weighted geometry means; radius includes the entire geometry set.
        for sid, s in siteinfo.items():
            gs = s["geom"]
            if not gs:
                continue
            x = sum(g[0] for g in gs) / len(gs)
            y = sum(g[1] for g in gs) / len(gs)
            radius = max(math.hypot(g[0] - x, g[1] - y) + g[2] for g in gs)
            lon, lat = transform.transform(x, y)
            s.update(x=x, y=y, lon=lon, lat=lat, radius=radius)
            group = f"{x:.3f},{y:.3f}"
            db.execute(
                "update sites set longitude=?,latitude=?,easting=?,northing=?,geometry_radius_m=?,position_group=? where site_id=?",
                (lon, lat, x, y, radius, group, sid),
            )
        db.execute("create index positions_site on positions(site_id)")
        db.execute(
            "update positions set within_site_weight=1.0/(select count(*) from positions q where q.site_id=positions.site_id)"
        )
        for row in source.execute("select * from ingaendelamning"):
            db.execute(
                "insert into components values(?,?,?,?)",
                (
                    row["id"],
                    uuidmap.get(row["lamning_uuid"]),
                    row["lamningstyp"],
                    row["antal"],
                ),
            )
        for row in source.execute("select * from egenskap"):
            db.execute(
                "insert into properties values(?,?,?,?,?)",
                (
                    row["id"],
                    uuidmap.get(row["lamning_uuid"]),
                    row["ingaendelamning_id"],
                    row["typ"],
                    row["varde"],
                ),
            )
        tables = [x[0] for x in source.execute("select table_name from gpkg_contents")]
        for table in tables:
            cols = [x[1] for x in source.execute('pragma table_info("' + table + '")')]
            if "lagesosakerhet_i_meter" in cols:
                for r in source.execute(
                    'select lamning_uuid,max(lagesosakerhet_i_meter) as uncertainty from "'
                    + table
                    + '" group by lamning_uuid'
                ):
                    if r["uncertainty"] is not None:
                        db.execute(
                            "update sites set source_uncertainty_m=? where uuid=?",
                            (r["uncertainty"], r["lamning_uuid"]),
                        )
        db.commit()
        print("Source geometry and morphology built", flush=True)
        labs = collections.Counter(r["lab_id"].strip() for r in rows)
        unknown = collections.Counter()
        reasons_count = collections.Counter()
        # Only the dated subset requires exact distance to source geometry. No centroids used to adjudicate displacement.
        geomcache = {}
        dated_ids = {x["site_id"] for x in rows}
        source.execute("pragma cache_size=-100000")
        for table in ["point", "linestring", "polygon"]:
            for r in source.execute("select * from " + table):
                sid = uuidmap.get(r["lamning_uuid"])
                if sid in dated_ids:
                    try:
                        geomcache.setdefault(sid, []).append(
                            gpkg_geometry(r["geometri"])
                        )
                    except Exception:
                        pass
        for r in rows:
            aid = "C14:" + r["fid"]
            sid = r["site_id"]
            si = siteinfo.get(sid)
            why = []
            feat = context_feature(r["context_type"])
            dist = None
            bounds = [num(r["cal_95_min"]), num(r["cal_95_max"])]
            lo = hi = None
            if all(v is not None for v in bounds):
                lo, hi = sorted(bounds)
            check = calibration.get(r["fid"])
            if not check or check["maximum_endpoint_difference_years"] > 5:
                why.append("calibration_endpoint_disagreement_or_unavailable")
            elif lo is not None:
                lo = min(lo, check["independent_hpd_envelope"][0])
                hi = max(hi, check["independent_hpd_envelope"][1])
            if not si:
                why.append("no_exact_current_KMR_site")
            elif not si["geom"]:
                why.append("no_valid_KMR_geometry")
            elif "överförd" in (si["status"] or "").lower():
                why.append("KMR_transferred_site")
            elif "felregistrering" in (si["status"] or "").lower():
                why.append("KMR_erroneous_registration")
            if r["location_precision"] != "archaeological site":
                why.append("not_site_precision")
            if r["c14_data_status"] != "ok" or r["calibration_status"] != "calibrated":
                why.append("source_date_not_ok")
            if r["assessed_relevant"] != "J":
                why.append("context_date_not_reported_relevant")
            if valid(r["comment"]):
                why.append("source_comment_requires_adjudication")
            if labs[r["lab_id"].strip()] > 1:
                why.append("duplicate_lab_identifier_requires_adjudication")
            if lo is None or hi is None or not lo < hi:
                why.append("invalid_calendar_interval")
            if not feat:
                why.append("unmapped_context_type")
                unknown[r["context_type"]] += 1
            lon, lat = num(r["longitude"]), num(r["latitude"])
            east, north = num(r["easting_3006"]), num(r["northing_3006"])
            if lon is None or lat is None or not (9 <= lon <= 25 and 54 <= lat <= 70):
                why.append("invalid_source_coordinate")
            if east is not None and north is not None and sid in geomcache:
                pt = Point(east, north)
                dist = min(g.distance(pt) for g in geomcache[sid])
                tlo, tla = transform.transform(east, north)
                if (
                    lon is not None
                    and lat is not None
                    and (abs(tlo - lon) > 0.001 or abs(tla - lat) > 0.001)
                ):
                    why.append("source_CRS_disagreement")
                if dist > 1000:
                    why.append("source_point_over_1km_from_KMR_geometry")
            else:
                why.append("geometry_crosscheck_unavailable")
            material = r["material"].lower()
            species = r["species"].lower()
            short = int(
                material in ("förkolnat frö", "frö (ej förkolnat)")
                or (
                    material == "växtdelar"
                    and any(
                        t in species
                        for t in ["korn", "vete", "havre", "råg", "hasselnöt"]
                    )
                )
            )
            risk = (
                "short_lived_terrestrial"
                if short
                else (
                    "possible_inbuilt_age"
                    if material in ("träkol", "trä", "bark", "kol i järn")
                    else "reservoir_or_material_review_needed"
                )
            )
            if risk == "reservoir_or_material_review_needed":
                why.append("material_reservoir_not_resolved")
            pub = "|".join(
                r[k] for k in ("author", "publication_year", "title", "journal")
            )
            group = (
                "investigation:" + r["uppdragsnummer"]
                if valid(r["uppdragsnummer"])
                else "publication:" + digest(pub.encode())[:20]
            )
            disposition = "eligible_reported_context" if not why else "withheld"
            db.execute(
                "insert into assertions values(" + ",".join("?" * 20) + ")",
                (
                    aid,
                    sid,
                    r["lab_id"],
                    r["context_id"],
                    group,
                    feat[0] if feat else None,
                    r["context_type"],
                    r["material"],
                    r["species"],
                    lo,
                    hi,
                    "sample_date_reportedly_relevant_to_context",
                    risk,
                    short,
                    lon,
                    lat,
                    dist,
                    disposition,
                    json.dumps(why),
                    json.dumps(r, ensure_ascii=False),
                ),
            )
            reasons_count.update(why)
            if not why:
                for pid, start, end, family in PERIODS:
                    if start <= lo and hi < end:
                        db.execute(
                            "insert into assertion_periods values(?,?)", (aid, pid)
                        )
                        db.execute(
                            "insert into panel_units values(?,?,?,?,?) on conflict(period_id,feature_id,site_id) do update set short_lived=max(short_lived,excluded.short_lived)",
                            (pid, feat[0], sid, group, short),
                        )
        db.executescript(
            "create index assertions_site on assertions(site_id); create index assertions_group on assertions(evidence_group); create index ap_assertion on assertion_periods(assertion_id);"
        )
        db.commit()
        for pid, f in db.execute(
            "select distinct period_id,feature_id from panel_units"
        ).fetchall():
            units = db.execute(
                "select u.site_id,s.position_group,s.county,u.short_lived from panel_units u join sites s using(site_id) where period_id=? and feature_id=?",
                (pid, f),
            ).fetchall()
            counties = collections.Counter(x[2] for x in units)
            positions = len({x[1] for x in units})
            short = sum(x[3] for x in units)
            groups = db.execute(
                "select count(distinct a.evidence_group) from assertions a join assertion_periods ap using(assertion_id) where ap.period_id=? and a.feature_id=?",
                (pid, f),
            ).fetchone()[0]
            db.execute(
                "insert into panel_support values(?,?,?,?,?,?,?,?,?)",
                (
                    pid,
                    f,
                    len(units),
                    positions,
                    groups,
                    len(counties),
                    max(counties.values()) / len(units),
                    short,
                    int(positions >= 5 and groups >= 3),
                ),
            )
        metadata = {
            "release_id": "SW-ARCH-CONTEXT-1",
            "status": "conditional_input_release_for_reported_radiocarbon_context_panels_only",
            "sources": "KMR native subset for all Strucke v2 site identifiers; full radiocarbon catalogue retained",
            "date_rule": "Strict containment of conservative union of source and independently calculated 95.4% HPD outer envelopes; endpoints differing >5 years withheld. No median allocation, temporal weights or automatic cultural attribution. Broad and fine panels are alternative families.",
            "primary_unit": "one site x period x normalized context feature; assay multiplicity never increases primary site weight; preserve all evidence groups in assertions",
            "geometry_rule": "EPSG3006 source geometries represented by centroids with within-site weights; site coordinates are means of geometry centroids, not context locations. Preserve geometry extent and source uncertainty. Source coordinate checked against actual KMR geometry within 1000m.",
            "missingness": "Unrecorded feature is unknown. Period comparator universe is dated eligible context sites, not all national KMR sites. National inventory is an observation-frame reference, not a period-specific absence map.",
            "dating_risk": "Primary reported-context tier includes wood/charcoal inbuilt-age risk. Short-lived seed subset supplied for required chronology sensitivity. Bone, shell and unresolved materials withheld; source comments withheld pending review.",
            "limitations": [
                "Radiocarbon-selected contexts, not a complete national period-aware archaeological census.",
                "KMR types/components are undated unless independently linked; never inherit context dates.",
                "No Battle Axe cultural assignment follows from a chronological bin.",
                "No mortality, migration date, ancient ethnicity or continuity inferred.",
                "Regional modelling only at >=50km scales subject to exposure/uncertainty checks.",
                "Border effects and ecology/recording controls remain model-design responsibilities.",
                "Panel count gate is a minimum, not a power or historical-readiness certificate.",
                "SHM museum enrichment not part of this primary release.",
            ],
            "source_bundle_sha256": file_hash(SOURCE),
            "script_sha256": file_hash(Path(__file__)),
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "shapely": __import__("shapely").__version__,
                "pyproj": __import__("pyproj").__version__,
            },
            "period_authorities": [
                "https://historiska.se/utforska-historien/kunskapsbank/de-aldsta-metallerna/",
                "https://historiska.se/skola/arkeologens-dotter/",
                "https://historiska.se/utforska-historien/tidsaldrar/",
            ],
            "period_convention": "400 CE Roman/Migration; 750 CE Vendel/Viking; 1050 CE Viking/Medieval selected operational boundaries; published conventions vary (375/793/1100); source ranges retained, boundary-sensitive cases not culture assignments.",
        }
        db.executemany(
            "insert into metadata values(?,?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in metadata.items()],
        )
        db.commit()
        assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
        db.close()
        source.close()
        final = output_path
        # Extraction may reside on another filesystem. Publish from a temporary sibling.
        with tempfile.TemporaryDirectory(prefix=".sclara_publish_", dir=OUT) as sibling:
            staged = Path(sibling) / final.name
            shutil.copyfile(temp, staged)
            staged.replace(final)
        dump(
            DOC / "build_audit.json",
            {
                "withholding_reasons": dict(reasons_count),
                "unmapped_contexts": dict(unknown),
                "bad_geometries": bad_geometry,
            },
        )
        print("Canonical archaeological database built", flush=True)
        if (DOC / "dictionary_decisions.json").exists():
            db = sqlite3.connect(final)
            apply_upgrade(db)
            db.close()
        if (DOC / "chronology_decisions.json").exists():
            db = sqlite3.connect(final)
            apply_corrections(db)
            db.close()
        finalize()
        comparison_panels()
        final_source_reconciliation()
        verify()

    def finalize():
        import numpy as np
        from pyproj import Geod

        db = sqlite3.connect(output_path)
        db.executescript(
            """
     drop view if exists model_panel_units;
     drop view if exists short_lived_panel_units;
     drop view if exists period_recording_sites;
     drop view if exists panel_evidence_links;
     create view model_panel_units as
     select u.*,s.longitude,s.latitude,s.geometry_radius_m,s.source_uncertainty_m,s.position_group,s.county
     from panel_units u join panel_support p using(period_id,feature_id) join sites s using(site_id)
     where p.minimum_gate_pass=1 and (select count(*) from panel_support q where q.period_id=u.period_id and q.minimum_gate_pass=1)>=3;
     create view short_lived_panel_units as select * from model_panel_units where short_lived=1;
     create view period_recording_sites as select distinct u.period_id,u.site_id,s.longitude,s.latitude,s.position_group,s.county from panel_units u join sites s using(site_id);
     create view panel_evidence_links as select distinct ap.period_id,a.feature_id,a.site_id,a.evidence_group,a.assertion_id from assertions a join assertion_periods ap using(assertion_id);
     """
        )
        db.commit()
        db.close()

    def comparison_panels():
        """Apply functional and dating qualifications without overwriting source rows."""
        q = json.loads(resource("comparison_decisions.json"))
        rules = q["function_and_quality_rules"]
        db = sqlite3.connect(output_path)
        db.row_factory = sqlite3.Row
        db.executescript(
            "\n    create table comparison_assertions(period_id text,assertion_id text,site_id text,feature_id text,quality text,eligible integer,decision_json text,primary key(period_id,assertion_id));\n    create table comparison_sites(period_id text,feature_id text,site_id text,primary key(period_id,feature_id,site_id));\n    create table comparison_recording_sites(period_id text,site_id text,primary key(period_id,site_id));\n    create table comparison_support(period_id text,feature_id text,tier text,sites integer,counties integer,gate_pass integer,primary key(period_id,feature_id,tier));\n    create table calibration_checks(assertion_id text primary key,independent_start real,independent_end real,maximum_endpoint_difference real);\n    "
        )
        corrected_dates = verify_comparison_dates(q, db)
        db.execute(
            "create table comparison_source_reviews(assertion_id text primary key,review_json text)"
        )
        db.executemany(
            "insert into comparison_source_reviews values(?,?)",
            [
                (aid, json.dumps(review, ensure_ascii=False))
                for aid, review in q.get("comparison_source_reviews", {}).items()
            ],
        )
        db.execute(
            "create table comparison_date_corrections(assertion_id text primary key,correction_json text)"
        )
        db.executemany(
            "insert into comparison_date_corrections values(?,?)",
            [
                (aid, json.dumps(review, ensure_ascii=False))
                for aid, review in corrected_dates.items()
            ],
        )
        periods = [
            ("Pre2800", -4000, -2800),
            ("Late_Neolithic", -2300, -1700),
            ("Early_Bronze", -1700, -1100),
            ("Migration", 400, 550),
        ]
        prior_reviews = {
            r["index_assertion_id"]: r.get("source_review", {})
            for r in q["older_reviews"]
        }
        for period, start, end in periods:
            rr = [
                dict(r)
                for r in db.execute(
                    "select * from assertions where disposition='eligible_reported_context' and start_year>=? and end_year<? order by assertion_id",
                    (start, end),
                )
            ]
            for row in rr:
                aid = row["assertion_id"]
                patch = {}
                eligible = True
                if period == "Pre2800":
                    patch = q["older_rules"].get(aid, {})
                    eligible = patch.get("eligible", True)
                if period == "Early_Bronze" and aid in ["C14:14586", "C14:14587"]:
                    eligible = False
                    patch = {
                        "eligible": False,
                        "reason": "Original report contradicts the catalogue laboratory age; withhold unsupported Bronze membership. No replacement phase assigned.",
                    }
                if aid in corrected_dates:
                    correction = corrected_dates[aid]
                    lo, hi = correction["calendar_envelope"]
                    eligible = eligible and lo >= start and hi < end
                    patch = dict(patch, original_report_date_correction=correction)
                if eligible:
                    db.execute(
                        "insert or ignore into comparison_recording_sites values(?,?)",
                        (period, row["site_id"]),
                    )
                if row["feature_id"] not in ["posthole", "hearth"]:
                    continue
                if period != "Pre2800":
                    patch = dict(rules["decisions"].get(aid, {}), **patch)
                    eligible = eligible and patch.get("comparison_eligible", True)
                row.update(
                    {
                        k: v
                        for k, v in patch.items()
                        if k in ["material", "short_lived", "context_id", "species"]
                    }
                )
                quality = (
                    "Q1"
                    if row["short_lived"]
                    else ("Q2" if aid in rules["q2_qualified_ids"] else "Q3")
                )
                review = prior_reviews.get(aid, {}) if period == "Pre2800" else {}
                if period == "Pre2800" and (
                    patch.get("quality") == "Q2_original_qualified"
                    or review.get("quality") == "Q2_original_qualified"
                ):
                    quality = "Q2"
                decision = {
                    "changes": patch,
                    "original_report_review": review,
                    "source_assertions_preserved": True,
                }
                db.execute(
                    "insert into comparison_assertions values(?,?,?,?,?,?,?)",
                    (
                        period,
                        aid,
                        row["site_id"],
                        row["feature_id"],
                        quality,
                        int(eligible),
                        json.dumps(decision, ensure_ascii=False),
                    ),
                )
                if eligible:
                    db.execute(
                        "insert or ignore into comparison_sites values(?,?,?)",
                        (period, row["feature_id"], row["site_id"]),
                    )
            for feature in ["posthole", "hearth"]:
                for tier, levels in [
                    ("primary", ["Q1", "Q2", "Q3"]),
                    ("Q1", ["Q1"]),
                    ("Q1_Q2", ["Q1", "Q2"]),
                ]:
                    rows = db.execute(
                        "select distinct a.site_id,s.county from comparison_assertions a join sites s using(site_id) where a.period_id=? and a.feature_id=? and a.eligible=1 and a.quality in ("
                        + ",".join("?" for _ in levels)
                        + ")",
                        [period, feature] + levels,
                    ).fetchall()
                    sites = len({r[0] for r in rows})
                    counties = len({r[1] for r in rows})
                    db.execute(
                        "insert into comparison_support values(?,?,?,?,?,?)",
                        (
                            period,
                            feature,
                            tier,
                            sites,
                            counties,
                            int(sites >= 5 and counties >= 2),
                        ),
                    )
        # Reproduce all previously frozen site and recording-universe memberships.
        expected = q["panel_validation"]
        for i, period in enumerate(expected["period_order"]):
            for j, feature in enumerate(expected["feature_order"]):
                found = [
                    r[0]
                    for r in db.execute(
                        "select site_id from comparison_sites where period_id=? and feature_id=? order by site_id",
                        (period, feature),
                    )
                ]
                assert found == expected["site_sets"][2 * i + j], (period, feature)
            found = [
                r[0]
                for r in db.execute(
                    "select site_id from comparison_recording_sites where period_id=? order by site_id",
                    (period,),
                )
            ]
            assert found == expected["recording_site_sets"][i], period
        check = json.loads(
            (DOC / "calibration_verification.json").read_text(encoding="utf-8")
        )
        db.executemany(
            "insert into calibration_checks values(?,?,?,?)",
            [
                (
                    "C14:" + r["fid"],
                    *r["independent_hpd_envelope"],
                    r["maximum_endpoint_difference_years"],
                )
                for r in check["checks"]
            ],
        )
        metadata = {
            "schema": "sclara-sweden-archaeology-1",
            "source_scope": "Native source subset for all radiocarbon-linked sites, not the full national inventory.",
            "comparison_scope": "Reported posthole-minus-hearth landscapes; primary dating tier admits stated inbuilt-age and context-event uncertainty.",
            "source_bundle_sha256": file_hash(SOURCE),
            "comparison_decisions": "data/sources/archaeology.zip::comparison_decisions.json",
            "source_assertion_preservation": "Original catalogue rows and earlier exploratory panels remain verbatim; comparison_date_corrections records the recalibrated final-comparison override. Corrections and comparison exclusions are explicit; excluded measurements are not automatically reassigned to another phase.",
        }
        db.executemany(
            "insert or replace into metadata values(?,?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in metadata.items()],
        )
        db.commit()
        db.close()

    def verify():
        db = sqlite3.connect(output_path)
        assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
        assert db.execute("select count(*) from assertions").fetchone()[0] == 30296
        assert not db.execute(
            "select 1 from panel_units u left join sites s using(site_id) where s.uuid is null limit 1"
        ).fetchone()
        assert not db.execute(
            'select 1 from assertion_periods ap join assertions a using(assertion_id) join periods p using(period_id) where a.disposition!="eligible_reported_context" or a.start_year<p.start_year or a.end_year>=p.end_year_exclusive limit 1'
        ).fetchone()
        assert not db.execute(
            "select 1 from positions group by site_id having abs(sum(within_site_weight)-1)>1e-8 limit 1"
        ).fetchone()
        print(
            "Comparison panel support:",
            db.execute(
                "select period_id,feature_id,sites,counties from comparison_support where tier='primary' order by period_id,feature_id"
            ).fetchall(),
        )
        db.close()

    def final_source_reconciliation():
        """Retain catalogue assertions; apply documented site corrections to final comparisons."""
        patch = json.loads(resource("source_reconciliation.json"))
        db = sqlite3.connect(output_path)
        db.row_factory = sqlite3.Row
        db.execute(
            "create table source_reconciliation(id text primary key,details text)"
        )
        db.executemany(
            "insert into source_reconciliation values(?,?)",
            [
                (k, json.dumps(v, ensure_ascii=False))
                for k, v in patch["decisions"].items()
            ],
        )
        site = patch["corrected_site"]
        columns = list(site)
        db.execute(
            "insert into sites("
            + ",".join(columns)
            + ") values("
            + ",".join("?" for _ in columns)
            + ")",
            [site[k] for k in columns],
        )
        pos = patch["corrected_position"]
        columns = list(pos)
        db.execute(
            "insert into positions("
            + ",".join(columns)
            + ") values("
            + ",".join("?" for _ in columns)
            + ")",
            [pos[k] for k in columns],
        )
        old = patch["old_site_id"]
        new = site["site_id"]
        db.execute(
            "create table source_site_corrections(assertion_id text primary key,original_site_id text,corrected_site_id text,reason text)"
        )
        rows = db.execute(
            "select assertion_id,source_row_json from assertions where site_id=?",
            (old,),
        ).fetchall()
        assert len(rows) == 10 and all(
            json.loads(r["source_row_json"])["raa_id"] == "Örtofta 51" for r in rows
        )
        db.executemany(
            "insert into source_site_corrections values(?,?,?,?)",
            [
                (
                    r["assertion_id"],
                    old,
                    new,
                    "Original report and official register identify Örtofta 51; catalogue mapped to Örtofta31:1.",
                )
                for r in rows
            ],
        )
        for table in [
            "comparison_assertions",
            "comparison_sites",
            "comparison_recording_sites",
        ]:
            db.execute("update " + table + " set site_id=? where site_id=?", (new, old))
        db.executescript(
            "drop view period_recording_sites; create view period_recording_sites as select distinct u.period_id,u.site_id,s.longitude,s.latitude,s.position_group,s.county from panel_units u join sites s using(site_id) union select distinct u.period_id,u.site_id,s.longitude,s.latitude,s.position_group,s.county from comparison_recording_sites u join sites s using(site_id);"
        )
        db.execute(
            "insert or replace into metadata values(?,?)",
            (
                "final_source_reconciliation",
                json.dumps(
                    {
                        "source": "data/sources/archaeology.zip::source_reconciliation.json",
                        "original_assertions_and_exploratory_panels_preserved": True,
                        "new_native_geometry": "Official RAÄ SOCH polygon, projected centroid under inherited geometry convention",
                        "missing_positional_uncertainty": "Not supplied in RDF; unknown, represented without displacement under inherited rule, not measured zero error.",
                    }
                ),
            ),
        )

        def dms(value, positive, negative):
            seconds = round(abs(value) * 3600, 1)
            degrees = int(seconds // 3600)
            minutes = int((seconds - degrees * 3600) // 60)
            sec = seconds - degrees * 3600 - minutes * 60
            return f"{degrees} deg {minutes:02d} min {sec:04.1f} sec {positive if value>=0 else negative}"

        db.execute(
            "create table site_coordinates(site_id text primary key,latitude_dms text,longitude_dms text,coordinate_meaning text,source text,source_uncertainty_m real,geometry_radius_m real)"
        )
        db.executemany(
            "insert into site_coordinates values(?,?,?,?,?,?,?)",
            [
                (
                    r["site_id"],
                    dms(r["latitude"], "N", "S"),
                    dms(r["longitude"], "E", "W"),
                    "Published register geometry representative point, not an excavated sample position",
                    (
                        "Official RAÄ SOCH polygon for corrected Örtofta 51"
                        if r["site_id"] == new
                        else "RAÄ KMR native source geometry"
                    ),
                    r["source_uncertainty_m"],
                    r["geometry_radius_m"],
                )
                for r in db.execute("select * from sites")
            ],
        )
        db.commit()
        assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
        db.close()

    parent = Path(work_dir).resolve() if work_dir is not None else output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sclara_sources_", dir=parent) as name:
        DOC = Path(name).resolve()
        if not DOC.is_relative_to(parent) or DOC == parent:
            raise ValueError("Invalid temporary preparation directory")
        for member in [
            "kmr_dated_sites.sqlite",
            "intcal20.14c",
            "dictionary_decisions.json",
            "chronology_decisions.json",
        ]:
            (DOC / member).write_bytes(resource(member))
        build()
    return {"assertions": 30296}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["paternal", "archaeology"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    function = prepare_paternal if args.kind == "paternal" else prepare_archaeology
    print(function(args.source, args.output, args.work_dir))


if __name__ == "__main__":
    main()

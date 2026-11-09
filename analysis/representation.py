"""Fixed branch-resolution and shared-recording-opportunity sensitivities.

The archaeological transforms and composite contrasts remain fixed. Changed
cohorts are explicitly described; these comparisons do not supply independent
replication or recover a historical population from modern origins.
"""
from pathlib import Path
import argparse
import collections
import copy
import gzip
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.branches import coarsen_records, flexible_cohort
from analysis.infer import arrays, encode, primary_designs, read, save, sha
from analysis.landscapes import measure_landscapes
from analysis.measure import (
    Calibration, construct, feature_definitions, later_environment, later_pool,
    lineage_category, load_position_streams, measure_later, project_later, source_member,
)
from analysis.model import make_designs


def read_pool(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def reference_positions(original, release):
    coordinates = sorted({
        (r["source_lat"], r["source_lon"]) for r in original["records"]
        if r["reconciled_screen_candidate"]
    })
    positions = []
    for lat, lon in coordinates:
        key = hashlib.sha256((format(lat, ".12g")+","+format(lon, ".12g")).encode()).hexdigest()[:16]
        pid = "POS-" + key
        positions.append(dict(position_id=pid, lat=lat, lon=lon,
                              draw_seed=release["spatial"]["seed"]+":"+pid))
    for p in original["spatial_positions"]:
        q = next(x for x in positions if x["position_id"] == p["position_id"])
        if (q["lat"], q["lon"], q["draw_seed"]) != (p["lat"], p["lon"], p["draw_seed"]):
            raise ValueError("Reference reconstruction changed a retained paternal position")
    return positions


def cohort(genetic, spec, positions, controls, xy, scope):
    index = {p["position_id"]: i for i, p in enumerate(positions)}
    units, excluded = [], []
    for unit in genetic["analysis_units"]:
        labels = {
            lineage_category(genetic["lineage_paths"][node], spec["ancestral_nodes"])
            for node in unit["branch_node_ids"]
        }
        reason = None
        if len(labels) != 1:
            reason = "mixed_category"
        elif scope == "common" and not all(
            np.isfinite(controls[index[p["position_id"]]]).all() for p in unit["localities"]
        ):
            reason = "terrain"
        if reason:
            excluded.append(dict(unit_id=unit["unit_id"], reason=reason))
        else:
            units.append(dict(unit, category=labels.pop()))
    counts = np.array([len(u["localities"]) for u in units], dtype=int)
    local = np.array([index[p["position_id"]] for u in units for p in u["localities"]], dtype=int)
    return dict(
        positions=positions, units=units, excluded=excluded, local=local, counts=counts,
        starts=np.r_[0, np.cumsum(counts)[:-1]],
        y=np.array([spec["categories"].index(u["category"]) for u in units], dtype=int),
        blocks=np.array([u["dependence_block"] for u in units]),
        unit_ids=np.array([u["unit_id"] for u in units]), controls=controls, xy=xy,
    )


def restricted_records(genetic, records):
    """Keep shared source memberships inside the already constructed cut units."""
    result = dict(genetic)
    units = []
    for source in genetic["analysis_units"]:
        localities = []
        for local in source["localities"]:
            selected = sorted(set(local["record_ids"]) & records)
            if selected:
                localities.append(dict(local, record_ids=selected))
        if localities:
            for local in localities:
                local["weight"] = 1/len(localities)
            selected = sorted({r for p in localities for r in p["record_ids"]})
            curated = source.get("full_cut_curated_record_ids", source.get("curated_record_ids", []))
            units.append(dict(source, record_ids=selected, localities=localities,
                              full_cut_curated_record_ids=list(curated),
                              curated_record_ids=sorted(set(curated) & set(selected))))
    result["analysis_units"] = units
    return result


def quadratic_reference(c):
    xy = c["xy"][c["local"]] / 100000
    weights = np.repeat(1/c["counts"], c["counts"])
    mu = np.average(xy, axis=0, weights=weights)
    sd = np.sqrt(np.average((xy-mu)**2, axis=0, weights=weights))
    z = (xy-mu)/sd
    q = np.column_stack([z[:, 1]**2, z[:, 1]*z[:, 0], z[:, 0]**2])
    qm = np.average(q, axis=0, weights=weights)
    qs = np.sqrt(np.average((q-qm)**2, axis=0, weights=weights))
    return dict(coordinate_center=mu, coordinate_sd=sd, polynomial_center=qm, polynomial_sd=qs)


def cell_reference(original, positions, xy, scores, config):
    coordinate_index = {(p["lat"], p["lon"]): i for i, p in enumerate(positions)}
    local = np.array([
        coordinate_index[r["source_lat"], r["source_lon"]]
        for r in original["records"] if r["reconciled_screen_candidate"]
    ])
    exposure = scores.reshape(len(positions), -1)
    result = {}
    for origin, frozen in config["cells"].items():
        cells, inverse = np.unique(
            np.floor((xy[local]-int(origin))/config["cell_size_m"]).astype(int),
            axis=0, return_inverse=True,
        )
        centers = {tuple(cell): exposure[local][inverse == i].mean(0)
                   for i, cell in enumerate(cells)}
        centers.update({tuple(cell): np.asarray(center)
                        for cell, center in zip(frozen["coordinates"], frozen["exposure_centers"])})
        result[origin] = centers
    return result


def common_designs(c, measured, spec, centers, quadratic):
    config = copy.deepcopy(spec["common"]["design"])
    for origin, frozen in config["cells"].items():
        cells = np.unique(np.floor(
            (c["xy"][c["local"]]-int(origin))/config["cell_size_m"]
        ).astype(int), axis=0)
        frozen["coordinates"] = cells
        frozen["exposure_centers"] = np.array([centers[origin][tuple(cell)] for cell in cells])
    result = {}
    for is_quadratic in (False, True):
        for origin, d in make_designs(
            c, measured, config, quadratic=is_quadratic, quadratic_reference=quadratic
        ).items():
            key = ("quadratic_" if is_quadratic else "linear_") + origin
            result[key] = dict(
                x=d["X"], y=d["y"], counts=d["counts"], starts=c["starts"], scales=d["scales"],
                contrasts=d["vectors"], categories=spec["categories"], scope="common",
            )
    return result


def later_designs(c, measured, spec):
    config = spec["later"]["design"]
    result = {}
    for i, key in enumerate(config["jobs"]):
        x = np.column_stack([measured["controls"][i], measured["scores"][:, i]])[c["local"]]
        contrasts = {}
        for name, weights in config["lineage_contrasts"].items():
            v = np.zeros((len(spec["categories"])-1, x.shape[1]))
            v[:, -1] = weights[1:]
            contrasts[name] = v.ravel()
        result[key] = dict(
            x=x, y=c["y"], counts=c["counts"], starts=c["starts"],
            scales=np.array([2]+[.5]*(x.shape[1]-2)+[1]), categories=spec["categories"],
            contrasts=contrasts, scope="later",
        )
    return result


def support_report(c, categories, baseline_records, gate):
    records = {r for u in c["units"] for r in u["record_ids"]}
    per_category = {
        name: dict(units=int(np.sum(c["y"] == i)),
                   blocks=len(set(c["blocks"][c["y"] == i])))
        for i, name in enumerate(categories)
    }
    passed = all(
        row["units"] >= gate["minimum_units_each_category"]
        and row["blocks"] >= gate["minimum_blocks_each_category"]
        for row in per_category.values()
    )
    return dict(
        units=len(c["units"]), locality_rows=len(c["local"]),
        unique_positions=len(set(c["local"])), blocks=len(set(c["blocks"])),
        source_records=len(records), retained_ids=sorted(records),
        recovered_records=sorted(records-baseline_records),
        lost_baseline_records=sorted(baseline_records-records),
        categories=per_category, excluded=c["excluded"], support_passed=passed,
        analysis_units=c["units"],
    )


def prepare(data, measurements, specification, extension, output):
    """Build all extension designs without inspecting their fitted outcomes."""
    if (output/"prepared.json").exists():
        raise FileExistsError("Prepared extension inputs exist; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    spec, contract = read(specification), read(extension)
    original = read(data/"paternal.json")
    paternal_archive, archaeology_archive = data/"sources/paternal.zip", data/"sources/archaeology.zip"
    tree = json.loads(source_member(paternal_archive, "references/ftdna_tree.json"))
    release_bytes = source_member(paternal_archive, "decisions/broad_release_specification.json")
    release = json.loads(release_bytes)
    if hashlib.sha256(source_member(paternal_archive, "observations.zip")).hexdigest() != release["raw_sha256"]:
        raise ValueError("Paternal source differs from the frozen grouping specification")
    positions = reference_positions(original, release)
    reference_genetic = dict(original, spatial_positions=positions)
    reference_cohort = flexible_cohort(reference_genetic, archaeology_archive, spec, scope="common")
    controls, xy = reference_cohort["controls"], reference_cohort["xy"]
    calibration = Calibration(archaeology_archive)
    pool = read_pool(measurements/"common_pool.json.gz")
    common, _ = measure_landscapes(
        pool, spec["common"]["periods"], spec["common"]["families"], calibration,
        spec["common"]["rules"]["seed"], spec["common"]["position_streams"],
        positions, spec["common"]["basis"],
    )
    np.savez_compressed(output/"reference_common.npz", **common)
    print("Measured common profile at", len(positions), "screened origins", flush=True)
    pool, radii = later_pool(data/"archaeology.sqlite", archaeology_archive, spec,
                              read(data/"adjudications.json"), calibration)
    definitions = feature_definitions(spec["later"])
    registry = load_position_streams(data, spec, archaeology_archive)
    configuration = construct(pool, definitions, radii, registry)
    later = measure_later(configuration, positions, 0)
    environment = later_environment(
        positions, archaeology_archive, paternal_archive, density_reference=original["spatial_positions"]
    )
    later["scores"], later["controls"] = project_later(
        later, spec["later"]["axes"], definitions, environment
    )
    np.savez_compressed(output/"reference_later.npz", **later, environment=environment)
    baseline = {
        scope: cohort(original, spec, positions, controls, xy, scope)
        for scope in ("common", "later")
    }
    centers = cell_reference(original, positions, xy, common["scores"], spec["common"]["design"])
    quadratic = quadratic_reference(baseline["common"])
    baseline_designs = {
        **common_designs(baseline["common"], common, spec, centers, quadratic),
        **later_designs(baseline["later"], later, spec),
    }
    expected = primary_designs(measurements, spec)
    equivalence = {}
    for key, d in baseline_designs.items():
        errors = {}
        for field in ("x", "y", "counts", "scales"):
            a, b = np.asarray(d[field]), np.asarray(expected[key][field])
            if a.shape != b.shape:
                raise ValueError((key, field, "baseline shape changed", a.shape, b.shape))
            error = float(np.max(np.abs(a-b)))
            if error > 1e-10:
                raise ValueError((key, field, "baseline design changed", error))
            errors[field] = error
        for name, vector in d["contrasts"].items():
            if not np.array_equal(vector, expected[key]["contrasts"][name]):
                raise ValueError((key, name, "baseline contrast changed"))
        equivalence[key] = errors
    print("All eight baseline designs reproduce on the extended reference", flush=True)
    baseline_records = {
        scope: {r for u in c["units"] for r in u["record_ids"]}
        for scope, c in baseline.items()
    }
    jobs, reports, posterior, withheld = {}, {}, [], []
    unit_designs = {}
    for cap in contract["branch_resolution"]["caps"]:
        genetic, mapping = coarsen_records(
            original, tree, release, hashlib.sha256(release_bytes).hexdigest(),
            paternal_archive.name, spec, cap,
        )
        for view in contract["branch_resolution"]["cohorts"]:
            for scope in ("common", "later"):
                g = genetic if view == "full_cut" else restricted_records(genetic, baseline_records[scope])
                c = cohort(g, spec, positions, controls, xy, scope)
                prefix = f"cap{cap}_{view}_{scope}"
                report = support_report(
                    c, spec["categories"], baseline_records[scope],
                    contract["branch_resolution"]["support_gate"],
                )
                reports[prefix] = dict(report, cap=cap, view=view, scope=scope, mapping=mapping)
                print(prefix, report["units"], "units;", report["source_records"],
                      "records; support", report["support_passed"], flush=True)
                if not report["support_passed"]:
                    withheld.append(prefix)
                    continue
                designs = (
                    common_designs(c, common, spec, centers, quadratic)
                    if scope == "common" else later_designs(c, later, spec)
                )
                centroids = np.add.reduceat(xy[c["local"]], c["starts"], axis=0)/c["counts"][:, None]
                cells = np.floor(centroids/200000).astype(int)
                unit_designs[prefix] = dict(
                    blocks=c["blocks"], cells=[f"{a},{b}" for a,b in cells],
                    unit_ids=c["unit_ids"], counts=c["counts"], local=c["local"],
                )
                for base, d in designs.items():
                    name = f"cap{cap}_{view}_{base}"
                    jobs[name] = dict(
                        d, job_id=name, base_design=base, cohort_key=prefix, cap=cap, view=view,
                        seed=contract["branch_resolution"]["posterior_seed_base"]+len(jobs),
                        scope="Fixed-profile branch-resolution sensitivity; "+scope,
                    )
                    if view == "full_cut" and base in contract["branch_resolution"]["posterior_designs"]:
                        posterior.append(name)
    reference = dict(
        positions=positions, quadratic=quadratic,
        regional_centers={
            origin: [dict(coordinates=list(cell), center=value) for cell,value in sorted(values.items())]
            for origin,values in centers.items()
        },
        baseline_equivalence=equivalence,
        center_rule=contract["branch_resolution"]["regional_reference"],
        later_density_reference_sha256=sha(data/"paternal.json"),
    )
    save(output/"reference.json", reference)
    save(output/"selection.json", dict(cohorts=reports, withheld=withheld))
    save(output/"unit_designs.json", unit_designs)
    save(output/"designs.json", dict(jobs=jobs, posterior_jobs=posterior))
    save(output/"extension_specification.json", contract)
    save(output/"prepared.json", dict(
        input_hashes={str(p):sha(p) for p in (
            data/"paternal.json", data/"adjudications.json", specification, extension,
            measurements/"common_pool.json.gz", paternal_archive, archaeology_archive
        )},
        outputs={name:sha(output/name) for name in (
            "reference.json","selection.json","unit_designs.json","designs.json","extension_specification.json"
        )},
        program_sha256=sha(Path(__file__)),
        baseline_equivalence=equivalence, conditional_designs=len(jobs),
        posterior_jobs=posterior, scope="Design construction only; no extension fit performed.",
    ))



def array_design(d):
    result = dict(d)
    for field in ("x", "y", "counts", "starts", "scales"):
        result[field] = np.asarray(d[field])
    result["contrasts"] = {k: np.asarray(v) for k,v in d["contrasts"].items()}
    return result


def fit_modes(output, specification, extension):
    """Fit the complete declared design family and export the sampled subset."""
    from analysis.infer import fit_design
    import io
    import shutil
    import sqlite3
    import time

    prepared = read(output/"prepared.json")
    for name, expected in prepared["outputs"].items():
        if sha(output/name) != expected:
            raise ValueError("Changed prepared extension input: "+name)
    if sha(extension) != prepared["input_hashes"][str(extension)]:
        raise ValueError("Extension specification changed after preparation")
    if (output/"modes.json").exists():
        raise FileExistsError("Completed extension modes exist")
    designs = read(output/"designs.json")
    modes, checks, sampled = {}, {}, {}
    db = sqlite3.connect(output/"conditional.sqlite")
    db.execute("create table if not exists metadata(key text primary key,value text not null)")
    db.execute("create table if not exists fits(job text,family text,replicate text,status text,details text,arrays blob,sha256 text,primary key(job,family,replicate))")
    if db.execute("select count(*) from fits").fetchone()[0]:
        raise FileExistsError("Partial mode fits require explicit inspection before rerunning")
    db.execute("insert into metadata values(?,?)",("input_hashes",encode(dict(
        prepared=sha(output/"prepared.json"), model=sha(Path(__file__).with_name("model.py")),
        implementation=sha(Path(__file__)),
    ))))
    db.commit()
    for name, raw in designs["jobs"].items():
        started = time.monotonic()
        d = array_design(raw)
        model, fit, result = fit_design(d)
        detail = dict(
            units=len(d["y"]), categories=dict(collections.Counter(
                d["categories"][i] for i in d["y"])),
            contrast_names=list(d["contrasts"]), gradient_max=fit["gradient"],
            minimum_curvature=fit["curvature"], iterations=fit["iterations"],
            seconds=time.monotonic()-started, uncertainty="Conditional mode and inverse-Hessian standard error",
        )
        stream=io.BytesIO()
        np.savez_compressed(stream,**result)
        blob=stream.getvalue()
        db.execute("insert into fits values(?,?,?,?,?,?,?)",(
            name,"conditional","baseline","passed",encode(detail),blob,hashlib.sha256(blob).hexdigest()
        ))
        db.commit()
        modes[name]=result
        checks[name]=detail
        if name in designs["posterior_jobs"]:
            rng=np.random.default_rng(d["seed"])
            probes=[]
            for v in (fit["mode"],fit["mode"]+rng.normal(0,.02,len(fit["mode"])),
                      fit["mode"]+rng.normal(0,.1,len(fit["mode"]))):
                value,gradient=model.evaluate(v)
                probes.append(dict(v=v,log_density=-value,gradient=-gradient))
            sampled[name]=dict(d,density_checks=probes)
        print("Fitted",name,"gradient",fit["gradient"],flush=True)
    db.close()
    save(output/"modes.json",modes)
    save(output/"numerical_checks.json",checks)
    posterior=output/"posterior"
    posterior.mkdir(exist_ok=True)
    save(posterior/"designs.json",dict(jobs=sampled))
    save(posterior/"specification.json",dict(base=read(specification),extensions=read(extension)))
    shutil.copyfile(Path(__file__).with_name("mixture.stan"),posterior/"mixture.stan")
    save(posterior/"design_freeze.json",dict(
        specification_sha256=sha(posterior/"specification.json"),
        designs_sha256=sha(posterior/"designs.json"),stan_sha256=sha(posterior/"mixture.stan"),
        sampler_sha256=sha(Path(__file__).with_name("sample.R")),
    ))
    print("Exported",len(sampled),"current-profile branch posterior jobs",flush=True)




def fit_shared_support(data, measurements, specification, extension, output):
    """Apply the declared whole-unit shared-opportunity restriction."""
    from analysis.infer import fit_design
    import io
    import sqlite3
    spec, contract = read(specification), read(extension)["common_support"]
    target=output/"common_support"
    if (target/"support.json").exists():
        raise FileExistsError("Completed common-support evaluation exists")
    target.mkdir(parents=True,exist_ok=True)
    c=arrays(measurements/"common_cohort.npz")
    measured=arrays(measurements/"common.npz")
    indices=[spec["common"]["design"]["periods"].index(p) for p in contract["periods"]]
    probability=(measured["recording_draws"] >= np.log1p(
        contract["minimum_recorded_sites_each_period"]
    )).mean(0)[:,indices]
    position_keep=(probability >= contract["minimum_draw_fraction_each_period"]).all(1)
    keep=np.array([
        position_keep[c["local"][start:start+n]].all()
        for start,n in zip(c["starts"],c["counts"])
    ])
    categories={
        category:dict(units=int(np.sum(keep & (c["y"]==i))),
                      blocks=len(set(c["blocks"][keep & (c["y"]==i)])))
        for i,category in enumerate(spec["categories"])
    }
    gate=contract["support_gate"]
    passed=all(v["units"]>=gate["minimum_units_each_category"]
               and v["blocks"]>=gate["minimum_blocks_each_category"] for v in categories.values())
    support=dict(
        support_passed=passed, units=int(keep.sum()), omitted_units=int((~keep).sum()),
        categories=categories, kept_unit_ids=c["unit_ids"][keep],
        omitted_unit_ids=c["unit_ids"][~keep], source_periods=contract["periods"],
        local_recording_probability=probability, position_keep=position_keep,
        rule=contract["unit_rule"], interpretation=contract["interpretation"],
    )
    save(target/"support.json",support)
    print("Shared recording support:",int(keep.sum()),"units; gate",passed,flush=True)
    if not passed:
        return
    original=primary_designs(measurements,spec)
    common=make_designs(c,measured,spec["common"]["design"])
    jobs,modes,weight_reference={}, {}, {}
    db=sqlite3.connect(target/"conditional.sqlite")
    db.execute("create table metadata(key text primary key,value text not null)")
    db.execute("create table fits(job text,family text,replicate text,status text,details text,arrays blob,sha256 text,primary key(job,family,replicate))")
    db.execute("insert into metadata values(?,?)",("input_hashes",encode(dict(
        specification=sha(specification),extension=sha(extension),
        measurements=sha(measurements/"common.npz"),cohort=sha(measurements/"common_cohort.npz"),
        implementation=sha(Path(__file__)),model=sha(Path(__file__).with_name("model.py")),
    ))))
    dep=spec["common"]["inference"]["sensitivity"]["crossed_dependence"]
    block_labels,block_index=np.unique(c["blocks"],return_inverse=True)
    for base in contract["designs"]:
        before=original[base]
        d=dict(before,x=before["x"][np.repeat(keep,c["counts"])],
               y=before["y"][keep],counts=before["counts"][keep])
        d["starts"]=np.r_[0,np.cumsum(d["counts"])[:-1]]
        name="shared_support_"+base
        d.update(job_id=name,base_design=base,scope="Shared local recording-opportunity sensitivity")
        _,fit,result=fit_design(d)
        detail=dict(units=len(d["y"]),contrast_names=list(d["contrasts"]),
                    gradient_max=fit["gradient"],minimum_curvature=fit["curvature"],
                    uncertainty="Conditional mode and inverse-Hessian standard error")
        stream=io.BytesIO();np.savez_compressed(stream,**result);blob=stream.getvalue()
        db.execute("insert into fits values(?,?,?,?,?,?,?)",(
            name,"conditional","baseline","passed",encode(detail),blob,hashlib.sha256(blob).hexdigest()
        ));db.commit()
        jobs[name]=d;modes[name]=result
        if base in contract["crossed_dependence"]["designs"]:
            origin=base.split("_")[-1]
            cell_labels,cell_index=np.unique(common[origin]["centroid_labels"],return_inverse=True)
            weight_reference[name]=dict(
                block_seed=dep["block_seed"],cell_seed=dep["cell_seeds"][origin],
                blocks=len(block_labels),cells=len(cell_labels),
                block_index=block_index[keep],cell_index=cell_index[keep],
                replicates=contract["crossed_dependence"]["replicates_per_design"],
            )
        print("Fitted",name,"gradient",fit["gradient"],flush=True)
    db.close()
    save(target/"designs.json",dict(jobs=jobs))
    save(target/"modes.json",modes)
    save(target/"weight_reference.json",weight_reference)


def initialize_reweighting(directory, section):
    global _REWEIGHT, _LIMIT
    _LIMIT=threadpool_limits(limits=1)
    directory=Path(directory)
    designs=read(directory/"designs.json")["jobs"]
    modes=read(directory/"modes.json")
    contract=read(directory/"extension_specification.json") if section=="branches" else None
    weights={}
    if section=="branches":
        units=read(directory/"unit_designs.json")
        cfg=contract["branch_resolution"]["crossed_dependence"]
        chosen={name:d for name,d in designs.items()
                if d["view"]=="full_cut" and d["base_design"] in cfg["designs"]}
        for name,d in chosen.items():
            c=units[d["cohort_key"]]
            blocks,bi=np.unique(c["blocks"],return_inverse=True)
            cells,ci=np.unique(c["cells"],return_inverse=True)
            seed=cfg["seed_base"]+d["cap"]*100+(1000 if d["cohort_key"].endswith("_later") else 0)
            weights[name]=dict(
                block=np.random.default_rng(seed).exponential(size=(cfg["replicates_per_design"],len(blocks))),
                cell=np.random.default_rng(seed+1).exponential(size=(cfg["replicates_per_design"],len(cells))),
                block_index=bi,cell_index=ci,
            )
    else:
        chosen={}
        for name,r in read(directory/"weight_reference.json").items():
            chosen[name]=designs[name]
            weights[name]=dict(
                block=np.random.default_rng(r["block_seed"]).exponential(size=(r["replicates"],r["blocks"])),
                cell=np.random.default_rng(r["cell_seed"]).exponential(size=(r["replicates"],r["cells"])),
                block_index=np.array(r["block_index"]),cell_index=np.array(r["cell_index"]),
            )
    _REWEIGHT=dict(
        designs={name:array_design(d) for name,d in chosen.items()},
        modes={name:np.asarray(modes[name]["mode"]) for name in chosen},weights=weights,
    )


def reweight_one(job):
    from analysis.infer import fit_design
    import io
    import time
    name,replicate=job
    d=_REWEIGHT["designs"][name];r=_REWEIGHT["weights"][name]
    weights=r["block"][replicate,r["block_index"]]*r["cell"][replicate,r["cell_index"]]
    weights*=len(d["y"])/weights.sum()
    errors=[];started=time.monotonic()
    for initial in (_REWEIGHT["modes"][name],None):
        try:
            _,fit,result=fit_design(d,initial,weights)
            break
        except (ValueError,np.linalg.LinAlgError) as error:
            errors.append(str(error))
    else:
        return job,"failed",dict(errors=errors,units=len(d["y"])),None
    detail=dict(
        units=len(d["y"]),contrast_names=list(d["contrasts"]),
        gradient_max=fit["gradient"],minimum_curvature=fit["curvature"],
        iterations=fit["iterations"],numerical_attempt_errors=errors,
        seconds=time.monotonic()-started,
        uncertainty="Conditional mode under crossed block/cell reweighting; not a posterior draw",
    )
    stream=io.BytesIO();np.savez_compressed(stream,**result)
    return job,"passed",detail,stream.getvalue()


def reweight(directory,section,workers):
    """Resume the finite dependence sensitivity; retain failures explicitly."""
    import concurrent.futures
    import sqlite3
    initialize_reweighting(directory,section)
    identity=dict(
        designs=sha(directory/"designs.json"),modes=sha(directory/"modes.json"),
        weights=sha(directory/("unit_designs.json" if section=="branches" else "weight_reference.json")),
        contract=sha(directory/"extension_specification.json") if section=="branches" else None,
        model=sha(Path(__file__).with_name("model.py")),implementation=sha(Path(__file__)),
    )
    jobs=[(name,i) for name,r in _REWEIGHT["weights"].items() for i in range(len(r["block"]))]
    db=sqlite3.connect(directory/"conditional.sqlite")
    previous=db.execute("select value from metadata where key='reweight_hashes'").fetchone()
    if previous and json.loads(previous[0])!=identity:
        raise ValueError("Reweighting inputs or implementation changed")
    db.execute("insert or ignore into metadata values(?,?)",("reweight_hashes",encode(identity)))
    db.commit()
    done={(name,int(rep)) for name,rep in db.execute(
        "select job,replicate from fits where family='crossed_dependence'"
    )}
    pending=[job for job in jobs if job not in done]
    print(section,"dependence fits:",len(jobs),"total;",len(pending),"pending",flush=True)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,initializer=initialize_reweighting,initargs=(str(directory),section)
    ) as pool:
        futures=[pool.submit(reweight_one,job) for job in pending]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            (name,rep),status,detail,blob=future.result()
            db.execute("insert into fits values(?,?,?,?,?,?,?)",(
                name,"crossed_dependence",str(rep),status,encode(detail),blob,
                hashlib.sha256(blob).hexdigest() if blob else None,
            ));db.commit()
            if i%50==0 or i==len(pending):
                print(section,"dependence fits",i,"/",len(pending),flush=True)
    failed=list(db.execute("select job,replicate,details from fits where status='failed'"))
    db.close()
    if failed:
        raise RuntimeError(("Numerical failures retained for review",failed))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "fit-modes", "common-support", "reweight-branches", "reweight-support"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--measurements", type=Path, default=Path("build/measurements"))
    parser.add_argument("--specification", type=Path, default=Path("analysis/specification.json"))
    parser.add_argument("--extension", type=Path, default=Path("analysis/extension_specification.json"))
    parser.add_argument("--output", type=Path, default=Path("build/representation"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            prepare(args.data,args.measurements,args.specification,args.extension,args.output)
        elif args.action == "fit-modes":
            fit_modes(args.output,args.specification,args.extension)
        elif args.action == "common-support":
            fit_shared_support(args.data,args.measurements,args.specification,args.extension,args.output)
        else:
            section="branches" if args.action == "reweight-branches" else "support"
            directory=args.output if section == "branches" else args.output/"common_support"
            reweight(directory,section,args.workers)


if __name__ == "__main__":
    main()

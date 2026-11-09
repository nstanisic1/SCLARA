"""Collect complete fresh calculations in the retained-results directory layout."""

from pathlib import Path
import argparse
import json
import shutil
from threadpoolctl import threadpool_limits
from .draws import compact_directory
from .results import connect, read, save, sha, verify_conditionals, verify_posterior


POSTERIOR_FILES = (
    ".json",
    "_draws.csv.gz",
    "_contrasts.csv",
    "_parameter_diagnostics.csv",
)


def collect(build, output):
    """Require complete diagnosed batches; leave the published snapshots untouched."""
    if output.exists():
        raise FileExistsError(f"Use a new collection directory: {output}")
    copies = []
    checks = {}
    for family, directory, count in (
        ("current", "inference", 8),
        ("appraisal", "appraisal", 10),
        ("branches", "branches", 4),
    ):
        folder = build / directory
        designs = read(folder / "designs.json")["jobs"]
        if len(designs) != count:
            raise ValueError((family, "Unexpected posterior family size"))
        checks[family] = verify_posterior(folder / "results", folder / "designs.json")
        freeze = read(folder / "design_freeze.json")
        for filename, key in (
            ("designs.json", "designs_sha256"),
            ("mixture.stan", "stan_sha256"),
            ("specification.json", "specification_sha256"),
        ):
            if sha(folder / filename) != freeze[key]:
                raise ValueError((family, filename, "Changed frozen input"))
        for job in designs:
            receipt = read(folder / "results" / (job + ".json"))
            if receipt["design_sha256"] != sha(folder / "designs.json") or receipt[
                "model_sha256"
            ] != sha(folder / "mixture.stan"):
                raise ValueError(
                    (family, job, "Receipt does not match generated design")
                )
            for suffix in POSTERIOR_FILES:
                copies.append(
                    (folder / "results" / (job + suffix), Path(family) / (job + suffix))
                )
    conditional = build / "inference/conditional.sqlite"
    checks["conditional"] = verify_conditionals(
        conditional, build / "inference/designs.json"
    )
    copies.append((conditional, Path("current/conditional.sqlite")))
    for name in ("modes.json", "support.json", "single_site_omissions.json"):
        copies.append((build / "appraisal" / name, Path("appraisal") / name))
    for source, target, tables in (
        (
            "benchmarks/benchmarks.sqlite",
            "benchmarks.sqlite",
            {"fits": 19400, "summaries": 67},
        ),
        (
            "paired/paired.sqlite",
            "chronology.sqlite",
            {
                "paired_fit": 258,
                "paired_measurement": 129,
                "chronology_membership": 2316,
                "paired_summary": 240,
            },
        ),
    ):
        path = build / source
        with connect(path) as db:
            counts = {
                table: db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                for table in tables
            }
            fit_table = "fits" if target == "benchmarks.sqlite" else "paired_fit"
            statuses = dict(
                db.execute(f'SELECT status,COUNT(*) FROM "{fit_table}" GROUP BY status')
            )
            if counts != tables or statuses != {"passed": tables[fit_table]}:
                raise ValueError(
                    (target, "Incomplete or failed calculation", counts, statuses)
                )
        checks[target] = {"tables": counts, "statuses": statuses}
        copies.append((path, Path(target)))
    copies.append((build / "northern/evidence.json", Path("northern.json")))
    # Check every file before creating output, avoiding a silently partial collection.
    inputs = [
        {"path": dst.as_posix(), "bytes": src.stat().st_size, "sha256": sha(src)}
        for src, dst in copies
    ]
    if len({item["path"] for item in inputs}) != len(inputs):
        raise ValueError("Duplicate collection destination")
    from .extensions import collect as collect_extensions, verify_source_batches

    extension_checks = verify_source_batches(build)
    output.mkdir(parents=True)
    for source, destination in copies:
        destination = output / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    extension_checks = collect_extensions(build, output, checks=extension_checks)
    conversions = {
        family: compact_directory(output / family)
        for family in ("current", "appraisal", "branches")
    }
    inputs = [
        {
            "path": p.relative_to(output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha(p),
        }
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    provenance = {
        "schema": "sclara-fresh-results-1",
        "scope": "Fresh specified calculations. The earlier outcome-informed historical screen is an archived development record and is not fitted by this collection.",
        "checks": checks,
        "lossless_draw_storage": conversions,
        "extensions": extension_checks,
        "files": inputs,
    }
    save(output / "provenance.json", provenance)
    return {
        "files": len(inputs),
        "bytes": sum(item["bytes"] for item in inputs),
        "posterior_jobs": 30,
        "conditional_fits": 5862,
        "evaluation_method_records": 4800,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--output", type=Path, default=Path("build/results"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(collect(args.build, args.output), indent=2))


if __name__ == "__main__":
    main()

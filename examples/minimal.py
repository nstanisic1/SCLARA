"""Run a synthetic source-to-measurement-to-contrast example in seconds.

All sites, origins, dates and labels are invented. Calendar distributions are
supplied directly; they are not calibrated radiocarbon observations. The fixed
building-minus-hearth axis is specified for teaching, not estimated by PCA.
"""

from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from analysis.landscapes import measure_landscapes
from analysis.model import LabelMixture


class SuppliedDates:
    """Synthetic calendar likelihoods indexed by nominal sample-age keys."""

    def distribution(self, age, error):
        years = {2450: [-550, -500, -450], 1450: [450, 500, 550], 1950: [-50, 0, 50]}
        return np.array(years[age]), np.array([0.2, 0.6, 0.2])


def run(output, all_multiple=False):
    rng = np.random.default_rng(613704)
    periods = [("earlier", -1000, 0), ("later", 0, 1000)]
    families = ["building", "hearth"]
    sites, geometry, assertions = {}, [], []
    centers = [(58.1, 13.6), (58.7, 15.3), (58.6, 13.8), (58.0, 15.0)]
    for period in range(2):
        for family, label in enumerate(families):
            for member in range(3):
                site = "S" + str(len(sites) + 1)
                latitude, longitude = np.array(
                    centers[2 * period + family]
                ) + rng.normal(0, 0.12, 2)
                sites[site] = dict(
                    latitude=float(latitude),
                    longitude=float(longitude),
                    source_uncertainty_m=1000.0,
                    position_group=site,
                )
                geometry.append(
                    dict(
                        position_id=site + "-point",
                        site_id=site,
                        latitude=float(latitude),
                        longitude=float(longitude),
                    )
                )
                assertions.append(
                    dict(
                        assertion_id=site + "-assay",
                        site_id=site,
                        family=label,
                        calibration_key=site + "-assay",
                        age=2450 if period == 0 else 1450,
                        error=30,
                    )
                )
    # One boundary-dated context shares its date across both reported features.
    site = "S13"
    sites[site] = dict(
        latitude=58.35, longitude=14.5, source_uncertainty_m=2000.0, position_group=site
    )
    geometry.append(
        dict(position_id=site + "-point", site_id=site, latitude=58.35, longitude=14.5)
    )
    for family in families:
        assertions.append(
            dict(
                assertion_id=site + "-" + family,
                site_id=site,
                family=family,
                calibration_key=site + "-shared-assay",
                age=1950,
                error=30,
            )
        )
    pool = dict(sites=sites, geometry=geometry, rows=assertions)
    positions = [
        dict(position_id="P" + str(i), lat=float(lat), lon=float(lon))
        for i, (lat, lon) in enumerate(
            rng.uniform([57.8, 13.0], [58.9, 15.8], size=(240 if all_multiple else 144, 2))
        )
    ]
    # Default: 24 alternative-origin units; optional: all 120 have two origins.
    counts = np.full(120, 2, dtype=int) if all_multiple else np.r_[np.ones(96, dtype=int), np.full(24, 2, dtype=int)]
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    basis = dict(
        center=[0.0, 0.0],
        scale=[0.2, 0.2],
        loadings=[[1 / np.sqrt(2)], [-1 / np.sqrt(2)]],
        eigenvalues=[1.0],
    )
    measurements, support = measure_landscapes(
        pool,
        periods,
        families,
        SuppliedDates(),
        seed=613705,
        streams={},
        positions=positions,
        basis=basis,
        n=64,
        position_seed_prefix="SCLARA-SYNTHETIC-EXAMPLE",
    )
    coordinates = np.array([[p["lon"], p["lat"]] for p in positions])
    standardized = (coordinates - coordinates.mean(0)) / coordinates.std(0)
    x = np.column_stack(
        [np.ones(len(positions)), standardized, measurements["scores"][:, :, 0]]
    )
    generating = np.array([0.0, -0.1, 0.1, 0.7, 0.25])
    # Average probabilities within each unit before drawing its single label.
    probability = np.add.reduceat(expit(x @ generating), starts) / counts
    labels = (rng.random(len(counts)) < probability).astype(int)
    fit = LabelMixture(
        x, labels, counts, [2.0, 0.5, 0.5, 1.0, 1.0], categories=2
    ).optimize()
    vectors = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, -1]])
    mean = vectors @ fit["mode"]
    covariance = vectors @ fit["covariance"] @ vectors.T
    sd = np.sqrt(np.diag(covariance))
    contrasts = [
        dict(
            contrast=name,
            mode=float(m),
            conditional_sd=float(s),
            conditional_95_interval=[float(m - 1.96 * s), float(m + 1.96 * s)],
        )
        for name, m, s in zip(
            ["B_minus_A_earlier", "B_minus_A_later", "earlier_minus_later"], mean, sd
        )
    ]
    summary = dict(
        synthetic=True,
        units=len(counts),
        all_multiple_origins=all_multiple,
        locality_rows=int(counts.sum()),
        sites=len(sites),
        categories=["A", "B"],
        category_counts=np.bincount(labels, minlength=2).tolist(),
        basis="Fixed illustrative building-minus-hearth contrast; the same scale in both periods.",
        uncertainty="Inverse-Hessian conditional intervals; not sampled posterior intervals or calibrated historical confidence.",
        contrasts=contrasts,
        contrast_covariance=covariance.tolist(),
        numerical_checks={
            key: fit[key] for key in ["gradient", "curvature", "iterations"]
        },
        support=support,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources.json").write_text(
        json.dumps(
            dict(
                synthetic=True,
                pool=pool,
                positions=positions,
                periods=periods,
                families=families,
                basis=basis,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "arrays.npz",
        x=x,
        labels=labels,
        counts=counts,
        mode=fit["mode"],
        covariance=fit["covariance"],
        **measurements
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                synthetic=True,
                units=len(counts),
                gradient=fit["gradient"],
                contrasts=contrasts,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/example"))
    parser.add_argument("--all-multiple", action="store_true", help="Give every synthetic unit two alternative origins")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.all_multiple)


if __name__ == "__main__":
    main()

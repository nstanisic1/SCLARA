"""Dated archaeological proximities and projection through a fixed profile basis.

Study configurations supply periods, feature families, source eligibility and
calibration distributions. These functions neither load project files nor fit
lineage models. Distances are in kilometres; source positional radii are metres.
"""
from __future__ import annotations

import collections
import hashlib
import math
import random

import numpy as np


def distance(a, b):
    """Return pairwise great-circle distances for latitude/longitude arrays."""
    p, q = np.radians(a), np.radians(b)
    value = np.sin((p[:, 0, None] - q[None, :, 0]) / 2) ** 2
    value += (
        np.cos(p[:, 0, None])
        * np.cos(q[None, :, 0])
        * np.sin((p[:, 1, None] - q[None, :, 1]) / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def position_draws(latitude, longitude, radius_m, seed, n):
    """Draw reproducible displacements using a uniform-disk radial law."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        delta = radius_m / 1000 * math.sqrt(rng.random()) / 6371.0088
        theta = 2 * math.pi * rng.random()
        p, longitude_radians = math.radians(latitude), math.radians(longitude)
        q = math.asin(
            math.sin(p) * math.cos(delta)
            + math.cos(p) * math.sin(delta) * math.cos(theta)
        )
        z = longitude_radians + math.atan2(
            math.sin(theta) * math.sin(delta) * math.cos(p),
            math.cos(delta) - math.sin(p) * math.sin(q),
        )
        draws.append([math.degrees(q), (math.degrees(z) + 180) % 360 - 180])
    return np.array(draws)


def measure_landscapes(
    pool,
    periods,
    families,
    calibration,
    seed,
    streams,
    positions,
    basis,
    n=64,
    proximity_scale_km=50.0,
    recording_radius_km=75.0,
    position_seed_prefix="SCLARA-FOOTPRINT-1",
    paternal_draws=None,
    origin_weights=None,
    omitted_sites=(),
    candidate_pairs=(),
):
    """Measure dated feature proximity and project through an existing basis.

    ``pool`` contains source assertions, sites and representative geometry rows.
    Assertions supply a shared calibration key, laboratory age/error, site and
    feature family. Geometry rows supply stable position IDs and coordinates.
    ``calibration.distribution(age, error)`` returns calendar years and their
    probabilities. ``streams`` maps existing position IDs to (latitude,
    longitude, radius, seed), preserving earlier draws when sources are added.
    ``positions`` are represented paternal-origin locations, without labels.

    Periods are named half-open calendar intervals. The caller supplies the
    frozen family order and basis: center, scale, loadings and eigenvalues.
    The basis is not estimated here. The recording measure counts unique
    position groups. Multiple geometries of one site use their strongest
    proximity. The seed prefix preserves the inherited Swedish random streams;
    another study can supply its own fixed namespace.

    Optional paternal draws replace nominal origin coordinates in each paired
    draw, including recording proximity. Whole-site omissions recompute nearest
    features. Optional origin weights produce an outcome-independent site
    contribution: the weighted whole-site deletion loss, averaged equally over
    draws, periods and feature families. Candidate site pairs additionally
    receive their exact joint deletion losses, using the nearest three sites.

    No eligible site gives zero proximity; one eligible site supplies its own
    proximity, with zero proximity after its deletion. These are observations
    within the supplied source universe, not imputations for unknown coverage.
    The caller must retain coverage and support qualifications. Draw-level
    support counts expose empty and singleton cases separately from exposure.

    Return numerical arrays and metadata defining site order and draw support.
    Unknown source positional error must remain distinguished from measured
    zero error in the supplied provenance, even when no displacement is used.
    """
    if n < 1 or proximity_scale_km <= 0 or recording_radius_km <= 0:
        raise ValueError("Draw count and distance scales must be positive")
    if not periods or not families or len(set(families)) != len(families):
        raise ValueError("Nonempty periods and distinct feature families are required")
    if not positions:
        raise ValueError("At least one represented origin is required")
    if candidate_pairs and origin_weights is None:
        raise ValueError("Pair deletion contributions require origin weights")

    sites = pool["sites"]
    site_ids = sorted(sites)
    site_index = {site: i for i, site in enumerate(site_ids)}
    pairs = [tuple(sorted(pair)) for pair in candidate_pairs]
    if len(set(pairs)) != len(pairs) or any(
        len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(sites)
        for pair in pairs
    ):
        raise ValueError("Candidate pairs must be distinct pairs of included sites")
    pair_indices = np.array(
        [[site_index[a], site_index[b]] for a, b in pairs], dtype=int
    ).reshape(-1, 2)
    pair_keys = pair_indices[:, 0] * len(site_ids) + pair_indices[:, 1]
    pair_order = np.argsort(pair_keys)
    sorted_pair_keys = pair_keys[pair_order]
    pair_extra = np.zeros(len(pairs))
    geometry = pool["geometry"]
    assignments = {}
    for row in pool["rows"]:
        key = row["calibration_key"]
        if key in assignments:
            continue
        years, probability = calibration.distribution(row["age"], row["error"])
        cumulative = np.cumsum(probability)
        random_seed = int.from_bytes(
            hashlib.sha256((str(seed) + ":" + key).encode()).digest()[:8], "little"
        )
        quantiles = np.random.default_rng(random_seed).random(n)
        dates = years[np.searchsorted(cumulative, quantiles)]
        assignments[key] = np.array(
            [
                next(
                    (i for i, (_, low, high) in enumerate(periods) if low <= date < high),
                    -1,
                )
                for date in dates
            ],
            dtype="int16",
        )

    groups = collections.defaultdict(list)
    for item in geometry:
        groups[item["latitude"], item["longitude"]].append(item)
    draws = {}
    new_streams = []
    for coordinates, group in groups.items():
        previous = [streams[item["position_id"]] for item in group if item["position_id"] in streams]
        if previous:
            chosen = previous[0]
            if not all(value == chosen for value in previous):
                raise ValueError("Conflicting frozen positional streams")
            latitude, longitude, radius, stream_seed = chosen
        else:
            latitude, longitude = coordinates
            radius = max(sites[item["site_id"]]["source_uncertainty_m"] or 0 for item in group)
            position_ids = sorted(item["position_id"] for item in group)
            stream_seed = position_seed_prefix + ":" + ",".join(position_ids)
            new_streams.append(
                dict(position_ids=position_ids, center=[latitude, longitude], radius_m=radius, seed=stream_seed)
            )
        scenario = position_draws(latitude, longitude, radius, stream_seed, n)
        for item in group:
            draws[item["position_id"]] = scenario

    geometry_draws = np.array(
        [draws[item["position_id"]] for item in geometry], dtype=float
    ).reshape(len(geometry), n, 2).transpose(1, 0, 2)
    geometry_sites = np.array([site_index[item["site_id"]] for item in geometry], dtype=int)
    order = np.argsort(geometry_sites, kind="stable")
    ordered_sites = geometry_sites[order]
    starts = (
        np.r_[0, np.flatnonzero(np.diff(ordered_sites)) + 1]
        if len(ordered_sites) else np.array([], dtype=int)
    )
    if not np.array_equal(ordered_sites[starts], np.arange(len(site_ids))):
        raise ValueError("Every included site requires representative geometry")
    nominal = np.array([[position["lat"], position["lon"]] for position in positions])
    centers = np.array(
        [[sites[site]["latitude"], sites[site]["longitude"]] for site in site_ids],
        dtype=float,
    ).reshape(-1, 2)
    recorded = distance(nominal, centers) <= recording_radius_km
    proximity = np.zeros((n, len(nominal), len(periods), len(families)))
    recording = np.zeros((n, len(nominal), len(periods)))
    nearest = np.full(proximity.shape, -1, dtype="int32")
    support = []
    omitted = set(omitted_sites)
    if not omitted.issubset(sites):
        raise ValueError("Omitted sites must belong to the supplied source pool")
    if paternal_draws is not None:
        paternal_draws = np.asarray(paternal_draws, dtype=float)
        if paternal_draws.shape != (n, len(nominal), 2) or not np.isfinite(paternal_draws).all():
            raise ValueError("Paternal draws must supply finite latitude/longitude pairs for every draw and origin")
    contribution = np.zeros(len(site_ids))
    if origin_weights is not None:
        origin_weights = np.asarray(origin_weights, dtype=float)
        if origin_weights.shape != (len(nominal),) or not np.all(np.isfinite(origin_weights) & (origin_weights >= 0)) or origin_weights.sum() <= 0:
            raise ValueError("Origin weights must be finite and nonnegative with positive total")
        contribution_weights = origin_weights / (origin_weights.sum() * n * len(periods) * len(families))

    for draw in range(n):
        coordinates = nominal if paternal_draws is None else paternal_draws[draw]
        if paternal_draws is not None:
            recorded = distance(coordinates, centers) <= recording_radius_km
        site_proximity = np.exp(-distance(coordinates, geometry_draws[draw]) / proximity_scale_km)
        if len(starts):
            site_proximity = np.maximum.reduceat(site_proximity[:, order], starts, axis=1)
        feature_sites = [[set() for _ in families] for _ in periods]
        recording_sites = [set() for _ in periods]
        for row in pool["rows"]:
            if row["site_id"] in omitted:
                continue
            period = int(assignments[row["calibration_key"]][draw])
            if period < 0:
                continue
            recording_sites[period].add(row["site_id"])
            if row["family"] in families:
                feature_sites[period][families.index(row["family"])].add(row["site_id"])
        draw_support = []
        for period in range(len(periods)):
            distinct = {
                sites[site]["position_group"]: site
                for site in sorted(recording_sites[period])
            }
            indices = [site_index[site] for site in distinct.values()]
            recording[draw, :, period] = np.log1p(recorded[:, indices].sum(1))
            for family, included_sites in enumerate(feature_sites[period]):
                indices = np.array([site_index[site] for site in sorted(included_sites)], dtype=int)
                draw_support.append(len(indices))
                if len(indices) == 0:
                    continue
                values = site_proximity[:, indices]
                best = values.argmax(1)
                proximity[draw, :, period, family] = values[np.arange(len(nominal)), best]
                nearest[draw, :, period, family] = indices[best]
                if origin_weights is not None:
                    second = (
                        np.partition(values, -2, axis=1)[:, -2]
                        if len(indices) >= 2 else np.zeros(len(nominal))
                    )
                    gap = values[np.arange(len(nominal)), best] - second
                    np.add.at(contribution, indices[best], gap * contribution_weights)
                    if pairs and len(indices) >= 2:
                        remaining = values.copy()
                        remaining[np.arange(len(nominal)), best] = -1
                        second_best = remaining.argmax(1)
                        first_site, second_site = indices[best], indices[second_best]
                        keys = (
                            np.minimum(first_site, second_site) * len(site_ids)
                            + np.maximum(first_site, second_site)
                        )
                        found = np.searchsorted(sorted_pair_keys, keys)
                        valid = found < len(sorted_pair_keys)
                        valid[valid] &= sorted_pair_keys[found[valid]] == keys[valid]
                        third = (
                            np.partition(values, -3, axis=1)[:, -3]
                            if len(indices) >= 3 else np.zeros(len(nominal))
                        )
                        # Summed single losses omit the second-to-third gap
                        # precisely when a pair contains both nearest sites.
                        np.add.at(
                            pair_extra, pair_order[found[valid]],
                            (second[valid] - third[valid]) * contribution_weights[valid],
                        )
        support.append(draw_support)

    center = np.array(basis["center"])
    scale = np.array(basis["scale"])
    loadings = np.array(basis["loadings"])
    eigenvalues = np.array(basis["eigenvalues"][:loadings.shape[1]])
    if center.shape != (len(families),) or scale.shape != center.shape or loadings.shape[0] != len(families):
        raise ValueError("Basis dimensions do not match the feature-family order")
    if np.any(scale <= 0) or np.any(eigenvalues <= 0):
        raise ValueError("Retained basis scales and eigenvalues must be positive")
    mean = proximity.mean(0)
    relative = mean - mean.mean(2, keepdims=True)
    scores = ((relative - center) / scale) @ loadings / np.sqrt(eigenvalues)
    score_draws = (
        (proximity - proximity.mean(3, keepdims=True) - center) / scale
    ) @ loadings / np.sqrt(eigenvalues)
    if np.max(abs(scores - score_draws.mean(0))) >= 1e-12:
        raise ArithmeticError("Mean projection and projected draw mean differ")
    arrays = dict(
        proximity_draws=proximity,
        recording_draws=recording,
        scores=scores,
        score_draws=score_draws,
        nearest_site=nearest,
    )
    metadata = dict(
        site_ids=site_ids,
        draw_support_quantiles=np.quantile(support, [0, .5, 1], axis=0).tolist(),
        new_position_streams=new_streams,
    )
    if origin_weights is not None:
        metadata["nearest_gap_contribution"] = dict(zip(site_ids, contribution.tolist()))
    if pairs:
        joint = contribution[pair_indices].sum(1) + pair_extra
        metadata["joint_gap_contribution"] = [
            dict(sites=list(pair), contribution=float(value))
            for pair, value in zip(pairs, joint)
        ]
    return arrays, metadata

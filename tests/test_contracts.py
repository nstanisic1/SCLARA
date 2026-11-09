"""Small scientific contract checks for reusable dates, measurement and likelihood."""
import itertools
import math
import unittest
import numpy as np
from scipy.special import logsumexp
from analysis.dates import overlaps_period
from analysis.landscapes import measure_landscapes
from analysis.measure import material_dating_properties
from analysis.model import LabelMixture


class Calendar:
    def distribution(self, age, error):
        return np.array([5.0]), np.array([1.0])


def fixture(proximities=(0.9, 0.9, 0.1)):
    sites, geometry, rows = {}, [], []
    for i, value in enumerate(proximities):
        sid = str(i)
        longitude = math.degrees(-50 * math.log(value) / 6371.0088)
        sites[sid] = dict(latitude=0, longitude=longitude, source_uncertainty_m=0,
                          position_group=sid)
        geometry.append(dict(site_id=sid, position_id=sid, latitude=0, longitude=longitude))
        rows.append(dict(site_id=sid, calibration_key=sid, age=100, error=10, family="feature"))
    return dict(sites=sites, geometry=geometry, rows=rows)


def measured(pool, positions=None, families=None, **kwargs):
    families = ["feature"] if families is None else families
    size = len(families)
    basis = dict(center=[0]*size, scale=[1]*size,
                 loadings=np.eye(size).tolist(), eigenvalues=[1]*size)
    return measure_landscapes(
        pool, [("period", 0, 10)], families, Calendar(), 73, {},
        [dict(lat=0, lon=0)] if positions is None else positions, basis, n=3, **kwargs
    )


class DateMembership(unittest.TestCase):
    def test_points_boundaries_and_intervals(self):
        for bounds, expected in [
            ([100, 100], True), ([1, 1], True), ([400, 400], False),
            ([0, 1], False), ([0, 2], True), ([399, 401], True),
            ([400, 500], False), (None, False), ([], False), ([None, 100], False)
        ]:
            with self.subTest(bounds=bounds):
                self.assertEqual(overlaps_period(bounds, 1, 400), expected)

    def test_invalid_assertions_are_not_silently_plotted(self):
        for bounds in ([10, 2], [float("nan"), 20], ["100", "200"], [1]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                overlaps_period(bounds, 1, 400)


class MeasurementContract(unittest.TestCase):
    def test_empty_observed_universe(self):
        arrays, metadata = measured(fixture(()), origin_weights=[1])
        self.assertTrue(np.all(arrays["proximity_draws"] == 0))
        self.assertTrue(np.all(arrays["nearest_site"] == -1))
        self.assertEqual(metadata["draw_support_quantiles"], [[0], [0], [0]])

    def test_one_site_and_an_empty_family(self):
        arrays, metadata = measured(fixture((0.8,)), families=["feature", "absent"],
                                    origin_weights=[1])
        np.testing.assert_allclose(arrays["proximity_draws"][..., 0], 0.8, atol=1e-13)
        np.testing.assert_array_equal(arrays["proximity_draws"][..., 1], 0)
        self.assertAlmostEqual(metadata["nearest_gap_contribution"]["0"], 0.4)
        removed, _ = measured(fixture((0.8,)), omitted_sites=["0"])
        np.testing.assert_array_equal(removed["proximity_draws"], 0)

    def test_redundant_pair_has_joint_influence(self):
        _, metadata = measured(fixture(), origin_weights=[1], candidate_pairs=[("0", "1")])
        self.assertAlmostEqual(metadata["nearest_gap_contribution"]["0"], 0)
        self.assertAlmostEqual(metadata["nearest_gap_contribution"]["1"], 0)
        self.assertAlmostEqual(metadata["joint_gap_contribution"][0]["contribution"], 0.8)

    def test_pair_contributions_equal_explicit_remeasurement(self):
        pool = fixture((0.95, 0.95, 0.7, 0.4))
        # One assay asserted in two families must share its date allocation.
        pool["rows"].append(dict(pool["rows"][2], family="reference"))
        origins = [dict(lat=0, lon=0), dict(lat=0.3, lon=0.7), dict(lat=1, lon=1)]
        weights = np.array([0.2, 0.5, 0.3])
        pairs = list(itertools.combinations(pool["sites"], 2))
        arrays, metadata = measured(pool, origins, ["feature", "reference"],
                                    origin_weights=weights, candidate_pairs=pairs)
        for row in metadata["joint_gap_contribution"]:
            removed, _ = measured(pool, origins, ["feature", "reference"],
                                   omitted_sites=row["sites"])
            loss = arrays["proximity_draws"] - removed["proximity_draws"]
            expected = float(np.average(loss.mean(axis=(0, 2, 3)), weights=weights))
            with self.subTest(pair=row["sites"]):
                self.assertAlmostEqual(row["contribution"], expected, places=13)

    def test_measurement_is_unchanged_by_influence_calculation(self):
        plain, _ = measured(fixture())
        nominated, _ = measured(fixture(), origin_weights=[1], candidate_pairs=[("0", "1")])
        for key in plain:
            np.testing.assert_array_equal(plain[key], nominated[key])


class SharedAssays(unittest.TestCase):
    def test_shared_dates_and_duplicate_assertion_invariance(self):
        class BoundaryDates:
            def distribution(self, age, error):
                return np.array([-1,1]), np.array([.5,.5])
        pool=fixture((.8,))
        pool["rows"].append(dict(pool["rows"][0],family="reference"))
        def measure(p):
            return measure_landscapes(p,[("early",-2,0),("late",0,2)],
                ["feature","reference"],BoundaryDates(),73,{},[dict(lat=0,lon=0)],
                dict(center=[0,0],scale=[1,1],loadings=[[1],[1]],eigenvalues=[1]),n=64)[0]
        original=measure(pool)
        np.testing.assert_array_equal(original["proximity_draws"][...,0],original["proximity_draws"][...,1])
        self.assertTrue(np.all((original["proximity_draws"]>0).sum(axis=2)==1))
        pool["rows"].append(dict(pool["rows"][0]))
        repeated=measure(pool)
        for key in original:np.testing.assert_array_equal(original[key],repeated[key])


class ReconciledAncientBranch(unittest.TestCase):
    def test_conflicts_are_not_overruled_by_priority(self):
        from analysis.ancient import reconciled_plot_branch
        person=dict(path_conflict=False,broad_parent_conflict=False,
                    qualified_targets=["I1","I1_Z58"],reported_broad_parents=["I1"])
        self.assertEqual(reconciled_plot_branch(person),"I1_Z58")
        self.assertEqual(reconciled_plot_branch(dict(person,path_conflict=True)),"other_or_unresolved")
        self.assertEqual(reconciled_plot_branch(dict(person,broad_parent_conflict=True)),"other_or_unresolved")
        self.assertEqual(reconciled_plot_branch(dict(person,qualified_targets=[])),"I1")
        self.assertEqual(reconciled_plot_branch(None),"other_or_unresolved")


class MaterialRisk(unittest.TestCase):
    def test_material_does_not_imply_activity_phase(self):
        self.assertEqual(material_dating_properties("sädeskorn", "skalkorn"),
                         (1, "short_lived_terrestrial"))
        self.assertEqual(material_dating_properties("hasselnötsskal", "hassel"),
                         (1, "short_lived_terrestrial"))
        self.assertEqual(material_dating_properties("träkol", "björk"),
                         (0, "possible_inbuilt_age"))
        self.assertEqual(material_dating_properties("ben", ""),
                         (0, "reservoir_or_material_review_needed"))


class LocalityMixture(unittest.TestCase):
    def test_single_multiple_and_mixed_units(self):
        rng = np.random.default_rng(2001)
        for counts in ([1, 1, 1], [2, 3, 2], [1, 2, 1]):
            x = np.column_stack([np.ones(sum(counts)), rng.normal(size=sum(counts))])
            y = np.array([0, 1, 2])
            model = LabelMixture(x, y, counts, [2, 1], 3)
            coefficients = np.array([0.2, -0.3, 0.6, 0.4])
            logits = np.column_stack([np.zeros(len(x)), x @ coefficients.reshape(2, 2).T])
            probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, None])
            expected, start = 0.0, 0
            for label, count in zip(y, counts):
                expected -= math.log(probabilities[start:start+count, label].mean())
                start += count
            expected += 0.5 * coefficients @ model.q @ coefficients
            self.assertAlmostEqual(model.evaluate(coefficients)[0], expected, places=13)


class TransportDiagnostics(unittest.TestCase):
    def test_equal_unit_weight_and_unrepresented_positions(self):
        from analysis.measure import common_transport_diagnostics
        first = np.array([1, -1, 0]) / np.sqrt(2)
        second = np.array([1, 1, -2]) / np.sqrt(6)
        coordinates = np.array([[1, 0], [-1, 0], [0, 2]])
        values = 0.5 + 0.1 * coordinates @ np.stack([first, second])
        values = np.vstack([values, [0.99, 0.01, 0.99]])
        arrays = dict(proximity_draws=values[None, :, None, :])
        cohort = dict(local=np.array([0, 1, 2]), counts=np.array([1, 2]))
        common = dict(periods=[["test", 0, 10]], families=["a", "b", "c"],
                      basis=dict(center=[0.1, 0.2, 0.3], scale=[1, 1, 1],
                                 loadings=first[:, None].tolist()))
        result = common_transport_diagnostics(arrays, cohort, common)
        # Coordinate variances under weights (1/2, 1/4, 1/4) are 11/16 and 12/16.
        self.assertAlmostEqual(result["periods"][0]["retained_variance_fraction"], 11 / 23)
        self.assertEqual(result["represented_positions"], 3)
        common["basis"]["loadings"] = np.stack([first, second], axis=1).tolist()
        result = common_transport_diagnostics(arrays, cohort, common)
        self.assertAlmostEqual(result["periods"][0]["retained_variance_fraction"], 1)

    def test_zero_variance_is_undefined(self):
        from analysis.measure import common_transport_diagnostics
        result = common_transport_diagnostics(
            dict(proximity_draws=np.zeros((2, 2, 1, 2))),
            dict(local=np.array([0, 1]), counts=np.array([1, 1])),
            dict(periods=[["empty", 0, 10]], families=["a", "b"],
                 basis=dict(center=[0, 0], scale=[1, 1], loadings=[[1], [0]])),
        )
        self.assertIsNone(result["periods"][0]["retained_variance_fraction"])
        self.assertEqual(result["periods"][0]["proximity_equivalent_coverage_fraction"], [0, 0])


if __name__ == "__main__":
    unittest.main()

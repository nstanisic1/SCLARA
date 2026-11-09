"""Stable positional draws and explicit source-identity contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis.ancient import canonical_identity_ids, validated_alias_edges
from analysis.coordinates import ancient_display_name
from analysis.landscapes import position_draws
from analysis.measure import load_position_streams, registered_position_draws


class PositionIdentity(unittest.TestCase):
    def setUp(self):
        self.geometry = [dict(position_id=i, latitude=60., longitude=15.) for i in ['a', 'b']]
        self.radii = {(60., 15.): 100}
        self.registry = dict(schema='sclara-preserved-later-position-streams-1', draws=64,
                             positions={i: dict(latitude=60., longitude=15., radius_m=100,
                                               seed='original stream') for i in ['a', 'b']})

    def test_roundoff_order_and_shared_identity(self):
        expected = position_draws(60., 15., 100, 'original stream', 64)
        changed = copy.deepcopy(self.geometry[::-1])
        for row in changed: row['latitude'] = np.nextafter(60., 61.)
        radii = {(row['latitude'], row['longitude']): 100 for row in changed}
        draws = registered_position_draws(changed, radii, self.registry)
        self.assertTrue(np.array_equal(draws[:, 0], expected))
        self.assertTrue(np.array_equal(draws[:, 1], expected))

    def test_changed_geometry_radius_and_membership_rejected(self):
        changed = copy.deepcopy(self.geometry); changed[0]['latitude'] += .001
        for geometry, radii in [(changed, self.radii), (self.geometry, {(60., 15.): 101}),
                                (self.geometry[:1], self.radii),
                                (self.geometry + [dict(position_id='c', latitude=60., longitude=15.)], self.radii),
                                (self.geometry + [self.geometry[0]], self.radii)]:
            with self.assertRaises(ValueError): registered_position_draws(geometry, radii, self.registry)

    def test_registry_consistency(self):
        for field, value in [('seed', 'changed'), ('latitude', float('nan')), ('radius_m', -1)]:
            registry = copy.deepcopy(self.registry); registry['positions']['b'][field] = value
            with self.assertRaises(ValueError): registered_position_draws(self.geometry, self.radii, registry)
        registry = copy.deepcopy(self.registry); registry['draws'] = 32
        with self.assertRaises(ValueError): registered_position_draws(self.geometry, self.radii, registry)

    def test_source_and_registry_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'source.zip'; archive.write_bytes(b'archaeological source')
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            registry = dict(self.registry, archaeological_source_archive_sha256=digest)
            path = root/'streams.json'; path.write_text(json.dumps(registry))
            specification = dict(source_archives={'archaeology.zip': digest}, later=dict(
                position_stream_registry=dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            self.assertEqual(load_position_streams(root, specification, archive), registry)
            archive.write_bytes(b'changed source')
            with self.assertRaises(ValueError): load_position_streams(root, specification, archive)
            archive.write_bytes(b'archaeological source'); path.write_text('{}')
            with self.assertRaises(ValueError): load_position_streams(root, specification, archive)


class AncientIdentity(unittest.TestCase):
    def setUp(self):
        self.alias = dict(left='Original', right='Mapped', canonical_id='Mapped',
                          left_source=['study1', 'samples', 2], right_source=['study2', 'samples', 8],
                          basis='Explicit published crosswalk', primary_citation='Author (2020), Table 1.',
                          primary_url='https://example.org/publication')
        self.raw = {('study1', 'samples', 2): {'ID': 'Original'},
                    ('study2', 'samples', 8): {'popId': 'Mapped'}}

    def test_exact_source_identifiers_and_stable_mapped_id(self):
        self.assertEqual(len(validated_alias_edges(self.raw, [self.alias])), 1)
        groups = {'group': ['Original', 'Mapped'], 'other': ['original']}
        ids = canonical_identity_ids(groups, [self.alias])
        self.assertEqual(ids, {'Original': 'Mapped', 'Mapped': 'Mapped', 'original': 'original'})
        raw = copy.deepcopy(self.raw); raw[('study1', 'samples', 2)]['ID'] = 'original'
        with self.assertRaises(ValueError): validated_alias_edges(raw, [self.alias])
        with self.assertRaises(ValueError): validated_alias_edges({}, [self.alias])

    def test_conflicting_or_external_canonical_id_rejected(self):
        groups = {'group': ['Original', 'Mapped']}
        for aliases in [[dict(self.alias, canonical_id='Absent')],
                        [self.alias, dict(self.alias, canonical_id='Original')]]:
            with self.assertRaises(ValueError): canonical_identity_ids(groups, aliases)

    def test_display_correction_is_source_scoped(self):
        row = dict(parent_id='Landbog√•rden', source_locator='S01_tables.xlsx / S5.1. Clustered Ancient Samples / 4252')
        self.assertEqual(ancient_display_name(row), 'Landbogården')
        self.assertEqual(row['parent_id'], 'Landbog√•rden')
        self.assertEqual(ancient_display_name(dict(row, source_locator='another source')), row['parent_id'])


if __name__ == '__main__':
    unittest.main()

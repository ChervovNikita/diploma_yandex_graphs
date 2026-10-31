import unittest

from compile_native_matrix import BACKBONES, native_cell_admission


class DeferralContract(unittest.TestCase):
    def fixtures(self):
        profiles = {'pubmed/' + b: {'reservation_mib': 8} for b in BACKBONES}
        del profiles['pubmed/GT']
        profiles['pubmed/GAT-sep'] = {'reservation_mib': 81}
        return profiles, {'allocation/gpu0': 80 * 1024**2}

    def test_full_request_cannot_silently_omit_failed_cells(self):
        profiles, capacities = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'explicit preserved deferral'):
            native_cell_admission(['pubmed'], profiles, capacities, {})

    def test_disabled_subset_preserves_each_unmaterialized_reason(self):
        profiles, capacities = self.fixtures()
        admitted, blocked = native_cell_admission(['pubmed'], profiles, capacities,
            {'pubmed/GT': 'OOM in qualification', 'pubmed/GAT-sep': 'No worker has sufficient headroom'})
        self.assertEqual(len(admitted), 6)
        self.assertEqual(set(blocked), {'pubmed/GT', 'pubmed/GAT-sep'})
        self.assertTrue(all(not row['study_materialized'] for row in blocked.values()))

    def test_unrelated_or_admitted_cell_cannot_be_pruned(self):
        profiles, capacities = self.fixtures()
        for name in ['cora/GCN', 'pubmed/SAGE']:
            with self.assertRaises(ValueError):
                native_cell_admission(['pubmed'], profiles, capacities, {name: 'prune'})

if __name__ == '__main__':
    unittest.main()

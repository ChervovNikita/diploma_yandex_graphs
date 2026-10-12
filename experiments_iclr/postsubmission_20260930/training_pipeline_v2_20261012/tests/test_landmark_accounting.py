"""Paid landmark sources, private views and separate own-selected bodies."""
import unittest

from compile_master_plan import bank_roster, fit_semantics, graph_check, typed_counts
from test_master_plan import run


def landmark(identifier, arm, stage, body=0, parents=()):
    config = {'stage': 'comparative', 'landmark_stage': stage, 'landmark_arm': arm,
              'architecture': 'original_gnnm' if arm == 'original_boundary' else 'landmark_memory',
              'is_original_gnnm': arm == 'original_boundary', 'source_optimizer_count': 1,
              'physical_slow_bodies': 1,
              'members': 4 if arm in {'shared_factor4', 'original_boundary', 'B4', 'T16', 'LM4'} else 1}
    if stage == 'continuation':
        config['lifecycle'] = {'schema': 'paid-parent-continuation-v1',
            'parent_fit_ids': list(parents), 'new_optimizer': True,
            'parameter_initialization': 'selected_parent_no_optimizer_reuse'}
    return run(identifier, 'model_fit', arm, config, parents, body, maximum=5000)


class LandmarkAccountingTests(unittest.TestCase):
    def test_memory_pays_factor_source_and_both_complete_budgets(self):
        source = landmark('source', 'factor_m1', 'source')
        memory = landmark('memory', 'LM4', 'continuation', parents=['source'])
        study = {'study_id': 'fixture', 'runs': [source, memory]}
        graph_check([study], {})
        counts = typed_counts([study])
        self.assertEqual(counts['model_optimizer_step_call_total'], 10000)
        self.assertEqual(counts['totals']['paid_optimizer_starts'], 2)
        self.assertEqual(counts['totals']['physical_slow_body_instances'], 2)
        self.assertEqual(counts['totals']['paid_continuation_segments'], 1)
        source['config']['landmark_arm'] = 'ordinary_m1'
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            graph_check([study], {})

    def test_I4_memory_control_needs_four_separate_paid_ancestors(self):
        sources = [landmark('s'+str(b), 'factor_m1', 'source', b) for b in range(4)]
        fits = [landmark('c'+str(b), 'S64', 'continuation', b, ['s'+str(b)]) for b in range(4)]
        ids = [r['run_id'] for r in fits]
        bank = run('I4', 'ensemble_assembly', 'S64_I4', {'stage': 'bank_export', 'fit_parents': ids}, ids)
        study = {'study_id': 'fixture', 'runs': sources+fits+[bank]}
        graph_check([study], {})
        self.assertTrue(bank_roster(study)[0]['genuine_I4'])
        fits[1]['config']['lifecycle']['parent_fit_ids'] = ['s0']
        fits[1]['parents'] = ['s0']
        with self.assertRaisesRegex(ValueError, 'paid source acquisitions'):
            bank_roster(study)

    def test_private_views_are_not_independent_bodies(self):
        for arm, stage in [('original_boundary', 'source'), ('B4', 'continuation'), ('T16', 'continuation')]:
            with self.subTest(arm=arm):
                row = landmark('fit', arm, stage)
                semantics = fit_semantics(row)
                self.assertEqual(semantics['physical_slow_bodies'], 1)
                self.assertEqual(semantics['optimizer_objects'], 1)
                row['config']['members'] = 1
                with self.assertRaisesRegex(ValueError, 'route counts'):
                    fit_semantics(row)


if __name__ == '__main__':
    unittest.main()

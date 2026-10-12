"""Paid ancestry and byte authority checks; no scientific fits or tensor reads."""
import copy
from pathlib import Path
import tempfile
import unittest

from engine import checkpoint_receipt
from run_study import construct_adapter, paid_parent_checkpoints
from compile_master_plan import bank_roster, fit_semantics, graph_check, typed_counts
from test_master_plan import run


def fit(identifier, arm, pt_stage, body=0, parents=()):
    config = {'stage': 'comparative', 'pt_stage': pt_stage, 'pt_arm': arm,
              'architecture': 'retained_pt_routes', 'members': 4 if arm == 'separable4' else 1,
              'source_optimizer_count': 1, 'physical_slow_bodies': 4 if arm == 'separable4' else 1}
    if pt_stage == 'continuation':
        config['lifecycle'] = {'schema': 'paid-parent-continuation-v1',
            'parent_fit_ids': list(parents), 'new_optimizer': True,
            'parameter_initialization': 'selected_parent_no_optimizer_reuse'}
    return run(identifier, 'model_fit', arm, config, parents, body, maximum=5000)


class FakeRegistry:
    def __init__(self, rows): self.rows = rows
    def run(self, study, identifier): return self.rows[identifier]


class PaidContinuationTests(unittest.TestCase):
    def test_legacy_factory_keeps_old_signature_and_context_is_explicit(self):
        row = {'config': {}}
        self.assertEqual(construct_adapter(lambda run: run, execution_context={}, run=row), row)
        def aware(run, execution_context): return execution_context
        self.assertEqual(construct_adapter(aware, execution_context={'registry': 'explicit'}, run=row),
                         {'registry': 'explicit'})
        row = fit('continued', 'continued_p1', 'continuation', parents=['source'])
        with self.assertRaisesRegex(ValueError, 'context-aware'):
            construct_adapter(lambda run: run, execution_context={}, run=row)

    def fixture(self, root):
        source = fit('source', 'ordinary_p1', 'source')
        continuation = fit('continued', 'continued_p1', 'continuation', parents=['source'])
        path = root / 'study' / 'source' / 'selected.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'opaque engineering checkpoint, not a torch model')
        result = {'status': 'complete', 'completed_steps': 5000,
                  'selected': {'step': 3999},
                  'selected_checkpoint': checkpoint_receipt(path, fit_id='source', step=3999)}
        registry = FakeRegistry({'source': {'state': 'complete', 'contract': source, 'result': result}})
        return registry, continuation, path

    def test_complete_paid_checkpoint_is_verified_without_tensor_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, continuation, path = self.fixture(root)
            authority = paid_parent_checkpoints(root, {'study_id': 'study'}, continuation, registry, root)
            self.assertEqual(authority['source']['checkpoint'], path)
            path.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'byte authority'):
                paid_parent_checkpoints(root, {'study_id': 'study'}, continuation, registry, root)

    def test_unpaid_or_partial_source_cannot_be_a_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, continuation, _ = self.fixture(root)
            registry.rows['source']['result']['completed_steps'] = 4999
            with self.assertRaisesRegex(ValueError, 'complete'):
                paid_parent_checkpoints(root, {'study_id': 'study'}, continuation, registry, root)
            continuation['config'].pop('lifecycle')
            with self.assertRaisesRegex(ValueError, 'direct paid'):
                paid_parent_checkpoints(root, {'study_id': 'study'}, continuation, registry, root)

    def test_typed_master_counts_both_training_stages_once(self):
        source = fit('source', 'ordinary_p1', 'source')
        private = fit('private', 'separable4', 'continuation', parents=['source'])
        study = {'study_id': 'fixture', 'runs': [source, private]}
        graph_check([study], {})
        quantities = typed_counts([study])
        self.assertEqual(quantities['model_optimizer_step_call_total'], 10000)
        self.assertEqual(quantities['totals']['paid_optimizer_starts'], 2)
        self.assertEqual(quantities['totals']['physical_slow_body_instances'], 5)
        self.assertEqual(quantities['totals']['initial_training_segments'], 1)
        self.assertEqual(quantities['totals']['paid_continuation_segments'], 1)
        private['config']['physical_slow_bodies'] = 1
        with self.assertRaisesRegex(ValueError, 'PT owns'):
            fit_semantics(private)

    def test_mismatched_source_or_checkpoint0_is_rejected_by_master(self):
        source = fit('source', 'ordinary_p1', 'source')
        private = fit('private', 'private4', 'continuation', parents=['source'])
        source['selection']['checkpoint0'] = True
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            graph_check([{'study_id': 'fixture', 'runs': [source, private]}], {})
        source['selection']['checkpoint0'] = False
        source['body'] = 1
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            graph_check([{'study_id': 'fixture', 'runs': [source, private]}], {})

    def continued_bank(self):
        sources = [fit('source' + str(body), 'ordinary_p1', 'source', body) for body in range(4)]
        continuations = [fit('continued' + str(body), 'continued_p1', 'continuation', body,
                             parents=['source' + str(body)]) for body in range(4)]
        single = run('M1', 'ensemble_assembly', 'continued_p1',
                     {'stage': 'bank_export', 'fit_parents': ['continued0']}, ['continued0'])
        bank = run('I4', 'ensemble_assembly', 'continued_p1',
                   {'stage': 'bank_export', 'fit_parents': [r['run_id'] for r in continuations]},
                   [r['run_id'] for r in continuations])
        return {'study_id': 'fixture', 'runs': sources + continuations + [single, bank]}

    def test_continued_I4_requires_four_equivalent_paid_source_learners(self):
        study = self.continued_bank()
        banks = bank_roster(study)
        self.assertTrue(banks[-1]['genuine_I4'])
        self.assertTrue(banks[-1]['member0_reused'])
        study['runs'][1]['config']['learning_rate'] = .1
        with self.assertRaisesRegex(ValueError, 'different learning'):
            bank_roster(study)

    def test_sixteen_views_require_four_distinct_complete_GRU_learners(self):
        study = self.continued_bank()
        for row in study['runs']:
            if row['kind'] == 'model_fit':
                row['config']['gru_stage'] = row['config'].pop('pt_stage')
                arm = row['config'].pop('pt_arm')
                row['config']['gru_arm'] = 'G1' if arm == 'continued_p1' else arm
                row['config']['architecture'] = 'private_gru_memory'
                row['config']['members'] = 4 if row['config']['gru_stage'] == 'continuation' else 1
        bank = study['runs'][-1]
        with self.assertRaisesRegex(ValueError, 'one-route'):
            bank_roster(study)
        bank['config']['independent_bank_contract'] = {'schema': 'four-own-four-view-v1',
            'bodies': 4, 'views_per_body': 4, 'served_views': 16}
        graph_check([study], {})
        records = bank_roster(study)
        self.assertEqual(records[-1]['routes'], 16)
        self.assertEqual(records[-1]['views_per_body'], 4)
        self.assertTrue(records[-1]['genuine_I16'])
        self.assertTrue(records[-1]['member0_reused'])
        counts = typed_counts([study])
        self.assertEqual(counts['totals']['genuine_four_body_four_view_I16_banks'], 1)
        self.assertEqual(counts['totals']['initial_training_segments'], 4)
        self.assertEqual(counts['totals']['paid_continuation_segments'], 4)
        bank['config']['independent_bank_contract']['served_views'] = 4
        with self.assertRaisesRegex(ValueError, 'sixteen-view'):
            bank_roster(study)


if __name__ == '__main__': unittest.main()

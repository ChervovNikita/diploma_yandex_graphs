"""Metadata checks plus root-gated tiny CPU fixtures, never benchmark data.

Local preparation runs CoordinateMetadataTests only. The root sets
GNNM_COORDINATE_CPU_FIXTURES=1 after source admission to execute tensor fixtures.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from compile_coordinate_matrix import (ARMS, I4_PARENTS, TEMPLATES, count_function,
    make_study, matching_width, model_bytes, numeric_domain, validate_recipes)

PACKAGE = Path(__file__).resolve().parents[1]
COORDINATE = PACKAGE / 'fixtures/coordinate_models.py'
if not COORDINATE.exists():
    COORDINATE = PACKAGE.parent / 'coordinate_ensemble_source_v1/models.py'
NATIVE = PACKAGE / 'fixtures/native_models.py'
if not NATIVE.exists():
    NATIVE = PACKAGE.parent / 'common_wrapper_native_source_20261010_v1/models.py'
SPECIFICATION = PACKAGE / 'fixtures/coordinate_specification.json'
if not SPECIFICATION.exists():
    SPECIFICATION = PACKAGE.parent / 'coordinate_stage1_assessment_source_v2/SPECIFICATION_v2.json'


class CoordinateMetadataTests(unittest.TestCase):
    def setUp(self):
        self.formula = count_function(COORDINATE)
        self.recipes = validate_recipes(json.loads(SPECIFICATION.read_text()))

    def test_exact_historical_aliases_and_equipped_single_controls(self):
        self.assertEqual(set(ARMS), {'original', 'factor', 'coordinate', 'permutation', 'bias_only',
            'single', 'untied', 'heads', 'gt_sep_single'})
        self.assertEqual(len(TEMPLATES), 13)
        self.assertEqual(len(I4_PARENTS), 6)
        self.assertNotIn('untied', I4_PARENTS)
        self.assertNotIn('heads', I4_PARENTS)

    def test_hidden_index_bytes_and_actual_source_defaults_survive_dedup(self):
        graph = {'features': 745, 'classes': 8}
        h, layers, members = 256, 3, 4
        self.assertEqual(model_bytes(self.formula, 'coordinate', 745, h, 8, layers)
            - model_bytes(self.formula, 'factor', 745, h, 8, layers), 8 * members * layers * 9 * h)
        for name, arm, members in TEMPLATES:
            domain = numeric_domain('amazon-photo', arm, members, graph, self.formula, self.recipes)
            anchor = self.recipes['AmazonPhoto_' + arm + '_stage1']
            expected = {'learning_rate': anchor['optimizer']['lr'], 'weight_decay': anchor['optimizer']['weight_decay'],
                'width': anchor['width'], 'depth': anchor['num_layers'], 'dropout': anchor['dropout']}
            self.assertIn(expected, domain)
            self.assertEqual(len(domain), len({json.dumps(p, sort_keys=True) for p in domain}))
            self.assertEqual(len(domain), 100)  # int0/float0.0 must not create a duplicate anchor.
        for arm, expected in [('single', 267), ('untied', 122), ('heads', 239)]:
            self.assertEqual(matching_width(self.formula, arm, 745, 256, 8, 3), expected)

    def test_disabled_roster_counts_and_exact_member0_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'split_id': split, 'node_inputs': {'graph': 'graph.pt', 'train': 't' + str(split), 'select': 's' + str(split)},
                'endpoint_inputs': {key: key + str(split) for key in ('cal_ids', 'evaluate_ids', 'cal_targets', 'evaluate_targets')}} for split in (0, 1)]
            custody = {'dataset_id': 'amazon-photo', 'splits': rows, 'model_graph': {'features': 745, 'classes': 8},
                'request': {'graph_policy': 'fixture', 'evaluation_exposure': 'engineering'}, 'request_hash': 'fixture', 'provider_files': []}
            source = {'git_commit': 'fixture', 'files': [{'path': 'source.py', 'bytes': 1, 'sha256': 'a' * 64}]}
            profile = {'reservation_mib': 1024, 'wall_seconds': 100, 'estimated_seconds_per_update': .1, 'evidence': ['fixture arithmetic']}
            study, counts, banks = make_study(root, custody, root, source, profile, self.recipes, self.formula,
                'frozen/runtime', 'frozen/coordinate.py', 'frozen/native.py')
            self.assertEqual(counts['atomic_fit_run_groups_by_stage']['comparative'], 93 * 2)
            self.assertEqual(counts['paid_optimizer_starts_by_stage']['comparative'], 102 * 2)
            self.assertEqual(counts['physical_bodies_by_stage']['comparative'], 102 * 2)
            self.assertEqual(counts['comparative_banks'], 57 * 2)
            self.assertEqual(counts['genuine_I4_banks_and_reused_member0'], 18 * 2)
            self.assertEqual(counts['served_route_views'], 174 * 2)
            self.assertEqual(counts['search_selection_jobs'], 13)
            self.assertEqual(counts['atomic_fit_run_groups_by_stage']['search'], 2600)
            self.assertEqual(counts['paid_optimizer_starts_by_stage']['search'], 3200)
            self.assertEqual(counts['actual_optimizer_updates'], 0)
            runs = {r['run_id']: r for r in study['runs']}
            for bank in banks:
                if bank['body0_reused']:
                    self.assertEqual(runs[bank['fit_parents'][0]]['body'], 0)
                    self.assertEqual([runs[i]['body'] for i in bank['fit_parents']], [0, 1, 2, 3])
                    self.assertTrue(all(runs[i]['config']['members'] == 1 for i in bank['fit_parents']))
                    self.assertTrue(all(runs[i]['config']['source_optimizer_count'] == 1 for i in bank['fit_parents']))
            fits = [r for r in study['runs'] if r['kind'] == 'model_fit']
            self.assertTrue(all(r['budget'] == {'unit': 'epochs', 'policy': 'full_budget', 'maximum': 2000} for r in fits))
            self.assertTrue(all(r['config']['fusion'] == 'mean_probabilities' for r in fits))
            self.assertTrue(all(r['config']['source_optimizer_count'] == 4 for r in fits if r['config']['source_arm'] == 'untied'))


CPU_ADMITTED = os.environ.get('GNNM_COORDINATE_CPU_FIXTURES') == '1' and importlib.util.find_spec('torch') is not None


@unittest.skipUnless(CPU_ADMITTED, 'Root CPU source admission required')
class CoordinateCPUFixtures(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copyfile(COORDINATE, self.root / 'coordinate.py')
        shutil.copyfile(NATIVE, self.root / 'native.py')
        torch.save({'x': torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32,
            'edge_index': torch.tensor([[0,0,1,2,2,3,4,5,6], [1,1,0,1,2,4,3,6,5]])}, self.root / 'graph.pt')
        torch.save({'ids': torch.arange(6), 'y': torch.tensor([0,0,1,1,2,2])}, self.root / 'train.pt')
        torch.save({'ids': torch.tensor([6,7]), 'y': torch.tensor([0,1])}, self.root / 'select.pt')
        self.config = {'source_arm': 'coordinate', 'backbone': 'SAGE', 'target_law': 'multiclass_ce',
            'classes': 3, 'coordinate_source': 'coordinate.py', 'native_model_source': 'native.py',
            'node_inputs': {'graph': 'graph.pt', 'train': 'train.pt', 'select': 'select.pt'},
            'width': 8, 'depth': 1, 'ffn_multiplier': 1, 'heads': 2, 'normalization': 'LayerNorm',
            'dropout': 0., 'learning_rate': .001, 'weight_decay': 0., 'permutation_seed_offset': 1000,
            'members': 4, 'source_requested_members': 4, 'source_optimizer_count': 1,
            'source_minimum_cycles': 1000, 'source_patience_cycles': 200, 'fusion': 'mean_probabilities'}
        self.study = {'dataset': 'synthetic_engineering_fixture', 'data': {}, 'source': {}}
        self.run = {'run_id': 'fixture', 'seed': 101, 'body': 0, 'config': self.config,
            'selection': {'metric': 'pooled_nll', 'direction': 'min', 'ties': 'first'}}

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, arm='coordinate', members=None, body=0):
        from adapters.coordinate import make_adapter
        run = copy.deepcopy(self.run)
        run['body'] = body
        config = run['config']
        config['source_arm'] = arm
        config['backbone'] = 'GT-sep' if arm == 'gt_sep_single' else 'SAGE'
        config['members'] = members or (1 if arm in {'single', 'gt_sep_single'} else 4)
        config['source_requested_members'] = 4 if arm in {'single', 'gt_sep_single'} else config['members']
        config['source_optimizer_count'] = 4 if arm == 'untied' else 1
        return make_adapter(torch=self.t, project_root=self.root, study=self.study, run=run, device='cpu')

    def test_all_source_arms_shape_and_ownership(self):
        from engine import inspect_ownership
        for arm in ARMS:
            with self.subTest(arm=arm):
                adapter = self.make(arm)
                logits = adapter.logits_fn(adapter.model, False, 0)
                self.assertEqual(tuple(logits.shape), (1 if arm in {'single', 'gt_sep_single'} else 4, 8, 3))
                inspect_ownership(adapter)
                self.assertEqual(adapter.validate(0)['fusion'], 'mean_probabilities')

    def test_exact_source_maps_bias_flags_and_live_gradients(self):
        adapter = self.make('coordinate')
        base = adapter.model.base
        conv = base.residual_modules[0].module.conv
        self.assertIsNone(conv.lin_r.bias)
        self.assertEqual(tuple(conv.lin_l.bias.shape), (4, 8))
        self.assertIsNotNone(conv.lin_l.input_index)
        self.assertEqual(tuple(base.input_be_block.R.shape), (4, 4))
        self.assertFalse(hasattr(base.input_be_block.W, 'input_index'))
        result = adapter.train_step(1)
        self.assertEqual(result['operation_counts']['optimizer_updates'], 1)
        self.assertIsNotNone(conv.lin_l.weight.grad)
        self.assertIsNotNone(conv.lin_l.R.grad)
        self.assertIsNotNone(base.input_be_block.R.grad)
        for arm in ('factor', 'bias_only', 'permutation'):
            context = self.make(arm).model.base.residual_modules[0].module.conv.lin_l
            self.assertEqual(context.R is None, arm != 'factor')
            self.assertEqual(context.input_index is None, arm != 'permutation')

    def test_four_optimizer_states_and_controller_actual_calls(self):
        from engine import run_trajectory, restore_checkpoint
        adapter = self.make('untied')
        self.assertEqual(len(adapter.optimizer.optimizers), 4)
        self.assertEqual(len({id(o) for o in adapter.optimizer.optimizers}), 4)
        contract = {'fit_id': 'fixture', 'budget_steps': 2, 'budget_unit': 'epochs', 'validation_interval': 1,
            'validate_initial': False, 'prefixes': [], 'inherited_fit_ids': [], 'adapter_spec': {}}
        result = run_trajectory(self.t, adapter, contract, self.root / 'trajectory')
        self.assertEqual(result['operation_counts']['optimizer_updates'], 8)
        self.assertEqual(result['operation_counts']['source_cycles'], 2)
        selected = self.t.load(self.root / 'trajectory/selected.pt', weights_only=False)
        self.assertEqual(selected['optimizer_state']['optimizer_count'], 4)
        self.assertTrue(all(state['state'] for state in selected['optimizer_state']['optimizer_states']))
        replacement = self.make('untied')
        restore_checkpoint(self.t, self.root / 'trajectory/selected.pt', replacement, expected_fit_id='fixture')
        self.assertEqual(len(replacement.optimizer.state_dict()['optimizer_states']), 4)

    def test_genuine_independent_m1_controls_and_fixed_index_restore(self):
        from engine import inspect_independence
        adapters = [self.make('coordinate', members=1, body=b) for b in range(4)]
        inspect_independence(adapters)
        self.assertEqual(len({a.provenance['actual_model_seed'] for a in adapters}), 4)
        self.assertEqual(len({a.provenance['actual_permutation_seed'] for a in adapters}), 4)
        state = copy.deepcopy(adapters[0].model.state_dict())
        indices = adapters[0].model.base.residual_modules[0].module.conv.lin_l.input_index
        with self.t.no_grad():
            indices.zero_()
        adapters[0].model.load_state_dict(state)
        self.assertTrue(self.t.equal(indices, state['base.residual_modules.0.module.conv.lin_l.input_index']))


if __name__ == '__main__':
    unittest.main()

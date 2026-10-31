"""Prospective metadata/DAG qualification; no numerical libraries or role arrays."""
import copy
import json
from pathlib import Path
import unittest

import compile_native_matrix as native
import compile_coordinate_matrix as coordinate
from compile_master_plan import reference_matches
from spec import validate

PACKAGE = Path(__file__).resolve().parents[1]


def custody(dataset, masks):
    return {'dataset_id': dataset, 'model_graph': {'features': 745, 'classes': 8},
            'request': {'graph_policy': 'synthetic_metadata_only', 'evaluation_exposure': 'development'},
            'request_hash': 'synthetic', 'provider_files': [],
            'splits': [{'split_id': i,
                        'node_inputs': {'graph': 'graph.pt', 'train': 'train' + str(i) + '.pt', 'select': 'select' + str(i) + '.pt'},
                        'endpoint_inputs': {role + suffix: role + suffix + str(i) + '.pt'
                                            for role in ('cal', 'evaluate') for suffix in ('_ids', '_targets')}}
                       for i in masks]}


def source():
    return {'git_commit': 'synthetic-engineering', 'files': [{'path': 'synthetic-source', 'sha256': '0' * 64, 'bytes': 1}]}


PROFILE = {'reservation_mib': 32, 'wall_seconds': 1000000, 'estimated_seconds_per_update': .01}


class PracticalMetadataTests(unittest.TestCase):
    def setUp(self):
        self.native_policy = json.loads((PACKAGE / 'docs/PRACTICAL_NATIVE_POLICY_A1_20261012.json').read_text())
        self.coordinate_policy = json.loads((PACKAGE / 'docs/PRACTICAL_COORDINATE_POLICY_A1_20261012.json').read_text())
        rows = json.loads((PACKAGE / 'docs/MATRIX_AXES_DRAFT.json').read_text())['baseline_recipe_catalogue']['rows']
        self.native_recipes = {r['native_constructor_name'].split('(')[-1].rstrip(')'): r
                               for r in rows if r.get('readiness', {}).get('adapter_available')}
        specification = PACKAGE / 'fixtures/coordinate_specification.json'
        if not specification.exists():
            specification = PACKAGE.parent / 'coordinate_stage1_assessment_source_v2/SPECIFICATION_v2.json'
        self.coordinate_recipes = coordinate.validate_recipes(json.loads(specification.read_text()))

    def native_study(self, dataset, backbone, policy=None):
        return native.make_study(PACKAGE, custody(dataset, range(10) if dataset == 'amazon-photo' else [0]),
                                 PACKAGE, backbone, self.native_recipes[backbone], source(), PROFILE,
                                 'native_fixed', practical_policy=policy)

    def test_native_enumerates_all_mandatory_cohorts_and_exact_costs(self):
        counts = []
        for dataset in self.native_policy['datasets']:
            for backbone in self.native_policy['backbones']:
                study, count, banks = self.native_study(dataset, backbone, self.native_policy)
                validate(study)
                self.assertTrue(all(r['matched'] for r in reference_matches(study)))
                self.assertTrue(all(r['budget']['maximum'] == 5000 for r in study['runs'] if r['kind'] == 'model_fit'))
                search = [r for r in study['runs'] if r['config']['stage'] == 'search']
                self.assertEqual(sum(r['config']['practical_source_anchor'] for r in search), len(self.native_policy['template_rosters'][backbone]) * 2)
                self.assertEqual(study['data']['evidence']['practical_comparative_split_roster'], self.native_policy['datasets'][dataset]['comparative_split_ids'])
                self.assertTrue(study['study_id'].startswith('native-practical-'))
                counts.append(count)
        self.assertEqual(sum(c['initial_training_segments'] for c in counts), 1809)
        self.assertEqual(sum(c['full_budget_optimizer_updates'] for c in counts), 9045000)
        self.assertEqual(sum(sum(c['physical_body_instances_by_stage'].values()) for c in counts), 2247)

    def test_coordinate_enumerates_source_shapes_I4_and_four_optimizer_cost(self):
        study, counts, banks = coordinate.make_study(PACKAGE, custody('amazon-photo', range(10)), PACKAGE,
            source(), PROFILE, self.coordinate_recipes, None, 'frozen/runtime', 'frozen/coordinate.py', 'frozen/native.py',
            practical_policy=self.coordinate_policy)
        self.assertTrue(all(r['matched'] for r in reference_matches(study)))
        self.assertEqual(counts['atomic_fit_run_groups_by_stage'], {'search': 260, 'comparative': 930})
        self.assertEqual(sum(counts['paid_optimizer_starts_by_stage'].values()), 1340)
        self.assertEqual(counts['full_budget_optimizer_step_calls'], 2680000)
        self.assertEqual(counts['genuine_I4_banks_and_reused_member0'], 180)
        self.assertEqual(set(counts['search_configuration_counts'].values()), {10})
        for bank in banks:
            if bank['body0_reused']:
                self.assertEqual(len(set(bank['fit_parents'])), 4)
                self.assertNotIn(bank['arm'], ['untied', 'heads'])
        for r in study['runs']:
            if r['kind'] == 'model_fit':
                self.assertEqual(r['budget']['maximum'], 2000)
                if r['config']['source_arm'] == 'untied':
                    self.assertEqual(r['config']['source_optimizer_count'], 4)

    def test_policy_cannot_omit_reference_or_pad_ineffective_points(self):
        mutations = [lambda p: p['template_rosters']['SAGE'].remove('original'),
                     lambda p: p['numeric_points'].append(copy.deepcopy(p['numeric_points'][0])),
                     lambda p: p.update(optimizer_updates=100),
                     lambda p: p.update(scalar_coefficients='neighbors')]
        for change in mutations:
            policy = copy.deepcopy(self.native_policy); change(policy)
            with self.assertRaises(ValueError):
                self.native_study('citeseer', 'SAGE', policy)
        policy = copy.deepcopy(self.coordinate_policy)
        policy['numeric_points_by_dataset']['amazon-photo']['single'][0]['width'] = 128
        with self.assertRaises(ValueError):
            coordinate.coordinate_practical_domain(policy, 'amazon-photo', 'single', 'single', self.coordinate_recipes)

    def test_exact_mask_scope_is_prospective_and_preserves_parent_custody(self):
        original = custody('amazon-photo', range(10))
        policy = copy.deepcopy(self.coordinate_policy)
        policy['datasets']['amazon-photo']['comparative_split_ids'] = [0, 4, 9]
        resolved = native.practical_custody(original, policy)
        self.assertEqual(len(original['splits']), 10)
        self.assertEqual([r['split_id'] for r in resolved['splits']], [0, 4, 9])
        self.assertEqual(resolved['practical_full_prepared_masks'], list(range(10)))
        for masks in ([0, 0], [4, 9], [0, 99]):
            policy['datasets']['amazon-photo']['comparative_split_ids'] = masks
            with self.assertRaises(ValueError):
                native.practical_custody(original, policy)

    def test_unrestricted_native_domain_and_resource_deferrals_are_preserved(self):
        study, counts, banks = self.native_study('citeseer', 'GCN')
        self.assertEqual(counts['optimizer_starts_by_stage']['search'], 1000)
        self.assertEqual(study['study_id'], 'native-v2-citeseer-gcn')
        self.assertNotIn('practical_policy', study['source'])
        profiles = {'citeseer/SAGE': {'reservation_mib': 8}, 'citeseer/GCN': {'reservation_mib': 8}}
        admitted, blocked = native.native_cell_admission(['citeseer'], profiles, {'allocation/gpu0': 16 * 1024**2}, {}, ['SAGE', 'GCN'])
        self.assertEqual(admitted, ['citeseer/GCN', 'citeseer/SAGE'])
        self.assertFalse(blocked)
        with self.assertRaises(ValueError):
            native.native_cell_admission(['citeseer'], profiles, {'allocation/gpu0': 16 * 1024**2}, {})

    def test_original_five_are_rejected_even_if_policy_is_tampered(self):
        study, counts, banks = self.native_study('citeseer', 'SAGE', self.native_policy)
        for dataset in ('roman-empire', 'amazon-ratings', 'minesweeper', 'questions', 'tolokers'):
            changed = copy.deepcopy(study); changed['dataset'] = dataset
            with self.assertRaisesRegex(ValueError, 'Original GNNM paper experiments'):
                validate(changed)


if __name__ == '__main__':
    unittest.main()

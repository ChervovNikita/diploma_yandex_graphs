"""Finite scientific-contract enumeration; no molecular arrays or models."""
import copy
from pathlib import Path
import unittest

from compile_molecular_matrix import make_study, parameter_domain, validate_practical_policy
from compile_molecular_boundary_matrix import append_boundary_reference, BOUNDARY_PIN
from compile_master_plan import typed_counts, reference_matches
from test_molecular_matrix import inputs


def policy():
    return {'schema': 'molecular-practical-policy-v1', 'datasets': ['ogbg-molhiv', 'ogbg-molbace'],
            'epoch_domains': [300], 'coefficient_policy': 'native_fixed',
            'numeric_search_points': [{'width': 256, 'depth': 5, 'weight_decay': 0,
                'learning_rate': rate, 'dropout': dropout}
                for rate in [.00025, .001, .004] for dropout in [.1, .3, .5]]}


class MolecularPracticalTests(unittest.TestCase):
    def test_full_grid_stays_unchanged_and_practical_anchor_is_effective(self):
        _, recipe, *_ = inputs()
        self.assertEqual(len(parameter_domain(recipe, 'be_unit_contrastive', 'neighbors')), 1080)
        self.assertEqual(len(validate_practical_policy(policy(), recipe)), 9)
        for arm in ('single', 'be_unit', 'be_unit_contrastive'):
            self.assertEqual(len(parameter_domain(recipe, arm, 'native_fixed', policy())), 9)
        for mutate in (lambda p: p['numeric_search_points'].pop(),
                       lambda p: p['numeric_search_points'][0].update(width=512),
                       lambda p: p.update(epoch_domains=[100]),
                       lambda p: p.update(coefficient_policy='neighbors')):
            invalid = policy(); mutate(invalid)
            with self.assertRaises(ValueError):
                validate_practical_policy(invalid, recipe)
        with self.assertRaises(ValueError):
            parameter_domain(recipe, 'single', 'neighbors', policy())

    def test_two_complete_binary_graph_studies_include_original_and_genuine_I4(self):
        studies = []
        for dataset, graphs in [('ogbg-molhiv', 32901), ('ogbg-molbace', 1210)]:
            custody, recipe, source, molecular_source, profile, exposure = inputs()
            custody['dataset'] = dataset
            custody['pack_descriptives'] = {'train': {'graphs': graphs}}
            source['practical_policy'] = {'contract': policy(), 'unrestricted_grid_unchanged': True}
            base, *_ = make_study(Path.cwd(), custody, recipe, source, molecular_source, profile,
                300, exposure, 'native_fixed', 'frozen-practical', policy())
            pin = {'path': 'frozen-original/models.py', 'bytes': BOUNDARY_PIN[0], 'sha256': BOUNDARY_PIN[1]}
            spec, detail, banks, domains = append_boundary_reference(base, 'frozen-practical', pin, [], 9)
            self.assertEqual(detail['initial_training_segments'], 225)
            self.assertEqual(detail['original_boundary_graph_fit_count'], 21)
            self.assertEqual(detail['optimizer_starts_by_stage'], {'search': 162, 'comparative': 63})
            self.assertEqual(detail['full_budget_epochs'], 67500)
            self.assertEqual(detail['bank_exports'], 39)
            self.assertEqual(next(iter(domains.values()))['configuration_count'], 9)
            self.assertTrue(all(r['matched'] for r in reference_matches(spec)))
            self.assertTrue(all(r['config']['train_minibatches_per_epoch'] == (graphs+127)//128
                                for r in spec['runs'] if r['kind'] == 'model_fit'))
            studies.append(spec)
        counts = typed_counts(studies)
        self.assertEqual(counts['totals']['atomic_fit_run_groups'], 450)
        self.assertEqual(counts['model_optimizer_step_call_total'], 18090000)
        self.assertEqual(counts['totals']['genuine_separately_selected_I4_banks'], 24)
        self.assertEqual(counts['totals']['reused_exact_member0_fit_references'], 24)


if __name__ == '__main__':
    unittest.main()

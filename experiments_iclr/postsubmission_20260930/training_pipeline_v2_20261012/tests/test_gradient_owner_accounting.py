"""Source gradient/optimizer ownership cannot be collapsed to one generic step."""
import unittest

from compile_master_plan import bank_roster, fit_semantics, typed_counts
from test_master_plan import run


def core(identifier, condition, *, owners=1, bodies=1, members=4, body=0):
    return run(identifier, 'model_fit', condition,
        {'stage': 'comparative', 'architecture': 'masked_core', 'core_condition': condition,
         'source_optimizer_count': owners, 'optimizer_calls_per_cycle': owners,
         'physical_slow_bodies': bodies, 'members': members,
         'macro_cycle_law': 'one_complete_old_state_all_owner_cycle'},
        body=body, budget_unit='epochs', maximum=5000)


class GradientOwnerAccountingTests(unittest.TestCase):
    def test_cmcl_assignment_replay_and_private_vjp_costs_are_charged(self):
        row = run('fit', 'model_fit', 'private_cmcl',
            {'stage': 'comparative', 'architecture': 'private_cmcl', 'cmcl_condition': 'private_cmcl',
             'source_optimizer_count': 1, 'physical_slow_bodies': 1, 'members': 4}, maximum=5000)
        semantics = fit_semantics(row)
        self.assertEqual(semantics['known_optimizer_step_calls'], 5000)
        self.assertEqual(semantics['known_source_TRAIN_forward_calls'], 40000)
        self.assertEqual(semantics['known_source_TRAIN_backward_calls'], 40000)
        row['config']['cmcl_condition'] = 'own_floor'
        self.assertEqual(fit_semantics(row)['known_source_TRAIN_forward_calls'], 20000)

    def test_masked_multi_owner_cycles_cannot_hide_optimizer_calls(self):
        row = core('fit', 'untied4_shared_decoder_core', owners=5, bodies=4)
        counts = typed_counts([{'study_id': 'fixture', 'runs': [row]}])
        self.assertEqual(counts['model_optimizer_step_call_total'], 25000)
        self.assertEqual(counts['totals']['paid_optimizer_starts'], 5)
        self.assertEqual(counts['totals']['physical_slow_body_instances'], 4)
        row['config']['optimizer_calls_per_cycle'] = 1
        with self.assertRaisesRegex(ValueError, 'every distinct'):
            fit_semantics(row)
        row['config']['optimizer_calls_per_cycle'] = 5
        row['budget']['unit'] = 'optimizer_updates'
        with self.assertRaisesRegex(ValueError, 'macro cycle'):
            fit_semantics(row)

    def test_common_selected_four_body_cohort_is_not_own_selected_I4(self):
        group = core('coupled', 'independent4_native', owners=4, bodies=4)
        bank = run('bank', 'ensemble_assembly', 'common_selected',
                   {'stage': 'bank_export', 'fit_parents': ['coupled']}, ['coupled'])
        entry = bank_roster({'study_id': 'fixture', 'runs': [group, bank]})[0]
        self.assertFalse(entry['genuine_I4'])
        fits = [core('fit'+str(b), 'single_native', members=1, body=b) for b in range(4)]
        ids = [r['run_id'] for r in fits]
        bank = run('I4', 'ensemble_assembly', 'own_selected',
                   {'stage': 'bank_export', 'fit_parents': ids}, ids)
        self.assertTrue(bank_roster({'study_id': 'fixture', 'runs': fits+[bank]})[0]['genuine_I4'])

    def test_private_adam_contract_keeps_gradient_law_unchanged(self):
        row = run('fit', 'model_fit', 'private_adam',
            {'stage': 'comparative', 'architecture': 'private_adam', 'core_condition': 'shared4_own',
             'source_optimizer_count': 1, 'physical_slow_bodies': 1, 'members': 4,
             'private_adam_normalization': 'private_R_S_epsilon_and_weight_decay_divided_by_M'}, maximum=5000)
        self.assertEqual(fit_semantics(row)['known_optimizer_step_calls'], 5000)
        row['config']['private_adam_normalization'] = 'divide_private_gradients_by_M'
        with self.assertRaisesRegex(ValueError, 'gradient scaling is unchanged'):
            fit_semantics(row)


if __name__ == '__main__':
    unittest.main()

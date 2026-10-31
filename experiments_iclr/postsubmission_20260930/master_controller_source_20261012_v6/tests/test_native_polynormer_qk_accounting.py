"""Native Polynormer two-view work is distinct from generic one-update work."""
import unittest

from compile_master_plan import POLYNORMER_QK_ARMS, fit_semantics, typed_counts
from test_master_plan import run


def qk(arm):
    operator,kind,members=POLYNORMER_QK_ARMS[arm]
    return run('qk-'+arm,'model_fit','native_'+arm,
               {'stage':'comparative','polynormer_qk_arm':arm,'polynormer_qk_operator':operator,
                'source_ownership_kind':kind,'architecture':'original_gnnm' if kind=='boundary4' else 'internal_be' if members==4 else 'single',
                'backbone':'native_Polynormer','members':members,'source_optimizer_count':1,
                'source_optimizer_calls_per_cycle':1,'physical_slow_bodies':1,
                'local_transition_updates':100,'source_TRAIN_forwards_per_update':2*members,
                'source_TRAIN_backwards_per_update':2*members,'is_original_gnnm':kind=='boundary4'},maximum=5000)


class NativePolynormerQKAccountingTests(unittest.TestCase):
    def test_all_arms_charge_two_complete_views_and_no_hidden_fit(self):
        for arm,(_,_,members) in POLYNORMER_QK_ARMS.items():
            with self.subTest(arm=arm):
                cost=fit_semantics(qk(arm))
                self.assertEqual(cost['optimizer_objects'],1)
                self.assertEqual(cost['physical_slow_bodies'],1)
                self.assertEqual(cost['known_optimizer_step_calls'],5000)
                self.assertEqual(cost['known_source_TRAIN_forward_calls'],10000*members)
                self.assertEqual(cost['known_source_TRAIN_backward_calls'],10000*members)
                self.assertEqual(cost['unstepped_pre_install_Adam_constructor_objects'],0 if arm in ('bare_native_single','original_gnnm') else 1)

    def test_generic_fields_cannot_hide_original_or_wrong_view_counts(self):
        for arm in ('native_tied_single','pre_sigmoid_split_be_init','original_gnnm'):
            for key,value in (('source_TRAIN_forwards_per_update',1),('source_TRAIN_backwards_per_update',1),
                              ('source_optimizer_count',2),('physical_slow_bodies',4),
                              ('is_original_gnnm',arm!='original_gnnm')):
                with self.subTest(arm=arm,key=key):
                    row=qk(arm);row['config'][key]=value
                    with self.assertRaisesRegex(ValueError,'Native Polynormer Q/K'):
                        fit_semantics(row)
        row=qk('native_tied_single');row['config']['polynormer_qk_arm']='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown native Polynormer'):
            fit_semantics(row)

    def test_typed_totals_separate_unstepped_constructor_overhead(self):
        rows=[qk(arm) for arm in POLYNORMER_QK_ARMS]
        result=typed_counts([{'study_id':'synthetic-qk','runs':rows}])
        totals=result['totals']
        self.assertEqual(totals['atomic_fit_run_groups'],10)
        self.assertEqual(totals['paid_optimizer_starts'],10)
        self.assertEqual(totals['unstepped_pre_install_Adam_constructor_objects'],8)
        self.assertEqual(result['model_optimizer_step_call_total'],50000)
        self.assertEqual(totals['partial_registered_known_source_TRAIN_forward_calls'],250000)
        self.assertEqual(totals['partial_registered_known_source_TRAIN_backward_calls'],250000)


if __name__=='__main__': unittest.main()

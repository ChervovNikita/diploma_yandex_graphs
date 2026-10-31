"""Pure lifecycle gating and no-unblinding checks."""
import copy
import unittest
from checkpoint_retention import release_plan,label_free_export_ready


def record():
    return {'fit_id':'synthetic-one','purpose':'synthetic_engineering','stage':'comparative',
        'fit_status':'complete','owner_terminal':True,'history_closed':True,'serving_equivalence_verified':True,
        'HPO_group_closed':True,'SELECT_history_saved':True,'label_free_bank_closed':True,
        'unfinished_state_dependents':[],'continuation_admissions_closed':True,
        'full_checkpoint_receipts':[{'path':'selected.pt','bytes':100,'sha256':'a'*64}],
        'serving_receipt':{'path':'selected_serving.pt','bytes':50,'sha256':'b'*64}}


class Lifetimes(unittest.TestCase):
    def test_terminal_complete_candidate_is_only_a_draft(self):
        x=release_plan(record());self.assertTrue(x['eligible']);self.assertFalse(x['delete_executed']);self.assertFalse(x['source_policy_installed'])
    def test_all_required_ownership_and_state_gates(self):
        for k,v in [('fit_status','failed'),('owner_terminal',False),('history_closed',False),
                ('serving_equivalence_verified',False),('label_free_bank_closed',False),
                ('unfinished_state_dependents',['paid-child']),('continuation_admissions_closed',False)]:
            r=record();r[k]=v;self.assertFalse(release_plan(r)['eligible'],k)
    def test_HPO_losers_wait_for_whole_HPO_SELECT_closure(self):
        r=record();r['stage']='search';r['label_free_bank_closed']=False
        self.assertTrue(release_plan(r)['eligible'])
        for k in ('HPO_group_closed','SELECT_history_saved'):
            s=copy.deepcopy(r);s[k]=False;self.assertFalse(release_plan(s)['eligible'])
    def test_historical_and_serving_deletion_rejected(self):
        r=record();r['purpose']='historical_evidence'
        with self.assertRaises(ValueError):release_plan(r)
        r=record();r['full_checkpoint_receipts'].append(r['serving_receipt'])
        with self.assertRaises(ValueError):release_plan(r)
    def test_label_free_gate_has_no_endpoint_scores_or_labels(self):
        r={'phase':'label_free_bank_export','all_fit_parents_complete':True,'selected_authorities_verified':True,
            'reads_CAL_or_EVALUATE_targets':False,'emits_predictive_scores':False,'registered_endpoint_ids_only':True}
        self.assertTrue(label_free_export_ready(r))
        for k,v in [('phase','calibration'),('all_fit_parents_complete',False),('selected_authorities_verified',False),
                ('reads_CAL_or_EVALUATE_targets',True),('emits_predictive_scores',True),('registered_endpoint_ids_only',False)]:
            s=dict(r);s[k]=v;self.assertFalse(label_free_export_ready(s),k)


if __name__=='__main__':unittest.main()

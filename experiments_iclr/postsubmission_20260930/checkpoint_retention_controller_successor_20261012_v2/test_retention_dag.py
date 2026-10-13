"""Pure conservative consumer/HPO leases and phase exception tests."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'runtime'))
from retention_lifecycle import consumer_leases,enabled,early_export_key,validate_plan_policy,SCHEMA


def fixture():
    rows=[{'run_id':'warm','kind':'model_fit','parents':[],'config':{'stage':'comparative'}},
        {'run_id':'post','kind':'model_fit','parents':['warm'],'config':{'stage':'comparative'}},
        {'run_id':'bank','kind':'ensemble_assembly','parents':['post'],'config':{'stage':'bank_export'}},
        {'run_id':'cal','kind':'calibration','parents':['bank'],'config':{'stage':'calibration'}},
        {'run_id':'evaluate','kind':'evaluation','parents':['bank','cal'],'config':{'stage':'evaluation'}}]
    study={'study_id':'toy','runs':rows,'release':{'checkpoint_retention_policy':{'schema':SCHEMA,'existing_output_migration':False,'audited_label_free_exports':{'bank':{'family':'node'}}}}}
    class Registry:
        def __init__(self):self.states={r['run_id']:'registered' for r in rows}
        def run(self,study,id):return {'state':self.states[id]}
        def consumer_records(self,study,id):
            children={r['run_id']:[] for r in rows}
            for r in rows:
                for p in r['parents']:children[p].append(r['run_id'])
            seen=set();todo=list(children[id])
            while todo:
                k=todo.pop()
                if k in seen:continue
                seen.add(k);todo.extend(children[k])
            return {'direct_ids':children[id],'transitive_ids':sorted(seen),'states':{k:self.states[k] for k in seen}}
    return study,Registry()


class DAG(unittest.TestCase):
    def test_full_paid_warm_waits_for_transitive_endpoint_consumers(self):
        s,r=fixture();r.states.update(warm='complete',post='complete',bank='complete')
        leases=consumer_leases(s,r,'warm');self.assertTrue(leases['paid_source_parent']);self.assertFalse(leases['closed'])
        self.assertEqual(leases['outstanding_full_state_consumers'],['cal','evaluate'])
        r.states.update(cal='complete',evaluate='complete');self.assertTrue(consumer_leases(s,r,'warm')['closed'])
    def test_completed_audited_bank_closes_own_selected_optimizer_lease(self):
        s,r=fixture();r.states.update(warm='complete',post='complete',bank='complete')
        self.assertTrue(consumer_leases(s,r,'post')['closed']);self.assertEqual(r.states['cal'],'registered')
    def test_failed_source_or_export_keeps_full_authority(self):
        s,r=fixture();r.states.update(post='failed',bank='failed')
        self.assertFalse(consumer_leases(s,r,'warm')['closed']);self.assertFalse(consumer_leases(s,r,'post')['closed'])
    def test_unknown_exporter_conservatively_keeps_full_state(self):
        s,r=fixture();s['release']['checkpoint_retention_policy']['audited_label_free_exports']={};r.states['bank']='complete'
        self.assertFalse(consumer_leases(s,r,'post')['closed'])
    def test_CAL_and_EVALUATE_never_enter_early_export_roster(self):
        s,r=fixture();byid={x['run_id']:x for x in s['runs']}
        self.assertEqual(early_export_key(s,byid['bank']),'toy/bank')
        self.assertIsNone(early_export_key(s,byid['cal']));self.assertIsNone(early_export_key(s,byid['evaluate']))
    def test_existing_output_migration_is_rejected(self):
        s,r=fixture();s['release']['checkpoint_retention_policy']['existing_output_migration']=True
        with self.assertRaises(ValueError):enabled(s)


if __name__=='__main__':unittest.main()

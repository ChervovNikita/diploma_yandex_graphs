"""Pure metadata tests; no graph/target decoding, numerical imports or fits."""
import copy
import unittest
from pathlib import Path

from adapters.label_stage import ARMS
from compile_paid_label_packet import make_study, SEARCH, COMPARE, NATIVE_DOMAIN, LABEL_DOMAIN
from compile_master_plan import graph_check, typed_counts, fit_semantics, bank_roster, reference_matches
from run_study import paid_parent_ids


def fixture(dataset):
    count,classes=(20,10) if dataset=='wikics' else (10,8)
    files={}
    def add(path):
        files[path]={'path':path,'bytes':0,'sha256':'0'*64}
        return path
    graph=add('graph.pt');splits=[]
    for split in range(count):
        nodes={'graph':graph,**{r:add(str(split)+'/'+r+'.pt') for r in ('train','select')}}
        ends={r+'_targets':add(str(split)+'/'+r+'_targets.pt') for r in ('cal','evaluate')}
        ends.update({r+'_ids':add(str(split)+'/'+r+'_ids.pt') for r in ('cal','evaluate')})
        splits.append({'split_id':split,'node_inputs':nodes,'endpoint_inputs':ends})
    custody={'dataset_id':dataset,'splits':splits,'files':list(files.values()),
        'model_graph':{'classes':classes,'nodes':11701 if classes==10 else 7650},
        'request':{'graph_policy':{'synthetic_metadata':True},'evaluation_exposure':{'previously_evaluated':True}},'request_hash':'0'*64}
    root=Path('/metadata-fixture')
    source={'git_commit':'0'*40,'files':[{'path':'synthetic_metadata_source.py','bytes':0,'sha256':'0'*64}]}
    labels={'capture':{'path':'capture.py','bytes':0,'sha256':'1'*64}}
    return make_study(root,custody,root/'inputs'/dataset,source,{},labels,'runtime')[0]


class PaidLabelPacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.studies=[fixture(d) for d in ('wikics','amazon-photo')]

    def label(self,study):return next(r for r in study['runs'] if r['kind']=='model_fit' and r['method'] in ARMS and r['config']['stage']=='comparative')

    def assert_bad_graph(self,mutation):
        study=copy.deepcopy(self.studies[0]);run=self.label(study);mutation(study,run)
        with self.assertRaises(ValueError):graph_check([study],{})

    def test_exact_finite_counts_and_source_owners(self):
        values=typed_counts(self.studies);t=values['totals']
        expected={'run_records':4800,'atomic_fit_run_groups':1444,'paid_optimizer_starts':2056,
            'physical_slow_body_instances':526,'registered_optimizer_updates':2630000,'registered_epochs':4590000,
            'known_registered_optimizer_step_calls':10280000,'partial_registered_known_source_TRAIN_forward_calls':24340000,
            'partial_registered_known_source_TRAIN_backward_calls':21280000,'unstepped_pre_install_Adam_constructor_objects':204,
            'paid_native_capture_jobs':94,'search_selection_jobs':22,'bank_exports':1080,'scalar_CAL_fit_jobs':1080,
            'registered_scalar_CAL_updates':540000,'evaluation_jobs':1080,'genuine_separately_selected_I4_banks':90,
            'served_route_views':2970,'reused_exact_member0_fit_references':90}
        self.assertEqual({k:t[k] for k in expected},expected)
        self.assertTrue(values['model_optimizer_step_call_total_fully_determined'])
        self.assertEqual(values['actual_model_optimizer_updates'],0)
        for study in self.studies:
            for r in study['runs']:
                if r['kind']=='model_fit' and r['method'] in ARMS:
                    arm=ARMS[r['method']];s=fit_semantics(r)
                    self.assertEqual(s['optimizer_objects'],arm[6]);self.assertEqual(s['physical_slow_bodies'],0)
                    self.assertEqual(s['known_optimizer_step_calls'],5000*arm[6])
                    self.assertEqual(s['known_source_TRAIN_forward_calls'],5000*arm[4])
                    self.assertEqual(s['known_source_TRAIN_backward_calls'],5000*arm[5])

    def test_all_masks_domains_and_original_gnnm_new_data(self):
        self.assertEqual(len(NATIVE_DOMAIN),9);self.assertEqual(len(LABEL_DOMAIN),3)
        self.assertEqual(SEARCH,(81101,81103));self.assertEqual(COMPARE,(81201,81203,81207))
        excluded={'roman-empire','amazon-ratings','minesweeper','questions','tolokers'}
        for study,count in zip(self.studies,(20,10)):
            self.assertNotIn(study['dataset'],excluded)
            for method in ('original_gnnm',*ARMS):
                fits=[r for r in study['runs'] if r['kind']=='model_fit' and r['method']==method]
                comparative=[r for r in fits if r['config']['stage']=='comparative']
                self.assertEqual({(r['split'],r['seed']) for r in comparative},{(s,k) for s in range(count) for k in COMPARE})
                for r in fits:
                    self.assertEqual(r['budget']['maximum'],5000);self.assertFalse(r['selection']['checkpoint0'])
                    self.assertEqual(r['config']['historical_native_horizon'],1100)
            self.assertTrue(all(v['matched'] for v in reference_matches(study)))
        self.assertEqual(sum(r['method']=='original_gnnm' and r['kind']=='model_fit' for s in self.studies for r in s['runs']),126)

    def test_capture_is_paid_reused_and_phase_acyclic(self):
        for study in self.studies:
            records,deps=graph_check([study],{})
            whole_search=[k for k,v in records.items() if v['run']['config']['stage'] in ('search','search_selection')]
            captures=[r for r in study['runs'] if r['config']['stage']=='native_capture']
            self.assertTrue(captures)
            for capture in captures:
                self.assertNotIn(study['study_id']+'/'+capture['run_id'],whole_search)
                users=[r for r in study['runs'] if r['kind']=='model_fit' and r['config'].get('capture_parent')==capture['run_id']]
                self.assertEqual(len(users),27 if capture['run_id'].startswith('capture-label-source-search') else 9)
                self.assertEqual({paid_parent_ids(r)[0] for r in users},{capture['config']['native_fit_parent']})
            comparative=[k for k,v in records.items() if v['run']['config']['stage']=='comparative']
            # A conceptual whole-search closure depends on only these search records.
            # No search record may be downstream of a comparative acquisition/capture.
            def ancestors(key):
                result=set(deps[key])
                for p in deps[key]:result.update(ancestors(p))
                return result
            for key in whole_search:self.assertFalse(set(comparative)&ancestors(key))

    def test_missing_capture_paid_fit_and_wrong_native_rejected(self):
        self.assert_bad_graph(lambda s,r:r['parents'].remove(r['config']['capture_parent']))
        self.assert_bad_graph(lambda s,r:s['runs'].__setitem__(slice(None),[x for x in s['runs'] if x['run_id']!=paid_parent_ids(r)[0]]))
        def wrong(s,r):
            native=next(x for x in s['runs'] if x['run_id']==paid_parent_ids(r)[0]);native['config']['polynormer_qk_arm']='original_gnnm'
        self.assert_bad_graph(wrong)

    def test_short_old_fit_wrong_capture_or_free_lifecycle_rejected(self):
        def old(s,r):next(x for x in s['runs'] if x['run_id']==paid_parent_ids(r)[0])['budget']['maximum']=1100
        def capture(s,r):
            c=next(x for x in s['runs'] if x['run_id']==r['config']['capture_parent'])
            c['config']['capture_source']={**c['config']['capture_source'],'sha256':'2'*64}
        def wrong_phase(s,r):next(x for x in s['runs'] if x['run_id']==r['config']['capture_parent'])['config']['stage']='search_selection'
        self.assert_bad_graph(old);self.assert_bad_graph(capture);self.assert_bad_graph(wrong_phase)
        self.assert_bad_graph(lambda s,r:r['config']['lifecycle'].__setitem__('new_label_optimizer',False))
        self.assert_bad_graph(lambda s,r:r['selection'].__setitem__('checkpoint0',True))

    def test_genuine_i4_distinct_ancestry_and_required_references(self):
        for study in self.studies:
            banks=bank_roster(study);i4=[b for b in banks if b['genuine_I4']]
            self.assertEqual(len(i4),len({b['split'] for b in i4})*3)
            self.assertTrue(all(b['member0_reused'] and b['routes']==4 for b in i4))
            damaged=copy.deepcopy(study)
            bank=next(r for r in damaged['runs'] if r['kind']=='ensemble_assembly' and r['method']=='native_independent4')
            fits={r['run_id']:r for r in damaged['runs']};fits[bank['config']['fit_parents'][1]]['config']['initialization_seed']=fits[bank['config']['fit_parents'][0]]['config']['initialization_seed']
            with self.assertRaises(ValueError):bank_roster(damaged)
            missing=copy.deepcopy(study);missing['runs']=[r for r in missing['runs'] if r['method']!='original_gnnm']
            self.assertTrue(any(not c['matched'] for c in reference_matches(missing)))


if __name__=='__main__':unittest.main()

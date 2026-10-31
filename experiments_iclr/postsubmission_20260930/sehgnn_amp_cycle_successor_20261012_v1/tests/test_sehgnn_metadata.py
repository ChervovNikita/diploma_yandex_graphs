"""Finite source roster and controller scope; standard library only."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from compile_sehgnn_packet import make_study,ARMS,DOMAIN,SEARCH,COMPARE
from compile_master_plan import typed_counts,graph_check,bank_roster,fit_semantics
from spec import validate
from run_study import load_factory


def fixture():
    files=[];splits=[]
    def file(path):files.append({'path':path,'bytes':1,'sha256':'a'*64});return path
    feature={v:file('features/'+v+'.pt') for v in ('full','actor','director','keyword')}
    for split in range(10):splits.append({'split_id':split,'node_inputs':{r:file(str(split)+'/'+r+'.pt') for r in ('train','select')},'endpoint_inputs':{r+f:file(str(split)+'/'+r+f+'.pt') for r in ('cal','evaluate') for f in ('_ids','_targets')},'label_caches':{v:file(str(split)+'/'+v+'.pt') for v in feature}})
    c={'dataset_id':'hgb-imdb-native5-development','class_dim':5,'splits':splits,'files':files,'graph_feature_caches':feature,'preprocessing':'native','prior_evaluation_exposure':'development','schema':'fixture','node_counts':[4932,2393,6124,7971],'raw_input_files':{}}
    return make_study(Path('/project'),c,Path('/project/custody'),{'git_commit':'fixture','files':[{'path':'runtime/a.py','bytes':1,'sha256':'a'*64}]},{'modules':{},'metadata':{}},'runtime')


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.study=fixture()
    def test_finite_counts(self):
        s=self.study;c=typed_counts([s])['totals'];self.assertEqual(len(ARMS),20);self.assertEqual(len(DOMAIN),9)
        for key,value in [('atomic_fit_run_groups',1230),('paid_optimizer_starts',1518),('physical_slow_body_instances',1518),('known_registered_optimizer_step_calls',906000),('AMP_bounded_commit_fit_groups',612),('registered_known_sehgnn_optimizer_attempts',1518000),('bounded_optimizer_step_calls_minimum',0),('bounded_optimizer_step_calls_maximum',612000),('run_records',3320),('bank_exports',690),('scalar_CAL_fit_jobs',690),('evaluation_jobs',690),('registered_scalar_CAL_updates',345000),('search_selection_jobs',20),('genuine_separately_selected_I4_banks',90),('reused_exact_member0_fit_references',90)]:self.assertEqual(c[key],value,key)
    def test_attempt_commit_bounds(self):
        c=typed_counts([self.study]);self.assertFalse(c['model_optimizer_step_call_total_fully_determined']);self.assertIsNone(c['model_optimizer_step_call_total']);self.assertEqual(c['native_SeHGNN_optimizer_attempt_total'],1518000);self.assertEqual(c['native_SeHGNN_optimizer_commit_bounds'],[906000,1518000]);self.assertEqual(c['model_optimizer_step_call_minimum'],906000);self.assertEqual(c['model_optimizer_step_call_maximum'],1518000);self.assertEqual(c['finite_bounded_trial_groups'],0);self.assertEqual(c['native_AMP_bounded_commit_fit_groups'],612)
    def test_source_reverse_trial_costs(self):
        rows={r['method']:r for r in self.study['runs'] if r['kind']=='model_fit'}
        for arm,vjp in [('native_pool_credit',15),('source_view_supervision',15),('assigned_source_supply',18),('COMMON_cycle',18),('uncoupled_source_contrast',18)]:
            x=fit_semantics(rows[arm]);self.assertEqual(x['known_source_TRAIN_forward_calls'],10138);self.assertEqual(x['known_source_TRAIN_backward_calls'],4000+198*vjp);self.assertEqual(x['known_sehgnn_scalar_leaf_reverse_calls'],2376);self.assertEqual(x['known_sehgnn_private_trial_forward_upper_bound'],8712)
        self.assertEqual(fit_semantics(rows['shared_own_only'])['known_source_TRAIN_forward_calls'],4000)
        self.assertEqual(fit_semantics(rows['independent_ADD_PS'])['optimizer_objects'],4)
    def test_original_new_data_and_masks(self):
        originals=[r for r in self.study['runs'] if r['kind']=='model_fit' and r['method']=='original_gnnm'];self.assertEqual(len(originals),48);self.assertEqual({r['split'] for r in originals},set(range(10)));self.assertTrue(all(r['config']['fusion']=='mean_logits' for r in originals))
        for old in ('Roman Empire','Amazon Ratings','Minesweeper','Questions','Tolokers'):
            s=copy.deepcopy(self.study);s['dataset']=old
            with self.assertRaises(ValueError):validate(s)
    def test_prefixes_width_horizon_are_narrow(self):
        for field,value in [('width',256),('classes',3),('sehgnn_epochs',200),('historical_state_inherited',True)]:
            s=copy.deepcopy(self.study);s['runs'][0]['config'][field]=value
            with self.assertRaises(ValueError):graph_check([s],{})
        s=copy.deepcopy(self.study);s['runs'][0]['budget']['prefixes'][0]['close_on_selector_stop']=True
        with self.assertRaises(ValueError):graph_check([s],{})
    def test_genuine_I4_and_grouped_controls(self):
        banks=bank_roster(self.study);self.assertEqual(len(banks),690);self.assertEqual(sum(r['genuine_I4'] for r in banks),90)
        grouped=[r for r in banks if r['method'] in ('independent_ADD_P','independent_ADD_PS')];self.assertEqual(len(grouped),60);self.assertTrue(all(not r['genuine_I4'] for r in grouped))
    def test_packet_imports_own_namespace_without_torch(self):
        root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parents[1]/'runtime'
        for entry in ('sehgnn.py:make_adapter','sehgnn_endpoints.py:export_bank','sehgnn_endpoints.py:scalar_calibration','sehgnn_endpoints.py:evaluate_bank'):
            file,fun=entry.split(':');factory=load_factory(root,str((base/'adapters'/file).relative_to(root))+':'+fun);self.assertTrue(callable(factory));self.assertTrue(factory.__module__.startswith('gnnm_packet_'))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()

"""Metadata-only link plan coverage and unused resource-helper guards."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from compile_master_plan import graph_check,reference_matches
from compile_link_matrix import make_study
from partition import DEFAULT_WORKERS
from prepare_link_plan import make_plan
from qualify_link_cuda import construct_native
from spec import fingerprint
from test_link import config


class LinkPlanTests(unittest.TestCase):
    def fixture(self):
        cfg=config();custody={k:cfg[k] for k in ('link_inputs','link_source','sampler','metric','support')}
        source={'git_commit':'synthetic','files':list(cfg['link_source'].values())}
        profile={'reservation_mib':32768,'wall_seconds':250000000,'estimated_seconds_per_epoch':1800,
            'estimated_selection_seconds':60,'estimated_export_seconds':7200,'evidence':'unmeasured synthetic fixture'}
        exposure={'protected_confirmation':False,'evaluate_purpose':'development','prior_exposure':'synthetic'}
        sealed={'nodes':8,'endpoint_inputs':{'evaluate_queries':{'path':'synthetic/queries.pt','bytes':1,'sha256':'0'*64},
            'evaluate_targets':{'path':'synthetic/targets.pt','bytes':1,'sha256':'1'*64}}}
        return make_study(custody,source,profile,exposure,sealed=sealed)[0]

    def test_exact_study_is_unchanged_and_all_1276_jobs_assigned_once(self):
        study=self.fixture();before=fingerprint(study);caps={w:80*1024**3 for w in DEFAULT_WORKERS}
        plan,summary=make_plan(study,'frozen/STUDY_DISABLED.json','frozen/runtime',caps,'fresh/link_plan')
        self.assertEqual(fingerprint(study),before)
        self.assertEqual(plan['studies'],[{'path':'frozen/STUDY_DISABLED.json','spec_hash':before}])
        keys=[key for group in plan['groups'] for key in group['run_keys']]
        self.assertEqual(len(keys),1276);self.assertEqual(len(set(keys)),1276)
        self.assertEqual({g['gpu_assignment'] for g in plan['groups']},{'allocation/gpu0'})
        self.assertEqual((summary['model_fits'],summary['registered_epochs'],summary['source_bound_declared_optimizer_updates']),(1250,125000,2125000))
        self.assertEqual((summary['CAL_fit_jobs'],summary['member0_copy_exports'],summary['evaluation_jobs']),(0,6,12))
        self.assertEqual(summary['plan_hash'],fingerprint(plan))
        self.assertFalse(plan['scientific_launch_enabled']);self.assertTrue(plan['human_GO_required'])
        self.assertNotIn('whole_calibration',plan['global_gates'])
        records,_=graph_check([study],{})
        self.assertEqual(set(records),set(keys));self.assertTrue(all(r['matched'] for r in reference_matches(study)))

    def test_three_capacities_and_memory_admission_are_required(self):
        study=self.fixture()
        with self.assertRaisesRegex(ValueError,'three logical'):make_plan(study,'s.json','runtime',{'allocation/gpu0':80*1024**3},'out')
        with self.assertRaisesRegex(ValueError,'accommodate'):make_plan(study,'s.json','runtime',{w:16*1024**3 for w in DEFAULT_WORKERS},'out')

    def test_unused_qualifier_copies_exact_native_constructor_controls(self):
        calls=[]
        def capture(name):
            def constructor(*args,**kwargs):calls.append((name,args,kwargs));return name
            return constructor
        native=SimpleNamespace(GCN=capture('GCN'),CNLinkPredictor=capture('NCN'),IncompleteCN1Predictor=capture('NCNC'))
        for predictor in ('NCN','NCNC'):
            cfg=config(predictor=predictor);encoder,decoder=construct_native(native,128,cfg)
            self.assertEqual((encoder,decoder),('GCN',predictor))
            self.assertEqual(calls[-2][1],(128,256,256,1,.1))
            self.assertEqual(calls[-1][1],(256,256,1,3,.3))
            if predictor=='NCNC':self.assertEqual(calls[-1][2]['depth'],cfg['predictor']['completion']['depth'])
        tree=ast.parse((Path(__file__).resolve().parents[1]/'qualify_link_cuda.py').read_text())
        calls_ast=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in {'Adam','AdamW','step'} for n in calls_ast))
        loads=[n for n in calls_ast if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr=='load']
        self.assertEqual(len(loads),1)
        inputs=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inputs' for t in n.targets))
        self.assertIsInstance(inputs,ast.DictComp)
        self.assertEqual(ast.literal_eval(inputs.generators[0].iter),('graph','train'))


if __name__=='__main__':unittest.main()

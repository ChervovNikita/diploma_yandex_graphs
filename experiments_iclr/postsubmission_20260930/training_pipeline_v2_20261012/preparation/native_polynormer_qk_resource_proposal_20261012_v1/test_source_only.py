"""Concrete source/proposal invariants; no Torch, data arrays, fit or CUDA."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

from qualify_native_qk_resources import validate_proposal,warm_steps,fingerprint,H_SHA,Z_SHA

HERE=Path(__file__).resolve().parent
V=HERE.parents[1];P=V.parent
MIRROR=V/'preparation/native_polynormer_qk_cpu_handoff_20261012_v1/allocation_artifacts'


class SourceOnlyTests(unittest.TestCase):
    def setUp(self):self.proposal=json.loads((HERE/'PROPOSAL.json').read_text())

    def test_exact_roster_source_run_identity_and_no_original_five(self):
        cases=validate_proposal(self.proposal)
        studies={}
        for case in cases:
            path=case['study_receipt']['path']
            if path not in studies:
                raw=(MIRROR/path).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),case['study_receipt']['sha256'])
                studies[path]=json.loads(gzip.decompress(raw))
            study=studies[path];self.assertEqual(fingerprint(study),case['study_hash'])
            run=next(r for r in study['runs'] if r['run_id']==case['source_run_id'])
            self.assertEqual(fingerprint(run),case['source_run_hash'])
            self.assertEqual(run['config'],case['source_run_config'])
            self.assertEqual(run['config']['polynormer_qk_arm'],case['arm'])
            self.assertEqual(case['GRAPH_TRAIN_bindings'],{k:run['config']['node_inputs'][k] for k in ('graph','train')})
        self.assertEqual(len(cases),8);self.assertEqual({c['dataset'] for c in cases},{'wikics','amazon-photo'})

    def test_bad_geometry_steps_and_rosters_fail(self):
        for change in ('width','warm_steps','arm','branch','count'):
            p=copy.deepcopy(self.proposal)
            if change=='width':p['requested_cases'][0]['shape_config']['width']=256
            elif change=='warm_steps':p['requested_cases'][0]['warm_callback_steps']=[1,2,3,4]
            elif change=='arm':p['requested_cases'][0]['arm']='native_tied_single'
            elif change=='branch':p['requested_cases'][0]['branch']='learned_global'
            else:p['requested_cases'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):validate_proposal(p)
        self.assertEqual(warm_steps(),(1,1,1,1))

    def test_exact_streamed_source_has_two_views_four_routes_and_final_Adam(self):
        p=P/'pre_sigmoid_qk_operator_adapter_source_20261009_v1/operator_adapter.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'be9974399e090b424c57fe4f77ba10126f575d8e89454945ce51f0f7c61b9631')
        function=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_streamed_f_step')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call)]
        ranges=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='range']
        self.assertTrue(any(len(n.args)==1 and isinstance(n.args[0],ast.Constant) and n.args[0].value==2 for n in ranges))
        forwards=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='member_forward']
        backwards=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='backward']
        commits=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='step']
        self.assertEqual((len(forwards),len(backwards),len(commits)),(1,1,1))
        self.assertLess(forwards[0].lineno,backwards[0].lineno);self.assertLess(backwards[0].lineno,commits[0].lineno)
        for case in self.proposal['requested_cases']:
            self.assertEqual(case['source_run_config']['members'],4)
            self.assertEqual(case['expected_actions_per_cycle']['model_forwards'],8)
            self.assertEqual(case['expected_actions_per_cycle']['backward_calls'],8)

    def test_helper_uses_restored_step1_native_body_hooks_and_no_selected_snapshot(self):
        tree=ast.parse((HERE/'qualify_native_qk_resources.py').read_text())
        probe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='node_probe')
        calls=[n for n in ast.walk(probe) if isinstance(n,ast.Call)]
        train=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='train_step']
        self.assertEqual(len(train),1);self.assertEqual(ast.dump(train[0].args[0]),"Name(id='step', ctx=Load())")
        loop=next(n for n in ast.walk(probe) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='warm_steps')
        self.assertEqual(ast.dump(loop.target),"Name(id='step', ctx=Store())")
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='restore_state' for n in ast.walk(loop.body[0])))
        setters=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='set_global']
        captures=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='capture_state']
        self.assertLess(setters[0].lineno,min(n.lineno for n in captures))
        hooks=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='register_forward_hook']
        self.assertEqual(ast.unparse(hooks[0].func.value),'wrapper.body')
        self.assertFalse(any(isinstance(n,ast.Subscript) and isinstance(n.ctx,ast.Store) and isinstance(n.slice,ast.Constant) and n.slice.value=='snapshot' for n in ast.walk(probe)))
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('validate','observe_validation','evaluate','export') for n in calls))

    def test_reviewed_guard_source_bytes_and_disabled_admission(self):
        bindings=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
        self.assertEqual(bindings['role_optimizer_cap_helper']['sha256'],H_SHA)
        self.assertEqual(bindings['tensor_commit_tree_helper']['sha256'],Z_SHA)
        self.assertEqual(hashlib.sha256((V/'qa/practical_zero_update_resources.py').read_bytes()).hexdigest(),H_SHA)
        self.assertEqual(hashlib.sha256((V/'qa/private_source_zero_update_resources.py').read_bytes()).hexdigest(),Z_SHA)
        admission=json.loads((HERE/'ENGINEERING_ADMISSION_TEMPLATE.json').read_text())
        self.assertFalse(admission['root_source_review_confirmed']);self.assertEqual(admission['approved_case_ids'],[])
        self.assertFalse(admission['scientific_GO']);self.assertEqual(admission['max_actual_optimizer_updates'],0)
        self.assertEqual(admission['runtime_package'],'master_controller_source_20261012_v6/runtime')
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()

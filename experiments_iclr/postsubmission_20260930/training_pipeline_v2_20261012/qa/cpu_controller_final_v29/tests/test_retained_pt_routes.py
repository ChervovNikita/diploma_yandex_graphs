"""Synthetic source/ancestry checks only; no benchmark TRAIN/CAL/TEST decoders."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from adapters.retained_pt_routes import (make_adapter, receipt, compile_source, PINS, Streams, tensor_hash)
from compile_retained_pt_matrix import make_study
from engine import run_trajectory, capture_checkpoint, restore_checkpoint, inspect_ownership, inspect_independence
from spec import fingerprint

SOURCE_PATHS = {
    'pt':'native_meanSAGE_private_continuation_PT_core_source_20261011_v2/core.py',
    'native':'native_meanSAGE_attribute_peer_feedback_core_source_20261011_v1/core.py',
    'original':'common_wrapper_native_source_20261010_v1/models.py',
    'original_driver':'common_wrapper_native_source_20261010_v1/run_common.py',
}


class FakeRegistry:
    def __init__(self): self.rows = {}
    def run(self,study,identifier): return self.rows[identifier]


class RetainedPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t = torch; torch.set_num_threads(1)
        cls.sources = Path(os.environ.get('GNNM_REFERENCE_PROJECT',Path(__file__).resolve().parents[2]))
        for rel in SOURCE_PATHS.values():
            if not (cls.sources/rel).is_file(): raise RuntimeError('Required exact pinned source missing: '+rel)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.bindings = {}
        for key,relative in SOURCE_PATHS.items():
            destination = self.root/relative; destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(self.sources/relative,destination)
            self.bindings[key] = receipt(self.root,destination); self.assertEqual(self.bindings[key]['sha256'],PINS[key])
        t = self.t; x = t.arange(40,dtype=t.float32).reshape(8,5)/37
        edges = t.tensor([[i for i in range(8)]+[i for i in range(8)],[i for i in range(8)]+[(i+1)%8 for i in range(8)]])
        packs = {'graph':{'x':x,'edge_index':edges},'train':{'ids':t.tensor([0,1,2,3]),'y':t.tensor([0,1,2,0])},'select':{'ids':t.tensor([4,5]),'y':t.tensor([1,2])}}
        self.inputs = {}
        for key,value in packs.items():
            path = self.root/(key+'.pt'); t.save(value,path); self.inputs[key] = receipt(self.root,path)
        self.study = {'study_id':'synthetic-pt','stage':'engineering','dataset':'wikics',
            'data':{'roles':{'train':{'0':self.inputs['train']},'select':{'0':self.inputs['select']}}},'source':{'files':list(self.bindings.values()),'git_commit':'synthetic'}}
        self.registry = FakeRegistry(); self.context = {'registry':self.registry,'output_root':self.root/'out'}

    def run_config(self,arm,body=0,parent=None,identifier=None):
        source = arm in ('ordinary_p1','factor_f1','factor_s4','original_gnnm')
        members = 4 if arm in ('factor_s4','original_gnnm','head_only','private4','middle_state_mean','separable4') else 1
        config = {'pt_arm':arm,'pt_stage':'source' if source else 'continuation','architecture':'original_gnnm' if arm=='original_gnnm' else 'retained_pt_routes',
            'weight_decay':0.,'learning_rate':.001,'dropout':.4,'width':128 if arm=='original_gnnm' else 32,'depth':2,'classes':3,'members':members,
            'physical_slow_bodies':4 if arm=='separable4' else 1,'source_optimizer_count':1,'paired_seed':10103,'initialization_seed':10103+1000003*body,
            'pt_sources':self.bindings,'node_inputs':self.inputs,'fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','stage':'search'}
        if parent: config['lifecycle'] = {'schema':'paid-parent-continuation-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}
        return {'run_id':identifier or arm+'-b'+str(body),'kind':'model_fit','method':arm,'split':0,'seed':10103,'body':body,'adapter':'adapters/retained_pt_routes.py:make_adapter',
            'config':config,'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':2},
            'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False},'parents':[parent] if parent else []}

    def adapter(self,run):
        return make_adapter(torch=self.t,project_root=self.root,study=self.study,run=run,device='cpu',execution_context=self.context)

    def paid_source(self,arm='ordinary_p1',body=0):
        run = self.run_config(arm,body); adapter = self.adapter(run)
        result = run_trajectory(self.t,adapter,{'fit_id':run['run_id'],'budget_steps':2,'budget_unit':'optimizer_updates',
            'validation_interval':1,'validate_initial':False,'prefixes':[],'recovery_interval':2,'inherited_fit_ids':[],
            'adapter_spec':{'run':run,'synthetic_only':True}},self.context['output_root']/self.study['study_id']/run['run_id'])
        self.registry.rows[run['run_id']] = {'state':'complete','contract':run,'result':result}
        return run,adapter,result

    def test_source_anchor_and_persistent_stream_ownership(self):
        t = self.t; from torch.nn import functional as F
        run = self.run_config('ordinary_p1'); adapter = self.adapter(run)
        namespace = {'__name__':'native_reference'}
        exec((self.root/SOURCE_PATHS['native']).read_text(),namespace)
        namespace.update(NODES=8,FEATURES=5,HIDDEN=32,CLASSES=3,BLOCKS=(10103,))
        outer = t.get_rng_state().clone(); raw = namespace['build_arm'](t,F,'P1',10103)['models'][0]
        self.assertTrue(t.equal(outer,t.get_rng_state()))
        self.assertEqual(set(raw.state_dict()),set(adapter.model.native.state_dict()))
        for key,value in raw.state_dict().items(): self.assertTrue(t.equal(value,adapter.model.native.state_dict()[key]))
        adapter.model.eval(); raw.eval(); graph = t.load(self.root/'graph.pt',weights_only=True)
        A = t.sparse_coo_tensor(graph['edge_index'].flip(0),t.ones(16),(8,8)).coalesce(); degree = t.full((8,1),2.)
        self.assertTrue(t.equal(adapter.model(),raw(graph['x'],A,degree)))
        streams = Streams(t,'cpu',[11,13]); outer = t.get_rng_state().clone(); before = copy.deepcopy(streams.states)
        with streams.scope(0): t.rand(19)
        self.assertTrue(t.equal(outer,t.get_rng_state())); self.assertFalse(t.equal(before[0]['cpu'],streams.states[0]['cpu'])); self.assertTrue(t.equal(before[1]['cpu'],streams.states[1]['cpu']))
        first = self.adapter(self.run_config('factor_f1',0)); second = self.adapter(self.run_config('factor_f1',1))
        self.assertEqual(inspect_independence([first,second])['independent_adapters'],2)

    def test_zero_residual_bias_coordinates_middle_and_delayed_gradient(self):
        parent,source,result = self.paid_source(); base = parent['run_id']; t = self.t
        adapters = {arm:self.adapter(self.run_config(arm,parent=base)) for arm in ('head_only','mixed_single','bias_coordinate_single','private4','middle_state_mean')}
        source.model.eval()
        expected = source.model()
        for adapter in adapters.values():
            adapter.model.eval(); actual = adapter.model()
            self.assertTrue(t.allclose(actual,expected.expand_as(actual),atol=1e-7,rtol=0))
            self.assertFalse(adapter.optimizer.state); self.assertEqual(adapter.custody['parent']['checkpoint']['sha256'],result['selected_checkpoint']['sha256'])
        mixed,bias = adapters['mixed_single'],adapters['bias_coordinate_single']
        for left,right in zip(mixed.model.continuations[0].memory.blocks,bias.model.continuations[0].memory.blocks):
            self.assertTrue(t.equal(left.G1,right.G1))
        self.assertEqual(bias.model.continuations[0].memory.e.numel(),32)
        self.assertEqual(float(bias.model.continuations[0].memory.e.mean(1).abs().max()),0)
        mixed.model.train(); z = mixed.model(); loss = t.stack([mixed.F.cross_entropy(v[mixed.train_ids],mixed.train_y) for v in z]).mean(); loss.backward()
        memory = mixed.model.continuations[0].memory
        self.assertGreater(float(memory.Wo.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum())==0 for block in memory.blocks for p in block.parameters()))
        mixed.optimizer.step(); mixed.optimizer.zero_grad(set_to_none=True)
        z = mixed.model(); t.stack([mixed.F.cross_entropy(v[mixed.train_ids],mixed.train_y) for v in z]).mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in memory.blocks[0].W.values()),0)
        # G1 can remain zero while zero G2 delays its reachability. No all-groups-by-step2 assertion.
        private,middle = adapters['private4'],adapters['middle_state_mean']
        private.model.eval(); middle.model.eval()
        _,details = middle.model.continuations[0](return_details=True)
        average = t.stack(details['first_states']).mean(0)
        self.assertEqual(len({id(v) for v in details['second_banks']}),1)
        self.assertTrue(t.equal(average,details['second_banks'][0]['H']))
        _,pdetails = private.model.continuations[0](return_details=True)
        self.assertEqual(len({id(v) for v in pdetails['second_banks']}),4)

    def test_actual_parent_hash_full_horizon_and_cache_guards(self):
        parent,source,result = self.paid_source(); run = self.run_config('mixed_single',parent=parent['run_id'])
        adapter = self.adapter(run); checkpoint = self.root/'continuation.pt'
        payload = capture_checkpoint(self.t,adapter,fit_id=run['run_id'],step=1,validation={'accuracy':0.},decision={'save':True})
        self.t.save(payload,checkpoint); restore_checkpoint(self.t,checkpoint,adapter,expected_fit_id=run['run_id'])
        self.registry.rows[parent['run_id']]['result']['completed_steps'] = 1
        with self.assertRaisesRegex(ValueError,'complete source horizon'): self.adapter(run)
        self.registry.rows[parent['run_id']]['result']['completed_steps'] = 2
        self.registry.rows[parent['run_id']]['result']['selected_checkpoint']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'hash changed'): self.adapter(run)
        with self.t.no_grad(): adapter.model.continuations[0].fixed_H0.add_(.01)
        with self.assertRaisesRegex(ValueError,'mutated'): adapter.model()
        adapter.custody['parent']['selected_step'] += 1
        with self.assertRaisesRegex(ValueError,'ancestry changed'): restore_checkpoint(self.t,checkpoint,adapter,expected_fit_id=run['run_id'])

    def test_bias_coordinate_mean_gradients_are_real_optimizer_redundancy(self):
        parent,source,result = self.paid_source(); adapter = self.adapter(self.run_config('bias_coordinate_single',parent=parent['run_id']))
        adapter.train_step(1); metrics = adapter.train_step(2); e = adapter.model.continuations[0].memory.e
        self.assertGreater(float(e.grad.abs().sum()),0)
        for block in range(2):
            for route in range(1,4): self.assertTrue(self.t.equal(e.grad[block,0],e.grad[block,route]))
        self.assertEqual(metrics['operation_counts']['pt_native_head_decode_calls'],4)
        self.assertEqual(metrics['operation_counts']['pt_block_bank_calls'],2)
        self.assertEqual(metrics['operation_counts']['pt_graph_propagations'],10)

    def test_new_separable_and_full_native_continued_controls(self):
        parent,source,result = self.paid_source(); sep = self.adapter(self.run_config('separable4',parent=parent['run_id']))
        self.assertEqual(len(sep.model.continuations),4); stem = sep.model.continuations[0].native.bodies[0].layers[0]
        self.assertTrue(all(body.native.bodies[0].layers[0] is stem for body in sep.model.continuations))
        live = [{id(p) for p in body.parameters() if p.requires_grad} for body in sep.model.continuations]
        self.assertTrue(all(not live[i]&live[j] for i in range(4) for j in range(i)))
        inspect_ownership(sep); sep.model.train(); self.assertEqual(sep.model().shape,(16,8,3)); sep.model.eval(); self.assertEqual(sep.model().shape,(4,8,3))
        self.assertEqual(sep.model.operation_counts(True)['pt_native_head_decode_calls'],16)
        self.assertEqual(sep.model.operation_counts(True)['pt_graph_propagations'],40)
        for native_arm,continued_arm in [('ordinary_p1','continued_p1'),('factor_f1','continued_f1')]:
            if native_arm=='ordinary_p1': actual_parent,actual_source = parent,source
            else: actual_parent,actual_source,_ = self.paid_source(native_arm)
            adapter = self.adapter(self.run_config(continued_arm,parent=actual_parent['run_id']))
            self.assertTrue(all(p.requires_grad for p in adapter.model.parameters())); self.assertFalse(adapter.optimizer.state)
            for key,value in actual_source.model.native.state_dict().items(): self.assertTrue(self.t.equal(value,adapter.model.native.state_dict()[key]))
            adapter.model.train(); self.assertEqual(adapter.model().shape,(4,8,3)); adapter.model.eval(); self.assertEqual(adapter.model().shape,(1,8,3))
            before = tensor_hash(adapter.model.native.bodies[0].layers[0].aggregate_weight); adapter.train_step(1)
            self.assertNotEqual(before,tensor_hash(adapter.model.native.bodies[0].layers[0].aggregate_weight))

    def test_original_native_reference_and_no_cal_test_fit_authority(self):
        adapter = self.adapter(self.run_config('original_gnnm')); adapter.model.eval()
        self.assertEqual(adapter.model().shape,(4,8,3)); self.assertEqual(adapter.fusion,'mean_logits')
        bad = self.run_config('ordinary_p1'); bad['config']['node_inputs']['evaluate'] = self.inputs['select']
        with self.assertRaisesRegex(ValueError,'Only GRAPH'): self.adapter(bad)

    def test_full_finite_roster_counts_and_source_reuse(self):
        totals = {}; studies = []
        for dataset,size in [('wikics',20),('amazon-photo',10)]:
            files = []; rows = []
            for split in range(size):
                node = {'graph':'graph.pt','train':f'train{split}.pt','select':f'select{split}.pt'}
                end = {k:f'{k}{split}.pt' for k in ('cal_ids','evaluate_ids','cal_targets','evaluate_targets')}
                rows.append({'split_id':split,'node_inputs':node,'endpoint_inputs':end,'evaluate_purpose':'development'})
                for path in list(node.values())+list(end.values()): files.append({'path':path,'bytes':1,'sha256':'a'*64})
            files = list({v['path']:v for v in files}.values())
            custody = {'dataset_id':dataset,'splits':rows,'files':files,'status':'prepared','benchmark_fits':0,'predictive_scores':0,'model_graph':{'classes':3},
                       'request':{'graph_policy':'unchanged','evaluation_exposure':'development'},'request_hash':'fixture'}
            study,counts,banks = make_study(self.root,custody,self.root,self.study['source'],self.bindings,{'*':{'reservation_mib':1,'wall_seconds':10000,'estimated_seconds_per_update':1,'evidence':'synthetic'}},'runtime')
            studies.append(study)
            for k,v in counts.items():
                if type(v)==int: totals[k]=totals.get(k,0)+v
            for bank in banks:
                if bank['genuine_I4']: self.assertEqual(len(bank['fit_parents']),4)
            ids = {r['run_id']:r for r in study['runs']}
            for run in study['runs']:
                if run['kind']=='model_fit' and run['config']['pt_stage']=='continuation':
                    parent = ids[run['config']['lifecycle']['parent_fit_ids'][0]]
                    self.assertEqual(parent['config']['pt_stage'],'source'); self.assertEqual((run['split'],run['seed'],run['body']),(parent['split'],parent['seed'],parent['body']))
        for key,value in {'atomic_fit_groups':3140,'physical_body_instances':3518,'source_acquisition_fits':1052,'continuations':2088,'search_selection_jobs':24,'comparative_banks':1620,'all_run_records':8024,'full_budget_optimizer_updates':15700000}.items(): self.assertEqual(totals[key],value)


if __name__=='__main__': unittest.main()

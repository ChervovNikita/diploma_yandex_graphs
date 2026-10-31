"""Synthetic/source CPU fixtures; no benchmark fit or CUDA dispatch claim."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from adapters.gru_memory import (PINS,make_adapter,prepare_tour_capsule,load_source,compile_gru,compile_source,receipt,tensor_hash)
from adapters.gru_memory_endpoints import export_bank,I16_CONTRACT
from compile_gru_memory_matrix import make_study,SEARCH_SEEDS,COMPARATIVE_SEEDS
from engine import run_trajectory,capture_checkpoint,restore_checkpoint,inspect_independence

SOURCE_PATHS={'gru':'native_meanSAGE_private_GRU_core_source_20261011_v1/core.py',
    'backend':'native_meanSAGE_private_GRU_core_source_20261011_v1/gru_backend.py',
    'tours':'native_meanSAGE_tour_scan_preparation_source_20261011_v1/tours.py',
    'native':'native_meanSAGE_attribute_peer_feedback_core_source_20261011_v1/core.py',
    'original':'common_wrapper_native_source_20261010_v1/models.py','original_driver':'common_wrapper_native_source_20261010_v1/run_common.py'}


class Registry:
    def __init__(self): self.rows={}
    def run(self,study,identifier): return self.rows[identifier]


class GRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch; torch.set_num_threads(1); cls.reference=Path(os.environ.get('GNNM_REFERENCE_PROJECT',Path(__file__).resolve().parents[2]))
        for p in SOURCE_PATHS.values():
            if not (cls.reference/p).is_file(): raise RuntimeError('Exact GRU fixture source missing: '+p)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name); self.bindings={}
        for key,relative in SOURCE_PATHS.items():
            target=self.root/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(self.reference/relative,target)
            self.bindings[key]=receipt(self.root,target); self.assertEqual(self.bindings[key]['sha256'],PINS[key])
        runtime=Path(__file__).resolve().parents[1]
        for rel in ('engine.py','spec.py','run_study.py','registry.py','adapters/__init__.py','adapters/classifier.py','adapters/retained_pt_routes.py','adapters/gru_memory.py'):
            target=self.root/'runtime'/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(runtime/rel,target)
        t=self.t; x=t.arange(40,dtype=t.float32).reshape(8,5)/37; pairs=[(i,i) for i in range(8)]+[(i,i+1) for i in range(7)]+[(i+1,i) for i in range(7)]
        packs={'graph':{'x':x,'edge_index':t.tensor(pairs,dtype=t.long).T.contiguous()},
            'train':{'ids':t.tensor([0,1,2,3]),'y':t.tensor([0,1,2,0])},'select':{'ids':t.tensor([4,5]),'y':t.tensor([1,2])}}
        self.inputs={}
        for key,value in packs.items():
            path=self.root/(key+'.pt'); t.save(value,path); self.inputs[key]=receipt(self.root,path)
        capsule=prepare_tour_capsule(t,self.root,self.inputs['graph'],self.bindings,10103); path=self.root/'tours.pt'; t.save(capsule,path)
        self.capsule=capsule; self.tours={'capsule':receipt(self.root,path),'payload_hash':capsule['payload_hash']}
        self.study={'study_id':'synthetic-gru','stage':'engineering','dataset':'wikics','data':{'roles':{'train':{'0':self.inputs['train']},'select':{'0':self.inputs['select']}}},'source':{'git_commit':'synthetic','files':list(self.bindings.values())}}
        self.registry=Registry(); self.context={'registry':self.registry,'output_root':self.root/'out'}

    def run_config(self,arm,body=0,parent=None):
        source=arm in ('ordinary_p1','factor_f1','factor_s4','original_gnnm'); members=4 if arm in ('B4','G1','GF1','GM4','GN4','factor_s4','original_gnnm') else 1
        c={'architecture':'original_gnnm' if arm=='original_gnnm' else 'private_gru_memory','gru_arm':arm,'gru_stage':'source' if source else 'continuation',
            'source_optimizer_count':1,'physical_slow_bodies':1,'members':members,'width':128 if arm=='original_gnnm' else 32,'depth':2,'classes':3,'weight_decay':0.,'learning_rate':.001,'dropout':.4,
            'paired_seed':10103,'initialization_seed':10103+1000003*body,'gru_sources':self.bindings,'node_inputs':self.inputs,'gru_tours':self.tours if arm in ('B4','G1','GF1','GM4','GN4') else None,
            'fusion':'mean_logits' if arm=='original_gnnm' else 'mean_probabilities','stage':'search'}
        if parent: c['lifecycle']={'schema':'paid-parent-continuation-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}
        return {'run_id':arm+'-b'+str(body),'kind':'model_fit','method':arm,'split':0,'seed':10103,'body':body,'adapter':'runtime/adapters/gru_memory.py:make_adapter','config':c,
            'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':2},'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False},'parents':[parent] if parent else []}

    def adapter(self,run): return make_adapter(torch=self.t,project_root=self.root,study=self.study,run=run,device='cpu',execution_context=self.context)
    def fit_synthetic(self,run):
        adapter=self.adapter(run); result=run_trajectory(self.t,adapter,{'fit_id':run['run_id'],'budget_steps':2,'budget_unit':'optimizer_updates','validation_interval':1,
            'validate_initial':False,'prefixes':[],'recovery_interval':2,'inherited_fit_ids':run['parents'],'adapter_spec':{'run':run,'synthetic_only':True}},self.context['output_root']/self.study['study_id']/run['run_id'])
        self.registry.rows[run['run_id']]={'state':'complete','contract':run,'result':result}; return adapter,result
    def source(self,body=0,arm='ordinary_p1'):
        run=self.run_config(arm,body); adapter,result=self.fit_synthetic(run); return run,adapter,result

    def test_exact_source_construction_anchor_and_shape_binding(self):
        t=self.t; from torch.nn import functional as F
        native=compile_source(self.root,self.bindings['native'],dropout=.4,dimensions=(34493,8415,32,5),seed=10103)
        ancestor=native.build_arm(t,F,'P1',10103)['models'][0]; H=t.zeros(34493,32); backend=load_source(self.root,self.bindings['backend'])
        raw=load_source(self.root,self.bindings['gru']); bound_core=compile_gru(self.root,self.bindings['gru'],nodes=34493,features=8415,classes=5,seed=10103,dropout=.4)
        outside=t.get_rng_state().clone()
        for arm in ('B4','G1','GF1','GM4','GN4'):
            first=raw.build_arm(t,F,arm,10103,[ancestor],[H],gru_backend=backend); second=bound_core.build_arm(t,F,arm,10103,[ancestor],[H],gru_backend=backend)
            self.assertEqual(raw.inspect_parameter_ownership(first),bound_core.inspect_parameter_ownership(second))
            for key,value in first['models'][0].state_dict().items(): self.assertTrue(t.equal(value,second['models'][0].state_dict()[key]))
        self.assertTrue(t.equal(outside,t.get_rng_state()))
        original=self.adapter(self.run_config('original_gnnm')); original.model.eval()
        self.assertEqual(original.model().shape,(4,8,3)); self.assertEqual(original.fusion,'mean_logits')

    def test_complete_tours_packing_reversal_and_rng_custody(self):
        t=self.t; backend=load_source(self.root,self.bindings['backend']); c=self.capsule
        self.assertEqual(c['components'],1); self.assertTrue(all(v['length']==15 for v in c['tours']))
        for tour,pack in zip(c['tours'],c['packed']):
            self.assertTrue(t.equal(tour['tokens'],pack['packed_tokens'][pack['unpack_to_tokens']]))
            self.assertEqual(int(tour['occurrences'].sum()),15); self.assertTrue(bool((tour['occurrences']>0).all()))
            self.assertTrue(t.equal(tour['tokens'],tour['tokens'][tour['reverse_index']][tour['reverse_index']]))
            self.assertEqual(pack['batch_sizes'].device.type,'cpu')
        graph=t.load(self.root/'graph.pt',weights_only=True); graph['edge_index']=t.tensor([(i,i) for i in range(8)]+[(0,1),(1,0),(2,3),(3,2)],dtype=t.long).T
        path=self.root/'disconnected.pt'; t.save(graph,path); outside=t.get_rng_state().clone(); new=prepare_tour_capsule(t,self.root,receipt(self.root,path),self.bindings,10103)
        self.assertEqual(new['components'],6); self.assertEqual(new['tours'][0]['length'],10); self.assertTrue(t.equal(outside,t.get_rng_state()))
        for p in new['packed']: self.assertEqual(int(p['batch_sizes'][0]),6)

    def test_tensor_list_reset_after_live_factor_derivatives_cpu(self):
        t=self.t; backend=load_source(self.root,self.bindings['backend']); tour=self.capsule['tours'][0]; packed=self.capsule['packed'][0]; g=t.Generator().manual_seed(41); d=32
        values=[t.randn(8,d,generator=g,dtype=t.float64)*.1,t.randn(3*d,d,generator=g,dtype=t.float64)*.05,t.randn(3*d,d,generator=g,dtype=t.float64)*.05,
            t.randn(3,d,generator=g,dtype=t.float64)*.1+1,t.randn(3,d,generator=g,dtype=t.float64)*.1+1,t.randn(3,d,generator=g,dtype=t.float64)*.1+1,
            t.randn(3*d,generator=g,dtype=t.float64)*.01,t.randn(3*d,generator=g,dtype=t.float64)*.01,t.randn(d,d,generator=g,dtype=t.float64)*.02]
        first=[v.clone().requires_grad_() for v in values]; second=[v.clone().requires_grad_() for v in values]
        def weights(v):
            H,A,B,rx,rh,s,bi,bh,Wo=v
            return [(s[:,:,None]*A.reshape(3,d,d)*rx[:,None,:]).reshape(3*d,d),(s[:,:,None]*B.reshape(3,d,d)*rh[:,None,:]).reshape(3*d,d),bi,bh]
        def manual(v):
            H,*_=v; A,B,bi,bh=weights(v); tokens=tour['tokens']; offsets=tour['offsets'].tolist(); full=[]
            def sequence(ids):
                h=H.new_zeros(d); out=[]
                for index in ids:
                    a=(H[index]@A.T+bi).reshape(3,d); b=(h@B.T+bh).reshape(3,d)
                    r=(a[0]+b[0]).sigmoid(); z=(a[1]+b[1]).sigmoid(); n=(a[2]+r*b[2]).tanh(); h=z*h+(1-z)*n; out.append(h)
                return t.stack(out)
            for lo,hi in zip(offsets[:-1],offsets[1:]):
                ids=tokens[lo:hi].tolist(); full.append((sequence(ids)+sequence(ids[::-1]).flip(0))*.5)
            node=H.new_zeros(H.shape).index_add(0,tokens,t.cat(full))/tour['occurrences'][:,None]
            return node
        node=backend.node_memory(t,first[0],weights(first),packed,True); reference=manual(second)
        self.assertTrue(t.allclose(node,reference,atol=1e-10,rtol=1e-10))
        probe=t.randn(8,d,generator=g,dtype=t.float64)
        def final(v,node):
            z=v[0]+.25*(node@v[-1]); return z/t.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1.)
        left,right=final(first,node),final(second,reference); (left*probe).sum().backward(); (right*probe).sum().backward()
        for a,b in zip(first,second): self.assertIsNotNone(a.grad); self.assertTrue(t.allclose(a.grad,b.grad,atol=1e-9,rtol=1e-9))
        self.assertTrue(all(float(v.grad.abs().sum())>0 for v in first))

    def test_paid_source_zero_readout_delayed_memory_and_live_GN4(self):
        parent,source,result=self.source(); t=self.t; source.model.eval(); expected=source.model()
        adapters={arm:self.adapter(self.run_config(arm,parent=parent['run_id'])) for arm in ('B4','G1','GF1','GM4','GN4')}
        for arm,adapter in adapters.items():
            adapter.model.eval(); self.assertTrue(t.allclose(adapter.model(),expected.expand(4,-1,-1),atol=1e-7,rtol=0)); self.assertFalse(adapter.optimizer.state)
            self.assertFalse(any('fixed_H' in key for key in adapter.model.state_dict()))
            self.assertEqual(adapter.custody['parent']['checkpoint']['sha256'],result['selected_checkpoint']['sha256'])
        g1=adapters['G1']; metrics=g1.train_step(1); memory=g1.model.memory_body.memory
        self.assertGreater(float(memory.W_o.grad.abs().sum()),0); self.assertEqual(float(memory.W_ih.grad.abs().sum()),0)
        self.assertEqual(metrics['operation_counts']['backward_calls'],4); self.assertEqual(metrics['operation_counts']['GRU_direction_forward_API_calls'],8)
        g1.train_step(2); self.assertGreater(float(memory.W_ih.grad.abs().sum()),0)
        gn=adapters['GN4']; self.assertIsNone(gn.custody['H_hash']); self.assertFalse(hasattr(gn.model.memory_body,'fixed_H'))
        gn.train_step(1); first=gn.model.memory_body.native.bodies[0].layers[0]
        self.assertIsNone(first.aggregate_weight.grad); self.assertGreater(float(first.aggregate_r.grad.abs().sum()),0)
        self.assertTrue(first.aggregate_r.requires_grad and first.bias_offset.requires_grad)
        self.registry.rows[parent['run_id']]['result']['selected_checkpoint']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'byte authority'): self.adapter(self.run_config('G1',parent=parent['run_id']))

    def test_factor_control_private_rows_and_immutable_restore(self):
        parent,source,result=self.source(); t=self.t
        g1=self.adapter(self.run_config('G1',parent=parent['run_id'])); gf=self.adapter(self.run_config('GF1',parent=parent['run_id'])); gm=self.adapter(self.run_config('GM4',parent=parent['run_id']))
        with t.no_grad():
            read=t.eye(32)*.03
            for a in (g1,gf,gm): a.model.memory_body.memory.W_o.copy_(read); a.model.eval()
        self.assertTrue(t.allclose(g1.model(),gf.model(),atol=1e-7,rtol=0)); self.assertEqual(gm.model.memory_body.memory.r_x.shape,(4,3,32))
        gm.model.train(); out=gm.model.view(0); gm.F.cross_entropy(out[gm.train_ids],gm.train_y).backward(); fast=gm.model.memory_body.memory.r_x.grad
        self.assertGreater(float(fast[0].abs().sum()),0); self.assertEqual(float(fast[1:].abs().sum()),0)
        payload=capture_checkpoint(t,gf,fit_id='gf-fixture',step=1,validation={'accuracy':0.},decision={'save':True}); path=self.root/'gf.pt'; t.save(payload,path)
        restore_checkpoint(t,path,gf,expected_fit_id='gf-fixture')
        before=gf.custody['H_hash']; self.assertEqual(before,tensor_hash(gf.model.memory_body.fixed_H))
        with t.no_grad(): gf.model.memory_body.fixed_H.add_(.01)
        with self.assertRaisesRegex(ValueError,'mutated'): gf.model()

    def test_four_own_four_view_export_and_native_continued(self):
        fits=[]; adapters=[]
        for body in range(4):
            parent,source,result=self.source(body); run=self.run_config('G1',body,parent['run_id']); adapter,result=self.fit_synthetic(run); fits.append(run['run_id']); adapters.append(adapter)
        self.assertEqual(inspect_independence(adapters)['independent_adapters'],4)
        ids={}
        for role,values in [('cal',[6]),('evaluate',[7])]:
            path=self.root/(role+'_ids.pt'); self.t.save(self.t.tensor(values),path); ids[role+'_ids']=str(path.relative_to(self.root))
        out=self.root/'export'; out.mkdir(); run={'parents':fits,'config':{'fit_parents':fits,'endpoint_inputs':ids,'fusion':'mean_probabilities','independent_bank_contract':I16_CONTRACT}}
        exported=export_bank(project_root=self.root,study=self.study,run=run,output_root=self.context['output_root'],output_dir=out,registry=self.registry,device='cpu')
        self.assertEqual(exported['served_views'],16); self.assertFalse(exported['CAL_targets_read']); bank=self.t.load(out/'cal_logits.pt',weights_only=True)
        self.assertEqual(bank['logits'].shape,(16,1,3)); self.assertEqual(bank['view_body_ids'].tolist(),sum(([i]*4 for i in range(4)),[]))
        parent=self.registry.rows['ordinary_p1-b0']['contract']; continued=self.adapter(self.run_config('continued_p1',parent=parent['run_id']))
        self.assertTrue(all(p.requires_grad for p in continued.model.parameters())); self.assertFalse(continued.optimizer.state)
        continued.model.train(); self.assertEqual(continued.model().shape,(4,8,3)); continued.model.eval(); self.assertEqual(continued.model().shape,(1,8,3))

    def test_finite_roster_and_explicit_sixteen_view_bank_contract(self):
        totals={}
        for dataset,size in [('wikics',20),('amazon-photo',10)]:
            files=[]; rows=[]
            for split in range(size):
                node={'graph':'graph.pt','train':f'train{split}.pt','select':f'select{split}.pt'}; end={k:f'{k}{split}.pt' for k in ('cal_ids','evaluate_ids','cal_targets','evaluate_targets')}
                rows.append({'split_id':split,'node_inputs':node,'endpoint_inputs':end,'evaluate_purpose':'development'})
                for p in list(node.values())+list(end.values()): files.append({'path':p,'bytes':1,'sha256':'a'*64})
            custody={'dataset_id':dataset,'splits':rows,'files':list({v['path']:v for v in files}.values()),'status':'prepared','benchmark_fits':0,'predictive_scores':0,'model_graph':{'classes':3},'request':{'graph_policy':'unchanged','evaluation_exposure':'development'},'request_hash':'synthetic'}
            tours={'dataset_id':dataset,'capsules':[{'seed':seed,'capsule':{'path':'tours.pt','bytes':1,'sha256':'b'*64},'payload_hash':'c'*64} for seed in SEARCH_SEEDS+COMPARATIVE_SEEDS]}
            study,counts,banks=make_study(self.root,custody,self.root,tours,self.study['source'],self.bindings,{'*':{'reservation_mib':1,'wall_seconds':10000,'estimated_seconds_per_update':1,'evidence':'synthetic'}},'runtime')
            for k,v in counts.items():
                if type(v)==int: totals[k]=totals.get(k,0)+v
            byid={r['run_id']:r for r in study['runs']}
            for bank in banks:
                if bank['genuine_I16']:
                    endpoint=byid[bank['export']]; self.assertEqual(endpoint['config']['independent_bank_contract'],I16_CONTRACT)
                    self.assertEqual([byid[p]['config']['members'] for p in bank['fit_parents']],[4]*4)
        for k,v in {'atomic_fit_groups':3014,'physical_body_instances':3014,'source_acquisition_fits':1052,'continuations':1962,'search_selection_jobs':22,'comparative_banks':1530,'genuine_I16_banks':180,'all_run_records':7626,'full_budget_optimizer_updates':15070000}.items(): self.assertEqual(totals[k],v)


if __name__=='__main__': unittest.main()

"""Synthetic CPU source algebra/custody proof; no benchmark fit or CUDA."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest

from adapters.landmark_memory import (PINS,SOURCE_PATHS,NATIVE,make_adapter,prepare_graph_capsule,compile_core,isolated_helpers,receipt,tensor_hash)
from adapters.landmark_memory_endpoints import export_bank
from compile_landmark_memory_matrix import make_study,SEARCH_SEEDS,COMPARATIVE_SEEDS
from engine import run_trajectory,capture_checkpoint,restore_checkpoint,inspect_independence
from compile_master_plan import graph_check,typed_counts,bank_roster


class Registry:
    def __init__(self): self.rows={}
    def run(self,study,identifier): return self.rows[identifier]


class LandmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch; torch.set_num_threads(1); cls.reference=Path(os.environ.get('GNNM_REFERENCE_PROJECT',Path(__file__).resolve().parents[2]))
        for path in SOURCE_PATHS.values():
            if not (cls.reference/path).is_file(): raise RuntimeError('Exact landmark source fixture missing: '+path)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name); self.bindings={}
        for key,relative in SOURCE_PATHS.items():
            path=self.root/relative; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(self.reference/relative,path)
            self.bindings[key]=receipt(self.root,path); self.assertEqual(self.bindings[key]['sha256'],PINS[key])
        runtime=Path(__file__).resolve().parents[1]
        for rel in ('engine.py','spec.py','run_study.py','registry.py','adapters/__init__.py','adapters/classifier.py','adapters/node_losses.py','adapters/retained_pt_routes.py','adapters/polyformer.py','adapters/polyformer_boundary.py','adapters/landmark_memory.py'):
            path=self.root/'runtime'/rel; path.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(runtime/rel,path)
        t=self.t; g=t.Generator().manual_seed(71); x=t.rand(72,5,generator=g)*3
        links=[(i,i+1) for i in range(71)]; pairs=[(i,i) for i in range(72)]+links+[(v,u) for u,v in links]
        packs={'graph':{'x':x,'edge_index':t.tensor(pairs,dtype=t.long).T.contiguous()},'train':{'ids':t.arange(36),'y':t.arange(36)%3},'select':{'ids':t.arange(36,60),'y':t.arange(36,60)%3}}
        self.inputs={}
        for key,value in packs.items():
            path=self.root/(key+'.pt'); t.save(value,path); self.inputs[key]=receipt(self.root,path)
        self.capsule=prepare_graph_capsule(t,self.root,self.inputs['graph'],self.bindings,10103); path=self.root/'geometry.pt'; t.save(self.capsule,path)
        self.geometry={'capsule':receipt(self.root,path),'payload_hash':self.capsule['payload_hash']}
        self.study={'study_id':'synthetic-landmark','stage':'engineering','dataset':'wikics','data':{'roles':{'train':{'0':self.inputs['train']},'select':{'0':self.inputs['select']}}},'source':{'git_commit':'synthetic','files':list(self.bindings.values())}}
        self.registry=Registry(); self.context={'registry':self.registry,'output_root':self.root/'out'}

    def run_config(self,arm,body=0,parent=None):
        source=arm in ('ordinary_m1','factor_m1','shared_factor4','original_boundary'); members=4 if arm in ('B4','T16','LM4','shared_factor4','original_boundary') else 1
        config={'architecture':'original_gnnm' if arm=='original_boundary' else 'landmark_memory','landmark_arm':arm,'landmark_stage':'source' if source else 'continuation',
            'is_original_gnnm':arm=='original_boundary','source_optimizer_count':1,'physical_slow_bodies':1,'members':members,'width':256,'depth':2,'classes':3,
            'weight_decay':0.,'learning_rate':.005,'dropout':.5,'paired_seed':10103,'initialization_seed':10103+1000003*body,
            'landmark_sources':self.bindings,'node_inputs':self.inputs,'landmark_graph':self.geometry,'fusion':'mean_logits' if arm=='original_boundary' else 'mean_probabilities','stage':'search'}
        if parent: config['lifecycle']={'schema':'paid-parent-continuation-v1','parent_fit_ids':[parent],'new_optimizer':True,'parameter_initialization':'selected_parent_no_optimizer_reuse'}
        return {'run_id':arm+'-b'+str(body),'kind':'model_fit','method':arm,'split':0,'seed':10103,'body':body,'adapter':'runtime/adapters/landmark_memory.py:make_adapter','config':config,
            'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':2},'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False},'parents':[parent] if parent else []}

    def adapter(self,run): return make_adapter(torch=self.t,project_root=self.root,study=self.study,run=run,device='cpu',execution_context=self.context)
    def fit_synthetic(self,run):
        adapter=self.adapter(run); result=run_trajectory(self.t,adapter,{'fit_id':run['run_id'],'budget_steps':2,'budget_unit':'optimizer_updates','validation_interval':1,
            'validate_initial':False,'prefixes':[],'recovery_interval':2,'inherited_fit_ids':run['parents'],'adapter_spec':{'run':run,'synthetic_only':True}},self.context['output_root']/self.study['study_id']/run['run_id'])
        self.registry.rows[run['run_id']]={'state':'complete','contract':run,'result':result}; return adapter,result
    def source(self,body=0,arm='factor_m1'):
        run=self.run_config(arm,body); adapter,result=self.fit_synthetic(run); return run,adapter,result

    def test_exact_author_factor_and_unmodified_core_anchor(self):
        t=self.t; p,b=isolated_helpers(self.bindings); outside=t.get_rng_state().clone(); tokens=[t.zeros(72,500) for _ in range(3)]
        ordinary=p.build_bank(t,self.root,tokens,native=NATIVE,classes=3,dataset='PubMed',members=1,factorized=False,initialization_seed=9101,dropout_seeds=[41])
        factor=p.build_bank(t,self.root,tokens,native=NATIVE,classes=3,dataset='PubMed',members=1,factorized=True,initialization_seed=9101,dropout_seeds=[41])
        self.assertEqual(sum(v.numel() for v in ordinary.parameters()),2069875); self.assertEqual(sum(v.numel() for v in factor.parameters()),2083818); self.assertEqual(factor.factor_sites,23)
        for key,value in ordinary.body.state_dict().items(): self.assertTrue(t.equal(value,factor.body.state_dict()[key]))
        factor.eval(); ordinary.eval(); self.assertTrue(t.equal(ordinary(),factor())); self.assertTrue(t.equal(outside,t.get_rng_state()))
        head=factor.body.lin3; self.assertEqual(set(dict(head.named_parameters())),{'weight','bias','r','s'})
        with t.no_grad(): head.s.fill_(2); head.bias.fill_(.7)
        probe=t.ones(2,256); self.assertTrue(t.allclose(head(probe),t.nn.functional.linear(probe*head.r[0],head.weight)*head.s[0]+head.bias))
        raw={}; path=self.root/SOURCE_PATHS['core']; exec(compile(path.read_text(),str(path),'exec'),raw)
        bound_core=compile_core(self.root,self.bindings['core'],nodes=19717,features=500,classes=3,seed=9101,dropout=.5)
        H=t.zeros(19717,256); geometry={'anchors':t.arange(64),'bias':t.zeros(19717,64)}; factors=p.factor_definitions(t,self.root)
        for arm in ('B1','T16','S64','FS64','LM4'):
            first=raw['build_arm'](t,t.nn.functional,arm,9101,[factor.body],[{'H':H,'U':H}],geometry,FactorLinear=factors['FactorLinear'])
            second=bound_core.build_arm(t,t.nn.functional,arm,9101,[factor.body],[{'H':H,'U':H}],geometry,FactorLinear=factors['FactorLinear'])
            self.assertEqual(raw['inspect_parameter_ownership'](first),bound_core.inspect_parameter_ownership(second))
            for key,value in first['models'][0].state_dict().items(): self.assertTrue(t.equal(value,second['models'][0].state_dict()[key]))
        original=self.adapter(self.run_config('original_boundary')); original.model.eval(); self.assertEqual(original.model().shape,(4,72,3)); self.assertEqual(original.fusion,'mean_logits')
        self.assertEqual(original.model.native_bank.boundary_sites,('lin1','lin3')); self.assertEqual(original.model.native_bank.factor_sites,0)

    def test_complete_normalize_mono_geometry_private_rng_and_disconnection(self):
        t=self.t; c=self.capsule; x=t.load(self.root/'graph.pt',weights_only=True)['x']; normalized=(x-x.min()); normalized=normalized/normalized.sum(-1,keepdim=True).clamp_min(1)
        self.assertTrue(t.equal(c['tokens'][0],normalized)); self.assertEqual(len(c['tokens']),3); geometry=c['geometry']; self.assertEqual(len(geometry['anchors'].unique()),64)
        expected=(t.arange(72)[:,None]-geometry['anchors'][None]).abs().clamp_max(16).to(t.uint8)
        self.assertTrue(t.equal(geometry['distance'],expected)); self.assertTrue(t.equal(geometry['bias'],expected.float().div(-4)))
        edges=geometry['support']; self.assertFalse(bool((edges[0]==edges[1]).any())); self.assertEqual(edges.shape,(2,142))
        graph=t.load(self.root/'graph.pt',weights_only=True); graph['edge_index']=t.tensor([(i,i) for i in range(72)]+[(0,1),(1,0),(0,1)],dtype=t.long).T
        path=self.root/'disconnected.pt'; t.save(graph,path); outside=t.get_rng_state().clone(); new=prepare_graph_capsule(t,self.root,receipt(self.root,path),self.bindings,10103)
        self.assertTrue(t.equal(outside,t.get_rng_state())); self.assertTrue(t.equal(new['geometry']['anchors'],geometry['anchors']))
        for column,anchor in enumerate(new['geometry']['anchors'].tolist()):
            self.assertEqual(int(new['geometry']['distance'][anchor,column]),0)
            if anchor>1: self.assertEqual(int((new['geometry']['distance'][:,column]==16).sum()),71)
        self.assertEqual(new['geometry']['support'].shape,(2,2)); self.assertFalse(new['labels_or_predictions_used'])

    def test_paid_selected_H_U_zero_readout_and_delayed_gradients(self):
        parent,source,result=self.source(); t=self.t; source.model.eval(); expected=source.model()
        for arm in ('B1','B4','T16','S64','FS64','LM4'):
            adapter=self.adapter(self.run_config(arm,parent=parent['run_id'])); adapter.model.eval(); body=adapter.model.memory_body
            self.assertTrue(t.allclose(adapter.model(),expected.expand(adapter.model.eval_views,-1,-1),atol=1e-5,rtol=1e-4)); self.assertFalse(adapter.optimizer.state)
            self.assertEqual(adapter.custody['parent']['checkpoint']['sha256'],result['selected_checkpoint']['sha256']); self.assertFalse(any('fixed_H' in k or 'fixed_U' in k for k in adapter.model.state_dict()))
            self.assertTrue(t.equal(body.fixed_U,body.fixed_H/t.linalg.vector_norm(body.fixed_H,dim=1,keepdim=True).clamp_min(1)))
            self.assertEqual({n for n,p in body.native.named_parameters() if p.requires_grad},{'lin3.weight','lin3.bias','lin3.r','lin3.s'})
            self.assertFalse(body.native.training); self.assertFalse(adapter.initial_diagnostic['checkpoint_eligible'])
        lm=self.adapter(self.run_config('LM4',parent=parent['run_id'])); metrics=lm.train_step(1); memory=lm.model.memory_body.memory
        self.assertGreater(float(memory.maps['Wo'].weight.grad.abs().sum()),0)
        self.assertEqual(float(memory.E.grad.abs().sum()),0)
        for name in ('Aq','Ak','Av','Bq','Bk','Bv'): self.assertEqual(float(memory.maps[name].weight.grad.abs().sum()),0)
        lm.train_step(2); self.assertTrue(any(float(memory.maps[name].weight.grad.abs().sum())>0 for name in ('Aq','Ak','Av','Bq','Bk','Bv')))
        self.assertEqual(metrics['operation_counts']['backward_calls'],4); self.assertEqual(metrics['operation_counts']['landmark_common_map_products'],3)
        self.assertFalse(any(p.grad is not None for p in lm.model.memory_body.native.parameters() if not p.requires_grad))
        # Explicit high-norm cache proves H is not silently normalized/clipped.
        p,_=isolated_helpers(self.bindings); core=compile_core(self.root,self.bindings['core'],nodes=72,features=5,classes=3,seed=10103,dropout=.5)
        H=t.ones(72,256)*2; U=H/t.linalg.vector_norm(H,dim=1,keepdim=True); geometry=self.capsule['geometry']
        model=core.build_arm(t,t.nn.functional,'S64',10103,[source.model.native_bank.body],[{'H':H,'U':U}],geometry,FactorLinear=p.factor_definitions(t,self.root)['FactorLinear'])['models'][0]
        hidden=model.memory(H,U,geometry['anchors'],geometry['bias'],0,model.bank_constants()); self.assertTrue(t.equal(hidden,H)); self.assertGreater(float(t.linalg.vector_norm(hidden,dim=1).min()),1)
        self.registry.rows[parent['run_id']]['result']['selected_checkpoint']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'byte authority'): self.adapter(self.run_config('S64',parent=parent['run_id']))

    def test_shared_projection_tape_equals_eager_mean_view_CE(self):
        parent,_,_=self.source(); t=self.t
        for arm in ('T16','LM4'):
            left=self.adapter(self.run_config(arm,parent=parent['run_id'])); right=self.adapter(self.run_config(arm,parent=parent['run_id']))
            with t.no_grad():
                weight=t.linspace(-.03,.03,8192).reshape(256,32)
                for a in (left,right): a.model.memory_body.memory.maps['Wo'].weight.copy_(weight)
            right.model.train(); right.optimizer.zero_grad(set_to_none=True); constants=right.model.constants(); logits=t.stack([right.model.view(v,constants) for v in range(4)])
            eager=t.stack([t.nn.functional.cross_entropy(z[right.train_ids],right.train_y) for z in logits]).mean(); eager.backward()
            metrics=left.train_step(1); self.assertAlmostEqual(metrics['ce'],float(eager),places=6)
            for (name,a),(other,b) in zip(left.model.named_parameters(),right.model.named_parameters()):
                self.assertEqual(name,other)
                if a.requires_grad: self.assertIsNotNone(a.grad); self.assertTrue(t.allclose(a.grad,b.grad,atol=2e-6,rtol=2e-5),name)
            self.assertEqual(metrics['operation_counts']['landmark_common_projected_banks'],1)

    def test_FS64_exact_unit_factors_private_offsets_and_immutable_restore(self):
        parent,_,_=self.source(); t=self.t; adapters={a:self.adapter(self.run_config(a,parent=parent['run_id'])) for a in ('S64','FS64','T16','LM4')}
        with t.no_grad():
            for a in adapters.values(): a.model.memory_body.memory.maps['Wo'].weight.copy_(t.linspace(-.04,.04,8192).reshape(256,32)); a.model.eval()
        self.assertTrue(t.allclose(adapters['S64'].model(),adapters['FS64'].model(),atol=1e-6,rtol=1e-6))
        fs=adapters['FS64']; self.assertEqual(sum(v.numel() for name,v in fs.model.memory_body.memory.named_parameters() if name.endswith(('.r','.s'))),1568)
        self.assertEqual(adapters['T16'].model.memory_body.memory.E.shape,(16,32)); self.assertEqual(adapters['LM4'].model.memory_body.memory.E.shape,(4,16,32))
        lm=adapters['LM4']; lm.model.train(); z=lm.model.view(0,lm.model.constants()); lm.F.cross_entropy(z[lm.train_ids],lm.train_y).backward(); grad=lm.model.memory_body.memory.E.grad
        self.assertGreater(float(grad[0].abs().sum()),0); self.assertEqual(float(grad[1:].abs().sum()),0)
        payload=capture_checkpoint(t,fs,fit_id='fs',step=1,validation={'accuracy':0.},decision={'save':True}); path=self.root/'fs.pt'; t.save(payload,path); restore_checkpoint(t,path,fs,expected_fit_id='fs')
        self.assertEqual(fs.custody['H_hash'],tensor_hash(fs.model.memory_body.fixed_H))
        with t.no_grad(): fs.model.memory_body.fixed_H.add_(.01)
        with self.assertRaisesRegex(ValueError,'mutated'): fs.model()

    def test_actual_four_paid_own_one_view_export_and_native_continuation(self):
        fits=[]; adapters=[]
        for body in range(4):
            parent,_,_=self.source(body); run=self.run_config('S64',body,parent['run_id']); adapter,_=self.fit_synthetic(run); fits.append(run['run_id']); adapters.append(adapter)
        self.assertEqual(inspect_independence(adapters)['independent_adapters'],4)
        ids={}
        for role,values in [('cal',[60,61,62]),('evaluate',[63,64,65])]:
            path=self.root/(role+'_ids.pt'); self.t.save(self.t.tensor(values),path); ids[role+'_ids']=str(path.relative_to(self.root))
        out=self.root/'export'; out.mkdir(); run={'parents':fits,'config':{'fit_parents':fits,'endpoint_inputs':ids,'fusion':'mean_probabilities','landmark_memory_I4':True}}
        exported=export_bank(project_root=self.root,study=self.study,run=run,output_root=self.context['output_root'],output_dir=out,registry=self.registry,device='cpu')
        self.assertEqual(exported['served_views'],4); self.assertFalse(exported['CAL_targets_read']); bank=self.t.load(out/'cal_logits.pt',weights_only=True)
        self.assertEqual(bank['logits'].shape,(4,3,3)); self.assertEqual(bank['view_body_ids'].tolist(),[0,1,2,3]); self.assertEqual(len({a['actual_paid_parent']['fit_id'] for a in exported['ancestry']}),4)
        factor=self.adapter(self.run_config('continued_f1',parent='factor_m1-b0')); self.assertTrue(all(p.requires_grad for p in factor.model.parameters())); self.assertFalse(factor.optimizer.state)
        factor.model.train(); self.assertEqual(factor.model().shape,(4,72,3)); factor.model.eval(); self.assertEqual(factor.model().shape,(1,72,3))
        ordinary,_,_=self.source(0,'ordinary_m1'); native=self.adapter(self.run_config('continued_m1',parent=ordinary['run_id'])); self.assertEqual(native.model.native_bank.factor_sites,0)

    def test_finite_roster_paid_sources_views_and_all_reference_cohorts(self):
        totals={}; studies=[]
        for dataset,size in [('wikics',20),('amazon-photo',10)]:
            files=[]; rows=[]
            for split in range(size):
                node={'graph':'graph.pt','train':f'train{split}.pt','select':f'select{split}.pt'}; end={k:f'{k}{split}.pt' for k in ('cal_ids','evaluate_ids','cal_targets','evaluate_targets')}
                rows.append({'split_id':split,'node_inputs':node,'endpoint_inputs':end,'evaluate_purpose':'development'})
                files += [{'path':path,'bytes':1,'sha256':'a'*64} for path in list(node.values())+list(end.values())]
            custody={'dataset_id':dataset,'splits':rows,'files':list({v['path']:v for v in files}.values()),'status':'prepared','benchmark_fits':0,'predictive_scores':0,'model_graph':{'classes':3},'request':{'graph_policy':'unchanged','evaluation_exposure':'development'},'request_hash':'synthetic'}
            graph_custody={'dataset_id':dataset,'capsules':[{'seed':seed,'capsule':{'path':'geometry.pt','bytes':1,'sha256':'b'*64},'payload_hash':'c'*64} for seed in SEARCH_SEEDS+COMPARATIVE_SEEDS]}
            study,counts,banks=make_study(self.root,custody,self.root,graph_custody,self.study['source'],self.bindings,{'*':{'reservation_mib':1,'wall_seconds':10000,'estimated_seconds_per_update':1,'evidence':'synthetic'}},'runtime'); studies.append(study)
            for key,value in counts.items():
                if type(value)==int: totals[key]=totals.get(key,0)+value
            self.assertEqual(len(bank_roster(study)),size*3*18)
        graph_check(studies,{}); paid=typed_counts(studies); self.assertEqual(paid['model_optimizer_step_call_total'],15700000)
        for key,value in {'atomic_fit_groups':3140,'physical_body_instances':3140,'source_acquisition_fits':1052,'continuations':2088,'search_selection_jobs':24,'comparative_banks':1620,'genuine_I4_banks':540,'landmark_memory_I4_banks':180,'all_run_records':8024,'full_budget_optimizer_updates':15700000}.items(): self.assertEqual(totals[key],value)


if __name__=='__main__': unittest.main()

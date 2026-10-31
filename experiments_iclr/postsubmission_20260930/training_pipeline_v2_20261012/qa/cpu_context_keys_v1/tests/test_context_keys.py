"""Exact-source synthetic objective/ownership/RNG/restore and finite metadata."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from adapters.context_keys import ARMS,REFERENCES,SOURCE_PINS,MODES,make_adapter,relocate_objective,source_module,tensor_hash,rng_state
from compile_context_keys_matrix import make_study
from compile_master_plan import bank_roster,reference_matches,typed_counts
from engine import capture_checkpoint,restore_checkpoint,inspect_independence,run_trajectory

try:
    import numpy as np
    import torch
except ImportError:
    np=torch=None

PACKAGE=Path(__file__).resolve().parents[1]
SOURCE_PATHS={'native':'common_wrapper_native_source_20261010_v1/models.py',
    'routes':'shared_fast_graph_model_interface_20261010_v1/common_routes.py',
    'factors':'portable_internal_be_public_interface_20261007_v2/core/factors.py',
    'objective':'native_SAGE_fixed_context_key_source_20261011_v1/fixed_context_key_objective.py',
    'driver':'native_SAGE_fixed_context_key_source_20261011_v1/run_family.py'}


def fixture_source(name):
    staged=PACKAGE/'fixtures/context_keys'/(name+'.py')
    return staged if staged.exists() else PACKAGE.parent/SOURCE_PATHS[name]


def receipt(root,path):
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def custody():
    files=[];splits=[]
    def file(path):files.append({'path':path,'bytes':1,'sha256':'0'*64});return path
    graph=file('graph.pt')
    for split in range(20):
        node={'graph':graph,'train':file('sp'+str(split)+'/train.pt'),'select':file('sp'+str(split)+'/select.pt')}
        endpoint={name:file('sp'+str(split)+'/'+name+'.pt') for name in ('cal_ids','cal_targets','evaluate_ids','evaluate_targets')}
        splits.append({'split_id':split,'node_inputs':node,'endpoint_inputs':endpoint,'role_counts':{'train':8,'select':4,'cal':4,'evaluate':4},
            'role_class_counts':{'train':{str(c):2 for c in range(4)}}})
    return {'dataset_id':'wikics','model_graph':{'nodes':20,'features':140,'classes':4},'splits':splits,'files':files,
        'request':{'graph_policy':'synthetic metadata','evaluation_exposure':'development synthetic'},'request_hash':'0'*64}


class ContextKeyMetadataTests(unittest.TestCase):
    def test_all_frozen_sources_match_and_complete_exact_finite_counts(self):
        for name,pin in SOURCE_PINS.items():self.assertEqual(hashlib.sha256(fixture_source(name).read_bytes()).hexdigest(),pin)
        sources={name:{'path':'source/'+name+'.py','bytes':1,'sha256':pin} for name,pin in SOURCE_PINS.items()}
        spec,banks,counts=make_study(custody(),'data',{'git_commit':'synthetic','files':list(sources.values())},sources,'runtime',
            {'reservation_mib':32768,'estimated_seconds_per_update':1,'wall_seconds':250000000})
        self.assertEqual((counts['search_model_fits'],counts['comparative_model_fits'],counts['model_fits']),(270,1500,1770))
        self.assertEqual((counts['registered_optimizer_updates'],counts['native_factual_forwards'],counts['query_maps']),(8850000,17040000,1176))
        self.assertEqual((counts['search_selection_jobs'],counts['bank_exports'],counts['scalar_CAL_fit_jobs'],counts['evaluation_jobs'],counts['run_records']),(12,900,900,900,4482))
        self.assertTrue(all(r['budget']['maximum']==5000 for r in spec['runs'] if r['kind']=='model_fit'))
        self.assertEqual(len(bank_roster(spec)),900)
        self.assertTrue(all(r['matched'] for r in reference_matches(spec)))
        self.assertEqual(typed_counts([spec])['model_optimizer_step_call_total'],8850000)
        self.assertFalse(spec['release']['scientific_launch_enabled'])


@unittest.skipIf(torch is None,'Allocation CPU numerical fixture required')
class ContextKeyNumericalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.sources={}
        for name in SOURCE_PINS:
            path=self.root/'sources'/(name+'.py');path.parent.mkdir(exist_ok=True);shutil.copyfile(fixture_source(name),path)
            self.sources[name]=receipt(self.root,path)
        generator=torch.Generator().manual_seed(981)
        x=torch.randn(12,140,generator=generator);ids=torch.tensor([7,0,5,2,6,1,4,3]);labels=torch.tensor([3,0,1,2,2,1,0,3])
        edges=torch.stack([torch.arange(12).repeat(2),torch.cat([torch.arange(12),torch.arange(12).roll(1)])])
        self.inputs={}
        for name,value in {'graph':{'x':x,'edge_index':edges},'train':{'ids':ids,'y':labels},'select':{'ids':torch.tensor([8,9,10,11]),'y':torch.tensor([0,1,2,3])}}.items():
            path=self.root/(name+'.pt');torch.save(value,path);self.inputs[name]=receipt(self.root,path)
        self.study={'stage':'engineering','dataset':'synthetic-context-keys','data':{'roles':{name:{'0':self.inputs[name]} for name in ('train','select')}},
            'source':{'files':list(self.sources.values())},'runs':[]}

    def tearDown(self):self.temp.cleanup()

    def row(self,arm='shared4_key_private',body=0):
        members=4 if arm.startswith('shared4_') or arm=='reference_original_GNNM' else 1
        return {'run_id':arm+'-'+str(body),'method':arm,'split':0,'seed':71201,'body':body,'kind':'model_fit',
            'config':{'context_key_arm':arm,'backbone':'SAGE','target_law':'multiclass_ce','classes':4,'width':128,'depth':2,'weight_decay':0,
                'context_coefficient':.05,'learning_rate':.001,'dropout':.2,'members':members,'node_inputs':self.inputs,
                'context_key_sources':self.sources,'paired_seed':71201,'initialization_seed':71201+1000003*body},
            'budget':{'unit':'optimizer_updates','policy':'full_budget','maximum':3},
            'selection':{'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'save_every':1}}

    def adapter(self,arm='shared4_key_private',body=0):return make_adapter(torch=torch,project_root=self.root,study=self.study,run=self.row(arm,body),device='cpu')

    def test_every_arm_native_parameters_query_ownership_and_row_order(self):
        for arm in ARMS+REFERENCES:
            with self.subTest(arm=arm):
                adapter=self.adapter(arm);result=adapter.train_step(1)
                self.assertEqual(result['operation_counts']['optimizer_updates'],1)
                self.assertEqual(tuple(adapter.model().shape),(adapter.model.members,12,4))
                owned=[id(p) for group in adapter.optimizer.param_groups for p in group['params']]
                self.assertEqual(len(owned),len(set(owned)));self.assertEqual(set(owned),{id(p) for p in adapter.model.parameters()})
                self.assertEqual(len(adapter.optimizer.param_groups),2 if arm in ARMS else 1)
                if arm in ARMS:
                    self.assertTrue(all(not value.requires_grad for value in adapter.targets.values()))
                    self.assertEqual(adapter.model.query_map.weight.shape,(140,128))
                    self.assertIsNotNone(adapter.model.query_map.weight.grad)
                    self.assertTrue(torch.equal(adapter.targets['train_ids'],adapter.train_ids))
                expected=torch.tensor([7,0,5,2,6,1,4,3]) if adapter.model.members==4 else torch.arange(8)
                self.assertTrue(torch.equal(adapter.train_ids,expected))

    def test_source_dimension_equations_and_gradients_equal_unchanged_source(self):
        source=source_module(self.root,self.sources['objective'],'objective')
        adapted=relocate_objective(source,580,300,128);generator=torch.Generator().manual_seed(602)
        x=torch.randn(580,300,generator=generator).numpy();edges=np.stack([np.arange(580),np.arange(580)])
        arrays,_=source.prepare_targets(np,x,edges,np.arange(580,dtype=np.int64),np.arange(580,dtype=np.int64)%10)
        target=source.tensor_bank(torch,arrays,np.arange(580,dtype=np.int64),'cpu')
        q1=source.make_query_map(torch,torch.nn.functional,71,'cpu');q2=adapted.make_query_map(torch,torch.nn.functional,71,'cpu')
        self.assertTrue(torch.equal(q1.weight,q2.weight))
        for mode in ('key_private','key_mixture','key_uniform','projected_peer_private','key_moment_private','key_shuffled_private'):
            a=torch.randn(4,580,128,generator=generator,requires_grad=True);b=a.detach().clone().requires_grad_()
            first=source.context_auxiliary(torch,torch.nn.functional,a,target,mode,query_map=q1)[0]
            second=adapted.context_auxiliary(torch,torch.nn.functional,b,target,mode,query_map=q2)[0]
            self.assertTrue(torch.equal(first,second));g1=torch.autograd.grad(first,(a,q1.weight));g2=torch.autograd.grad(second,(b,q2.weight))
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(g1,g2)))

    def test_auxiliary_gradient_permissions_and_both_live_peer_sides(self):
        adapter=self.adapter('shared4_projected_peer_private');adapter.model.train();adapter.model()
        hidden=adapter.model.last_hidden[:,adapter.train_ids];qmap=adapter.model.query_map;target=adapter.targets
        loss=adapter.objective.context_auxiliary(torch,torch.nn.functional,hidden,target,'projected_peer_private',query_map=qmap)[0]
        queries=torch.nn.functional.normalize(qmap(hidden),dim=-1,eps=1e-12)
        scores=torch.bmm(queries,queries.transpose(1,2))/.2
        manual=(torch.logsumexp(scores.masked_fill(~target['full_allowed'][None],float('-inf')),-1)-(target['weights']*scores).sum(-1)).mean()
        live_grad=torch.autograd.grad(loss,(hidden,qmap.weight),retain_graph=True);manual_grad=torch.autograd.grad(manual,(hidden,qmap.weight),retain_graph=True)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(live_grad,manual_grad)))
        stopped=queries@queries.detach().transpose(1,2)/.2
        stopped_loss=(torch.logsumexp(stopped.masked_fill(~target['full_allowed'][None],float('-inf')),-1)-(target['weights']*stopped).sum(-1)).mean()
        self.assertFalse(torch.allclose(live_grad[0],torch.autograd.grad(stopped_loss,hidden,retain_graph=True)[0]))
        adapter.optimizer.zero_grad(set_to_none=True);loss.backward()
        self.assertIsNone(adapter.model.body.body.output_linear.weight.grad)
        self.assertIsNotNone(adapter.model.body.body.input_linear.r.grad)
        self.assertIsNotNone(adapter.model.body.body.residual_modules[0].module.conv.lin_l.weight.grad)

    def test_shuffle_moves_only_Q_axes_and_float32_moments_are_not_normalized(self):
        adapter=self.adapter();graph=torch.load(self.root/'graph.pt',weights_only=True)
        arrays,_=adapter.objective.prepare_targets(np,graph['x'].numpy(),graph['edge_index'].numpy(),np.arange(8,dtype=np.int64),np.arange(8,dtype=np.int64)%2)
        t=adapter.objective.tensor_bank(torch,arrays,np.arange(8,dtype=np.int64),'cpu');p=t['shuffle_permutation']
        self.assertTrue(torch.equal(t['shuffled_weights'],t['weights'][:,p[:,None],p[None,:]]))
        self.assertTrue(torch.equal(t['shuffled_moments'],torch.bmm(t['shuffled_weights'],t['keys'])))
        self.assertFalse(torch.allclose(t['moments'].norm(dim=-1),torch.ones(4,8)))
        self.assertTrue(torch.equal(t['full_allowed'],t['train_ids'][:,None]!=t['train_ids'][None,:]))

    def test_private_RNG_disjoint_I4_and_same_query_initial_bytes(self):
        outer=rng_state(torch,np);a=self.adapter('genuinefactorizedI4_key_assigned',0);b=self.adapter('genuinefactorizedI4_key_assigned',1)
        self.assertEqual(inspect_independence([a,b])['independent_adapters'],2)
        self.assertTrue(torch.equal(a.model.query_map.weight,b.model.query_map.weight))
        self.assertNotEqual(a.model.query_map.weight.untyped_storage().data_ptr(),b.model.query_map.weight.untyped_storage().data_ptr())
        self.assertEqual((a.context_index,b.context_index),(0,1));a.train_step(1)
        self.assertTrue(torch.equal(torch.get_rng_state(),outer['torch_cpu']))
        self.assertEqual(rng_state(torch,np)['numpy'],outer['numpy'])

    def test_full_synthetic_trajectory_strict_restore_replay_and_target_custody(self):
        row=self.row();a=self.adapter();result=run_trajectory(torch,a,row,self.root/'fit')
        self.assertEqual(result['completed_steps'],3)
        restored=self.adapter();path=self.root/'fit'/result['selected']['path']
        restore_checkpoint(torch,path,restored,expected_fit_id=row['run_id'])
        before=restored.model.query_map.weight.detach().clone();restored.train_step(4);self.assertFalse(torch.equal(before,restored.model.query_map.weight))
        restore_checkpoint(torch,path,restored,expected_fit_id=row['run_id'])
        restored.targets['keys'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'bytes changed'):restore_checkpoint(torch,path,restored,expected_fit_id=row['run_id'])


if __name__=='__main__':unittest.main()

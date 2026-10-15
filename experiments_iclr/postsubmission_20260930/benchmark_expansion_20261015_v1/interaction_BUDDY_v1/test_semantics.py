"""Interaction algebra and ingredient mechanics; no native source model or fit."""
import copy
import hashlib
import importlib.util
from importlib import import_module
from pathlib import Path
import sys
import unittest
import torch

HERE = Path(__file__).resolve().parent
def load(path,package=False):
    path = Path(path); name = '_interaction_test_'+hashlib.sha256(str(path).encode()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,path,**({'submodule_search_locations':[str(path.parent)]} if package else {}))
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return sys.modules[name]
own = load(HERE/'objectives.py'); link = load(HERE.parent/'link/__init__.py',True)
resp = load(HERE.parent/'structured_responsibility_control_v2/__init__.py',True)
responsibility = import_module(resp.__name__+'.objectives'); sampler = import_module(resp.__name__+'.sampler')
aggregation = load(HERE.parent/'conditional_aggregation_v1/core.py')
portable = load(HERE.parent/'portable_private_input_v2/core.py')


class SemanticTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(73)
        shared = torch.randn(6,1,dtype=torch.float64,requires_grad=True)
        private = torch.randn(4,6,dtype=torch.float64,requires_grad=True)
        preheads = torch.randn(4,6,5,dtype=torch.float64,requires_grad=True)
        logits = [shared[:,0]+private[m] for m in range(4)]
        groups = [1,1,1,1,0,0]
        endpoints = [(0,1),(1,2),(2,3),(0,4),(1,4),(2,4)]
        planning = sys.modules[link.__name__+'.planning']; keys = planning.query_keys(endpoints)
        signature = planning.authenticate(keys,groups,'TRAIN')
        aligned = dict(preheads=preheads,endpoints=endpoints,relation_ids=[0]*6,groups=groups,route_query_signatures=[signature]*4)
        rows = [[{'keys':keys,'groups':groups}] for _ in range(4)]
        labels=torch.tensor(groups,dtype=torch.float64)
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(z,labels) for z in logits]
        mean = torch.stack(logits).mean(0)
        pool = torch.nn.functional.binary_cross_entropy_with_logits(mean,labels)
        kw = dict(link=link,responsibility=responsibility,sampler=sampler,aggregation=aggregation,
            aligned=aligned,raw_logits=logits,native_losses=losses,native_pool=pool,seed=11,actual_route_rows=rows)
        return shared,private,preheads,kw

    def test_I1_exact_original_private_contrast_credit(self):
        shared,private,hidden,kw = self.fixture()
        actual = own.build(torch,method='interaction_geometry_input',gate=None,**kw)
        expected = link.build_objectives(torch,family='BUDDY',logits=torch.stack(kw['raw_logits']),
            native_member_losses=kw['native_losses'],native_pool_loss=kw['native_pool'],seed=11,
            contrast='endpoint',responsibility='none',aux_credit='private',coefficient=.05,
            temperature=.2,anchor_limit=1024,negative_limit=16,role='TRAIN',**kw['aligned'])
        torch.testing.assert_close(actual['shared_objective'],expected['shared_objective'],atol=0,rtol=0)
        torch.testing.assert_close(actual['private_objective'],expected['private_objective'],atol=0,rtol=0)
        self.assertFalse(actual['scalar_update'])
        self.assertIsNone(torch.autograd.grad(actual['shared_objective'],hidden,retain_graph=True,allow_unused=True)[0])
        self.assertGreater(float(torch.autograd.grad(actual['private_objective'],hidden)[0].norm()),0)

    def test_I2_exact_corrected_endpoint_v2_and_private_pool(self):
        shared,private,hidden,kw = self.fixture()
        value = own.build(torch,method='interaction_responsibility_private_pool',gate=None,**kw)
        keys = kw['actual_route_rows'][0][0]['keys']
        batch = sampler.make_query_batch([k[:2] for k in keys],scope=own.QUERY_SCOPE,directed=False)
        weights = torch.tensor(sampler.raw_responsibilities(batch,11,mode='endpoint',floor=.25),dtype=torch.float64)
        records = responsibility.native_record_losses(torch,torch.stack(kw['raw_logits']),kw['aligned']['groups'],'BUDDY')
        W = responsibility.reduce_native(torch,records,kw['aligned']['groups'],'BUDDY',weights).mean()
        torch.testing.assert_close(value['shared_objective'],torch.stack(kw['native_losses']).mean(),atol=0,rtol=0)
        torch.testing.assert_close(value['private_objective'],.5*W+.5*kw['native_pool'],atol=0,rtol=0)
        self.assertFalse(value['scalar_update']); self.assertEqual(value['interaction_diagnostics']['responsibility']['hash'],'BLAKE2b64/link-resp-v2/52bit-midpoint')
        self.assertIsNone(torch.autograd.grad(value['private_objective'],hidden,allow_unused=True)[0])

    def test_I2_rejects_changed_actual_route_order(self):
        _,_,_,kw = self.fixture(); kw['actual_route_rows'] = copy.deepcopy(kw['actual_route_rows'])
        kw['actual_route_rows'][3][0]['keys'] = list(reversed(kw['actual_route_rows'][3][0]['keys']))
        with self.assertRaises(ValueError):
            own.build(torch,method='interaction_responsibility_private_pool',gate=None,**kw)

    def test_I3_native_mean_J_exact_and_one_scalar(self):
        _,_,_,kw = self.fixture()
        gate = aggregation.make_aggregator(torch,mode='mean',task_kind='binary',seed=11,reference=kw['raw_logits'][0])
        value = own.build(torch,method='interaction_input_meanJ',gate=gate,**kw)
        expected = .5*torch.stack(kw['native_losses']).mean()+.5*kw['native_pool']
        torch.testing.assert_close(value['shared_objective'],expected,atol=0,rtol=0)
        self.assertIs(value['shared_objective'],value['private_objective']); self.assertTrue(value['scalar_update'])
        self.assertEqual(sum(p.numel() for p in gate.parameters()),0)

    def test_I4_initial_mean_and_live_readout_gradient(self):
        _,_,_,kw = self.fixture()
        gate = aggregation.make_aggregator(torch,mode='conditional',task_kind='binary',seed=11,reference=kw['raw_logits'][0])
        value = own.build(torch,method='interaction_input_conditionalJ',gate=gate,**kw)
        expected = .5*torch.stack(kw['native_losses']).mean()+.5*kw['native_pool']
        torch.testing.assert_close(value['shared_objective'],expected,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in gate.parameters()),20)
        grads = torch.autograd.grad(value['shared_objective'],tuple(gate.parameters()))
        self.assertGreater(sum(float(g.norm()) for g in grads),0)
        weights = gate(kw['raw_logits'])['weights']
        self.assertTrue((weights>=.05).all()); torch.testing.assert_close(weights.sum(0),torch.ones(6,dtype=torch.float64))

    def test_two_BUDDY_input_branches_rank8_zero_initial_and_RNG_isolated(self):
        torch.manual_seed(91);state=torch.get_rng_state().clone();parameters=0
        for index,width in enumerate((16,1024)):
            anchor=torch.zeros(width,16 if index==0 else 128,dtype=torch.float64)
            residual=portable.build_residual(torch,anchor,width=width,seed=11+901003+104729*index)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            parameters+=sum(p.numel() for p in residual.parameters())
            expected='separate_per_route_orthonormal_frames_with_unavoidable_overlap' if width==16 else 'one_disjoint_orthonormal_frame'
            self.assertEqual(residual.metadata()['actual_frame'],expected)
            x=torch.arange(2*width,dtype=torch.float64).reshape(2,width)/(2*width)
            for member in range(4):self.assertTrue(torch.equal(residual(x,member),x))
            self.assertEqual(residual.metadata()['rank'],8)
        self.assertEqual(parameters,66560)

    def test_no_source_loss_renormalization_or_drop_in_I2(self):
        _,_,_,kw = self.fixture()
        result = own.build(torch,method='interaction_responsibility_private_pool',gate=None,**kw)
        d = result['interaction_diagnostics']['responsibility']
        self.assertEqual(d['query_count'],6); self.assertEqual(d['native_positive_count'],4); self.assertEqual(d['native_negative_count'],2)
        self.assertEqual(d['native_reduction'],'joint batch mean')
        self.assertFalse(d['unbiased_original_risk_claim']); self.assertFalse(d['normalized_marginals_matched'])

    def test_caller_source_loss_mismatch_rejected(self):
        _,_,_,kw = self.fixture(); kw['native_losses'][0] = kw['native_losses'][0]+1
        with self.assertRaises(ValueError):
            own.build(torch,method='interaction_input_meanJ',gate=None,**kw)


if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()

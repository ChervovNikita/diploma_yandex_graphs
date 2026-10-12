"""Narrow synthetic CPU tests of the actual pinned PolyFormer source bank.

No optimizer action, actual-data tensor, CUDA kernel, fit or prediction score.
"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from adapters.polyformer import make_adapter
from adapters.route_recompute import route_recompute

def tensor_digest(torch,value):
    def visit(v):
        if torch.is_tensor(v):return str((str(v.dtype),tuple(v.shape))).encode()+v.detach().cpu().contiguous().numpy().tobytes()
        if isinstance(v,dict):return b''.join(str(k).encode()+visit(v[k]) for k in sorted(v))
        if isinstance(v,(list,tuple)):return b''.join(visit(x) for x in v)
        return repr(v).encode()
    return hashlib.sha256(visit(value)).hexdigest()

class RouteRecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch;torch.set_num_threads(2)
        cls.root=Path(os.environ['GNNM_REFERENCE_PROJECT']).resolve()
        cls.native={'K':2,'hidden':16,'nlayer':2,'n_head':4,'d_ffn':8,'q':1.6,'multi':2.,'dropout':.35,'dprate':.4}
        cls.states=[]

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=self.root);self.folder=Path(self.temp.name);t=self.t
        graph={'x':t.arange(12*11,dtype=t.float32).reshape(12,11)/100,
          'edge_index':t.tensor([[i for i in range(12)]+[i for i in range(12)],[(i+1)%12 for i in range(12)]+[i for i in range(12)]])}
        payloads={'graph':graph,'train':{'ids':t.arange(9),'y':t.arange(9)%3},'select':{'ids':t.arange(9,12),'y':t.arange(9,12)%3}}
        self.inputs={}
        for key,value in payloads.items():
            p=self.folder/(key+'.pt');t.save(value,p);self.inputs[key]=str(p.relative_to(self.root))

    def tearDown(self):self.temp.cleanup()

    def adapter(self,paired=True):
        loss={'variant':'two_view_class_full_cross_view','coefficients':{'alignment_weight':.05,'temperature':.2}} if paired else \
          {'variant':'gncl_within_route_supcon','coefficients':{'pool_weight':.5,'supcon_weight':.05,'temperature':.2}}
        config={'polyformer_mode':'polyformer_internal_be','fusion':'mean_probabilities','node_inputs':self.inputs,
          'native':dict(self.native),'preprocessing':'pinned_mono_base','train_views':2 if paired else 1,'loss':loss,
          'dropout_seeds':[81231,81233,81237,81241],'classes':3,'initialization_seed':82101,
          'learning_rate':.005,'attn_lr':.0005,'weight_decay':0.,'attn_weight_decay':1e-4,'factor_initialization':'unit'}
        adapter=make_adapter(torch=self.t,project_root=self.root,study={'dataset':'synthetic_route_recompute','data':{},'source':{}},
          run={'run_id':'fixture','method':'paired' if paired else 'gncl_supcon','config':config,
            'budget':{'maximum':5000},'selection':{'metric':'accuracy','direction':'max','ties':'first','patience_steps':5000}},device='cpu')
        # Manual resident source-optimizer values, with no initialization/update
        # kernel. The utility must not touch these or any learnable parameter.
        for index,p in enumerate(adapter.model.parameters()):
            adapter.optimizer.state[p]={'step':self.t.tensor(3.),'exp_avg':self.t.full_like(p,.01+index*.0001),'exp_avg_sq':self.t.full_like(p,.02)}
        return adapter

    def objective(self,adapter):
        bank=adapter.model;bank.train();logits=adapter.logits_fn(bank,True,1);hidden=bank.last_hidden
        terms=adapter.objective_callback(adapter,1,logits,hidden)
        return terms['objective'],(logits.detach().clone(),hidden.detach().clone()),terms

    def compare_callback(self,paired):
        t=self.t;eager=self.adapter(paired);wrapped=self.adapter(paired)
        initial=t.get_rng_state().clone();parameters=tensor_digest(t,eager.model.state_dict());moments=tensor_digest(t,eager.optimizer.state_dict())
        loss,outputs,terms=self.objective(eager);loss.backward();eager_runtime=eager.model.capture_runtime_state();eager_global=t.get_rng_state().clone()
        t.set_rng_state(initial)
        with route_recompute(t,wrapped.model) as handle:
            checked,observed,observed_terms=self.objective(wrapped)
            for first,last in zip(outputs,observed):t.testing.assert_close(first,last,rtol=0,atol=0)
            t.testing.assert_close(loss.detach(),checked.detach(),rtol=0,atol=0)
            self.assertEqual(terms['operation_counts'] if 'operation_counts' in terms else None,
              observed_terms['operation_counts'] if 'operation_counts' in observed_terms else None)
            advanced=tensor_digest(t,wrapped.model.capture_runtime_state())
            checked.backward()
            self.assertEqual(advanced,tensor_digest(t,wrapped.model.capture_runtime_state()))
        for (name,p),(name2,q) in zip(eager.model.named_parameters(),wrapped.model.named_parameters()):
            self.assertEqual(name,name2);self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:t.testing.assert_close(p.grad,q.grad,rtol=2e-6,atol=2e-7,msg=name)
        self.assertEqual(tensor_digest(t,eager_runtime),tensor_digest(t,wrapped.model.capture_runtime_state()))
        self.assertTrue(t.equal(eager_global,t.get_rng_state()))
        self.assertEqual(parameters,tensor_digest(t,eager.model.state_dict()));self.assertEqual(parameters,tensor_digest(t,wrapped.model.state_dict()))
        self.assertEqual(moments,tensor_digest(t,eager.optimizer.state_dict()));self.assertEqual(moments,tensor_digest(t,wrapped.optimizer.state_dict()))
        count=8 if paired else 4;report=handle.report()
        self.assertEqual(report['logical_source_route_forwards'],count);self.assertEqual(report['additional_full_source_route_forwards'],count)
        self.assertEqual(report['total_completed_training_route_forwards'],2*count);self.assertEqual(report['recompute_overhead_forward_ratio'],1.)
        self.assertEqual(report['token_bank_copies_created'],0);self.assertFalse(t.cuda.is_initialized())
        self.states.append({'case':'paired_class_full' if paired else 'shared4_gncl_supcon','work':report,
          'outputs_equal':True,'gradients_equal':True,'model_and_optimizer_tensors_unchanged':True,'logical_and_global_RNG_equal':True})

    def test_paired_class_full_four_route_callback(self):self.compare_callback(True)

    def test_shared4_gncl_supcon_callback(self):self.compare_callback(False)

    def test_ordered_repeated_stream_slots_do_not_advance_on_replay(self):
        t=self.t;a=self.adapter();b=self.adapter();a.model.train();b.model.train()
        order=[(2,1),(0,1),(3,0),(1,2),(2,1),(3,3)]
        def run(bank):
            pairs=[bank.forward_route(route,slot=slot) for route,slot in order]
            return sum(z.square().mean()+h.square().mean()*.01 for z,h in pairs),pairs
        loss,first=run(a.model);loss.backward()
        with route_recompute(t,b.model) as handle:
            value,second=run(b.model);logical=tensor_digest(t,b.model.capture_runtime_state());value.backward()
            self.assertEqual(logical,tensor_digest(t,b.model.capture_runtime_state()))
        for x,y in zip(first,second):
            for left,right in zip(x,y):t.testing.assert_close(left,right,rtol=0,atol=0)
        for (name,p),(_,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
            if p.grad is not None:t.testing.assert_close(p.grad,q.grad,rtol=2e-6,atol=2e-7,msg=name)
        self.assertEqual(tensor_digest(t,a.model.capture_runtime_state()),tensor_digest(t,b.model.capture_runtime_state()))
        self.assertEqual([(r['route'],r['slot']) for r in handle.records],order)
        self.assertEqual(handle.report()['additional_full_source_route_forwards'],len(order))

    def test_runtime_restore_repeats_stochastic_branch_and_preserves_caller_eval(self):
        t=self.t;a=self.adapter();b=self.adapter()
        saved=copy.deepcopy(a.capture_state());b.restore_state(copy.deepcopy(saved))
        eager,first,_=self.objective(a);a.model.eval();eager.backward()
        with route_recompute(t,b.model) as handle:
            value,second,_=self.objective(b);b.model.eval();advanced=tensor_digest(t,b.model.capture_runtime_state());value.backward()
            self.assertFalse(b.model.training);self.assertTrue(all(not m.training for m in b.model.modules()))
            self.assertEqual(advanced,tensor_digest(t,b.model.capture_runtime_state()))
        for (name,p),(_,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
            if p.grad is not None:t.testing.assert_close(p.grad,q.grad,rtol=2e-6,atol=2e-7,msg=name)
        for x,y in zip(first,second):t.testing.assert_close(x,y,rtol=0,atol=0)
        a.restore_state(copy.deepcopy(saved));b.restore_state(copy.deepcopy(saved));a.model.zero_grad(set_to_none=True);b.model.zero_grad(set_to_none=True)
        eager,branch,_=self.objective(a);eager.backward()
        with route_recompute(t,b.model):value,repeat,_=self.objective(b);value.backward()
        for x,y in zip(branch,repeat):t.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(tensor_digest(t,a.model.capture_runtime_state()),tensor_digest(t,b.model.capture_runtime_state()))

    def test_evaluation_no_grad_and_disabled_paths_are_eager(self):
        t=self.t;a=self.adapter();b=self.adapter();a.model.eval();b.model.eval()
        runtime=tensor_digest(t,b.model.capture_runtime_state());expected=a.model()
        with route_recompute(t,b.model) as handle:
            observed=b.model();t.testing.assert_close(expected,observed,rtol=0,atol=0)
        self.assertEqual(handle.report()['logical_source_route_forwards'],0);self.assertEqual(handle.report()['additional_full_source_route_forwards'],0)
        self.assertEqual(handle.counts['bypassed_route_calls'],4);self.assertEqual(runtime,tensor_digest(t,b.model.capture_runtime_state()))
        a.model.train();b.model.train()
        with t.no_grad():
            expected=a.model()
            with route_recompute(t,b.model) as handle:observed=b.model()
        t.testing.assert_close(expected,observed,rtol=0,atol=0);self.assertEqual(handle.counts['bypassed_route_calls'],4)
        self.assertEqual(tensor_digest(t,a.model.capture_runtime_state()),tensor_digest(t,b.model.capture_runtime_state()))
        with route_recompute(t,b.model,enabled=False) as handle:_,_,_=self.objective(b)
        self.assertEqual(handle.counts['logical_checkpoint_route_calls'],0);self.assertEqual(handle.counts['bypassed_route_calls'],8)

    def test_recompute_error_restores_logical_stream_modes_and_route_owner(self):
        t=self.t;a=self.adapter();bank=a.model;function=bank.forward_route.__func__
        with self.assertRaisesRegex(RuntimeError,'synthetic replay failure'):
            with route_recompute(t,bank) as handle:
                loss,_,_=self.objective(a);bank.eval();advanced=tensor_digest(t,bank.capture_runtime_state())
                hook=bank.body.lin1.register_forward_hook(lambda *unused:(_ for _ in ()).throw(RuntimeError('synthetic replay failure')))
                try:loss.backward()
                finally:hook.remove()
        self.assertIs(bank.forward_route.__func__,function);self.assertNotIn('forward_route',bank.__dict__)
        self.assertFalse(bank.training);self.assertEqual(advanced,tensor_digest(t,bank.capture_runtime_state()))
        self.assertGreater(handle.counts['failed_route_invocations'],0)

if __name__=='__main__':unittest.main()

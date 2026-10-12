"""New synthetic CPU-only source/place/state checks; no optimizer action."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from adapters.endpoint_value import SOURCES,receipt,render_sources,construct_source,make_adapter

class EndpointSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch;torch.set_num_threads(2)
        cls.root=Path(os.environ['GNNM_REFERENCE_PROJECT']).resolve()
        cls.bindings=json.loads(Path(os.environ['GNNM_ENDPOINT_SOURCE_BINDINGS']).read_text())
        cls.edges=torch.tensor([[0,1,2,3,4,5,6,0,1,2,3,4,5,6],[0,1,2,3,4,5,6,1,2,3,4,5,6,0]],dtype=torch.long)
        cls.x=torch.arange(35,dtype=torch.float32).reshape(7,5)/35

    def build(self,arm='E4',body=0):
        return construct_source(self.t,self.root,self.bindings,nodes=7,features=5,classes=3,seed=76101,body=body,arm=arm,
          dropout=.2,learning_rate=.001,device='cpu',edge_metadata={'nodes':7,'edges':14,'self_edges':7})

    def test_provider_binding_is_narrow_and_stable(self):
        rendered=render_sources(self.root,self.bindings,7,5,3)
        self.assertEqual(rendered['binding_changes'],['native_input_dim','native_classes','stem_factor_input_dim']+['installer_full_graph_nodes']*4)
        self.assertEqual(set(rendered['source_method_AST_hashes']),{'seeds','native_body','make','scope','training_outputs'})
        self.assertIn('class NodeValueSAGEConv',rendered['installer'])
        self.assertNotIn('11701',rendered['installer'])

    def test_identity_constructor_matches_BASE_and_preserves_native_parameters(self):
        source,model,optimizer,streams,_=self.build('E4');base,bmodel,_,bstreams,_=self.build('BASE')
        e=dict(model.named_parameters());b=dict(bmodel.named_parameters())
        self.assertTrue(set(b).issubset(e))
        for key,value in b.items():self.assertTrue(self.t.equal(value,e[key]),key)
        model.eval();bmodel.eval();batch={'x':self.x,'edge_index':self.edges,'graph':type('G',(),{'edge_index':self.edges})()}
        with self.t.no_grad():
            a=model(batch)[0];z=bmodel(batch)[0]
        self.t.testing.assert_close(a,z,rtol=2e-6,atol=2e-6)
        for a,b in zip(streams,bstreams):self.assertTrue(self.t.equal(a['cpu'],b['cpu']))
        self.assertEqual(sum(m.value_source_builder.r.numel()+m.value_source_builder.s.numel() for m in model.modules() if hasattr(m,'value_source_builder')),2176)
        self.assertFalse(self.t.cuda.is_initialized())

    def test_live_endpoint_and_receiver_controls_preserve_self_VALUE(self):
        source,_,_,_,_=self.build('E4');t=self.t
        source_values=t.arange(4*128,dtype=t.float32).reshape(4,128)/512
        target_values=source_values.flip(0);mask=t.tensor([True,False,True,False])
        left=t.nn.Linear(256,16);right=t.nn.Linear(256,16)
        for mode in ('endpoint','node'):
            moved=source.value_helper.transport_values(t,source_values,target_values,left,right,stalk_dim=4,self_edge=mask,conditioning=mode)
            self.assertTrue(t.equal(moved[mask],source_values[mask]))
            self.assertFalse(t.equal(moved[~mask],source_values[~mask]))
            left.zero_grad(set_to_none=True);right.zero_grad(set_to_none=True);moved.square().mean().backward()
            self.assertTrue(t.isfinite(left.weight.grad).all());self.assertGreater(float(left.weight.grad.abs().sum()),0)

    def test_factory_packed_own_gradient_restore_and_genuine_ownership(self):
        t=self.t
        with tempfile.TemporaryDirectory(dir=self.root) as raw:
            folder=Path(raw)
            payloads={'graph':{'x':self.x,'edge_index':self.edges},'train':{'ids':t.tensor([0,1,2]),'y':t.tensor([0,1,2])},
              'select':{'ids':t.tensor([3,4]),'y':t.tensor([0,1])}}
            inputs={}
            for key,value in payloads.items():
                p=folder/(key+'.pt');t.save(value,p);inputs[key]=receipt(self.root,p)
            def create(arm,body=0):
                config={'endpoint_arm':arm,'width':128,'depth':2,'classes':3,'weight_decay':0,'learning_rate':.001,'dropout':.2,
                  'members':4 if arm in ('E4','N4','BASE','original_gnnm') else 1,'source_optimizer_count':1,
                  'node_inputs':inputs,'endpoint_sources':self.bindings}
                return make_adapter(torch=t,project_root=self.root,study={'dataset':'synthetic_endpoint_fixture','data':{},'source':{}},
                  run={'config':config,'seed':76101,'body':body,'run_id':arm+'_'+str(body)},device='cpu')
            adapter=create('E4');saved=copy.deepcopy(adapter.capture_state());parameters=dict(adapter.model.named_parameters())
            before={name:p.detach().clone() for name,p in parameters.items()}
            initial=copy.deepcopy(adapter.external_capture())
            adapter.model.train();outputs=adapter.logits_fn(adapter.model,True,1)
            loss=t.nn.functional.cross_entropy(outputs[:,adapter.train_ids].flatten(0,1),adapter.train_y.repeat(4));loss.backward()
            packed={name:None if p.grad is None else p.grad.clone() for name,p in parameters.items()}
            adapter.model.zero_grad(set_to_none=True);adapter.external_restore(initial);adapter.model.train()
            for member in range(4):
                current=adapter.logits_fn(adapter.model,True,1)
                # Restore before each full forward so the route uses its same
                # source stream. Only this route's factual loss is accumulated.
                adapter.external_restore(initial);adapter.model.train()
                (t.nn.functional.cross_entropy(current[member,adapter.train_ids],adapter.train_y)/4).backward()
            for name,p in parameters.items():
                if packed[name] is not None:t.testing.assert_close(p.grad,packed[name],rtol=2e-4,atol=2e-5)
                self.assertTrue(t.equal(before[name],p.detach()))
            adapter.restore_state(saved);restored=adapter.capture_state()
            for first,last in zip(saved['external']['streams'],restored['external']['streams']):self.assertTrue(t.equal(first['cpu'],last['cpu']))
            self.assertEqual(saved['external']['counters'],restored['external']['counters'])
            singles=[create('equipped_M1',body) for body in range(4)]
            ids=[{p.data_ptr() for p in a.model.parameters()} for a in singles]
            self.assertTrue(all(not ids[i]&ids[j] for i in range(4) for j in range(i)))
            self.assertEqual(len({a.provenance['actual_model_seed'] for a in singles}),4)
            original=create('original_gnnm')
            self.assertEqual(original.fusion,'mean_logits')
            names=[n for n,_ in original.model.named_parameters()]
            self.assertTrue(any(n.endswith('R') for n in names));self.assertTrue(any(n.endswith('S') for n in names));self.assertTrue(any(n.endswith('B') for n in names))
            self.assertTrue(all(not a.optimizer.state for a in [adapter,*singles,original]))

if __name__=='__main__':unittest.main()

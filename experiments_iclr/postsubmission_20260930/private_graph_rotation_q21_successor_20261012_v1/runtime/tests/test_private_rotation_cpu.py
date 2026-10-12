"""Deferred synthetic CPU source tests. No CUDA or scientific dataset arrays."""
import ast
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from adapters.private_rotation import ARMS,CONTRACT,SOURCES,base_train,make_adapter,receipt

ROOT=Path(os.environ['GNNM_REFERENCE_PROJECT']).resolve()
PACKET=Path(__file__).resolve().parents[2]
BINDINGS=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text())


def fixture(directory,backbone,arm,body=0,features=300,classes=10):
    import torch
    n=18;x=torch.arange(n*features,dtype=torch.float32).reshape(n,features)/(n*features)
    edges=torch.tensor([[i for i in range(n) for _ in range(2)],[j for i in range(n) for j in (i,(i+1)%n)]])
    ids=torch.tensor([4,0,5,1,3,2,9,8,7,6]);train={'ids':ids,'y':ids%classes}
    select={'ids':torch.tensor([10,11,12]),'y':torch.tensor([0,1,2])};inputs={}
    for role,payload in [('graph',{'x':x,'edge_index':edges}),('train',train),('select',select)]:
        path=directory/(role+'.pt');torch.save(payload,path);inputs[role]=receipt(ROOT,str(path.relative_to(ROOT)))
    c={'rotation_arm':arm,'members':ARMS[arm][1],'physical_slow_bodies':1,'source_optimizer_count':1,
      'backbone':backbone,'width':128,'depth':2,'classes':classes,'weight_decay':0,'learning_rate':.001,'dropout':.2,
      'rotation_contract':CONTRACT,'rotation_sources':BINDINGS,'node_inputs':inputs}
    run={'run_id':'synthetic-rotation-'+arm,'seed':77101,'body':body,'config':c}
    study={'dataset':'wikics','data':{'scope':'synthetic_CPU_engineering_only;18nodes;no_scientific_data'},'source':{'files':list(BINDINGS.values())}}
    return study,run


def literal_reference(adapter,run):
    """Untouched literal constructor methods on original300/10 dimensions."""
    path=ROOT/BINDINGS['family']['path'];tree=ast.parse(path.read_bytes())
    family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family')
    methods=[copy.deepcopy(next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in ('seeds','make','scope','logits')]
    fit=next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    loop=next(n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    train=ast.parse('def unit(self,model,optimizer,streams,members,bank):\n    torch=self.torch\n    return train_loss\n').body[0]
    train.body=[train.body[0],copy.deepcopy(loop.body[0]),copy.deepcopy(loop.body[1]),train.body[-1]];methods.append(train)
    code=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='UntouchedLiteral',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
    env={'math':math};exec(compile(code,str(path),'exec'),env);source=env['UntouchedLiteral']()
    for key in ('torch','F','cfg','device','cuda','Model','routes','factors','corrections','set_seed','data','batch','train_ids'):
        setattr(source,key,getattr(adapter.source_family,key))
    namespace=base_train(ROOT/BINDINGS['base_train']['path']);namespace['F']=source.F;source.train_step=namespace['train_step']
    kind,members=ARMS[run['config']['rotation_arm']];m,o,s=source.make(run['seed'],run['body'],kind,True,members)
    return source,m,o,s


class RotationCPU(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;self.threads=torch.get_num_threads();torch.set_num_threads(2)
    def tearDown(self):self.t.set_num_threads(self.threads)
    def equal(self,left,right):
        t=self.t
        if t.is_tensor(left):self.assertTrue(t.equal(left,right))
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for k in left:self.equal(left[k],right[k])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):self.equal(a,b)
        else:self.assertEqual(left,right)

    def test_real_source_constructor_gradients_steps_and_streams_both_backbones(self):
        t=self.t
        for backbone in ('GAT','SAGE'):
            for arm in ARMS:
                with self.subTest(backbone=backbone,arm=arm),tempfile.TemporaryDirectory(dir=ROOT,prefix='rotation-CPU-') as tmp:
                    study,run=fixture(Path(tmp),backbone,arm);a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
                    f,m,o,s=literal_reference(a,run);members=ARMS[arm][1]
                    self.equal(a.model.state_dict(),m.state_dict());self.equal(a.optimizer.state_dict(),o.state_dict());self.equal(a.source_streams,s)
                    self.assertEqual(len(a.optimizer.param_groups),1)
                    parameters=list(a.model.parameters());owned=a.optimizer.param_groups[0]['params']
                    self.assertEqual(len(owned),len(parameters));self.assertEqual({id(p) for p in owned},{id(p) for p in parameters})
                    for step in (1,2):
                        global_rng=t.get_rng_state().clone();v=a.train_step(step);expected=f.unit(m,o,s,members,members>1)
                        self.assertEqual(v['objective'],expected);self.equal(a.model.state_dict(),m.state_dict());self.equal(a.optimizer.state_dict(),o.state_dict())
                        self.equal([p.grad for p in a.model.parameters()],[p.grad for p in m.parameters()]);self.equal(a.source_streams,s)
                        self.assertTrue(t.equal(global_rng,t.get_rng_state()));self.assertEqual(v['operation_counts']['model_forwards'],members)
                        self.assertEqual(v['operation_counts']['backward_calls'],1);self.assertEqual(v['operation_counts']['autograd_grad_VJP_calls'],0)

    def test_equipment_placement_initial_identity_first_signs_and_I4_seed_ancestry(self):
        t=self.t
        for backbone in ('GAT','SAGE'):
            with tempfile.TemporaryDirectory(dir=ROOT,prefix='rotation-CPU-equipment-') as tmp:
                study,run=fixture(Path(tmp),backbone,'shared4_paired_local');a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
                sites=a.model._correction_sites;expected_map='linear_2' if backbone=='GAT' else 'linear_1'
                self.assertEqual(len(sites),2);self.assertTrue(all(s['path'].endswith(expected_map) for s in sites))
                self.assertEqual(sum(s['extra_private_parameters'] for s in sites),2048 if backbone=='GAT' else 4096)
                for module in a.model.modules():
                    if isinstance(module,a.source_family.factors.FactorLinear):
                        self.assertTrue(t.equal(module.r,t.ones_like(module.r)));self.assertTrue(t.equal(module.s,t.ones_like(module.s)))
                    if hasattr(module,'u1'):
                        self.assertTrue(t.equal(module.u1,module.u2));self.assertNotEqual(module.u1.data_ptr(),module.u2.data_ptr())
                        x=t.arange(module.weight.shape[1],dtype=t.float32)[None]/100
                        expected=t.nn.functional.linear(x,module.weight,module.bias)
                        self.assertTrue(t.allclose(module(x),expected,rtol=1e-5,atol=1e-5))
                for body in range(4):
                    st,r=fixture(Path(tmp),backbone,'paired_local_M1',body);b=make_adapter(torch=t,project_root=ROOT,study=st,run=r,device='cpu')
                    self.assertEqual(b.provenance['actual_model_seed'],77101+1000003*body)
                    for shared,single in zip(sites,b.model._correction_sites):self.assertEqual(shared['orientation_seeds'][body],single['orientation_seeds'][0])
                st,r=fixture(Path(tmp),backbone,'rademacher_allmap_M1',3);b=make_adapter(torch=t,project_root=ROOT,study=st,run=r,device='cpu')
                generator=t.Generator(device='cpu').manual_seed(77101+1000003*3+2000003);signs=t.randint(0,2,(1,300),generator=generator)*2-1
                self.assertTrue(t.equal(b.model.input_linear.r,signs.to(b.model.input_linear.r)))
                for name,module in b.model.named_modules():
                    if isinstance(module,b.source_family.factors.FactorLinear):
                        self.assertTrue(t.equal(module.s,t.ones_like(module.s)))
                        if name!='input_linear':self.assertTrue(t.equal(module.r,t.ones_like(module.r)))

    def test_bank_flattened_mean_gradient_and_private_one_quarter(self):
        t=self.t
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='rotation-CPU-gradient-') as tmp:
            study,run=fixture(Path(tmp),'SAGE','shared4_paired_local');a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
            a.model.train();state=copy.deepcopy(a.source_streams);logits=a.source_family.logits(a.model,a.source_streams,True,a.train_ids)
            losses=[t.nn.functional.cross_entropy(logits[i],a.train_y) for i in range(4)]
            parameters=list(a.model.parameters());expected=t.autograd.grad(t.stack(losses).mean(),parameters,retain_graph=True,allow_unused=True)
            module=next(m for m in a.model.modules() if hasattr(m,'u1'))
            own=t.autograd.grad(losses[0],module.u1,retain_graph=True)[0];pooled=t.autograd.grad(t.stack(losses).mean(),module.u1)[0]
            self.assertTrue(t.allclose(pooled[0],own[0]/4,rtol=1e-6,atol=1e-7));self.assertTrue(t.equal(own[1:],t.zeros_like(own[1:])))
            a.source_streams[:]=state;a.train_step(1)
            for p,g in zip(parameters,expected):
                if g is None:self.assertIsNone(p.grad)
                else:self.assertTrue(t.allclose(p.grad,g,rtol=1e-5,atol=1e-6))

    def test_selected_state_stream_moment_roundtrip_and_context_free_reconstruction(self):
        t=self.t
        for arm in ARMS:
            with tempfile.TemporaryDirectory(dir=ROOT,prefix='rotation-CPU-state-') as tmp:
                study,run=fixture(Path(tmp),'GAT',arm);a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
                a.train_step(1);metrics=a.validate(1);a.observe_validation(1,metrics)
                p=copy.deepcopy(a.model.state_dict());o=copy.deepcopy(a.optimizer.state_dict());s=copy.deepcopy(a.capture_state())
                a.train_step(2);after=copy.deepcopy(a.model.state_dict());moments=copy.deepcopy(a.optimizer.state_dict());streams=copy.deepcopy(a.source_streams)
                # Factory accepts ordinary endpoint reconstruction without any
                # TRAIN lifecycle context, as the frozen export adapter expects.
                b=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
                b.model.load_state_dict(p);b.optimizer.load_state_dict(o);b.restore_state(s);b.train_step(2)
                self.equal(after,b.model.state_dict());self.equal(moments,b.optimizer.state_dict());self.equal(streams,b.source_streams)
                b.model.eval();logits=b.logits_fn(b.model,False,2);self.assertEqual(tuple(logits.shape),(ARMS[arm][1],18,10))
                self.assertTrue(t.equal(b.probabilities(logits),logits.softmax(-1).mean(0)))

    def test_nonlegacy_feature_class_provider_on_both_real_backbones(self):
        t=self.t
        for backbone in ('GAT','SAGE'):
            for arm in ARMS:
                with tempfile.TemporaryDirectory(dir=ROOT,prefix='rotation-CPU-provider-') as tmp:
                    study,run=fixture(Path(tmp),backbone,arm,features=7,classes=3)
                    a=make_adapter(torch=t,project_root=ROOT,study=study,run=run,device='cpu')
                    self.assertEqual(a.model.body.input_linear.weight.shape[1] if hasattr(a.model,'body') else a.model.input_linear.weight.shape[1],7)
                    a.train_step(1);a.model.eval();self.assertEqual(tuple(a.logits_fn(a.model,False,1).shape),(ARMS[arm][1],18,3))


if __name__=='__main__':unittest.main()

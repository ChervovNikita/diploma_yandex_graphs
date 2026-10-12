"""New-port CPU QA on fresh small fixtures only. No provider/outcome loading."""
import copy,json,math,sys
from pathlib import Path
from types import SimpleNamespace
RUNTIME=Path(__file__).resolve().parents[1];PACKET=RUNTIME.parent;ROOT=PACKET.parent
sys.path.insert(0,str(RUNTIME))

def same(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    import torch,torch_geometric
    from adapters import native_combinations as port
    from adapters import common_native_wrapper as common
    from engine import inspect_ownership,inspect_independence
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    bindings=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text());cases=[];independence=[];steps=0
    for features,classes in ((300,10),(745,8)):
      nodes=2*classes+8
      x=torch.linspace(-.5,.5,nodes*features).reshape(nodes,features)
      # Repeated factual record and an isolated final node; preserve multiplicity.
      sources=torch.arange(nodes-2);targets=sources+1
      edges=torch.cat((torch.stack((sources,targets)),torch.stack((targets,sources)),torch.tensor([[0,0],[1,1]]),torch.arange(nodes).repeat(2,1)),dim=1)
      train_ids=torch.arange(2*classes);train_y=torch.arange(2*classes)%classes;select_ids=torch.arange(2*classes,2*classes+3);select_y=torch.arange(3)%classes
      for family,roster in port.FAMILIES.items():
       for backbone in roster['backbones']:
        def construct(arm,body=0):
            if arm in port.CONTROLS:
                source,model,opt,streams,primitives,rendered=common.construct_source(torch,ROOT,port.bindings_common(bindings),features=features,classes=classes,
                    seed=88101,body=body,arm=arm,backbone=backbone,dropout=.2,learning_rate=.001,device='cpu')
                labels=torch.zeros(nodes,dtype=torch.long);labels[train_ids]=train_y;mask=torch.zeros(nodes,dtype=torch.bool);mask[train_ids]=True
                source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask);source.train_ids=train_ids
                source.batch={'x':x,'edge_index':edges,'graph':SimpleNamespace(edge_index=edges)}
                bank=arm=='shared4_unchanged'
                def train_once():
                    if bank:return primitives['packed_train_step'](source,model,opt,streams,4)
                    with source.scope(streams,0):return primitives['train_step_tabm'](model,source.data,opt,False) if arm=='original_gnnm' else primitives['train_step'](model,opt,source.data,False)
                return SimpleNamespace(model=model,optimizer=opt,source=source,streams=streams,train_step=lambda step:train_once(),
                    capture_state=lambda:{'streams':copy.deepcopy(streams)},restore_state=lambda value:streams.__setitem__(slice(None),copy.deepcopy(value['streams'])))
            source,model,opt,streams,rendered=port.construct_source(torch,ROOT,bindings,family=family,arm=arm,features=features,classes=classes,train_count=len(train_ids),
                seed=88101,body=body,backbone=backbone,dropout=.2,learning_rate=.001,device='cpu')
            return port.attach_adapter(torch,source,model,opt,streams,family=family,arm=arm,x=x,edges=edges,train_ids=train_ids,train_y=train_y,
                select_ids=select_ids,select_y=select_y,provenance={'actual_model_seed':source.seeds(88101,body)[0],'classes':classes,'QA':'fresh synthetic CPU'})
        for arm in roster['arms']+port.CONTROLS:
            adapter=construct(arm);inspect_ownership(adapter)
            before=(copy.deepcopy(adapter.model.state_dict()),copy.deepcopy(adapter.optimizer.state_dict()),copy.deepcopy(adapter.capture_state()))
            result=adapter.train_step(1);steps+=1
            (PACKET/'CPU_QA_PROGRESS.json').write_text(json.dumps({'toy_optimizer_steps':steps,'current_case':{'family':family,'arm':arm,'backbone':backbone,'features':features},'benchmark_fits':0})+'\n')
            after=(copy.deepcopy(adapter.model.state_dict()),copy.deepcopy(adapter.optimizer.state_dict()),copy.deepcopy(adapter.capture_state()))
            loss=result['objective'] if isinstance(result,dict) else result
            assert math.isfinite(loss) and any(not torch.equal(before[0][k],after[0][k]) for k in before[0])
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
            adapter.model.load_state_dict(before[0]);adapter.optimizer.load_state_dict(before[1]);adapter.restore_state(before[2])
            replay=adapter.train_step(1);steps+=1
            (PACKET/'CPU_QA_PROGRESS.json').write_text(json.dumps({'toy_optimizer_steps':steps,'current_case':{'family':family,'arm':arm,'backbone':backbone,'features':features},'benchmark_fits':0})+'\n')
            assert same(result,replay) and same(after[0],adapter.model.state_dict()) and same(after[1],adapter.optimizer.state_dict()) and same(after[2],adapter.capture_state())
            if arm not in port.CONTROLS:
                adapter.model.eval();rng=copy.deepcopy(adapter.capture_state())
                validation=adapter.validate(1);served=adapter.predict_rows(torch.arange(2*classes+3,nodes),1)
                assert math.isfinite(validation['pooled_nll']) and served.shape==(port.arm_shape(family,arm)[0],5,classes)
                assert same(rng,adapter.capture_state())
                if family=='nonlocal_label_retrieval':
                    try:adapter.predict_rows(train_ids,1)
                    except ValueError:pass
                    else:raise AssertionError('TRAIN query labels admitted into serving')
                if family=='shared_recurrent_class_decoder':
                    source,target,inverse=adapter.source.decoder_support
                    assert int(source.eq(target).sum())==nodes and int(((source==0)&(target==1)).sum())==3
                    assert float(inverse[-1])==1.0
                    if arm=='joint_untied4_graphP_commonB':
                        assert isinstance(adapter.model,torch.nn.ModuleList) and len(adapter.model)==4
                        sets=[{p.data_ptr() for p in body.parameters()} for body in adapter.model]
                        assert all(not sets[a]&sets[b] for a in range(4) for b in range(a))
            cases.append({'family':family,'arm':arm,'backbone':backbone,'features':features,'classes':classes,'ownership':True,'optimizer_and_external_RNG_replay_exact':True})
        for bank_arm,single in {**roster['i4'],**port.CONTROL_I4}.items():
            learners=[construct(single,body) for body in range(4)]
            assert inspect_independence(learners)['independent_adapters']==4
            assert len({learner.source.seeds(88101,b)[0] for b,learner in enumerate(learners)})==4
            independence.append({'family':family,'bank_arm':bank_arm,'backbone':backbone,'features':features,'classes':classes,'separate_complete_models_optimizers_RNG':True})
    # Independent algebra checks for source GNCL, query exclusion and decoder.
    src,_,_,_,_=port.construct_source(torch,ROOT,bindings,family='gncl_supcon',arm='shared4_GNCL_SupCon',features=5,classes=3,train_count=6,
        seed=88101,body=0,backbone='SAGE',dropout=.2,learning_rate=.001,device='cpu')
    hidden=torch.linspace(-1,1,4*6*7).reshape(4,6,7).requires_grad_();targets=torch.arange(6)%3
    actual,_=src.within_route_supcon(hidden,targets);z=torch.nn.functional.normalize(hidden,dim=-1);scores=z@z.transpose(-2,-1)/.2
    expected=[]
    for m in range(4):
      for i in range(6):
        allowed=[j for j in range(6) if j!=i];positive=[j for j in allowed if targets[j]==targets[i]]
        expected.append(torch.logsumexp(scores[m,i,allowed],0)-scores[m,i,positive].mean())
    expected=torch.stack(expected).mean();torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(torch.autograd.grad(actual,hidden,retain_graph=True)[0],torch.autograd.grad(expected,hidden)[0],rtol=1e-5,atol=1e-6)
    logits=torch.linspace(-2,2,4*6*3).reshape(4,6,3)
    pool=-logits.softmax(-1).mean(0).log().gather(1,targets[:,None]).mean()
    torch.testing.assert_close(src.probability_pool_ce(logits,targets),pool)
    memory=common.load_module(ROOT,bindings['memory']);anchors=torch.arange(6);gen=torch.Generator().manual_seed(88101+5000011)
    q=memory.common_half_query(torch,anchors,targets,3,generator=gen)
    assert len(q)==3 and all(targets[q].eq(c).sum()==1 for c in range(3))
    h=torch.randn(1,8,5,requires_grad=True);z=torch.randn(1,8,3,requires_grad=True)
    live=memory.label_memory(torch,z,h,anchors,targets,q,mode='fit')
    assert not torch.isin(q,live['support_rows']).any()
    grad=torch.autograd.grad(-live['retrieval_log_probs'].gather(-1,targets[q][None,:,None]).mean(),h)[0]
    assert grad[:,q].abs().sum()>0 and grad[:,live['support_rows']].abs().sum()>0
    detached=memory.label_memory(torch,z,h,anchors,targets,q,mode='fit',stop_attention_gradient=True)
    assert not detached['retrieval_log_probs'].requires_grad
    serve=memory.label_memory(torch,z,h,anchors,targets,torch.tensor([6,7]),mode='serve')
    torch.testing.assert_close(serve['mixed_log_probs'].exp(),.5*serve['native_log_probs'].exp()+.5*serve['retrieval_log_probs'].exp())
    decoder=common.load_module(ROOT,bindings['decoder']);z=torch.randn(4,5,3,requires_grad=True);B=torch.randn(3,3,requires_grad=True)
    source=torch.tensor([0,1,2,3,4,0,0]);target=torch.tensor([0,1,2,3,4,1,1]);inverse=torch.tensor([1.,1/3,1.,1.,1.])
    actual,_=decoder.decode(torch,z,B,support=(source,target,inverse));q=z.softmax(-1)
    for step in range(5):
        rows=torch.stack([q[:,source[target==i]].mean(1) for i in range(5)],1);scores=z+rows@B;q=scores.softmax(-1)
    torch.testing.assert_close(actual,scores,rtol=1e-6,atol=1e-6)
    gradients=torch.autograd.grad(actual.square().mean(),(z,B));assert all(g.abs().sum()>0 for g in gradients)
    assert not torch.cuda.is_initialized() and len(cases)==120 and len(independence)==42 and steps==240
    result={'schema':'native-combinations-new-port-CPU-QA-v1','status':'passed','native_cases':cases,'I4_independence_cases':independence,
        'synthetic_toy_optimizer_steps':steps,'algebra_checks':['GNCL probability CE','within-route SupCon value/gradient','half-query per-class/exclusion/live-vs-detached attention','literal logaddexp mixture','five decoder steps and live gradients/multiplicity/isolate'],
        'torch_version':torch.__version__,'pyg_version':torch_geometric.__version__,'python':sys.executable,'runtime_pin_claimed':False,
        'real_tensor_reads':0,'scientific_fits':0,'scientific_scores':0,'CUDA_operations':0,'old_closed_audits_rerun':False}
    (PACKET/'VALIDATION_SYNTHETIC.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','native_cases':len(cases),'I4_cases':len(independence),'synthetic_toy_optimizer_steps':steps,'scientific_fits':0}))

if __name__=='__main__':main()

"""Tensor-only link export authority fixtures; no native model or forward."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint');parser.add_argument('--checkpoint-sha256');args=parser.parse_args()
    root,output=Path(args.root).resolve(),Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(root) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh own fixture output and hidden CUDA required')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    started=time.perf_counter();output.mkdir(parents=True)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import adapters
    import run_fit
    from common import receipt, digest, load_module
    folder=output/'fixture_sources';folder.mkdir()
    (folder/'callback.py').write_text('# synthetic callback source fixture\n')
    (folder/'policy.py').write_text('# synthetic policy source fixture\n')
    (folder/'session.py').write_text('import copy\ndef selected_mean_sidecar(session):\n    if session.sidecar is None: raise ValueError("stale fixture sidecar")\n    return copy.deepcopy(session.sidecar)\n')
    source_pins=[receipt(folder/name) for name in ('callback.py','policy.py','session.py')]
    (folder/'MANIFEST.json').write_text(json.dumps({'files':source_pins},sort_keys=True))
    provider_pin=receipt(folder/'MANIFEST.json');all_pins=source_pins+[provider_pin]
    provider,_=load_module(folder/'session.py',root,receipt(folder/'session.py')['sha256'])
    n=160084;ids=torch.arange(n);endpoints=torch.stack((ids,ids+1),dim=1)
    labels=torch.cat((torch.ones(60084),torch.zeros(100000)))
    served=torch.linspace(-1.,1.,n);mean=served*.75
    base={'outputs':served,'labels':labels,'ids':ids,'query_endpoints':endpoints,'loss':.7,'metrics':{'hits@50':.1},
          'same_bank_mean_outputs':mean,'same_bank_mean_loss':.8,'same_bank_mean_metrics':{'hits@50':.05},
          'role':'VALID','complete_source_VALID':True,'TEST_accessed':False,
          'complete_VALID_positive_records':60084,'complete_VALID_negative_records':100000}
    job={'id':'synthetic-link-authority-fixture','procedure':'interaction_input_conditionalJ',
         'callback_module':str(folder/'callback.py'),'callback_sha256':receipt(folder/'callback.py')['sha256'],
         'link_policy_module':str(folder/'policy.py'),'link_policy_sha256':receipt(folder/'policy.py')['sha256'],
         'method_settings':{'manifest_sha256':provider_pin['sha256']}}
    class Inner:
        def __init__(self):
            self.session=SimpleNamespace(native=SimpleNamespace(updates=17,valid={'edge':endpoints[:60084],'edge_neg':endpoints[60084:]}),
                interaction_metadata={'method_id':job['procedure'],'source_receipts':all_pins},sidecar=None)
            self.callback_receipt=receipt(folder/'callback.py');self.extension_receipt=receipt(folder/'session.py')
            self.dependency_receipts=all_pins;self.completed_epochs=1;self.model=object();self.optimizers=[];self.valid_calls=0
            self.weights=torch.tensor([.1,.2]);self.valid_mutation=None
        def state(self):return {'weights':self.weights.clone(),'epoch':self.completed_epochs,'updates':self.session.native.updates,'nested':(None,True,3.5,{'x':[1,'a']})}
        def load_state(self,state):
            self.weights=state['weights'].clone();self.completed_epochs=state['epoch'];self.session.native.updates=state['updates'];self.session.sidecar=None
        def train_epoch(self):
            self.completed_epochs+=1;self.session.native.updates+=17;self.session.sidecar=None
            return {'epoch':self.completed_epochs,'complete_native_TRAIN':True}
        def validate(self):
            self.valid_calls+=1;r=copy.deepcopy(base)
            if self.valid_mutation:self.valid_mutation(r)
            self.session.sidecar={'schema':'same-call-NCN-interaction-VALID-sidecar-v1','role':'VALID','TEST_accessed':False,
                'method_id':job['procedure'],'source_updates':self.session.native.updates,'ids':r['ids'].clone(),
                'labels':r['labels'].clone(),'query_endpoints':r['query_endpoints'].clone(),
                'actual_served_raw_logits':r['outputs'].clone(),'actual_loss':r['loss'],'actual_metrics':copy.deepcopy(r['metrics']),
                'same_bank_mean_raw_logits':r['same_bank_mean_outputs'].clone(),'same_bank_mean_loss':r['same_bank_mean_loss'],
                'same_bank_mean_metrics':copy.deepcopy(r['same_bank_mean_metrics']),'selected_checkpoint_bound':False,
                'caller_must_bind_loaded_selected_epoch_checkpoint':True,'extra_forward_calls':0}
            return r
        def close(self):pass
    checkpoint=output/'synthetic-selected.bin';checkpoint.write_bytes(b'fixture selected checkpoint bytes')
    check_pin=receipt(checkpoint);checks=[]
    def fixture(bind=True):
        inner=Inner();adapter=adapters.LinkSameCallMeanAdapter(torch,job,None,root,inner,provider)
        binding={'runtime_sha256':{name:receipt(Path(adapters.__file__).with_name(name))['sha256'] for name in
                    ('run_fit.py','adapters.py','bookkeeping.py','common.py','controllers.py','source_rng.py')},
            'policy_sha256':job['link_policy_sha256'],'callback_sha256':inner.callback_receipt['sha256'],
            'extension_sha256':inner.extension_receipt['sha256'],
            'dependency_sha256':{p['path']:p['sha256'] for p in all_pins}}
        state=inner.state();adapter.load_state(state)
        payload={'job_hash':digest(job),'runtime_binding':binding,'selector':{'selected_epoch':1},'adapter':copy.deepcopy(state)}
        kwargs={'selected_checkpoint':check_pin,'job_hash':digest(job),'runtime_binding':binding}
        if bind:adapter.bind_loaded_selection(payload,**kwargs)
        return adapter,inner,payload,kwargs
    def export(adapter,actual,kwargs):return adapter.selected_mean_export(actual,selected_epoch=1,**kwargs)
    def rejects(name,operation):
        try:operation()
        except (ValueError,KeyError):checks.append({'name':name,'expected_refusal':True});return
        raise AssertionError('Expected refusal missing: '+name)
    a,i,p,k=fixture();v=a.validate();out=export(a,v,k)
    assert i.valid_calls==1 and torch.equal(out['raw_logits'],mean) and not torch.equal(out['raw_logits'],served)
    assert torch.equal(out['query_endpoints'],endpoints) and out['extra_forward_calls']==0
    checks.append({'name':'selected-actual-and-mean-preserved-one-VALID-zero-extra-forward','passed':True})
    a,i,p,k=fixture(False);v=a.validate();rejects('unbound-selection',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();a.load_state(i.state());rejects('stale-after-restore',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();a.train_epoch();rejects('stale-after-TRAIN',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();a.validate();rejects('earlier-VALID-serial',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();i.session.sidecar=None;rejects('source-cache-invalidated-outside-wrapper',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();v['outputs'][0]+=1;rejects('changed-actual-logit',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();v['same_bank_mean_VALID']['outputs'][0]+=1;rejects('changed-mean-logit',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();v['query_endpoints'][0,0]+=1;rejects('changed-query-endpoint',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();v['same_bank_mean_VALID']['metrics']['hits@50']+=.1;rejects('changed-mean-native-metric',lambda:export(a,v,k))
    a,i,p,k=fixture();v=a.validate();bad=copy.deepcopy(k);bad['runtime_binding']['policy_sha256']='0'*64;rejects('wrong-link-policy',lambda:export(a,v,bad))
    a,i,p,k=fixture();v=a.validate();bad=copy.deepcopy(k);bad['runtime_binding']['extension_sha256']='0'*64;rejects('wrong-outer-extension',lambda:export(a,v,bad))
    a,i,p,k=fixture();v=a.validate();bad=copy.deepcopy(k);bad['job_hash']='0'*64;rejects('wrong-job',lambda:export(a,v,bad))
    a,i,p,k=fixture();v=a.validate();rejects('wrong-selected-epoch',lambda:a.selected_mean_export(v,selected_epoch=2,**k))
    a,i,p,k=fixture(False);p['adapter']['weights'][0]+=1;rejects('selected-payload-not-loaded-state',lambda:a.bind_loaded_selection(p,**k))
    a,i,p,k=fixture();v=a.validate();checkpoint.write_bytes(b'changed fixture checkpoint');rejects('checkpoint-bytes-changed',lambda:export(a,v,k));checkpoint.write_bytes(b'fixture selected checkpoint bytes')
    a,i,p,k=fixture();i.valid_mutation=lambda r:r.update(TEST_accessed=True);rejects('TEST-role',lambda:a.validate())
    a,i,p,k=fixture();i.valid_mutation=lambda r:r.update(complete_VALID_negative_records=99999);rejects('partial-VALID',lambda:a.validate())
    a,i,p,k=fixture();i.valid_mutation=lambda r:r['labels'].__setitem__(0,0);rejects('wrong-positive-negative-role',lambda:a.validate())
    a,i,p,k=fixture();i.valid_mutation=lambda r:r['query_endpoints'].__setitem__((0,0),-1);rejects('wrong-native-query-order',lambda:a.validate())
    a,i,p,k=fixture();i.valid_mutation=lambda r:r['same_bank_mean_outputs'].__setitem__(0,float('nan'));rejects('nonfinite-mean',lambda:a.validate())
    # New link CPU mode is separately bounded; old node/OGB mode is untouched.
    original_controller=run_fit.controller
    run_fit.controller=lambda *args,**kw:(SimpleNamespace(track='matched_extended'),None,None)
    cpu={'schema':'native-full-epoch-fit-v1','family':'link','TEST_access':False,'recipe':{'dataset':'ogbl-collab'},
         'execution_mode':'cpu_link_engineering_v1','engineering_only':True,'device':'cpu','seed':0,'procedure':'interaction_input_conditionalJ',
         'cpu_engineering_limits':{'maximum_epochs':2,'maximum_optimizer_updates':34},
         'policy':{'maximum_epochs':2,'minimum_epochs':2,'evaluate_TRAIN_each_epoch':False}}
    try:
        run_fit.validate_job(cpu,root);checks.append({'name':'explicit-finite-link-CPU-mode','passed':True})
        bad=copy.deepcopy(cpu);bad['cpu_engineering_limits']['maximum_optimizer_updates']=35;rejects('unbounded-CPU-work',lambda:run_fit.validate_job(bad,root))
        bad=copy.deepcopy(cpu);bad['policy']['maximum_epochs']=3;rejects('extended-CPU-epoch-cap',lambda:run_fit.validate_job(bad,root))
        bad=copy.deepcopy(cpu);bad['engineering_only']=False;rejects('scientific-CPU-mode',lambda:run_fit.validate_job(bad,root))
    finally:run_fit.controller=original_controller
    class Opt:
        param_groups=[]
        def __init__(self):self.steps=0
        def step(self):self.steps+=1
    owner=Opt();observer=run_fit.OptimizerObserver(torch,[owner],maximum_completed=1);owner.step()
    rejects('optimizer-limit-before-second-step',owner.step);assert owner.steps==1 and observer.completed==[1];observer.close()
    actual_state=None
    if args.checkpoint:
        path=Path(args.checkpoint).resolve()
        if not path.is_relative_to(root) or receipt(path)['sha256']!=args.checkpoint_sha256:raise ValueError('Own actual checkpoint receipt differs')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        a._state_signature(saved['session'])
        import engineering_link_policy
        policy_job=copy.deepcopy(saved['job'])
        policy_job.update(engineering_only=True,execution_mode='cpu_link_engineering_v1',device='cpu',
            policy=copy.deepcopy(engineering_link_policy.FINITE_POLICY),
            cpu_engineering_limits={'maximum_epochs':2,'maximum_optimizer_updates':34},
            link_policy_module=str(Path(engineering_link_policy.__file__)),
            link_policy_sha256=receipt(engineering_link_policy.__file__)['sha256'])
        policy_job['recipe']['_repository_root']=str(root)
        finite=run_fit.validate_job(policy_job,root)
        assert (finite.maximum_epochs,finite.minimum_epochs,finite.patience,finite.native_family)==(2,2,1,'NCNC')
        selector=engineering_link_policy.create_selector(finite)
        role={'role':'VALID','complete_source_VALID':True,'complete_VALID_positive_records':60084,
              'complete_VALID_negative_records':100000,'metrics':{'hits@50':.2}}
        first=selector.observe_validation(1,role);second=selector.observe_validation(2,role)
        assert first['save_selected'] and not first['stop'] and not second['save_selected'] and second['stop'] and second['selected_epoch']==1
        restored=engineering_link_policy.create_selector(finite);restored.load_state(selector.state());assert restored.state()==selector.state()
        checks.append({'name':'actual-NCNC-finite-policy-dispatch-first-tie-two-epochs-selector-restore','passed':True})
        bad=copy.deepcopy(policy_job);bad['policy']['maximum_epochs']=3
        rejects('finite-policy-horizon-change',lambda:engineering_link_policy.create_policy(bad,root))
        bad=copy.deepcopy(policy_job);bad['recipe']['architecture']='NCN'
        rejects('finite-policy-native-family-change',lambda:engineering_link_policy.create_policy(bad,root))
        actual_state={'receipt':receipt(path),'exact_recursive_state_signature_supported':True,'native_models_created':0}
    usage=resource.getrusage(resource.RUSAGE_SELF)
    report={'schema':'runtime-link-v5-synthetic-authority-contract-v1','status':'passed','checks':checks,
            'checks_count':len(checks),'actual_checkpoint_shape_check':actual_state,
            'native_models':0,'native_forward_calls':0,'native_optimizer_updates':0,'native_VALID_calls':0,
            'synthetic_optimizer_steps':1,'TEST_accessed':False,'elapsed_seconds':time.perf_counter()-started,
            'process_peak_RSS_KiB':usage.ru_maxrss,'runtime_sources':{name:receipt(Path(adapters.__file__).with_name(name)) for name in
                ('run_fit.py','adapters.py','bookkeeping.py','common.py','controllers.py','source_rng.py')}}
    (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':len(checks),'native_models':0,'native_optimizer_updates':0,'elapsed_seconds':report['elapsed_seconds'],'process_peak_RSS_KiB':usage.ru_maxrss}))

if __name__=='__main__':main()

"""Allocation CPU-only synthetic engineering; no scientific fit/score/update."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback


def route():
    if socket.gethostname() != 'anogena-2-0':
        raise ValueError('Exact one-GPU allocation hostname required')
    uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()
    if uuids != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Sole authorized allocation UUID required')


def receipt(root,p):
    p=Path(p).resolve();raw=p.read_bytes()
    return {'path':str(p.relative_to(root)), 'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('project-root','runtime','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--prior-check')
    args=parser.parse_args();route();root=Path(args.project_root).resolve()
    runtime=(root/args.runtime).resolve();output=(root/args.output).resolve()
    if not runtime.is_relative_to(root) or not output.is_relative_to(root) or output.exists():
        raise ValueError('Fresh project-only qualification directory required')
    output.mkdir();os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,str(runtime))
    import torch
    import numpy as np
    from adapters.native_polynormer_qk import ARMS, SOURCES, CONTRACT, make_adapter
    from compile_native_polynormer_qk_matrix import make_study
    from engine import capture_checkpoint, restore_checkpoint, inspect_independence, _equal_tree
    torch.set_num_threads(2)
    if torch.cuda.is_initialized(): raise ValueError('CPU engineering may not initialize CUDA')
    sources={key:receipt(root,root/relative) for key,(relative,digest) in SOURCES.items()}
    if any(sources[key]['sha256']!=pin[1] for key,pin in SOURCES.items()): raise ValueError('Pinned source changed')
    fixture=output/'synthetic_inputs';fixture.mkdir()
    generator=torch.Generator().manual_seed(19)
    n=16;feature_count=7
    graph={'x':torch.randn(n,feature_count,generator=generator),
           'edge_index':torch.stack((torch.arange(n).repeat(2),torch.cat(((torch.arange(n)+1)%n,(torch.arange(n)-1)%n))))}
    train={'ids':torch.arange(6),'y':torch.arange(6)%3}
    select={'ids':torch.arange(6,10),'y':torch.arange(4)%3}
    for key,payload in (('graph',graph),('train',train),('select',select)):
        torch.save(payload,fixture/(key+'.pt'))
    bindings={key:receipt(root,fixture/(key+'.pt')) for key in ('graph','train','select')}
    load_orig=torch.load;step_orig=torch.optim.Adam.step
    allowed={str((root/row['path']).resolve()) for row in bindings.values()}
    def guarded_load(file,*a,**kw):
        if str(Path(file).resolve()) not in allowed: raise ValueError('Only synthetic engineering input/state loads allowed')
        return load_orig(file,*a,**kw)
    def denied_step(*a,**kw): raise ValueError('No optimizer step authorized for engineering')
    torch.load=guarded_load;torch.optim.Adam.step=denied_step
    results=[];prior=None
    if args.prior_check:
        prior_path=(root/args.prior_check).resolve()
        if not prior_path.is_relative_to(root): raise ValueError('Confined exact prior CPU evidence required')
        prior=json.loads(prior_path.read_text())
        if (prior.get('error')!='Zero-coordinate/native start changed' or prior.get('fits')!=0 or prior.get('optimizer_updates')!=0
                or [r.get('arm') for r in prior.get('completed_checks',[])]!=list(ARMS)[:8]):
            raise ValueError('Expected eight equipped/operator checks only')
        original_runtime=prior_path.parent.parent/'native_polynormer_qk_cpu_stage_20261012_v1/runtime'
        original_manifest=json.loads((original_runtime.parent/'MANIFEST.json').read_text())
        for row in original_manifest['files']:
            if not row['path'].startswith('runtime/'): continue
            name=row['path'][len('runtime/'):]
            original=(original_runtime/name).read_bytes()
            if (hashlib.sha256(original).hexdigest()!=row['sha256'] or len(original)!=row['bytes']
                    or original!=(runtime/name).read_bytes()):
                raise ValueError('Prior constructor source bytes changed')
        results=copy.deepcopy(prior['completed_checks'])
    def construct(arm,body=0,seed=78201):
        operator,kind,members=ARMS[arm]
        config={'polynormer_qk_arm':arm,'members':members,'physical_slow_bodies':1,'source_optimizer_count':1,
                'width':512,'depth':7,'global_layers':2,'heads':1,'local_transition_updates':100,
                'learning_rate':.001,'in_dropout':.5,'dropout':.5,'global_dropout':.5,'classes':3,
                'source_contract':CONTRACT,'node_inputs':bindings,'polynormer_qk_sources':sources}
        study={'dataset':'synthetic-new-data-engineering','data':{'engineering_synthetic':True},'source':{'files':list(sources.values())}}
        run={'run_id':'synthetic-'+arm+'-body'+str(body),'seed':seed,'body':body,'config':config}
        return make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu')
    try:
        native_reference=None
        for arm in ARMS:
            if prior is not None and arm in list(ARMS)[:8]: continue
            adapter=construct(arm);model=adapter.model;body=model.models[0].body
            members=ARMS[arm][2];operator=ARMS[arm][0]
            assert adapter.provenance['actual_model_seed']==78201
            assert not adapter.optimizer.state and len(adapter.source_session.optimizers)==1
            assert len(adapter.source_session.streams)==members
            assert all(p.device.type=='cpu' for p in model.parameters())
            factor_modules=[(name,m) for name,m in model.named_modules() if hasattr(m,'r') and hasattr(m,'s')]
            if arm=='bare_native_single': assert not factor_modules
            if arm=='original_gnnm':
                assert not factor_modules and adapter.fusion=='mean_logits'
                for name in ('lin_in','pred_local','pred_global'):
                    block=getattr(body,name)
                    assert block.W.bias is None and torch.equal(block.S,torch.ones_like(block.S)) and not block.B.any()
                    assert set(block.R.unique().tolist())<=({-1.,1.} if name=='lin_in' else {1.})
            if ARMS[arm][1]=='be_init':
                seed_generator=torch.Generator().manual_seed(78201+900001)
                expected=(torch.randint(0,2,body.lin_in.r.shape,generator=seed_generator)*2-1).to(body.lin_in.r)
                assert torch.equal(body.lin_in.r,expected)
                assert all(torch.equal(m.s,torch.ones_like(m.s)) for _,m in factor_modules)
                assert all(torch.equal(m.r,torch.ones_like(m.r)) for name,m in factor_modules if name!='models.0.body.lin_in')
            for k in body.global_attn.k_lins:
                for name in ('gamma','delta'):
                    if hasattr(k,name): assert not getattr(k,name).any()
            if operator=='full_qk':
                for q,k in zip(body.global_attn.q_lins,body.global_attn.k_lins):
                    for (qn,qp),(kn,kp) in zip(q.named_parameters(),k.named_parameters()):
                        assert qn==kn and torch.equal(qp,kp) and qp.untyped_storage().data_ptr()!=kp.untyped_storage().data_ptr()
            model.eval();shapes={}
            for global_flag in (False,True):
                model.set_global(global_flag)
                with torch.no_grad():
                    logits=adapter.logits_fn(model,False,0);pooled=adapter.probabilities(logits)
                assert logits.shape==(members,n,3) and pooled.shape==(n,3) and torch.isfinite(pooled).all()
                assert torch.allclose(pooled.sum(-1),torch.ones(n),rtol=1e-6,atol=1e-6)
                shapes[str(global_flag)]=list(logits.shape)
                if global_flag and arm=='native_tied_single': native_reference=logits.clone()
                if global_flag and arm in ('active_reversible_exp_single','pre_sigmoid_split_single','full_qk_single'):
                    assert torch.equal(logits,native_reference), 'Zero-coordinate/native start changed'
            results.append({'arm':arm,'constructor_initializer_serving':'pass','parameter_count':sum(p.numel() for p in model.parameters()),'shapes':shapes})
            del adapter,model,body;gc.collect()
        # One source-specific transition and genuine engine serializer proof.
        adapter=construct('pre_sigmoid_split_be_init');session=adapter.source_session
        parameter=next(adapter.model.parameters())
        adapter.optimizer.state[parameter]={'step':torch.tensor(7.),'exp_avg':torch.full_like(parameter,.125),'exp_avg_sq':torch.full_like(parameter,.25)}
        adapter.observe_validation(1,{'selection_metric':.5})
        chosen=copy.deepcopy(adapter.capture_state()['external']['local']['snapshot'])
        adapter.observe_validation(2,{'selection_metric':.5})
        assert adapter.capture_state()['external']['local']['step']==1, 'Strict-first local tie changed'
        with torch.no_grad(): parameter.add_(.125)
        adapter.optimizer.state[parameter]['exp_avg'].fill_(.75)
        session.steps=100
        torch.rand(3);np.random.rand(3)
        for i,stream in enumerate(session.streams): stream['cpu']=torch.Generator().manual_seed(900+i).get_state()
        state_before=adapter.capture_state();assert state_before['external']['local']['step']==1
        end_rng=torch.get_rng_state().clone();end_streams=copy.deepcopy(session.streams)
        checkpoint=output/'engineering_nested_state.pt';allowed.add(str(checkpoint.resolve()))
        payload=capture_checkpoint(torch,adapter,fit_id='synthetic-transition',step=100,validation={},decision={})
        torch.save(payload,checkpoint)
        with torch.no_grad(): parameter.zero_()
        session.steps=4;session._operator_failed=True;adapter.model.set_global(True)
        restore_checkpoint(torch,checkpoint,adapter,expected_fit_id='synthetic-transition',verify=True)
        assert _equal_tree(torch,state_before,adapter.capture_state())
        checks={}
        def synthetic_transition_callback(batch,y):
            checks['model_selected']=_equal_tree(torch,chosen['model'],adapter.model.state_dict())
            checks['Adam_selected']=_equal_tree(torch,chosen['optimizer'],adapter.optimizer.state_dict())
            checks['global_enabled']=all(body.body._global for body in adapter.model.models)
            checks['end_local_global_rng_preserved']=torch.equal(torch.get_rng_state(),end_rng)
            checks['end_local_private_rng_preserved']=_equal_tree(torch,session.streams,end_streams)
            checks['step_not_rewound']=session.steps==100
            if not all(checks.values()): raise AssertionError(checks)
            # Stub records zero scientific actions: no native training callback.
            return {'loss':torch.tensor(0.),'own_mean':torch.tensor(0.)}
        session.train_step=synthetic_transition_callback
        transition=adapter.custom_train(adapter,101)
        assert transition['operation_counts']['local_model_Adam_restores']==1
        assert adapter.capture_state()['external']['local']['transition_completed']
        results.append({'transition_at101':'pass','synthetic_control_state_only':True,'checks':checks,'engine_nested_restore':'pass',
                        'engineering_state_receipt':receipt(root,checkpoint),'actual_TRAIN_callbacks':0,'actual_optimizer_updates':0})
        del adapter,session,parameter,payload,chosen;gc.collect()
        independent=[construct('native_tied_single',body=b) for b in range(4)]
        ownership=inspect_independence(independent)
        assert [a.provenance['actual_model_seed'] for a in independent]==[78201+1009*b for b in range(4)]
        results.append({'genuine_I4_CPU_constructor_ownership':'pass','ownership':ownership,'seed_law':[78201+1009*b for b in range(4)]})
        del independent;gc.collect()
        for dataset in ('wikics','amazon-photo'):
            custody_path=root/'training_pipeline_v2_20261012/prepared_data'/dataset/'CUSTODY.json'
            custody=json.loads(custody_path.read_text())
            source={'git_commit':'engineering-metadata-only','files':list(sources.values())}
            study,counts,banks=make_study(root,custody,custody_path.parent,source,sources,args.runtime)
            by_id={r['run_id']:r for r in study['runs']}
            for bank in banks:
                if not bank['genuine_I4']: continue
                parents=[by_id[i] for i in bank['fit_parents']]
                assert [p['body'] for p in parents]==[0,1,2,3]
                assert len({p['config']['selected_config_from'] for p in parents})==1
                assert len({json.dumps(p['config'],sort_keys=True) for p in parents})==4 # seed only differs
                stripped=[{k:v for k,v in p['config'].items() if k!='initialization_seed'} for p in parents]
                assert all(c==stripped[0] for c in stripped)
                assert len({p['config']['initialization_seed'] for p in parents})==4
            results.append({'dataset':dataset,'complete_mask_genuine_I4_ancestry':'pass','counts':counts})
        if torch.cuda.is_initialized(): raise ValueError('CPU qualification initialized CUDA')
        result={'status':'passed','scope':'synthetic CPU native source engineering only; no actual data/input geometry; no accuracy qualification',
                'results':results,'runtime':{'torch':torch.__version__,'numpy':np.__version__},
                'runtime_receipts':[receipt(root,runtime/p) for p in ('adapters/native_polynormer_qk.py','engine.py','spec.py','compile_native_polynormer_qk_matrix.py')],
                'actual_optimizer_updates':0,'scientific_fits':0,'predictive_scores':0,'CUDA_initialized':False,'GO':False}
        if prior is not None:
            result['prior_check_receipt']=receipt(root,prior_path)
            result['prior_stage_manifest_receipt']=receipt(root,original_runtime.parent/'MANIFEST.json')
            result['inherited_constructor_checks']=8
        result['bare_vs_equipped_control_arithmetic']='Separate declared controls; bare uses fused affine bias, factor affine adds bias after scales; no bit-equality requirement.'
        (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'passed','constructor_arms':len(ARMS),'transition':'pass','engine_restore':'pass','independent4':'pass','fits':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (output/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed_checks':results,
                                                     'fits':0,'optimizer_updates':0,'GO':False},indent=2)+'\n')
        raise
    finally:
        torch.load=load_orig;torch.optim.Adam.step=step_orig


if __name__=='__main__': main()

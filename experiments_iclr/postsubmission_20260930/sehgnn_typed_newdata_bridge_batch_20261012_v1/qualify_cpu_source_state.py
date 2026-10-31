"""Synthetic full-width source constructor/gradient/state QA; deny all Adam."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation required before project operations')
    root=Path(args.project_root).resolve();stage=Path(__file__).resolve().parent;out=root/args.output
    if out.exists() or not out.resolve().is_relative_to(root):raise ValueError('Fresh confined synthetic CPU QA required')
    os.environ['CUDA_VISIBLE_DEVICES']='';out.mkdir();sys.path.insert(0,str(stage/'runtime'))
    import torch
    from source_utils import bind_source
    from adapters.sehgnn import make_adapter,SUPPLY
    from engine import inspect_ownership,capture_checkpoint,restore_checkpoint,_equal_tree
    helper_path=root/'private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py'
    if hashlib.sha256(helper_path.read_bytes()).hexdigest()!='effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0':raise ValueError('Exact reviewed denial helper required')
    spec=importlib.util.spec_from_file_location('sehgnn_denial',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    torch.set_num_threads(2);source=bind_source(root);custody=json.loads((root/'sehgnn_typed_newdata_custody_20261012_v2/CUSTODY.json').read_text())
    N=16;g=torch.Generator().manual_seed(71301)
    full={k:torch.randn(N,shape[-1],generator=g) for k,shape in custody['feature_shapes'].items()}
    protocol=json.loads((root/source['metadata']['protocol']['path']).read_text());labels={k:torch.randn(N,5,generator=g) for k in protocol['label_paths']}
    feature_views={'full':full};label_views={'full':labels}
    for family,symbol in (('actor','A'),('director','D'),('keyword','K')):
        feature_views[family]={k:torch.zeros_like(v) if symbol in k else v.clone() for k,v in full.items()}
        label_views[family]={k:torch.zeros_like(v) if symbol in k else v.clone() for k,v in labels.items()}
    y=(torch.arange(N)[:,None]+torch.arange(5)[None])%2
    fixture={'features':feature_views,'labels':label_views,'train':{'ids':torch.arange(8),'y':y[:8].float()},'select':{'ids':torch.arange(8,12),'y':y[8:12].float()}}
    arms=('native_single','factorized_native_single','original_gnnm','shared_own_only','ADD_PS','rank1_single_P','rank4_single_P')
    results=[];started=time.monotonic()
    try:
        for arm in arms:
            cfg={'sehgnn_arm':arm,'sehgnn_epochs':1000,'historical_state_inherited':False,'learning_rate':.001,'dropout':.5,'sehgnn_sources':source}
            study={'stage':'engineering','study_id':'synthetic_sehgnn_source_state','data':{'synthetic':True},'source':{'files':[]}}
            run={'run_id':'CPU_'+arm,'seed':81301,'body':0,'split':0,'config':cfg}
            adapter=make_adapter(torch=torch,project_root=root,study=study,run=run,device='cpu',execution_context={'engineering_synthetic':True,'fixture':fixture})
            inspect_ownership(adapter);parameters=sum(p.numel() for p in adapter.model.parameters());before={k:p.detach().clone() for k,p in adapter.model.named_parameters()}
            with helper.OptimizerDenial(torch) as denied:value=adapter.train_step(1)
            if denied.denied_step_calls!=1 or any(not torch.equal(before[k],p) for k,p in adapter.model.named_parameters()):raise ValueError('Synthetic source changed parameters or Adam owner count')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in adapter.model.parameters()):raise ValueError('Nonfinite source gradient')
            with torch.no_grad():z=adapter.row_logits_fn(torch.arange(N))
            if z.shape!=(4 if arm in ('original_gnnm','shared_own_only','ADD_PS') else 1,N,5) or not torch.isfinite(z).all():raise ValueError('Source full-width serving shape/finiteness differs')
            state=adapter.capture_state();payload=capture_checkpoint(torch,adapter,fit_id=run['run_id'],step=1,validation={},decision={});path=out/(arm+'_engineering_state.pt');torch.save(payload,path)
            adapter.session.counters[next(iter(adapter.session.counters))]+=7
            restore_checkpoint(torch,path,adapter,expected_fit_id=run['run_id'],verify=True)
            if not _equal_tree(torch,state,adapter.capture_state()):raise ValueError('Native member RNG/counter/scaler/engine state restore differs')
            path.unlink()
            results.append({'arm':arm,'full_native_width512_classes5_feature_channels25_label_channels12':True,'parameters':parameters,
                'source_constructor_gradient_serving_state':'pass','source_attempted_Adam_calls':denied.denied_step_calls,'actual_optimizer_updates':0})
            print(json.dumps({'source_CPU_case':arm,'parameters':parameters,'status':'passed','updates':0,'CUDA':False}),flush=True)
            del adapter,payload,before,state,z
            import gc;gc.collect()
        if torch.cuda.is_initialized():raise ValueError('CPU source QA initialized CUDA')
        result={'status':'passed','scope':'Synthetic16-row source engineering at actual native512/512/full25+12channels/5classes; no real input, warm teacher, scientific fit or predictive score',
            'cases':results,'elapsed_seconds':time.monotonic()-started,'source_module_bindings':source,'actual_optimizer_updates':0,'scientific_fits':0,'predictive_scores':0,'CUDA':False,'GO':False}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'passed','cases':len(results),'updates':0,'CUDA':False,'GO':False}))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'status':'failed','error':str(e),'traceback':traceback.format_exc(),'completed':results,'actual_optimizer_updates':0,'scientific_fits':0,'GO':False},indent=2)+'\n');raise


if __name__=='__main__':main()

"""Unused engineering helper: exact NCN/NCNC CUDA forward/backward, zero updates.

Opens only the frozen study JSON and hash-bound GRAPH/TRAIN packs. Synthetic
signs on TRAIN queries exercise decoder/backward; no labels are fetched from
SELECT or TEST. This is not a complete epoch or optimizer-storage profile.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import socket
import subprocess
import sys
import time


def construct_native(native,features,config):
    e,p=config['encoder'],config['predictor']
    encoder=native.GCN(features,e['hidden'],e['hidden'],e['layers'],e['dropout'],
        ln=e['ln'],res=e['res'],max_x=-1,conv_fn='gcn',jk=e['jk'],
        edrop=e['edrop'],xdropout=e['xdropout'],taildropout=e['taildropout'])
    cls=native.CNLinkPredictor if p['name']=='NCN' else native.IncompleteCN1Predictor
    options={key:p[key] for key in ('edrop','ln','cndeg','use_xlin','tailact','twolayerlin','beta')}
    if p['name']=='NCNC':options.update(p['completion'])
    decoder=cls(e['hidden'],e['hidden'],1,p['layers'],p['dropout'],**options)
    return encoder,decoder


def bind_metadata(root,path):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():raise ValueError('Confine qualifier inputs')
    raw=path.read_bytes()
    return json.loads(raw),{'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def qualify(root,study,run,output):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch_sparse import SparseTensor
    from adapters.link import bound_file,native_modules,pair_keys,validate_config
    config=validate_config(run['config'],run)
    if run['config'].get('stage')!='search' or 'selected_config_from' in config:raise ValueError('Probe an exact frozen search constructor; no selected ancestry guessing')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.get_default_dtype()!=torch.float32:
        raise ValueError('Require the sole allocation CUDA GPU and float32 native arithmetic')
    result={'status':'started','engineering_only':True,'run_id':run['run_id'],'config':config,
        'optimizer_created':False,'optimizer_updates':0,'scientific_fits':0,'scientific_metrics_computed':0,
        'SELECT_pack_opened':False,'TEST_query_or_target_pack_opened':False,'synthetic_target_law':'fixed random Rademacher signs on two TRAIN query panels',
        'not_full_epoch_or_optimizer_storage_qualification':True,'scientific_launch_enabled':False,
        'operation_counts':{'encoder_attempts':0,'encoder_completed':0,'decoder_attempts':0,'decoder_completed':0,'backward_attempts':0,'backward_completed':0}}
    tick=time.monotonic();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    result['baseline_allocated_bytes']=torch.cuda.memory_allocated();result['baseline_reserved_bytes']=torch.cuda.memory_reserved()
    try:
        inputs={role:torch.load(bound_file(root,config['link_inputs'][role]),map_location='cpu',weights_only=True) for role in ('graph','train')}
        if set(inputs['graph'])!={'x','raw_edge_index'} or set(inputs['train'])!={'positive'}:raise ValueError('GRAPH/TRAIN schema changed')
        x,raw,pairs=inputs['graph']['x'],inputs['graph']['raw_edge_index'],inputs['train']['positive']
        if x.dtype!=torch.float32 or x.ndim!=2 or not bool(torch.isfinite(x).all()):raise ValueError('Complete finite float32 features required')
        if pairs.dtype!=torch.long or pairs.ndim!=2 or pairs.shape[1]!=2 or int(pairs.min())<0 or int(pairs.max())>=len(x):raise ValueError('Complete TRAIN pair shape/range required')
        if raw.dtype!=torch.long or raw.shape!=(2,2*len(pairs)):raise ValueError('Complete bidirectional TRAIN multiset required')
        expected=torch.cat((pairs,pairs.flip(1)),0)
        if not torch.equal(pair_keys(torch,raw.T,len(x),undirected=False).sort().values,pair_keys(torch,expected,len(x),undirected=False).sort().values):raise ValueError('GRAPH is not exact TRAIN support')
        n=len(pairs);batch=config['training']['batch_size']
        if n<batch:raise ValueError('Native dropped-tail batch would be empty')
        seed=config['initialization_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        native,utils=native_modules(root,config['link_source']);encoder,decoder=construct_native(native,x.shape[1],config)
        encoder=encoder.cuda();decoder=decoder.cuda();encoder.train();decoder.train()
        x=x.cuda();raw=raw.cuda();pairs=pairs.cuda()
        def support(values):return SparseTensor.from_edge_index(values.T,sparse_sizes=(len(x),len(x))).to_device(values.device,non_blocking=True).to_symmetric()
        full=support(pairs)  # Retain the fixed serving graph, as the fit adapter does.
        ids=next(iter(utils.PermIterator('cuda',n,batch,True)))
        keep=torch.ones(n,dtype=torch.bool,device='cuda');keep[ids]=False;masked=support(pairs[keep])
        query1=pairs[ids].T;query2=pairs[ids.flip(0)].T
        signs1=torch.randint(0,2,(batch,),device='cuda').float().mul_(2).sub_(1)
        signs2=torch.randint(0,2,(batch,),device='cuda').float().mul_(2).sub_(1)
        parameters=list(encoder.parameters())+list(decoder.parameters());versions=[p._version for p in parameters]
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        counts=result['operation_counts'];counts['encoder_attempts']+=1;h=encoder(x,masked);counts['encoder_completed']+=1
        counts['decoder_attempts']+=1;a=decoder.multidomainforward(h,masked,query1,cndropprobs=[]).flatten();counts['decoder_completed']+=1
        counts['decoder_attempts']+=1;b=decoder.multidomainforward(h,masked,query2,cndropprobs=[]).flatten();counts['decoder_completed']+=1
        if a.shape!=(batch,) or b.shape!=(batch,) or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('Native query coverage/nonfinite scores')
        loss=-F.logsigmoid(a*signs1).mean()-F.logsigmoid(b*signs2).mean()
        counts['backward_attempts']+=1;loss.backward();counts['backward_completed']+=1;end.record();torch.cuda.synchronize()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Nonfinite native resource gradients')
        if versions!=[p._version for p in parameters]:raise ValueError('A model parameter changed during the zero-update probe')
        result.update(status='complete',nodes=len(x),TRAIN_records=n,batch_records=batch,synthetic_query_records=2*batch,
            CUDA_forward_backward_seconds=start.elapsed_time(end)/1000.,parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
            gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in parameters if p.grad is not None),parameter_versions_unchanged=True,
            input_pack_receipts={role:config['link_inputs'][role] for role in ('graph','train')})
    except BaseException as error:
        result.update(status='failed',failure_type=type(error).__name__,failure=str(error),operation_counts_are_completed_lower_bounds=True)
        raise
    finally:
        result.update(wall_seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (Path(output)/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','study','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--runtime',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise RuntimeError('Verify the exact sole-GPU allocation before project access')
    root=args.project_root.resolve();output=args.output.resolve();runtime=(root/args.runtime).resolve()
    if not runtime.is_relative_to(root) or not output.is_relative_to(root) or output.exists():raise ValueError('Use confined frozen runtime and fresh probe output')
    study,source=bind_metadata(root,args.study);rows=[r for r in study['runs'] if r['run_id']==args.run_id]
    if len(rows)!=1 or rows[0]['kind']!='model_fit':raise ValueError('Select one exact frozen search model constructor')
    sys.path.insert(0,str(runtime));output.mkdir(parents=True)
    (output/'STUDY_RECEIPT.json').write_text(json.dumps(source,indent=2)+'\n')
    result=qualify(root,study,rows[0],output)
    print(json.dumps({k:result[k] for k in ('status','optimizer_updates','peak_allocated_bytes','peak_reserved_bytes','CUDA_forward_backward_seconds')}))


if __name__=='__main__':main()

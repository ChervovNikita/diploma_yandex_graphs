"""One real resource trajectory, metrics private; never a scientific fit."""
import argparse
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from guard import admit,identity,json_write,sha,ROOT,SUITE
from source_ops import bootstrap,execute,contract


def main():
    started=time.monotonic();os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args()
    job,config,output,live,runtime,_=admit(args.job,worker=True)
    output.mkdir(mode=0o700)
    progress={'stage':'runtime_startup','completed_modes':[],'TRAIN_updates':0}
    torch=None;ns=None
    try:
        versions=runtime.runtime_versions()
        import numpy as np
        import torch
        from ogb.linkproppred import Evaluator as LinkEvaluator
        from ogb.graphproppred import Evaluator as GraphEvaluator
        sys.path.insert(0,str(SUITE))
        for name in ('models','factors','data','objectives','selection'):
            loaded=sys.modules.get(name)
            if loaded is not None and Path(loaded.__file__).resolve()!=SUITE/(name+'.py'):
                raise ValueError('Conflicting suite module: '+name)
        from models import Ensemble,native_sources
        from data import load_projection,batches,valid_batches,bound
        from objectives import objective,own_supervision
        from factors import factor_counts
        from selection import local_transition,finite_predictions,finite_state
        torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
        if torch.cuda.device_count()!=1:raise ValueError('One exact visible singleton CUDA device')
        device=torch.cuda.get_device_properties(0)
        ns=dict(job=job,config=config,output=output,PHASE=runtime.PHASE,ROOT=SUITE,
            json=json,random=random,np=np,torch=torch,LinkEvaluator=LinkEvaluator,
            GraphEvaluator=GraphEvaluator,Ensemble=Ensemble,native_sources=native_sources,
            load_projection=load_projection,batches=batches,valid_batches=valid_batches,
            bound=bound,objective=objective,own_supervision=own_supervision,
            factor_counts=factor_counts,local_transition=local_transition,
            finite_predictions=finite_predictions,finite_state=finite_state)
        progress['stage']='data_support_model_optimizer_RNG_startup'
        selected=bootstrap(SUITE/'run.py',ns)
        model,opts,train,valid=ns['model'],ns['opts'],ns['train'],ns['valid']
        if any(p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
            raise ValueError('Exact float32 model dtype; no autocast/precision substitution')
        finite_state(model,opts)
        counters={};original_forward=ns['forward'];phase='startup';mode='local' if job['task']=='wikics' else 'native'
        def measured_forward(batch):
            logits,representation=original_forward(batch)
            row=counters.setdefault(mode,{}).setdefault(phase,{'member_forwards':0,'calls':0,'objects_per_member':0,'batch_shapes':[]})
            row['member_forwards']+=model.members;row['calls']+=1;row['objects_per_member']+=logits.shape[1]
            shape={'objects':logits.shape[1]}
            if 'x' in batch:shape['graph_nodes']=len(batch['x'])
            if 'edge_index' in batch:shape['graph_edges']=batch['edge_index'].shape[1]
            if 'adj' in batch:shape['support_nnz']=batch['adj'].nnz()
            if 'graph' in batch:
                shape.update(graphs=int(batch['graph'].num_graphs),graph_nodes=int(batch['graph'].num_nodes),
                    graph_edges=int(batch['graph'].edge_index.shape[1]))
            row['batch_shapes'].append(shape)
            return logits,representation
        ns['forward']=measured_forward
        molecular_scope=None
        if job['task']=='molhiv':
            molecular_scope={'TRAIN_batch_policy':'first full 128 of exact source epoch1 shuffled grouping',
                'worst_later_TRAIN_batch_qualified':False,'population':{}}
            for role,payload in [('TRAIN',train),('VALID',valid)]:
                nodes=payload['node_ptr'].diff();edges=payload['edge_ptr'].diff()
                molecular_scope['population'][role]={'graphs':len(payload['y']),'nodes':int(nodes.sum()),
                    'edges':int(edges.sum()),'max_single_graph_nodes':int(nodes.max()),'max_single_graph_edges':int(edges.max())}
        modes=['local','global'] if job['task']=='wikics' else ['native']
        for mode in modes:
            epoch=1 if mode!='global' else config['training']['local_epochs']+1
            if mode=='global':
                # Source helper restores the resource local snapshot and Adam,
                # retaining current live RNG, then switches the native phase.
                local_transition(model,opts,lambda name:torch.load(output/name,map_location='cuda:0',weights_only=False),ns['ordinary_independent'])
            progress['stage']=mode+'_TRAIN_update';phase='TRAIN';model.train()
            generator=batches(job['task'],train,config['training'],epoch,job['seed'],'cuda:0')
            batch,y=next(generator)
            expected=580 if job['task']=='wikics' else (131072 if job['task']=='collab' else 128)
            if len(y)!=expected:raise ValueError('No shortened representative TRAIN batch')
            if job['task']=='collab' and (len(batch['x'])!=235868 or int((y==1).sum())!=65536 or int((y==0).sum())!=65536):
                raise ValueError('Complete Collab graph/full 65536 positive target plus negative batch')
            if job['task']=='molhiv':
                ptr=batch['graph'].ptr.diff();edges=batch['graph'].edge_index
                molecular_scope['measured_TRAIN_batch']={'graphs':128,'nodes':int(batch['graph'].num_nodes),
                    'edges':int(edges.shape[1]),'max_single_graph_nodes':int(ptr.max())}
            ns.update(batch=batch,y=y,epoch=epoch)
            execute(selected['TRAIN_batch'],ns,SUITE/'run.py')
            # Finite gradients are checked in the unmodified step before Adam;
            # finite parameters and optimizer moments are checked after Adam.
            progress['TRAIN_updates']+=1
            counters[mode]['backward_calls']=1;counters[mode]['optimizer_steps']=len(opts)
            del generator,batch,y
            progress['stage']=mode+'_complete_VALID';phase='VALID'
            metric,per=ns['evaluate']()  # Original complete evaluator, including per-member workloads.
            expected_valid=5274 if job['task']=='wikics' else (160084 if job['task']=='collab' else 4113)
            if counters[mode]['VALID']['objects_per_member']!=expected_valid:
                raise ValueError('Incomplete VALID forward/evaluator population')
            progress['stage']=mode+'_checkpoint_serialization'
            # Fixed first-save sentinel branches exercise the exact checkpoint
            # code; these snapshots are private resource artifacts, not selectors.
            ns.update(metric=metric,per=per,best=-float('inf'),best_local=-float('inf'),
                own_best=[-float('inf')]*model.members,own_local=[-float('inf')]*model.members)
            execute(selected['checkpoint_branches'],ns,SUITE/'run.py')
            # Source save helper packs complete model/Adam/RNG even for the
            # ordinary independent bank; charge this conservative output work.
            ns['save'](output/('RESOURCE_STATE_'+mode+'.pt'),epoch,metric,per)
            json_write(output/('CLOSED_PREDICTIVE_'+mode+'.json'),{'VALID':metric,'members':per,'resource_only':True})
            progress['completed_modes'].append(mode)
            torch.cuda.synchronize()
            json_write(output/'PROGRESS.json',{**progress,'predictive_scores_closed':True})
        progress['stage']='output_accounting'
        files=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
        torch.cuda.synchronize()
        candidate={'schema':'internal-be-resource-worker-candidate-v1','worker_completed':True,'passed':False,
            'qualifier_identity':identity(job,runtime),'cell_identity':runtime.cell_identity(job),'seed':job['seed'],
            'supervisor_receipt_sha256':sha(live),'worker_pid':os.getpid(),'worker_start_ticks':runtime.proc_start(os.getpid()),
            'versions':versions,'runtime_pin_sha256':sha(SUITE/'RUNTIME_PIN.json'),
            'cuda_device':{'visible_index':0,'name':device.name,'total_memory_bytes':device.total_memory,
                'capability':[device.major,device.minor],'physical_uuid_from_exact_allocation_guard':runtime.GPU},
            'parameters':factor_counts(model),'finite_parameters_gradients_optimizer_outputs':True,
            'source_scope_contract':contract(SUITE/'run.py'),
            'work':{'two_view_TRAIN_backward_Adam':True,'complete_VALID_evaluation':True,'checkpoint_serialization':True,
                'predictive_scores_closed':True,'members':model.members,'own_views':2,'modes':modes,
                'committed_TRAIN_updates':progress['TRAIN_updates'],'counters':counters,'molecular_scope':molecular_scope,
                'TRAIN_batch_is_representative_not_complete_epoch':True,'resource_only_not_fit':True},
            'peak_CUDA_allocated_bytes':torch.cuda.max_memory_allocated(0),
            'peak_CUDA_reserved_bytes':torch.cuda.max_memory_reserved(0),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'artifact_storage_bytes':sum(x['bytes'] for x in files),'closed_artifacts':files,
            'worker_inclusive_seconds':time.monotonic()-started,'TEST_access':False,'automatic_retry':False}
        json_write(output/'WORKER_RESOURCE.json',candidate)
        json_write(output/'WORKER_OUTPUT_CUSTODY.json',{'WORKER_RESOURCE_sha256':sha(output/'WORKER_RESOURCE.json'),
            'worker_seconds_through_candidate_write':time.monotonic()-started,'predictive_scores_closed':True})
    except Exception as error:
        failure={'schema':'internal-be-resource-worker-failure-v1','worker_completed':False,'passed':False,
            'error_type':type(error).__name__,'stage':progress,'seed':job['seed'],
            'worker_pid':os.getpid(),'worker_start_ticks':runtime.proc_start(os.getpid()),
            'qualifier_identity':identity(job,runtime),'worker_inclusive_seconds':time.monotonic()-started,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'predictive_scores_closed':True,'automatic_retry':False,'TEST_access':False}
        if torch is not None and torch.cuda.is_initialized():
            failure.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(0))
        json_write(output/'FAILURE.json',failure)
        raise


if __name__=='__main__':main()

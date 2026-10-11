"""Sole full-input GM4 update, checkpoint0 restore, complete SELECT replay."""
import math
from pathlib import Path
import time
from runtime import BLOCKS,Runtime,receipt,require,write

def evaluate(rt,model,roles):
    model.eval()
    with rt.t.no_grad():
        logits=rt.forward(model)
        require(logits.shape==(4,34493,5) and rt.t.isfinite(logits).all().item(),'Finite complete four-view SELECT forward')
        ce=rt.t.stack([rt.F.cross_entropy(z[roles['select_ids']],roles['select_y']) for z in logits]).mean()
        penalty=model.regularizer();values=[float(v.item()) for v in (ce+penalty,ce,penalty)]
        require(all(math.isfinite(v) for v in values),'Finite SELECT total/CE/native penalty')
    del logits,ce,penalty;rt.guard();return values

def train_update(rt,model,optimizer,streams,roles):
    model.train();optimizer.zero_grad(set_to_none=True);own=[]
    for view in range(4):
        logits=rt.view(model,view,streams)
        require(logits.shape==(34493,5) and rt.t.isfinite(logits).all().item(),'Complete live full-input view')
        ce=rt.F.cross_entropy(logits[roles['train_ids']],roles['train_y'])
        require(rt.t.isfinite(ce).item(),'Finite own TRAIN CE');own.append(float(ce.item()))
        (ce/4).backward();del logits,ce;rt.guard()
    penalty=model.regularizer();value=float(penalty.item());require(math.isfinite(value),'Finite native first-layer penalty')
    if penalty.requires_grad:penalty.backward()
    live=dict((n,p) for n,p in model.named_parameters() if p.requires_grad)
    require(all(p.grad is not None and rt.t.isfinite(p.grad).all().item() for p in live.values()),'Every live parameter gradient finite, including disclosed zero-start gates')
    gradients={n:dict(nonzero=bool((p.grad!=0).any().item()),max_absolute=float(p.grad.abs().max().item())) for n,p in live.items()}
    require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'Frozen first-layer parameters have no gradients')
    optimizer.step();rt.guard()
    require(optimizer.state['tf1_global']['updates']==1 and all(p in optimizer.state and optimizer.state[p] for p in live.values()),'Exactly one source Adam update with all live moments')
    return dict(mean_own_CE=sum(own)/4,native_penalty=value,total=sum(own)/4+value,view_CE=own,gradients=gradients,streamed_view_backwards=4,regularizer_backward=int(penalty.requires_grad),Adam_steps=1)

def run(config,custody):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);rt=None
    result=dict(stage='qualify',complete=False,status='running',fits=0,TRAIN_updates=0,Adam_steps=0,SELECT_checkpoints=0,SELECT_labels_read=True,CAL_DEV_truth_read=False,TEST_read=False,retry=False,automatic_fit_release=False,custody=custody,role_custody=config['custody'],source_manifest=config['source_manifest'],derivative=config['derivative'],derivative_owner_end=config['derivative_owner_end'])
    try:
        require(config.get('qualification_block')==BLOCKS[0] and config.get('qualification_arm')=='GM4','One fixed full-input GM4 block10103 qualification')
        rt=Runtime(config,output);roles=rt.roles(BLOCKS[0]);result.update(graph=rt.graph,tour_bindings=rt.tour_bindings,packing_bindings=rt.packing_bindings,canonical_CPU_capsule=rt.capsule_receipt,canonical_CPU_capsule_fingerprint=rt.capsule_fingerprint,reconstruction_and_packing_seconds=rt.packing_seconds)
        ancestor,hidden=rt.inherited_P1(BLOCKS[0]);result['inherited_P1']=rt.ancestor_records
        core,optimizers,audit=rt.construct('GM4',BLOCKS[0],[ancestor],[hidden]);model=core['models'][0];optimizer=optimizers[0];streams=rt.streams(BLOCKS[0],0)
        require(model.fixed_H is hidden,'Shared immutable H registered without copy')
        require(optimizer.state['tf1_global']['updates']==0 and len(optimizer.state)==1,'Fresh Adam checkpoint0 has empty per-parameter moments')
        result['tail_seed_law']='block+459000151+1000003*b+1009*j';result['initial_private_stream_fingerprints']=[rt.fp(s) for s in streams]
        tick=time.monotonic();select0=evaluate(rt,model,roles);result['checkpoint0_SELECT_seconds']=time.monotonic()-tick;result['SELECT_checkpoints']=1
        meta=dict(block=BLOCKS[0],arm='GM4',body=0,selected_step=0,SELECT_total=select0[0],SELECT_CE=select0[1],SELECT_L2=select0[2],role_custody=config['custody'],source_manifest=config['source_manifest'],core_manifest_sha256=custody['core_manifest_sha256'],H_fingerprint=rt.fp(hidden),qualification_only=True)
        path=output/'checkpoint0.pt';digest=rt.save(path,model,optimizer,streams,meta);result['checkpoint0']=receipt(path);result['checkpoint0_fingerprint']=digest;result['checkpoint0_SELECT']=select0
        global_before=rt.fp(rt.rng());tick=time.monotonic()
        with rt.t.profiler.profile(activities=[rt.t.profiler.ProfilerActivity.CPU]) as profile:
            actual=train_update(rt,model,optimizer,streams,roles)
        result['actual_TRAIN']=actual;result['TRAIN_seconds']=time.monotonic()-tick;result['TRAIN_updates']=1;result['Adam_steps']=1
        require(rt.fp(rt.rng())==global_before,'Private tail training altered outside global RNG')
        operations={event.key:event.count for event in profile.key_averages() if 'gru' in event.key or 'cudnn_rnn' in event.key}
        result['backend_operations']=operations;result['runtime']=dict(torch=rt.t.__version__,numpy=rt.np.__version__,cudnn=rt.t.backends.cudnn.version())
        require(operations.get('aten::gru')==8 and operations.get('aten::_cudnn_rnn')==8 and operations.get('aten::_cudnn_rnn_backward')==8,'Actual complete bidirectional four-tour cuDNN TRAIN dispatch')
        del profile
        with rt.t.no_grad():
            for parameter in model.parameters():parameter.add_(1)
            for values in optimizer.state.values():
                for value in values.values():
                    if rt.t.is_tensor(value):value.add_(1)
            optimizer.state['tf1_global']['updates']+=7;optimizer.param_groups[0]['lr']=.7
        for module in model.modules():module.training=not module.training
        streams[:]=[rt.fresh_stream(101+view) for view in range(4)];rt.restore_rng(rt.fresh_stream(997))
        result['poisoned_state_differs']=rt.fp(rt.state(model,optimizer,streams,meta))!=digest
        require(result['poisoned_state_differs'],'Restore challenge must change state')
        tick=time.monotonic();restored=rt.restore(path,model,optimizer,streams,meta)
        require(restored==digest and optimizer.state['tf1_global']['updates']==0 and len(optimizer.state)==1,'Exact checkpoint0 optimizer restore, including empty moments')
        result['strict_model_optimizer_all_module_modes_global_and_private_rng_restore']=True;result['restored_Adam_updates']=0;result['restored_per_parameter_moment_states']=0
        replay=evaluate(rt,model,roles);result['restored_SELECT_seconds']=time.monotonic()-tick;result['restored_SELECT']=replay;result['SELECT_checkpoints']=2
        result['SELECT_replay_delta']=replay[0]-select0[0]
        require(math.isclose(replay[0],select0[0],rel_tol=1e-5,abs_tol=1e-7),'Restored SELECT total within inherited disclosed reduction tolerance')
        result['SELECT_replay_tolerance']=dict(rel_tol=1e-5,abs_tol=1e-7);rt.guard()
        result.update(complete=True,status='pass',construction=audit,cache_immutable=True,meaning='One actual full-input source/resource/restore qualification; no scientific utility claim')
    except BaseException as error:
        result.update(status='failed_retained',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if rt is not None:result['costs']=rt.costs()
        write(output/'QUALIFICATION.json',result)
    return result

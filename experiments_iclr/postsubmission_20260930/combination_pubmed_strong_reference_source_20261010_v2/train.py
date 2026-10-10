"""Fresh honest native references; only a new separately reviewed root release runs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
from admission import admit
from source import bind, fingerprint, sha, write
from reference_plan import description, body_roster, SERVING


def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime,r.ru_stime


def timed(torch,call):
    started=time.monotonic();user,system=cpu()
    torch.cuda.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();value=call();end.record();torch.cuda.synchronize()
    u,s=cpu()
    return value,dict(wall_seconds=time.monotonic()-started,CUDA_seconds=begin.elapsed_time(end)/1000,
                      CPU_user_seconds=u-user,CPU_system_seconds=s-system)


def add(timings,phase,row):
    target=timings.setdefault(phase,{k:0. for k in row})
    for k,v in row.items():target[k]+=v


def check_bounds(spec,output,started):
    if time.monotonic()-started >= spec['limits']['external_active_seconds']:
        raise TimeoutError('Frozen active wall cap; no shortened completion')
    total=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > spec['limits']['output_bytes']:raise RuntimeError('Frozen scientific output cap')


def runtime_and_data(spec,valid=True):
    import numpy as np
    import torch
    providers=dict(torch=str(torch.__version__),numpy=str(np.__version__))
    for name in ('torch-geometric','scipy','torch-scatter','torch-sparse'):
        providers[name]=importlib.metadata.version(name)
    if providers != spec['frozen_providers']:raise ValueError('Exact actual qualification providers required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
    with np.load(bind(spec['train_bundle']),allow_pickle=False) as archive:
        if set(archive.files) != {'x','edge_index','train_ids','train_y'}:raise ValueError('Only four TRAIN arrays admitted')
        arrays={k:archive[k].copy() for k in archive.files}
    if {k:fingerprint(v) for k,v in arrays.items()} != spec['_train_custody']['array_fingerprints']:
        raise ValueError('Exact TRAIN fingerprints required')
    if arrays['x'].dtype != np.float32 or arrays['x'].shape != (19717,500) or not np.isfinite(arrays['x']).all():
        raise ValueError('Complete finite float32 normalized factual features required')
    if arrays['edge_index'].dtype != np.int64 or arrays['edge_index'].shape != (2,88648):raise ValueError('Complete ordered factual edges required')
    roles={}
    sources=[('train',arrays,11829,[2461,4643,4725])]
    if valid:
        with np.load(bind(spec['valid_bundle']),allow_pickle=False) as archive:
            if set(archive.files) != {'valid_ids','valid_y'}:raise ValueError('Only projected VALID arrays admitted')
            validation={k:archive[k].copy() for k in archive.files}
        if {k:fingerprint(v) for k,v in validation.items()} != spec['_validation_custody']['array_fingerprints']:
            raise ValueError('Exact VALID fingerprints required')
        sources.append(('valid',validation,3942,[820,1547,1575]))
    for name,source,n,counts in sources:
        ids,y=source[name+'_ids'],source[name+'_y']
        if ids.dtype != np.int64 or y.dtype != np.int64 or ids.shape != (n,) or y.shape != ids.shape or len(np.unique(ids)) != n or ids.min()<0 or ids.max()>=19717:
            raise ValueError('Complete unique exact role IDs/labels required')
        if y.min()<0 or y.max()>=3 or np.bincount(y,minlength=3).tolist() != counts:raise ValueError('Frozen complete role class populations required')
        roles[name.upper()]=(torch.from_numpy(ids).to('cuda:0'),torch.from_numpy(y).to('cuda:0'))
    if valid and np.intersect1d(arrays['train_ids'],validation['valid_ids']).size:raise ValueError('Role overlap')
    tensor={k:torch.from_numpy(v) for k,v in arrays.items()}
    return np,torch,providers,tensor,roles


def fit_body(spec,body_spec,tensor,roles,output,started,timings):
    from adapter import fresh_single
    from metrics import classification,signatures,verify_counts,verify_floats
    import torch
    output.mkdir(exist_ok=False)
    session,clock=timed(torch,lambda:fresh_single(spec['condition'],body_spec['initialization_seed'],tensor))
    add(timings,'fit_construction_and_preprocessing',clock)
    best_correct,best_epoch,best_metrics,best_signatures,epochs=-1,None,None,None,0
    writes=0;path=output/'SELECTED_PREDICTOR.pt'
    views=4 if spec['condition']=='single_mean4_dropout' else 1
    with (output/'EPOCHS.jsonl').open('x') as trace:
        for epoch in range(1,2001):
            check_bounds(spec,output.parent,started)
            training,clock=timed(torch,lambda:session.train_step(audit=False));add(timings,'native_updates',clock)
            (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'regular_VALID_forwards',clock)
            tick=time.monotonic();metrics=classification(logits,probabilities,*roles['VALID'])
            # Author-native raw-logit maximum, never a pooled selector.
            correct=int(logits[0,roles['VALID'][0]].max(1)[1].eq(roles['VALID'][1]).sum().item())
            if correct != metrics['members'][0]['correct']:raise ValueError('Native selector count differs')
            add(timings,'regular_VALID_metrics',dict(wall_seconds=time.monotonic()-tick))
            improved=correct > best_correct
            if improved:
                best_correct,best_epoch,best_metrics=correct,epoch,metrics
                best_signatures=signatures(logits,probabilities,*roles['VALID'])
                tick=time.monotonic()
                selected=dict(schema='PubMed-own-selected-native-body-v1',record_id=spec['record_id'],
                              body_id=body_spec['body_id'],initialization_seed=body_spec['initialization_seed'],
                              factual_dropout_seeds=body_spec['factual_dropout_seeds'],selected_epoch=epoch,
                              selector=spec['selector'],selected_VALID=metrics,selected_prediction_signatures=best_signatures,
                              body={k:v.detach().cpu().clone() for k,v in session.bodies[0].state_dict().items()},
                              resumable=False,optimizer_history_included=False,labels_included=False)
                temporary=output/'SELECTED_PREDICTOR.partial.pt';torch.save(selected,temporary);os.replace(temporary,path)
                writes+=1;del selected
                add(timings,'checkpoint_write',dict(wall_seconds=time.monotonic()-tick))
            epochs=epoch
            tick=time.monotonic()
            trace.write(json.dumps(dict(epoch=epoch,TRAIN=training,VALID=metrics,selected=improved,best_epoch=best_epoch),allow_nan=False)+'\n');trace.flush()
            write(output.parent/'PROGRESS.json',dict(complete=False,record_id=spec['record_id'],body_id=body_spec['body_id'],
                  private_epoch=epoch,private_best_epoch=best_epoch,private_max_epochs=2000,
                  counters=session.counters,inclusive_seconds=time.monotonic()-started))
            add(timings,'trace_and_progress_write',dict(wall_seconds=time.monotonic()-tick))
            del probabilities,logits
            if epoch-best_epoch >=250:break
    def restore():
        selected=torch.load(path,map_location='cpu')
        for k,v in dict(record_id=spec['record_id'],body_id=body_spec['body_id'],initialization_seed=body_spec['initialization_seed'],selected_epoch=best_epoch,selector=spec['selector']).items():
            if selected.get(k)!=v:raise ValueError('Own-selected body identity changed')
        if selected['selected_prediction_signatures'] != best_signatures:raise ValueError('Selected signature custody changed')
        session.bodies[0].load_state_dict(selected['body'],strict=True)
        return selected
    selected,clock=timed(torch,restore);add(timings,'private_selected_restore',clock)
    (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'private_selected_forwards',clock)
    metrics=classification(logits,probabilities,*roles['VALID'])
    signature=signatures(logits,probabilities,*roles['VALID'])
    if signature != best_signatures:raise ValueError('Restored exact native predictions/correct masks changed')
    verify_counts(metrics,best_metrics);verify_floats(metrics,best_metrics)
    if session.counters != dict(updates=epochs,factual_forwards=epochs*(views+1)+1,masked_forwards=0,
                                backwards=epochs*views,optimizer_steps=epochs,serving_forwards=epochs+1,preprocessing_banks=1):
        raise ValueError('Complete private native work counters differ')
    row=dict(**body_spec,epochs_executed=epochs,selected_epoch=best_epoch,
             stopped_by='patience250' if epochs-best_epoch>=250 else 'max2000',
             selected_VALID=metrics,selected_prediction_signatures=signature,counters=dict(session.counters),
             preprocessing_seconds=session.preparation_seconds,selected_checkpoint_writes=writes,
             selected_predictor=dict(path=str(path.relative_to(output.parent)),sha256=sha(path),bytes=path.stat().st_size),
             own_selector=True,common_bank_selector=False)
    write(output/'BODY_COMPLETE.json',row)
    return session,logits,selected['body'],row


def serving_benchmark(session,expected_signatures,roles,timings):
    import torch
    from metrics import signatures
    warmup,timed_rows=[],[]
    for i in range(SERVING['warmup_calls']+SERVING['measured_calls']):
        result,clock=timed(torch,session.factual_probabilities)
        probabilities,logits=result
        if signatures(logits,probabilities,*roles['VALID']) != expected_signatures:
            raise ValueError('Serving benchmark changed selected predictions')
        if i<SERVING['warmup_calls']:
            warmup.append(clock);add(timings,'serving_warmup',clock)
        else:
            timed_rows.append(clock);add(timings,'serving_measured',clock)
        del probabilities,logits
    return dict(protocol=SERVING,warmup=warmup,measured=timed_rows,
                mean={k:sum(r[k] for r in timed_rows)/len(timed_rows) for k in timed_rows[0]},
                includes_full_graph_native_predictor_and_probability_pool=True,
                excludes_construction_restore_preprocessing=True,
                separate_construction_restore_preprocessing_costs_recorded=True)


def fit(spec,output,started):
    timings={};bodies=[];session=None
    output.mkdir(parents=True,exist_ok=False)
    try:
        tick=time.monotonic();u,s=cpu()
        np,torch,providers,tensor,roles=runtime_and_data(spec)
        u2,s2=cpu();add(timings,'numeric_imports_and_input',dict(wall_seconds=time.monotonic()-tick,CPU_user_seconds=u2-u,CPU_system_seconds=s2-s))
        from adapter import assembled_I4
        from metrics import classification,repair_diagnostics,signatures
        states=[]
        for body_spec in spec['body_roster']:
            session,logits,state,row=fit_body(spec,body_spec,tensor,roles,output/body_spec['body_id'],started,timings)
            bodies.append(row)
            if spec['condition']=='independent4_own':
                states.append(state);del session,logits;session=None
        assembly_counters=None;assembly_preprocessing=0.
        if spec['condition']=='independent4_own':
            session,clock=timed(torch,lambda:assembled_I4(spec['seed'],tensor,states));add(timings,'I4_assembly_construction_preprocessing_restore',clock)
            assembly_preprocessing=session.preparation_seconds;del states
            (probabilities,logits),clock=timed(torch,session.factual_probabilities);add(timings,'assembled_selected_forwards',clock)
            for m,row in enumerate(bodies):
                own=logits[m:m+1];own_prob=own.softmax(-1).mean(0)
                if signatures(own,own_prob,*roles['VALID']) != row['selected_prediction_signatures']:
                    raise ValueError('I4 assembly changed an individually selected native body')
        else:probabilities=logits.softmax(-1).mean(0)
        tick=time.monotonic()
        readouts={name:dict(classification=classification(logits,probabilities,*role),
                            repair=repair_diagnostics(logits,probabilities,*role),
                            prediction_signatures=signatures(logits,probabilities,*role)) for name,role in roles.items()}
        add(timings,'selected_TRAIN_VALID_readouts',dict(wall_seconds=time.monotonic()-tick))
        benchmark=serving_benchmark(session,readouts['VALID']['prediction_signatures'],roles,timings)
        if spec['condition']=='independent4_own':
            assembly_counters=dict(session.counters)
            if assembly_counters != dict(updates=0,factual_forwards=56,masked_forwards=0,backwards=0,optimizer_steps=0,serving_forwards=56,preprocessing_banks=1):
                raise ValueError('Fresh I4 serving assembly must never train')
        else:
            extra=SERVING['warmup_calls']+SERVING['measured_calls']
            if session.counters['updates'] != bodies[0]['epochs_executed'] or session.counters['serving_forwards'] != bodies[0]['epochs_executed']+1+extra:
                raise ValueError('Single serving benchmark counter mismatch')
        tick=time.monotonic();payload=output/'SELECTED_MEMBER_LOGITS.npz'
        np.savez_compressed(payload,factual_member_logits=logits.detach().cpu().numpy())
        add(timings,'logit_payload_write',dict(wall_seconds=time.monotonic()-tick))
        check_bounds(spec,output,started)
        members=4 if spec['condition']=='independent4_own' else 1
        result=dict(schema='PubMed-strong-reference-complete-v1',complete=True,record_id=spec['record_id'],
                    condition=spec['condition'],seed=spec['seed'],source_manifest_sha256=spec['source_manifest_sha256'],
                    release_sha256=spec['_release_sha256'],providers=providers,split_identity=spec['split_identity'],split_seed=190111,
                    bodies=bodies,private_horizons_complete=True,selector=spec['selector'],max_epochs_per_body=2000,patience_per_body=250,
                    selected_VALID=readouts['VALID']['classification'],selected_prediction_signatures=readouts['VALID']['prediction_signatures'],
                    selected_readouts=readouts,serving_benchmark=benchmark,timings=timings,
                    assembly_counters=assembly_counters,assembly_preprocessing_seconds=assembly_preprocessing,
                    final_session_counters=dict(session.counters),fresh_training_body_trajectories=len(bodies),
                    body_constructions=len(bodies)+(4 if members==4 else 0),preprocessing_banks=len(bodies)+(1 if members==4 else 0),
                    predictor_parameters=members*2069875,decoder_parameters=0,
                    selected_state_verification=dict(predictions_and_correct_masks_exact=True,counts_and_epoch_exact=True,
                        float_absolute_tolerance=2e-6,float_relative_tolerance=2e-6,tolerance_is_restore_verification_only=True),
                    complete_saved_member_logits=dict(path=payload.name,sha256=sha(payload),bytes=payload.stat().st_size,
                        keys=['factual_member_logits'],factual_shape=[members,19717,3],contains_labels_or_role_ids=False,server_only=True),
                    train_bundle_sha256=spec['train_bundle']['sha256'],valid_bundle_sha256=spec['valid_bundle']['sha256'],
                    inclusive_seconds=time.monotonic()-started,CPU_user_seconds=cpu()[0],CPU_system_seconds=cpu()[1],
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                    output_payload_bytes_before_COMPLETE=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                    TEST_access=False,TEST_scored=False,CORE=False,masking=False,HPO=False,automatic_retry=False,
                    post_screen_exploration=True,paper_score_recalculation=False,further18_activated=False,
                    partial_family_comparison_allowed=False,whole_nine_plus_three_anchors_required=True,owner_success_not_inferred=True)
        write(output/'COMPLETE.json',result);return result
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,record_id=spec['record_id'],error_type=type(error).__name__,error=str(error),
            inclusive_seconds=time.monotonic()-started,bodies_completed=bodies,timings=timings,
            partial_current_counters=None if session is None else session.counters,
            automatic_retry=False,TEST_access=False,partial_family_comparison_allowed=False))
        raise


def main():
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('describe','run'),default='describe')
    parser.add_argument('--release',type=Path);parser.add_argument('--release-sha256')
    args=parser.parse_args()
    if args.mode=='describe':print(json.dumps(description(),indent=2,sort_keys=True));return
    if args.release is None or args.release_sha256 is None:parser.error('New exact root release required')
    spec,output=admit(args.release,args.release_sha256);result=fit(spec,output,started)
    print(json.dumps(dict(complete=result['complete'],record_id=spec['record_id'],output=str(output),owner_success_not_inferred=True)))


if __name__=='__main__':main()

"""Full frozen DBLP TRAIN-loss resource qualification; never driver main/selection."""
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
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as s:json.dump(value,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def rss():
    # Linux KiB; current and process-lifetime peak scopes are reported separately.
    observed={}
    for line in Path('/proc/self/status').read_text().splitlines():
        for field in ('VmRSS:','VmSize:','VmPeak:'):
            if line.startswith(field):observed[field[:-1]+'_bytes']=int(line.split()[1])*1024
    return dict(current_RSS_bytes=observed['VmRSS_bytes'],process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**observed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',required=True);p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--run-name',required=True);p.add_argument('--GPU-release',help='Root JSON explicitly admits GPU qualification after original cohort closes')
    args=p.parse_args();require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.', '..'),'Fresh simple run name')
    out=HERE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=False)
    result=dict(status='preparing',device=args.device,rows=[],qualification_only=True,training_driver_main_called=False,
        validation_or_test_scored=False,model_selection=False,labels_saved=False,scientific_study_outcomes=False,
        source_development_pool_opened=True,consecutive_updates_per_arm=6,exact_fit_style_lifetimes=True,only_frozen_first_split_TRAIN_targets_materialized=True,
        DGL_import_required=False)
    originals=[];started=time.perf_counter();torch=None
    try:
        binding=json.loads((HERE/'BINDINGS.json').read_text())
        require(sha(args.freeze)==binding['freeze']['sha256'],'Exact adopted freeze changed')
        frozen=json.loads(Path(args.freeze).read_text());result['study_freeze_sha256']=sha(args.freeze)
        require(frozen['study_adopted_by_root'] is True and frozen['test_labels_closed'] is True,'Adopted development-only freeze required')
        require(frozen['arms']==binding['arms'] and frozen['seeds']==[131,137,139,149,151],'Frozen arm/seed geometry changed')
        for record in binding['source_records']:
            require(sha(record['path'])==record['sha256'],'Runtime source changed: '+record['path'])
        originals=list(binding['source_records'])+[dict(path=args.freeze,sha256=binding['freeze']['sha256'])]
        result['source_custody_before']=True
        if args.device=='cpu':require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and resource.getrlimit(resource.RLIMIT_AS)==(16*2**30,)*2,'CPU qualification requires CUDA hidden and preimport16GiB AS')
        else:
            require(args.GPU_release is not None,'GPU qualification remains closed without root release')
            release=json.loads(Path(args.GPU_release).read_text())
            require(release['GPU_qualification_authorized'] is True and release['original_cohort_closed'] is True
                    and release['study_freeze_sha256']==binding['freeze']['sha256'] and release['run_name']==args.run_name
                    and release['qualification_manifest_sha256']==sha(HERE/'MANIFEST.json')
                    and os.environ.get('CUDA_VISIBLE_DEVICES')==frozen['GPU_uuid'],'Exact once-cohort-closed GPU resource release required')
            result['GPU_release_sha256']=sha(args.GPU_release)
        import torch
        require(torch.__version__=='2.1.2+cu118','Pinned Torch2.1.2+cu118 runtime required')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result['environment']=dict(torch=torch.__version__,python=sys.version,threads=torch.get_num_threads(),
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),DGL_imported='dgl' in sys.modules)
        require('dgl' not in sys.modules,'Qualification closure unexpectedly imported DGL')
        device=torch.device(args.device)
        if device.type=='cuda':require(torch.cuda.device_count()==1,'Exactly released one GPU required')
        phase=Path(binding['canonical_phase'])
        driver=load('resource_pinned_DBLP_driver',phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2/train_dblp.py')
        inputs=load('resource_pinned_DBLP_inputs',phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2/dblp_inputs.py')
        families=load('resource_pinned_DBLP_families',phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2/families.py')
        onecycle=load('resource_pinned_DBLP_onecycle',phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2/onecycle_state.py')
        implementation=load('resource_pinned_HGT',phase/binding['implementation_source'])
        tick=time.perf_counter();schema,attributes,edges=inputs.stream_schema(frozen['archive'],frozen['members'])
        require(schema['member_sha256']==frozen['member_sha256'] and schema['node_counts']==frozen['expected_node_counts']
                and schema['input_dims']==frozen['expected_input_dims'],'Full graph input identity differs')
        require({str(r['raw_id']):(r['source'],r['target']) for r in schema['relations']}==
                {k:tuple(v) for k,v in frozen['expected_relations'].items()},'Exact released relation geometry required')
        result['stream_materialization_seconds']=time.perf_counter()-tick
        # This exact existing source pool contains TRAIN+VAL labels. No validation
        # targets are constructed or scored; only first frozen split TRAIN labels
        # enter Torch. No ZIP label member is opened by stream_schema.
        development=inputs.read_development_labels(frozen['development_labels'],frozen['archive']['sha256'])
        require(development['source_label_member_sha256']==frozen['source_label_member_sha256'],'Source TRAIN_VAL pool identity differs')
        split_record=frozen['splits'][0];seed=split_record['seed']
        split=json.loads(inputs.verified(split_record['descriptor']).read_text());train_ids=split['train_ids']
        require(seed==split['seed']==131 and train_ids==sorted(set(train_ids)) and len(train_ids)==974,'Exact first frozen TRAIN IDs required')
        selected_ids=set(train_ids)
        train_lookup={node:label for node,label in zip(development['node_ids'],development['labels']) if node in selected_ids}
        classes=len(development['train_class_schema'])
        require(len(train_lookup)==974 and set(train_lookup.values())==set(range(classes)),'TRAIN establishes all classes')
        labels=torch.tensor([train_lookup[node] for node in train_ids],dtype=torch.long,device=device)
        ids=torch.tensor(train_ids,dtype=torch.long,device=device)
        del development,train_lookup,selected_ids
        for r in (frozen['archive'],frozen['development_labels'],split_record['descriptor']):originals.append(r)
        result['label_scope']=dict(source_pack='explicit frozen TRAIN_VAL_ONLY pool',loss_targets='974 frozen seed131 TRAIN only',validation_targets_constructed=False,ZIP_label_members_opened=[])
        tick=time.perf_counter();graph,features,schema=inputs.materialize(schema,attributes,edges,implementation,frozen['relation_row_order'],device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        result['native_tensor_materialization_seconds']=time.perf_counter()-tick;result['graph_schema']=schema
        del attributes,edges
        tick=time.perf_counter();models=families.build(implementation,graph,schema['input_dims'],classes,seed,seed+900001,device,frozen['arms'])
        if device.type=='cuda':torch.cuda.synchronize(device)
        result['all_seven_resident_model_build_seconds']=time.perf_counter()-tick
        def equal(a,b):
            if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
            if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
            if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        require(equal(models['CP'].core.state_dict(),models['global_BE'].core.state_dict())
                and equal(models['CP'].core.state_dict(),models['untied_HGT'].cores[0].state_dict()),'Paired common/member0 geometry differs')
        require(all(torch.equal(cp.a,be.a) and torch.equal(cp.b,be.b)
                    for cp,be in zip(models['CP'].factors,models['global_BE'].factors)),'Paired CP/global base factors differ')
        result['paired_CP_global_untied_geometry_verified']=True;result['models_resident_before_arms']=rss()
        for arm in frozen['arms']:
            arm_out=out/arm;arm_out.mkdir();row=dict(arm=arm,status='preparing',seed=seed)
            model=models[arm];optimizer=scheduler=streams=logits=loss=saved=member_logits=checkpoint=state=None
            try:
                torch.manual_seed(seed+700001)
                if device.type=='cuda':torch.cuda.manual_seed(seed+700001);torch.cuda.reset_peak_memory_stats(device)
                streams=implementation.MemberStreams([seed+700001+1009*m for m in range(4)],device)
                optimizer=torch.optim.AdamW(model.parameters(),weight_decay=1e-4)
                scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=300,max_lr=1e-3,pct_start=.05)
                row['before_epoch']=rss();row['parameters']=implementation.parameter_count(model)
                def measured(callback):
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    start=time.perf_counter();value=callback()
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    return value,time.perf_counter()-start
                row['per_update_memory']=[]
                for update in range(1,7):
                    observed=dict(update=update,before_TRAIN=rss())
                    # Retain previous eval/checkpoint refs through next TRAIN forward,
                    # and TRAIN logits/loss through eval, matching unchanged fit.
                    model.train();optimizer.zero_grad()
                    logits,observed['TRAIN_forward_seconds']=measured(lambda:model(graph,features,'0',streams))
                    loss=implementation.mean_member_ce(logits,ids,labels)
                    require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)),'Nonfinite TRAIN forward/loss')
                    _,observed['TRAIN_backward_seconds']=measured(loss.backward)
                    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite TRAIN gradients')
                    row['nonzero_gradient_tensors']=sum(int(p.grad is not None and bool((p.grad!=0).any())) for p in model.parameters())
                    require(row['nonzero_gradient_tensors']>0,'No TRAIN gradients')
                    def step():optimizer.step();scheduler.step(update)
                    _,observed['optimizer_scheduler_seconds']=measured(step)
                    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite updated parameters')
                    observed['after_TRAIN']=rss()
                    model.eval()
                    with torch.no_grad():member_logits,observed['score_free_eval_forward_seconds']=measured(lambda:model(graph,features,'0'))
                    require(tuple(member_logits.shape)[1:]==(4057,classes) and bool(torch.isfinite(member_logits).all()),'Full target output geometry/finite check')
                    checkpoint=dict(schema='DBLP_resource_qualification_disposable_update_v2',qualification_only=True,arm=arm,seed=seed,update=update,
                        model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=onecycle.portable_state(scheduler),
                        optimizer_parameter_names=[name for name,_ in model.named_parameters()],member_RNG=streams.state_dict(),
                        torch_CPU_RNG=torch.get_rng_state(),torch_device_RNG=None,
                        study_freeze_sha256=binding['freeze']['sha256'],training_packet_manifest_sha256=binding['training_manifest_sha256'])
                    def write_update():
                        torch.save(checkpoint,arm_out/'DISPOSABLE_CURRENT_UPDATE_STATE.pt')
                        torch.save(member_logits.detach().cpu(),arm_out/'DISPOSABLE_CURRENT_UNSCORED_LOGITS.pt')
                    _,observed['state_and_logits_write_seconds']=measured(write_update)
                    observed['after_eval_and_checkpoint']=rss()
                    row['per_update_memory'].append(observed)
                    with (arm_out/'RESOURCE_UPDATES.jsonl').open('a') as stream:stream.write(json.dumps(observed,allow_nan=False)+'\n')
                for key in ('TRAIN_forward_seconds','TRAIN_backward_seconds','optimizer_scheduler_seconds','score_free_eval_forward_seconds'):
                    row[key]=sum(item[key] for item in row['per_update_memory'])/6
                logits=member_logits
                state=dict(schema='DBLP_resource_qualification_disposable_state_v1',qualification_only=True,arm=arm,seed=seed,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=onecycle.portable_state(scheduler),
                    optimizer_parameter_names=[name for name,_ in model.named_parameters()],member_RNG=streams.state_dict(),
                    torch_CPU_RNG=torch.get_rng_state(),torch_device_RNG=None if device.type=='cpu' else torch.cuda.get_rng_state(device),
                    study_freeze_sha256=binding['freeze']['sha256'],training_packet_manifest_sha256=binding['training_manifest_sha256'])
                def save():torch.save(state,arm_out/'DISPOSABLE_STATE.pt');torch.save(logits.detach().cpu(),arm_out/'DISPOSABLE_UNSCORED_LOGITS.pt')
                _,row['state_and_logits_write_seconds']=measured(save)
                saved,row['weights_only_read_seconds']=measured(lambda:torch.load(arm_out/'DISPOSABLE_STATE.pt',map_location=device,weights_only=True))
                require(equal(saved['model'],model.state_dict()) and equal(saved['optimizer'],optimizer.state_dict())
                        and equal(saved['scheduler'],onecycle.portable_state(scheduler))
                        and equal(saved['member_RNG'],streams.state_dict())
                        and equal(saved['torch_CPU_RNG'].cpu(),torch.get_rng_state()),'Saved state/RNG custody differs')
                require(saved['optimizer_parameter_names']==[name for name,_ in model.named_parameters()],'Optimizer name order differs')
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);onecycle.restore_state(scheduler,saved['scheduler'])
                streams.load_state_dict(saved['member_RNG']);torch.set_rng_state(saved['torch_CPU_RNG'].cpu())
                if device.type=='cuda':torch.cuda.set_rng_state(saved['torch_device_RNG'].cpu(),device)
                row.update(status='qualified',optimizer_updates=6,validation_or_test_scored=False,model_selection=False,
                    full_state_rng_custody=True,checkpoint_sha256=sha(arm_out/'DISPOSABLE_STATE.pt'),after_epoch=rss(),
                    GPU_peak_allocated_bytes=None if device.type=='cpu' else torch.cuda.max_memory_allocated(device),
                    GPU_peak_reserved_bytes=None if device.type=='cpu' else torch.cuda.max_memory_reserved(device))
                row['observed_maxcap_epoch_seconds']=sum(row[key] for key in ('TRAIN_forward_seconds','TRAIN_backward_seconds','optimizer_scheduler_seconds','score_free_eval_forward_seconds','state_and_logits_write_seconds'))
            except Exception as error:
                row.update(status='qualification_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),memory=rss())
            finally:
                write(arm_out/'RECEIPT.json',row);result['rows'].append(row)
                with (out/'ARM_TERMINALS.jsonl').open('a') as s:s.write(json.dumps(row,allow_nan=False)+'\n')
                del models[arm],model,optimizer,scheduler,streams,logits,loss,saved,member_logits,checkpoint,state
        complete=len(result['rows'])==7 and all(r['status']=='qualified' for r in result['rows'])
        result['status']='qualified' if complete else 'qualification_incomplete'
        if complete:
            observed=5*300*sum(r['observed_maxcap_epoch_seconds'] for r in result['rows'])
            result['forecast']=dict(cases=35,maximum_epochs_per_case=300,upper_case_epoch_count=10500,
                observed_family_cost_maxcap_seconds=observed,safety_factor=2,safety_factor_forecast_seconds=2*observed,
                safety_factor_forecast_hours=2*observed/3600,initial_graph_build_seconds=result['stream_materialization_seconds']+result['native_tensor_materialization_seconds'],
                scope='Measured six full-graph consecutive TRAIN epochs per family with fit-style tensor lifetimes, per-update unscored evaluation/checkpoint writes, repeated5 seeds x300; safety factor2 is planning allowance, not a guaranteed upper bound. No validation score or fitted comparison.',
                CPU_estimate_is_not_GPU_measurement=device.type=='cpu')
    except Exception as error:
        result.update(status='qualification_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
    finally:
        try:
            for record in originals:require(sha(record['path'])==record['sha256'],'Original changed: '+record['path'])
            result['originals_preserved']=True
        except Exception as error:result.update(status='original_preservation_failed',preservation_error=str(error))
        result.update(wall_seconds=time.perf_counter()-started,memory=rss(),DGL_imported='dgl' in sys.modules)
        write(out/'RESOURCE_QUALIFICATION.json',result)
    print(json.dumps(dict(status=result['status'],output=str(out),forecast=result.get('forecast'),memory=result['memory'],rows=[{k:r.get(k) for k in ('arm','status','observed_maxcap_epoch_seconds')} for r in result['rows']])))
    return 0 if result['status']=='qualified' else 1


if __name__=='__main__':raise SystemExit(main())

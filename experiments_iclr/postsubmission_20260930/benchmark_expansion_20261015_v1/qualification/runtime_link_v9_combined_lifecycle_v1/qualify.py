"""One real two-epoch NCN or BUDDY lifecycle, selected export and exact resume."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback
from support import guard,inside,receipt,require,sources,load_admission,validate_interpreter,ROOT,atomic_json,digest


class ControlledCompletedEpochStop(BaseException):pass
class EngineeringWallLimit(BaseException):pass


def main():
    guard()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--admission',required=True)
    args=parser.parse_args();here=inside(Path(__file__).parent)
    admission,job_path,case=load_admission(args.admission,here);family=case['case_id'];limits=case['limits'];layout=case['layout'];maps=layout['optimizer_maps_per_epoch']
    output=inside(args.output);require(not output.exists(),'Fresh own qualifier output required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit admitted allocation GPU0 visibility required')
    interpreter=validate_interpreter(here);require(interpreter['Torch_imported']is False,'Verify actual venv before Torch import')
    output.mkdir(parents=True);started=time.perf_counter();tensor_library=None;live={'phase':None,'objective':None};handles=[]
    report={'schema':'runtime-link-v9-combined-native-lifecycle-result-v1','status':'running','case_id':family,'stage':'imports',
        'admission':receipt(args.admission),'job':receipt(job_path),'script':receipt(__file__),'limits':limits,'interpreter':interpreter,
        'native_constructions_attempted':0,'native_constructions_completed':0,'TRAIN_epochs_started':0,'TRAIN_epochs_completed':0,
        'updates_attempted':0,'updates_completed':0,'optimizer_attempts':0,'optimizer_completed':0,'reverse_attempts':0,'reverse_completed':0,
        'extra_reverse_attempts':0,'VALID_attempts':0,'VALID_completed':0,'native_VALID_attempts':0,'native_VALID_completed':0,
        'mean_exports_attempted':0,'mean_exports_completed':0,'native_forwards':{'TRAIN':0,'VALID':0},'native_forward_attempts':{'TRAIN':0,'VALID':0},'TRAIN_task_query_records':0,
        'TRAIN_source_iterator_records':0,'provider_sampler_calls':0,'provider_TRAIN_hydration_calls':0,'provider_VALID_hydration_calls':0,
        'restore_checks':[],'global_RNG_restore_checks':[],'epoch_receipts':[],'VALID_receipts':[],'export_receipts':[],'invocations':[],'snapshots':[],
        'update_cost_ledger_rows':0,'update_cost_chain_sha256':digest([]),'scientific_fits':0,'TEST_accessed':False,'automatic_retries':0,
        'full_campaign_GO':False,'reporting_scope':'approved exploratory TRAIN/VALID; later confirmation is not an admission gate',
        'reporting_ROOT_REVIEW':admission['reporting_ROOT_REVIEW'],'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}

    def write():
        usage=resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter()-started,process_user_seconds=usage.ru_utime,process_system_seconds=usage.ru_stime,process_peak_RSS_KiB=usage.ru_maxrss)
        if tensor_library is not None:
            report['CUDA_context_initialized']=tensor_library.cuda.is_initialized()
            if tensor_library.cuda.is_initialized():
                report['cuda_peak_allocated_bytes']=max(report.get('cuda_peak_allocated_bytes',0),tensor_library.cuda.max_memory_allocated())
                report['cuda_peak_reserved_bytes']=max(report.get('cuda_peak_reserved_bytes',0),tensor_library.cuda.max_memory_reserved())
        atomic_json(output/'RESULT.json',report)

    def begin(key):require(report[key]<limits[key],'Hard bound exhausted before '+key);report[key]+=1;write()
    def limit(*_):raise EngineeringWallLimit('Frozen single-case wall watchdog exhausted')
    signal.signal(signal.SIGTERM,limit);signal.signal(signal.SIGALRM,limit);signal.alarm(case['wall_watchdog_seconds']);write()
    try:
        report['sources_before']=sources(here,admission['driver_manifest_sha256'])
        runtime=here.parents[1]/'runtime_link_v9';sys.path.insert(0,str(runtime))
        import common
        import adapters
        import run_fit
        import torch
        import numpy as np
        tensor_library=torch;torch.set_num_threads(2);torch.set_num_interop_threads(1)
        job=json.loads(job_path.read_text());run_fit.validate_job(job,ROOT)
        require(job['procedure']==case['method']and job['seed']==case['seed']and job['recipe']['native_fixed_config']==case['native_fixed_config'],'Frozen case changed')
        report.update(torch=str(torch.__version__),threads=2,interop_threads=1,job_hash=common.digest(job))
        observer_pin=json.loads((here/'SOURCE_CLOSURE.json').read_text())['reporting_observer_source']
        require(receipt(inside(observer_pin['path']))==observer_pin,'Reporting observer source differs')
        observer_module,_=common.load_module(observer_pin['path'],ROOT,observer_pin['sha256'])
        reporting_observer=None
        report['reporting_observer_source']=observer_pin

        def exact(a,b,name):
            if torch.is_tensor(a):require(torch.is_tensor(b)and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.detach().cpu()),'Exact tensor differs: '+name)
            elif isinstance(a,np.ndarray):require(isinstance(b,np.ndarray)and a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),'Exact array differs: '+name)
            elif isinstance(a,dict):
                require(isinstance(b,dict)and set(a)==set(b),'Exact mapping differs: '+name)
                for k in a:exact(a[k],b[k],name+'/'+str(k))
            elif isinstance(a,(tuple,list)):
                require(type(a)is type(b)and len(a)==len(b),'Exact sequence differs: '+name)
                for i,(x,y)in enumerate(zip(a,b)):exact(x,y,name+'/'+str(i))
            else:require(type(a)is type(b)and a==b,'Exact scalar differs: '+name)

        def identity(value):
            v=value.detach().cpu().contiguous();return {'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(memoryview(v.numpy()).cast('B')).hexdigest()}
        original_create=adapters.create_adapter;original_restore_rng=run_fit.restore_rng
        original_backward=torch.Tensor.backward;original_grad=torch.autograd.grad

        def restored_rng(t,value,**kw):
            original_restore_rng(t,value,**kw);exact(value,common.global_rng(t,device='cuda:0'),'all_global_RNG')
            report['global_RNG_restore_checks'].append({'exact':True,'invocation':len(report['invocations'])+1});write()

        def backward(tensor,*a,**kw):
            require(live['phase']=='TRAIN'and live['objective']is tensor,'Only authentic scalar-J TRAIN objective may call backward')
            begin('reverse_attempts');result=original_backward(tensor,*a,**kw);report['reverse_completed']+=1;write();return result

        def forbidden_grad(*a,**kw):report['extra_reverse_attempts']+=1;write();raise ValueError('No extra autograd collection belongs to either scalar-J case')

        def measured_create(t,descriptor,policy,root):
            begin('native_constructions_attempted');owner=original_create(t,descriptor,policy,root);report['native_constructions_completed']+=1
            n=owner.session.native;exact(owner.session.layout,layout,'native_epoch_layout')
            population=int(n.positive.shape[1])if family=='NCN'else len(n.positive)
            require(len(owner.optimizers)==1 and n.members==4 and population==case['native_positive_population'],'One original four-route Adam and source TRAIN required')
            require(all(p.device.type=='cuda'for p in owner.model.parameters()),'Native model device differs')
            require(owner.extension_authority['mean_protocol']==('same_call'if family=='NCN'else'provider'),'Expected v8 extension source protocol differs')
            source_pins=[owner.callback_receipt,owner.extension_receipt,owner.policy_receipt,*owner.dependency_receipts]
            for p in source_pins:exact(receipt(p['path']),p,'source_receipt')
            report.setdefault('construction_sources',[]).append(source_pins)
            original_train,original_valid,original_load,original_export,original_close=owner.train_epoch,owner.validate,owner.load_state,owner.selected_mean_export,owner.close
            original_update=owner.session.train_update;original_native_valid=owner.session.validate;original_step=owner.optimizers[0].step
            builder=owner.session.c6 if family=='NCN'else owner.session.objectives;build_name='native_objective'if family=='NCN'else'build';original_build=getattr(builder,build_name)
            module=n.encoder if family=='NCN'else n.model
            def forward_pre_hook(*_):
                phase=live['phase'];require(phase in('TRAIN','VALID'),'Native forward outside admitted phase')
                cap=case['total_exact_work']['native_'+phase+'_model_or_encoder_forwards']
                require(report['native_forward_attempts'][phase]<cap,'Native forward bound exhausted before call')
                report['native_forward_attempts'][phase]+=1
            def forward_hook(*_):
                require(live['phase']in('TRAIN','VALID'),'Native forward outside admitted phase');report['native_forwards'][live['phase']]+=1
            pre_handle=module.register_forward_pre_hook(forward_pre_hook);handle=module.register_forward_hook(forward_hook);handles.extend([pre_handle,handle])
            def build(*a,**kw):
                terms=original_build(*a,**kw)
                if family=='NCN':live['objective']=terms['J']
                else:
                    require(terms['scalar_update']and terms['shared_objective']is terms['private_objective']and terms['shared_objective']is terms['logged_objective'],'One authentic joint scalar required')
                    live['objective']=terms['shared_objective']
                return terms
            def step(*a,**kw):
                require(live['phase']=='TRAIN','Optimizer outside TRAIN');begin('optimizer_attempts');value=original_step(*a,**kw);report['optimizer_completed']+=1;write();return value
            def update():
                begin('updates_attempted');before=(report['optimizer_completed'],report['reverse_completed'],n.updates,report['native_forwards']['TRAIN'])
                provider=owner.session.base if family=='BUDDY'else None
                prior=(provider.ledger['sampler_calls'],provider.ledger['TRAIN_hydration_calls'])if provider is not None else None
                value=original_update();after=(report['optimizer_completed'],report['reverse_completed'],n.updates,report['native_forwards']['TRAIN'])
                require(tuple(b-a for a,b in zip(before,after))==(1,1,1,4),'Exactly one update/Adam/backward/four native routes required')
                work=value['work'];progress=value['epoch_progress'];batch=progress['consumed_maps_in_current_epoch']
                expected_queries=layout['task_query_records_per_epoch']if family=='NCN'else(794 if batch==maps else 1024)
                if family=='NCN':
                    expected_queries=131072
                    require(work['positive_query_records']==65536 and work['negative_query_records']==65536 and work['captured_native_task_query_records']==131072,'Whole native NCN positive/negative task batch required')
                    sampler=n.permutation[n.cursor-n.batch_size:n.cursor]if hasattr(n,'permutation')else n.sampler_state()['permutation'][n.cursor-n.batch_size:n.cursor]
                    sampler_id=identity(sampler)
                else:
                    require(work['query_records']==expected_queries and work['captured_source_query_records']==expected_queries and work['native_BUDDY_forwards']==4,'Full native BUDDY batch/tail differs')
                    now=(provider.ledger['sampler_calls'],provider.ledger['TRAIN_hydration_calls'])
                    require(tuple(b-a for a,b in zip(prior,now))==(1,1),'One genuine provider sampler/hydration per update required')
                    report['provider_sampler_calls']+=1;report['provider_TRAIN_hydration_calls']+=1
                    recipe=value['actual_source_recipe_receipt'];sampler_id=recipe
                    require(sum(value['interaction_diagnostics']['source_group_counts'].values())==expected_queries,'Actual mixed BUDDY labels do not cover source batch')
                    require(owner.session._last_VALID is None,'TRAIN retained stale provider sidecar')
                require(work['reverse_collections']==1 and work['optimizer_transitions']==1 and work['extra_source_forward_views']==0 and owner._cache is None,'Unexpected native reverse/step/extra forward or stale runtime cache')
                report['updates_completed']+=1;report['TRAIN_task_query_records']+=expected_queries
                report['TRAIN_source_iterator_records']+=65536 if family=='NCN'else expected_queries
                row={'trajectory_update':n.updates,'epoch_progress':progress,'source_sampler_identity':sampler_id,'work':work,'elapsed_seconds':time.perf_counter()-started}
                report['update_cost_chain_sha256']=digest([report['update_cost_chain_sha256'],row]);report['update_cost_ledger_rows']+=1
                with (output/'UPDATE_COST.jsonl').open('a')as stream:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                live['objective']=None;write();return value
            def train():
                begin('TRAIN_epochs_started');live['phase']='TRAIN';before=report['updates_completed']
                try:value=original_train()
                finally:live['phase']=None;live['objective']=None;write()
                complete=value.get('source_loss_complete')if family=='NCN'else value.get('native_source_epoch_loss_complete')
                require(value['complete_native_TRAIN']is True and value['optimizer_maps']==maps and value['native_epoch_optimizer_maps']==maps and complete is True and report['updates_completed']-before==maps,'Exactly one complete source epoch required')
                expected={'optimizer_transitions':maps,'encoder_forwards':4*maps,'reverse_collections':maps}if family=='NCN'else{'optimizer_transitions':maps,'native_BUDDY_forwards':4*maps,'query_records':layout['task_query_records_per_epoch'],'reverse_collections':maps}
                exact(value['native_work_totals'],expected,'epoch_native_work');report['TRAIN_epochs_completed']+=1
                report['epoch_receipts'].append({'native_layout':layout,'returned_epoch':value});write();return value
            def native_valid():
                require(report['native_VALID_attempts']<4,'No fifth native VALID');report['native_VALID_attempts']+=1;write()
                value=original_native_valid();report['native_VALID_completed']+=1;return value
            def valid():
                begin('VALID_attempts');live['phase']='VALID';before=(report['native_forwards']['VALID'],report['optimizer_completed'],report['reverse_completed'],n.updates)
                provider=owner.session.base if family=='BUDDY'else None;prior=provider.ledger['VALID_hydration_calls']if provider is not None else None
                try:value=original_valid()
                finally:live['phase']=None;write()
                after=(report['native_forwards']['VALID'],report['optimizer_completed'],report['reverse_completed'],n.updates)
                require(tuple(b-a for a,b in zip(before,after))==((4 if family=='NCN'else 8),0,0,0),'VALID changed TRAIN clock or native route count')
                require(value['role']=='VALID'and value['TEST_accessed']is False and value['complete_source_VALID']is True and(value['complete_VALID_positive_records'],value['complete_VALID_negative_records'])==(60084,100000),'Whole official VALID required')
                exact(value['ids'],torch.arange(160084),'VALID IDs');exact(value['labels'],torch.cat((torch.ones(60084,dtype=torch.long),torch.zeros(100000,dtype=torch.long))),'VALID labels')
                exact(value['query_endpoints'],torch.cat((n.valid['edge'],n.valid['edge_neg'])).cpu(),'VALID endpoint order')
                if provider is not None:
                    require(provider.ledger['VALID_hydration_calls']-prior==2 and value['production_native_VALID_inputs_verified']is True,'Both actual BUDDY VALID input roles must authenticate')
                    report['provider_VALID_hydration_calls']+=2
                side=value['same_bank_mean_VALID'];require(side['extra_forward_calls']==0,'Extra mean inference')
                report['VALID_completed']+=1;report['VALID_receipts'].append({'source_updates':n.updates,'completed_epoch':owner.completed_epochs,
                    'ids':identity(value['ids']),'labels':identity(value['labels']),'query_endpoints':identity(value['query_endpoints']),
                    'actual_logits':identity(value['outputs']),'mean_logits':identity(side['outputs']),'actual_loss':value['loss'],'actual_metrics':value['metrics'],
                    'mean_loss':side['loss'],'mean_metrics':side['metrics'],'authority':side['authority'],'work':value['work']});write();return value
            def load_state(value):
                refs={k:id(p)for k,p in n.source_parameter_references.items()};original_load(value);exact(value,owner.state(),'complete_model_Adam_sampler_scope_RNG_modes_epoch')
                require(refs=={k:id(p)for k,p in n.source_parameter_references.items()}and owner._cache is None,'Restore replaced source parameters or retained runtime cache')
                if family=='BUDDY':require(owner.session._last_VALID is None,'Restore retained provider sidecar')
                report['restore_checks'].append({'exact':True,'completed_epoch':owner.completed_epochs,'source_updates':n.updates,'state_signature_sha256':common.digest(owner._state_signature(value))});write()
            def export(actual,**context):
                begin('mean_exports_attempted');before=copy.deepcopy(report['native_forwards']);before_valid=report['native_VALID_attempts'];value=original_export(actual,**context)
                require(value is not None and value['extra_forward_calls']==0,'Selected mean export missing/extra inference')
                exact(before,report['native_forwards'],'zero_extra_export_forwards');require(before_valid==report['native_VALID_attempts'],'Export called native VALID')
                for key in('ids','labels','query_endpoints'):exact(value[key],actual[key],'emitted_mean/'+key)
                exact(value['raw_logits'],actual['same_bank_mean_VALID']['outputs'],'emitted_current_same_call_mean');exact(value['authority'],actual['same_bank_mean_VALID']['authority'],'emitted_authority')
                report['mean_exports_completed']+=1;report['export_receipts'].append({k:copy.deepcopy(value[k])for k in('job_hash','job_id','selected_epoch','selected_checkpoint','runtime_binding','authority','loss','metrics')});write();return value
            def close():pre_handle.remove();handle.remove();setattr(builder,build_name,original_build);original_close()
            owner.train_epoch,owner.validate,owner.load_state,owner.selected_mean_export,owner.close=train,valid,load_state,export,close
            owner.session.train_update,owner.session.validate=update,native_valid;owner.optimizers[0].step=step;setattr(builder,build_name,build)
            write();return owner
        adapters.create_adapter=measured_create;run_fit.restore_rng=restored_rng;torch.Tensor.backward=backward;torch.autograd.grad=forbidden_grad
        folder=output/'own_fit'
        def snapshot(name):
            destination=output/'snapshots'/name;destination.mkdir(parents=True);rows=[]
            for filename in('JOB.json','RESULT.json','CURVE.jsonl','EVENTS.jsonl','PROGRESS.json','selected.pt','recovery.pt','selected_VALID.pt'):
                p=folder/filename
                if p.exists():
                    shutil.copy2(p,destination/filename);a,b=receipt(p),receipt(destination/filename);require(a['sha256']==b['sha256'],'Snapshot differs');rows.append({'original':a,'copy':b})
            report['snapshots'].append({'name':name,'files':rows});write()
        def invoke(name,expected_updates,resume=None,interrupt=False):
            previous=run_fit.atomic_json;interrupted=False
            def durable_progress(path,value):
                nonlocal interrupted
                previous(path,value)
                if interrupt and Path(path)==folder/'PROGRESS.json'and value['train']['epoch']==1:
                    for filename in('recovery.pt','selected.pt','CURVE.jsonl','PROGRESS.json'):
                        with(folder/filename).open('rb')as stream:os.fsync(stream.fileno())
                    with(output/'UPDATE_COST.jsonl').open('rb')as stream:os.fsync(stream.fileno())
                    fd=os.open(folder,os.O_RDONLY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                    interrupted=True;raise ControlledCompletedEpochStop('Declared stop after durable epoch1 recovery/CURVE/PROGRESS')
            report['stage']=name;write();begin_time=time.perf_counter()
            if interrupt:run_fit.atomic_json=durable_progress
            try:result=run_fit.run(copy.deepcopy(job),ROOT,folder,resume=resume)
            finally:run_fit.atomic_json=previous
            observed=reporting_observer.record_invocation(result,name)
            report['reporting_observer_invocations']=len(reporting_observer.invocations)
            report['invocations'].append({'observer_metadata':observed,'name':name,'outer_wall_seconds':time.perf_counter()-begin_time,'expected_controlled_interruption':interrupt,'result':copy.deepcopy(result),'result_receipt':receipt(folder/'RESULT.json')});write()
            require(result['optimizer_calls_this_invocation']=={'attempts':[expected_updates],'completed':[expected_updates]},'Invocation optimizer cost differs')
            if interrupt:require(interrupted and result['status']=='failed'and result['failure']['type']=='ControlledCompletedEpochStop'and result['completed_epochs']==1 and result['training_epochs_this_invocation']==1,'Expected durable epoch1 stop was not preserved')
            else:require(result['status']=='complete','Native invocation failed: '+str(result.get('failure')))
            exact(json.loads((folder/'JOB.json').read_text()),job,'immutable JOB');gc.collect();torch.cuda.empty_cache();return result
        def own_export():
            value=torch.load(folder/'selected_VALID.pt',map_location='cpu',weights_only=False);checkpoint=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
            mean=value['same_bank_mean_VALID'];p=receipt(folder/'selected.pt');epoch=checkpoint['selector']['selected_epoch']
            require(value['schema']=='own-selected-VALID-logits-v1'and mean['schema']=='own-selected-same-bank-mean-VALID-v1'and value['role']==mean['role']=='VALID'and value['TEST_accessed']is False and mean['TEST_accessed']is False and epoch in(1,2)and mean['extra_forward_calls']==0,'Persisted actual/mean export structure differs')
            for obj in(value,mean):require(obj['job_hash']==common.digest(job)and obj['selected_epoch']==epoch and obj['selected_checkpoint']==p and(obj['complete_VALID_positive_records'],obj['complete_VALID_negative_records'])==(60084,100000),'Persisted JOB/checkpoint/epoch/VALID authority differs')
            exact(mean['runtime_binding'],checkpoint['runtime_binding'],'export_runtime_binding')
            for key in('ids','labels','query_endpoints'):exact(value[key],mean[key],'persisted_actual_mean/'+key)
            for key,name in[('ids','ids'),('labels','labels'),('query_endpoints','query_endpoints'),('actual_logits','raw_logits')]:exact(identity(value[name]),mean['authority'][key],'persisted_actual_authority/'+key)
            exact(identity(mean['raw_logits']),mean['authority']['mean_logits'],'persisted_mean_authority')
            for key in report['export_receipts'][-1]:exact(mean[key],report['export_receipts'][-1][key],'persisted_emitted/'+key)
            require(bool(torch.isfinite(value['raw_logits']).all())and bool(torch.isfinite(mean['raw_logits']).all()),'Nonfinite persisted export');return value
        with observer_module.observe(run_fit,job,output/'reporting_observer',binding_spec=None) as reporting_observer:
            invoke('controlled_completed_epoch1',maps,interrupt=True);snapshot('after_controlled_epoch1')
            resumed=invoke('resume_remaining_epoch2',maps,resume=folder/'recovery.pt');require(resumed['training_epochs_this_invocation']==1,'Resume must train only epoch2')
            first=own_export();snapshot('after_resumed_epoch2');selected_pin,recovery_pin=receipt(folder/'selected.pt'),receipt(folder/'recovery.pt')
            final=invoke('stopped_finalization_resume',0,resume=folder/'recovery.pt');require(final['finalization_only_resume']and final['training_epochs_this_invocation']==0,'Stopped resume performed TRAIN')
            second=own_export();snapshot('after_stopped_finalization')
            # Only authority/state/role identity across required finalizations; no numerical replay/tolerance test.
            for key in('job_hash','selected_epoch','selected_checkpoint','ids','labels','query_endpoints','policy'):exact(first[key],second[key],'stopped_actual_identity/'+key)
            for key in('job_hash','job_id','selected_epoch','selected_checkpoint','runtime_binding','ids','labels','query_endpoints'):exact(first['same_bank_mean_VALID'][key],second['same_bank_mean_VALID'][key],'stopped_mean_identity/'+key)
            exact(receipt(folder/'selected.pt'),selected_pin,'unchanged_selected_checkpoint');exact(receipt(folder/'recovery.pt'),recovery_pin,'unchanged_recovery_checkpoint')
            curve=[json.loads(line)for line in(folder/'CURVE.jsonl').read_text().splitlines()];require(len(curve)==2 and[row['train']['epoch']for row in curve]==[1,2],'Exactly two completed-epoch rows')
            metric=[row['validation']['metrics']['hits@50']for row in curve];winner=1 if metric[0]>=metric[1]else 2
            require(first['selected_epoch']==second['selected_epoch']==winner,'Own actual first-best selector differs')
            report['actual_selector_audit']={'original_trajectory_hits_at_50':metric,'selected_epoch':winner,'first_ties':True,'mean_used_for_selection':False}
            for key in('native_constructions_attempted','TRAIN_epochs_started','updates_attempted','optimizer_attempts','reverse_attempts','VALID_attempts','mean_exports_attempted'):require(report[key]==limits[key],'Final hard-bound count differs: '+key)
            for key,expected in{'native_constructions_completed':3,'TRAIN_epochs_completed':2,'updates_completed':2*maps,'optimizer_completed':2*maps,'reverse_completed':2*maps,'extra_reverse_attempts':0,'VALID_completed':4,'native_VALID_attempts':4,'native_VALID_completed':4,'mean_exports_completed':2}.items():require(report[key]==expected,'Final work count differs: '+key)
            exact(report['native_forwards'],{'TRAIN':8*maps,'VALID':16 if family=='NCN'else 32},'actual native forward totals')
            exact(report['native_forward_attempts'],report['native_forwards'],'all_native_forwards_completed')
            require(report['TRAIN_task_query_records']==case['total_exact_work']['TRAIN_task_query_records']and report['TRAIN_source_iterator_records']==case['total_exact_work']['TRAIN_source_iterator_records'],'Full TRAIN population/tail charge differs')
            require(len(report['restore_checks'])==4 and len(report['global_RNG_restore_checks'])==4,'All recovery/selected restores required')
            if family=='BUDDY':require(report['provider_sampler_calls']==report['provider_TRAIN_hydration_calls']==3170 and report['provider_VALID_hydration_calls']==8,'Production provider work counters differ')
            report['sources_after']=sources(here,admission['driver_manifest_sha256']);exact(report['sources_before'],report['sources_after'],'frozen_source_preservation')
            report.update(status='passed',stage='terminal',no_duplicate_uninterrupted_trajectory=True,no_numerical_reexecution_test=True,scientific_convergence_inferred=False)
        observed_index=json.loads((output/'reporting_observer/INDEX.json').read_text())
        require(observed_index['exports']==2 and observed_index['invocations']==3 and observed_index['extra_native_work']==0 and observed_index['RNG_calls']==0 and observed_index['tensor_deserializations']==0,'Reporting observer count/scope differs')
        report['reporting_observer_index']=receipt(output/'reporting_observer/INDEX.json');write()
    except BaseException as error:report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=24)})
    finally:
        signal.alarm(0);report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text();write()
    print(json.dumps({key:report.get(key)for key in('status','case_id','stage','elapsed_seconds','optimizer_completed','reverse_completed','VALID_completed','failure')}),flush=True)
    return 0 if report['status']=='passed'else 1


if __name__=='__main__':sys.exit(main())

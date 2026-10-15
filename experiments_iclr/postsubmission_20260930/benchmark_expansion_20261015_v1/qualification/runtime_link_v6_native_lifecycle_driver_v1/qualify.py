"""One actual frozen NCNC worker trajectory: durable epoch interruption and two resumes."""
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
from support import guard, inside, receipt, require, sources, load_admission, LIMITS, ROOT, atomic_json


class ControlledCompletedEpochStop(BaseException):
    pass


class EngineeringWallLimit(BaseException):
    pass


def main():
    guard()  # Before project reads, tensor imports, or source/data loading.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    here = inside(Path(__file__).parent)
    admission, job_path = load_admission(args.admission, here)
    device = admission['device']; output = inside(args.output)
    require(not output.exists(), 'Preserve prior attempt; fresh qualifier output required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if device=='cpu' else '0'), 'Explicit admitted device visibility required')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema':'runtime-link-v6-native-NCNC-lifecycle-result-v1','status':'running','stage':'imports',
        'device':device,'admission':receipt(args.admission),'job':receipt(job_path),
        'script':receipt(__file__),'limits':LIMITS,'native_constructions_attempted':0,
        'native_constructions_completed':0,'TRAIN_epochs_started':0,'TRAIN_epochs_completed':0,
        'updates_attempted':0,'updates_completed':0,'optimizer_attempts':0,'optimizer_completed':0,
        'reverse_attempts':0,'reverse_completed':0,'extra_reverse_attempts':0,
        'VALID_attempts':0,'VALID_completed':0,'native_VALID_attempts':0,'native_VALID_completed':0,
        'mean_exports_attempted':0,'mean_exports_completed':0,'encoder_forwards':{'TRAIN':0,'VALID':0},
        'restore_checks':[],'global_RNG_restore_checks':[],'update_receipts':[],'epoch_receipts':[],
        'VALID_receipts':[],'export_receipts':[],'invocations':[],'snapshots':[],
        'scientific_fits':0,'TEST_accessed':False,'automatic_retries':0,'full_campaign_GO':False,
        'memory_events_before':Path('/sys/fs/cgroup/memory.events').read_text()}
    tensor_library = None
    live = {'phase':None,'objective':None,'owner':None}

    def write():
        use = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter()-started,process_user_seconds=use.ru_utime,
            process_system_seconds=use.ru_stime,process_peak_RSS_KiB=use.ru_maxrss)
        if tensor_library is not None:
            report['CUDA_context_initialized'] = tensor_library.cuda.is_initialized()
            if device!='cpu' and tensor_library.cuda.is_initialized():
                report['cuda_peak_allocated_bytes'] = max(report.get('cuda_peak_allocated_bytes',0),tensor_library.cuda.max_memory_allocated())
                report['cuda_peak_reserved_bytes'] = max(report.get('cuda_peak_reserved_bytes',0),tensor_library.cuda.max_memory_reserved())
        atomic_json(output/'RESULT.json',report)

    def begin(key):
        require(report[key] < LIMITS[key], 'Hard bound exhausted before '+key)
        report[key] += 1; write()

    def limit(*_):
        raise EngineeringWallLimit('One admitted lifecycle wall budget exhausted')

    signal.signal(signal.SIGTERM,limit); signal.signal(signal.SIGALRM,limit)
    signal.alarm(5400 if device=='cpu' else 1800)
    if device=='cpu':
        resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
    write()
    try:
        runtime = here.parents[1]/'runtime_link_v6'
        report['sources_before'] = sources(here,admission['driver_manifest_sha256'])
        sys.path.insert(0,str(runtime))
        import common
        import adapters
        import run_fit
        import torch
        import numpy as np
        tensor_library = torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        report.update(torch=str(torch.__version__),threads=2,interop_threads=1)
        job = json.loads(job_path.read_text())
        require(job['device']==device and job['procedure']=='interaction_input_conditionalJ'
            and job['seed']==0 and job['split']==0 and job['engineering_only'] is True,'Fixed native conditional-J job required')
        run_fit.validate_job(job,ROOT)
        report['job_hash'] = common.digest(job)

        def exact(a,b,name):
            if torch.is_tensor(a):
                require(torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
                    and torch.equal(a.detach().cpu(),b.detach().cpu()),'Exact tensor differs: '+name)
            elif isinstance(a,np.ndarray):
                require(isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype
                    and np.array_equal(a,b),'Exact array differs: '+name)
            elif isinstance(a,dict):
                require(isinstance(b,dict) and set(a)==set(b),'Exact mapping differs: '+name)
                for k in a:exact(a[k],b[k],name+'/'+str(k))
            elif isinstance(a,(tuple,list)):
                require(type(a) is type(b) and len(a)==len(b),'Exact sequence differs: '+name)
                for i,(x,y) in enumerate(zip(a,b)):exact(x,y,name+'/'+str(i))
            else:
                require(type(a) is type(b) and a==b,'Exact scalar differs: '+name)

        def identity(value):
            value=value.detach().cpu().contiguous()
            return {'shape':list(value.shape),'dtype':str(value.dtype),
                    'sha256':hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest()}

        original_create=adapters.create_adapter
        original_restore_rng=run_fit.restore_rng
        original_backward=torch.Tensor.backward
        original_grad=torch.autograd.grad

        def restored_rng(t,value,**kw):
            original_restore_rng(t,value,**kw)
            exact(value,common.global_rng(t,device=device),'all_global_RNG')
            report['global_RNG_restore_checks'].append({'exact':True,'invocation':len(report['invocations'])+1})
            write()

        def backward(tensor,*a,**kw):
            require(live['phase']=='TRAIN' and live['objective'] is tensor,
                'Only the actual one scalar-J TRAIN objective may call backward')
            begin('reverse_attempts')
            value=original_backward(tensor,*a,**kw)
            report['reverse_completed']+=1; write(); return value

        def forbidden_grad(*a,**kw):
            report['extra_reverse_attempts']+=1;write()
            raise ValueError('No additional gradient collection belongs to conditional-J')

        def measured_create(t,descriptor,policy,root):
            begin('native_constructions_attempted')
            owner=original_create(t,descriptor,policy,root)
            report['native_constructions_completed']+=1
            n=owner.session.native
            require(len(owner.optimizers)==1 and n.members==4 and int(n.positive.shape[1])==1179052,
                'One original four-route NCNC Adam and complete source TRAIN required')
            layout=owner.session.layout
            expected={'family':'NCNC','batch_size':65536,'optimizer_maps_per_epoch':17,
                'source_iterator_records_per_epoch':1114112,'task_query_records_per_epoch':2228224,
                'dropped_positive_tail':64940,'included_partial_tail':0}
            exact(layout,expected,'native_epoch_layout')
            require(all(p.device.type==('cpu' if device=='cpu' else 'cuda') for p in owner.model.parameters()),'Native model device differs')
            annotated=owner.metadata()['native_integration']['native_binding']['adapter_manifest_receipt']
            require(annotated['adapter_hashes_verified'] is True and owner.dependency_annotations=={annotated['path']:{'adapter_hashes_verified':True}},'Authentic native annotation must survive canonical receipt dispatch')
            report.setdefault('construction_dependency_annotations',[]).append(copy.deepcopy(owner.dependency_annotations))
            source_pins=[owner.callback_receipt,owner.extension_receipt,owner.policy_receipt,*owner.dependency_receipts]
            for pin in source_pins:exact(receipt(pin['path']),pin,'source_receipt')
            report.setdefault('construction_sources',[]).append(source_pins)
            original_train,original_valid=owner.train_epoch,owner.validate
            original_update=owner.session.train_update
            original_native_valid=owner.session.validate
            original_load=owner.load_state
            original_export=owner.selected_mean_export
            original_step=owner.optimizers[0].step
            original_build=owner.session.objectives.build
            original_close=owner.close

            def encoder_hook(*_):
                require(live['phase'] in ('TRAIN','VALID'),'Unexpected native encoder call outside admitted phase')
                report['encoder_forwards'][live['phase']]+=1

            handle=n.encoder.register_forward_hook(encoder_hook)

            def build(*a,**kw):
                result=original_build(*a,**kw)
                require(result['scalar_update'] and result['shared_objective'] is result['private_objective']
                    and result['shared_objective'] is result['logged_objective'],'One authentic scalar-J objective required')
                live['objective']=result['shared_objective']
                return result

            def step(*a,**kw):
                require(live['phase']=='TRAIN','Optimizer outside TRAIN')
                begin('optimizer_attempts')
                value=original_step(*a,**kw)
                report['optimizer_completed']+=1;write();return value

            def update():
                begin('updates_attempted')
                before=(report['optimizer_completed'],report['reverse_completed'],n.updates,report['encoder_forwards']['TRAIN'])
                value=original_update()
                after=(report['optimizer_completed'],report['reverse_completed'],n.updates,report['encoder_forwards']['TRAIN'])
                require(tuple(b-a for a,b in zip(before,after))==(1,1,1,4),'One native update/Adam/backward/four-encoder route required')
                work=value['work']; routing=work['NCNC_output_routing']
                observation=value['interaction_diagnostics']['NCNC_top_level_observer']
                require(work['positive_query_records']==65536 and work['negative_query_records']==65536
                    and work['reverse_collections']==1 and work['optimizer_transitions']==1
                    and work['extra_source_forward_views']==0 and work['captured_source_query_records']==131072,
                    'Original full native update work differs')
                require(routing['task_only_output_scope_verified'] and observation['task']==8
                    and observation['task']==routing['dispatch_counts']['task']
                    and observation['recursive_ignored']==routing['dispatch_counts']['recursive']
                    and observation['recursive_ignored']>0 and observation['recursive_query_objects_used_in_AUX'] is False
                    and not owner.session._depth and not n.task_call_scope.depth,'NCNC recursive output routing differs')
                require(owner.session._last_VALID is None,'TRAIN retained stale VALID authority')
                indices=n.permutation[n.cursor-n.batch_size:n.cursor] if hasattr(n,'permutation') else n.sampler_state()['permutation'][n.cursor-n.batch_size:n.cursor]
                report['updates_completed']+=1
                report['update_receipts'].append({'trajectory_update':n.updates,'native_epoch':n.epoch,'cursor':n.cursor,
                    'native_sampler_indices':identity(indices),'source_query_recipe':value['actual_source_recipe_receipt'],
                    'work':work,'top_level_observer':observation,'elapsed_seconds':time.perf_counter()-started})
                live['objective']=None;write();return value

            def train():
                begin('TRAIN_epochs_started');live['phase']='TRAIN'
                before=report['updates_completed']
                try:value=original_train()
                finally:live['phase']=None;live['objective']=None;write()
                require(value['complete_native_TRAIN'] is True and value['optimizer_maps']==17
                    and value['native_epoch_optimizer_maps']==17 and value['native_source_epoch_loss_complete'] is True
                    and report['updates_completed']-before==17,'Complete authentic 17-map epoch required')
                exact(value['native_work_totals'],{'optimizer_transitions':17,'encoder_forwards':68,'reverse_collections':17},'epoch_native_work')
                report['TRAIN_epochs_completed']+=1
                report['epoch_receipts'].append({'native_layout':layout,'returned_epoch':value});write();return value

            def native_valid():
                require(report['native_VALID_attempts']<4,'No fifth native VALID')
                report['native_VALID_attempts']+=1;write()
                value=original_native_valid();report['native_VALID_completed']+=1;return value

            def valid():
                begin('VALID_attempts');live['phase']='VALID'
                before=(report['encoder_forwards']['VALID'],report['optimizer_completed'],report['reverse_completed'],n.updates)
                try:value=original_valid()
                finally:live['phase']=None;write()
                after=(report['encoder_forwards']['VALID'],report['optimizer_completed'],report['reverse_completed'],n.updates)
                require(tuple(b-a for a,b in zip(before,after))==(4,0,0,0),'VALID work differs or changed TRAIN clock')
                require(value['role']=='VALID' and value['TEST_accessed'] is False and value['complete_source_VALID'] is True
                    and (value['complete_VALID_positive_records'],value['complete_VALID_negative_records'])==(60084,100000),'Complete official VALID required')
                exact(value['ids'],torch.arange(160084),'canonical VALID IDs')
                exact(value['labels'],torch.cat((torch.ones(60084,dtype=torch.long),torch.zeros(100000,dtype=torch.long))),'canonical VALID labels')
                exact(value['query_endpoints'],torch.cat((n.valid['edge'],n.valid['edge_neg'])).cpu(),'native VALID endpoint order')
                routing=value['work']['NCNC_output_routing']
                require(routing['task_only_output_scope_verified'] and not owner.session._depth and not n.task_call_scope.depth,'VALID task/recursive routing differs')
                side=value['same_bank_mean_VALID']
                require(side['extra_forward_calls']==0,'Extra mean inference')
                report['VALID_completed']+=1
                report['VALID_receipts'].append({'source_updates':n.updates,'completed_epoch':owner.completed_epochs,
                    'ids':identity(value['ids']),'labels':identity(value['labels']),
                    'query_endpoints':identity(value['query_endpoints']),'actual_logits':identity(value['outputs']),
                    'mean_logits':identity(side['outputs']),'actual_loss':value['loss'],'actual_metrics':value['metrics'],
                    'mean_loss':side['loss'],'mean_metrics':side['metrics'],'authority':side['authority'],'work':value['work']})
                write();return value

            def load_state(value):
                reference_ids={k:id(p) for k,p in n.source_parameter_references.items()}
                original_load(value)
                exact(value,owner.state(),'complete_native_model_Adam_sampler_scope_rng_modes_epoch')
                require(reference_ids=={k:id(p) for k,p in n.source_parameter_references.items()},'Restore replaced native parameters')
                require(owner.session._last_VALID is None and owner._cache is None,'Restore retained stale validation authority')
                report['restore_checks'].append({'exact':True,'completed_epoch':owner.completed_epochs,
                    'source_updates':n.updates,'state_signature_sha256':common.digest(owner._state_signature(value))})
                write()

            def export(actual,**context):
                begin('mean_exports_attempted')
                before=copy.deepcopy(report['encoder_forwards']);before_valid=report['native_VALID_attempts']
                value=original_export(actual,**context)
                require(value is not None and value['extra_forward_calls']==0,'Selected mean export missing or extra inference')
                exact(before,report['encoder_forwards'],'zero-extra-export-encoder-work')
                require(before_valid==report['native_VALID_attempts'],'Export triggered native VALID')
                for key in ('ids','labels','query_endpoints'):exact(value[key],actual[key],'emitted_mean/'+key)
                exact(value['raw_logits'],actual['same_bank_mean_VALID']['outputs'],'emitted_same_call_mean')
                exact(value['authority'],actual['same_bank_mean_VALID']['authority'],'emitted_authority')
                report['mean_exports_completed']+=1
                report['export_receipts'].append({k:copy.deepcopy(value[k]) for k in
                    ('job_hash','job_id','selected_epoch','selected_checkpoint','runtime_binding','authority','loss','metrics')})
                write();return value

            def close():
                handle.remove();owner.session.objectives.build=original_build
                original_close()

            owner.train_epoch,owner.validate,owner.load_state= train,valid,load_state
            owner.session.train_update,owner.session.validate=update,native_valid
            owner.selected_mean_export,owner.close=export,close
            owner.optimizers[0].step=step;owner.session.objectives.build=build
            write();return owner

        adapters.create_adapter=measured_create
        run_fit.restore_rng=restored_rng
        torch.Tensor.backward=backward;torch.autograd.grad=forbidden_grad
        folder=output/'own_fit'

        def snapshot(name):
            destination=output/'snapshots'/name;destination.mkdir(parents=True)
            rows=[]
            for filename in ('JOB.json','RESULT.json','CURVE.jsonl','EVENTS.jsonl','PROGRESS.json','selected.pt','recovery.pt','selected_VALID.pt'):
                path=folder/filename
                if path.exists():
                    shutil.copy2(path,destination/filename)
                    original,copied=receipt(path),receipt(destination/filename)
                    require(original['sha256']==copied['sha256'],'Snapshot bytes differ')
                    rows.append({'original':original,'copy':copied})
            report['snapshots'].append({'name':name,'files':rows});write()

        def invoke(name,expected_updates,resume=None,interrupt=False):
            previous=run_fit.atomic_json;interrupted=False
            def durable_progress(path,value):
                nonlocal interrupted
                previous(path,value)
                if interrupt and Path(path)==folder/'PROGRESS.json' and value['train']['epoch']==1:
                    # The source's atomic writes are now synced before the declared fault.
                    for filename in ('recovery.pt','selected.pt','CURVE.jsonl','PROGRESS.json'):
                        with (folder/filename).open('rb') as stream:os.fsync(stream.fileno())
                    fd=os.open(folder,os.O_RDONLY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                    interrupted=True
                    raise ControlledCompletedEpochStop('Declared fault after durable epoch1 recovery/CURVE/PROGRESS')
            report['stage']=name;write();begin_time=time.perf_counter()
            if interrupt:run_fit.atomic_json=durable_progress
            try:result=run_fit.run(copy.deepcopy(job),ROOT,folder,resume=resume)
            finally:run_fit.atomic_json=previous
            report['invocations'].append({'name':name,'outer_wall_seconds':time.perf_counter()-begin_time,
                'expected_controlled_interruption':interrupt,'result':copy.deepcopy(result),'result_receipt':receipt(folder/'RESULT.json')})
            write()
            require(result['optimizer_calls_this_invocation']=={'attempts':[expected_updates],'completed':[expected_updates]},'Invocation optimizer costs differ')
            if interrupt:
                require(interrupted and result['status']=='failed' and result['failure']['type']=='ControlledCompletedEpochStop'
                    and result['completed_epochs']==1 and result['training_epochs_this_invocation']==1,'Expected durable epoch1 fault was not preserved')
            else:require(result['status']=='complete','Unexpected native worker failure: '+str(result.get('failure')))
            exact(json.loads((folder/'JOB.json').read_text()),job,'immutable JOB')
            gc.collect()
            if device!='cpu':torch.cuda.empty_cache()
            return result

        def own_export():
            value=torch.load(folder/'selected_VALID.pt',map_location='cpu',weights_only=False)
            checkpoint=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
            mean=value['same_bank_mean_VALID'];pin=receipt(folder/'selected.pt');epoch=checkpoint['selector']['selected_epoch']
            require(value['schema']=='own-selected-VALID-logits-v1' and mean['schema']=='own-selected-same-bank-mean-VALID-v1'
                and value['role']==mean['role']=='VALID' and value['TEST_accessed'] is False and mean['TEST_accessed'] is False
                and epoch in (1,2) and mean['extra_forward_calls']==0,'Persisted actual/mean structure differs')
            for obj in (value,mean):
                require(obj['job_hash']==common.digest(job) and obj['selected_epoch']==epoch
                    and obj['selected_checkpoint']==pin
                    and (obj['complete_VALID_positive_records'],obj['complete_VALID_negative_records'])==(60084,100000),
                    'Persisted JOB/checkpoint/epoch/VALID authority differs')
            exact(mean['runtime_binding'],checkpoint['runtime_binding'],'export_runtime_binding')
            for key in ('ids','labels','query_endpoints'):exact(value[key],mean[key],'persisted_actual_mean/'+key)
            for key,name in [('ids','ids'),('labels','labels'),('query_endpoints','query_endpoints'),('actual_logits','raw_logits')]:
                exact(identity(value[name]),mean['authority'][key],'persisted_actual_authority/'+key)
            exact(identity(mean['raw_logits']),mean['authority']['mean_logits'],'persisted_mean_authority')
            for key in report['export_receipts'][-1]:exact(mean[key],report['export_receipts'][-1][key],'persisted_emitted/'+key)
            require(bool(torch.isfinite(value['raw_logits']).all()) and bool(torch.isfinite(mean['raw_logits']).all()),'Nonfinite persisted logits')
            return value

        def floating(a,b,name):
            require(a.shape==b.shape and a.dtype==b.dtype,'Repeated floating shape/dtype differs: '+name)
            aa,bb=a.detach().cpu().double(),b.detach().cpu().double();delta=(aa-bb).abs()
            result={'name':name,'bit_equal':bool(torch.equal(a,b)),
                'max_absolute_error':float(delta.max()),
                'max_relative_error':float((delta/bb.abs().clamp_min(1e-30)).max()),
                'rtol':1e-4,'atol':1e-5,'allclose':bool(torch.allclose(aa,bb,rtol=1e-4,atol=1e-5))}
            report.setdefault('repeated_inference',[]).append(result);write()
            require(result['allclose'],'Fixed repeated-inference tolerance failed: '+name)

        invoke('controlled_completed_epoch1',17,interrupt=True)
        snapshot('after_controlled_epoch1')
        resumed=invoke('resume_remaining_epoch2',17,resume=folder/'recovery.pt')
        require(resumed['training_epochs_this_invocation']==1,'Resume must train only epoch2')
        first=own_export();snapshot('after_resumed_epoch2')
        selected_pin,recovery_pin=receipt(folder/'selected.pt'),receipt(folder/'recovery.pt')
        final=invoke('stopped_finalization_resume',0,resume=folder/'recovery.pt')
        require(final['finalization_only_resume'] and final['training_epochs_this_invocation']==0,'Stopped resume performed TRAIN')
        second=own_export();snapshot('after_stopped_finalization')
        for key in ('job_hash','selected_epoch','selected_checkpoint','ids','labels','query_endpoints','policy'):
            exact(first[key],second[key],'repeated_actual_identity/'+key)
        for key in ('job_hash','job_id','selected_epoch','selected_checkpoint','runtime_binding','ids','labels','query_endpoints'):
            exact(first['same_bank_mean_VALID'][key],second['same_bank_mean_VALID'][key],'repeated_mean_identity/'+key)
        floating(first['raw_logits'],second['raw_logits'],'selected actual logits')
        floating(first['same_bank_mean_VALID']['raw_logits'],second['same_bank_mean_VALID']['raw_logits'],'selected mean logits')
        for label,one,two in [('actual',first['validation'],second['validation']),
                             ('mean',first['same_bank_mean_VALID'],second['same_bank_mean_VALID'])]:
            exact(set(one['metrics']),set(two['metrics']),label+' metric names')
            values1=[float(one['loss'])]+[float(one['metrics'][k]) for k in sorted(one['metrics'])]
            values2=[float(two['loss'])]+[float(two['metrics'][k]) for k in sorted(two['metrics'])]
            floating(torch.tensor(values1,dtype=torch.float64),torch.tensor(values2,dtype=torch.float64),label+' loss/metrics')
        exact(receipt(folder/'selected.pt'),selected_pin,'unchanged selected checkpoint')
        exact(receipt(folder/'recovery.pt'),recovery_pin,'unchanged recovery checkpoint')
        curve=[json.loads(line) for line in (folder/'CURVE.jsonl').read_text().splitlines()]
        require(len(curve)==2 and [row['train']['epoch'] for row in curve]==[1,2],'Only two completed epoch curve rows')
        metric=[row['validation']['metrics']['hits@50'] for row in curve]
        winner=1 if metric[0]>=metric[1] else 2
        require(first['selected_epoch']==second['selected_epoch']==winner,'Selection must be first best actual original trajectory VALID')
        report['actual_selector_audit']={'original_trajectory_hits_at_50':metric,'selected_epoch':winner,'first_ties':True,'mean_used_for_selection':False}
        expected={'native_constructions_attempted':3,'native_constructions_completed':3,'TRAIN_epochs_started':2,'TRAIN_epochs_completed':2,
            'updates_attempted':34,'updates_completed':34,'optimizer_attempts':34,'optimizer_completed':34,
            'reverse_attempts':34,'reverse_completed':34,'extra_reverse_attempts':0,
            'VALID_attempts':4,'VALID_completed':4,'native_VALID_attempts':4,'native_VALID_completed':4,
            'mean_exports_attempted':2,'mean_exports_completed':2}
        for key,value in expected.items():require(report[key]==value,'Final exact work count differs: '+key)
        exact(report['encoder_forwards'],{'TRAIN':136,'VALID':16},'all actual encoder forwards')
        require(len(report['restore_checks'])==4 and len(report['global_RNG_restore_checks'])==4,'All recovery and selected restores must be checked')
        report['sources_after']=sources(here,admission['driver_manifest_sha256'])
        exact(report['sources_before'],report['sources_after'],'frozen source preservation')
        report.update(status='passed',stage='terminal',scientific_convergence_inferred=False,
            convergence_field_interpretation='Inherited patience-stop field is not scientific convergence evidence',
            no_extra_mean_inference=True,no_duplicate_uninterrupted_trajectory=True)
    except BaseException as error:
        report.update(status='failed',failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(limit=24)})
    finally:
        signal.alarm(0)
        report['memory_events_after']=Path('/sys/fs/cgroup/memory.events').read_text()
        write()
    print(json.dumps({key:report.get(key) for key in ('status','stage','elapsed_seconds','optimizer_completed','reverse_completed','VALID_completed','failure')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())

"""Five isolated CPU seed workers; exact sealed v2 fit and selection functions."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode=True
PACKET=Path(__file__).resolve().parent
SEEDS=(131,137,139,149,151)
ARMS=('native_HGT','global_BE','shared_relation','CP','unrestricted','untied_HGT','wider_BE')


class ControllerInterrupted(Exception):pass


def install_controller_signals():
    previous={number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
    def interrupt(number,frame):
        for owned in previous:signal.signal(owned,signal.SIG_IGN)
        raise ControllerInterrupted('Controller interrupted by signal '+str(number))
    for number in previous:signal.signal(number,interrupt)
    return previous


def restore_controller_signals(previous):
    for number,handler in previous.items():signal.signal(number,handler)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def verify(row):
    path=Path(row['path']);require(sha(path)==row['sha256'],'Fingerprint mismatch: '+str(path))
    if 'bytes' in row:require(path.stat().st_size==row['bytes'],'Byte length mismatch: '+str(path))
    return path


def write(path,value):
    with Path(path).open('x') as s:json.dump(value,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def packet_guard():
    manifest=json.loads((PACKET/'MANIFEST.json').read_text());msha=sha(PACKET/'MANIFEST.json')
    require(json.loads((PACKET/'SEAL.json').read_text())['manifest_sha256']==msha,'Scheduler seal mismatch')
    for r in manifest['payload']:verify(dict(r,path=str(PACKET/r['path'])))
    binding=json.loads((PACKET/'BINDINGS.json').read_text())
    for r in binding['source_records']:verify(r)
    frozen=json.loads(verify(binding['freeze']).read_text())
    driver=load('parallel_exact_v2_driver',verify(binding['driver_source']))
    return binding,frozen,driver,msha


def scheduler_preservation(expected):
    require(sha(PACKET/'MANIFEST.json')==expected,'Scheduler manifest changed during execution')
    for r in json.loads((PACKET/'MANIFEST.json').read_text())['payload']:verify(dict(r,path=str(PACKET/r['path'])))


def prior_failure_guard(binding,release):
    """Hash custody of the complete closed v1 failure; never load its tensors."""
    inventory_record=release['original_failed_run_inventory']
    require(inventory_record==binding['original_failed_run_inventory'],'Exact preserved original cohort inventory required')
    inventory=json.loads(verify(inventory_record).read_text())
    original=Path(binding['original_failed_run_directory'])
    require(inventory['schema']=='HGT35_closed_failure_inventory_v1'
            and inventory['root_observed'] is True and inventory['status']=='closed_incomplete'
            and inventory['original_run_directory']==str(original)
            and inventory['outcome_metrics_requested'] is False
            and inventory['prior_selected_checkpoints_reused'] is False,
            'Root-observed closed original failure inventory required')
    records=inventory['files']
    require(isinstance(records,list) and records and len({r['path'] for r in records})==len(records),
            'Exact original file inventory required')
    observed=set()
    for path in original.rglob('*'):
        require(not path.is_symlink(),'Original failed-run symlink is not admitted')
        if path.is_file():observed.add(str(path))
    require({r['path'] for r in records}==observed,'Complete original failure files must remain present')
    for row in records:
        path=Path(row['path'])
        require(path.is_absolute() and path.resolve()==path and original in path.parents
                and isinstance(row.get('bytes'),int) and not isinstance(row['bytes'],bool),
                'Original failure records require exact absolute custody')
        verify(row)
    old=json.loads((original/'PARALLEL_STUDY.json').read_text())
    study=json.loads((original/'STUDY.json').read_text())
    require(old['status']=='incomplete' and old['originals_preserved'] is True
            and old['final_labels_closed'] is True
            and [(r['seed'],r['arm']) for r in old['rows']]==[(s,a) for s in SEEDS for a in ARMS]
            and [(r['seed'],r['arm']) for r in study['rows']]==[(s,a) for s in SEEDS for a in ARMS]
            and all(r['status']==('selected' if r['arm']=='native_HGT' else 'resource_deferred')
                    for r in old['rows'])
            and [r['status'] for r in study['rows']]==[r['status'] for r in old['rows']]
            and study['summary']['status']=='incomplete'
            and study['summary']['successful_subset_scored'] is False
            and old['closure']['comparison'] is None,
            'Exact closed 5-selected/30-resource-deferred v1 cohort required')
    binding['source_records']=binding['source_records']+[inventory_record]+records
    return inventory_record


def runtime_qualification_guard(binding,release):
    """Root-observed six-update evidence replaces insufficient v1 one-step evidence."""
    qualification=release['resource_qualification']
    require(qualification['sha256']==release['resource_qualification_sha256'],
            'Root qualification descriptor/hash differs')
    qualified=json.loads(verify(qualification).read_text())
    transport_records=release['resource_transport']
    transport=json.loads(verify(transport_records['receipt']).read_text())
    verify(transport_records['wrapper']);verify(transport_records['remote_code'])
    require(transport_records['wrapper']['sha256']==binding['qualified_resource_wrapper_sha256'],
            'Exact sealed six-update qualification wrapper required')
    actual=transport['remote_receipt'];preflight=actual['preflight']
    require(release['root_observed_resource_qualification'] is True
            and transport['exit_code']==0 and transport['helper_sha256']==transport_records['wrapper']['sha256']
            and actual['exit_code']==0 and actual['status']=='completed'
            and preflight['memory_limit_bytes']==16*2**30 and preflight['RSS_limit_bytes']==14*2**30
            and preflight['CPU_threads']==1 and preflight['wall_limit_seconds']==1200 and preflight['CUDA_VISIBLE_DEVICES']==''
            and preflight['GPU_computation'] is False and actual['qualification_result']==qualified,
            'Actual bounded preimport16GiB/six-update CPU transport required')
    require(qualified['status']=='qualified' and qualified['device']=='cpu'
            and qualified['originals_preserved'] is True
            and qualified['paired_CP_global_untied_geometry_verified'] is True
            and qualified['study_freeze_sha256']==binding['freeze']['sha256']
            and qualified['qualification_only'] is True and qualified['training_driver_main_called'] is False
            and qualified['validation_or_test_scored'] is False and qualified['model_selection'] is False
            and qualified['scientific_study_outcomes'] is False
            and [r['arm'] for r in qualified['rows']]==list(ARMS)
            and all(r['status']=='qualified' and r['optimizer_updates']==6
                    and r['full_state_rng_custody'] is True
                    and isinstance(r['per_update_memory'],list) and len(r['per_update_memory'])==6
                    and r['validation_or_test_scored'] is False and r['model_selection'] is False
                    for r in qualified['rows']),
            'All-seven six-consecutive-update score-free full-graph qualification required')
    for row in qualified['rows']:
        require([item['update'] for item in row['per_update_memory']]==list(range(1,7)),
                'Exactly six consecutive observed updates required')
        for item in row['per_update_memory']:
            for phase in ('before_TRAIN','after_TRAIN','after_eval_and_checkpoint'):
                memory=item[phase]
                require(all(isinstance(memory[field],int) and not isinstance(memory[field],bool)
                            and memory[field]>0 for field in ('VmSize_bytes','VmPeak_bytes','VmRSS_bytes'))
                        and memory['VmSize_bytes']<=16*2**30 and memory['VmPeak_bytes']<=16*2**30
                        and memory['VmRSS_bytes']<=14*2**30,
                        'Per-update virtual/resident memory evidence exceeds declared caps or is absent')
    binding['resource_qualification']=qualification
    binding['source_records']=binding['source_records']+list(transport_records.values())
    return qualification


def admission_guard(binding,frozen,driver,release,scheduler_sha,run_name):
    require(tuple(frozen['seeds'])==SEEDS and tuple(frozen['arms'])==ARMS,'Exact five-seed/seven-arm order required')
    driver.freeze_guard(frozen,release,binding['freeze']['sha256'],binding['training_manifest_sha256'])
    require(release['run_name']==run_name and release['device']=='cpu' and release['mode']=='parallel_CPU_five_seeds'
            and release['scheduler_manifest_sha256']==scheduler_sha
            and release['parallel_workers']==5 and release['threads_per_worker']==1
            and release['worker_address_space_limit_bytes']==16*2**30
            and release['worker_RSS_limit_bytes']==14*2**30
            and release['runtime_repair_only'] is True and release['fresh_complete35_rerun'] is True
            and release['resume_from_prior_selected_checkpoints'] is False,
            'Exact fresh runtime-only CPU repair release required')
    require(isinstance(release['worker_wall_budget_seconds'],int) and not isinstance(release['worker_wall_budget_seconds'],bool)
            and release['worker_wall_budget_seconds']>0,'Explicit positive worker wall budget required')
    require(isinstance(release['required_free_host_bytes'],int) and not isinstance(release['required_free_host_bytes'],bool)
            and release['required_free_host_bytes']>=96*2**30,'At least96GiB current host headroom required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','All CPU workers require CUDA hidden')
    runtime_qualification_guard(binding,release)
    preserved=prior_failure_guard(binding,release)
    return dict(study_freeze_sha256=binding['freeze']['sha256'],prepared_manifest_sha256=binding['training_manifest_sha256'],
        scheduler_manifest_sha256=scheduler_sha,resource_qualification_sha256=binding['resource_qualification']['sha256'],
        original_failed_run_inventory_sha256=preserved['sha256'],runtime_repair_only=True,
        fresh_complete35_rerun=True,resume_from_prior_selected_checkpoints=False,
        test_labels_closed=True,device='cpu',parallel_workers=5,threads_per_worker=1)


def resource_screen(required_bytes):
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):values['host_available_bytes']=int(line.split()[1])*1024
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':values['cgroup_available_bytes']=int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text())
    except (OSError,ValueError):pass
    free=min(values.values());affinity=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
    reasons=[]
    if free<required_bytes:reasons.append('insufficient_five_worker_host_memory')
    if affinity<5 or load>.75*affinity:reasons.append('current_CPU_contention')
    return dict(status='resource_deferred' if reasons else 'passed',reasons=reasons,
        free_bytes=free,required_free_bytes=required_bytes,affinity_CPUs=affinity,one_minute_load=load,**values)


def missing_rows(seed,reason,status='resource_deferred',attempted=False):
    return [dict(seed=seed,arm=arm,status=status,attempted=attempted,reason=reason,
        final_labels_closed=True,successful_subset_scored=False) for arm in ARMS]


def collect(seed,seed_out,exit_record):
    path=seed_out/'ARM_TERMINALS.jsonl';rows=[]
    started=set();started_path=seed_out/'ARM_STARTED.jsonl'
    if started_path.exists():
        try:started_lines=started_path.read_text().splitlines()
        except (OSError,UnicodeError):started_lines=[]
        for line in started_lines:
            try:record=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(record,dict) and record.get('seed')==seed and record.get('arm') in ARMS:started.add(record['arm'])
    def invalid(reason):
        failed=missing_rows(seed,reason,status='failed')
        for row in failed:row['attempted']=row['arm'] in started
        return failed
    if path.exists():
        try:lines=path.read_text().splitlines()
        except (OSError,UnicodeError):return invalid('unreadable_worker_receipt')
        for index,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:break  # Partial last line retained as evidence.
                return invalid('malformed_worker_receipt')
            if not isinstance(row,dict) or row.get('seed')!=seed or row.get('arm') not in ARMS:
                return invalid('wrong_seed_arm_or_shape_worker_receipt')
            if row['arm'] in started:row['attempted']=True
            rows.append(row)
    expected=[a for a in ARMS if any(r['arm']==a for r in rows)]
    if len({r['arm'] for r in rows})!=len(rows) or [r['arm'] for r in rows]!=expected or any(r.get('status') not in ('selected','failed','resource_deferred') for r in rows):
        return invalid('invalid_duplicate_order_or_nonterminal_worker_receipt')
    by_arm={r['arm']:r for r in rows}
    worker_failure=exit_record['exit_code']!=0 or exit_record['status']!='completed'
    for arm in ARMS:
        if arm not in by_arm:by_arm[arm]=dict(seed=seed,arm=arm,status='resource_deferred' if worker_failure else 'failed',
            attempted=arm in started,reason='worker_missing_terminal',worker_exit=exit_record,final_labels_closed=True,successful_subset_scored=False)
    return [by_arm[arm] for arm in ARMS]


def closure(rows,driver,binding,run_out,comparison_permitted=True):
    require([(r['seed'],r['arm']) for r in rows]==[(s,a) for s in SEEDS for a in ARMS],'Exactly35 ordered terminals required')
    require(all(r['status'] in ('selected','failed','resource_deferred') for r in rows),'Nonterminal worker row')
    selected=all(r['status']=='selected' and r.get('selected_state_replay') is True and r.get('checkpoint_bindings_verified') is True for r in rows)
    for r in rows:
        if r['status']!='selected':continue
        for key,name in (('selected_checkpoint','selected.pt'),('selected_logits','selected_member_logits.pt'),
                         ('selection_receipt','SELECTION.json'),('training_trace','TRACE.jsonl')):
            expected=run_out.resolve()/f"seed{r['seed']}"/r['arm']/name
            require(Path(r[key]['path']).is_absolute() and Path(r[key]['path'])==expected
                    and expected.resolve()==expected,
                    'Selected artifact outside exact seed/arm output slot')
            require(isinstance(r[key].get('bytes'),int) and not isinstance(r[key]['bytes'],bool)
                    and r[key]['bytes']>=0 and isinstance(r[key].get('sha256'),str),
                    'Every selected artifact requires hash and byte count')
            verify(r[key])
        original_selection=json.loads(Path(r['selection_receipt']['path']).read_text())
        require(original_selection.get('status')=='selected' and original_selection.get('seed')==r['seed']
                and original_selection.get('arm')==r['arm'] and 'selection' in original_selection
                and all(r.get(key)==value for key,value in original_selection.items()),
                'Terminal row differs from byte-bound original v2 selection receipt')
        require(r['checkpoint_binding_sha256']==binding['study_checkpoint_binding_sha256'],'Selected checkpoint study/scheduler custody differs')
    if not selected or not comparison_permitted:return dict(status='incomplete',all35_terminals=True,all35_selected_fits=selected,
        comparison_permitted=comparison_permitted,successful_subset_scored=False,comparison=None,final_labels_closed=True)
    return dict(status='complete_development_summary',all35_terminals=True,all35_selected_fits=True,
        successful_subset_scored=False,comparison=driver.paired_development(rows,list(ARMS)),final_labels_closed=True)


def run_workers(commands,out,wall_seconds,memory_limit_bytes,env,limit_resources=True,rss_limit_bytes=None):
    """Only orchestration: no Torch import or numerical functions in controller."""
    children={};records={};logs={}
    try:
        for seed,command in commands:
            stdout=(out/f'worker{seed}.stdout.log').open('w');stderr=(out/f'worker{seed}.stderr.log').open('w');logs[seed]=(stdout,stderr)
            # Defer controller signals until the new owned PID is registered.
            previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,(signal.SIGINT,signal.SIGTERM))
            def limits():
                signal.signal(signal.SIGINT,signal.SIG_DFL);signal.signal(signal.SIGTERM,signal.SIG_DFL)
                signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
                if limit_resources:resource.setrlimit(resource.RLIMIT_AS,(memory_limit_bytes,memory_limit_bytes))
            try:
                process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,preexec_fn=limits)
                children[seed]=dict(process=process,start=time.monotonic(),peak_RSS_bytes=0)
            except Exception as error:
                records[seed]=dict(status='spawn_failed',exit_code=1,error_type=type(error).__name__,error_message=str(error))
                write(out/f'worker{seed}.exit.json',records[seed])
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
        while any(seed not in records for seed in children):
            for seed,child in children.items():
                if seed in records:continue
                process=child['process'];elapsed=time.monotonic()-child['start'];status=None
                try:
                    for line in Path('/proc',str(process.pid),'status').read_text().splitlines():
                        if line.startswith('VmRSS:'):child['peak_RSS_bytes']=max(child['peak_RSS_bytes'],int(line.split()[1])*1024)
                except OSError:pass
                if process.poll() is not None:status='completed'
                elif elapsed>wall_seconds:status='wall_budget_exhausted'
                elif child['peak_RSS_bytes']>(memory_limit_bytes if rss_limit_bytes is None else rss_limit_bytes):status='RSS_budget_exhausted'
                if status:
                    if process.poll() is None:
                        process.terminate()
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                    records[seed]=dict(status=status,exit_code=process.returncode,wall_seconds=elapsed,peak_observed_RSS_bytes=child['peak_RSS_bytes'])
                    write(out/f'worker{seed}.exit.json',records[seed])
            if any(seed not in records for seed in children):time.sleep(.5)
    finally:
        for seed,child in children.items():
            if child['process'].poll() is None:
                child['process'].terminate()
                try:child['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:child['process'].kill();child['process'].wait()
            if seed not in records:
                records[seed]=dict(status='controller_aborted',exit_code=child['process'].returncode,
                    wall_seconds=time.monotonic()-child['start'],peak_observed_RSS_bytes=child['peak_RSS_bytes'])
                write(out/f'worker{seed}.exit.json',records[seed])
        for stdout,stderr in logs.values():stdout.close();stderr.close()
    return records


def worker(args,binding,frozen,driver,admission):
    require(args.worker_seed in SEEDS,'Unknown seed worker')
    run_out=PACKET/'runs'/args.run_name
    require(args.controller_out==str(run_out.resolve()),'Worker output must belong to admitted controller')
    marker=json.loads((run_out/'STUDY_STARTED.json').read_text());require(marker['root_release_sha256']==sha(args.admission),'Worker root release custody differs')
    seed=args.worker_seed;out=run_out/f'seed{seed}';out.mkdir(exist_ok=False)
    before=list(binding['source_records'])+[binding['freeze'],binding['resource_qualification'],frozen['archive'],frozen['development_labels'],
        dict(path=str(Path(args.admission).resolve()),sha256=admission['root_release_sha256'],bytes=admission['root_release_bytes'])]
    split_record=next(r for r in frozen['splits'] if r['seed']==seed);before.append(split_record['descriptor'])
    rows=[];result=dict(seed=seed,status='preparing',final_labels_closed=True,device='cpu');started=time.perf_counter();torch=None
    try:
        for r in before:verify(r)
        import torch
        require(torch.__version__=='2.1.2+cu118' and sys.version_info[:3]==(3,11,14)
                and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Exact qualified CPU-only runtime required')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        result['runtime']=dict(torch=torch.__version__,python=sys.version,threads=torch.get_num_threads(),pid=os.getpid(),CUDA_VISIBLE_DEVICES='')
        phase=Path(binding['canonical_phase']);v2=phase/'graph_heterogeneous_dblp_training_preparation_20261003_v2'
        inputs=load('worker_exact_DBLP_inputs',v2/'dblp_inputs.py');families=load('worker_exact_DBLP_families',v2/'families.py')
        onecycle=load('worker_exact_onecycle',v2/'onecycle_state.py');implementation=load('worker_exact_HGT',phase/binding['implementation_source'])
        schema,attributes,edges=inputs.stream_schema(frozen['archive'],frozen['members'])
        require(schema['member_sha256']==frozen['member_sha256'] and schema['node_counts']==frozen['expected_node_counts']
                and schema['input_dims']==frozen['expected_input_dims'],'Exact full graph identity differs')
        require({str(r['raw_id']):(r['source'],r['target']) for r in schema['relations']}=={k:tuple(v) for k,v in frozen['expected_relations'].items()},'Raw relation geometry differs')
        development=inputs.read_development_labels(frozen['development_labels'],frozen['archive']['sha256'])
        require(development['source_label_member_sha256']==frozen['source_label_member_sha256'],'Development source identity differs')
        split,train_labels,val_labels=inputs.verify_split(split_record['descriptor'],development,seed)
        graph,features,schema=inputs.materialize(schema,attributes,edges,implementation,frozen['relation_row_order'],torch.device('cpu'))
        driver.write(out/'GRAPH_SCHEMA.json',schema);del attributes,edges
        checkpoint_bindings=dict(admission,archive=frozen['archive'],development_labels=frozen['development_labels'],
            implementation_manifest_sha256=frozen['implementation_manifest_sha256'],graph_schema=schema)
        models=families.build(implementation,graph,schema['input_dims'],len(development['train_class_schema']),seed,seed+900001,torch.device('cpu'),frozen['arms'])
        for arm in ARMS:
            case=out/arm;arm_start=time.perf_counter()
            with (out/'ARM_STARTED.jsonl').open('a') as s:s.write(json.dumps(dict(seed=seed,arm=arm))+'\n')
            try:
                bindings=dict(checkpoint_bindings,split=split_record)
                row=driver.fit(torch,implementation,models[arm],graph,features,split,train_labels,val_labels,seed,arm,case,bindings,onecycle)
                saved=torch.load(case/'selected.pt',map_location='cpu',weights_only=True)
                require(saved['bindings']==bindings and saved['seed']==seed and saved['arm']==arm
                        and saved['optimizer_parameter_names']==[n for n,_ in models[arm].named_parameters()],
                        'Selected checkpoint binding/name custody differs')
                del saved
                row.update(attempted=True,checkpoint_bindings_verified=True,checkpoint_binding_sha256=binding['study_checkpoint_binding_sha256'],
                    **{key:dict(path=str(case/name),sha256=sha(case/name),bytes=(case/name).stat().st_size)
                       for key,name in (('selected_checkpoint','selected.pt'),('selected_logits','selected_member_logits.pt'),
                                        ('selection_receipt','SELECTION.json'),('training_trace','TRACE.jsonl'))})
            except Exception as error:
                case.mkdir(exist_ok=True);resource_failure=isinstance(error,MemoryError) or 'out of memory' in str(error).lower() or "can't allocate memory" in str(error).lower()
                row=dict(seed=seed,arm=arm,status='resource_deferred' if resource_failure else 'failed',attempted=True,
                    error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),
                    paid_wall_seconds=time.perf_counter()-arm_start,final_labels_closed=True,successful_subset_scored=False)
                if not (case/'FAILURE.json').exists():driver.write(case/'FAILURE.json',row)
            finally:del models[arm]
            rows.append(row)
            with (out/'ARM_TERMINALS.jsonl').open('a') as s:s.write(json.dumps(row,allow_nan=False)+'\n')
        result['status']='completed' if all(r['status']=='selected' for r in rows) else 'failed_or_deferred'
    except Exception as error:
        result.update(status='worker_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
        done={r['arm'] for r in rows}
        for row in missing_rows(seed,'worker_prerequisite_failed',status='failed'):
            if row['arm'] not in done:
                rows.append(row)
                with (out/'ARM_TERMINALS.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
    finally:
        try:
            for r in before:verify(r)
            scheduler_preservation(admission['scheduler_manifest_sha256'])
            result['originals_preserved']=True
        except Exception as error:result.update(status='original_preservation_failed',preservation_error=str(error));result['originals_preserved']=False
        result.update(rows=rows,wall_seconds=time.perf_counter()-started,
            process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out/'WORKER_RESULT.json',result)
    return 0 if result['status']=='completed' else 1


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--admission',required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--worker-seed',type=int);p.add_argument('--controller-out');p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(argv);require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.', '..'),'Simple fresh run name')
    binding,frozen,driver,scheduler_sha=packet_guard();release_bytes=Path(args.admission).read_bytes();release=json.loads(release_bytes)
    admission=admission_guard(binding,frozen,driver,release,scheduler_sha,args.run_name)
    admission.update(root_release_sha256=hashlib.sha256(release_bytes).hexdigest(),root_release_bytes=len(release_bytes))
    if args.worker_seed is not None:return worker(args,binding,frozen,driver,admission)
    out=PACKET/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=False)
    result=dict(status='preparing',rows=[],worker_exits={},final_labels_closed=True,device='cpu',GPU_execution=False);started=time.perf_counter()
    records=list(binding['source_records'])+[binding['freeze'],binding['resource_qualification'],frozen['archive'],frozen['development_labels']]+[r['descriptor'] for r in frozen['splits']]+[
        dict(path=str(Path(args.admission).resolve()),sha256=admission['root_release_sha256'],bytes=admission['root_release_bytes'])]
    previous_signals=install_controller_signals()
    try:
        for r in records:verify(r)
        screen=resource_screen(release['required_free_host_bytes']);result['resource_preflight']=screen
        if screen['status']=='resource_deferred':
            result.update(status='resource_deferred',rows=[r for s in SEEDS for r in missing_rows(s,'controller_resource_preflight')])
        elif args.preflight_only:result['status']='preflight_complete'
        else:
            require(not list((PACKET/'runs').glob('*/STUDY_STARTED.json')),'No automatic restart/replacement parallel study')
            require(not list((Path(binding['canonical_phase'])/'graph_heterogeneous_dblp_training_preparation_20261003_v2/runs').glob('*/STUDY_STARTED.json')),'Direct v2 study already started')
            write(out/'STUDY_STARTED.json',admission)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            commands=[(s,[sys.executable,'-B',str(Path(__file__).resolve()),'--admission',str(Path(args.admission).resolve()),'--run-name',args.run_name,
                '--worker-seed',str(s),'--controller-out',str(out.resolve())]) for s in SEEDS]
            result['worker_exits']=run_workers(commands,out,release['worker_wall_budget_seconds'],16*2**30,env,rss_limit_bytes=14*2**30)
            result['rows']=[r for s in SEEDS for r in collect(s,out/f'seed{s}',result['worker_exits'][s])]
            preserved=all((out/f'seed{s}/WORKER_RESULT.json').is_file()
                and json.loads((out/f'seed{s}/WORKER_RESULT.json').read_text()).get('originals_preserved') is True
                and json.loads((out/f'seed{s}/WORKER_RESULT.json').read_text()).get('status')=='completed'
                and result['worker_exits'][s]['exit_code']==0 and result['worker_exits'][s]['status']=='completed' for s in SEEDS)
            result['worker_preservation_and_completion_passed']=preserved
            for r in records:verify(r)
            scheduler_preservation(scheduler_sha)
            result['closure']=closure(result['rows'],driver,binding,out,comparison_permitted=preserved)
            result['status']=result['closure']['status']
    except (Exception,KeyboardInterrupt) as error:
        result.update(status='controller_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),closure=None)
        if (out/'STUDY_STARTED.json').exists():
            for seed in SEEDS:
                exit_path=out/f'worker{seed}.exit.json'
                if seed not in result['worker_exits'] and exit_path.is_file():
                    try:result['worker_exits'][seed]=json.loads(exit_path.read_text())
                    except (OSError,ValueError):pass
            result['rows']=[r for s in SEEDS for r in collect(s,out/f'seed{s}',result['worker_exits'].get(s,
                dict(exit_code=1,status='controller_aborted')))]
    finally:
        try:
            for r in records:verify(r)
            scheduler_preservation(scheduler_sha)
            result['originals_preserved']=True
        except Exception as error:result.update(status='original_preservation_failed',preservation_error=str(error),closure=None,originals_preserved=False)
        result.update(wall_seconds=time.perf_counter()-started,parallel_workers=5,threads_per_worker=1,scientific_merit_decided_by_resources=False)
        complete=result['status']=='complete_development_summary' and result.get('originals_preserved') is True
        summary=result['closure']['comparison'] if complete else dict(status='incomplete',
            all_frozen_terminals=len(result['rows'])==35,successful_subset_scored=False)
        try:
            write(out/'STUDY.json',dict(schema='HGB_DBLP_development_training_v1',rows=result['rows'],summary=summary,
                wall_seconds=result['wall_seconds'],original_inputs_verified_unchanged=result.get('originals_preserved') is True
                and result.get('worker_preservation_and_completion_passed',not (out/'STUDY_STARTED.json').exists()),
                preservation_error=result.get('preservation_error'),final_labels_closed=True,admission=admission,
                comparison_scope='HGT family only; competent native challengers remain separate required work'))
            write(out/'PARALLEL_STUDY.json',result)
        finally:restore_controller_signals(previous_signals)
    print(json.dumps(dict(status=result['status'],output=str(out))))
    return 0 if result['status'] in ('complete_development_summary','preflight_complete','resource_deferred') else 1


if __name__=='__main__':raise SystemExit(main())

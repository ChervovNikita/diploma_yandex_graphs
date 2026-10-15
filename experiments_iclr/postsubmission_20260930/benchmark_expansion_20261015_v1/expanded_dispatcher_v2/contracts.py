"""Production allocation contracts; no model or numerical imports."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=ROOT/'experiments_iclr/postsubmission_20260930'
EXPANSION=PHASE/'benchmark_expansion_20261015_v1'
HERE=EXPANSION/'expanded_dispatcher_v2'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')
GATES=('source_context','provider_split_evaluator','native_epoch_controller','selected_export_resume_rng',
       'mature_target_resource','reporting_statistics_access','source_reference_policy_or_matched_controller','campaign_scope_bridge')


def require(ok,message):
    if not ok:raise ValueError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def inside(path,root=ROOT):
    path=Path(path).resolve();require(path.is_relative_to(Path(root).resolve()),'Path escapes approved root')
    return path

def pin(path):
    raw=Path(path).read_bytes();return {'path':str(Path(path).resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def verify(value,root=ROOT):
    actual=pin(inside(value['path'],root))
    require(all(actual[k]==value[k]for k in('bytes','sha256')if k in value),'Receipt changed: '+value['path'])
    return actual

def read(path):return json.loads(Path(path).read_text())

def guard(worker):
    # Reject disabled routes before any project reads or host/device access.
    require(worker=='allocation/gpu0','Server77 disabled; explicit reauthorization and new route/runtime/shared-ledger qualification required')
    require(socket.gethostname()=='anogena-2-0','Wrong scientific allocation; no project access')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID],'Wrong allocation GPU')
    require(Path(__file__).resolve().parent==HERE,'Use the allocation source copy')


def load_legacy(pins):
    """Load exactly pinned stdlib reservation code, never old execute()."""
    for value in pins.values():verify(value)
    folder=Path(pins['supervisor.py']['path']).parent
    # The source module imports `contracts`; this package has a different module
    # name in that namespace. Load the old contract under its expected name only
    # during import, then restore this module's binding.
    previous=sys.modules.get('contracts')
    spec=importlib.util.spec_from_file_location('contracts',folder/'contracts.py');old=importlib.util.module_from_spec(spec)
    sys.modules['contracts']=old;spec.loader.exec_module(old)
    spec2=importlib.util.spec_from_file_location('_expanded_dispatcher_old_supervisor',folder/'supervisor.py')
    supervisor=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(supervisor)
    if previous is None:sys.modules.pop('contracts',None)
    else:sys.modules['contracts']=previous
    return supervisor,old


def validate_resources(resources,cost,legacy_contract):
    # Reuse predecessor's exact measured-resource and scoped-cost contract.
    def relative_pin(value):
        verify(value);return {'path':str(inside(value['path']).relative_to(ROOT)),'sha256':value['sha256']}
    r=dict(resources);r['evidence']=relative_pin(r['evidence'])
    c=dict(cost);c['evidence']=relative_pin(c['evidence'])
    legacy_contract.resource_contract(ROOT,r,c)


def source_hook_pins(job):
    required=[{'path':job['callback_module'],'sha256':job['callback_sha256']}]
    if job.get('session_transform'):required.append({k:job['session_transform'][k]for k in('path','sha256')})
    if job.get('link_policy_module'):required.append({'path':job['link_policy_module'],'sha256':job['link_policy_sha256']})
    if job.get('source_rng_emulation'):required.append({'path':job['source_rng_emulation']['audit'],'sha256':job['source_rng_emulation']['audit_sha256']})
    return required


def validate_python_environment(execution,source_files):
    """Keep the venv launch path lexical; pin its resolved binary separately."""
    value=execution['python_environment'];verify(value)
    require(source_files.get(value['path'])==value['sha256'],'Python environment missing from reviewed closure')
    spec=read(value['path'])
    require(spec.get('schema')=='qualified-lexical-venv-python-v1','Explicit lexical venv environment required')
    require(execution['python_executable']==spec['lexical_executable'] and execution['python_sha256']==spec['binary_sha256'],
            'Execution interpreter differs from exact environment binding')
    require(Path(spec['lexical_executable']).is_absolute()and str(Path(spec['lexical_executable']))==spec['lexical_executable'],
            'Absolute normalized lexical interpreter required')
    lexical=Path(spec['lexical_executable'])
    require(lexical.parent.name=='bin'and str(lexical.parent.parent)==spec['prefix']and spec['prefix']!=spec['base_prefix'],
            'Declared own virtual environment and base prefix required')
    require(str(lexical.resolve())==spec['resolved_executable']and pin(lexical)['sha256']==spec['binary_sha256'],
            'Virtual-environment symlink or actual binary changed')
    verify(spec['pyvenv_config'])
    require(Path(spec['pyvenv_config']['path'])==lexical.parent.parent/'pyvenv.cfg','Own venv configuration required')
    return spec


def validate_binding(task,binding,source_files,old_contract):
    require(set(binding)=={'execution','resources','measured_cost','gates'},'Exact reviewed binding fields required')
    execution=binding['execution'];kind=task['kind']
    require(set(execution)=={'runtime_entrypoint','python_executable','python_sha256','job_descriptor','scope','python_environment'},'Typed runtime/Python/job binding required')
    entry=verify(execution['runtime_entrypoint']);require(entry['path']in source_files,'Runtime entrypoint missing from reviewed closure')
    require(Path(entry['path']).name==('run_fit.py'if kind=='fit'else'aggregate_selected.py'),'Fixed native fit/export handler only')
    require(execution['scope']=='expanded_finite_campaign','Bounded-pilot runtime requires an explicit campaign-scope bridge')
    require(Path(execution['python_executable']).is_absolute()and len(execution['python_sha256'])==64,'Qualified interpreter bytes required')
    validate_python_environment(execution,source_files)
    validate_resources(binding['resources'],binding['measured_cost'],old_contract)
    for gate in GATES:
        require(gate in binding['gates'],'Missing per-task qualification: '+gate)
        value=binding['gates'][gate];verify(value);evidence=read(value['path'])
        require(evidence.get('status')=='PASS'and evidence.get('task_id')==task['id']and evidence.get('science_contract_hash')==task['science_contract_hash'],
                'Gate must bind this exact task/science contract: '+gate)
    if kind=='fit':
        original=read(task['source_descriptor']['path']);verify(task['source_descriptor'])
        descriptor=execution['job_descriptor'];verify(descriptor);job=read(descriptor['path'])
        require(job['schema']=='native-full-epoch-fit-v1'and job['TEST_access']is False,'Explicit native no-TEST job only')
        require(job['id']==task['id']and job['expanded_contract_hash']==task['science_contract_hash'],'Runtime job/science identity changed')
        require(digest(job['expanded_contract'])==task['science_contract_hash'],'Expanded contract bytes changed')
        # Same allocation path binding: no implicit relocation/provider mutation.
        require(job==original and digest(job)==task['runtime_job_hash'],'Allocation execution must use unchanged exact compiled descriptor')
        require(job['policy']['track']==('source_reference'if task['protocol_track']=='native_replica'else'matched_extended'),'Source and matched policies cannot be exchanged')
        require(job['device']=='cuda:0','Only actual allocation GPU0 target')
        for value in source_hook_pins(job):
            verify(value);require(source_files.get(value['path'])==value['sha256'],'Callback/transform/policy/RNG source absent from closure')
        require(not task.get('source_blocked_reason'),'Native source/context-blocked intention cannot be compiled as executable')
    else:
        require(execution['job_descriptor']is None and len(task['dependencies'])==4,'I4 uses exactly four existing exports, no job/model factory')
    return binding


def compile_release(skeleton_pin,bindings_path,output_path,output_root):
    guard('allocation/gpu0');verify(skeleton_pin);skeleton=read(skeleton_pin['path']);bindings=read(inside(bindings_path));output_path=inside(output_path)
    require(skeleton['schema']=='expanded-finite-launch-skeleton-v1','Expanded finite skeleton required')
    require(bindings['schema']=='expanded-production-bindings-v1'and bindings['skeleton']==skeleton_pin,'Bindings must pin exact skeleton')
    require(len(skeleton['tasks'])==6730 and sum(t['kind']=='fit'for t in skeleton['tasks'])==6460,'Exact expanded inventory required')
    source_files={}
    for value in bindings['source_files']:
        verify(value);source_files[value['path']]=value['sha256']
    old,old_contract=load_legacy(skeleton['legacy_supervisor_sources'])
    for name in ('contracts.py','engine.py','dispatcher.py','qualified_python_entry.py'):
        value=pin(HERE/name);require(source_files.get(value['path'])==value['sha256'],'Dispatcher source closure incomplete')
    require(set(bindings['tasks']).issubset({t['id']for t in skeleton['tasks']}),'Unknown execution binding task')
    tasks=[]
    for source in skeleton['tasks']:
        task=dict(source);binding=bindings['tasks'].get(task['id']);task['binding']=binding
        if binding:
            require(task['worker']=='allocation/gpu0','Disabled77 cannot receive executable binding')
            validate_binding(task,binding,source_files,old_contract)
        task['execution_status']='qualified'if binding else'blocked_missing_qualification'
        tasks.append(task)
    require(len({t['id']for t in tasks})==6730,'Disjoint task IDs required')
    outputs=[t['output_relative']for t in tasks];require(len(set(outputs))==6730,'Disjoint immutable task outputs required')
    indexed={t['id']:t for t in tasks}
    for task in tasks:
        for dep in task['dependencies']:require(indexed[dep]['worker']==task['worker'],'I4 dependencies must share physical worker')
    release={'schema':'expanded-production-dispatch-release-v1','skeleton':skeleton_pin,'source_files':bindings['source_files'],
        'source_bundle_sha256':digest(source_files),'tasks':tasks,'workers':list(WORKERS),'enabled_workers':['allocation/gpu0'],
        'maximum_owned_children':2,'output_root':str(inside(output_root)),'legacy_supervisor_sources':skeleton['legacy_supervisor_sources'],
        'all_tasks_qualified':all(t['binding']for t in tasks),'full_inventory_count':6730,'physical_fits':6460,'I4_exports':270,
        'human_GO':False,'automatic_retries':0,'TEST_access':False,'campaign_ETA':None}
    require(not Path(output_path).exists(),'Preserve prior release')
    Path(output_path).write_text(json.dumps(release,indent=2,sort_keys=True)+'\n')
    return {'release':pin(output_path),'qualified_tasks':sum(t['binding']is not None for t in tasks),'blocked_tasks':sum(t['binding']is None for t in tasks)}


def authority(release_path,go_path,readiness_path,worker):
    guard(worker)
    release_pin=pin(inside(release_path));release=read(release_path)
    require(release['schema']=='expanded-production-dispatch-release-v1','Reviewed production release required; skeleton is not executable')
    require(release['enabled_workers']==['allocation/gpu0']and worker=='allocation/gpu0','Only allocation execution authorized')
    require(release['maximum_owned_children']==2 and len(release['tasks'])==6730,'Finite/two-slot contract changed')
    inside(release['output_root']);verify(release['skeleton'])
    skeleton=read(release['skeleton']['path']);original={t['id']:t for t in skeleton['tasks']}
    require(len({t['id'] for t in release['tasks']})==6730 and set(original)=={t['id'] for t in release['tasks']},'Release must preserve exact disjoint skeleton IDs')
    for task in release['tasks']:
        require({k:v for k,v in task.items() if k not in ('binding','execution_status')}==original[task['id']],'Release changed science/placement/dependency contract')
    require(release['all_tasks_qualified']is True,'Full expanded campaign still has source/provider/runtime/resource qualification gaps')
    require(go_path and readiness_path,'Exact human GO and readiness required')
    go_pin=pin(inside(go_path));ready_pin=pin(inside(readiness_path));go=read(go_path);ready=read(readiness_path)
    record,budget=validate_reviewed_authority(release,release_pin,go,go_pin,ready,ready_pin,worker)
    old,old_contract=load_legacy(release['legacy_supervisor_sources'])
    files={v['path']:v['sha256']for v in release['source_files']}
    require(digest(files)==release['source_bundle_sha256'],'Source closure hash changed')
    for value in release['source_files']:verify(value)
    for task in release['tasks']:
        require(task['execution_status']=='qualified'and task['binding'],'Executable release contains blocked task')
        validate_binding(task,task['binding'],files,old_contract)
    own=[t for t in release['tasks']if t['worker']==worker]
    require(bool(own),'Worker must own explicit tasks')
    from qualified_python_entry import validate_runtime_environment
    for task in own:
        execution=task['binding']['execution']
        spec=validate_python_environment(execution,files)
        validate_runtime_environment(spec)  # Actual supervisor lexical venv, prefix and resolved binary.

    def recheck(task):
        guard(worker)
        for value in (release_pin,go_pin,ready_pin):verify(value)
        verify(go['human_instruction_evidence'])
        for value in release['source_files']:verify(value)
        for evidence in record['gates'].values():verify(evidence['evidence'])
        validate_binding(task,task['binding'],files,old_contract)
        validate_runtime_environment(validate_python_environment(task['binding']['execution'],files))
    return release,release_pin,old,budget,recheck

def validate_reviewed_authority(release,release_pin,go,go_pin,ready,ready_pin,worker):
    require(go.get('schema')=='expanded-production-human-GO-v1'and go.get('user_explicit_GO')is True,'Explicit human campaign GO required')
    require(go['manifest_sha256']==release_pin['sha256']and go['source_bundle_sha256']==release['source_bundle_sha256'],'GO binds different manifest/source')
    require(go['readiness']==ready_pin and worker in go['allowed_workers'],'GO readiness/worker mismatch')
    verify(go['human_instruction_evidence']);require(bool(go.get('review_evidence')),'GO must cite actual human instruction;root/human custody authenticates it')
    require(ready['schema']=='expanded-production-readiness-v1'and ready['manifest_sha256']==release_pin['sha256']and ready['source_bundle_sha256']==release['source_bundle_sha256'],'Readiness differs from exact release')
    record=ready['workers'][worker];budget=record['budget']
    require(set(budget)=={'gpu_headroom_bytes','host_ram_headroom_bytes','disk_headroom_bytes'}and all(type(v)is int and v>=0 for v in budget.values()),'Measured headroom budget required')
    for gate in ('route_identity_environment','complete_per_task_qualification','shared_claim_and_STOP_custody','failure_cost_and_resume_custody'):
        evidence=record['gates'][gate];require(evidence['status']=='PASS','Worker gate is not PASS: '+gate);verify(evidence['evidence'])
    return record,budget


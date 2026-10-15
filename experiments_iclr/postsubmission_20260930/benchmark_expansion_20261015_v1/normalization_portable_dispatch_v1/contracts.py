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

import bootstrap
from binding_rules import expected_entrypoint, normalized_execution_binding, verify_composed_job
from host_binding import load_config, guard as host_guard, config_path, HERE, WORKERS
SUCCESSOR_HERE=Path(__file__).resolve().parent
CONFIG, CONFIG_PIN = load_config()
ROOT=Path(CONFIG['project_root'])
PHASE=ROOT/CONFIG['phase_relative']
EXPANSION=PHASE/'benchmark_expansion_20261015_v1'
UUID=CONFIG['physical_gpu_uuid']
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
    require(worker==CONFIG['worker'],'Worker/config mismatch')
    host_guard(CONFIG)


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
    require(type(resources.get('cpu_slots'))is int and resources['cpu_slots']>0,'Reviewed positive CPU reservation required')
    require(type(resources.get('thread_count'))is int and 0<resources['thread_count']<=resources['cpu_slots'],'Explicit native thread envelope required')
    r={k:v for k,v in resources.items() if k not in('cpu_slots','thread_count','cpu_evidence')};r['evidence']=relative_pin(r['evidence'])
    verify(resources['cpu_evidence']);cpu=read(resources['cpu_evidence']['path'])
    require(cpu.get('status')=='PASS'and cpu.get('worker')==CONFIG['worker'] and cpu.get('cpu_slots')==resources['cpu_slots'] and cpu.get('thread_count')==resources['thread_count'],'CPU/thread receipt must bind exact worker envelope')
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
    family=task.get('family') if kind=='fit' else 'I4'
    expected_entry=expected_entrypoint(task, ROOT)
    require(entry['path']==str(expected_entry),'Only bundled reviewed native entrypoints are executable')
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
        if original.get('normalization_composition') is not None:
            verify_composed_job(original, ROOT, source_files)
            bound,receipt=original,normalized_execution_binding(original, ROOT, CONFIG['phase_relative'])
        else:
            from descriptor_binding import rebind_job
            from host_binding import ALLOCATION_ROOT, PHASE_RELATIVE
            bound,receipt=rebind_job(original,ALLOCATION_ROOT,PHASE_RELATIVE,str(ROOT),CONFIG['phase_relative'])
        require(digest(original)==task['canonical_runtime_job_hash'],'Canonical descriptor identity changed')
        require(job==bound and digest(job)==task['runtime_job_hash'],'Only explicit runtime path binding is allowed')
        require(task['runtime_binding']==receipt,'Exact relocation receipt required')
        require(job['policy']['track']==('source_reference'if task['protocol_track']=='native_replica'else'matched_extended'),'Source and matched policies cannot be exchanged')
        require(job['device']=='cuda:0','UUID-masked logical device required')
        for value in source_hook_pins(job):
            verify(value);require(source_files.get(value['path'])==value['sha256'],'Callback/transform/policy/RNG source absent from closure')
        require(not task.get('source_blocked_reason'),'Native source/context-blocked intention cannot be compiled as executable')
    else:
        require(execution['job_descriptor']is None and len(task['dependencies'])==4,'I4 uses exactly four existing exports, no job/model factory')
    return binding


def authority(run_spec, worker):
    """Every launch is bound to one reviewed local projection and global owner."""
    guard(worker)
    from ownership import validate as validate_ownership, CAMPAIGN
    from custody import digest as owner_digest
    for name in ('release','ownership','go','readiness'):
        require(run_spec.get(name),'Exact '+name+' pin required');verify(run_spec[name])
    release_pin=run_spec['release'];release=read(release_pin['path'])
    ownership_pin=run_spec['ownership'];ownership=read(ownership_pin['path'])
    require(release['schema']=='portable-host-production-release-v2','Reviewed host projection required')
    require(release['ownership']==ownership_pin and release['host']==worker.split('/')[0],'Projection owner/host mismatch')
    require(release['ordinary_failure_policy']=='retain_failure_continue_other_admissible_tasks','Explicit reviewed failure policy required')
    require(release['host_config_pins'][worker]==CONFIG_PIN,'Exact worker configuration required')
    verify(release['canonical_skeleton']);skeleton=read(release['canonical_skeleton']['path'])
    from prepare_projection import verify_canonical_plan
    verify_canonical_plan(release,skeleton)
    owners=validate_ownership(ownership,skeleton['tasks'])
    require(ownership['hosts'][release['host']]['project_root']==str(ROOT),'Immutable host root changed')
    require(ownership['hosts'][release['host']]['phase_relative']==CONFIG['phase_relative'],'Immutable host phase changed')
    expected=ROOT/CONFIG['phase_relative']/'benchmark_expansion_20261015_v1/campaign_outputs'/CAMPAIGN
    require(Path(release['output_root'])==expected and expected.resolve()==expected,'One canonical output/claim namespace required')
    own_ids={i for i,w in owners.items() if w.split('/')[0]==release['host']}
    require(len(release['tasks'])==len(own_ids) and {t['id'] for t in release['tasks']}==own_ids,'Projection must contain every host-owned task exactly once')
    original={t['id']:t for t in skeleton['tasks']}
    from prepare_projection import project_task
    for task in release['tasks']:
        expected_task=project_task(original[task['id']],owners[task['id']],ownership['hosts'][release['host']],read(task['source_descriptor']['path']) if task['kind']=='fit' else None)
        require({k:v for k,v in task.items() if k not in('binding','execution_status')}==expected_task,'Projection changed task/science/dependency contract')
    require(release['all_tasks_qualified'] is True,'Host projection has unresolved qualification gates')
    require(all(t['binding'] and t['execution_status']=='qualified' for t in release['tasks']),'Host projection contains blocked tasks')
    go=read(run_spec['go']['path']);ready=read(run_spec['readiness']['path'])
    require(go.get('schema')=='portable-human-GO-v2' and go.get('user_explicit_GO')is True,'Actual explicit campaign GO required')
    require(go['ownership_sha256']==owner_digest(ownership) and go['host_releases'][release['host']]==release_pin and worker in go['allowed_workers'],'GO binds different ownership/projection/worker')
    require(go['readiness']==run_spec['readiness'],'GO/readiness mismatch')
    verify(go['human_instruction_evidence'])
    require(ready['schema']=='portable-readiness-v2'and ready['ownership_sha256']==owner_digest(ownership)and ready['host_releases'][release['host']]==release_pin,'Readiness must bind exact host projection')
    record=ready['workers'][worker];budget=record['budget']
    accounting=record['memory_accounting']
    require(accounting['cgroup_v2_memory_paths']==CONFIG.get('cgroup_v2_memory_paths') and accounting['all_effective_ancestor_limits_accounted']is True,'Actual cgroup path/ancestor qualification required')
    verify(accounting['evidence'])
    require(set(budget)=={'gpu_headroom_bytes','host_ram_headroom_bytes','disk_headroom_bytes','host_cpu_slots'} and all(type(v)is int and v>=0 for v in budget.values()) and budget['host_cpu_slots']>0,'Qualified host resource budget required')
    for peer,entry in ready['workers'].items():
        if peer.split('/')[0]==release['host']:
            require(all(entry['budget'][k]==budget[k]for k in('host_cpu_slots','host_ram_headroom_bytes','disk_headroom_bytes')),'Host workers must use the same aggregate resource envelope')
    for gate in ('route_identity_environment','complete_per_task_qualification','shared_claim_and_STOP_custody','failure_cost_and_resume_custody'):
        evidence=record['gates'][gate];require(evidence['status']=='PASS','Missing worker gate: '+gate);verify(evidence['evidence'])
    _,old_contract=load_legacy(release['legacy_supervisor_sources'])
    files={v['path']:v['sha256']for v in release['source_files']}
    require(digest(files)==release['source_bundle_sha256'],'Source closure changed')
    for value in release['source_files']:verify(value)
    package=read(HERE/'MANIFEST.json')
    for row in package['files']:
        require(files.get(str(HERE/row['path']))==row['sha256'],'Portable package closure incomplete')
    require(files.get(str(HERE/'MANIFEST.json'))==pin(HERE/'MANIFEST.json')['sha256'],'Package manifest missing from reviewed closure')
    require(files.get(str(HERE/'run_gpu.py'))==pin(HERE/'run_gpu.py')['sha256'],'Launcher missing from reviewed closure')
    successor=read(SUCCESSOR_HERE/'MANIFEST.json')
    for row in successor['files']:
        require(files.get(str(SUCCESSOR_HERE/row['path']))==row['sha256'],'Normalization dispatcher closure incomplete')
    require(files.get(str(SUCCESSOR_HERE/'MANIFEST.json'))==pin(SUCCESSOR_HERE/'MANIFEST.json')['sha256'],
            'Normalization dispatcher manifest missing from reviewed closure')
    from qualified_python_entry import validate_runtime_environment
    own=[t for t in release['tasks'] if t['worker']==worker]
    require(own,'Worker has no explicit owned tasks')
    for task in own:
        require(task['binding']and task['execution_status']=='qualified','Task is not qualified')
        validate_binding(task,task['binding'],files,old_contract)
        require(task['binding']['resources']['thread_count']==2,'Preserve existing native two-thread setting; changing it requires a source successor')
        resource_gate=read(task['binding']['gates']['mature_target_resource']['path'])
        require(resource_gate.get('worker')==worker and resource_gate.get('GPU_UUID')==UUID and resource_gate.get('runtime_job_hash')==task['runtime_job_hash'],'Actual destination resource evidence required')
        validate_runtime_environment(validate_python_environment(task['binding']['execution'],files))
    def recheck(task):
        guard(worker)
        for name in ('release','ownership','go','readiness'):verify(run_spec[name])
        verify(go['human_instruction_evidence'])
        for value in release['source_files']:verify(value)
        for evidence in record['gates'].values():verify(evidence['evidence'])
        verify(accounting['evidence'])
        validate_binding(task,task['binding'],files,old_contract)
        validate_runtime_environment(validate_python_environment(task['binding']['execution'],files))
    return release,release_pin,ownership,budget,recheck

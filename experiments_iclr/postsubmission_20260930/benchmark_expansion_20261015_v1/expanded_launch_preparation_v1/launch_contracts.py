"""Expanded queue contracts. Source/JSON only; no numerical imports or launches."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=ROOT/'experiments_iclr/postsubmission_20260930'
EXPANSION=PHASE/'benchmark_expansion_20261015_v1'
HERE=EXPANSION/'expanded_launch_preparation_v1'
WORKERS=('allocation/gpu0','server77/gpu0','server77/gpu1')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
FIT_GATES=('source_context','provider_split_evaluator','native_epoch_controller','selected_export_resume_rng',
           'mature_target_resource','reporting_statistics_access')


def require(ok,reason):
    if not ok:raise ValueError(reason)


def guard():
    require(socket.gethostname()=='anogena-2-0','Wrong allocation; no project access')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID],
            'Wrong allocation GPU; no project access')
    require(Path(__file__).resolve().parent==HERE,'Use the guarded allocation source copy')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def inside(path):
    p=Path(path).resolve();require(p.is_relative_to(PHASE),'Path escapes phase')
    return p


def receipt(path):
    p=inside(path);require(p.suffix in {'.json','.jsonl','.py','.md','.txt'},'Only source/JSON metadata may be opened')
    raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def verify(pin):
    actual=receipt(pin['path'])
    require(all(actual[k]==pin[k] for k in ('bytes','sha256') if k in pin),'Receipt changed: '+pin['path'])
    return actual


def read(path,pin=None):
    if pin:verify(pin)
    else:receipt(path)
    return json.loads(inside(path).read_text())


def write(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with inside(path).open('xb') as out:out.write(raw)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def legacy(source_pins):
    """Reuse reviewed reservation, probe and terminal mechanisms; no run call."""
    folder=EXPANSION/'campaign_launcher_v1'
    for name in ('contracts.py','supervisor.py'):
        verify(source_pins[name])
    sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('_expanded_reviewed_legacy_supervisor',folder/'supervisor.py')
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
    return value


def check_host_policy(policy):
    require(policy['schema']=='expanded-host-policy-v1','Explicit host policy required')
    require(set(policy['workers'])==set(WORKERS),'Keep all three intended future workers explicit')
    enabled=[]
    for name,row in policy['workers'].items():
        require(row['maximum_children_per_physical_GPU']==2,'At most two owned children per physical GPU')
        if not row['assignment_enabled']:
            require(row['execution_enabled'] is False,'Disabled assignment cannot execute')
            continue
        if name!='allocation/gpu0':
            for role in ('explicit_host_reauthorization','route_and_device_identity','qualified_runtime_path_bridge','shared_claim_STOP_and_host_resource_ledger'):
                require(row.get(role), 'Future host assignment requires '+role)
                evidence=read(row[role]['path'],row[role]);require(evidence.get('status')=='pass','Future host gate is not PASS')
            require(row.get('physical_GPU_UUID') and row.get('qualified_profile'),'Verified future GPU identity/profile required')
        else:
            require(row['hostname']=='anogena-2-0' and row['physical_GPU_UUID']==UUID,'Literal allocation identity required')
        enabled.append(name)
    require(enabled,'At least one authorized planning worker required')
    return sorted(enabled)


def qualification_blocks(task,admission,host):
    reasons=[]
    if task.get('source_status')=='blocked':reasons.append(task['source_blocked_reason'])
    for gate in FIT_GATES if task['kind']=='fit' else ('selected_export_identity_and_role','mature_target_resource','reporting_statistics_access'):
        pin=admission.get('gates',{}).get(gate)
        if not pin:reasons.append('missing_'+gate);continue
        value=read(pin['path'],pin)
        if value.get('status')!='pass' or value.get('fit_or_task_id')!=task['id'] or value.get('science_contract_hash')!=task['science_contract_hash']:
            reasons.append('unqualified_'+gate)
    if not admission.get('execution_binding'):reasons.append('qualified_runtime_execution_binding_missing')
    if not admission.get('resources'):reasons.append('measured_resource_reservations_missing')
    if host['execution_enabled'] is not True:reasons.append('host_execution_not_enabled')
    return sorted(set(reasons))


def durable_action(task,state):
    """Restartable queue decisions; never steal a claim or infer a retry."""
    row=state.get(task['id'])
    if not row:return {'action':'pending_new','extra_fit_repeats':0}
    require(row['science_contract_hash']==task['science_contract_hash'],'Queue state belongs to changed science contract')
    status=row['state']
    if status in ('complete','no_selected_state'):
        pin=row.get('terminal')
        require(pin,'Terminal receipt required to skip completed work')
        terminal=read(pin['path'],pin)
        require(terminal.get('status')==status and terminal.get('TEST_accessed') is False,'Invalid terminal role/status')
        if task['kind']=='fit':
            require(terminal.get('schema')=='native-full-epoch-fit-result-v1' and terminal.get('job_id')==task['id'] and terminal.get('job_hash')==task['runtime_job_hash'],'Terminal belongs to another physical fit')
        else:
            require(terminal.get('schema')=='independent-own-selected-VALID-bank-result-v1','Own I4 export terminal schema required')
            require(sorted(m['job_id'] for m in terminal.get('members',[]))==sorted(task['dependencies']),'I4 terminal members differ from four own fit dependencies')
        return {'action':'skip_'+status,'terminal':pin,'extra_fit_repeats':0}
    if status=='interrupted':
        pin=row.get('resume_disposition')
        if not pin:return {'action':'blocked_interrupted_requires_reviewed_disposition','extra_fit_repeats':0}
        value=read(pin['path'],pin)
        require(value.get('explicit_resume_authorized') is True and value.get('job_id')==task['id'] and value.get('job_hash')==task['runtime_job_hash'],
                'Resume disposition must bind the unchanged physical job')
        require(value.get('complete_epoch_checkpoint_receipt') and value.get('prior_partial_work_is_preserved') is True,'Resume requires own recorded epoch checkpoint and partial-work custody')
        checkpoint=value['complete_epoch_checkpoint_receipt']
        require(inside(checkpoint['path']).name=='recovery.pt' and len(checkpoint['sha256'])==64,'Own recovery checkpoint receipt required')
        # The skeleton reads only its receipt. Runtime verifies/loads tensors after review/GO.
        return {'action':'resume_same_fit_after_gates','resume_checkpoint':checkpoint,'disposition':pin,'extra_fit_repeats':0}
    require(status in ('running','failed','claimed'),'Unknown durable claim state')
    return {'action':'blocked_'+status+'_requires_reconciliation','extra_fit_repeats':0}


def fixed_argv(task,execution_binding,output_root,resume=None):
    """Typed command proposal only. This package never calls Popen."""
    runtime=execution_binding['runtime_entrypoint'];verify(runtime)
    python=execution_binding['qualified_python_executable']
    output=inside(output_root)/task['output_relative']
    if task['kind']=='fit':
        require(Path(runtime['path']).name=='run_fit.py','Fixed native fit handler required')
        job=execution_binding['job_descriptor'];verify(job)
        argv=[python,'-B','-u',runtime['path'],'--project-root',str(ROOT),'--job',job['path'],'--output',str(output)]
        if resume:argv+=['--resume',resume['path']]
        return argv
    require(Path(runtime['path']).name=='aggregate_selected.py','Fixed export-only I4 handler required')
    members=execution_binding['own_selected_VALID_exports']
    require(len(members)==4,'I4 requires exactly four own exports')
    return [python,'-B','-u',runtime['path'],'--project-root',str(ROOT),'--members',*[m['path'] for m in members],'--output',str(output)]

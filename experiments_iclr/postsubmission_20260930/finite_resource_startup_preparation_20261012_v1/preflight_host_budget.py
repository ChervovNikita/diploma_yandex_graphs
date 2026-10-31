"""Finite source/startup admission check; no workers, fits or host changes.

One normal launcher can call this internally before its two worker slots.
Missing bounds remain pending; insufficient resources refuse startup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ALLOCATION_PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
SERVER77_PHASE=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def pin(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def confined(root, relative):
    value=Path(relative);path=(root/value).resolve()
    if value.is_absolute() or '..' in value.parts or not path.is_relative_to(root):
        raise ValueError('Project-relative preflight path required')
    return path


def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    gpus=[{'index':int(a),'uuid':b.strip(),'total_mib':int(c),'free_mib':int(d)}
          for a,b,c,d in (line.split(',') for line in raw.splitlines())]
    return {'hostname':socket.gethostname(),'gpus':gpus}


def assess(request, observed):
    pending=[];refused=[]
    required=('final_master_canonical_sha256','final_master_plan_receipt','root_resource_coverage_review_reference','runtime_admission_receipt',
              'disk_required_additional_bytes','disk_headroom_bytes','whole_host_maximum_live_RSS_bytes',
              'RAM_headroom_bytes','per_GPU_live_slot_budgets','maximum_finite_retry_storage_bytes')
    for name in required:
        if request.get(name) is None:pending.append(name)
    for name in ('disk_required_additional_bytes','disk_headroom_bytes','whole_host_maximum_live_RSS_bytes',
                 'RAM_headroom_bytes','maximum_finite_retry_storage_bytes'):
        number=request.get(name)
        if number is not None and (type(number) is not int or number<0):refused.append('invalid_nonnegative_byte_bound_'+name)
    if refused:
        return {'schema':'massive-run-host-startup-assessment-v1','status':'REFUSED','pending':pending,'refused':refused,
                'normal_host_execution':True,'host_settings_changed':False,'scientific_workers_started':0,
                'scientific_GO_created':False,'scientific_fits':0}
    if request.get('source_geometry_and_full_state_coverage_complete') is not True:
        pending.append('source_geometry_and_full_state_coverage_complete')
    if request.get('all_adapter_state_populated_native_L_paid_ancestors_HPO_cohorts_banks_tapes_included') is not True:
        pending.append('complete_retention_storage_accounting')
    if request.get('host_binding') is None:pending.append('physical_host_binding')
    elif request['host_binding']!={'hostname':observed['hostname'],'gpu_uuids':[r['uuid'] for r in observed['gpus']]}:
        refused.append('physical_host_binding_changed')
    if request.get('disk_required_additional_bytes') is not None and request.get('disk_headroom_bytes') is not None:
        need=request['disk_required_additional_bytes']+request['disk_headroom_bytes']
        if need>observed['disk_available_bytes']:refused.append('insufficient_project_filesystem_disk')
    if request.get('whole_host_maximum_live_RSS_bytes') is not None and request.get('RAM_headroom_bytes') is not None:
        need=request['whole_host_maximum_live_RSS_bytes']+request['RAM_headroom_bytes']
        if need>observed['RAM_available_bytes']:refused.append('insufficient_whole_host_RAM')
    budgets=request.get('per_GPU_live_slot_budgets')
    if budgets is not None:
        if len(budgets)!=len(observed['gpus']):refused.append('every_present_GPU_budget_required')
        for row in budgets:
            actual=next((r for r in observed['gpus'] if r['uuid']==row.get('uuid')),None)
            if actual is None:refused.append('unknown_GPU_UUID');continue
            # Two is an upper slot limit. The reservation broker may run one
            # large fit and delay a second until their combined budget fits.
            if row.get('maximum_worker_slots')!=2:refused.append('requested_two_slot_limit_changed')
            if row.get('largest_qualified_single_fit_mib') is None or row.get('GPU_headroom_mib') is None:
                pending.append('qualified_single_fit_GPU_envelope_'+actual['uuid']);continue
            if any(type(row[name]) is not int or row[name]<0 for name in ('largest_qualified_single_fit_mib','GPU_headroom_mib')):
                refused.append('invalid_GPU_byte_bound_'+actual['uuid']);continue
            if row['largest_qualified_single_fit_mib']+row['GPU_headroom_mib']>actual['free_mib']:
                refused.append('largest_single_fit_does_not_fit_present_free_VRAM_'+actual['uuid'])
            if row.get('two_slot_pair_admission_policy')!='sum_live_reservations_within_present_VRAM_and_whole_host_RSS_disk':
                pending.append('two_slot_pair_admission_policy_'+actual['uuid'])
    return {'schema':'massive-run-host-startup-assessment-v1','status':'REFUSED' if refused else ('PENDING' if pending else 'PASS'),
            'pending':pending,'refused':refused,'normal_host_execution':True,'host_settings_changed':False,
            'scientific_workers_started':0,'scientific_GO_created':False,'scientific_fits':0,
            'two_worker_slots_per_GPU_are_upper_bound_not_unconditional_concurrency':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--host',choices=('allocation','server77'),required=True)
    p.add_argument('--result',required=True);a=p.parse_args();observed=snapshot()
    # No seven-GPU metadata/source access occurs even if the wrong host flag is
    # passed. Physical identities are checked before project files are opened.
    if a.host=='allocation':
        if observed['hostname']!='anogena-2-0' or [r['uuid'] for r in observed['gpus']]!=[UUID]:
            raise ValueError('Authorized one-GPU allocation route required')
        expected=ALLOCATION_PHASE
    else:
        if len(observed['gpus'])!=2 or observed['hostname'].startswith('anogena'):
            raise ValueError('Authorized two-GPU server77 physical route required')
        expected=SERVER77_PHASE
    root=Path(a.project_root).resolve()
    if root!=expected:raise ValueError('Exact authorized project research root required')
    manifest=confined(root,a.manifest);value=json.loads(manifest.read_text())
    if value.get('schema')!='massive-run-three-GPU-host-budget-manifest-v1':raise ValueError('Reviewed finite host-budget schema required')
    request=value['hosts'][a.host];fs=os.statvfs(root)
    master_row=request.get('final_master_plan_receipt')
    if master_row:
        master_path=confined(root,master_row['path'])
        if pin(master_path)!={k:master_row[k] for k in ('bytes','sha256')}:raise ValueError('Final master file receipt changed')
        master=json.loads(master_path.read_text())
        canonical=json.dumps(master,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest()!=request['final_master_canonical_sha256']:
            raise ValueError('Physical preflight refers to another master run plan')
    observed['disk_available_bytes']=fs.f_bavail*fs.f_frsize
    memory={line.split(':',1)[0]:int(line.split(':',1)[1].strip().split()[0])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    observed['RAM_available_bytes']=memory['MemAvailable']
    receipt=request.get('runtime_admission_receipt')
    if receipt:
        authority=confined(root,receipt['path'])
        if pin(authority)!={k:receipt[k] for k in ('bytes','sha256')}:raise ValueError('Host runtime admission receipt changed')
        admitted=json.loads(authority.read_text())
        if admitted.get('admitted') is not True or not admitted.get('root_authorization_reference'):
            raise ValueError('Source runtime/device admission is not granted')
    assessment=assess(request,observed);assessment.update(manifest_pin=pin(manifest),observed=observed)
    result=confined(root,a.result)
    if result.exists():raise ValueError('Preserve every preflight outcome with a fresh result path')
    result.parent.mkdir(parents=True,exist_ok=True);result.write_text(json.dumps(assessment,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':assessment['status'],'pending':assessment['pending'],'refused':assessment['refused'],
                      'scientific_workers_started':0,'scientific_GO_created':False}))
    return 0 if assessment['status']=='PASS' else 3

if __name__=='__main__':sys.exit(main())

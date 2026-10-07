"""GPU77 host/provider and root-bound comparison custody; no numerical work."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def admit_host(root,phase,repo,job,bound,sha):
    config=json.loads((root/'HOST_CONTEXT.json').read_text())
    if job['kind'] not in ('native_fit','fit'):
        raise ValueError('This host port admits ordinary scientific fits only; no parity diagnostics')
    if str(repo)!=config['repository'] or Path.cwd().resolve()!=repo.resolve() or socket.gethostname()!=config['hostname']:
        raise ValueError('Exact authorized GPU77 repository and hostname required')
    selected=job.get('physical_gpu_uuid')
    if selected not in config['physical_gpu_inventory'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=selected:
        raise ValueError('Root-selected authorized physical GPU and visible singleton required')
    observed=[]
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total',
        '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    for row in rows:
        uuid,name,driver,memory=[value.strip() for value in row.split(',')]
        observed.append({'uuid':uuid,'name':name,'driver_version':driver,'memory_total_MiB':int(memory)})
    if [row['uuid'] for row in observed]!=config['physical_gpu_inventory'] or any(
        row['name']!=config['physical_gpu_name'] or row['memory_total_MiB']!=config['physical_gpu_memory_total_MiB'] for row in observed):
        raise ValueError('Exact authorized full two-GPU physical inventory required')
    python=config['python_executable']
    if (str(Path(sys.executable).absolute())!=python['path'] or str(Path(sys.executable).resolve())!=python['resolved_path']
        or sha(Path(sys.executable).resolve())!=python['sha256'] or sys.prefix!=config['runtime_prefix']):
        raise ValueError('Exact existing GPU77 interpreter/provider required')
    if os.environ.get('PYTHONPATH','')!='' or os.environ.get('PYTHONHOME'):
        raise ValueError('Use the existing self-contained GPU77 provider without an allocation overlay')
    bound(config['runtime_provider']);bound(config['runtime_environment_verification'])
    context=json.loads(bound(job['execution_context']).read_text())
    descriptor={'host_context_id':config['host_context_id'],'hostname':config['hostname'],
        'repository':config['repository'],'physical_gpu_uuid':selected,'python_executable':python['path'],
        'runtime_provider':config['runtime_provider'],'runtime_versions':config['runtime_versions']}
    if (context.get('root_adopted') is not True or context.get('device_runtime')!=descriptor
        or context.get('physical_inventory')!=observed or context.get('seed')!=job['seed']
        or context.get('data_manifest')!=job['data_manifest'] or context.get('plan')!=job['plan']
        or context.get('soft_seconds')!=job['soft_seconds'] or context.get('hard_seconds')!=job['hard_seconds']):
        raise ValueError('Exact root-bound actual host/runtime/data/plan/resource context required')
    authority=json.loads(bound(job['data_manifest']).read_text())
    if (authority['available']['sha256']!=config['safe_data_payload_sha256']
        or authority.get('TEST_labels_available_to_trainer') is not False
        or context.get('data_payload')!=authority['available']):
        raise ValueError('Local GPU77 safe data manifest must bind the same fixed TRAIN/VALID payload; TEST closed')
    copy_receipt=json.loads(bound(context['local_data_custody']).read_text())
    if (copy_receipt.get('complete') is not True or copy_receipt.get('local_repository')!=config['repository']
        or copy_receipt.get('local_hostname')!=config['hostname'] or copy_receipt.get('data_manifest')!=job['data_manifest']
        or copy_receipt.get('data_payload')!=authority['available'] or copy_receipt.get('TEST_labels_copied') is not False):
        raise ValueError('Exact local GPU77 data copy/acquisition custody required')
    block=json.loads(bound(context['comparison_block']).read_text())
    if (block.get('root_adopted') is not True or block.get('seed')!=job['seed']
        or block.get('fixed_arms')!=config['fixed_arms'] or block.get('data_manifest')!=job['data_manifest']):
        raise ValueError('Fixed seed comparison block and common data required')
    policy=block.get('comparison_policy')
    if policy not in ('matched_device_runtime','cross_runtime_limit_recorded'):
        raise ValueError('Root must prospectively record the device/runtime comparison scope')
    if policy=='matched_device_runtime' and block.get('device_runtime')!=descriptor:
        raise ValueError('Native baseline and candidates in a seed must share the declared physical GPU/runtime')
    if policy=='cross_runtime_limit_recorded' and not isinstance(block.get('cross_runtime_limit'),str):
        raise ValueError('Cross-runtime comparison needs an explicit root-recorded interpretation limit')
    if policy=='cross_runtime_limit_recorded' and not block['cross_runtime_limit'].strip():
        raise ValueError('Empty cross-runtime interpretation limit')
    freezes=job.get('donor_freezes',[]);origins=job.get('donor_origins',[])
    if len(origins)!=len(freezes):raise ValueError('Every donor requires exact training-origin custody')
    for freeze_binding,origin_binding in zip(freezes,origins):
        freeze=json.loads(bound(freeze_binding).read_text());origin=json.loads(bound(origin_binding).read_text())
        training_job_path=bound(origin['source_training_job']);training_job=json.loads(training_job_path.read_text())
        if (origin.get('complete_custody_record') is not True or origin.get('freeze')!=freeze_binding
            or training_job.get('source_manifest_sha256')!=freeze['source_manifest_sha256']
            or training_job.get('program_sha256')!=freeze['program_sha256']
            or (freeze.get('job_sha256') is not None and freeze['job_sha256']!=sha(training_job_path))):
            raise ValueError('Donor source training job and frozen output custody differs')
        bound(origin['training_terminal_evidence'])
        training_context=origin['training_device_runtime']
        original=config['original_allocation_donor_origin']
        if freeze['source_manifest_sha256']==original['source_manifest_sha256']:
            if (any(training_context.get(key)!=original[key] for key in ('hostname','repository','physical_gpu_uuid','python_executable'))
                or freeze['program_sha256']!=original['program_sha256'] or origin.get('trained_locally_on_gpu77') is not False):
                raise ValueError('Transferred allocation donor must retain its actual allocation training origin')
            bound(origin['transfer_custody'])
        else:
            acquired=json.loads(bound(training_job['execution_context']).read_text())['device_runtime']
            if training_context!=acquired or origin.get('trained_locally_on_gpu77') is not True:
                raise ValueError('Local donor must bind its actual GPU77 training context')
        if policy=='matched_device_runtime' and training_context!=descriptor:
            raise ValueError('Donor training differs from seed comparison GPU/runtime; record the cross-runtime limit')
        bound(freeze['selected']);bound(freeze['end'])
    return config


def verify_runtime_provider(config,modules,sha):
    if config is None:raise ValueError('Host context was not admitted')
    for name,module in modules.items():
        expected=config['core_module_provider'][name];path=Path(module.__file__).resolve()
        if str(path)!=expected['resolved_path'] or path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:
            raise ValueError('Actual imported GPU77 runtime provider differs: '+name)

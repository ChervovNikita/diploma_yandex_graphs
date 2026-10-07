"""Both authorized hosts, TRAIN-only fits and all-endpoint scoring admission."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parent;PHASE=ROOT.parent;REPO=ROOT.parents[2]
ARMS=('B0','BR','BC','S0','SC');SEEDS=(17,29,43)
VERSIONS={'torch':'2.1.2+cu118','numpy':'1.26.4','PyG':'2.7.0','torch_scatter':'2.1.2+pt21cu118','torch_sparse':'0.6.18+pt21cu118','CUDA':'11.8'}
HOSTS={
 '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs':{'hostname':'anogena-2-0','inventory':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],'python':'experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python'},
 '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git':{'hostname':'peptide','inventory':['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'],'python':'.gnnm_runtime/private_transfer_cp311_cu118_20261005_v1/bin/python'}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def bound(row):
    rel=Path(row['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Phase-relative file required')
    path=(PHASE/rel).resolve(strict=True)
    if not path.is_relative_to(PHASE) or sha(path)!=row['sha256']:raise ValueError('Bound bytes changed')
    return path

def guard(program,kind):
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();job=json.loads(args.job.read_text());out=args.output.resolve()
    if (job.get('root_execution_authorized') is not True or job.get('source_review_approved') is not True
        or job.get('TEST_access') is not False or job.get('retry') is not False or job.get('kind')!=kind):raise ValueError('Disabled exact root scientific job required')
    if str(REPO) not in HOSTS or Path.cwd().resolve()!=REPO or socket.gethostname()!=HOSTS[str(REPO)]['hostname']:raise ValueError('Authorized repository/host only')
    host=HOSTS[str(REPO)];uuid=job['physical_gpu_uuid']
    if uuid not in host['inventory'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise ValueError('Root selected visible singleton GPU required')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    observed=[dict(zip(('uuid','name','driver_version','memory_total_MiB'),[v.strip() for v in row.split(',')])) for row in rows]
    for row in observed:row['memory_total_MiB']=int(row['memory_total_MiB'])
    if [r['uuid'] for r in observed]!=host['inventory']:raise ValueError('Physical inventory changed')
    if str(Path(sys.executable).absolute())!=str(REPO/host['python']):raise ValueError('Pinned existing interpreter only')
    if out.exists() or not out.parent.is_dir() or not out.is_relative_to(PHASE) or str(out)!=job['output_directory']:raise ValueError('Fresh exact output required')
    if sha(program)!=job['program_sha256'] or sha(ROOT/'MANIFEST.json')!=job['source_manifest_sha256']:raise ValueError('Source program/manifest changed')
    for row in json.loads((ROOT/'MANIFEST.json').read_text())['files']:
        path=(ROOT/row['path']).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Source closure changed')
    for row in json.loads((ROOT/'INPUT_BINDINGS.json').read_text())['files']:bound(row)
    if not job.get('source_review_evidence'):raise ValueError('Independent source review required')
    for row in job['source_review_evidence']:bound(row)
    if type(job['soft_seconds']) is not int or type(job['hard_seconds']) is not int or not 0<job['soft_seconds']<job['hard_seconds'] or job.get('external_hard_bound_confirmed') is not True:raise ValueError('Finite owned caps required')
    plan=json.loads(bound(job['plan']).read_text());sealed=json.loads((ROOT/'PLAN.json').read_text())
    recipe=json.loads(json.dumps(plan));recipe['root_adopted']=sealed['root_adopted']
    if plan.get('root_adopted') is not True or recipe!=sealed:raise ValueError('Exact frozen development recipe required')
    context=json.loads(bound(job['execution_context']).read_text())
    if (context.get('root_adopted') is not True or context.get('repository')!=str(REPO) or context.get('hostname')!=host['hostname']
        or context.get('physical_inventory')!=observed or context.get('physical_gpu_uuid')!=uuid or context.get('plan')!=job['plan']
        or context.get('runtime_versions')!=VERSIONS or context.get('soft_seconds')!=job['soft_seconds'] or context.get('hard_seconds')!=job['hard_seconds']):raise ValueError('Exact actual host/device/runtime/plan/caps context required')
    if kind=='fit':
        if job.get('fits_authorized') is not True or job.get('VALID_values_access') is not False or job['arm'] not in ARMS or job['seed'] not in SEEDS:raise ValueError('TRAIN-only fixed fit required')
        if job['data_manifest']!=plan['TRAIN_projection']:raise ValueError('Exact TRAIN-only projection required')
        block=json.loads(bound(context['comparison_block']).read_text())
        if (block.get('root_adopted') is not True or block.get('seed')!=job['seed'] or block.get('arms')!=list(ARMS)
            or block.get('physical_gpu_uuid')!=uuid or block.get('repository')!=str(REPO) or block.get('runtime_versions')!=VERSIONS
            or block.get('donor_freeze')!=job['donor_freeze'] or block.get('donor_origin')!=job['donor_origin']
            or block.get('plan')!=job['plan']):raise ValueError('All five same-seed fits require one donor/device/runtime block')
        donor=json.loads(bound(job['donor_freeze']).read_text());origin=json.loads(bound(job['donor_origin']).read_text())
        if (donor.get('complete') is not True or donor.get('epochs')!=1100 or donor.get('seed')!=job['seed']
            or donor.get('kind')!='native_fit' or donor.get('source_manifest_sha256') not in plan['native_donor_source_manifests']
            or donor.get('program_sha256')!=plan['native_donor_program_sha256']
            or donor.get('data_manifest_sha256')!=plan['dataset']['sha256']):raise ValueError('Same complete native1100 selected donor required')
        bound(donor['selected']);bound(donor['end']);parent_job=bound(origin['source_training_job'])
        if (origin.get('complete_custody_record') is not True or origin.get('freeze')!=job['donor_freeze']
            or sha(parent_job)!=donor['job_sha256']):raise ValueError('Exact donor original job/custody required')
        bound(origin['training_terminal_evidence'])
        parent=json.loads(parent_job.read_text())
        if parent.get('source_manifest_sha256')!=donor['source_manifest_sha256'] or parent.get('program_sha256')!=donor['program_sha256']:raise ValueError('Donor source-parent differs')
        training=origin['training_device_runtime']
        matched=(training.get('repository')==str(REPO) and training.get('physical_gpu_uuid')==uuid
            and training.get('runtime_versions')==VERSIONS and training.get('python_executable')==str(REPO/host['python']))
        if not matched:
            if block.get('comparison_policy')!='cross_runtime_limit_recorded' or not block.get('cross_runtime_limit'):raise ValueError('Transferred donor origin needs explicit cross-runtime limitation')
            bound(origin['transfer_custody'])
        elif block.get('comparison_policy')!='matched_device_runtime':raise ValueError('Record the exact native/candidate device/runtime scope')
    elif kind=='score':
        if job.get('VALID_values_access') is not True or job.get('fits_authorized') is not False or job['data_manifest']!=plan['dataset']:raise ValueError('Once-only complete-matrix score role required')
        if job.get('score_seed') not in SEEDS:raise ValueError('Score one same-device seed block after all15 fits')
        records=job['endpoint_freezes']
        if {(r['seed'],r['arm']) for r in records}!={(s,a) for s in SEEDS for a in ARMS} or len(records)!=15:raise ValueError('All fifteen fixed endpoints must exist before VALID access')
        for row in records:
            freeze=json.loads(bound(row['freeze']).read_text())
            if (freeze.get('complete') is not True or freeze.get('arm')!=row['arm'] or freeze.get('seed')!=row['seed']
                or freeze.get('source_manifest_sha256')!=job['source_manifest_sha256'] or freeze.get('plan')!=job['plan']
                or freeze.get('VALID_values_access') is not False or freeze.get('class_path_updates')!=400):raise ValueError('Complete TRAIN-only endpoint required before any score')
            bound(freeze['model']);bound(freeze['training_job'])
            if row['seed']==job['score_seed']:
                parent_context=json.loads(bound(freeze['execution_context']).read_text())
                if freeze['physical_gpu_uuid']!=uuid or freeze['runtime']!=VERSIONS or parent_context['repository']!=str(REPO):raise ValueError('Endpoint serving must retain its seed training device/runtime')
    else:raise ValueError('Only fixed fitting or all-endpoint scoring')
    return args,job,out,plan,context

def runtime(context):
    import torch,numpy,torch_geometric,torch_scatter,torch_sparse
    versions={'torch':str(torch.__version__),'numpy':numpy.__version__,'PyG':torch_geometric.__version__,'torch_scatter':torch_scatter.__version__,'torch_sparse':torch_sparse.__version__,'CUDA':torch.version.cuda}
    if versions!=VERSIONS or torch.cuda.device_count()!=1:raise ValueError('Exact one-visible-GPU runtime required')
    if HOSTS[str(REPO)]['hostname']=='peptide':
        cfg=json.loads((ROOT/'GPU77_PROVIDER.json').read_text())
        if os.environ.get('PYTHONPATH','') or os.environ.get('PYTHONHOME') or sys.prefix!=cfg['runtime_prefix'] or str(Path(sys.executable).resolve())!=cfg['python_executable']['resolved_path'] or sha(Path(sys.executable).resolve())!=cfg['python_executable']['sha256']:raise ValueError('Exact self-contained GPU77 provider required')
        for name,module in {'torch':torch,'numpy':numpy,'torch_geometric':torch_geometric,'torch_scatter':torch_scatter,'torch_sparse':torch_sparse}.items():
            row=cfg['core_module_provider'][name];path=Path(module.__file__).resolve()
            if str(path)!=row['resolved_path'] or sha(path)!=row['sha256']:raise ValueError('Imported GPU77 provider changed')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    return torch,versions

def load_data(torch,job,training):
    authority=json.loads(bound(job['data_manifest']).read_text());data=torch.load(bound(authority['available']),map_location='cpu',weights_only=True)
    keys={'x','edge_index','train_ids','train_y'} if training else {'x','edge_index','train_ids','train_y','valid_ids','valid_y'}
    if set(data)!=keys or tuple(data['x'].shape)!=(11701,300) or tuple(data['edge_index'].shape)!=(2,442907) or len(data['train_ids'])!=580:raise ValueError('Full fixed WikiCS safe data roles/shapes required')
    if training and authority.get('qualified_reader_deserializes_VALID_TEST_values') is not False:raise ValueError('TRAIN-only projection required')
    if not training and (authority.get('TEST_labels_available_to_trainer') is not False or len(data['valid_ids'])!=5274):raise ValueError('Complete fixed VALID and absent TEST required')
    if data['x'].dtype!=torch.float32 or not bool(torch.isfinite(data['x']).all()) or any(data[k].dtype!=torch.long for k in keys-{'x'}):raise ValueError('Finite/type data checks')
    if (len(data['train_ids'].unique())!=580 or bool((data['train_ids']<0).any()) or bool((data['train_ids']>=11701).any())
        or bool((data['edge_index']<0).any()) or bool((data['edge_index']>=11701).any())
        or len(data['train_y'])!=580 or bool((data['train_y']<0).any()) or bool((data['train_y']>=10).any())):raise ValueError('Fixed supervision/index ranges required')
    return {k:(v.to('cuda:0') if k not in ('valid_ids','valid_y') else v) for k,v in data.items()}

def deadline(started,job):
    import time
    if time.monotonic()-started>job['soft_seconds']:raise TimeoutError('Fixed cap; preserve failure, no retry')

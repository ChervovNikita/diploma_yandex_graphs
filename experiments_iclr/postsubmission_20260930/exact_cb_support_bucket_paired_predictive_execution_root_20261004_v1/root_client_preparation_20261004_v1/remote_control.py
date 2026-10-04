"""Root-owned stdlib admission/launch/observation; execution needs external PASS pins."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE = ROOT.parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PYTHON = Path('/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12')
PYTHON_SHA = '14776d98474f987919376922a9995a20733e13b51d7d122873b068bf2e47d1b2'
SOURCE = PHASE/'exact_cb_support_bucket_paired_predictive_preparation_20261004_v1'
SOURCE_SHA = '0d0899d94f0a20d317f7e30491f3ed5b35c293d462f28a0baf3f4ea350e9d229'
SOURCE_SEAL_SHA = '00cfa256ce00a9dc19ae32f27b8535f5ea9b069ff92b48377a5244efe6f33e7d'
GPU = 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'
CAPS = {'wall_seconds':86400, 'host_RSS_bytes':32*1024**3,
        'cuda_peak_allocated_bytes':70*1024**3, 'cuda_peak_reserved_bytes':75*1024**3}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def descriptor(path):
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def check_pin(pin, root=PHASE):
    path = Path(pin['path'])
    if not path.is_absolute():
        path = root/path
    require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink(), 'Pin path outside owned phase')
    require(path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'], 'Pinned bytes differ: '+str(path))
    return path


def packet(folder, manifest_sha, seal_sha):
    require(sha(folder/'MANIFEST.json')==manifest_sha and sha(folder/'SEAL.json')==seal_sha, 'Packet binding differs')
    manifest = json.loads((folder/'MANIFEST.json').read_text())
    for row in manifest['files']:
        check_pin(dict(row, bytes=row.get('bytes',row.get('size')), path=str(folder/row['path'])), folder)
    seal = json.loads((folder/'SEAL.json').read_text())
    require(seal.get('manifest_sha256', seal.get('manifest',{}).get('sha256'))==manifest_sha, 'Seal/manifest link differs')
    return manifest


def physical(pid):
    base = Path('/proc')/str(pid)
    try:
        raw = (base/'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
        return {'PID':pid, 'parent_PID':int(fields[1]), 'process_group':int(fields[2]),
                'session':int(fields[3]), 'start_time_ticks':int(fields[19]), 'state':fields[0],
                'argv':[x.decode() for x in (base/'cmdline').read_bytes().split(bytes([0])) if x],
                'cwd':str((base/'cwd').resolve()), 'exe':str((base/'exe').resolve())}
    except FileNotFoundError:
        return None


def bound_queue(expected_sha):
    require(Path.cwd().resolve()==REPO and os.uname().nodename=='peptide', 'Exact ordinary host/repository required')
    require(PHASE==REPO/'experiments_iclr/postsubmission_20260930'
            and ROOT.name=='exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1', 'Wrong execution root')
    require(Path(sys.executable).resolve()==PYTHON.resolve() and sha(PYTHON)==PYTHON_SHA, 'Exact interpreter required')
    require('torch' not in sys.modules, 'Root queue control is stdlib only')
    path = ROOT/'QUEUE_RELEASE.json'
    require(not path.is_symlink() and sha(path)==expected_sha, 'Exact external queue release required')
    q = json.loads(path.read_text())
    require(q['schema']=='ncnc-exact-CB-fixed-nine-fit-queue-release-v1' and q['execution_enabled'] is True
            and q['root_authorization_reference'], 'Queue is not root released')
    packet(SOURCE, SOURCE_SHA, SOURCE_SEAL_SHA)
    packet(HERE, q['client_manifest_sha256'], q['client_seal_sha256'])
    plan = json.loads((HERE/'CLIENT_PLAN.json').read_text())
    require(q['source_manifest_sha256']==SOURCE_SHA and q['source_seal_sha256']==SOURCE_SEAL_SHA
            and q['queue_order']==plan['queue_order'] and q['queue_order_sha256']==plan['queue_order_sha256']
            and len(q['queue_order'])==9 and q['GPU_UUID']==GPU and q['fit_caps']==CAPS
            and q['automatic_retry_or_resume'] is False and q['TEST_access'] is False, 'Fixed family queue differs')
    review_pins = json.loads(check_pin(q['external_review_pins']).read_text())
    review_file = check_pin(review_pins['review_file'])
    review_manifest = check_pin(review_pins['review_manifest'])
    review_seal = check_pin(review_pins['review_seal'])
    require(review_file.parent==review_manifest.parent==review_seal.parent, 'One exact independent review packet required')
    packet(review_file.parent, review_pins['review_manifest']['sha256'], review_pins['review_seal']['sha256'])
    review = json.loads(review_file.read_text())
    require(review['status']=='PASS' and not review.get('blocking_findings')
            and review['candidate_manifest_sha256']==SOURCE_SHA and review['candidate_seal_sha256']==SOURCE_SEAL_SHA
            and review['reviewer_is_source_preparer'] is False and review['execution_authorized'] is False,
            'Fresh exact external independent PASS required')
    admission = json.loads(check_pin(q['root_admission']).read_text())
    require(admission['status']=='APPROVED_FIXED_NINE_FITS' and admission['client_manifest_sha256']==q['client_manifest_sha256']
            and admission['source_manifest_sha256']==SOURCE_SHA and admission['queue_order_sha256']==plan['queue_order_sha256'],
            'Root admission does not bind this queue')
    source_plan = json.loads((SOURCE/'PILOT_PLAN.json').read_text())
    require(q['family_id']==source_plan['family_id'] and sha(SOURCE/'PILOT_PLAN.json')==q['source_plan_sha256'], 'Fixed source plan differs')
    require(len(q['cell_releases'])==9, 'Exactly nine external single-cell releases required')
    for cell, pin in zip(q['queue_order'], q['cell_releases']):
        rpath = check_pin(pin, ROOT)
        r = json.loads(rpath.read_text())
        require(r['driver_manifest_sha256']==SOURCE_SHA and r['fit_supervision_manifest_sha256']==SOURCE_SHA
                and r['authorized_invocations']==[cell['invocation']] and r['authorized_stages']==['fit']
                and r['execution_enabled'] is True and r['cuda_visible_devices']==GPU and r['fit_caps']==CAPS,
                'Single-cell root release differs')
    return q, source_plan


def admission(expected_sha):
    q, plan = bound_queue(expected_sha)
    require(not (ROOT/'LAUNCH_ATTEMPT_SPENT.json').exists(), 'Queue launch already spent')
    mapping = json.loads((SOURCE/'INPUT_BINDINGS.json').read_text())['server_receipt_to_local_mirror_mapping']
    for row in mapping:
        check_pin(row['server_canonical_receipt'])
    for cell in q['queue_order']:
        require(not Path(cell['invocation']['output_directory']).exists()
                and not Path(cell['supervision_output_directory']).exists(), 'Prospective cell already has outputs')
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True,timeout=15)
    rows = [dict(zip(('uuid','name','total_MiB','free_MiB'), [x.strip() for x in line.split(',')]))
            for line in gpu.stdout.splitlines() if line.strip()]
    require({r['uuid'] for r in rows}=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',GPU}
            and all(r['name']=='NVIDIA A100 80GB PCIe' and int(r['total_MiB'])==81920 for r in rows), 'Physical GPU route differs')
    spec = importlib.util.spec_from_file_location('fixed_predictive_stdlib_gate',SOURCE/'pilot_common.py')
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    for cell, pin in zip(q['queue_order'],q['cell_releases']):
        context = common.preflight(pin['path'],'fit')
        inv = cell['invocation']
        common.admit_invocation(context,'fit',inv['unit'],inv['base_seed'],inv['output_directory'],False)
    result = {'UTC':utc(),'status':'PASS_FIXED_NINE_CELL_STDLIB_ADMISSION', 'queue_release_sha256':expected_sha,
              'source_manifest_sha256':SOURCE_SHA, 'client_manifest_sha256':q['client_manifest_sha256'],
              'queue_order_sha256':q['queue_order_sha256'], 'interpreter_sha256':PYTHON_SHA,
              'selected_GPU':next(r for r in rows if r['uuid']==GPU), 'GPU_observation_is_reservation':False,
              'numerical_execution':False,'new_QA_ladder':False,'TEST_access':False}
    write_new(ROOT/'PRENUMERICAL_ADMISSION.json',result)
    return result


def launch(expected_sha):
    q, plan = bound_queue(expected_sha)
    admitted = json.loads((ROOT/'PRENUMERICAL_ADMISSION.json').read_text())
    require(admitted['status']=='PASS_FIXED_NINE_CELL_STDLIB_ADMISSION'
            and admitted['queue_release_sha256']==expected_sha, 'Exact prior admission required')
    require(not (ROOT/'queue/run01').exists() and not (ROOT/'DETACHED_LAUNCH.json').exists(), 'Owned queue already exists')
    for cell in q['queue_order']:
        require(not Path(cell['invocation']['output_directory']).exists()
                and not Path(cell['supervision_output_directory']).exists(), 'Cell output exists')
    write_new(ROOT/'LAUNCH_ATTEMPT_SPENT.json', {'UTC':utc(),'source_manifest_sha256':SOURCE_SHA,
              'client_manifest_sha256':q['client_manifest_sha256'],'queue_release_sha256':expected_sha,
              'queue_order_sha256':q['queue_order_sha256'],'no_blind_retry':True})
    command = [str(PYTHON),'-B',str(HERE/'owned_queue.py'),'--queue-release-sha256',expected_sha]
    environment = {'CUDA_VISIBLE_DEVICES':GPU,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
                   'PYTHONPATH':str(REPO/'.gnnm_runtime/buddy_extra_v1/site'), 'PYTHONDONTWRITEBYTECODE':'1',
                   'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                   'GNNM_SSH_DESTINATION':'shmelev@192.168.18.77'}
    with (ROOT/'DETACHED_STDOUT.txt').open('xb') as out, (ROOT/'DETACHED_STDERR.txt').open('xb') as err:
        child = subprocess.Popen(command,cwd=REPO,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,
                                 stdout=out,stderr=err,start_new_session=True)
    identity = physical(child.pid)
    require(identity is not None and identity['argv']==command and identity['cwd']==str(REPO)
            and identity['exe']==str(PYTHON.resolve()) and identity['session']==child.pid
            and identity['process_group']==child.pid, 'Detached queue physical handle differs')
    result = {'UTC':utc(),'status':'OWNED_FIXED_NINE_FIT_QUEUE_LAUNCHED', 'queue_physical_identity':identity,
              'command':command,'environment':environment,'source_manifest_sha256':SOURCE_SHA,
              'client_manifest_sha256':q['client_manifest_sha256'],'queue_release_sha256':expected_sha,
              'queue_order_sha256':q['queue_order_sha256'], 'automatic_retry_or_resume':False,'TEST_access':False}
    write_new(ROOT/'DETACHED_LAUNCH.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('admit','launch'))
    parser.add_argument('--queue-release-sha256',required=True)
    args=parser.parse_args()
    result=admission(args.queue_release_sha256) if args.mode=='admit' else launch(args.queue_release_sha256)
    print(json.dumps(result,allow_nan=False))

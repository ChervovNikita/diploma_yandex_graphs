"""Own one finite, frozen complete-seed shard on an admitted 77 GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)

def identity(pid):
    q = Path('/proc', str(pid), 'stat')
    if not q.exists():
        return None
    v = q.read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, start_ticks=int(v[19]), state=v[0])

def stamp():
    return datetime.now(timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', choices=['shard0', 'shard1'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    assert str(repo) == '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
    assert socket.gethostname() == 'peptide' and Path.cwd() == repo
    freeze = json.loads((root/'FREEZE.json').read_text())
    ready = json.loads((root/'ACTUAL_READY_V1.json').read_text())
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ready['gpu_uuids']
    for row in freeze['bound_files']:
        assert hashlib.sha256((repo/row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    assert json.loads((root/'ACTUAL_QUALIFICATION_V1.json').read_text())['qualified']
    shard = freeze['shards'][args.shard]
    here = root/args.shard
    config = here/'CONFIG.json'
    output = here/'actual_family_v1'
    assert not output.exists() and not (here/'OWNER_START.json').exists()
    gpu = shard['gpu_uuid']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu
    source = repo/'experiments_iclr/postsubmission_20260930/SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py'
    normalized = dict(schema_version=1, study_id=freeze['study_id'], stage='F', K=None,
                      hostname=socket.gethostname(), runtime_fingerprint_sha256=ready['runtime_fingerprint_sha256'],
                      gpu_uuid=gpu, logical_device='cuda:0', seed_block=shard['seed_block'],
                      submitted_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                      training_family_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      safe_role_hashes={k: ready['roles'][k]['file']['sha256'] for k in ['train', 'valid']},
                      TEST_access=False)
    write(here/'EXECUTION_RECEIPT.json', dict(normalized, status='HOST_GPU_SOURCE_DATA_ADMITTED', UTC=stamp(), freeze_sha256=hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', DGLBACKEND='pytorch')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    command = [ready['runtime']['executable'], '-B', str(source), '--config', str(config), '--output', str(output), '--seed-block', *map(str, shard['seed_block'])]
    start = time.monotonic()
    with (here/'worker.stdout.log').open('x') as out, (here/'worker.stderr.log').open('x') as err:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
        child_id = identity(child.pid)
        write(here/'OWNER_START.json', dict(normalized, UTC=stamp(), owner=identity(os.getpid()), child=child_id, command=command, freeze_sha256=hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), finite_seconds=21600))
        timed_out = False
        try:
            code = child.wait(timeout=21600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    absent = identity(child.pid) is None
    cuda_absent = str(child.pid) not in [x.strip() for x in gpu_pids]
    complete = output/'COMPLETE_FAMILY.json'
    good = False
    if code == 0 and complete.exists():
        j = json.loads(complete.read_text())
        n = len(shard['seed_block'])
        good = (j['complete'] is True and j['groups'] == 5*n and j['optimizer_acquisition_bundles'] == 8*n
                and j['native_body_fit_records'] == 11*n and j['completed_seed_block'] == shard['seed_block']
                and j['stage'] == 'F' and j['K'] is None and not j['TEST_access'])
    success = code == 0 and good and absent and cuda_absent
    write(here/'OWNER_END.json', dict(normalized, UTC=stamp(), status='COMPLETE_EXIT_0' if success else 'FAILED_RETAINED',
          exit_code=code, scientific_success=success, complete_family=good, direct_child_wait=True,
          child=child_id, child_pid_absent=absent, owned_cuda_pid_absent=cuda_absent, timed_out=timed_out,
          seconds=time.monotonic()-start, complete_sha256=hashlib.sha256(complete.read_bytes()).hexdigest() if complete.exists() else None,
          overlap='Two prospective seed shards on distinct admitted GPUs; observed costs, not isolated benchmarks.'))
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

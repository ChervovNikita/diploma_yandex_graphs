from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'private_sheaf_post_all21_scientific_readout_20261009_v1'
S = P / 'private_sheaf_post_all21_analysis_source_20261009_v2'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '24c315a7-3c08-471f-b550-b9a3e1faf75d'
os.chdir(R)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

assert sha(A / 'RELEASE.json') == 'c326d4f19aee2d8a10b59f72738f38b5269b27a401480e178d9f49f6848117b6'
release = read(A / 'RELEASE.json')
assert sha(S / 'analysis.py') == release['analysis_source_sha256'] == '4af0e1e5590ed0b5bfe5c0f2d49f9a660509a6da714786261c36b286cc54fd7a'
custody = read(A / 'ROOT_CUSTODY.json')
assert sha(A / 'ROOT_CUSTODY.json') == release['root_custody']['sha256']
science_pids = {value[key] for value in custody['families'].values() for key in ('parent_pid', 'worker_pid')}
groups = {value[key] for value in custody['families'].values() for key in ('parent_group', 'worker_group')}
assert all(not Path('/proc', str(pid)).exists() for pid in science_pids)
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():
        continue
    try:
        raw = (folder / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        continue
    assert int(raw[raw.rfind(')') + 2:].split()[2]) not in groups
cuda_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
assert all(int(row.split(',')[1].strip()) not in science_pids for row in cuda_rows if len(row.split(',')) == 3)
closure = read(Path(release['centered_closure']['path']))
assert closure['complete'] is True and len(closure['logical_records']) == 21 and all(row['status'] == 'complete' for row in closure['logical_records'])
runtime_path = Path(release['qualified_runtime_binding']['path'])
assert sha(runtime_path) == release['qualified_runtime_binding']['sha256']
runtime = read(runtime_path)
output = Path(release['output_directory'])
assert not output.exists() and not (A / 'EXECUTION_RECEIPT.json').exists()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
    PYTHONPATH=os.pathsep.join([str(P / 'private_sheaf_dependency_overlay_20261009_v1'), *runtime['PYTHONPATH']]),
    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
env.pop('PYTHONHOME', None)
command = [runtime['python'], '-B', str(S / 'analysis.py'), '--execute', '--release', str(A / 'RELEASE.json'), '--output', str(output)]
started = time.perf_counter()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
with (A / 'READER.log').open('xb') as stream:
    completed = subprocess.run(command, cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
after = resource.getrusage(resource.RUSAGE_CHILDREN)
files = []
if completed.returncode == 0:
    analysis = read(output / 'ANALYSIS.json')
    assert analysis['all21_verified_before_numeric_or_reference_access'] is True
    assert analysis['exact_completed_result_reference_history_provenance'] is True
    assert analysis['source_sha256'] == release['analysis_source_sha256'] and analysis['release_sha256'] == sha(A / 'RELEASE.json')
    assert len(analysis['quality']) == 90 and len(analysis['contrasts']) == 4
    for path in sorted(output.iterdir()):
        assert path.name in {'ANALYSIS.json', 'EVERY_MEMBER_QUALITY.csv', 'REPORT.md'}
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
receipt = dict(schema='root-authorized-closed-all21-one-shot-reader-execution-v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), complete=completed.returncode == 0,
    exit_code=completed.returncode, command=command, hostname=socket.gethostname(), GPU_UUID=GPU,
    CUDA_VISIBLE_DEVICES='', CPU_only=True, release_sha256=sha(A / 'RELEASE.json'),
    reader_source_sha256=sha(S / 'analysis.py'), runtime_binding=release['qualified_runtime_binding'],
    analysis_wall_seconds=time.perf_counter() - started,
    analysis_CPU_user_seconds=after.ru_utime - before.ru_utime,
    analysis_CPU_system_seconds=after.ru_stime - before.ru_stime,
    reader_peak_RSS_bytes=after.ru_maxrss * 1024, reader_peak_RSS_is_child_highwater=True,
    all21_custody_reverified_before_invocation=True, complete_outputs=files,
    reader_log=dict(path=str(A / 'READER.log'), bytes=(A / 'READER.log').stat().st_size, sha256=sha(A / 'READER.log')),
    new_fits=0, model_forwards=0, new_predictions=0, thresholds_or_original_scores_modified=False,
    automatic_retry=False, other_workloads_changed=False)
with (A / 'EXECUTION_RECEIPT.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print(json.dumps(receipt, sort_keys=True))
assert completed.returncode == 0, 'Preserve reader failure without retry or source change'

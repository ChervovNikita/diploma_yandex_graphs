from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BOOT = '24c315a7-3c08-471f-b550-b9a3e1faf75d'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
os.chdir(R)

def read(path):
    return json.loads(path.read_text())

def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

handles = {
    'original18': ('geometry_only_core_scientific18_activation_root_20261009_v1', 563273, 6035067559, 563276, 6035067565),
    'centered3': ('geometry_only_core_centered3_activation_root_20261009_v1', 576849, 6037023042, 576853, 6037023053),
}
families = {}
all_groups = set()
cuda_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
cuda_pids = {int(row.split(',')[1].strip()) for row in cuda_rows if len(row.split(',')) == 3 and row.split(',')[0].strip() == GPU}
for family, (name, parent_pid, parent_birth, worker_pid, worker_birth) in handles.items():
    folder = P / name
    launch, terminal = read(folder / 'LAUNCH.json'), read(folder / 'TERMINAL.json')
    parent, worker = launch['parent'], terminal['child']
    assert (parent['pid'], parent['start_ticks'], parent['boot_id']) == (parent_pid, parent_birth, BOOT)
    assert (worker['pid'], worker['start_ticks'], worker['boot_id']) == (worker_pid, worker_birth, BOOT)
    assert parent['group'] == parent_pid and worker['group'] == worker_pid
    assert all(terminal[key] is True for key in ('complete', 'reaped', 'actual_worker_absent', 'actual_worker_CUDA_absent'))
    assert terminal['exit_code'] == 0
    assert not Path('/proc', str(parent_pid)).exists() and not Path('/proc', str(worker_pid)).exists()
    assert worker_pid not in cuda_pids and parent_pid not in cuda_pids
    all_groups.update([parent['group'], worker['group']])
    families[family] = dict(parent_pid=parent_pid, parent_birth=parent_birth, parent_group=parent['group'],
        worker_pid=worker_pid, worker_birth=worker_birth, worker_group=worker['group'], boot_id=BOOT,
        launch=binding(folder / 'LAUNCH.json'), terminal=binding(folder / 'TERMINAL.json'),
        reaped=True, exit_code=0, actual_parent_absent=True, actual_worker_absent=True, actual_worker_CUDA_absent=True)
group_members = []
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():
        continue
    try:
        raw = (folder / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        continue
    fields = raw[raw.rfind(')') + 2:].split()
    if int(fields[2]) in all_groups:
        group_members.append(dict(pid=int(folder.name), group=int(fields[2]), birth=int(fields[19]), state=fields[0]))
assert not group_members
for family in families.values():
    family.update(actual_parent_group_absent=True, actual_group_absent=True)

centered_root = P / 'geometry_only_core_centered3_execution_root_20261009_v1'
closure = read(centered_root / 'ALL21_CLOSURE.json')
expected = {('shared_fit', seed, -1) for seed in (7409, 8501, 9607)}
expected |= {('independent_pool', seed, -1) for seed in (7409, 8501, 9607)}
expected |= {('independent_member', seed, member) for seed in (7409, 8501, 9607) for member in range(4)}
expected |= {('centered_shared_fit', seed, -1) for seed in (7409, 8501, 9607)}
assert all(closure[key] is True for key in ('complete', 'original18_complete', 'centered3_complete'))
assert closure['required_logical_records'] == 21 and len(closure['logical_records']) == 21
assert {(row['kind'], row['base_seed'], row.get('member', -1)) for row in closure['logical_records']} == expected
assert all(row['status'] == 'complete' for row in closure['logical_records'])

# All science identities/groups/CUDA and status-only all21 closure passed above.
original_root = P / 'geometry_only_core_scientific18_execution_root_20261009_v1'
original = read(original_root / 'COMPLETE.json')
assert original['complete'] is True and len(original['records']) == 18 and all(row['status'] == 'complete' for row in original['records'])
runtime_path = P / 'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json'
runtime = read(runtime_path)
release = read(P / 'geometry_only_core_centered3_activation_root_20261009_v1/RELEASE.json')
origins_path = Path(release['original18_origins']['path'])
assert origins_path.is_relative_to(P) and origins_path.exists()
reader = P / 'private_sheaf_post_all21_analysis_source_20261009_v2/analysis.py'
assert hashlib.sha256(reader.read_bytes()).hexdigest() == '4af0e1e5590ed0b5bfe5c0f2d49f9a660509a6da714786261c36b286cc54fd7a'
custody = dict(schema='root-exact-post-all21-science-custody-template-v1', release_owner='root',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), hostname=socket.gethostname(), GPU_UUID=GPU,
    all21_complete_verified=True, families=families,
    observation=dict(bound_boot=BOOT, science_group_members=group_members, observed_CUDA_PIDs=sorted(cuda_pids), other_workloads_unchanged=True))
print(json.dumps(dict(custody=custody, centered_closure=binding(centered_root / 'ALL21_CLOSURE.json'),
    centered_records=binding(centered_root / 'CENTERED_RECORDS.json'), original_complete=binding(original_root / 'COMPLETE.json'),
    original18_origins=binding(origins_path), runtime_binding=binding(runtime_path), runtime=runtime,
    reader=binding(reader), git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    completed_records=21, outcomes_printed=False), sort_keys=True))

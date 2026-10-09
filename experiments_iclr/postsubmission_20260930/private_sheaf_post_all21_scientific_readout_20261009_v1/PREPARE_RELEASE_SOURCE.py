from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'private_sheaf_post_all21_scientific_readout_20261009_v1'
S = P / 'private_sheaf_post_all21_analysis_source_20261009_v2'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '24c315a7-3c08-471f-b550-b9a3e1faf75d'
os.chdir(R)

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def binding(path):
    path = Path(path).resolve(strict=True)
    assert path.is_relative_to(P)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

assert sha(S / 'analysis.py') == '4af0e1e5590ed0b5bfe5c0f2d49f9a660509a6da714786261c36b286cc54fd7a'
custody = json.loads(CUSTODY_JSON)
groups = set()
cuda_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True).splitlines()
cuda_pids = {int(row.split(',')[1].strip()) for row in cuda_rows if len(row.split(',')) == 3 and row.split(',')[0].strip() == GPU}
for value in custody['families'].values():
    for name in ('launch', 'terminal'):
        assert sha(Path(value[name]['path'])) == value[name]['sha256']
    terminal = read(Path(value['terminal']['path']))
    assert terminal['complete'] is terminal['reaped'] is terminal['actual_worker_absent'] is terminal['actual_worker_CUDA_absent'] is True
    assert terminal['exit_code'] == 0
    for key in ('parent_pid', 'worker_pid'):
        assert not Path('/proc', str(value[key])).exists() and value[key] not in cuda_pids
    groups.update([value['parent_group'], value['worker_group']])
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit():
        continue
    try:
        raw = (folder / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        continue
    assert int(raw[raw.rfind(')') + 2:].split()[2]) not in groups
closure_path = P / 'geometry_only_core_centered3_execution_root_20261009_v1/ALL21_CLOSURE.json'
closure = read(closure_path)
assert closure['complete'] is closure['original18_complete'] is closure['centered3_complete'] is True
assert closure['required_logical_records'] == len(closure['logical_records']) == 21
assert all(row['status'] == 'complete' for row in closure['logical_records'])
custody['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
custody['observation']['observed_CUDA_PIDs'] = sorted(cuda_pids)

centered_release = read(P / 'geometry_only_core_centered3_activation_root_20261009_v1/RELEASE.json')
origins_path = Path(centered_release['original18_origins']['path'])
assert sha(origins_path) == centered_release['original18_origins']['sha256'] == '26a7a654f7470dda5cdb97aa019505cb1852e1ea943435998f45fb4a6cc78f59'
origins = read(origins_path)
index = {(row['kind'], row['base_seed'], row.get('member', -1)): row for row in origins['records']}
assert len(index) == 18 and all(row['status'] == 'complete' for row in index.values())
records_path = closure_path.parent / 'CENTERED_RECORDS.json'
centered_records = read(records_path)
assert len(centered_records) == 3 and all(row['status'] == 'complete' for row in centered_records)
centered_index = {row['base_seed']: row for row in centered_records}
release = read(S / 'RELEASE_TEMPLATE_DISABLED.json')
assert release['enabled'] is False
release.update(enabled=True, output_directory=str(A / 'complete_analysis'))
release['original_complete'] = binding(Path(origins['original_complete']['path']))
assert release['original_complete']['sha256'] == origins['original_complete']['sha256'] == 'a90060864683d9104e59e6e63f86e918ff04d9faae436d6ea07719b6746e9fad'
release['original18_origins'] = binding(origins_path)
release['centered_closure'] = binding(closure_path)
release['centered_records'] = binding(records_path)
for row in release['references']:
    family, seed = row['family'], row['base_seed']
    if family == 'centered':
        folder = closure_path.parent / 'shared' / ('seed' + str(seed))
        ref = binding(folder / 'SELECTED_SERVING_REFERENCE.pt')
        assert ref['sha256'] == centered_index[seed]['serving_reference_sha256']
        row.update(ref)
        row['result'] = binding(folder / 'RESULT.json')
        row['histories'] = [binding(folder / 'HISTORY.jsonl')]
    else:
        kind = 'shared_fit' if family == 'vanilla' else 'independent_pool'
        origin = index[(kind, seed, -1)]
        artifacts = {value['label']: value for value in origin['artifacts']}
        ref = binding(Path(artifacts['serving_reference']['path']))
        assert ref['sha256'] == artifacts['serving_reference']['sha256']
        row.update(ref)
        row['result'] = binding(Path(origin['result']['path']))
        assert row['result']['sha256'] == origin['result']['sha256']
        histories = []
        for member in range(4) if family == 'independent' else [-1]:
            owner = index[('independent_member', seed, member)] if family == 'independent' else origin
            history_origin = next(value for value in owner['artifacts'] if value['label'] == 'history')
            history = binding(Path(history_origin['path']))
            assert history['sha256'] == history_origin['sha256']
            if family == 'independent':
                history['member'] = member
            histories.append(history)
        row['histories'] = histories
assert len(release['references']) == 9 and sum(len(row['histories']) for row in release['references']) == 18
runtime_path = P / 'staged_label_posterior_native_metadata_root_20261008_v1/RUNTIME.json'
assert sha(runtime_path) == '919b1053f66dfb962c8f9c054f1640d9d8ac1011c37cc88125a7300867c6f198'
runtime = read(runtime_path)
assert runtime['torch'] == release['torch_version'] and runtime['numpy'] == release['numpy_version']
A.mkdir(exist_ok=False)
write(A / 'ROOT_CUSTODY.json', custody)
release['root_custody'] = binding(A / 'ROOT_CUSTODY.json')
release['root_explicit_all21_readout_authorized'] = True
release['no_fits_or_threshold_changes'] = True
release['qualified_runtime_binding'] = binding(runtime_path)
write(A / 'RELEASE.json', release)
print(json.dumps(dict(release=binding(A / 'RELEASE.json'), custody=custody, root_custody=binding(A / 'ROOT_CUSTODY.json'),
    references_bound=9, own_histories_bound=18, release_value=release, output_directory=release['output_directory'],
    runtime=runtime, raw_array_or_checkpoint_download=False), sort_keys=True))

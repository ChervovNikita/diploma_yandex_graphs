"""PubMed LM4 finite reader admission and exact owner/artifact custody; stdlib only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (9101, 9203, 9307)
STAGE2 = ('B1', 'B4', 'T16', 'S64', 'FS64', 'LM4', 'I4[S64]', 'I4[FS64]')
ENDPOINTS = ('F1_b0', 'F1_b1', 'F1_b2', 'F1_b3') + STAGE2 + ('GA4',)
RAW_REFERENCES = ('F1_b0', 'GA4', 'B1', 'B4', 'T16', 'S64', 'FS64', 'I4[S64]', 'I4[FS64]')
SPEC_SHA = '60ec176d581f693633c537be7b82af63d6795af51ec01ede716d3918d7193bf5'
NOMINATION_SHA = '84d44d954104d0059db53f2393ecd1a99944784975d0b63dff6e366cb816e0fb'
ACCOUNTING = dict(stage1_fit_banks=12, stage1_trajectories=12, stage2_fit_banks=24,
    stage2_trajectories=42, total_new_fit_banks=36, total_new_trajectories=54,
    CAL_endpoints=39, CAL_scalar_updates=19500, inherited_fits_or_CAL=0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def route():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=15).splitlines() == [UUID], 'Excluded wrong sole GPU')
    require(Path.cwd() == REPO, 'Exact allocation repository required')


def receipt(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'Receipt outside authorized project')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(item):
    require(isinstance(item, dict) and set(item) in ({'path', 'sha256'}, {'path', 'bytes', 'sha256'}), 'Bound receipt schema')
    path = Path(item['path'])
    if not path.is_absolute():
        path = PHASE / path
    got = receipt(path)
    require(got['sha256'] == item['sha256'] and ('bytes' not in item or got['bytes'] == item['bytes']),
        'Bound file differs: ' + str(path))
    return Path(got['path'])


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_check(item):
    path = checked(item)
    manifest = json.loads(path.read_text())
    for rel, fixed in manifest['files'].items():
        got = receipt(path.parent / rel)
        require(got['bytes'] == fixed['bytes'] and got['sha256'] == fixed['sha256'], 'Sealed source differs: ' + rel)
    return path.parent


def identity(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]))


def owner_closed(end_item, stage, config_item=None, start_item=None, source_manifest=None):
    end_path = checked(end_item)
    end = json.loads(end_path.read_text())
    require(end['stage'] == stage and end['status'] == 'COMPLETE_EXIT_0' and end['exit_code'] == 0
        and end['timed_out'] is False and end['retry'] is False and end['direct_child_wait'] is True
        and end['child_pid_absent'] is True and end['owned_cuda_pid_absent'] is True,
        'Successful exact-stage direct owner closure')
    require(identity(end['child']['pid']) != end['child'], 'Owned worker remains live')
    if config_item is not None:
        config_path = checked(config_item)
        action = json.loads(config_path.read_text())
        start_path = checked(start_item)
        start = json.loads(start_path.read_text())
        source = Path(source_manifest['path']).parent
        binding = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
        interpreter = str(checked(binding['server_data_and_runtime']['runtime']['python']))
        command = [interpreter, '-B', '-u', str(source / 'worker.py'), str(config_path)]
        require(action['stage'] == stage and action['execution_enabled'] is True
            and action['source_manifest'] == source_manifest
            and start_path == Path(action['owner_output']) / 'OWNER_START.json'
            and end_path == Path(action['owner_output']) / 'OWNER_END.json', 'Producer CONFIG/source/owner-output linkage')
        for value in (start, end):
            require(value['stage'] == stage and value['command'] == command
                and value['config']==config_item and value['source_manifest']==source_manifest
                and value['root_decision']==action['root_decision'] and value['output']==action['output']
                and value['child_wall_seconds'] == action['limits']['wall_seconds']
                and value['external_wall_seconds'] == action['limits']['wall_seconds'] + 60,
                'Producer stage/command/CONFIG/finite-cap linkage')
        require(end['owner'] == start['owner'] and end['child'] == start['child']
            and start['child']['pgid'] == start['child']['pid'], 'Same producer owner/worker birth identities')
        name=dict(prepare='CUSTODY.json',qualify='QUALIFICATION.json',acquire='PANEL.json',
            calibrate='TEMPERATURES.json',score='DEVELOPMENT_REPORT.json')[stage]
        require(end['produced_artifact']==receipt(Path(action['output'])/name), 'Exact successful owner-produced artifact receipt')
        return end, action
    return end


def guard(config, start):
    require(time.monotonic() - start < config['limits']['wall_seconds']
        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= config['limits']['rss_bytes'],
        'Finite CPU reader wall/RSS cap')


def providers(config):
    actual = {name: importlib.metadata.version(name) for name in config['expected_runtime_versions']}
    require(actual == config['expected_runtime_versions'], 'Exact PolyFormer provider versions')
    return actual


def admit(config, stage):
    require(stage in ('calibrate', 'score') and config.get('stage') == stage
        and config.get('execution_enabled') is True, 'Exact root-enabled reader stage')
    require(config.get('seeds') == list(SEEDS) and config.get('endpoint_ids') == list(ENDPOINTS), 'Fixed complete roster')
    require(config.get('finite_external_owner') is True
        and os.environ.get('PILOT_OWNER_PID') == str(os.getppid()), 'Direct finite owner required')
    limits = config['limits']
    require(set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
        and all(type(v) is int and v > 0 for v in limits.values())
        and limits['cuda_bytes'] <= 32 * 1024**3 and limits['rss_bytes'] <= 16 * 1024**3, 'Positive finite source resource envelope')
    require(all(Path(config[k]).is_absolute() and Path(config[k]).resolve().is_relative_to(REPO)
        for k in ('output', 'owner_output')), 'Project-local fresh outputs')
    decision = json.loads(checked(config['root_decision']).read_text())
    require(stage in decision.get('admitted_stages', []) and decision.get('specification_sha256') == SPEC_SHA,
        'Explicit root readout admission for fixed LM4 nomination')
    require(source_check(config['source_manifest']) == HERE, 'Exact current reader source')
    require(receipt(HERE / 'FIXED_SPEC.json')['sha256'] == SPEC_SHA, 'Fixed nomination changed')
    binding = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    require(config['acquisition_source_manifest'] == binding['acquisition_source_manifest']
        and config['expected_runtime_versions'] == binding['server_data_and_runtime']['frozen_providers'],
        'Exact training-source/runtime binding')
    return dict(stage=stage, root_decision=config['root_decision'], source_manifest=config['source_manifest'],
        specification=receipt(HERE / 'FIXED_SPEC.json'), nomination_manifest_sha256=NOMINATION_SHA)

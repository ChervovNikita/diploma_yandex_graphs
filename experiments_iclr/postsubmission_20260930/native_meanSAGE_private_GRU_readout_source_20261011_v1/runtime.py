"""Finite reader admission and custody; stdlib-only module imports."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOCKS = (10103, 10427, 10949)
STAGE2 = ('B4', 'G1', 'GF1', 'GM4', 'GN4', 'GI16', 'GFI16')
ENDPOINTS = ('P1_b0', 'P1_b1', 'P1_b2', 'P1_b3') + STAGE2 + ('GA4', 'inherited_WI4')
RAW_REFERENCES = ('P1_b0', 'B4', 'G1', 'GF1', 'GN4', 'GI16', 'GFI16', 'GA4', 'inherited_WI4')
SPEC_SHA = 'daa1bfc411f4f6a2a8e5511dad4f45d3ec47435dfd67ef2c42fc8dc7a998c7e2'
ACCOUNTING = dict(cohort_fit_banks=33, cohort_optimizer_trajectories=51,
    cohort_new_fit_banks=30, cohort_new_optimizer_trajectories=48,
    inherited_P1_fit_banks=3, inherited_P1_optimizer_trajectories=3,
    external_WI4_fit_banks=3, external_WI4_optimizer_trajectories=12,
    derived_GA4_rows=3, extra_P1_ancestor_diagnostic_rows=9, serving_rows=39)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def route():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=15).splitlines() == [UUID], 'Excluded wrong sole GPU')
    require(Path.cwd() == REPO, 'Authorized allocation repository required')

def receipt(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'Receipt outside authorized project')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            digest.update(data)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())

def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'Exact receipt schema')
    require(receipt(item['path']) == item, 'Bound file differs: ' + item['path'])
    return Path(item['path'])

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def admit(config, stage):
    require(stage in ('calibrate', 'score') and config.get('stage') == stage
        and config.get('execution_enabled') is True, 'Exact root-enabled reader stage required')
    require(config.get('blocks') == list(BLOCKS) and config.get('endpoint_ids') == list(ENDPOINTS),
        'Fixed three-block/thirteen-endpoint interface required')
    require(all(Path(config[k]).is_absolute() and Path(config[k]).resolve().is_relative_to(REPO)
        for k in ('output', 'owner_output')), 'Fresh output/owner paths inside authorized project')
    require(config.get('finite_external_owner') is True
        and os.environ.get('PILOT_OWNER_PID') == str(os.getppid()), 'Direct finite owner required')
    limits = config['limits']
    require(set(limits) == {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
        and all(type(v) is int and v > 0 for v in limits.values()), 'Positive finite resource caps')
    require(config['expected_runtime_versions'] == {'numpy': '1.26.4', 'torch': '2.1.2+cu118'},
        'Exact inherited runtime versions required')
    decision = json.loads(checked(config['root_decision']).read_text())
    require(stage in decision.get('admitted_stages', []) and decision.get('v2_spec_sha256') == SPEC_SHA,
        'Root must explicitly admit exact v2 reader stage')
    manifest = json.loads(checked(config['source_manifest']).read_text())
    for rel, expected in manifest['files'].items():
        got = receipt(HERE / rel)
        require(got['bytes'] == expected['bytes'] and got['sha256'] == expected['sha256'],
            'Fixed reader source differs: ' + rel)
    require(receipt(HERE / 'FIXED_SPEC.json')['sha256'] == SPEC_SHA, 'V2 specification changed')
    boundary = json.loads((HERE / 'PROSPECTIVE_BOUNDARY.json').read_text())
    require(boundary['v2_spec_sha256'] == SPEC_SHA
        and boundary['overall_bound_CAL'] == dict(endpoints=39, new_endpoints=33,
            inherited_endpoints=6, total_steps=19500, new_steps=16500, inherited_steps=3000),
        'Prospective CAL boundary changed')
    require(set(config['inherited_sources']) == {'P1', 'WI4'}, 'Both inherited source families required')
    binding = json.loads((HERE / 'BINDING.json').read_text())
    require(config['inherited_sources'] == binding['inherited_sources']
        and config['custody'] == binding['role_custody'], 'Exact inherited families and role receipt')
    return dict(stage=stage, root_decision=config['root_decision'],
        source_manifest=config['source_manifest'], specification=receipt(HERE / 'FIXED_SPEC.json'),
        interpretation_boundary=receipt(HERE / 'PROSPECTIVE_BOUNDARY.json'))

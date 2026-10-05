"""Reviewed stdlib metadata helper subset; no numerical/GPU execution at import."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
RUNTIME_EVIDENCE = {'authorized_one_GPU_allocation': {'path':'citeseer_ncn_native_runtime_qualification_20261005_v2/RESULT.json','sha256':'2dfa804a4761c559b6fa2f954f98300b81dabd4ac7f8b2da97e5301b9d253496'}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_metadata(path):
    # Callers supply release/registry/queue/job/source/qualification/process metadata.
    # FREEZE, VALID history and tensor files are never passed to this helper.
    path = Path(path)
    require(path.name not in ('FREEZE.json','VALID_HISTORY.jsonl') and path.suffix == '.json',
            'Outcome-bearing FREEZE/history/tensor payload is not admitted by metadata reader')
    return json.loads(path.read_text(), parse_constant=reject_constant)


def reject_constant(value):
    raise ValueError('Nonfinite metadata constant: '+value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative(value):
    path = Path(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts, 'Require a phase-relative path')
    return path


def phase_file(value):
    path = PHASE/relative(value)
    require(path.resolve(strict=True).is_relative_to(PHASE.resolve()), 'Input leaves the project phase')
    for parent in [path, *path.parents]:
        if parent == PHASE.parent:
            break
        require(not parent.is_symlink(), 'Symlink custody is not admitted')
    require(path.is_file(), 'Bound artifact must be a file')
    return path


def binding(ref):
    require(isinstance(ref, dict) and isinstance(ref.get('sha256'), str)
            and len(ref['sha256']) == 64, 'Unresolved exact binding')
    path = phase_file(ref['path'])
    require(sha(path) == ref['sha256'], 'Bound bytes changed: '+ref['path'])
    return path


def reference(path):
    return {'path': str(Path(path).relative_to(PHASE)), 'sha256': sha(path)}


def verify_manifest(ref):
    manifest = binding(ref)
    rows = read_metadata(manifest)['files']
    require(rows and len({r['path'] for r in rows}) == len(rows), 'Complete unique source manifest required')
    for row in rows:
        path = phase_file(str((manifest.parent/relative(row['path'])).relative_to(PHASE)))
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source packet bytes changed')
    return manifest


def identity(owner, argv, cwd, expected_cwd):
    require(isinstance(owner, dict) and type(owner.get('PID')) is int and owner['PID'] > 0
            and owner.get('pgid') == owner['PID'] == owner.get('sid')
            and type(owner.get('start_ticks')) is int and owner['start_ticks'] > 0
            and owner.get('argv') == argv and cwd == expected_cwd,
            'Saved fresh process/session/argv/cwd identity differs')


def terminal(receipt):
    require(receipt.get('terminal_wait_observed') is True
            and receipt.get('exit_code_authority') == 'subprocess.Popen.wait/poll'
            and receipt.get('exit_code') == 0 and receipt.get('reason') is None
            and receipt.get('signals_sent') == [] and receipt.get('attempts') == 1
            and receipt.get('retry') is False and receipt.get('identity_admitted_for_signals') is True
            and receipt.get('scores_read') is False and receipt.get('signal_refusal') is None,
            'Actual successful owned Popen terminal authority required')


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Explicit UTC-offset metadata timestamp required')
    return parsed


def qualified_runtime(provider, job):
    ref = RUNTIME_EVIDENCE[provider]
    require(ref in job['runtime_qualification']['evidence'], 'Actual provider runtime evidence changed')
    metadata = read_metadata(binding(ref))
    require(metadata['status'] == 'PASS', 'Actual provider runtime qualification failed')
    if provider == 'authorized_one_GPU_allocation':
        versions = metadata['runtime_versions']
    else:
        versions = {name:metadata['providers'][name]['version'] for name in job['runtime_versions'] if name != 'CUDA'}
        versions['CUDA'] = metadata['torch_cuda']
    require(versions == job['runtime_versions'], 'Qualified runtime differs from the actual frozen job')
    return versions

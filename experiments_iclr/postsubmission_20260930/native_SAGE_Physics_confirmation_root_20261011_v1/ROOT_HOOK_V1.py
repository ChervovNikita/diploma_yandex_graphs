"""Root admission for the fixed complete Physics cohort; no numerical imports."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
H = Path(__file__).resolve().parent
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def receipt(path):
    p = Path(path).resolve()
    if not p.is_relative_to(R): raise ValueError('Receipt outside authorized repository')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def checked(item):
    if receipt(item['path']) != item: raise ValueError('Frozen receipt differs: ' + item['path'])
    return Path(item['path'])


def guard():
    if socket.gethostname() != 'anogena-2-0': raise ValueError('Excluded scientific route')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
    if uuids != [UUID] or Path.cwd() != R: raise ValueError('Exact sole GPU and project cwd required')


def bindings():
    guard()
    b = json.loads((H / 'FROZEN_BINDINGS_V1.json').read_text())
    if b['status'] != 'ROOT_COMPLETE_PHYSICS_COHORT_ADMITTED' or (b['banks'], b['optimizer_bodies']) != (60, 90):
        raise ValueError('Complete root cohort admission required')
    for row in b['bound_files']: checked(row)
    for row in b['qualification_receipts']:
        if json.loads(checked(row).read_text()).get('status') != 'PASS': raise ValueError('Closed execution qualification required')
    for row in b['qualification_owners']:
        owner = json.loads(checked(row).read_text())
        if owner['status'] != 'COMPLETE_EXIT_0' or owner['exit_code'] != 0 or not owner['child_pid_absent']:
            raise ValueError('Direct execution qualification owner closure required')
    return b


def custody(b, status):
    return dict(status=status, hostname=socket.gethostname(), gpu_uuid=UUID,
                frozen_bindings_sha256=receipt(H / 'FROZEN_BINDINGS_V1.json')['sha256'],
                protocol_sha256=b['protocol']['sha256'], banks=60, optimizer_bodies=90,
                paired_masks=10, CAL_CONFIRM_truth_access=False,
                no_retry_or_survivor_subset=True, overlap='Unchanged Penn94 on same authorized sole GPU; no isolated speed claim')


def validate_parent(spec, *, source_receipt):
    b = bindings()
    if (source_receipt != b['coupled_runner'] or spec['H'] != str(H)
            or spec['core_config'] != b['core_config'] or spec['author_config'] != b['author_config']
            or spec['root_bindings'] != receipt(H / 'FROZEN_BINDINGS_V1.json')
            or spec['root_hook'] != receipt(__file__) or spec['parent_wall_seconds'] != 172800
            or spec['core_port_directory'] != b['core_port_directory']
            or spec['e_directory'] != b['e_directory']
            or spec['author_context_directory'] != b['author_context_directory']
            or spec['core_release'] != b['core_release']):
        raise ValueError('Root integration differs from complete source/config freeze')
    return custody(b, 'ROOT_COUPLED_FIT_ADMITTED')


def validate_core(cfg, *, action, executing_source, port_receipt):
    b = bindings()
    if (action != 'fixed_family_fit' or cfg != json.loads(checked(b['core_config']).read_text())
            or Path(executing_source) != Path(b['e_directory']) / 'run_family.py'
            or port_receipt != b['expected_port_receipt']):
        raise ValueError('Preserved core recipe/source callback differs')
    return dict(stage='F', K=None, pilot_provenance_read=False,
                study_kind='prospective_complete_Physics_endpoint_shared_ensemble', action=action,
                source_sha256={v['path']: v['sha256'] for v in cfg['source_receipts'].values()},
                safe_role_sha256={k: v['sha256'] for k, v in cfg['safe_roles'].items()},
                root_runtime_admission=custody(b, 'ROOT_CONFIRMATION_FIT_ADMITTED'), baseline_policy='fresh_required')


def validate_author(cfg, *, action, source_receipts):
    b = bindings()
    if (action != 'fixed_Physics_author_meanSAGE_context_fits'
            or cfg != json.loads(checked(b['author_config']).read_text())
            or source_receipts != b['author_execution_sources']):
        raise ValueError('Preserved source-derived context recipe differs')
    return custody(b, 'ROOT_AUTHOR_CONTEXT_FITS_ADMITTED')

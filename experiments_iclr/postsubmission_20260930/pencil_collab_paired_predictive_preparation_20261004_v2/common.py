"""Stdlib admission and exact file custody for the disabled scientific PENCIL fits."""
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DATA_FILES = ('split/time/train.pt', 'split/time/valid.pt', 'raw/edge.csv.gz', 'raw/node-feat.csv.gz')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def pin(path, row):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Pinned file differs: ' + str(path))
    return path


def source_custody(plan):
    rows = []
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = HERE/row['path']
        require(path.resolve().is_relative_to(HERE), 'Pinned source escaped packet')
        pin(path, row)
        rows.append(dict(scope='source', path=str(path), bytes=row['bytes'], sha256=row['sha256']))
    return rows


def native_module_custody():
    for row in json.loads((HERE/'NATIVE_SOURCE_BINDINGS.json').read_text())['files']:
        relative = Path(row['path']).relative_to('native')
        parts = list(relative.with_suffix('').parts)
        if parts[-1] == '__init__': parts.pop()
        name = '.'.join(parts)
        module = sys.modules.get(name)
        require(module is not None and Path(module.__file__).resolve() == (HERE/row['path']).resolve(),
                'Native imported module differs: ' + name)


def control_receipt(row):
    path = Path(row['path'])
    require(path.resolve().is_relative_to(PHASE), 'Control receipt outside research root')
    return json.loads(pin(path, row).read_text())


def dependency_custody(plan, release):
    admission = control_receipt(release['dependency_admission'])
    require(admission.get('status') == 'ROOT_ADMITTED_FOR_NATIVE_PENCIL_PREDICTIVE_FITS'
            and admission.get('base_runtime_authority_sha256') == sha(HERE/'metadata/RUNTIME_AUTHORITY.json')
            and admission.get('distribution_versions') == plan['distribution_versions']
            and admission.get('installed_inventory_complete') is True
            and admission.get('installed_inventory_root') == plan['dependency_overlay']
            and admission.get('PYTHONPATH') == os.environ.get('PYTHONPATH'), 'Exact dependency admission required')
    # New package files are explicitly admitted after installation; no absence is silently repaired.
    rows = admission['added_package_source_pins']
    expected = plan['added_package_modules']
    require(len(rows) == len(expected) and {x['distribution'] for x in rows} == set(expected)
            and all(x['module'] == expected[x['distribution']] for x in rows), 'Exact newly resolved package closure must be pinned')
    for row in rows:
        path = Path(row['path'])
        require(path.resolve().is_relative_to(Path(plan['dependency_overlay'])), 'Added dependency outside owned overlay')
        pin(path, row)
    overlay = Path(plan['dependency_overlay'])
    installed = admission['installed_files']
    require(installed and len({row['path'] for row in installed}) == len(installed)
            and {Path(row['path']) for row in installed} == {x for x in overlay.rglob('*') if x.is_file()},
            'Complete observed owned-overlay file inventory required')
    for row in installed:
        path = Path(row['path'])
        require(path.resolve().is_relative_to(overlay) and not path.is_symlink(), 'Installed file escaped overlay')
        pin(path, row)
    for name, version in plan['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Distribution differs: ' + name)
    return admission


def gate(release_path, release_sha, seed):
    require('torch' not in sys.modules, 'Fresh stdlib admission must precede Torch import')
    plan = json.loads((HERE/'PLAN.json').read_text())
    runtime = json.loads((HERE/'metadata/RUNTIME_AUTHORITY.json').read_text())
    base = PHASE/plan['execution_directory']
    require(plan['seeds'] == [0,1,2] and seed in plan['seeds'], 'Exact paired scientific seed required')
    execution = base/('seed%d'%seed)
    require(sys.platform == 'linux' and os.uname().nodename == 'peptide'
            and Path.cwd() == Path(plan['repository']) and PHASE == Path(runtime['research_root']), 'Exact authorized host/root required')
    require(release_path.resolve() == base/'ROOT_RELEASE.json' and not release_path.is_symlink()
            and sha(release_path) == release_sha, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    require(release.get('status') == 'APPROVED' and release.get('root_authorization_reference')
            and release.get('source_manifest_sha256') == sha(HERE/'MANIFEST.json')
            and release.get('plan_sha256') == sha(HERE/'PLAN.json')
            and release.get('authorized_stage') == 'three_fresh_native_PENCIL_20_epoch_TRAIN_VALID_fits'
            and release.get('caps') == plan['caps'] and release.get('workload') == plan['workload']
            and release.get('allocator_policy') == plan['allocator_policy']
            and release.get('cuda_visible_devices') == plan['GPU_UUID']
            and release.get('TEST_reads') is False and release.get('scores_for_selection') is True
            and release.get('VALID_selection_only') is True and release.get('heldout_release') is False
            and release.get('initialization_source') == 'fresh_native_scratch_per_seed'
            and release.get('root_resource_adoption_reference')
            and release.get('state_donor') is False and release.get('automatic_retry') is False, 'Exact scientific TRAIN/VALID release required')
    source_custody(plan)
    evidence=json.loads((HERE/plan['resource_evidence']).read_text())
    require(evidence['adoption_eligible'] is True and evidence['supervisor_identity'] is None
            and evidence['physical_exit_code']==0 and evidence['physical_session_closed'] is True
            and evidence['resource']['source_manifest_sha256']=='211c140f7aa95e9af6395d841206160bb50aa8b20980b4b00d75a5a79a8291e6',
            'Exact completed resource evidence required')
    adoption=control_receipt(release['root_resource_adoption_reference'])
    require(adoption.get('status')=='ROOT_ADOPTED_COMPLETE_RESOURCE_ONLY'
            and adoption.get('source_manifest_sha256')==evidence['resource']['source_manifest_sha256']
            and adoption.get('predictive_result') is False and adoption.get('state_donation') is False,
            'Exact prior root resource adoption required')
    review = control_receipt(release['independent_source_review'])
    require(review.get('status') == 'PASS' and not review.get('blocking_findings')
            and review.get('candidate_manifest_sha256') == sha(HERE/'MANIFEST.json')
            and review.get('execution_authorized') is False, 'Independent exact source PASS required')
    require(Path(sys.executable).resolve() == Path(runtime['interpreter_path']).resolve()
            and sha(runtime['interpreter_path']) == runtime['interpreter_sha256'], 'Pinned RAPIDS Python required')
    dependency_custody(plan, release)
    for row in runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']]:
        path = Path(row['path'])
        require(path.is_file() and not path.is_symlink() and sha(path) == row['sha256']
                and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Base runtime file differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == plan['GPU_UUID']
            and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2'
            and os.environ.get('WANDB_MODE') == 'disabled'
            and os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'Fixed preimport resource environment required')
    metric_pin=json.loads((HERE/'metadata/DATA_AUTHORITY.json').read_text())['ogb_evaluator']
    pin(Path(metric_pin['path']), metric_pin)
    return plan, runtime, execution


def final_file_custody(plan, runtime, release_path, release_sha, data):
    require(sha(release_path) == release_sha, 'Root release changed')
    release = json.loads(release_path.read_text())
    rows = source_custody(plan)
    require(release['source_manifest_sha256'] == sha(HERE/'MANIFEST.json'), 'Manifest changed')
    for path, digest, scope in ((HERE/'MANIFEST.json', release['source_manifest_sha256'], 'source_manifest'),
                                (release_path, release_sha, 'root_release'),
                                (Path(runtime['interpreter_path']), runtime['interpreter_sha256'], 'interpreter')):
        require(sha(path) == digest, 'Final control/interpreter differs')
        rows.append(dict(scope=scope, path=str(path), bytes=path.stat().st_size, sha256=digest))
    for key in ('independent_source_review', 'dependency_admission', 'root_resource_adoption_reference'):
        row = release[key]; control_receipt(row); rows.append(dict(scope='control', **row))
    admission = dependency_custody(plan, release)
    for row in runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']] + admission['installed_files']:
        path = Path(row['path'])
        require(path.is_file() and not path.is_symlink() and sha(path) == row['sha256']
                and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Final runtime file differs')
        rows.append(dict(scope='runtime_file', path=str(path), bytes=path.stat().st_size, sha256=row['sha256']))
    authority = json.loads((HERE/'metadata/DATA_AUTHORITY.json').read_text())
    metric_pin=authority['ogb_evaluator'];metric_path=pin(Path(metric_pin['path']),metric_pin)
    rows.append(dict(scope='runtime_metric_file',path=str(metric_path),bytes=metric_pin['bytes'],sha256=metric_pin['sha256']))
    require(data == Path(authority['dataset_root']).resolve() and data.is_relative_to(PHASE), 'Data root differs')
    opened = []
    for name in DATA_FILES:
        path = data/name; row = authority['files'][name]
        require(path.resolve().is_relative_to(data), 'Data file escaped authority'); pin(path, row)
        opened.append(dict(relative_path=name, path=str(path), **row))
    return dict(schema='pencil-predictive-final-file-custody-v1', status='MATCH',
                source_manifest_sha256=release['source_manifest_sha256'], release_sha256=release_sha,
                files=rows, data_files=opened, GPU_operations=False, array_loads=False)

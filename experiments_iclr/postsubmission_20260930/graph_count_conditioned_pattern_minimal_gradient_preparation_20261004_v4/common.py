"""Stdlib gates/custody for exactly one disabled native gradient diagnostic."""
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DATA_FILES = ('split/time/train.pt', 'raw/edge.csv.gz', 'raw/node-feat.csv.gz')


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


def pinned_json(row):
    path = Path(row['path'])
    require(path.resolve().is_relative_to(PHASE), 'External receipt outside project')
    return json.loads(pin(path, row).read_text())


def source_custody(plan):
    rows = []
    for base, pins, scope in ((HERE, json.loads((HERE/'MANIFEST.json').read_text())['files'], 'source'),
                              (PHASE, plan['source_pins'], 'dependency')):
        for row in pins:
            path = base/row['path']
            require(path.resolve().is_relative_to(base), 'Pinned path escaped project')
            pin(path, row)
            rows.append(dict(scope=scope, path=str(path), bytes=row['bytes'], sha256=row['sha256']))
    return rows


def prerequisite_custody(plan, release):
    core_sha = plan['core_manifest_sha256']
    require(all(release[key] == row for key, row in plan['CPU_prerequisites'].items()),
            'Actual adopted core-v3 CPU receipt pins differ')
    review = pinned_json(release['independent_source_review'])
    require(review.get('status') == 'PASS' and not review.get('blocking_findings')
            and review.get('candidate_manifest_sha256') == sha(HERE/'MANIFEST.json')
            and review.get('execution_authorized') is False, 'Independent exact diagnostic source PASS required')
    q = pinned_json(release['CPU_QA'])
    terminal = pinned_json(release['CPU_supervisor_terminal'])
    custody = pinned_json(release['CPU_final_custody'])
    require(q.get('schema') == 'graph-count-conditioned-pattern-fabricated-cpu-qualification-v1'
            and q.get('status') == 'PASS' and q.get('source_manifest_sha256') == core_sha
            and all(q.get(k) is False for k in ('GPU_data_access', 'scientific_fit_admitted',
                    'TEST_supported', 'native_full_batch_resource_qualification'))
            and {'ragged_successor', 'vectorized_single_successor'} <= set(q['result']),
            'Successful exact core-v3 fabricated CPU QA required')
    require(terminal.get('overall_status') == 'PASS_FABRICATED_CPU_ONLY'
            and terminal.get('candidate_manifest_sha256') == core_sha
            and terminal.get('child_exit_code') == 0 and terminal.get('stop_reason') is None
            and terminal.get('physical_status') == 'REAPED_OWNED_SESSION_EMPTY'
            and terminal.get('final_source_closure_matches') is True
            and terminal.get('final_runtime_closure_matches') is True
            and terminal['collected_oracle_result']['status'] == 'COLLECTED_PASS'
            and terminal['collected_oracle_result']['qualification']['sha256'] == release['CPU_QA']['sha256']
            and terminal['collected_oracle_result']['child_final_custody']['sha256'] == release['CPU_final_custody']['sha256']
            and custody.get('stage') == 'fabricated_cpu' and custody.get('completed') is True
            and any(row['path'] == 'QUALIFICATION.json' and row['sha256'] == release['CPU_QA']['sha256']
                    and row['bytes'] == release['CPU_QA']['bytes'] for row in custody['files']),
            'Exact core-v3 physical CPU completion/custody required')
    for key in ('native_numerical', 'native_full_graph'):
        value = json.loads((PHASE/plan[key]).read_text())
        require(value.get('status') == 'PASS'
                and value['identity']['driver_manifest_sha256'] == plan['native_driver_manifest_sha256']
                and value['identity']['prototype_manifest_sha256'] == plan['prototype_manifest_sha256']
                and value['identity']['runtime_profile_id'] == plan['runtime_profile_id']
                and value['runtime_profile_transition']['status'] == 'TRANSITION_VERIFIED'
                and value['runtime_profile_transition']['RNG_exactly_unchanged'] is True,
                'Qualified exact native integration/profile receipt required')
    for key in ('native_numerical_terminal', 'native_full_graph_terminal'):
        value = json.loads((PHASE/plan[key]).read_text())
        require(value.get('status') == 'COMPLETE' and value.get('child_exit_code') == 0,
                'Qualified native physical terminal required')
    adoption = json.loads((PHASE/plan['CPU_root_adoption']).read_text())
    require(adoption.get('status') == 'ADOPTED_COMPLETE_CORE_V3_FABRICATED_CPU_LAW_AND_GRADIENT_QA_ONLY'
            and adoption['qualification']['source_manifest_sha256'] == core_sha
            and adoption['qualification']['qualification_descriptor']['sha256'] == release['CPU_QA']['sha256'],
            'Exact root adoption of CPU QA required')


def gate(release_path, release_sha):
    require('torch' not in sys.modules, 'Fresh stdlib admission must precede Torch')
    plan = json.loads((HERE/'PLAN.json').read_text())
    execution = PHASE/plan['execution_directory']
    require(sys.platform == 'linux' and Path.cwd() == Path(plan['repository'])
            and os.uname().nodename == 'peptide', 'Exact ordinary project host/root required')
    require(release_path.resolve() == execution/'ROOT_RELEASE.json' and not release_path.is_symlink()
            and sha(release_path) == release_sha, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    require(release.get('status') == 'APPROVED' and release.get('root_authorization_reference')
            and release.get('source_manifest_sha256') == sha(HERE/'MANIFEST.json')
            and release.get('plan_sha256') == sha(HERE/'PLAN.json')
            and release.get('authorized_stage') == 'one_native_TRAIN_gradient_diagnostic'
            and release.get('caps') == plan['caps'] and release.get('workload') == plan['workload']
            and release.get('runtime_profile_id') == plan['runtime_profile_id']
            and release.get('cuda_visible_devices') == plan['GPU_UUID']
            and release.get('fits') == 0 and release.get('optimizer_updates') == 0
            and release.get('VALID_TEST_reads') is False and release.get('automatic_retry') is False,
            'Exact bounded zero-update source release required')
    source_custody(plan)
    prerequisite_custody(plan, release)
    runtime = json.loads((PHASE/plan['runtime_authority']).read_text())
    require(PHASE == Path(runtime['research_root'])
            and Path(sys.executable).resolve() == Path(runtime['interpreter_path']).resolve()
            and sha(runtime['interpreter_path']) == runtime['interpreter_sha256'], 'Exact runtime interpreter required')
    for name, version in runtime['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Distribution version differs: ' + name)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == plan['GPU_UUID']
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1'
            and os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '2'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Fixed preimport runtime environment required')
    return plan, runtime, execution


def load_module(name, path):
    path = Path(path)
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path.resolve(), 'Shadowed module: ' + name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def module_custody(plan):
    for row in plan['imported_modules']:
        value = sys.modules.get(row['module'])
        require(value is not None and Path(value.__file__).resolve() == PHASE/row['path'],
                'Actual import closure differs: ' + row['module'])


def final_file_custody(plan, runtime, release_path, release_sha, data):
    require(sha(release_path) == release_sha, 'Root release changed')
    release = json.loads(release_path.read_text())
    require(release['source_manifest_sha256'] == sha(HERE/'MANIFEST.json'), 'Source manifest changed')
    rows = source_custody(plan)
    prerequisite_custody(plan, release)
    for path, digest, scope in ((HERE/'MANIFEST.json', release['source_manifest_sha256'], 'source_manifest'),
                                (release_path, release_sha, 'root_release'),
                                (Path(runtime['interpreter_path']), runtime['interpreter_sha256'], 'interpreter')):
        require(sha(path) == digest, 'Final control/interpreter differs')
        rows.append(dict(scope=scope, path=str(path), bytes=path.stat().st_size, sha256=digest))
    for key in ('independent_source_review', 'CPU_QA', 'CPU_supervisor_terminal', 'CPU_final_custody'):
        row = release[key]; pinned_json(row)
        rows.append(dict(scope='prerequisite', **row))
    for name, version in runtime['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Final distribution differs')
    for row in runtime['runtime_source_pins'] + runtime['runtime_binary_files'] + [runtime['negative_sampler']]:
        path = Path(row['path'])
        require(path.is_file() and not path.is_symlink() and sha(path) == row['sha256']
                and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Final runtime source/binary differs')
        rows.append(dict(scope='runtime_file', path=str(path), bytes=path.stat().st_size, sha256=row['sha256']))
    train = []
    if data is not None:
        authority = json.loads((PHASE/plan['data_authority']).read_text())
        require(data == Path(authority['dataset_root']).resolve() and data.is_relative_to(PHASE), 'TRAIN root differs')
        for name in DATA_FILES:
            path = data/name; row = authority['files'][name]
            require(path.resolve().is_relative_to(data), 'TRAIN file escaped project')
            pin(path, row)
            train.append(dict(relative_path=name, path=str(path), **row))
    return dict(schema='native-gradient-final-file-custody-v1', status='MATCH',
                source_manifest_sha256=release['source_manifest_sha256'], release_sha256=release_sha,
                files=rows, TRAIN_files=train, GPU_operations=False, array_loads=False)

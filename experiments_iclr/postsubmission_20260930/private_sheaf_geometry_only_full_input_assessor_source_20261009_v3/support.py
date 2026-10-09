"""Separate inactive assessor gates and exact V2 scientific/helper custody."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def exact_files(root, rows):
    for row in rows:
        path = (root/row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact pinned bytes: '+row['path'])


def source_checks():
    seal = read(HERE/'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False and sha(HERE/'MANIFEST.json') == seal['manifest_sha256'], 'Exact inactive assessor seal')
    exact_files(HERE, read(HERE/'MANIFEST.json')['files'])
    pins = read(HERE/'SOURCE_BINDINGS.json')
    exact_files(PHASE, pins['files'])
    return pins


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_core(pins):
    """Load unchanged V2 stdlib/local modules; no numerical provider is imported."""
    root = PHASE/pins['scientific_core_directory']
    support = load(root/'support.py', '_geometry_assessor_v2_support')
    support.source_checks()
    absent = object()
    previous = {name:sys.modules.get(name, absent) for name in ('support','bank','shared_fit')}
    try:
        sys.modules['support'] = support
        bank = load(root/'bank.py', '_geometry_assessor_v2_bank'); sys.modules['bank'] = bank
        fit = load(root/'shared_fit.py', '_geometry_assessor_v2_shared_fit'); sys.modules['shared_fit'] = fit
        qualifier = load(root/'qualifier.py', '_geometry_assessor_v2_qualification_checks')
    finally:
        for name, value in previous.items():
            if value is absent: sys.modules.pop(name, None)
            else: sys.modules[name] = value
    return support, bank, fit, qualifier


def admission(args, pins, core):
    release = read(args.release)
    require(release['enabled'] is True and release['release_owner'] == 'root'
            and release['action'] == 'geometry_engineering_qualify'
            and release['assessment_method'] == 'memory_bounded_sequential_M1_differential_v3'
            and release['assessor_source_seal_sha256'] == sha(HERE/'SEAL.json')
            and release['reference_source_sha256'] == sha(HERE/'reference.py'), 'Explicit new assessor/reference release')
    require(release['new_post_failure_resource_plan'] is True and release['failed_V2_joint_attempt_preserved'] is True
            and release['automatic_retry'] is False and release['repeat_failed_joint_method'] is False
            and release['one_live_autograd_graph'] is True and release['reference_bodies_live_at_once'] == 1
            and release['qualifier_seed'] == 7409 and release['owned_GPU_cap_bytes'] == 25769803776
            and release['owned_RSS_cap_bytes'] == 17179869184,
            'New prospective bounded assessment, preserved failed method/cost, unchanged 24GiB custody')
    require(release['prior_failure_receipt_sha256'] == pins['prior_failure']['sha256']
            == sha(PHASE/pins['prior_failure']['path']), 'Exact original resource failure evidence')
    failed = read(PHASE/pins['prior_failure']['path'])
    terminal = failed['metadata']['TERMINAL.json']
    require(terminal['complete'] is False and terminal['error']['message'] == 'Owned GPU cap exceeded'
            and terminal['exit_code'] == -15 and terminal['actual_worker_absent'] is True
            and terminal['actual_worker_CUDA_absent'] is True and terminal['reaped'] is True
            and terminal['automatic_retry'] is False and terminal['partial_files_and_costs_retained'] is True
            and failed['qualification_progress']['qualification_passed'] is False, 'Prior attempt remains failed with partial proof/cost')
    row = release['prior_partial_qualification']
    require(sha(row['path']) == row['sha256'], 'Exact retained server-only partial V2 qualification')
    partial = read(row['path'])
    counts = partial['counters']
    require(partial['status'] == 'started' and partial['qualification_passed'] is False
            and partial['identity']['source_seal_sha256'] == pins['scientific_core_seal_sha256']
            and counts['train_forwards_completed'] == counts['backwards_completed'] == counts['original_serving_forwards_completed'] == 4
            and counts['Adam_steps_completed'] == 1 and counts['restored_serving_forwards_completed'] == 0
            and len(partial['per_backward_gradient_observations']) == 4,
            'Preserved actual partial ownership/gradient/cache proof, never relabelled as complete')
    v2_pins = core.source_checks()
    result = core.admission(args, v2_pins, read(PHASE/pins['scientific_core_directory']/'PROTOCOL.json'))
    output = result[3]
    forbidden = [HERE,Path(row['path']).resolve().parent]+[PHASE/path for path in pins['protected_directories']]
    require(not any(output.is_relative_to(path) for path in forbidden), 'Fresh V3 output outside assessor/failure artifacts')
    return result

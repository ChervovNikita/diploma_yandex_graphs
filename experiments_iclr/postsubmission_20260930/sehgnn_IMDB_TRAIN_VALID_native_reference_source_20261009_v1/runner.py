"""Inactive literal native IMDB TRAIN/VALID qualifier and five-seed reference."""
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PROVIDERS = ('torch', 'numpy', 'dgl', 'torch_sparse', 'sklearn')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def bound_path(row):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}
            and isinstance(row['path'], str) and Path(row['path']).is_absolute()
            and type(row['bytes']) is int and row['bytes'] > 0
            and isinstance(row['sha256'], str) and len(row['sha256']) == 64, 'Complete root file identity')
    path = Path(row['path'])
    require(not path.is_symlink() and binding(path) == row, 'Exact reviewed root file identity')
    return path


def load_module(name, path):
    """Private module registration precedes execution, including dataclasses."""
    require(name not in sys.modules, 'Fresh owned module name')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Inactive immutable source packet')
    manifest = read(HERE / 'MANIFEST.json')
    for row in manifest['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Unchanged sealed reference source')
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'Unchanged pinned author/seam/helper provenance')
    require(sha(HERE / 'model.py') == sources['original_model_sha256'], 'Exact author model bytes')
    return sources, read(HERE / 'PROTOCOL.json')


def release_gate(args, sources, protocol):
    release = read(args.release)
    action = 'qualify_one_full_native_TRAIN_update' if args.qualify else 'fit_literal_five_seed_TRAIN_VALID_reference'
    seeds = [1] if args.qualify else [1, 2, 3, 4, 5]
    require(release['enabled'] is True and release['action'] == action and release['seeds'] == seeds
            and release['root_source_review_approved'] is True
            and release['development_input_scope_authorized'] is True
            and release['source_seal_sha256'] == sha(HERE / 'SEAL.json')
            and release['protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
            and release['seam_seal_sha256'] == sources['seam_seal_file_sha256'], 'Explicit reviewed root native release')
    require(str(args.input_root.resolve()) == release['input_root']
            and str(args.output.resolve()) == release['output_directory'] and not args.output.exists()
            and str(Path(sys.executable).absolute()) == release['python_executable']
            and platform.python_version() == release['python_version'], 'Fresh exact inputs/output/interpreter')
    require(isinstance(release['expected_runtime_versions'], dict)
            and set(release['expected_runtime_versions']) == set(PROVIDERS)
            and all(isinstance(v, str) and v for v in release['expected_runtime_versions'].values()), 'Every numerical provider prequalified; no guessed or missing version')
    require(release['device'] == 'cuda:0' and isinstance(release['cuda_device_uuid'], str)
            and release['cuda_device_uuid'].startswith('GPU-')
            and release['runtime_environment']['CUDA_VISIBLE_DEVICES'] == release['cuda_device_uuid']
            and set(release['runtime_environment']) == {'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'}
            and all(isinstance(v, str) for v in release['runtime_environment'].values()), 'Root-owned single-device runtime environment')
    require(all(os.environ.get(key) == value for key, value in release['runtime_environment'].items()), 'Exact environment adopted before provider imports')
    require(release['resource_budget']['host_RSS_bytes'] >= 32 * 1024**3
            and release['resource_budget']['device_bytes'] >= 24 * 1024**3
            and release['resource_budget']['root_owns_external_resource_monitor'] is True, 'Full literal resource allowance with root monitoring')
    require(isinstance(release['roles'], dict) and set(release['roles']) == {str(seed) for seed in range(1, 11)}, 'All ten already-frozen schema roles, source fits use literal subset')
    require(all(Path(row['path']).name == 'seed' + seed + '.json' for seed, row in release['roles'].items())
            and Path(release['schema_receipt']['path']).name == 'SCHEMA_REPORT.json', 'Only named JSON role/schema metadata descriptors')
    paths = {int(seed): bound_path(row) for seed, row in release['roles'].items()}
    schema_path = bound_path(release['schema_receipt'])
    require(sha(schema_path) == read(HERE / 'ROOT_REPORTED_SCHEMA_BINDING.json')['schema_report_sha256'], 'Adopted actual schema identity reported by root')
    schema = read(schema_path)
    require(schema['status'] == 'complete' and schema['schema_qualified'] is True
            and schema['all_ten_roles_qualified'] is True
            and schema['TEST_member_open_stat_hash_or_parse'] is False
            and schema['TEST_membership_known'] is False
            and schema['source_seal_sha256'] == sources['seam_seal_file_sha256'], 'Successful full schema and role adoption without TEST access')
    require(set(release['expected_input_files']) == {'node.dat', 'link.dat', 'label.dat'}
            and schema['actual_input_schema']['input_files'] == release['expected_input_files'], 'Only explicit development/schema input identities')
    require(len(schema['roles']) == 10 and {row['seed'] for row in schema['roles']} == set(range(1, 11)), 'Ten schema-qualified role records')
    for row in schema['roles']:
        require(row['sha256'] == release['roles'][str(row['seed'])]['sha256']
                and row['bytes'] == release['roles'][str(row['seed'])]['bytes'], 'Frozen role bytes adopted from schema receipt')
    roles = {seed: read(path) for seed, path in paths.items()}
    require(all(role['seed'] == seed and role['input_files'] == release['expected_input_files'] for seed, role in roles.items()), 'Root role/input identities')
    if not args.qualify:
        require(Path(release['native_qualification_receipt']['path']).name == 'COHORT_REPORT.json', 'Named native qualification JSON receipt only')
        qualification_path = bound_path(release['native_qualification_receipt'])
        qualified = read(qualification_path)
        require(qualified['status'] == 'complete' and qualified['complete'] is True
                and qualified['native_backbone_numerically_qualified'] is True
                and qualified['mode'] == 'qualification' and qualified['seeds'] == [1]
                and qualified['source_seal_sha256'] == release['source_seal_sha256']
                and qualified['schema_receipt']['sha256'] == release['schema_receipt']['sha256']
                and qualified['input_files'] == release['expected_input_files']
                and qualified['runtime']['versions'] == release['expected_runtime_versions']
                and qualified['runtime']['math_flags'] == release['expected_math_flags']
                and qualified['runtime']['cuda_device_uuid'] == release['cuda_device_uuid']
                and qualified['runtime']['python_executable'] == release['python_executable']
                and qualified['runtime']['environment'] == release['runtime_environment']
                and qualified['roles'] == release['roles'], 'Matching successful full-input native qualification before fits')
        child = qualified['fits']
        require(len(child) == 1 and child[0]['qualification_passed'] is True and child[0]['complete'] is True
                and child[0]['parameters']['total'] == protocol['predicted_native_model_parameters']
                and child[0]['counters']['actual_Adam_steps'] == child[0]['counters']['TRAIN_member_forwards']
                == child[0]['counters']['TRAIN_backwards'] == child[0]['counters']['epochs_completed'] == 1, 'Exactly one measured literal qualifying update')
    return release, roles


def runtime(release):
    """Numerical imports happen only after complete source/release/adoption gates."""
    providers = {name: importlib.import_module(name) for name in PROVIDERS}
    versions = {name: value.__version__ for name, value in providers.items()}
    require(versions == release['expected_runtime_versions'], 'Exact prequalified numerical provider versions')
    torch = providers['torch']
    require(torch.get_default_dtype() == torch.float32 and torch.cuda.is_available()
            and torch.cuda.device_count() == 1, 'Native FP32 default and one visible CUDA device')
    torch.cuda.init()
    device = torch.device(release['device'])
    torch.cuda.set_device(device)  # Required before Torch2.1 peak reset.
    torch.cuda.synchronize(device)
    prop = torch.cuda.get_device_properties(device)
    require(prop.total_memory >= release['resource_budget']['device_bytes'], 'Complete model device budget')
    flags = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
             'cudnn_deterministic': torch.backends.cudnn.deterministic,
             'cudnn_benchmark': torch.backends.cudnn.benchmark,
             'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
             'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
             'default_dtype': str(torch.get_default_dtype())}
    require(flags == release['expected_math_flags'], 'Native math flags adopted without modification')
    model_module = load_module('_owned_sehgnn_IMDB_model_v1', HERE / 'model.py')
    native = load_module('_owned_sehgnn_IMDB_helpers_v1', HERE / 'native_helpers.py')
    state = load_module('state_helpers', HERE / 'state_helpers.py')
    engine = load_module('_owned_sehgnn_IMDB_engine_v1', HERE / 'engine.py')
    record = dict(versions=versions, providers={name: binding(module.__file__) for name, module in providers.items()},
                  python_version=platform.python_version(), python_executable=str(Path(sys.executable).absolute()),
                  python_binary=binding(Path(sys.executable).resolve()), python_prefix=str(Path(sys.prefix).resolve()),
                  environment={key: os.environ.get(key) for key in release['runtime_environment']}, math_flags=flags,
                  cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                  cuda_device_uuid=release['cuda_device_uuid'], device=str(device), device_name=prop.name,
                  device_total_bytes=prop.total_memory, UUID_custody='one visible device; CUDA_VISIBLE_DEVICES exact root GPU UUID',
                  model_module=binding(model_module.__file__), native_module=binding(native.__file__), state_module=binding(state.__file__))
    rt = dict(providers, device=device, model_module=model_module, model_class=model_module.SeHGNN,
              native=native, remove_diag=providers['torch_sparse'].remove_diag, SparseTensor=providers['torch_sparse'].SparseTensor,
              protocol=read(HERE / 'PROTOCOL.json'))
    return rt, engine, record


def execute(args):
    sources, protocol = source_gate()
    release, roles = release_gate(args, sources, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    report = dict(status='started', complete=False, mode='qualification' if args.qualify else 'native_reference',
                  seeds=release['seeds'], source_seal_sha256=release['source_seal_sha256'],
                  protocol_sha256=release['protocol_sha256'], release=binding(args.release), schema_receipt=release['schema_receipt'],
                  roles=release['roles'], input_files=release['expected_input_files'], native_backbone_numerically_qualified=False,
                  TEST_truth=False, TEST_membership_known=False, TEST_file_access=False, fits=[])
    report_path = args.output / 'COHORT_REPORT.json'
    write(report_path, report)
    rt = engine = static = data = scalar = None
    try:
        rt, engine, report['runtime'] = runtime(release)
        if not args.qualify:
            qualified = read(Path(release['native_qualification_receipt']['path']))
            require(report['runtime'] == qualified['runtime'], 'Complete provider/module/interpreter/device runtime matches successful native qualification')
        torch = rt['torch']
        torch.cuda.reset_peak_memory_stats(rt['device'])
        costs = engine.Costs(args.output, torch, rt['device'])
        seam = HERE.parent / sources['seam_directory']
        loader = load_module('_owned_IMDB_role_loader_v1', seam / 'role_loader.py')
        loader.source_gate()
        with costs.measure('once_only_exact_frozen_development_inputs'):
            data = loader.load(args.input_root, Path(release['roles']['1']['path']), read(seam / 'SOURCE_EXPECTATIONS.json'))
            require(data.input_bindings == release['expected_input_files'], 'Exact once-loaded input hashes')
            for role in roles.values():
                data.with_roles(role)
        static = engine.prepare_static(rt, data, costs)
        report['feature_shapes'] = static.feature_shapes
        report['setup_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(rt['device'])
        report['setup_peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(rt['device'])
        # Literal author lifecycle: one master before source seed loop. It is not reset per seed.
        scalar = torch.cuda.amp.GradScaler()
        identity = dict(source_seal_sha256=release['source_seal_sha256'], protocol_sha256=release['protocol_sha256'],
                        schema_receipt_sha256=release['schema_receipt']['sha256'], input_files=release['expected_input_files'],
                        role_sha256s={seed: row['sha256'] for seed, row in release['roles'].items()},
                        release_sha256=sha(args.release), runtime=report['runtime'])
        for seed in release['seeds']:
            folder = args.output / ('seed' + str(seed))
            try:
                engine.run_one(rt, static, roles[seed], seed, folder, costs, scalar, identity, qualify=args.qualify)
            finally:
                if (folder / 'RESULT.json').exists():
                    report['fits'].append(read(folder / 'RESULT.json'))
                    write(report_path, report)
            # Read final cleanup records; a returned pre-finally state is not a success certificate.
            child = read(folder / 'RESULT.json')
            completion = read(folder / 'COMPLETE.json')
            require(child['status'] == 'complete' and child['complete'] is True
                    and completion['complete'] is True and (not args.qualify or child['qualification_passed'] is True), 'Complete fit, replay and cleanup')
        require(loader.bindings(loader.permitted_files(args.input_root)) == release['expected_input_files'], 'Inputs unchanged through full cohort')
        require(all(binding(Path(row['path'])) == row for row in release['roles'].values())
                and binding(Path(release['schema_receipt']['path'])) == release['schema_receipt']
                and binding(args.release) == report['release'], 'Adopted roles/schema/release unchanged through computation')
        report.update(status='complete', complete=True, native_backbone_numerically_qualified=True,
                      native_five_seed_recipe_completed=not args.qualify, master_AMP_final_state=scalar.state_dict(),
                      schema_qualification_is_distinct_from_numerical_backbone_qualification=True,
                      source_MODEL_and_helpers_exact=True, all_full_input_costs_charged=True)
    except BaseException as error:
        report.update(status='failed', complete=False, native_backbone_numerically_qualified=False,
                      failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if rt is not None:
            try:
                rt['torch'].cuda.synchronize(rt['device'])
                report['last_fit_peak_cuda_allocated_bytes'] = rt['torch'].cuda.max_memory_allocated(rt['device'])
                report['last_fit_peak_cuda_reserved_bytes'] = rt['torch'].cuda.max_memory_reserved(rt['device'])
                static = data = scalar = None
                gc.collect()
                rt['torch'].cuda.empty_cache()
            except BaseException as error:
                report.update(status='failed', complete=False, native_backbone_numerically_qualified=False,
                              cleanup_failure={'type': type(error).__name__, 'message': str(error)})
        end = resource.getrusage(resource.RUSAGE_SELF)
        if 'setup_peak_cuda_allocated_bytes' in report:
            report['cohort_peak_cuda_allocated_bytes'] = max([report['setup_peak_cuda_allocated_bytes']]
                + [fit.get('peak_cuda_allocated_bytes', 0) for fit in report['fits']])
            report['cohort_peak_cuda_reserved_bytes'] = max([report['setup_peak_cuda_reserved_bytes']]
                + [fit.get('peak_cuda_reserved_bytes', 0) for fit in report['fits']])
        report.update(seconds=time.perf_counter()-started, CPU_user_seconds=end.ru_utime-usage.ru_utime,
                      CPU_system_seconds=end.ru_stime-usage.ru_stime,
                      cumulative_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))
        write(report_path, report)
        write(args.output / 'COMPLETE.json', dict(status=report['status'], complete=report['complete'],
                                                native_backbone_numerically_qualified=report['native_backbone_numerically_qualified'],
                                                TEST_truth=False, TEST_membership_known=False))
        if report['status'] == 'failed' and 'failure' not in report:
            raise RuntimeError('Cohort cleanup failed; final receipts preserve the failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--qualify', action='store_true')
    modes.add_argument('--reference', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.qualify and not args.reference:
        print(json.dumps({'inactive': True, 'data_or_numerical_provider_or_model_import': False}))
        return
    require(args.release and args.input_root and args.output, 'Explicit root release and bound input/output paths')
    execute(args)


if __name__ == '__main__':
    main()

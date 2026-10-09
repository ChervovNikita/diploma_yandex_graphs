"""Disabled one-shot caller for the reviewed V2 complete-input P/S witness."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
V2_NAME = 'joint_source_additive_sehgnn_training_source_20261009_v2'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def bound(row):
    require(set(row) == {'path', 'bytes', 'sha256'}, 'Exact root file binding')
    path = Path(row['path'])
    require(path.is_absolute() and not path.is_symlink() and path.is_relative_to(PHASE)
            and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact phase-local root file bytes/hash')
    return path


def load(name, path):
    require(name not in sys.modules, 'Fresh exact module namespace')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact disabled qualifier seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Qualifier source unchanged')
    for row in read(HERE / 'SOURCE_BINDINGS.json')['files']:
        path = (PHASE / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Existing bootstrap/dependency unchanged')


def reviewed_source(cfg, old):
    row = cfg['reviewed_joint_source']
    directory = Path(row['directory'])
    require(directory.is_absolute() and directory.name == V2_NAME and directory.parent == PHASE
            and not directory.is_symlink(), 'Only the eventual separate V2, never blocked immutable V1')
    manifest, seal, adoption_path = (bound(row[key]) for key in ('manifest', 'seal', 'root_adoption'))
    require(manifest == directory / 'MANIFEST.json' and seal == directory / 'SEAL.json', 'Same exact V2 packet')
    require(row['seal']['sha256'] != old['blocked_V1_seal_sha256'], 'Blocked V1 cannot qualify')
    declaration = read(adoption_path)
    require(declaration['schema'] == 'root-reviewed-joint-source-v2-qualification-adoption-v1'
            and declaration['root_source_review_approved'] is True and declaration['reviewed_revision'] == 'v2'
            and declaration['source_manifest_sha256'] == row['manifest']['sha256']
            and declaration['source_seal_sha256'] == row['seal']['sha256']
            and declaration['unresolved_construction_or_role_blockers'] == []
            and declaration['qualification_only'] is True and declaration['scientific_training_adopted'] is False,
            'Actual root review adoption bound to exact V2; no assumed source approval')
    review_path = bound(declaration['independent_review'])
    require(review_path.suffix == '.json' and read(review_path), 'Exact saved focused independent review metadata')
    packet_seal = read(seal)
    require(packet_seal['runtime_disabled'] is True and packet_seal['manifest_sha256'] == row['manifest']['sha256'], 'Exact V2 disabled seal')
    for item in read(manifest)['files']:
        path = (directory / item['path']).resolve(strict=True)
        require(path.is_relative_to(directory) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Reviewed V2 payload bytes unchanged')
    sources = read(directory / 'SOURCE_BINDINGS.json')
    require(sources['modules'] == old['original_module_bindings'], 'No alternate native/loader/view/provider dependencies')
    return directory, declaration, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True, numerical_imports=False, qualification_execution=False)))
        return
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    source_gate()
    require(args.release is not None, 'Separate root qualification release')
    cfg, old = read(args.release), read(HERE / 'EXISTING_NATIVE_BINDINGS.json')
    require(cfg['schema'] == 'disabled-joint-additive-full-native-qualification-release-v1'
            and cfg['enabled'] is True and cfg['root_qualification_execution_approved'] is True
            and cfg['root_owns_existing_runtime_allocation_and_external_resource_monitor'] is True,
            'Root owns exact once-only execution/runtime/resources')
    require(cfg['action'] == 'qualify_reviewed_joint_additive_full_input_once'
            and cfg['root_entry_sha256'] == sha(__file__)
            and cfg['qualifier_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and cfg['plan_sha256'] == sha(HERE / 'PLAN.json'), 'Exact source and bounded plan')
    require(all(cfg[key] is False for key in ('quality_scoring', 'VALID_checkpoint_selection', 'scientific_training', 'automatic_retry', 'TEST_file_access')),
            'No fitting, quality selection or retry')
    require(set(cfg['resource_budget']) == {'seconds', 'host_RSS_bytes', 'device_bytes'}
            and all(type(value) is int and value > 0 for value in cfg['resource_budget'].values()), 'Exact meaningful root resource caps')
    directory, adoption, sources = reviewed_source(cfg, old)
    repo = PHASE.parents[1]
    require(Path.cwd() == repo and socket.gethostname() == 'anogena-2-0', 'Ordinary authorized repository/host')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU], 'Sole authorized GPU')
    entry, output = Path(cfg['entry_output_directory']), Path(cfg['qualification_output_directory'])
    protected = {HERE, directory, PHASE / 'joint_source_additive_sehgnn_training_source_20261009_v1',
        Path(old['input_root']), Path(old['role1']['path']).parent,
        *(Path(old[key]['path']).parent for key in ('qualified_native_runtime_release', 'native_qualification_receipt', 'native_five_reference_receipt')),
        *((PHASE / row['path']).parent for row in read(HERE / 'SOURCE_BINDINGS.json')['files'])}
    require(entry.is_absolute() and output.is_absolute() and entry.resolve() == entry and output.resolve() == output
            and entry.is_relative_to(PHASE) and output.is_relative_to(PHASE)
            and not entry.exists() and not output.exists() and entry != output
            and not entry.is_relative_to(output) and not output.is_relative_to(entry)
            and all(not candidate.is_relative_to(path) for candidate in (entry, output) for path in protected),
            'Separate fresh root outputs outside source/input/role/old-receipt scopes')
    entry.mkdir(parents=True, exist_ok=False)
    record = dict(status='started', complete=False, qualification_passed=False, release_sha256=sha(args.release),
        root_entry_sha256=sha(__file__), joint_source_seal_sha256=cfg['reviewed_joint_source']['seal']['sha256'],
        root_source_adoption_sha256=cfg['reviewed_joint_source']['root_adoption']['sha256'],
        plan_sha256=cfg['plan_sha256'], quality_scoring=False, VALID_checkpoint_selection=False,
        scientific_training=False, automatic_retry=False, TEST_file_access=False, original_sources_unchanged=True)
    write(entry / 'ROOT_REPORT.json', record)
    try:
        native_cfg = read(bound(old['qualified_native_runtime_release']))
        native_q = read(bound(old['native_qualification_receipt']))
        native_five = read(bound(old['native_five_reference_receipt']))
        role_path = bound(old['role1'])
        require(role_path.name == 'seed1.json', 'Existing exact role1 only')
        role = read(role_path)
        require(role['seed'] == 1 and role['input_files'] == old['expected_input_files']
                and native_cfg['roles']['1'] == old['role1'] and native_cfg['input_root'] == old['input_root']
                and native_cfg['expected_input_files'] == old['expected_input_files'], 'Existing canonical IMDB/input/role custody')
        require(native_q['complete'] is True and native_q['status'] == 'complete'
                and native_q['mode'] == 'qualification' and native_q['native_backbone_numerically_qualified'] is True
                and native_q['source_seal_sha256'] == native_cfg['source_seal_sha256'] == old['original_native_source_seal_sha256']
                and native_q['input_files'] == old['expected_input_files'], 'Actual existing native qualification')
        require(native_five['complete'] is True and native_five['status'] == 'complete'
                and native_five['native_five_seed_recipe_completed'] is True and native_five['seeds'] == [1, 2, 3, 4, 5]
                and native_five['runtime'] == native_q['runtime'], 'Existing native-five provenance only; no quality adoption')
        native = load('_joint_qualification_native', PHASE / 'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/runner.py')
        native.source_gate()
        rt, engine, runtime = native.runtime(native_cfg)
        require(runtime == native_q['runtime'], 'One actual unchanged qualified native runtime')
        for name in ('joint_contracts', 'joint_additive_adapter', 'joint_training'):
            require(name not in sys.modules, 'No previously imported V1 or other source')
        sys.path.insert(0, str(directory))
        import joint_training
        import joint_contracts
        modules = {'engine': engine}
        for key in joint_contracts.MODULE_KEYS:
            item = sources['modules'][key]
            require(sha(PHASE / item['path']) == item['sha256'], 'Exact bound dependency')
            if key != 'engine':
                modules[key] = load('_joint_qualification_' + key, PHASE / item['path'])
        joint_contracts.validate_modules(rt, modules, joint_contracts.source_gate())
        qualifier = load('_joint_full_input_qualifier', HERE / 'QUALIFY.py')
        costs = engine.Costs(entry, rt['torch'], rt['device'])
        result = qualifier.qualify(rt, modules, joint_training, joint_contracts, Path(old['input_root']),
            role_path, output, costs, cfg, old, adoption)
        require(result['complete'] is True and result['qualification_passed'] is True
                and result['actual_Adam_owner_steps'] == 8
                and result['peak_cuda_reserved_bytes'] <= cfg['resource_budget']['device_bytes'], 'Actual complete bounded witness/resources')
        record.update(status='complete', complete=True, qualification_passed=True,
            actual_Adam_owner_steps=8, native_runtime=runtime,
            qualification_report=dict(path=str(output / 'QUALIFICATION_REPORT.json'), bytes=(output / 'QUALIFICATION_REPORT.json').stat().st_size,
                sha256=sha(output / 'QUALIFICATION_REPORT.json')))
    except BaseException as error:
        record.update(status='failed', complete=False, qualification_passed=False,
            failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter() - started, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime,
            cumulative_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_is_process_lifetime_highwater=True, startup_runtime_setup_and_qualification_included=True)
        if record['complete'] and (record['seconds'] > cfg['resource_budget']['seconds'] or record['cumulative_RSS_peak_bytes'] > cfg['resource_budget']['host_RSS_bytes']):
            record.update(status='failed', complete=False, qualification_passed=False,
                failure=dict(type='ResourceCap', message='Actual inclusive wall/RSS exceeds root cap'))
        write(entry / 'ROOT_REPORT.json', record)
        write(entry / 'COMPLETE.json', {key: record[key] for key in ('status', 'complete', 'qualification_passed', 'quality_scoring', 'scientific_training', 'TEST_file_access')})
    require(record['complete'], 'Root qualification incomplete; preserve failure, no retry')
    print(json.dumps({key: record[key] for key in ('status', 'complete', 'qualification_passed', 'actual_Adam_owner_steps', 'seconds', 'CPU_user_seconds', 'CPU_system_seconds', 'cumulative_RSS_peak_bytes')}))


if __name__ == '__main__':
    main()

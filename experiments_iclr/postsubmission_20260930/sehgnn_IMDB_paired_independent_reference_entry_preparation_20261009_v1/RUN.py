"""Disabled narrow root entry for six fixed genuine independent4 reference groups."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PILOT_SEAL = '4350d456c4f646738f0fc71268e7641c13dd4a97ef36c841262fbf2a872cd9d0'
PAIRS = [[1, 1], [2, 2], [3, 3]]
VARIANTS = ['plain_native', 'untied_same_six_factors']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path):
    assert name not in sys.modules
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bound_path(row, basename=None):
    assert set(row) == {'path', 'bytes', 'sha256'} and Path(row['path']).is_absolute()
    path = Path(row['path'])
    assert not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    assert basename is None or path.name == basename
    return path


def source_gate():
    seal = read(HERE / 'SEAL.json')
    assert seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256']
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        assert path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
    return sources


def main():
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'inactive': True, 'reference_fits_or_provider_imports': False}))
        return
    assert args.release is not None
    sources = source_gate()
    cfg = read(args.release)
    assert cfg['enabled'] is True and cfg['root_source_review_approved'] is True and cfg['root_reference_fit_release_approved'] is True
    assert cfg['action'] == 'fit_fixed_paired_genuine_independent4_reference_family'
    assert cfg['root_entry_sha256'] == sha(__file__) and cfg['entry_source_seal_sha256'] == sha(HERE / 'SEAL.json')
    assert cfg['source_seal_sha256'] == PILOT_SEAL and cfg['native_and_independent_reference_competence_adopted'] is False
    assert cfg['root_owns_qualified_runtime_allocation_and_external_resource_monitor'] is True
    assert cfg['pairs'] == PAIRS and cfg['variants'] == VARIANTS and cfg['actual_body_fits'] == 24
    assert cfg['automatic_retry'] is False and cfg['comparative_outcome_publishing'] is False and cfg['TEST_file_access'] is False
    assert set(cfg['roles']) == {'1', '2', '3'} and digest(cfg['study']) == cfg['study_binding']
    assert cfg['study']['pilot_source_seal_sha256'] == PILOT_SEAL and cfg['study']['pairs'] == PAIRS
    assert cfg['study']['variants'] == VARIANTS and cfg['study']['roles'] == cfg['roles']
    assert cfg['study']['input_root'] == cfg['input_root'] and cfg['study']['expected_input_files'] == cfg['expected_input_files']
    assert cfg['study']['body_seeds'] == {str(base): [1000*base+m+1 for m in range(4)] for base in (1, 2, 3)}
    phase = HERE.parent
    repo = phase.parents[1]
    assert str(repo) == cfg['project_root'] and Path.cwd() == repo
    entry_output, family_output = Path(cfg['entry_output_directory']), Path(cfg['family_output_directory'])
    assert entry_output.is_absolute() and family_output.is_absolute() and not entry_output.exists() and not family_output.exists()
    assert family_output != entry_output and not family_output.is_relative_to(entry_output) and not entry_output.is_relative_to(family_output)
    entry_output.mkdir(parents=True, exist_ok=False)
    record = dict(status='started', complete=False, action=cfg['action'], source_seal_sha256=PILOT_SEAL,
        entry_source_seal_sha256=cfg['entry_source_seal_sha256'], root_entry_sha256=sha(__file__), release_sha256=sha(args.release),
        study_binding=cfg['study_binding'], pairs=PAIRS, variants=VARIANTS, declared_body_fits=24,
        reference_competence_pending=True, native_and_independent_reference_competence_adopted=False,
        startup_native_runtime_and_input_loading_inclusive=True, comparative_outcome_publishing=False,
        automatic_retry=False, TEST_file_access=False, family_output_directory=str(family_output))
    rt = engine = family = None
    write(entry_output / 'ROOT_REPORT.json', record)
    try:
        assert socket.gethostname() == 'anogena-2-0'
        assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
        native_cfg = read(bound_path(cfg['qualified_native_runtime_release']))
        assert cfg['input_root'] == native_cfg['input_root'] and cfg['expected_input_files'] == native_cfg['expected_input_files']
        assert all(cfg['roles'][seed] == native_cfg['roles'][seed] for seed in ('1', '2', '3'))
        roles = {int(seed): read(bound_path(row, 'seed' + seed + '.json')) for seed, row in cfg['roles'].items()}
        assert all(role['seed'] == seed and role['input_files'] == cfg['expected_input_files'] for seed, role in roles.items())
        native_q = read(bound_path(cfg['native_qualification_receipt'], 'COHORT_REPORT.json'))
        compressed_q = read(bound_path(cfg['integration_qualification_receipt'], 'QUALIFICATION_REPORT.json'))
        independent_q = read(bound_path(cfg['independent4_qualification_receipt'], 'QUALIFICATION_REPORT.json'))
        assert native_q['complete'] is True and native_q['status'] == 'complete' and native_q['native_backbone_numerically_qualified'] is True
        assert native_q['mode'] == 'qualification' and native_q['seeds'] == [1] and native_q['source_seal_sha256'] == sources['native_seal_sha256']
        assert native_cfg['source_seal_sha256'] == sources['native_seal_sha256'] and native_q['input_files'] == cfg['expected_input_files']
        assert compressed_q['complete'] is True and compressed_q['status'] == 'complete' and compressed_q['representative_integration_eligible'] is True
        assert compressed_q['source_seal_sha256'] == sources['compressed_qualifier_seal_sha256'] and compressed_q['quality_scoring'] is False
        assert compressed_q['native_qualification_binding'] == cfg['native_qualification_receipt']['sha256']
        assert compressed_q['projection_backend_identity_sha256'] == sources['projection_backend_identity_sha256']
        assert independent_q['complete'] is True and independent_q['status'] == 'complete' and independent_q['qualification_passed'] is True
        assert independent_q['source_seal_sha256'] == PILOT_SEAL and independent_q['real_native_body_updates'] == 8
        assert independent_q['quality_scoring'] is False and independent_q['VALID_checkpoint_selection'] is False
        assert independent_q['role_seed'] == 1 and all(row['complete'] is True for row in independent_q['variants'])
        native = load('_root_paired_reference_native', phase / 'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2' / 'runner.py')
        native.source_gate()
        rt, engine, runtime = native.runtime(native_cfg)
        assert runtime == native_q['runtime'] == compressed_q['runtime']
        record['native_runtime'] = runtime
        sys.path.insert(0, str(phase / 'sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2'))
        import family_driver
        import independent_controls
        pilot_sources = family_driver.source_gate()

        def dependency(name, key):
            matches = [phase / row['path'] for row in pilot_sources['files'] if row['sha256'] == pilot_sources[key] and row['path'].endswith('.py')]
            assert len(matches) == 1
            return load(name, matches[0])

        adapter = dependency('_root_paired_reference_adapter', 'adapter_source_sha256')
        metadata = dependency('_root_paired_reference_metadata', 'independent_metadata_sha256')
        independent = dependency('_root_paired_reference_control', 'independent_source_sha256')
        views = dependency('_root_paired_reference_views', 'view_source_sha256')
        assert cfg['study']['own_selection'] == independent.SELECTION
        assert cfg['study']['pilot_protocol_sha256'] == sha(family_driver.HERE / 'PROTOCOL.json')
        seam = phase / 'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1'
        loader = load('_root_paired_reference_loader', seam / 'role_loader.py')
        loader.source_gate()
        costs = engine.Costs(entry_output, rt['torch'], rt['device'])
        with costs.measure('root_entry_once_only_actual_frozen_development_input_load'):
            data = loader.load(Path(cfg['input_root']), Path(cfg['roles']['1']['path']), read(seam / 'SOURCE_EXPECTATIONS.json'))
            assert data.input_bindings == cfg['expected_input_files']
            for role in roles.values():
                data.with_roles(role)
        pilot_cfg = family_driver.PilotConfig(enabled=True, root_source_review_approved=True, root_scientific_release_approved=True,
            source_seal_sha256=PILOT_SEAL, native_qualification_binding=cfg['native_qualification_receipt']['sha256'],
            integration_qualification_binding=cfg['integration_qualification_receipt']['sha256'],
            independent4_qualification_binding=cfg['independent4_qualification_receipt']['sha256'],
            native_and_independent_reference_competence_adopted=False,
            role_bindings={int(seed): row['sha256'] for seed, row in cfg['roles'].items()},
            role_files={int(seed): row['path'] for seed, row in cfg['roles'].items()}, study_binding=cfg['study_binding'])
        with costs.measure('root_entry_complete_existing_paired_independent_reference_family', gpu=True):
            family = independent_controls.run_independent_family(rt, engine, adapter, metadata, independent, views,
                data, roles, family_output, pilot_cfg)
        assert family['complete'] is True and family['complete_groups'] == 6 and family['actual_body_fits'] == 24
        assert family['native_and_independent_reference_competence_pending'] is True
        assert loader.bindings(loader.permitted_files(Path(cfg['input_root']))) == cfg['expected_input_files']
        # Only resource fields enter this wrapper report; native own selector
        # and diagnostics remain in the existing driver's owned result files.
        resources = []
        for cell in family['cells']:
            result = read(Path(cell['output']) / 'RESULT.json')
            resources.append(dict(role_seed=cell['role_seed'], variant=cell['variant'], complete=result['complete'],
                peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'], peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes']))
        record['completed_cell_resources'] = resources
        assert all(row['complete'] is True and row['peak_cuda_reserved_bytes'] <= cfg['resource_budget']['device_bytes'] for row in resources)
        record.update(status='complete', complete=True, complete_groups=6, completed_actual_body_fits=24)
    except BaseException as error:
        record.update(status='failed', complete=False, failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        end = resource.getrusage(resource.RUSAGE_SELF)
        record.update(seconds=time.perf_counter()-started, CPU_user_seconds=end.ru_utime-usage.ru_utime,
            CPU_system_seconds=end.ru_stime-usage.ru_stime,
            cumulative_RSS_peak_bytes=int(end.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
            RSS_peak_is_process_lifetime_highwater=True)
        if record['complete'] and (record['seconds'] > cfg['resource_budget']['seconds'] or record['cumulative_RSS_peak_bytes'] > cfg['resource_budget']['host_RSS_bytes']):
            record.update(status='failed', complete=False, failure=dict(type='ResourceCap', message='Actual inclusive root entry wall/RSS exceeds declared cap'))
        write(entry_output / 'ROOT_REPORT.json', record)
        write(entry_output / 'COMPLETE.json', dict(status=record['status'], complete=record['complete'],
            reference_competence_pending=True, native_and_independent_reference_competence_adopted=False,
            comparative_outcome_publishing=False, automatic_retry=False, TEST_file_access=False))
    assert record['complete'] is True
    print(json.dumps({key: record[key] for key in ('status', 'complete', 'complete_groups', 'completed_actual_body_fits',
        'seconds', 'CPU_user_seconds', 'CPU_system_seconds', 'cumulative_RSS_peak_bytes', 'reference_competence_pending',
        'comparative_outcome_publishing', 'automatic_retry', 'TEST_file_access')}))


if __name__ == '__main__':
    main()

"""Root entry for the reviewed eight-update full-input independent4 witness."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'inactive': True}))
        return
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    activation = Path(__file__).resolve().parent
    phase = activation.parent
    repo = phase.parents[1]
    assert Path.cwd() == repo and socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    cfg = read(args.release)
    assert cfg['enabled'] is True and cfg['root_source_review_approved'] is True
    assert cfg['action'] == 'qualify_both_genuine_independent4_variants_one_actual_native_epoch_per_body'
    assert cfg['root_owns_existing_qualified_runtime_allocation_and_external_resource_monitor'] is True
    assert cfg['root_entry_sha256'] == sha(__file__)
    assert cfg['quality_scoring'] is False and cfg['TEST_file_access'] is False
    native_binding = cfg['qualified_native_runtime_release']
    assert sha(native_binding['path']) == native_binding['sha256']
    native_cfg = read(native_binding['path'])
    pilot = phase / 'sehgnn_paired_six_arm_source_supply_pilot_source_20261009_v2'
    assert sha(pilot / 'SEAL.json') == cfg['source_seal_sha256']
    native = load('_root_independent4_native', phase / 'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2' / 'runner.py')
    native.source_gate()
    rt, engine, runtime = native.runtime(native_cfg)
    native_report_path = Path(native_cfg['output_directory']) / 'COHORT_REPORT.json'
    assert sha(native_report_path) == cfg['native_qualification_binding']
    native_report = read(native_report_path)
    assert native_report['complete'] is True and native_report['status'] == 'complete'
    assert native_report['native_backbone_numerically_qualified'] is True
    assert native_report['mode'] == 'qualification' and native_report['seeds'] == [1]
    assert native_report['source_seal_sha256'] == native_cfg['source_seal_sha256']
    assert native_report['runtime'] == runtime
    sys.path.insert(0, str(pilot))
    import family_driver
    import independent4_qualification
    sources = family_driver.source_gate()

    def dependency(name, key):
        matches = [phase / row['path'] for row in sources['files'] if row['sha256'] == sources[key] and row['path'].endswith('.py')]
        assert len(matches) == 1
        return load(name, matches[0])

    adapter = dependency('_root_independent4_adapter', 'adapter_source_sha256')
    metadata = dependency('_root_independent4_metadata', 'independent_metadata_sha256')
    independent = dependency('_root_independent4_control', 'independent_source_sha256')
    seam = phase / 'imdb_role_isolated_public_schema_backbone_preparation_20261009_v1'
    loader = load('_root_independent4_loader', seam / 'role_loader.py')
    loader.source_gate()
    data = loader.load(Path(native_cfg['input_root']), Path(cfg['role_file']), read(seam / 'SOURCE_EXPECTATIONS.json'))
    assert data.input_bindings == native_cfg['expected_input_files']
    config = independent4_qualification.QualificationConfig(
        enabled=True, root_source_review_approved=True,
        source_seal_sha256=cfg['source_seal_sha256'],
        native_qualification_binding=cfg['native_qualification_binding'],
        role_binding=cfg['role_binding'], role_file=cfg['role_file'],
        qualification_binding=cfg['qualification_binding'])
    result = independent4_qualification.qualify(rt, engine, adapter, metadata,
        independent, data, read(cfg['role_file']), Path(cfg['output_directory']), config)
    assert result['complete'] is True and result['qualification_passed'] is True
    assert result['real_native_body_updates'] == 8
    assert result['peak_cuda_reserved_bytes'] <= cfg['resource_budget']['device_bytes']
    assert loader.bindings(loader.permitted_files(Path(native_cfg['input_root']))) == native_cfg['expected_input_files']
    end = resource.getrusage(resource.RUSAGE_SELF)
    receipt = dict(complete=True, native_runtime=runtime,
        root_entry_sha256=sha(__file__), release_sha256=sha(args.release),
        seconds=time.perf_counter() - started,
        CPU_user_seconds=end.ru_utime - usage.ru_utime,
        CPU_system_seconds=end.ru_stime - usage.ru_stime,
        cumulative_RSS_peak_bytes=int(end.ru_maxrss * 1024),
        native_setup_input_loading_and_witness_included=True,
        accuracy_reference_competence_established=False,
        quality_scoring=False, TEST_file_access=False)
    assert receipt['seconds'] <= cfg['resource_budget']['seconds']
    assert receipt['cumulative_RSS_peak_bytes'] <= cfg['resource_budget']['host_RSS_bytes']
    with (activation / 'ROOT_REPORT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

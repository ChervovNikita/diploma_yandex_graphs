#!/usr/bin/env python3
"""Disabled one-attempt fabricated bucket QA; source preparation is not QA PASS."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
CANDIDATE = RESEARCH / 'exact_cb_support_bucket_implementation_hypothesis_preparation_20261004_v2'
CANDIDATE_MANIFEST = '47d8fc112f82d05bfce84548f86538e2b259b67efaf3635c5d2de42c567a434b'
CORE = RESEARCH / 'graph_count_conditioned_pattern_loss_prototype_preparation_20261004_v3'
CORE_MANIFEST = '92ae9f79c15cf6de53e39f06c241780b8089d652d59070cf9d23ed521e2740cf'
EXECUTION = RESEARCH / 'exact_cb_support_bucket_cpu_qa_execution_root_20261004_v1'
INNER_RELEASE = EXECUTION / 'ROOT_RELEASE_fabricated_cpu.json'
OUTER_RELEASE = EXECUTION / 'ROOT_RELEASE_owned_supervisor.json'
ROOT_LOCK = EXECUTION / '.ROOT_LOCK_fabricated_cpu_owned_supervisor'
CAPS = {'wall_seconds': 900, 'peak_RSS_bytes': 4 * 1024**3}
EXPECTED_CASES = {'ragged_analytic_cases': 96, 'masked_ESP_cases': 12, 'zero_cases': 6, 'plan_cases': 5}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def pinned_json(path, pin):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32 * 1024**2
            and sha(path) == pin, 'Pinned JSON differs: ' + str(path))
    return json.loads(path.read_text())


def verify_rows(base, rows):
    require(isinstance(rows, list) and rows, 'Empty file closure')
    seen = set()
    for row in rows:
        path = base / row['path']
        require(row['path'] not in seen and path.resolve().is_relative_to(base.resolve())
                and path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
                and sha(path) == row['sha256'], 'File closure differs: ' + str(path))
        seen.add(row['path'])


def review(path, pin, manifest):
    require(path.resolve().is_relative_to(RESEARCH) and not path.is_symlink(), 'Review outside source project')
    value = pinned_json(path, pin)
    require(value.get('status') == 'PASS' and value.get('candidate_manifest_sha256') == manifest
            and value.get('execution_authorized') is False and not value.get('blocking_findings'),
            'Independent exact source PASS required')


def gate(path, pin, before_numerical=True):
    require(path == INNER_RELEASE and path.resolve() == path, 'Exact external child release required')
    release = pinned_json(path, pin)
    require(release.get('schema') == 'exact-CB-bucket-fabricated-cpu-root-release-v1'
            and release.get('status') == 'APPROVED' and release.get('authorized_stages') == ['fabricated_cpu']
            and release.get('root_authorization_reference') and release.get('caps') == CAPS
            and release.get('threads') == 2 and release.get('interop_threads') == 1
            and release.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(release.get(name) is False for name in
                    ('automatic_retry', 'scientific_fit_admitted', 'GPU_data_access', 'TEST_supported')),
            'Fabricated CPU-only authorization absent')
    require(release['source_manifest_sha256'] == CANDIDATE_MANIFEST
            and release['core_manifest_sha256'] == CORE_MANIFEST, 'Candidate/core identity differs')
    for folder, manifest in ((HERE, release['qa_manifest_sha256']), (CANDIDATE, CANDIDATE_MANIFEST), (CORE, CORE_MANIFEST)):
        value = pinned_json(folder / 'MANIFEST.json', manifest)
        require(value['source_only'] is True and value['execution_authorized'] is False, 'Source preparation flags differ')
        verify_rows(folder, value['files'])
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    verify_rows(RESEARCH, binding['external_input_pins'])
    core_review = binding['reused_core_independent_review']
    review(RESEARCH / core_review['path'], core_review['sha256'], CORE_MANIFEST)
    review(Path(release['independent_source_review_path']), release['independent_source_review_sha256'], CANDIDATE_MANIFEST)
    review(Path(release['independent_qa_review_path']), release['independent_qa_review_sha256'], release['qa_manifest_sha256'])
    require(ROOT_LOCK.resolve() == ROOT_LOCK and ROOT_LOCK.is_file() and not ROOT_LOCK.is_symlink()
            and ROOT_LOCK.stat().st_size <= 32768, 'Owned plain root lock required')
    lock = json.loads(ROOT_LOCK.read_text())
    outer = pinned_json(OUTER_RELEASE, lock['root_release_sha256'])
    require(outer.get('status') == 'APPROVED' and outer.get('caps') == CAPS
            and outer.get('candidate_manifest_sha256') == CANDIDATE_MANIFEST
            and lock['supervisor_pid'] == os.getppid() and lock['candidate_manifest_sha256'] == CANDIDATE_MANIFEST
            and lock['supervisor_manifest_sha256'] == release['qa_manifest_sha256']
            and outer['inner_cpu_release_sha256'] == pin and outer['supervisor_manifest_sha256'] == release['qa_manifest_sha256']
            and outer['independent_supervisor_review_sha256'] == release['independent_qa_review_sha256']
            and os.getsid(0) == os.getpid() and os.getpgrp() == os.getpid(),
            'Held physical supervisor/session release required')
    profile = release['runtime_profile']
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode and sys.platform in ('linux', 'darwin')
            and sys.version_info >= (3, 10) and Path(sys.executable).resolve() == Path(profile['python_executable'])
            and sha(Path(profile['python_executable'])) == profile['python_executable_sha256'], 'Pinned isolated runtime differs')
    if before_numerical:
        require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules), 'Torch imported before source gate')
    require(importlib.metadata.version('torch') == profile['torch_distribution_version'], 'Pinned Torch distribution version differs')
    return release, binding


def load_bound_modules(binding):
    loaded = {}
    for row in binding['module_load_order']:
        name, path = row['module'], RESEARCH / row['path']
        require(name not in sys.modules and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'],
                'Module preloaded or changed: ' + name)
        spec = importlib.util.spec_from_file_location(name, path)
        require(spec is not None and spec.loader is not None, 'Pinned file loader absent')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        source = path.read_bytes()
        require(hashlib.sha256(source).hexdigest() == row['sha256'], 'Module bytes changed before compilation')
        # Execute authenticated source bytes; no unlisted project .pyc can replace them.
        exec(compile(source, str(path), 'exec'), module.__dict__)
        require(Path(module.__file__).resolve() == path, 'Module origin differs: ' + name)
        loaded[name] = module
    require(loaded['bucketed_oracles'].bucketed_training_pattern_losses is loaded['conditional_loss'].training_pattern_losses
            and loaded['bucketed_oracles'].training_pattern_losses is loaded['_sealed_v3_conditional_loss'].training_pattern_losses
            and loaded['oracles'].training_pattern_losses is loaded['conditional_loss'].training_pattern_losses,
            'Actual candidate/direct comparator oracle binding differs')
    return loaded


def source_custody(binding):
    rows = []
    for row in binding['module_load_order']:
        module = sys.modules.get(row['module'])
        path = RESEARCH / row['path']
        require(module is not None and Path(module.__file__).resolve() == path
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Actual loaded source module differs')
        rows.append(dict(row, actual_file=str(path)))
    return rows


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, binding = gate(args.root_release, args.release_sha256)
    output = EXECUTION / 'fabricated_cpu/run01'
    require(output.resolve() == output and not output.exists(), 'Fresh CPU attempt required; no retry')
    output.mkdir(parents=True)
    ledger = {'comparison_reports': []}
    last_progress = [started]
    def check_cap():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
        require(time.monotonic() - started <= CAPS['wall_seconds'] and peak_bytes <= CAPS['peak_RSS_bytes'], 'CPU wall/RSS cap exceeded')
        if time.monotonic() - last_progress[0] >= 1:
            write(output / 'PROGRESS.json', {'bucketed_case_counts': ledger.get('bucketed_case_counts'),
                  'completed_comparison_reports': len(ledger['comparison_reports']),
                  'inclusive_wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': peak_bytes})
            last_progress[0] = time.monotonic()
        return peak_bytes
    try:
        # Exact source, interpreter hash and runtime version gates precede this first numerical import.
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '2'
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1
                and torch.__version__ == release['runtime_profile']['torch_distribution_version'], 'Exact Torch CPU runtime differs')
        modules = load_bound_modules(binding)
        before = source_custody(binding)
        write(output / 'SOURCE_CUSTODY.json', {'candidate_manifest_sha256': CANDIDATE_MANIFEST,
              'core_manifest_sha256': CORE_MANIFEST, 'qa_manifest_sha256': release['qa_manifest_sha256'],
              'module_load_order': binding['module_load_order'], 'before_QA': before, 'actual_module_bindings_checked': True})
        result = modules['bucketed_oracles'].qualify_bucketed(check_cap, ledger)
        require(result['cases'] == EXPECTED_CASES and len(ledger['comparison_reports']) == 3288, 'Exact bucket-only coverage differs')
        result['comparison_reports'] = ledger['comparison_reports']
        # Re-authenticate releases, exact source, interpreter hash and runtime version after QA.
        final_release, final_binding = gate(args.root_release, args.release_sha256, before_numerical=False)
        require(final_release == release and final_binding == binding, 'Final release/source binding differs')
        after = source_custody(binding)
        write(output / 'RUNTIME_SCOPE.json', {'python_version': sys.version, 'platform': sys.platform,
              'python_executable': str(Path(sys.executable).resolve()), 'torch_version': torch.__version__,
              'torch_distribution_version': importlib.metadata.version('torch'), 'threads': torch.get_num_threads(),
              'interop_threads': torch.get_num_interop_threads(), 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
              'source_modules_after_QA': after, 'complete_runtime_closure_claim': False,
              'scope_limit': 'Pinned interpreter hash and Torch distribution/version metadata; runtime package payloads, stdlib and OS libraries are not exhaustively pinned.'})
        peak = check_cap()
        write(output / 'QUALIFICATION.json', {'schema': 'exact-CB-bucket-fabricated-cpu-qualification-v1',
              'status': 'PASS', 'UTC': datetime.now(timezone.utc).isoformat(), 'source_manifest_sha256': CANDIDATE_MANIFEST,
              'core_manifest_sha256': CORE_MANIFEST, 'qa_manifest_sha256': release['qa_manifest_sha256'],
              'root_release_sha256': args.release_sha256, 'runtime_profile': release['runtime_profile'],
              'torch_version': torch.__version__, 'inclusive_wall_seconds': time.monotonic() - started, 'peak_RSS_bytes': peak,
              'result': result, 'actual_module_bindings_checked': True, 'complete_runtime_closure_claim': False,
              'GPU_data_access': False, 'scientific_fit_admitted': False, 'TEST_supported': False,
              'native_full_batch_resource_qualification': False, 'frozen_four_arm_screen_changed': False})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error),
              'incurred_oracle_ledger': ledger, 'inclusive_wall_seconds': time.monotonic() - started,
              'automatic_retry': False, 'scientific_fit_admitted': False})
        raise
    finally:
        rows = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
                for path in sorted(output.iterdir()) if path.is_file() and not path.name.endswith('.tmp')]
        write(output / 'FINAL_CUSTODY.json', {'stage': 'fabricated_cpu', 'completed': (output / 'QUALIFICATION.json').exists(), 'files': rows})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

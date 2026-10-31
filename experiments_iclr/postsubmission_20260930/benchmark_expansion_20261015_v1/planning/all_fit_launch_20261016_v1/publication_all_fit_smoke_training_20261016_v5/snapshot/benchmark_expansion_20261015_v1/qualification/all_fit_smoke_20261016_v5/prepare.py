"""Enumerate every deployed fit, preserving its exact descriptor and owner."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from support import (E, PORT, PORT_SHA, CANONICAL_SHA, COUNTS, digest, fit_directory,
                     guard, inside, module, own_source, pin, read, read_pinned, require, verify, write)


def interleave(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['worker'], row['cell_id'], row['scientific_method_id'], row['protocol_track'])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (row['seed'], row['split'], str(row['body']), row['id']))
    ordered = []
    for index in range(max(map(len, groups.values()))):
        for key in sorted(groups):
            if index < len(groups[key]):
                ordered.append(groups[key][index])
    for index, row in enumerate(ordered):
        row['smoke_order'] = index
    return ordered


def prepare(root, release_path, release_sha, candidates_path, candidates_sha, output):
    config, _ = guard(root)
    root = Path(root).resolve()
    source = own_source()
    require(pin(PORT / 'MANIFEST.json')['sha256'] == PORT_SHA, 'Frozen port manifest differs')
    release, release_pin = read_pinned(release_path, release_sha, root)
    candidates, candidates_pin = read_pinned(candidates_path, candidates_sha, root)
    host = config['worker'].split('/')[0]
    require(release['host'] == host and release['TEST_access'] is False, 'Wrong host release')
    canonical_pin = release['canonical_skeleton']
    require(canonical_pin['sha256'] == CANONICAL_SHA, 'Wrong canonical inventory')
    canonical = read(verify(canonical_pin, root))
    canonical_fits = {t['id']: t for t in canonical['tasks'] if t['kind'] == 'fit'}
    require(len(canonical_fits) == 6460 and len(canonical['tasks']) == 6730, 'Exact canonical fit universe required')
    by_id = {row['id']: row for row in candidates}
    require(len(by_id) == len(candidates) == len(release['tasks']), 'Candidate coverage differs')
    require(set(by_id) == {t['id'] for t in release['tasks']}, 'Candidate IDs differ')
    # Verify the frozen source closure once here; no dataset hashes in run-one.
    source_files = {row['path']: row for row in release['source_files']}
    require(digest({p: row['sha256'] for p, row in source_files.items()}) == release['source_bundle_sha256'], 'Source closure differs')
    for row in source_files.values():
        verify(row, root)
    registry_pin = release['reporting']['registry']
    registry = read(verify(registry_pin, root))
    presets = {}
    for preset_pin in registry['preset_files']:
        preset = read(verify(preset_pin, root))
        for fit_id, spec in preset['fit_specs'].items():
            require(fit_id not in presets, 'Duplicate reporting fit')
            presets[fit_id] = (preset_pin, spec)
    sys.path.insert(0, str(PORT))
    binding = module(PORT / 'binding_rules.py', '_all_fit_exact_binding')
    rows = []
    for task in release['tasks']:
        if task['kind'] != 'fit':
            continue
        candidate = by_id[task['id']]
        require(task['id'] in canonical_fits, 'Unexpected fit')
        require(task['canonical_task_sha256'] == digest(canonical_fits[task['id']]), 'Canonical task changed')
        descriptor = candidate['job_descriptor']
        job = read(verify(descriptor, root))
        binding.validate_job_binding(task, job, root)
        require(candidate['worker'] == task['worker'] and candidate['runtime_job_hash'] == task['runtime_job_hash'] == digest(job), 'Execution identity changed')
        entry = candidate['runtime_entrypoint']
        verify(entry, root)
        require(Path(entry['path']) == binding.expected_entrypoint(task, root), 'Wrong native runtime')
        preset_pin, spec = presets[task['id']]
        require(spec['descriptor'] == descriptor and spec['runtime_entrypoint'] == entry and spec['job_hash'] == task['runtime_job_hash'], 'Reporting descriptor differs')
        require(job['TEST_access'] is False and job['device'] == 'cuda:0', 'Native CUDA TRAIN/VALID job required')
        rows.append({'id': task['id'], 'worker': task['worker'], 'descriptor': descriptor,
            'runtime_entrypoint': entry, 'runtime_job_hash': task['runtime_job_hash'],
            'canonical_runtime_job_hash': task['canonical_runtime_job_hash'],
            'science_contract_hash': task['science_contract_hash'], 'execution_science_contract_hash': task['execution_science_contract_hash'],
            'preset': preset_pin, 'family': job['family'], 'recipe_id': job['recipe_id'], 'procedure': job['procedure'],
            'seed': job['seed'], 'split': job.get('split', 0), 'body': job.get('body'),
            'cell_id': task['cell_id'], 'scientific_method_id': task.get('scientific_method_id') or job['procedure'], 'protocol_track': task['protocol_track'],
            'smoke_output_relative': fit_directory(task['id']), 'TRAIN_updates_requested': 2})
    counts = {worker: sum(row['worker'] == worker for row in rows) for worker in COUNTS if worker.startswith(host + '/')}
    require(all(counts[w] == COUNTS[w] for w in counts), 'Host must cover every assigned fit')
    value = {'schema': 'all-fit-smoke-inventory-v1', 'host': host, 'project_root': str(root),
        'source_manifest': source, 'producer_manifest': pin(PORT / 'MANIFEST.json'),
        'release': release_pin, 'candidates': candidates_pin, 'reporting_index': registry_pin,
        'canonical_plan': canonical_pin, 'canonical_fit_ids': sorted(canonical_fits),
        'canonical_fit_ids_sha256': digest(sorted(canonical_fits)),
        'fits': interleave(rows), 'counts': counts, 'source_closure_verified_once': True,
        'source_bundle_sha256': release['source_bundle_sha256'], 'TRAIN_updates_per_fit': 2,
        'ordering': 'interleave exact worker/cell/method/track combinations before repeated seeds',
        'class_substitution': False, 'new_native_work': 0, 'full_training_admitted': False}
    output = inside(output, root)
    require(output.is_relative_to(E) and not output.exists(), 'Fresh project smoke inventory required')
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, value, exclusive=True)
    return {'inventory': pin(output), 'counts': counts, 'fits': len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'release', 'release-sha256', 'candidates', 'candidates-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    import json
    print(json.dumps(prepare(a.project_root, a.release, a.release_sha256, a.candidates, a.candidates_sha256, a.output)))


if __name__ == '__main__':
    main()

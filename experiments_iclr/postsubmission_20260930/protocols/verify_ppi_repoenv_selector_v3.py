"""Bounded selector-path engineering gate; no graphs, fits or GPU context.

Read all six actual repoenv locks, then simulate a legacy path and a changed
checkpoint digest in memory. Never modify a report, checkpoint or old lock.
Invoke through run_logged.py in the coordinator's existing repo environment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('ppi_repoenv_selector_subject', PHASE / 'protocols/ppi_study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    if study.torch.cuda.is_initialized():
        raise ValueError('Gate must begin without a CUDA context')
    confinement = study.confinement_gate()
    output = study.inside(args.output)
    if output.exists():
        raise ValueError('Refuse to overwrite selector gate evidence')
    manifest = PHASE / 'new_tasks/data/ppi/frozen_shareable_manifest_v2.json'
    checks, choice = [], {}
    with patch.object(study, 'PPI', side_effect=AssertionError('Engineering gate must not construct a graph loader')):
        for fraction in (0.1, 1.0):
            locks = study.selector_locks(fraction, manifest)
            choice[str(fraction)] = [locks[str(s)]['private_indices'] for s in (0, 1, 2)]
        if choice != {'0.1': [[1], [1], [0]], '1.0': [[0], [0], [0]]}:
            raise ValueError('Unexpected locked policy')
        checks.append({'check': 'actual_runner_reads_six_clean_locks', 'status': 'PASS'})

        real_read = Path.read_text
        comparison = PHASE / 'sharing_policy/PPI_REPOENV_REPRO_COMPARISON_v1.json'

        def changed_path(path, *args, **kwargs):
            value = real_read(path, *args, **kwargs)
            if path == comparison:
                record = json.loads(value)
                record['runs'][0]['reproduction'] = record['runs'][0]['original']
                return json.dumps(record)
            return value

        with patch.object(Path, 'read_text', changed_path):
            try:
                study.selector_locks(0.1, manifest)
            except ValueError as exc:
                if str(exc) != 'Fresh probe reproduction identity differs':
                    raise
                checks.append({'check': 'legacy_cpu_run_path_rejected', 'status': 'PASS'})
            else:
                raise AssertionError('Legacy path was accepted')

        real_sha = study.sha
        target = PHASE / 'sharing_policy/ppi_probe_v2_seed0_label100_repoenv_repro_v1/probe_epoch10.pt'

        def changed_checkpoint(path):
            return '0' * 64 if Path(path) == target else real_sha(path)

        with patch.object(study, 'sha', changed_checkpoint):
            try:
                study.selector_locks(1.0, manifest)
            except ValueError as exc:
                if str(exc) != 'Fresh probe checkpoint or input changed':
                    raise
                checks.append({'check': 'changed_fresh_checkpoint_digest_rejected', 'status': 'PASS'})
            else:
                raise AssertionError('Changed checkpoint digest was accepted')
    if study.torch.cuda.is_initialized():
        raise ValueError('Gate initialized a CUDA context')
    output.parent.mkdir(parents=True, exist_ok=True)
    study.write_json(output, {'status': 'PASS', 'scope': 'six actual lock reads and in-memory selector refusal engineering gate',
                             'checks': checks, 'private_indices': choice, 'source_sha256': study.sources(),
                             'gate_source_sha256': real_sha(Path(__file__)),
                             'shareable_manifest_sha256': real_sha(manifest),
                             'comparison_sha256': real_sha(comparison), 'confinement': confinement,
                             'dataset_loader_constructed': False, 'optimizer_steps': 0,
                             'GPU_context': False, 'validation_or_test_scored': False,
                             'evidence_bytes_mutated': False, 'not_research_performance_evidence': True})
    print(json.dumps({'status': 'PASS', 'checks': checks, 'GPU_context': False}), flush=True)


if __name__ == '__main__':
    main()

"""Compare fresh P-only fits with retained earlier fits; never modify evidence."""
import csv
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
REMOTE_PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runs = []
    for seed in range(3):
        for budget in (10, 100):
            stem = f'ppi_probe_v2_seed{seed}_label{budget}'
            old = PHASE / 'sharing_policy' / (stem + '_cpu')
            new = PHASE / 'sharing_policy' / (stem + '_repoenv_repro_v1')
            supervisor = PHASE / 'protocols/logged_runs' / f'ppi_probe_seed{seed}_label{budget}_repoenv_repro_v1'
            hashes = {}
            for name in ('contract.json', 'selector_lock.json', 'label_masks.pt',
                         'probe_epoch0.json', 'probe_epoch10.json', 'probe_epoch25.json',
                         'probe_epoch0.pt', 'probe_epoch10.pt', 'probe_epoch25.pt'):
                a, b = sha(old / name), sha(new / name)
                if a != b:
                    raise ValueError(f'Fresh fit mismatch: {stem}/{name}')
                hashes[name] = a
            old_sources = {str(f.relative_to(old / 'source')): sha(f)
                           for f in (old / 'source').rglob('*') if f.is_file()}
            new_sources = {str(f.relative_to(new / 'source')): sha(f)
                           for f in (new / 'source').rglob('*') if f.is_file()}
            if old_sources != new_sources:
                raise ValueError('Archived sources differ')
            with (old / 'fit_trace.csv').open() as stream:
                a = list(csv.DictReader(stream))
            with (new / 'fit_trace.csv').open() as stream:
                b = list(csv.DictReader(stream))
            if len(a) != len(b) or len(a) != 25:
                raise ValueError('Fit trajectory length differs')
            changes = sorted({key for x, y in zip(a, b) for key in x if x[key] != y[key]})
            if any(key != 'fit_epoch_seconds' for key in changes):
                raise ValueError('Predictive-loss/step trajectory differs')
            completion = json.loads((supervisor / 'completion.json').read_text())
            for key, name in (('environment_sha256', 'environment.json'),
                              ('command_sha256', 'command.json'),
                              ('log_sha256', 'stdout_stderr.log')):
                if completion[key] != sha(supervisor / name):
                    raise ValueError('Supervisor integrity differs')
            if completion['exit_code'] != 0:
                raise ValueError('Supervised fit failed')
            environment = json.loads((supervisor / 'environment.json').read_text())
            cache = environment['cache_and_temp']
            if len(cache) != 17 or any(not value or not Path(value).is_relative_to(REMOTE_PHASE)
                                       for value in cache.values()):
                raise ValueError('Cache/temp confinement mismatch')
            result = json.loads((new / 'result.json').read_text())
            if (result['validation_loader_accessed'] or result['test_loader_accessed']
                    or result['fit_graph_steps'] != 250):
                raise ValueError('Unexpected data/fit scope')
            runs.append({'seed': seed, 'label_fraction': budget / 100,
                         'original': str(old.relative_to(PHASE)),
                         'reproduction': str(new.relative_to(PHASE)),
                         'identical_artifact_sha256': hashes,
                         'archived_sources_identical': True,
                         'mean_graph_step_bce_and_steps_identical': True,
                         'fit_trace_changed_columns': changes,
                         'old_fit_trace_sha256': sha(old / 'fit_trace.csv'),
                         'new_fit_trace_sha256': sha(new / 'fit_trace.csv'),
                         'private_indices': result['private_indices'],
                         'fresh_training_graph_steps': 250,
                         'supervisor_completion_sha256': sha(supervisor / 'completion.json'),
                         'verified_cache_temp_paths': len(cache),
                         'validation_or_test_scored': False})
    report = {'status': 'PASS', 'scope': 'six fresh P-only training probe reproductions on the same CPU runtime',
              'runs': runs, 'comparison_script_sha256': sha(Path(__file__)),
              'prior_runs_cache_confinement_certified': False,
              'cross_runtime_reproduction_claimed': False,
              'predictive_improvement_claimed': False}
    destination = PHASE / 'sharing_policy/PPI_REPOENV_REPRO_COMPARISON_v1.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'runs': len(runs),
                      'choices10': [r['private_indices'] for r in runs if r['label_fraction'] == 0.1],
                      'choices100': [r['private_indices'] for r in runs if r['label_fraction'] == 1.0]}))


if __name__ == '__main__':
    main()

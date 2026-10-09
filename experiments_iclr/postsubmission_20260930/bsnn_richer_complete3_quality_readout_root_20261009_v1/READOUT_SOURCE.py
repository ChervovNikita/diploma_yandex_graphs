"""Read the complete stronger baseline after owned closure; no model/array access."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import statistics
import subprocess

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'bsnn_richer_full_three_seed_baseline_activation_root_20261009_v1'
D = P / 'bsnn_richer_full_three_seed_baseline_execution_root_20261009_v1'
O = P / 'bsnn_richer_complete3_quality_readout_root_20261009_v1'
SEEDS = (7409, 8501, 9607)
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def read(path):
    return json.loads(path.read_text())

def bind(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
launch, owner, terminal = read(A / 'LAUNCH.json'), read(A / 'WORKER_OWNER.json'), read(A / 'TERMINAL.json')
assert terminal['complete'] and terminal['exit_code'] == 0 and terminal['reaped']
assert terminal['actual_worker_absent'] and terminal['actual_worker_CUDA_absent']
identities = [launch['parent'], owner['child']]
for handle in identities:
    assert not Path('/proc', str(handle['pid'])).exists(), 'Owned process still present; no readout'
    try:
        os.killpg(handle['group'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('Owned group still present; no readout')
gpu_pids = {int(row.split(',')[0]) for row in subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines() if row.strip().isdigit()}
assert not gpu_pids.intersection(h['pid'] for h in identities)
complete = read(D / 'COMPLETE.json')
assert complete['complete'] and complete['failure'] is None and tuple(complete['required_seeds']) == SEEDS
assert len(complete['records']) == 3 and {row['seed'] for row in complete['records']} == set(SEEDS)
records, bindings = [], [bind(A / name) for name in ('LAUNCH.json', 'WORKER_OWNER.json', 'TERMINAL.json')]
bindings.append(bind(D / 'COMPLETE.json'))
for seed in SEEDS:
    f = D / ('seed' + str(seed)) / 'RESULT.json'
    row = read(f)
    assert row == next(x for x in complete['records'] if x['seed'] == seed)
    assert row['status'] == 'complete' and row['full_graph_used'] and not row['TEST_truth_present']
    assert row['model_parameter_buffer_state_exact'] and row['optimizer_state_exact'] and row['selected_streams_restored']
    assert row['native_args']['layers'] == 4 and row['native_args']['second_linear'] is True
    records.append(row)
    bindings.append(bind(f))
summary = {}
for role in ('train', 'valid'):
    summary[role] = {}
    for metric in ('auroc', 'nll', 'accuracy', 'brier'):
        values = [row['scores'][role][metric] for row in records]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        radius = 4.302652729911275 * sd / (3 ** .5)
        summary[role][metric] = dict(per_seed=values, mean=mean, sample_SD=sd, descriptive_seed95_interval=[mean-radius, mean+radius])
report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), complete_three_seed_baseline=True,
              fresh_selected_serving_scores=summary, original_records=records, inputs=bindings,
              terminal=terminal, exact_owned_process_group_and_CUDA_absence_verified=True,
              scientific_scope='Exploratory original Tolokers split0; descriptive optimizer-seed uncertainty only.',
              no_GNNM_comparison_performed=True, original18_and_centered3_unopened=True,
              accuracy_advantage_established=False, TEST_truth_accessed=False,
              checkpoint_or_prediction_arrays_loaded=False, automatic_retry=False)
O.mkdir(exist_ok=False)
(O / 'READOUT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
rows = ['# Completed stronger BSNN baseline', '', 'Three complete fresh fits of the fixed authentic Cayley d2/f32/L4 model. Scores come from fresh reconstruction of each selected model. This report opens no incomplete GNNM family.', '', '| Seed | Epochs | Selected epoch | TRAIN AUROC | VALID AUROC | VALID NLL | Seconds |', '| --- | --- | --- | --- | --- | --- | --- |']
for row in records:
    rows.append('| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.1f} |'.format(row['seed'], row['epochs_completed'], row['selected_epoch'], row['scores']['train']['auroc'], row['scores']['valid']['auroc'], row['scores']['valid']['nll'], row['complete_attempt_seconds']))
rows.extend(['', 'Mean VALID AUROC: {:.6f}; sample SD: {:.6f}.'.format(summary['valid']['auroc']['mean'], summary['valid']['auroc']['sample_SD']), '', 'The intervals describe three optimizer seeds on an already used development split. They do not establish graph-level generalization or a sharing mechanism. General NSD and native BSNN differ in geometry, conditioning, normalization, objective and stochastic serving. No shared-model result is compared here.', '', 'All failures and earlier weaker configurations remain in the ledger. Checkpoints and raw predictions remain server-only.'])
(O / 'REPORT.md').write_text('\n'.join(rows) + '\n')
print(json.dumps(dict(summary=summary, terminal=terminal, records=[{k: r[k] for k in ('seed', 'epochs_completed', 'selected_epoch', 'parameters', 'complete_attempt_seconds')} for r in records], readout_path=str(O))))

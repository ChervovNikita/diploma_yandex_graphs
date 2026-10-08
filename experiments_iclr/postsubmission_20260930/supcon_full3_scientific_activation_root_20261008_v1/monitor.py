"""Observe training progress and owned process identity without quality scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = ROOT / 'experiments_iclr/postsubmission_20260930'
ACTIVATION = PHASE / 'supcon_full3_scientific_activation_root_20261008_v1'
FAMILY = PHASE / 'canonical_SupCon_full3_gpu77_execution_root_20261008_v1'
assert Path.cwd() == ROOT and socket.gethostname() == 'peptide'


def process(saved):
    try:
        text = Path('/proc', str(saved['pid']), 'stat').read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rfind(')') + 2:].split()
    result = dict(pid=saved['pid'], start_ticks=int(fields[19]), state=fields[0])
    result['identity_matches'] = result['start_ticks'] == saved['start_ticks']
    return result


result = dict(UTC=datetime.now(timezone.utc).isoformat(), quality_scores_read=False,
              family_created=FAMILY.exists(), cells=[])
launch = ACTIVATION / 'LAUNCH.json'
if launch.exists():
    saved = json.loads(launch.read_text())
    result['launch'] = saved
    result['actual_owner'] = process(saved['owner']) if saved.get('owner') else None
status = FAMILY / 'CELL_STATUS.json'
if status.exists():
    for row in json.loads(status.read_text()):
        summary = {key: row[key] for key in ('seed', 'status', 'physical_gpu_uuid',
                   'owner', 'memory_wait_seconds', 'inclusive_fit_seconds', 'exit_code',
                   'actual_exit_and_reap', 'error') if key in row}
        if row.get('owner'):
            summary['actual_child'] = process(row['owner'])
        progress = FAMILY / 'cells' / ('supcon_eq2_' + str(row['seed'])) / 'PROGRESS.json'
        if progress.exists():
            values = json.loads(progress.read_text())
            summary['progress'] = {key: values[key] for key in ('epoch', 'steps',
                                   'complete_epochs', 'execution_accounting') if key in values}
        result['cells'].append(summary)
closure = FAMILY / 'CLOSURE.json'
if closure.exists():
    values = json.loads(closure.read_text())
    result['closure'] = {key: values[key] for key in ('family_accounted',
                        'all_new_fits_complete', 'fatal_error', 'inclusive_family_seconds',
                        'family_hard_cap_exceeded')}
result['GPU_inventory'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
                                                  '--format=csv,noheader,nounits'], text=True).strip()
if result.get('actual_owner') is None or result.get('closure', {}).get('fatal_error'):
    log = ACTIVATION / 'owner.log'
    if log.exists():
        result['owner_failure_log'] = log.read_text()[-4000:]
print(json.dumps(result))

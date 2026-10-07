"""Read only nonpredictive resource/terminal metadata on the exact allocation."""
import datetime,hashlib,json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
gpus=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
assert len(gpus)==1 and gpus[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ROOT=PHASE/'internal_BE_molhiv_be_init_contrastive6101_resource_execution_20261007_v1'
prefix='receipts/be_init_contrastive_6101_LIVE'
outer=['exit_code','wall_seconds','user_CPU_seconds','system_CPU_seconds','peak_RSS_bytes','input_blocks','output_blocks','supervisor_owner','resource_only','predictive_scores_opened','automatic_retry','resource_weights_used_as_fit_start']
supervisor=['outer_PID','outer_start_ticks','resource_supervisor_PID','resource_supervisor_start_ticks']
live=['schema','job_sha256','supervisor_pid','supervisor_start_ticks','hard_seconds','active_compute_seconds','cleanup_grace_seconds']
resource=['schema','passed','seed','status','exit_code','inclusive_seconds','peak_GPU_bytes','sampled_peak_driver_process_GPU_bytes','driver_samples','monitor_errors','parent_sampled_peak_child_RSS_bytes','work','worker_measurements','worker_resource_sha256','worker_failure_sha256','worker_failure_measurements','qualifier_manifest_sha256','job_sha256','failure_type','predictive_scores_closed','automatic_retry','TEST_access','scientific_fit_or_quality_admission','full_fit_time_forecast_claimed','terminal_receipt_sha256','total_output_storage_bytes','observed_log_live_terminal_storage_bytes']
terminal=['schema','status','exit_code','child_pid','child_start_ticks','reap_observed','inclusive_seconds','absolute_hard_seconds','active_compute_seconds','cleanup_grace_seconds','cap_exceeded','owned_worker_completed','live_receipt_sha256','predictive_scores_closed','automatic_retry','failure_type']
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':{},'resource_weights_or_prediction_values_transferred':False,'free_GPU_bytes':int(gpus[0].split(',')[1].strip())*1024**2}
for name,keys in [('OUTER_TERMINAL.json',outer),('SUPERVISOR_OWNER.json',supervisor),(prefix+'.json',live),(prefix+'_RESOURCE.json',resource),(prefix+'_TERMINAL.json',terminal)]:
    file=ROOT/name
    if not file.is_file():continue
    data=file.read_bytes();record=json.loads(data)
    out['records'][name]={'sha256':hashlib.sha256(data).hexdigest(),'selected_nonpredictive_metadata':{k:record[k] for k in keys if k in record}}
out['terminal_observed']=all(name in out['records'] for name in ('OUTER_TERMINAL.json',prefix+'_RESOURCE.json',prefix+'_TERMINAL.json'))
WIKI=PHASE/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
running=json.loads((WIKI/'RUNNING_CELL.json').read_text())
progress=json.loads((WIKI/'fits/outputs'/running['cell']/'PROGRESS.json').read_text())
out['WikiCS_post']={'cell':running['cell'],'arm':running['arm'],'seed':running['seed'],'progress':progress,'jobs_modified':False}
print(json.dumps(out))

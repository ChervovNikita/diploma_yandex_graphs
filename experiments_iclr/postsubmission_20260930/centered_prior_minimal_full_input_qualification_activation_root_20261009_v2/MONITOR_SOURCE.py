"""Read only exact owned geometry handles, work counts and status; no scores."""
from pathlib import Path
import datetime,json,socket,subprocess
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
A=P/'centered_prior_minimal_full_input_qualification_activation_root_20261009_v2'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def observed(saved):
 try:s=Path('/proc',str(saved['pid']),'stat').read_text()
 except FileNotFoundError:return dict(pid=saved['pid'],present=False,expected_birth=saved['start_ticks'])
 f=s[s.rfind(')')+2:].split()
 actual=dict(pid=saved['pid'],start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),state=f[0],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
 assert all(actual[k]==saved[k] for k in ['pid','start_ticks','group','session','boot_id']), 'Owned PID identity mismatch'
 actual['present']=True;return actual
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scores_opened=False,TEST_truth_accessed=False,other_processes_changed=False)
launch=json.loads((A/'LAUNCH.json').read_text());d['parent']=observed(launch['parent'])
if (A/'WORKER_OWNER.json').exists():
 owner=json.loads((A/'WORKER_OWNER.json').read_text());d['worker']=observed(owner['child'])
 d['worker_owner']=owner
if (A/'TERMINAL.json').exists():
 d['terminal']=json.loads((A/'TERMINAL.json').read_text())
cfg=json.loads((A/'RELEASE.json').read_text());q=Path(cfg['output_directory']);assert q.is_relative_to(P)
d['output_present']=q.exists();d['components']=[]
if q.exists():
 for path in sorted(q.glob('**/RESULT.json')):
  v=json.loads(path.read_text());d['components'].append(dict(path=str(path.relative_to(q)),**{k:v[k] for k in ['kind','base_seed','seed','status','last_attempted_epoch','epochs_completed','selected_epoch','counters','failure'] if k in v}))
 d['complete_file_present']=(q/'QUALIFICATION.json').exists()
 if (q/'QUALIFICATION.json').exists():
  v=json.loads((q/'QUALIFICATION.json').read_text());d['qualification']={k:v[k] for k in ['status','qualification_passed','centered_prior_probe_passed','centered_update_qualified','model_parameter_buffer_restore_exact','optimizer_restore_exact','training_RNG_restore_exact','finite_reconstructed_serving','counters','peak_CUDA_allocated_bytes','complete_attempt_seconds','failure'] if k in v}
d['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(d))

"""Observe fixed FiLM lifecycle; fetch scores only after whole pipeline closure."""
import argparse,base64,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r"""
import base64,datetime,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1')
def read(name):
 p=H/name;return json.loads(p.read_text()) if p.exists() else None
launch=read('QUEUE_LAUNCH_V1.json');end=read('PIPELINE_END_V1.json')
progress=read('actual_family_v1/PROGRESS.json') or {}
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=launch,queue_present=bool(launch and Path('/proc',str(launch['queue_PID'])).exists()),pipeline_start=read('PIPELINE_START_V1.json'),pipeline_end=end,physical_closure=read('PHYSICAL_ACQUISITION_CLOSURE_V1.json'),fit_owner_start=read('fit_owner_v1/OWNER_START.json'),fit_owner_end=read('fit_owner_v1/OWNER_END.json'),readout_owner_end=read('readout_owner_v1/OWNER_END.json'),fit_progress={k:progress[k] for k in ('arm','seed','member','update','completed_units') if k in progress},partial_predictive_metrics_read=False,TEST_access=False)
if end and end['complete_pipeline']:
 owner=value['readout_owner_end'];assert owner['complete_analysis'] and owner['child_pid_absent'] and owner['owned_cuda_pid_absent']
 full=read('complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json');assert full['complete'] and full['banks']==9 and full['all45_calibration_attempts_terminal_before_interpretation']
 value['complete_analysis']=full
elif end:
 value['failure_log']=(H/'queue_outer_stderr_v1.log').read_text()[-2500:]
 for stage in ('fit','readout'):
  q=H/(stage+'_owner_v1/stderr.log')
  if q.exists():value[stage+'_failure_log']=q.read_text()[-2500:]
print(json.dumps(value))
"""
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True);args=ap.parse_args()
 f=H/f'QUEUE_METADATA_V{args.version}.json';assert not f.exists()
 spec=importlib.util.spec_from_file_location('payload_transport',P/'publication/publish_exact_inventory_v19.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'FiLM-lifecycle','literal-route','whole-closure-only'])]
 r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr[-2000:]
 value=json.loads(r.stdout);complete=value.pop('complete_analysis',None)
 if complete:
  q=H/'COMPLETE_ANALYSIS_SUMMARY_V1.json'
  if q.exists():assert json.loads(q.read_text())==complete
  else:q.write_text(json.dumps(complete,indent=2)+'\n')
  value['complete_analysis_fetched']=str(q)
 f.write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({k:value.get(k) for k in ('UTC','queue_present','pipeline_end','physical_closure','fit_progress','complete_analysis_fetched','partial_predictive_metrics_read','TEST_access')}))
if __name__=='__main__':main()

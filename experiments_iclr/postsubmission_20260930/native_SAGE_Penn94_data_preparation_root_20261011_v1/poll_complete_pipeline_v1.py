"""Fetch the complete frozen Penn94 release only after whole pipeline closure."""
import argparse,base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r'''
import base64,datetime,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'native_SAGE_Penn94_data_preparation_root_20261011_v1'
def read(n):
 p=H/n;return json.loads(p.read_text()) if p.exists() else None
def pinned(item):
 p=Path(item['path']);assert p.resolve().is_relative_to(H)
 data=p.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 return json.loads(data)
end=read('PIPELINE_WATCH_END_V1.json')
v=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pipeline_end=end,watcher_present=Path('/proc/627089').exists(),partial_predictive_metrics_read=False)
v['stages']={n:read(n+'_v1_owner/OWNER_END.json') for n in ('calibrate','confirm')}
if end:
 assert end['complete'] and end['extra_native_fits']==end['extra_native_forwards']==0
 acquire=read('native_acquisition_v1_owner/OWNER_END.json');cal=pinned(end['calibration_owner']);confirm=pinned(end['confirmation_owner'])
 for owner in (acquire,cal,confirm):
  assert owner['exit_code']==0 and not owner['timed_out'] and all(owner[k] for k in ('direct_child_wait','child_pid_absent','owned_cuda_pid_absent','stage_terminal'))
 native=pinned(end['native']);calibrations=pinned(end['calibration']);full=pinned(end['confirmation'])
 assert native['terminal'] and native['banks']==70 and native['body_fit_units']==100 and len(native['records'])==70
 assert calibrations['terminal'] and calibrations['scalar_endpoints']==90 and len(calibrations['records'])==90
 assert full['terminal'] and full['complete_CONFIRM_release'] and full['CONFIRM_truth_access'] and len(full['rows'])==90
 v['complete_confirmation']=full
 v['final_source_config']={n:read(n) for n in ('QUALIFICATION_CONFIG_V2.json','ACQUISITION_ADMISSION_V1.json','CALIBRATION_ADMISSION_V1.json','CONFIRM_ADMISSION_V1.json')}
 v['closure_counts']={'banks':70,'body_attempts':100,'CAL_attempts':90,'CONFIRM_rows':90,'native_failures':sum(r['status']!='selected_finite' for r in native['records']),'calibration_failures':sum(r['status']!='finite_fixed_endpoint' for r in calibrations['records'])}
elif not v['watcher_present']:
 for name in ('pipeline_watcher_stderr.log','pipeline_watcher_stdout.log','calibrate_pipeline_stderr.log','confirm_pipeline_stderr.log'):
  p=H/name
  if p.exists():v[name]=p.read_text()[-4000:]
print(json.dumps(v))
'''
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True);a=ap.parse_args()
 output=H/f'COMPLETE_PIPELINE_OBSERVATION_V{a.version}.json';assert not output.exists()
 spec=importlib.util.spec_from_file_location('penn_complete_transport',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'Penn-complete-pipeline','literal-route','whole-fixed-release'])]
 r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr[-2000:]+r.stdout[-2000:]
 v=json.loads(r.stdout);full=v.pop('complete_confirmation',None);bindings=v.pop('final_source_config',None)
 if full:
  f=H/'COMPLETE_CONFIRM_V1.json'
  if f.exists():assert json.loads(f.read_text())==full
  else:f.write_text(json.dumps(full,indent=2)+'\n')
  q=H/'FINAL_SOURCE_CONFIG_BINDINGS_V1.json'
  if q.exists():assert json.loads(q.read_text())==bindings
  else:q.write_text(json.dumps(bindings,indent=2)+'\n')
  v['complete_confirmation_fetched']=str(f);v['final_source_config_fetched']=str(q)
 output.write_text(json.dumps(v,indent=2)+'\n')
 print(json.dumps({k:v.get(k) for k in ('UTC','watcher_present','pipeline_end','closure_counts','complete_confirmation_fetched','partial_predictive_metrics_read')}))
if __name__=='__main__':main()

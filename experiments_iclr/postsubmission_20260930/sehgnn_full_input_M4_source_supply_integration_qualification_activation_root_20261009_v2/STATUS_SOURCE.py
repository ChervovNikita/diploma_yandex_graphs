from pathlib import Path
import json,socket,subprocess,datetime,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_full_input_M4_source_supply_integration_qualification_activation_root_20261009_v2'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def observed(h):
 f=Path('/proc',str(h['pid']),'stat')
 if not f.exists():return dict(pid=h['pid'],present=False)
 raw=f.read_text();v=raw[raw.rfind(')')+2:].split();assert int(v[19])==h['start_ticks'];return dict(pid=h['pid'],present=True,birth=int(v[19]),state=v[0])
starter=json.loads((A/'STARTER.json').read_text());launch=json.loads((A/'LAUNCH.json').read_text());d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=observed(starter['parent']),child=observed(launch['child']),launch=launch,terminal_present=(A/'TERMINAL.json').exists())
if d['terminal_present']:
 terminal=json.loads((A/'TERMINAL.json').read_text());d['terminal']=terminal
 assert terminal['child_reaped'] and terminal['original_pid_absent']
 assert not d['parent']['present'] and not d['child']['present']
 groups={starter['parent']['group'],launch['child']['group']}
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:v=(q/'stat').read_text()
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  assert int(v[v.rfind(')')+2:].split()[2]) not in groups
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines():assert row.strip() not in {str(h) for h in groups}
 out=Path(json.loads((A/'RELEASE.json').read_text())['output_directory']);report=out/'QUALIFICATION_REPORT.json';d['qualification']=json.loads(report.read_text());d['qualification_binding']=dict(path=str(report),bytes=report.stat().st_size,sha256=hashlib.sha256(report.read_bytes()).hexdigest())
else:
 d['worker_tail']=(A/'worker.stderr').read_text()[-1800:]
print(json.dumps(d))

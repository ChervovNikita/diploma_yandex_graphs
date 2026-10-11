"""Retain compact complete-family results; full per-class evidence stays remote."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

HERE=Path(__file__).resolve().parent
P=HERE.parent
remote=r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';E=P/'Physics_dropout_pilot_acquisition_root_20261011_v1';H=P/'Physics_dropout_pilot_readout_after_whole_acquisition_root_20261011_v1'
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def closed(p):
 a=json.loads(p.read_text());assert a['status']=='COMPLETE_EXIT_0' and a['exit_code']==0 and a['direct_child_wait'] and a['child_pid_absent'] and a['owned_cuda_pid_absent'];return a
owners={str(p.relative_to(P)):closed(p) for p in [E/'owner_v1/OWNER_END.json',H/'calibrate_owner_v1/OWNER_END.json',H/'score_owner_v1/OWNER_END.json']}
watch=json.loads((H/'WATCH_END.json').read_text());assert watch['complete'] and watch['all69states_before42CAL_before_onewholeDEV']
path=H/'score_v1/DEVELOPMENT_REPORT.json';bound=receipt(path);assert bound==watch['report'] and bound['sha256']=='53e7d8d14a008e16bc47c7ae9947f69823303ff6bd8888622adf04c7380cbf2b'
a=json.loads(path.read_text());assert a['complete'] and len(a['records'])==42 and a['actual_scalar_updates']==21000 and not a['survivor_subset']
panel=json.loads((E/'acquisition_v1/PANEL.json').read_text());assert panel['complete'] and (panel['banks'],panel['selected_states'],panel['exports'])==(42,69,42)
summary={k:v for k,v in a.items() if k not in ('comparisons','records')}
summary['full_report_receipt']=bound
summary['records']=[{k:v for k,v in r.items() if k!='classes'} for r in a['records']]
summary['comparisons']=[]
for r in a['comparisons']:
 row={k:v for k,v in r.items() if k not in ('class_quality_deltas','per_block_error_diagnostics')}
 row['per_block_error_diagnostics']=[{k:v for k,v in d.items() if k!='classes'} for d in r['per_block_error_diagnostics']]
 summary['comparisons'].append(row)
summary['all_per_class_results_retained_in_full_server_report']=True
summary['owners']=owners
summary['watch']=watch
summary['publication_scope']='Every arm, whole-population metric, candidate contrast, frozen gate and cost; class detail retained under bound full report.'
data=(json.dumps(summary,indent=2)+'\n').encode()
print(json.dumps(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())))
'''
spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cmd=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,'repo','GPU','complete_summary'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',cmd]
r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60)
metadata=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr)
if r.returncode:
    metadata['stdout']=r.stdout
    (HERE/'TRANSPORT.json').write_text(json.dumps(metadata,indent=2)+'\n')
    raise SystemExit(r.returncode)
row=json.loads(r.stdout);data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
with (HERE/'SUMMARY.json').open('xb') as f:f.write(data)
metadata.update(bytes=row['bytes'],sha256=row['sha256'])
(HERE/'TRANSPORT.json').write_text(json.dumps(metadata,indent=2)+'\n')
a=json.loads(data)
print(json.dumps(dict(complete=a['complete'],full_report=a['full_report_receipt'],gates=a['gates'],arms=[dict(arm=r['arm'],accuracy=r['metrics']['accuracy_pct']['mean'],mean_member=r['metrics']['mean_member_accuracy_pct']['mean']) for r in a['arm_summaries']]),indent=2))

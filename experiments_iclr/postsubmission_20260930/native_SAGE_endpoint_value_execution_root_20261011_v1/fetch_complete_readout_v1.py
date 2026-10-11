"""Fetch only compact, completely closed endpoint-study evidence."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_SAGE_endpoint_value_execution_root_20261011_v1')
endpath=H/'READER_WATCH_END_V1.json'
if not endpath.exists():print(json.dumps(dict(status='READER_NOT_CLOSED',predictive_outputs_read=False)))
else:
 end=json.loads(endpath.read_text())
 if not end.get('complete_analysis'):print(json.dumps(dict(status='READER_FAILURE_RETAINED',end=end,predictive_outputs_read=False)))
 else:
  summarypath=H/'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json';b=summarypath.read_bytes()
  assert hashlib.sha256(b).hexdigest()==end['summary']['sha256']
  summary=json.loads(b);assert summary['complete'] and summary['banks']==15 and summary['optimizer_acquisition_bundles']==24 and summary['all75_calibration_attempts_terminal_before_interpretation']
  rows=[]
  paths=[endpath,summarypath,H/'fit_owner_v1/OWNER_END.json',H/'COMPLETE_READER_INPUT_SPEC_V1.json']
  paths += [Path(row['path']) for row in summary['partitions']]
  expected={row['path']:row for row in summary['partitions']}
  for p in paths:
   assert p.resolve().is_relative_to(H) and p.stat().st_size<2000000
   b=p.read_bytes();row=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
   if str(p) in expected:assert row==expected[str(p)]
   rows.append(dict(relative=str(p.relative_to(H)),data=base64.b64encode(b).decode(),**row))
  print(json.dumps(dict(status='COMPLETE_CLOSED_READOUT',files=rows)))
'''

def main():
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'complete-readout','literal-route','compact-only'])]
    r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stderr[-2500:]
    obj=json.loads(r.stdout)
    if obj['status']!='COMPLETE_CLOSED_READOUT':print(json.dumps(obj));return
    D=H/'complete_closed_evidence_v1';D.mkdir(exist_ok=False)
    for row in obj['files']:
        rel=Path(row['relative']);assert not rel.is_absolute() and '..' not in rel.parts
        b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
        p=D/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    (D/'FETCH_RECEIPT.json').write_text(json.dumps(dict(status=obj['status'],file_receipts=[{k:v for k,v in row.items() if k!='data'} for row in obj['files']],exit_code=r.returncode),indent=2)+'\n')
    summary=json.loads((D/'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json').read_text())
    print(json.dumps(dict(status=obj['status'],files=len(obj['files']),frozen_policy=summary['frozen_policy']),indent=2))

if __name__=='__main__':main()

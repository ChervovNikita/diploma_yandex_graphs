"""Fetch compact whole FiLM evidence after successful9/18/45 ownership closure."""
import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
H=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_PyG_FiLM_WikiCS_comparator_execution_root_20261011_v1')
def pinned(item):
 p=Path(item['path']);assert p.resolve().is_relative_to(H)
 data=p.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];return data
end=json.loads((H/'PIPELINE_END_V1.json').read_text());assert end['complete_pipeline'] and end['status']=='COMPLETE_EXIT_0'
for field in ('fit_owner_end','readout_owner_end'):
 v=json.loads(pinned(end[field]));assert v['exit_code']==0 and v['child_pid_absent'] and v['owned_cuda_pid_absent']
summary=json.loads((H/'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json').read_text())
assert summary==payload['summary'] and summary['complete'] and summary['banks']==9 and summary['optimizer_acquisition_bundles']==summary['native_body_fit_records']==18 and summary['all45_calibration_attempts_terminal_before_interpretation']
files=[]
for item in summary['partitions']:
 p=Path(item['path']);assert p.parent==H/'complete_analysis_v1' and item['bytes']<2000000
 files.append(dict(name=p.name,receipt=item,data=base64.b64encode(pinned(item)).decode()))
print(json.dumps(dict(complete=True,files=files)))
'''
def main():
 summary=json.loads((H/'COMPLETE_ANALYSIS_SUMMARY_V1.json').read_text());out=H/'complete_closed_evidence_v1';assert not out.exists()
 spec=importlib.util.spec_from_file_location('FiLM_complete_fetch_transport',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'complete-FiLM-partitions','literal-route','all-fixed-arms'])]
 r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps({'summary':summary}).encode())).decode(),capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr[-1000:]
 v=json.loads(r.stdout);assert v['complete'];out.mkdir();receipts=[]
 for item in v['files']:
  data=base64.b64decode(item['data']);pin=item['receipt'];assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256']
  p=out/item['name'];assert p.parent==out and not p.exists();p.write_bytes(data);receipts.append(dict(local_path=str(p),original_receipt=pin))
 (out/'FETCH_RECEIPTS.json').write_text(json.dumps({'complete':True,'files':receipts,'native_arrays_or_checkpoints_copied':False},indent=2)+'\n')
 print(json.dumps({'complete':True,'partitions':len(receipts),'bytes':sum(item['original_receipt']['bytes'] for item in receipts)}))
if __name__=='__main__':main()

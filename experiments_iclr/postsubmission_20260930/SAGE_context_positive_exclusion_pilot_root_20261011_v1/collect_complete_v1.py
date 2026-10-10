import argparse,base64,hashlib,json,shlex,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True);version=parser.parse_args().version
receipt=H/f'COMPLETE_READOUT_COLLECTION_V{version}.json';assert version>0 and not receipt.exists()
REMOTE="import base64,hashlib,json,socket,subprocess\nfrom pathlib import Path\nassert socket.gethostname()=='anogena-2-0';assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']\nH=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/SAGE_context_positive_exclusion_pilot_root_20261011_v1')\nread=lambda q:json.loads(q.read_text()) if q.exists() else None\nend=read(H/'READER_JOB_END_V1.json');value=dict(owner_end=read(H/'OWNER_END.json'),reader_start=read(H/'READER_EXECUTION_START_V1.json'),reader_end=end,reader_stderr=(H/'reader.stderr.log').read_text()[-5000:] if (H/'reader.stderr.log').exists() else None,files=[])\nif end and end['success']:\n A=H/'complete_analysis_v1'\n for q in sorted(A.glob('*.json')):\n  if q.name=='COMPLETE_REPORT.json' or q.name.startswith(('SAGE_ARM_DETAILS','CALIBRATION_FIT_DETAILS','CHECKPOINT_CUSTODY','ARCHIVE_CUSTODY','REUSED_OOF_ARCHIVE_CUSTODY')):continue\n  assert q.stat().st_size<2000000\n  data=q.read_bytes();value['files'].append(dict(name=q.name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data=base64.b64encode(data).decode()))\nprint('CONTEXT_COMPLETE_JSON='+json.dumps(value))\n"
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -B -c '+shlex.quote(REMOTE)]
r=subprocess.run(args,text=True,capture_output=True,timeout=55)
assert r.returncode==0,r.stderr[-2500:]
v=json.loads(next(x.split('=',1)[1] for x in r.stdout.splitlines() if x.startswith('CONTEXT_COMPLETE_JSON=')))
files=v.pop('files');folder=H/'complete_analysis_v1'
if files:
 folder.mkdir(exist_ok=True)
 for row in files:
  q=folder/row['name'];assert q.parent==folder;data=base64.b64decode(row.pop('data'));assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  if q.exists():assert q.read_bytes()==data
  else:q.write_bytes(data)
v['collected_files']=files
receipt.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(owner_end=v['owner_end'],reader_end=v['reader_end'],reader_stderr=v['reader_stderr'],files=len(files))))

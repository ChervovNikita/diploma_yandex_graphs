"""Fetch small, stored evidence after complete-study closure."""
import argparse,base64,hashlib,json,shlex,subprocess,zlib
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
mode,identity=sys.argv[1:]
if mode=='typed':
 r=subprocess.run(['python3','-B',str(P/'closed_family_scalar_reader_source_20261010_v2/reader.py'),'--study','typed42','--output-id',identity],capture_output=True,text=True)
 if r.returncode:
  print(json.dumps(dict(reader_exit_code=r.returncode,reader_stdout=r.stdout,reader_stderr=r.stderr)));sys.exit(r.returncode)
 print(r.stdout)
else:
 H=P/'pubmed_factorized_I4_reference_source_20261010_v1';A=P/'factorized_I4_reference_root_activation_20261010_v1'
 start=json.loads((A/'comparison'/'STARTER.json').read_text());assert not Path('/proc',str(start['PID'])).exists()
 done=json.loads((H/'comparison'/'FAMILY_COMPLETE.json').read_text());assert done['complete']
 c=json.loads((H/'comparison'/'cells'/'factorized_I4_complete_comparison'/'COMPLETE.json').read_text())
 assert c['complete'] and c['complete_family'] and c['TEST_access'] is False
 files=[]
 for key in ('comparison','costs'):
  row=c[key];q=Path(row['path']);assert q.resolve().is_relative_to(P);b=q.read_bytes();assert len(b)<2000000 and hashlib.sha256(b).hexdigest()==row['sha256']
  files.append(dict(path=str(q.relative_to(P)),bytes=len(b),sha256=row['sha256'],base64_zlib=base64.b64encode(zlib.compress(b,9)).decode()))
 print(json.dumps(dict(pending=False,complete_family=True,files=files)))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',choices=('typed','factorized'),required=True);p.add_argument('--id',required=True);args=p.parse_args()
 assert args.id.isidentifier()
 target=HERE/args.id;target.mkdir(exist_ok=False)
 argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.study)+' '+shlex.quote(args.id)]
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
 (target/'TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
 assert r.returncode==0,r.stderr
 v=json.loads(r.stdout)
 if v['pending']:
  print(json.dumps(v));return
 rows=[dict(v,path=v['output']+'/CLOSED_SCALAR_EVIDENCE.json')] if args.study=='typed' else v['files']
 saved=[]
 for row in rows:
  b=zlib.decompress(base64.b64decode(row['base64_zlib']));assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
  q=target/'fetched'/row['path'];assert q.resolve().is_relative_to((target/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);saved.append(str(q))
 (target/'CUSTODY.json').write_text(json.dumps(dict(study=args.study,saved=saved,whole_family_closed_before_quality=True,TEST_access=False),indent=2)+'\n')
 print(json.dumps(dict(study=args.study,pending=False,saved=saved)))

if __name__=='__main__':main()

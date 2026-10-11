"""Download the fixed PyG GitHub release inside the authorized repository."""
from datetime import datetime,timezone
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE=r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,urllib.request,time,datetime,zipfile
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);out=repo/'experiments_iclr/postsubmission_20260930/native_SAGE_GitHub_confirmation_root_20261011_v1'
raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
assert not raw.is_symlink() and not out.is_symlink()
path=raw/'github.npz';partial=raw/'github.npz.partial';receipt=out/'PROVIDER_BYTE_RECEIPT_V1.json'
assert not path.exists() and not partial.exists() and not receipt.exists()
start=time.monotonic();digest=hashlib.sha256();count=0
with urllib.request.urlopen(payload['url'],timeout=45) as response:
 headers=dict(response.headers);final_URL=response.geturl();status=response.status
 with partial.open('xb') as f:
  while True:
   b=response.read(1048576)
   if not b:break
   count+=len(b);assert count<=512*1024**2 and time.monotonic()-start<300
   f.write(b);digest.update(b)
with zipfile.ZipFile(partial) as z:
 names=z.namelist();assert set(names)=={'features.npy','target.npy','edges.npy'},names
partial.rename(path)
r=dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset='PyG GitHub',provider_revision=payload['provider_revision'],
 url=payload['url'],final_URL=final_URL,HTTP_status=status,response_headers=headers,path=str(path),bytes=count,sha256=digest.hexdigest(),
 NPZ_member_names=names,seconds=time.monotonic()-start,arrays_or_targets_decoded=False,predictive_scores_read=False,model_fits=0,scientific_fit_admitted=False)
receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
'''

def main():
    payload=dict(url='https://graphmining.ai/datasets/ptg/github.npz',provider_revision='76ff9c2ce18c8cebf52122b57e2aeadce9793d10')
    admission=dict(UTC=datetime.now(timezone.utc).isoformat(),operation='Fixed published PyG release byte acquisition only',
        selection='Existing complete E quality result and scoped history finding: no GitHub-dataset outcome in3587 permitted authored notes; no universal absence claim',
        target='Full released graph, prospective label-free60/20/20split; no official mask claim',
        metric='Micro-F1 equals accuracy for exhaustive single-label binary predictions',
        dataset_selected_before_new_target_or_outcome_access=True,source_and_inputs_audit_pending=True,
        data_decoding_or_fits_admitted=False,**payload)
    file=H/'PROVIDER_BYTE_ADMISSION_V1.json';assert not file.exists();file.write_text(json.dumps(admission,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,R,UUID,'fixed-GitHub-provider-bytes'])]
    r=m.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(),capture_output=True,text=True,timeout=350)
    (H/'PROVIDER_BYTE_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stdout[-2000:]
    v=json.loads(r.stdout);(H/'PROVIDER_BYTE_RECEIPT_V1.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps({k:v[k] for k in ('complete','UTC','bytes','sha256','seconds','model_fits','scientific_fit_admitted')}))

if __name__=='__main__':main()

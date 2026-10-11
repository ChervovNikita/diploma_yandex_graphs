"""Acquire the reserved public Penn94 blobs; no decode, fit or score."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REVISION='82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6'
REMOTE=r'''
import base64,zlib,json,sys,pathlib,socket,subprocess,hashlib,urllib.request,time,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15).stdout.splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
out=phase/'native_SAGE_Penn94_data_preparation_root_20261011_v1';raw=out/'raw'
raw.mkdir(parents=True,exist_ok=True)
rows=[];start=time.monotonic()
for item in payload['files']:
 path=raw/pathlib.PurePosixPath(item['path']).name
 assert not path.is_symlink()
 cached=path.exists()
 if cached:data=path.read_bytes()
 else:
  url='https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/'+payload['revision']+'/'+item['path']
  with urllib.request.urlopen(url,timeout=45) as response:data=response.read(item['bytes_from_tree']+1)
 assert len(data)==item['bytes_from_tree']
 blob=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
 assert blob==item['git_blob_sha']
 if not cached:path.write_bytes(data)
 rows.append({'path':str(path.relative_to(repo)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha':blob,'already_cached':cached})
receipt={'complete':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':payload['revision'],'files':rows,'seconds':time.monotonic()-start,'payloads_decoded':False,'model_fits':0,'prediction_or_TEST_metrics_read':False,'population':'published Penn94 split0 by index, reserved before new outcomes'}
target=out/'PINNED_PAYLOAD_RECEIPT_V1.json'
assert not target.exists()
target.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

def main():
    metadata=json.loads((PHASE/'shared_message_closest_prior_v1/new_complete_tasks_v1/PAYLOAD_ENDPOINT_METADATA.json').read_text())
    rows=[r for r in metadata if r['path'] in ('data/facebook100/Penn94.mat','data/splits/fb100-Penn94-splits.npy')]
    assert len(rows)==2
    admission={'UTC':datetime.now(timezone.utc).isoformat(),'population':'Penn94 published split0',
               'selection_basis':'First split index and saved source reservation; no new predictive outcomes.',
               'known_project_history':'Saved source-only opportunity and scoped ledger/CONTEXT/README/tracker checks; no universal absence-of-access claim.',
               'operation':'Pinned public byte acquisition only; subsequent decoding/role construction and fitting remain separate.',
               'revision':REVISION,'files':rows,'scientific_execution_admitted':False}
    (HERE/'PAYLOAD_ACQUISITION_ADMISSION_V1.json').write_text(json.dumps(admission,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('penn94_payload_transport',PHASE/'publication/publish_exact_inventory_v19.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',
          '/Users/alex/.ssh/mlspace__private_key_anogena.txt','-p','2222',
          'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
          shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID,'pinned-Penn94-data-only'])]
    result=transport.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps({'revision':REVISION,'files':rows}).encode())).decode(),capture_output=True,text=True,timeout=120)
    receipt={'UTC':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,
             'stdout':result.stdout,'stderr':result.stderr,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'ACQUISITION_TRANSPORT_V1.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert result.returncode==0,result.stderr[-1000:]
    value=json.loads(result.stdout);assert value['complete']
    (HERE/'PINNED_PAYLOAD_RECEIPT_V1.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()

"""Acquire official arxiv release bytes inside the scientific repository only."""
from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

HERE = Path(__file__).resolve().parent
P = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE = r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,time,hashlib,urllib.request,datetime,zipfile
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
repo=pathlib.Path(sys.argv[1]);phase=repo/'experiments_iclr/postsubmission_20260930'
out=phase/'native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1'
out.mkdir(exist_ok=True);raw=out/'raw';raw.mkdir(exist_ok=True)
assert not out.is_symlink() and not raw.is_symlink()
receipt=out/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V1.json'
assert not receipt.exists()
archive=raw/'arxiv.zip';partial=raw/'arxiv.zip.partial'
assert not archive.exists() and not partial.exists()
start=time.monotonic();digest=hashlib.sha256();count=0
request=urllib.request.Request(payload['requested_url'],headers={'User-Agent':'GNNM-research-official-provider'})
with urllib.request.urlopen(request,timeout=45) as response:
 final_url=response.geturl();status=response.status;headers=dict(response.headers)
 with partial.open('xb') as f:
  while True:
   data=response.read(1048576)
   if not data:break
   count+=len(data)
   assert count<=1024**3 and time.monotonic()-start<300
   digest.update(data);f.write(data)
assert count>0
with zipfile.ZipFile(partial) as z:
 members=[dict(name=i.filename,bytes=i.file_size,CRC=i.CRC) for i in z.infolist()]
 assert all(not pathlib.PurePosixPath(i['name']).is_absolute() and '..' not in pathlib.PurePosixPath(i['name']).parts for i in members)
 assert any(i['name'].endswith('/raw/node-feat.csv.gz') for i in members)
partial.rename(archive)
r=dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset='ogbn-arxiv',
 official_provider_revision=payload['provider_revision'],canonical_url=payload['canonical_url'],requested_url=payload['requested_url'],final_url=final_url,
 https_transport_amendment=True,HTTP_status=status,response_headers=headers,path=str(archive),bytes=count,sha256=digest.hexdigest(),
 members=members,seconds=time.monotonic()-start,archive_decoded=False,targets_or_predictive_outputs_read=False,model_fits=0,scientific_fit_admitted=False)
receipt.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
'''

def main():
    expected = HERE/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V1.json'
    assert not expected.exists()
    canonical = 'http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
    payload = dict(canonical_url=canonical, requested_url=canonical.replace('http:', 'https:', 1),
                   provider_revision='61e9784ca76edeaa6e259ba0f836099608ff0586')
    admission = dict(UTC=datetime.now(timezone.utc).isoformat(), operation='official release byte acquisition only',
        source_specification='ogbn_arxiv_official_provider_SAGE_confirmation_source_audit_20261011_v1/PROVIDER_PREPARATION_SPECIFICATION.json',
        reason='Complete endpoint-conditioned WikiCS evidence supports a fixed independent graph confirmation with capable same-information references.',
        HTTPS_transport_amendment='Same official SNAP archive path; record response redirects/bytes/hash; no alternate release.',
        unused_history_audit_pending=True, population_labels_or_predictive_scores_used_for_selection=False,
        data_decoding_or_fits_admitted=False, **payload)
    (HERE/'OFFICIAL_ARCHIVE_BYTE_ADMISSION_V1.json').write_text(json.dumps(admission,indent=2)+'\n')
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
          '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
          shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,REPO,UUID,'official-arxiv-bytes-only'])]
    r=mod.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode(),capture_output=True,text=True,timeout=350)
    (HERE/'OFFICIAL_ARCHIVE_BYTE_TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stderr[-1500:]
    receipt=json.loads(r.stdout);assert receipt['complete']
    expected.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('complete','UTC','bytes','sha256','seconds','model_fits','scientific_fit_admitted')}))

if __name__=='__main__':main()

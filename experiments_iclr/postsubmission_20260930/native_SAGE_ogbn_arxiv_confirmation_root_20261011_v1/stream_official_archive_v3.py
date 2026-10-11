"""Stream the exact official archive through this Mac, retaining no local payload."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

H=Path(__file__).resolve().parent
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
REMOTE=r'''
import json,sys,socket,subprocess,pathlib,time,hashlib,datetime,zipfile,termios,tty
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
repo=pathlib.Path(sys.argv[1]);out=repo/'experiments_iclr/postsubmission_20260930/native_SAGE_ogbn_arxiv_confirmation_root_20261011_v1'
raw=out/'raw';raw.mkdir(parents=True,exist_ok=True)
assert not out.is_symlink() and not raw.is_symlink()
archive=raw/'arxiv.zip';partial=raw/'arxiv.zip.partial';receipt=out/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V3.json'
assert not archive.exists() and not partial.exists() and not receipt.exists()
expected=int(sys.argv[3]);assert expected==83058288
prior=termios.tcgetattr(0);tty.setraw(0);print('GNNM_OFFICIAL_ARXIV_STREAM_READY',flush=True)
start=time.monotonic();count=0;d=hashlib.sha256()
with partial.open('xb') as f:
 while count<expected:
  b=sys.stdin.buffer.read(min(1048576,expected-count))
  assert b and time.monotonic()-start<300
  count+=len(b);d.update(b);f.write(b)
termios.tcsetattr(0,termios.TCSANOW,prior)
with zipfile.ZipFile(partial) as z:
 members=[dict(name=i.filename,bytes=i.file_size,CRC=i.CRC) for i in z.infolist()]
 assert all(not pathlib.PurePosixPath(i['name']).is_absolute() and '..' not in pathlib.PurePosixPath(i['name']).parts for i in members)
 assert any(i['name'].endswith('/raw/node-feat.csv.gz') for i in members)
partial.rename(archive)
r=dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset='ogbn-arxiv',official_provider_revision='61e9784ca76edeaa6e259ba0f836099608ff0586',
 canonical_url='http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip',requested_url='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip',
 transport='Official HTTPS bytes streamed through Mac via literal authorized SSH; no Mac payload file',path=str(archive),bytes=count,sha256=d.hexdigest(),members=members,
 seconds=time.monotonic()-start,archive_decoded=False,targets_or_predictive_outputs_read=False,model_fits=0,scientific_fit_admitted=False)
receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
'''

def main():
    final=H/'OFFICIAL_ARCHIVE_ACQUISITION_RECEIPT_V3.json';assert not final.exists()
    url='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
    (H/'OFFICIAL_ARCHIVE_BYTE_ADMISSION_V3.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),
        operation='Stream exact official HTTPS release bytes through Mac after remote HTTP and TLS timeouts',requested_url=url,
        no_Mac_payload_file=True,no_data_decoding_or_fit_admission=True,prior_attempts_preserved=['V1','V2']),indent=2)+'\n')
    start=time.monotonic();d=hashlib.sha256();count=0
    with urllib.request.urlopen(url,timeout=45) as response:
        headers=dict(response.headers);expected=int(response.headers['Content-Length']);assert expected==83058288
        cmd=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,R,UUID,str(expected)])]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as p:
            ready=p.stdout.readline().strip();assert ready==b'GNNM_OFFICIAL_ARXIV_STREAM_READY',ready
            while True:
                block=response.read(1048576)
                if not block:break
                count+=len(block);assert count<=expected and time.monotonic()-start<300
                d.update(block);p.stdin.write(block);p.stdin.flush()
            assert count==expected
            p.stdin.close();p.stdin=None
            stdout,stderr=p.communicate(timeout=45)
            (H/'OFFICIAL_ARCHIVE_BYTE_TRANSPORT_V3.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=p.returncode,
                stdout=stdout.decode(),stderr=stderr.decode(),headers=headers,final_URL=response.geturl(),HTTP_status=response.status,
                bytes=count,sha256=d.hexdigest(),seconds=time.monotonic()-start,no_Mac_payload_file=True),indent=2)+'\n')
            assert p.returncode==0,stdout.decode()[-2500:]
            receipt=json.loads(stdout);assert receipt['complete'] and receipt['sha256']==d.hexdigest() and receipt['bytes']==count
            receipt['HTTPS_response_headers']=headers;receipt['local_stream_sha256']=d.hexdigest();receipt['total_stream_seconds']=time.monotonic()-start
            final.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps({k:receipt[k] for k in ('complete','UTC','bytes','sha256','total_stream_seconds','model_fits','scientific_fit_admitted')}))

if __name__=='__main__':main()

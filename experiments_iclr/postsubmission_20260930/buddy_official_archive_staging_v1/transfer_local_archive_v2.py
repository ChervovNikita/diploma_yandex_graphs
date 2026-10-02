"""Transfer unopened official archive bytes after the remote TLS acquisition failed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'local_transfer_v2'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CODE = '''
from datetime import datetime, timezone
import hashlib,json,pathlib,subprocess,sys,time
repo=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; size=int(sys.argv[3]); digest=sys.argv[4]
if repo.resolve()!=repo: raise RuntimeError('Actual repository required')
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if gpu.stdout.strip().splitlines()!=[expected]: raise RuntimeError('Authorized one-GPU route differs')
root=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True)
if pathlib.Path(root.stdout.strip()).resolve()!=repo: raise RuntimeError('Actual project Git root differs')
folder=repo/'experiments_iclr/postsubmission_20260930/buddy_official_archive_staging_v1/root_transfer_v2'
if not folder.resolve().is_relative_to(repo): raise RuntimeError('Transfer path escapes project')
folder.mkdir(parents=True,exist_ok=False)
started=time.monotonic(); received=0; actual=hashlib.sha256()
record=dict(schema='official-archive-byte-transfer-v2',UTC=datetime.now(timezone.utc).isoformat(),GPU_uuid=expected,
 dataset_members_opened=False,test_members_opened=False,archive_extracted=False,GPU_compute=False)
try:
 with (folder/'collab.zip').open('xb') as stream:
  while True:
   block=sys.stdin.buffer.read(1024*1024)
   if not block: break
   received+=len(block)
   if received>size: raise RuntimeError('Transfer exceeds declared size')
   actual.update(block);stream.write(block)
 if received!=size or actual.hexdigest()!=digest: raise RuntimeError('Archive byte identity differs')
 record.update(status='OFFICIAL_ARCHIVE_STAGED_MEMBERS_UNOPENED',archive_sha256=digest,archive_bytes=received)
except Exception as error:
 record.update(status='TRANSFER_FAILED',error_type=type(error).__name__,error=str(error));raise
finally:
 record['seconds']=time.monotonic()-started
 (folder/'STAGING_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\\n')
 print(json.dumps(record,sort_keys=True))
'''


def main():
    receipt_path = LOCAL / 'TRANSFER_RECEIPT.json'
    if receipt_path.exists():
        raise RuntimeError('Single-use transfer already attempted')
    archive = LOCAL / 'collab.zip'
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    count = archive.stat().st_size
    if count != 121625147:
        raise RuntimeError('Archive size differs from official HEAD receipt')
    with zipfile.ZipFile(archive) as packed:
        members = []
        for member in packed.infolist():
            p = Path(member.filename)
            if p.is_absolute() or '..' in p.parts:
                raise RuntimeError('Unsafe official archive path')
            members.append(dict(name=member.filename, bytes=member.file_size,
                                compressed_bytes=member.compress_size, CRC=member.CRC))
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),
                   URL='https://snap.stanford.edu/ogb/data/linkproppred/collab.zip',
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   HTTP_headers_sha256=hashlib.sha256((LOCAL / 'HTTP_HEADERS.txt').read_bytes()).hexdigest(),
                   archive_sha256=digest.hexdigest(), archive_bytes=count, members=members,
                   dataset_members_opened=False, heldout_pairs_or_labels_opened=False,
                   archive_extracted=False, ssh_destination=LOGIN, GPU_compute=False)
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', LOGIN]
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', CODE,
                          REPO, GPU, str(count), digest.hexdigest()])
    with archive.open('rb') as stream:
        result = subprocess.run([*ssh, command], stdin=stream,
                                capture_output=True, text=True, timeout=300)
    receipt.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   status='TRANSFER_COMPLETE' if result.returncode == 0 else 'TRANSFER_FAILED')
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'archive_bytes', 'archive_sha256', 'stdout')}, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

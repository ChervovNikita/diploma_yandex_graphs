"""Stream only the two named audit payloads to verified private server custody."""
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = '61c5ca6d6c744aa0a5d2ed7d0aa886fa57e839f5673da4b8c9c7698ae258d5bb'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
FOLDER = REPO + '/author_archives/mac_cleanup_20261003_large_audits_v1'
SSH = ['ssh', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222', '-o', 'IdentitiesOnly=yes',
       '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no', '-o', 'StrictHostKeyChecking=yes',
       'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']

STAGE = r'''
import base64,hashlib,json,os,stat,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);repo=Path(p['repo']);folder=Path(p['folder'])
assert str(repo)=='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert repo.resolve()==repo and subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'],text=True).strip()==str(repo)
assert folder==repo/'author_archives/mac_cleanup_20261003_large_audits_v1' and folder.parent.resolve()==folder.parent
os.umask(0o077);folder.mkdir(mode=0o700,exist_ok=True);assert folder.resolve()==folder
assert set(p['files'])=={'EXACT_FILE_MANIFEST.json','recover_large_audits.py','RECOVERY.txt','verify_large_audit_archive.py'}
for name,row in p['files'].items():
    data=base64.b64decode(row['base64'],validate=True);assert hashlib.sha256(data).hexdigest()==row['sha256'];path=folder/name
    if path.exists():assert path.resolve()==path and path.is_file() and path.read_bytes()==data
    else:
        with path.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
ignore=folder/'.gitignore'
if not ignore.exists():ignore.write_text('*\n')
assert subprocess.run(['git','-C',str(repo),'check-ignore','--quiet',str(folder/'large_audits.tar.gz')]).returncode==0
print(json.dumps({'small_recovery_files_staged':4,'folder':str(folder)}))
'''

RECEIVE = r'''
from datetime import datetime,timezone
import hashlib,json,os,shutil,stat,subprocess,sys
from pathlib import Path
expected=EXPECTED_VALUE
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');folder=repo/'author_archives/mac_cleanup_20261003_large_audits_v1'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert repo.resolve()==repo and subprocess.check_output(['git','-C',str(repo),'rev-parse','--show-toplevel'],text=True).strip()==str(repo)
assert folder.resolve()==folder;os.umask(0o077)
for name,digest in expected.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
partial=folder/'large_audits.tar.gz.partial';final=folder/'large_audits.tar.gz';h=hashlib.sha256();count=0
with partial.open('xb') as out:
    for block in iter(lambda:sys.stdin.buffer.read(8*1024*1024),b''):
        out.write(block);h.update(block);count+=len(block)
    out.flush();os.fsync(out.fileno())
verification=json.loads(subprocess.check_output(['python3','-B',str(folder/'verify_large_audit_archive.py'),str(partial)],text=True))
assert verification['archive_sha256']==h.hexdigest() and verification['archive_bytes']==count
assert verification['manifest_sha256']==expected['EXACT_FILE_MANIFEST.json']
assert verification['recovery_hashes']=={name:expected[name] for name in ['recover_large_audits.py','RECOVERY.txt']}
os.link(partial,final,follow_symlinks=False);partial.unlink();os.chmod(final,0o400)
manifest=json.loads((folder/'EXACT_FILE_MANIFEST.json').read_bytes());scratch=folder/'restore_verification_v1'
restored=json.loads(subprocess.check_output(['python3','-B',str(folder/'recover_large_audits.py'),str(final),str(scratch)],text=True))
observed=set()
for directory,names,filenames in os.walk(scratch,followlinks=False):
    for name in names:assert stat.S_ISDIR((Path(directory)/name).lstat().st_mode)
    for name in filenames:
        path=Path(directory)/name;assert stat.S_ISREG(path.lstat().st_mode);observed.add(str(path.relative_to(scratch)))
assert observed=={row['phase_relative_path'] for row in manifest['files']}
for row in manifest['files']:
    path=scratch/row['phase_relative_path'];s=path.lstat();assert s.st_size==row['bytes'] and stat.S_IMODE(s.st_mode)==row['mode'] and s.st_mtime_ns==row['mtime_ns']
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==row['sha256']
assert restored['restored_files']==2 and restored['restored_bytes']==33537166
assert scratch==folder/'restore_verification_v1' and scratch.resolve()==scratch;shutil.rmtree(scratch)
verification.update(archive_path=str(final),archive_mode='0400',repo=str(repo),GPU_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac',
    recovery_test='BOTH_RESTORED_PATHS_BYTES_SHA256_MODES_MTIMES_VERIFIED',recovery_scratch_removed=True,
    archive_exclusive_creation_no_overwrite=True,verified_at_UTC=datetime.now(timezone.utc).isoformat())
with (folder/'CUSTODY_VERIFICATION.json').open('xb') as out:out.write((json.dumps(verification,indent=2)+'\n').encode());out.flush();os.fsync(out.fileno())
print(json.dumps(verification,indent=2))
'''


class Sink:
    def __init__(self, output): self.output=output; self.sha=hashlib.sha256(); self.bytes=0
    def write(self, block): self.sha.update(block); self.bytes+=len(block); return self.output.write(block)
    def flush(self): self.output.flush()


class Source:
    def __init__(self, source): self.source=source; self.sha=hashlib.sha256(); self.bytes=0
    def read(self, size):
        block=self.source.read(size); self.sha.update(block); self.bytes+=len(block); return block


def identity(path):
    s=path.lstat(); return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def main():
    payload=dict(repo=REPO,folder=FOLDER,files={});small={}
    for name in ['EXACT_FILE_MANIFEST.json','recover_large_audits.py','RECOVERY.txt','verify_large_audit_archive.py']:
        data=(HERE/name).read_bytes(); small[name]=data
        payload['files'][name]=dict(base64=base64.b64encode(data).decode('ascii'),sha256=hashlib.sha256(data).hexdigest())
    assert payload['files']['EXACT_FILE_MANIFEST.json']['sha256']==MANIFEST_SHA
    manifest=json.loads(small['EXACT_FILE_MANIFEST.json'])
    assert [Path(row['path']).name for row in manifest['files']]==['FILE_INVENTORY.json','REMOTE_GIT_TREE.z']
    subprocess.run(SSH+['python3 -B -c '+shlex.quote(STAGE)],input=json.dumps(payload).encode(),capture_output=True,check=True)
    remote=RECEIVE.replace('EXPECTED_VALUE',repr({name:row['sha256'] for name,row in payload['files'].items()}))
    with (HERE/'TRANSFER.log').open('w') as log:
        child=subprocess.Popen(SSH+['python3 -B -c '+shlex.quote(remote)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
        sink=Sink(child.stdin)
        try:
            with gzip.GzipFile(fileobj=sink,mode='wb',compresslevel=1,mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,mode='w|',bufsize=1024*1024) as packed:
                    for name in ['EXACT_FILE_MANIFEST.json','recover_large_audits.py','RECOVERY.txt']:
                        data=small[name];info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;packed.addfile(info,io.BytesIO(data))
                    for row in manifest['files']:
                        path=Path(row['path']);assert path.resolve()==path;before=identity(path)
                        info=tarfile.TarInfo('files/'+path.name);info.size=row['bytes'];info.mode=0o600
                        with path.open('rb') as input_file:checked=Source(input_file);packed.addfile(info,checked)
                        assert identity(path)==before and checked.sha.hexdigest()==row['sha256'] and checked.bytes==row['bytes']
            child.stdin.close();output=child.stdout.read();assert child.wait()==0,'Server offload failed; inspect TRANSFER.log'
        except BaseException:
            try:child.stdin.close()
            except Exception:pass
            child.wait();raise
    remote_result=json.loads(output);assert remote_result['archive_sha256']==sink.sha.hexdigest() and remote_result['archive_bytes']==sink.bytes
    receipt=dict(schema='gnnm-large-audit-offload-receipt-v1',status='VERIFIED_SERVER_ARCHIVE_AND_FULL_RECOVERY_LOCAL_COPIES_RETAINED',
                 server=SSH[-1]+':2222',manifest_sha256=MANIFEST_SHA,exact_file_count=2,original_bytes=manifest['bytes'],
                 remote_verification=remote_result,no_large_local_archive_created=True,files_removed=0,
                 updated_at_UTC=datetime.now(timezone.utc).isoformat())
    (HERE/'OFFLOAD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(remote_result,indent=2))


if __name__=='__main__':main()

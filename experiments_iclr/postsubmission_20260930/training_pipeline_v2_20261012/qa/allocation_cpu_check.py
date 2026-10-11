"""Stage immutable engineering fixtures and run them on allocation CPU only."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

PACKAGE = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
REMOTE = r'''
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
transport=repo/'experiments_iclr/postsubmission_20260930/training_pipeline_v2_20261012/qa'/(sys.argv[1]+'_transport')
encoded=(transport/'payload.b64').read_text()
assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))
root=repo/'experiments_iclr/postsubmission_20260930/training_pipeline_v2_20261012/qa'/payload['label']
assert root.resolve().is_relative_to(repo.resolve()) and not root.exists()
root.mkdir(parents=True)
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
 target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(root/'SOURCE_INVENTORY.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in payload['files']],indent=2)+'\n')
tmp=root/'tmp';tmp.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         GNNM_REFERENCE_PROJECT=str(repo/'experiments_iclr/postsubmission_20260930'))
tick=time.monotonic()
child=subprocess.Popen([str(repo/'.venv/bin/python'),'-B','-m','unittest','discover','-s','tests','-p',payload['pattern'],'-v'],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
fields=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()
(root/'START.json').write_text(json.dumps({'child_pid':child.pid,'child_start_ticks':int(fields[19]),'parent_pid':os.getpid(),'engineering_only':True,'scientific_training_runs':0})+'\n')
try:
 stdout,stderr=child.communicate(timeout=300)
except subprocess.TimeoutExpired:
 child.kill();stdout,stderr=child.communicate();stderr+='\nCPU engineering timeout retained.\n'
value={'exit_code':child.returncode,'wall_seconds':time.monotonic()-tick,'scientific_training_runs':0,'engineering_only':True,'stdout':stdout,'stderr':stderr}
(root/'RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
sys.exit(child.returncode)
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    if not args.label.replace("_", "").isalnum():
        raise ValueError("Use a plain immutable label")
    out = PACKAGE / "qa" / args.label
    if out.exists():
        raise FileExistsError(out)
    files = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "qa" in path.relative_to(PACKAGE).parts:
            continue
        raw = path.read_bytes()
        files.append({"path": str(path.relative_to(PACKAGE)), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "data": base64.b64encode(raw).decode()})
    # The pinned native model is an engineering fixture, not a trained run.
    source = PACKAGE.parent / "common_wrapper_native_source_20261010_v1/models.py"
    raw = source.read_bytes()
    files.append({"path": "fixtures/native_models.py", "bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()})
    for relative in ("docs/MATRIX_AXES_DRAFT.json", "docs/NODE_PROVIDER_ROLE_CATALOGUE.json", "start_massive.sh"):
        raw = (PACKAGE / relative).read_bytes()
        files.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "data": base64.b64encode(raw).decode()})
    arg = base64.b64encode(zlib.compress(json.dumps({"label": args.label, "files": files,
                                                  "pattern": args.pattern}).encode())).decode()
    prefix = ["ssh", "-tt", "-p", "2222", "-i", "/Users/alex/.ssh/mlspace__private_key_anogena.txt",
           "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
           "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
           "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"]
    out.mkdir()
    (out / "SOURCE_INVENTORY.json").write_text(json.dumps([{k:v for k,v in r.items() if k!='data'} for r in files], indent=2)+"\n")
    transfer = r'''
import hashlib,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/training_pipeline_v2_20261012/qa')
root=base/(sys.argv[1]+'_transport');target=root/'payload.b64';offset=int(sys.argv[2])
if offset==0:
 assert not root.exists();root.mkdir(parents=True)
else:assert target.stat().st_size==offset
with target.open('ab') as stream:stream.write(sys.argv[3].encode())
print(target.stat().st_size)
'''
    for offset in range(0, len(arg), 24000):
        command = prefix + [shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-c", transfer,
                                        args.label, str(offset), arg[offset:offset+24000]])]
        chunk = subprocess.run(command, capture_output=True, text=True, timeout=45)
        (out / ("TRANSFER_" + str(offset) + ".json")).write_text(json.dumps({"offset":offset,
            "exit_code":chunk.returncode,"stdout":chunk.stdout,"stderr":chunk.stderr},indent=2)+"\n")
        if chunk.returncode:
            print(chunk.stderr)
            raise SystemExit(chunk.returncode)
    cmd = prefix + [shlex.join(["/usr/bin/python3", "-I", "-S", "-B", "-c", REMOTE,
                                args.label, hashlib.sha256(arg.encode()).hexdigest()])]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=330)
    (out / "TRANSPORT.json").write_text(json.dumps({"UTC": datetime.now(timezone.utc).isoformat(),
        "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2)+"\n")
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()

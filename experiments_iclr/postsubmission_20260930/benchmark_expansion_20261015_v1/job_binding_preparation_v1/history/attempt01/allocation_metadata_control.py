"""Narrow source transfer and metadata-only preparation for this new package."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
helper_path = PHASE/'reference_wrapper_redesign_20261013_v1/smoke_20261014/remote_ops.py'
spec = importlib.util.spec_from_file_location('_reviewed_allocation_transport', helper_path)
guarded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guarded)
REMOTE = guarded.REMOTE_ROOT+'/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/job_binding_preparation_v1'
SOURCE_NAMES = ('prepare_descriptors.py', 'allocation_metadata_control.py', 'README.md', 'PREPARATION_MANIFEST.json')


def upload():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in SOURCE_NAMES:
            archive.add(HERE/name, arcname=name, recursive=False)
    packet = buffer.getvalue()
    code = guarded.GUARD + '''import io,sys,tarfile,hashlib,json
from pathlib import Path
root=Path(ROOT_LITERAL)
root.mkdir(exist_ok=True)
print("READY",flush=True)
packet=sys.stdin.buffer.read(SIZE_LITERAL)
assert len(packet)==SIZE_LITERAL and hashlib.sha256(packet).hexdigest()==HASH_LITERAL
rows=[]
with tarfile.open(fileobj=io.BytesIO(packet),mode="r:gz") as archive:
 for item in archive.getmembers():
  assert item.isfile() and item.name in NAMES_LITERAL
  target=root/item.name
  assert target.resolve().is_relative_to(root) and not target.is_symlink()
  raw=archive.extractfile(item).read()
  if target.exists():
   assert target.read_bytes()==raw, "Existing preparation source differs; preserve it"
  else:
   with target.open("xb") as stream: stream.write(raw)
  rows.append({"path":str(target),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
print(json.dumps({"guard":"passed","source_files":rows,"fits_started":0}),flush=True)
'''
    code = code.replace('ROOT_LITERAL', repr(REMOTE)).replace('SIZE_LITERAL', str(len(packet))).replace('HASH_LITERAL', repr(hashlib.sha256(packet).hexdigest())).replace('NAMES_LITERAL', repr(SOURCE_NAMES))
    proc = subprocess.Popen(guarded.SSH+['stty raw -echo; python3 -B -c '+shlex.quote(code)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    first = proc.stdout.readline()
    if first.strip() != b'READY':
        proc.terminate()
        out, err = proc.communicate(timeout=15)
        raise RuntimeError((first+out+err).decode(errors='replace'))
    out, err = proc.communicate(input=packet, timeout=60)
    if proc.returncode:
        raise RuntimeError((out+err).decode(errors='replace'))
    return json.loads(out.decode().strip())


def run_metadata_preparation():
    code = 'import runpy,sys\nsys.argv=["prepare_descriptors.py","--output-name","generated_v1"]\nrunpy.run_path('+repr(REMOTE+'/prepare_descriptors.py')+',run_name="__main__")\n'
    return json.loads(guarded.call(code, timeout=300))


if __name__ == '__main__':
    transfer = upload()
    (HERE/'TRANSFER_RECEIPT.json').write_text(json.dumps(transfer, sort_keys=True, indent=2)+'\n')
    try:
        result = run_metadata_preparation()
    except Exception as exc:
        (HERE/'PREPARATION_FAILURE.json').write_text(json.dumps({'stage':'metadata_only_preparation','error':str(exc),'fits_started':0},indent=2)+'\n')
        raise
    (HERE/'READINESS.json').write_text(json.dumps(result['readiness'], sort_keys=True, indent=2)+'\n')
    (HERE/'REMOTE_READINESS_RECEIPT.json').write_text(json.dumps(result['readiness_receipt'], sort_keys=True, indent=2)+'\n')
    print(json.dumps({'status':result['readiness']['status'], 'descriptors':result['readiness']['descriptor_count'], 'remote_output':result['readiness']['remote_output'], 'fits_started':0}))

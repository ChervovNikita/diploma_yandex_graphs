"""Read exact owned compact metadata once; leave payload tensors on the server."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
DEST = HERE / 'compact_retrieval_v1'
RESULT_SHA = '942992e040cb1cc820965707bf7c20b071efce5c29857dee03ab0a2eca06d421'


def main():
    assert not DEST.exists()
    code = r'''from pathlib import Path
import base64,hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
uuids=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
stage=phase/'citeseer_frame_mechanism_analysis_execution_stage_20261005_v1'
output=phase/'citeseer_frame_mechanism_analysis_execution_20261005_v1'
paths={
 'stage_EXECUTION_START.json':stage/'EXECUTION_START.json',
 'stage_EXECUTION_RESULT.json':stage/'EXECUTION_RESULT.json',
 'stage_ROOT_ADMISSION.json':stage/'ROOT_ADMISSION.json',
 'stage_stdout.log':stage/'stdout.log',
 'stage_stderr.log':stage/'stderr.log',
 'START.json':output/'START.json',
 'RESULTS.json':output/'RESULTS.json',
 'STRATIFIED_MRR.json':output/'STRATIFIED_MRR.json',
 'FRAME_MAP_DESCRIPTORS.json':output/'FRAME_MAP_DESCRIPTORS.json',
}
def sha(raw):return hashlib.sha256(raw).hexdigest()
result=json.loads((output/'RESULTS.json').read_text())
assert sha((output/'RESULTS.json').read_bytes())=='942992e040cb1cc820965707bf7c20b071efce5c29857dee03ab0a2eca06d421'
files=[];total=0
for name,path in paths.items():
 assert path.resolve(strict=True).is_relative_to(phase.resolve(strict=True))
 raw=path.read_bytes();total+=len(raw)
 assert len(raw)<=250000 and total<=1000000
 if name in result.get('files',{}):
  ref=result['files'][name]
  assert ref['bytes']==len(raw) and ref['sha256']==sha(raw)
 files.append(dict(name=name,remote_relative=str(path.relative_to(phase)),bytes=len(raw),sha256=sha(raw),base64=base64.b64encode(raw).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),GPU_UUIDs=uuids,read_only=True,TEST_access=False,files=files,total_bytes=total)))
'''
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    completed = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code])],
                               capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr
    package = json.loads(completed.stdout)
    assert package['read_only'] is True and package['TEST_access'] is False
    decoded = {}
    for row in package['files']:
        assert Path(row['name']).name == row['name']
        raw = base64.b64decode(row['base64'], validate=True)
        assert len(raw) == row['bytes']
        assert hashlib.sha256(raw).hexdigest() == row['sha256']
        decoded[row['name']] = raw
    assert hashlib.sha256(decoded['RESULTS.json']).hexdigest() == RESULT_SHA
    result = json.loads(decoded['RESULTS.json'])
    for name in ('STRATIFIED_MRR.json', 'FRAME_MAP_DESCRIPTORS.json'):
        ref = result['files'][name]
        assert len(decoded[name]) == ref['bytes']
        assert hashlib.sha256(decoded[name]).hexdigest() == ref['sha256']
    execution = json.loads(decoded['stage_EXECUTION_RESULT.json'])
    assert execution['status'] == 'COMPLETE' and execution['exit_code'] == 0
    assert execution['mechanism_RESULTS_sha256'] == RESULT_SHA
    DEST.mkdir()
    for name, raw in decoded.items():
        with (DEST / name).open('xb') as stream:
            stream.write(raw)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),
                   files=[{key: value for key, value in row.items() if key != 'base64'} for row in package['files']],
                   total_bytes=package['total_bytes'],
                   hostname=package['hostname'], GPU_UUIDs=package['GPU_UUIDs'],
                   remote_read_only=True, hashes_authenticated=True, TEST_access=False,
                   transport_stderr=completed.stderr,
                   tensors_checkpoints_predictions_downloaded=False)
    with (DEST / 'RETRIEVAL_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: receipt[key] for key in ('total_bytes', 'hashes_authenticated', 'hostname', 'TEST_access')}))


if __name__ == '__main__':
    main()

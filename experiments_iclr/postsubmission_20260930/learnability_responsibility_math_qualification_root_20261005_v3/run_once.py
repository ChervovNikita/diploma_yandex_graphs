"""Run one source-bound CPU fixture on the authorized allocation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PREP = HERE.parent / 'learnability_weighted_graph_responsibility_operator_20261005_v2'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
RUNTIME = REPO + '/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python'
EXPECTED_SOURCE = 'fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3'
REMOTE = r'''
from pathlib import Path
import hashlib,json,socket,subprocess,sys,types,traceback
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.load(sys.stdin)
for name in ['operator','fixture']:
 assert hashlib.sha256(payload[name].encode()).hexdigest()==payload[name+'_sha256']
output=repo/'experiments_iclr/postsubmission_20260930/learnability_responsibility_math_qualification_root_20261005_v3/CPU_RESULT.json'
assert not output.exists()
sys.path.insert(0,str(repo/'.venv/lib/python3.11/site-packages'))
module=types.ModuleType('response_operator_fixture');sys.modules[module.__name__]=module
fixture=types.ModuleType('root_math_fixture');sys.modules[fixture.__name__]=fixture
try:
 exec(compile(payload['operator'],'source_bound_response_operator.py','exec'),module.__dict__)
 assert module.SOURCE_RELEASED is False
 exec(compile(payload['fixture'],'root_math_fixture.py','exec'),fixture.__dict__)
 value=fixture.run(module)
 import torch
 assert Path(torch.__file__).resolve().is_relative_to(repo)
 value['torch_module_path']=torch.__file__
except Exception as error:
 value={'status':'FAIL_CPU_MATH_FIXTURE','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'scientific_data_access':False,'TEST_access':False,'model_fits':0,'predictive_evidence':False}
value.update(operator_sha256=payload['operator_sha256'],fixture_sha256=payload['fixture_sha256'],hostname=socket.gethostname())
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')
print(json.dumps(value))
raise SystemExit(0 if value['status']=='PASS_CPU_MATH_FIXTURE_ONLY' else 1)
'''


def main():
    assert not (HERE / 'EXECUTION_RELEASE.json').exists(), 'One execution identity only'
    manifest = json.loads((PREP / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        data = (PREP / row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    source = (PREP / 'response_operator.py').read_text()
    fixture = (HERE / 'check_math.py').read_text()
    payload = {'operator': source, 'fixture': fixture,
               'operator_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'fixture_sha256': hashlib.sha256(fixture.encode()).hexdigest()}
    assert payload['operator_sha256'] == EXPECTED_SOURCE
    release = {'UTC': datetime.now(timezone.utc).isoformat(),
               'purpose': 'One CPU tensor engineering fixture: balance/positivity, private partial and mixed derivative',
               'operator_sha256': payload['operator_sha256'], 'fixture_sha256': payload['fixture_sha256'],
               'source_manifest_sha256': hashlib.sha256((PREP / 'MANIFEST.json').read_bytes()).hexdigest(),
               'scientific_fits_authorized': False, 'native_model_qualified': False,
               'TEST_access': False, 'predictive_claim_authorized': False}
    (HERE / 'EXECUTION_RELEASE.json').write_text(json.dumps(release, indent=2) + '\n')
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=20', LOGIN,
               'cd ' + shlex.quote(REPO) + ' && exec ' + shlex.join([RUNTIME, '-B', '-c', REMOTE])]
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=50)
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'exit_code': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr,
               'remote_wrapper_sha256': hashlib.sha256(REMOTE.encode()).hexdigest(),
               'source_unchanged': True, 'TEST_access': False, 'scientific_fits': 0}
    (HERE / 'TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.stdout:
        value = json.loads(result.stdout)
        (HERE / 'CPU_RESULT.json').write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(value))
    else:
        print(json.dumps({'exit_code': result.returncode, 'stderr': result.stderr}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

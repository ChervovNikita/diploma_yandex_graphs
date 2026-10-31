"""Freeze the allocation-qualified common controller without launching it."""
import json
from pathlib import Path

from prepare_molecular_packets_allocation import GUARD, invoke

HERE = Path(__file__).resolve().parent / 'common_controller_20261012_v1'
REMOTE = GUARD + r'''
import importlib
qa=phase/'training_pipeline_v2_20261012/qa/cpu_controller_final_v25'
result=json.loads((qa/'RESULT.json').read_text())
assert result['exit_code']==0 and 'Ran 41 tests' in result['stderr'] and 'skipped=' not in result['stderr']
sys.path.insert(0,str(qa));master=importlib.import_module('compile_master_plan')
stage=phase/'master_controller_source_20261012_v1';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir()
inventory={row['path']:row for row in json.loads((qa/'SOURCE_INVENTORY.json').read_text())}
for name in master.CONTROLLER_FILES:
 row=inventory[name];raw=(qa/name).read_bytes()
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 target=runtime/name;target.write_bytes(raw);target.chmod(0o555 if name.endswith('.sh') else 0o444)
def receipt(path):
 raw=path.read_bytes()
 return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
bindings=master.controller_bindings(phase,str(runtime.relative_to(phase)))
remote=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin']).strip()
policy={'schema':'phase-git-rendezvous-v1','remote_name':'origin',
 'remote_url_sha256':hashlib.sha256(remote).hexdigest(),
 'max_compressed_host_bytes':8388608,'max_uncompressed_proof_bytes':67108864}
host={'allocation':{'status':'verified','hostname':socket.gethostname(),
 'gpu_uuids':['GPU-44039938-fd82-41d2-fefd-de71514e2fac'],
 'verification_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},
 'server77':{'status':'pending_live_verification_before_launch',
  'reason':'Current user authorization is allocation-only; no remote connection attempted'}}
value={'schema':'frozen-master-controller-v1','runtime_package':str(runtime.relative_to(phase)),
 'files':bindings,'engineering_QA':receipt(qa/'RESULT.json'),'passed_engineering_tests':41,
 'scientific_fits':0,'benchmark_scores':0,'phase_rendezvous':policy,'host_bindings':host,
 'start_commands_are_disabled_until_GO':True}
for name,item in [('MANIFEST.json',value),('HOST_BINDINGS.json',host),('RENDEZVOUS_POLICY.json',policy)]:
 (stage/name).write_text(json.dumps(item,indent=2,allow_nan=False)+'\n')
print(json.dumps(value))
'''


def main():
    assert not HERE.exists()
    HERE.mkdir()
    transport = invoke(REMOTE, [], HERE, 'FREEZE_TRANSPORT', 120)
    if transport['exit_code'] != 0:
        raise RuntimeError('Common-controller freeze failed; transport retained')
    value = json.loads(transport['stdout'])
    (HERE/'MANIFEST.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'frozen': True, 'runtime_package': value['runtime_package'],
                      'engineering_tests': value['passed_engineering_tests'], 'training': False}))


if __name__ == '__main__':
    main()

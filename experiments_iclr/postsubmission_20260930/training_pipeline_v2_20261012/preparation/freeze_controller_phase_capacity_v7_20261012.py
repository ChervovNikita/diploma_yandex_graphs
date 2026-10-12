"""Raise only the bounded hash-metadata transport cap; preserve controller v6."""
import base64,hashlib,json,zlib
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
BASE=Path(__file__).resolve().parent
HERE=BASE/'controller_phase_capacity_root_20261012_v7'
PACKAGE=BASE.parent
REMOTE=GUARD+r'''
from importlib import import_module
payload=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
prior_dir=phase/'master_controller_source_20261012_v6'
prior=json.loads((prior_dir/'MANIFEST.json').read_text())
stage=phase/'master_controller_source_20261012_v7';assert not stage.exists();stage.mkdir()
runtime=stage/'runtime';runtime.mkdir();unchanged=[]
def receipt(p):return {'path':str(p.relative_to(phase)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row
 target=runtime/p.name;raw=p.read_bytes()
 if p.name=='phase_rendezvous.py':
  assert raw.count(b'MAX_COMPRESSED_BYTES = 16 * 1024**2')==1
  raw=raw.replace(b'MAX_COMPRESSED_BYTES = 16 * 1024**2',b'MAX_COMPRESSED_BYTES = 32 * 1024**2')
 else:unchanged.append(row)
 target.write_bytes(raw);target.chmod(0o555 if target.suffix=='.sh' else 0o444)
assert (runtime/'phase_rendezvous.py').read_bytes()==base64.b64decode(payload['phase_source'])
qa=stage/'CPU_QA';qa.mkdir();tests=qa/'tests';tests.mkdir()
(tests/'test_phase_rendezvous.py').write_bytes(base64.b64decode(payload['test_source']))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(runtime)+':'+str(tests))
cmd=[str(repo/'.venv/bin/python'),'-B','-m','unittest',
 'test_phase_rendezvous.PhaseRendezvousTests.test_received_bytes_are_atomic_immutable_and_decompression_is_bounded',
 'test_phase_rendezvous.PhaseRendezvousTests.test_runtime_requires_plan_bound_existing_destination_and_limits',
 'test_phase_rendezvous.PhaseRendezvousTests.test_exact_roster_size_bound_qualifies_transport_without_real_results','-v']
v=subprocess.run(cmd,cwd=qa,env=env,capture_output=True,text=True,timeout=60)
result={'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'command':cmd,'fits':0,'GO':False}
(qa/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');assert v.returncode==0,v.stderr
sys.path.insert(0,str(runtime));master=import_module('compile_master_plan');rendezvous=import_module('phase_rendezvous')
assert rendezvous.MAX_COMPRESSED_BYTES==32*1024**2 and rendezvous.MAX_PROOF_BYTES==64*1024**2
policy=dict(prior['phase_rendezvous'],max_compressed_host_bytes=rendezvous.MAX_COMPRESSED_BYTES)
manifest=dict(prior,files=master.controller_bindings(phase,str(runtime.relative_to(phase))),
 runtime_package=str(runtime.relative_to(phase)),phase_rendezvous=policy,engineering_QA=receipt(qa/'RESULT.json'),
 passed_engineering_tests=3,prior_controller_manifest=receipt(prior_dir/'MANIFEST.json'),
 unchanged_controller_files=unchanged,source_scope='Only16to32MiB bounded hash metadata aggregate; prediction and learning unchanged',
 scientific_fits=0,benchmark_scores=0,start_commands_are_disabled_until_GO=True)
for name,value in [('MANIFEST.json',manifest),('RENDEZVOUS_POLICY.json',policy),('HOST_BINDINGS.json',prior['host_bindings'])]:
 (stage/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'manifest':manifest,'result':result,'all_source_changes_only_capacity_constant':True,'fits':0,'GO':False}))
'''
def main():
    assert not HERE.exists();HERE.mkdir()
    (HERE/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes())
    payload={k:base64.b64encode((PACKAGE/v).read_bytes()).decode() for k,v in
        [('phase_source','phase_rendezvous.py'),('test_source','tests/test_phase_rendezvous.py')]}
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode()
    v=invoke(REMOTE,[encoded],HERE,'FREEZE_AND_NARROW_QA',120)
    assert v['exit_code']==0,'Transport capacity engineering failure preserved'
    x=json.loads(v['stdout']);(HERE/'MANIFEST.json').write_text(json.dumps(x['manifest'],indent=2)+'\n')
    (HERE/'RESULT.json').write_text(json.dumps(x['result'],indent=2)+'\n')
    print(json.dumps({'frozen_controller':x['manifest']['runtime_package'],'passing_tests':3,'fits':0,'GO':False}))
if __name__=='__main__':main()

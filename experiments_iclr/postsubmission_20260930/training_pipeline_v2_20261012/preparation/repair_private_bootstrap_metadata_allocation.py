"""Retain failed v1; inherit passed numeric checks; repair metadata only."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import zlib
import prepare_private_bootstrap_allocation as p
from prepare_molecular_packets_allocation import GUARD,TRANSFER,invoke

NAME=p.NAME+'_metadata_repair_v3'
OUT=p.PACKAGE/'preparation'/p.LABEL
PREFIX=GUARD+r'''
import ast
transport=phase/(sys.argv[1]+'_transport');encoded=(transport/'payload.b64').read_text();assert hashlib.sha256(encoded.encode()).hexdigest()==sys.argv[2]
payload=json.loads(zlib.decompress(base64.b64decode(encoded)));stage=phase/'massive_private_bootstrap_packet_20261012_v1'
old=stage/'runtime';runtime=stage/'runtime_v3';assert not runtime.exists();shutil.copytree(old,runtime)
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def save(path,value):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');path.chmod(0o444)
row=payload['test'];raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
path=runtime/'tests/test_private_bootstrap_source.py';path.chmod(0o644);path.write_bytes(raw);path.chmod(0o444)
row=payload['compiler'];raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
compiler=runtime/'compile_private_bootstrap_matrix.py';compiler.chmod(0o644);compiler.write_bytes(raw);compiler.chmod(0o444)
prior=stage/'receipts/SYNTHETIC_CPU_SOURCE_QA_RESULT.json';qa=json.loads(prior.read_text());assert qa['exit_code']==1 and 'FAILED (errors=1)' in qa['stderr'] and 'test_finite_all_masks_source_reference_roster_and_paid_I4' in qa['stderr']
def methods(path):
 tree=ast.parse(path.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PrivateBootstrapSourceTests')
 return {n.name:ast.dump(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
original=methods(old/'tests/test_private_bootstrap_source.py');repaired=methods(path);numeric=[n for n in original if n.startswith('test_') and 'finite_all_masks' not in n]
assert len(numeric)==4 and all(original[n]==repaired[n] for n in numeric)
checked=[]
for source in sorted(old.rglob('*')):
 if not source.is_file() or source.relative_to(old).as_posix() in ('tests/test_private_bootstrap_source.py','compile_private_bootstrap_matrix.py'):continue
 copy=runtime/source.relative_to(old);assert source.read_bytes()==copy.read_bytes();checked.append({'source':receipt(source),'copy':receipt(copy)})
inherit={'schema':'private-bootstrap-CPU-numeric-four-check-inheritance-v1','prior_result':receipt(prior),'prior_scope':'four numeric/source checks passed;one pure metadata source fixture failed',
 'inherited_passed_numerical_test_names':numeric,'numerical_test_method_ASTs_identical':True,'runtime_bytes_identical_except_repaired_pure_metadata_fixture_and_compiler_tags':checked,'metadata_compiler_change':'honest architecture/one optimizer and actual initialization seeds;unchanged source adapter/numerics',
 'additional_numerical_test_runs':0,'CUDA_executions':0,'scientific_fits':0,'GO_created':False}
save(stage/'NUMERIC_QA_INHERITANCE_METADATA_REPAIR_V3.json',inherit)
save(stage/'SOURCE_INVENTORY_METADATA_REPAIR_V3.json',{'owned_runtime':[receipt(path) for path in sorted(runtime.rglob('*')) if path.is_file()],
 'pinned_original_sources':json.loads((stage/'SOURCE_BINDINGS.json').read_text()),'prior_runtime_preserved':True,'common_controller_modified':False,'source_contract':payload['contract']})
'''


def main():
 test=p.PACKAGE/'tests/test_private_bootstrap_source.py';raw=test.read_bytes()
 compiler=p.PACKAGE/'compile_private_bootstrap_matrix.py';code=compiler.read_bytes()
 payload={'test':{'data':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()},'compiler':{'data':base64.b64encode(code).decode(),'sha256':hashlib.sha256(code).hexdigest()},'contract':p.CONTRACT}
 encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
 for offset in range(0,len(encoded),24000):
  r=invoke(TRANSFER,[NAME,str(offset),encoded[offset:offset+24000]],OUT,'METADATA_REPAIR_TRANSFER_'+str(offset),60)
  if r['exit_code']!=0:raise RuntimeError(r['stderr'])
 tail=p.REMOTE[p.REMOTE.index('env=dict'):]
 tail=tail.replace("[python,'-B','-m','unittest','discover','-s','tests','-p','test_private_bootstrap_source.py','-v']", "[python,'-B','-c',\"import sys,unittest;sys.path.insert(0,'tests');import test_private_bootstrap_source as m;r=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromName('PrivateBootstrapSourceTests.test_finite_all_masks_source_reference_roster_and_paid_I4',m));raise SystemExit(not r.wasSuccessful())\"]")
 tail=tail.replace("tmp=stage/'engineering_tmp'","tmp=stage/'engineering_tmp_metadata_v3'")
 tail=tail.replace('SYNTHETIC_CPU_SOURCE_QA','METADATA_REPAIR_ONLY_QA').replace("'Ran 5 tests'","'Ran 1 test'")
 tail=tail.replace("'SOURCE_INVENTORY.json'","'SOURCE_INVENTORY_METADATA_REPAIR_V3.json'")
 tail=tail.replace("'result':receipt(stage/'receipts/METADATA_REPAIR_ONLY_QA_RESULT.json'),", "'result':receipt(stage/'receipts/METADATA_REPAIR_ONLY_QA_RESULT.json'),'four_numeric_checks_inherited':receipt(stage/'NUMERIC_QA_INHERITANCE_METADATA_REPAIR_V3.json'),'new_metadata_checks':1,'numerical_checks_repeated':0,")
 tail=tail.replace("'source_protocol_changes':payload['contract']['new_protocol']", "'source_protocol_changes':payload['contract']['new_protocol'],'serialization_adaptation':'new plain Torch TRAIN/source/seed-bound artifact retains exact float64 raw Exponential and weights;historical archive used NPZ'")
 tail=tail.replace("paths += [phase/r['path'] for r in plan['studies']]", "paths += [stage/'NUMERIC_QA_INHERITANCE_METADATA_REPAIR_V3.json']+[phase/r['path'] for r in plan['studies']]")
 r=invoke(PREFIX+tail,[NAME,hashlib.sha256(encoded.encode()).hexdigest()],OUT,'METADATA_REPAIR_TRANSPORT',600)
 for line in r['stdout'].splitlines():
  if not line.startswith('{'):continue
  value=json.loads(line)
  for row in value.pop('mirror_files',[]):
   raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['receipt']['sha256'];target=OUT/'allocation_artifacts'/row['receipt']['path'];assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  print(json.dumps(value,indent=2))
 if r['exit_code']!=0:print(r['stderr'],r['stdout'][-6000:]);raise RuntimeError('Retained metadata repair failure')


if __name__=='__main__':main()

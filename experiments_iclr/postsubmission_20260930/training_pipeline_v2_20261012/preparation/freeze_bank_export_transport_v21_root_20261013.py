"""Admit digest-only bank-export closure to the existing phase transport."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD,invoke
OUT=Path(__file__).resolve().parent/'controller_v21_bank_export_transport_root_20261013_v1'
REMOTE=GUARD+r'''
import ast,copy,importlib.util
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
base=phase/'master_controller_source_20261013_v20/MANIFEST.json';prior=json.loads(base.read_text());stage=phase/'master_controller_source_20261013_v21';assert not stage.exists();stage.mkdir();runtime=stage/'runtime';runtime.mkdir();before={}
for row in prior['files']:
 p=phase/row['path'];assert receipt(p)==row;(runtime/p.name).write_bytes(p.read_bytes());before[p.name]=p.read_bytes()
p=runtime/'phase_rendezvous.py';text=p.read_text()
old='ALLOWED_STAGES = {"search", "search_selection", "comparative", "calibration"}';assert text.count(old)==1
text=text.replace(old,'ALLOWED_STAGES = {"search", "search_selection", "comparative", "bank_export", "calibration"}',1)
old='"comparative": "model_fit", "calibration": "calibration"}';assert text.count(old)==1
text=text.replace(old,'"comparative": "model_fit", "bank_export": "ensemble_assembly", "calibration": "calibration"}',1);p.write_text(text)
def functions(raw):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a,b=functions(before[p.name]),functions(p.read_bytes());assert {k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}=={'check_proof'}
for name,raw in before.items():
 if name!='phase_rendezvous.py':assert (runtime/name).read_bytes()==raw
for q in runtime.glob('*.py'):compile(q.read_text(),str(q),'exec')
sys.path.insert(0,str(runtime))
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load(phase/'master_controller_source_20261013_v20/runtime/phase_rendezvous.py','old_bank_export_transport');new=load(p,'new_bank_export_transport')
from spec import canonical,fingerprint
digest='0'*64;key='synthetic-study/synthetic-bank';run={'kind':'ensemble_assembly','config':{'stage':'bank_export'}};runs={key:{'run':run}};placement={key:{'worker':'allocation/gpu0'}}
plan={'global_gates':{'whole_bank_exports':{'path':'synthetic-never-installed/whole_bank_exports.json','source_stages':['bank_export'],'required_for_stages':['calibration'],'required_hosts':['allocation'],'all_packet_source_run_keys':[key]}}}
try:old.gates_for(plan)
except ValueError as e:old_failure=str(e)
else:raise AssertionError('Original transport failure not reproduced')
proof={'status':'complete','host':'allocation','closed_phase':'whole_bank_exports','plan_hash':digest,'run_keys':[key],
 'results':{key:{'state':'complete','result_hash':'1'*64,'contract_hash':fingerprint(run)}},'closure_authority':'whole_assigned_host_registry','terminal_roster':True,'all_successful':True}
raw=(canonical(proof)+'\n').encode();new.check_proof(raw,plan,digest,runs,placement,'whole_bank_exports','allocation')
assert new.decode_proof(new.encode_proof(raw))==raw
bad=copy.deepcopy(proof);bad['results'][key]['state']='failed';bad['all_successful']=False
try:new.check_proof((canonical(bad)+'\n').encode(),plan,digest,runs,placement,'whole_bank_exports','allocation')
except ValueError:pass
else:raise AssertionError('Failed bank export closed')
badplan=copy.deepcopy(plan);badplan['global_gates']['whole_bank_exports']['source_stages']=['evaluation']
try:new.gates_for(badplan)
except ValueError:pass
else:raise AssertionError('Evaluation transport admitted')
badproof=copy.deepcopy(proof);badproof['predictions']=[0]
try:new.check_proof((canonical(badproof)+'\n').encode(),plan,digest,runs,placement,'whole_bank_exports','allocation')
except ValueError:pass
else:raise AssertionError('Non-digest payload admitted')
qa={'old_failure':old_failure,'bank_export_digest_roundtrip':True,'failed_bank_rejected':True,'evaluation_transport_rejected':True,'prediction_payload_rejected':True,'synthetic_memory_only':True,'proof_installed_or_transferred':False,'fits':0,'GO':False}
(stage/'BANK_EXPORT_TRANSPORT_PROOF.json').write_text(json.dumps(qa,indent=2)+'\n')
manifest=dict(prior);manifest.update(runtime_package=str(runtime.relative_to(phase)),files=[receipt(q) for q in sorted(runtime.iterdir()) if q.is_file()],prior_controller_manifest=receipt(base),bank_export_transport_proof=receipt(stage/'BANK_EXPORT_TRANSPORT_PROOF.json'),changed_AST_functions_against_v20={'phase_rendezvous.py':['check_proof']},new_scope='Cumulative v20 plus bank_export/ensemble_assembly digest-only terminal closure; no prediction/heldout payload transport.',scientific_fits=0,GO_created=False)
(stage/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
for q in runtime.iterdir():
 if q.is_file():q.chmod(0o444)
print(json.dumps({'manifest':manifest,'proof':qa}))
'''
def main():
 OUT.mkdir();(OUT/'SOURCE_SCRIPT.py').write_bytes(Path(__file__).read_bytes());r=invoke(REMOTE,[],OUT,'FREEZE',90)
 assert r['exit_code']==0,'Preserve failure'
 v=json.loads(r['stdout']);(OUT/'MANIFEST.json').write_text(json.dumps(v['manifest'],indent=2)+'\n');print(json.dumps({'runtime':v['manifest']['runtime_package'],'proof':v['proof']}))
if __name__=='__main__':main()

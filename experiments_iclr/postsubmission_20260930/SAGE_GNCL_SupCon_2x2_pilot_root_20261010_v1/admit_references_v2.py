"""Admit exactly the immutable original own-only SAGE reference banks, no forwards."""
import hashlib,json,shlex,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
REMOTE=r'''import ast,hashlib,json,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
u='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[u]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
assert not (H/'actual_family_v1').exists()
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def read(q):return json.loads(q.read_text())
bpath=P/'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
assert sha(bpath)=='bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
b=read(bpath);O=Path(b['family_root']);assert O.is_relative_to(P)
owner=read(O.parent/'OWNER_END.json');assert owner['scientific_success'] and owner['direct_child_wait'] and owner['child_pid_absent'] and owner['owned_cuda_pid_absent'] and not owner['TEST_access']
assert sha(O.parent/'OWNER_END.json')==b['owner_end_sha256']
assert sha(O/'COMPLETE_FAMILY.json')==b['complete_sha256']==owner['complete_sha256']
old=read(O/'COMPLETE_FAMILY.json');assert old['complete'] and old['groups']==old['expected_groups']==21 and old['fit_units']==old['expected_fit_units']==39 and not old['TEST_access']
assert sha(Path(b['config']))==sha(H/'CONFIG.json')==old['config_sha256']==b['config_sha256']
assert read(H/'CONFIG.json')==b['configuration']
for row in b['frozen_inputs']:
 q=R/row['path'];assert q.resolve().is_relative_to(R);assert sha(q)==row['sha256']
old_source=Path(b['family_source']);new_source=P/'native_SAGE_GNCL_SupCon_2x2_source_20261010_v1/run_family.py'
assert old_source.resolve().is_relative_to(P)
def methods(q):
 cls=next(n for n in ast.parse(q.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Family')
 return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
o,n=methods(old_source),methods(new_source)
unchanged=['make','seeds','scope','logits','metrics','progress','synchronize']
assert all(o[k]==n[k] for k in unchanged)
old_index={(r['arm'],r['seed']):r for r in old['results']};banks=[];checkpoints=[]
expected_arms={'ordinary_M1','ordinary_genuine_I4','factorized_allmap_M1','factorized_allmap_genuine_I4','shared4_unchanged'}
assert len(b['banks'])==15 and {(x['arm'],x['seed']) for x in b['banks']}=={(a,s) for a in expected_arms for s in [7301,7403,7507]}
for bank in b['banks']:
 r=old_index[bank['arm'],bank['seed']];archive=Path(bank['archive']);folder=archive.parent
 assert archive.resolve().is_relative_to(P) and sha(archive)==bank['archive_sha256']
 assert read(folder/'RESULT.json')==r
 assert len(r['fits'])==(4 if 'genuine_I4' in bank['arm'] else 1)
 assert bank['checkpoints']==[f['selected_state'] for f in r['fits']]
 for f in r['fits']:
  ck=Path(f['selected_state']);assert ck.resolve().is_relative_to(P)
  assert ck.stat().st_size==f['costs']['selected_checkpoint_bytes']
  assert read(ck.parent/'RESULT.json')==f
  checkpoints.append(dict(path=str(ck),sha256=sha(ck),bytes=ck.stat().st_size,arm=bank['arm'],seed=bank['seed'],member=f['member'],selected_step=f['selected_step'],digest_scope='first current checkpoint byte digest; historic path/size/fit metadata and archive hashes matched; no historical checkpoint digest claimed'))
 banks.append(dict(arm=bank['arm'],seed=bank['seed'],archive=str(archive),archive_sha256=sha(archive),selected_fits=len(r['fits'])))
assert len(checkpoints)==33
q=read(H/'actual_qualification_v1/QUALIFICATION.json');assert q['qualified'] and not q['VALID_quality_scored'] and not q['TEST_access']
value=dict(UTC=datetime.now(timezone.utc).isoformat(),admitted=True,reference_bindings_sha256=sha(bpath),banks=15,selected_fits=33,exact_source_config_role_seed_selector_custody_compatible=True,qualification_passed=True,TEST_access=False,new_outcomes_seen=False,no_native_forward=True,original_owner_fits_retained=39,original_complete_sha256=b['complete_sha256'],original_owner_end_sha256=b['owner_end_sha256'],configuration_sha256=sha(H/'CONFIG.json'),unchanged_native_methods=unchanged,archive_records=banks,checkpoint_records=checkpoints,current_checkpoint_digests_not_historical_claim=True)
a=H/'REFERENCE_ADMISSION.json'
if a.exists():
 prior=read(a);assert {k:v for k,v in prior.items() if k!='UTC'}=={k:v for k,v in value.items() if k!='UTC'};value=prior
else:a.write_text(json.dumps(value,indent=2)+'\n')
print('REFERENCE_JSON='+json.dumps(value))'''
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
p=subprocess.run(args,text=True,capture_output=True,timeout=90)
receipt=dict(exit_code=p.returncode,stdout=p.stdout if p.returncode else 'reference receipt collected',stderr=p.stderr)
(H/'REFERENCE_ADMISSION_TRANSPORT_V2.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert p.returncode==0,p.stderr[-3000:]
rows=[x for x in p.stdout.splitlines() if x.startswith('REFERENCE_JSON=')];assert len(rows)==1
value=json.loads(rows[0].split('=',1)[1]);(H/'REFERENCE_ADMISSION.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(admitted=value['admitted'],banks=value['banks'],selected_fits=value['selected_fits'],no_native_forward=value['no_native_forward'])))

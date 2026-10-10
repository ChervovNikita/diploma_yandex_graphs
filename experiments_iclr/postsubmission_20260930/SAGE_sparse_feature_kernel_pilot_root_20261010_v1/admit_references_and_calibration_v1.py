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
H=P/'SAGE_sparse_feature_kernel_pilot_root_20261010_v1'
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
old_source=Path(b['family_source']);new_source=P/'native_SAGE_sparse_feature_kernel_source_20261010_v1/run_family.py'
assert old_source.resolve().is_relative_to(P)
def methods(q):
 cls=next(n for n in ast.parse(q.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Family')
 return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
o,n=methods(old_source),methods(new_source)
unchanged=['seeds','scope','metrics','progress','synchronize']
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
prior_reader=P/'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
prior_analysis=prior_reader/'complete_analysis_v1'
reader_end=read(prior_reader/'READER_JOB_END_V1.json');assert reader_end['success'] and reader_end['exit_code']==0 and not reader_end['TEST_access']
summary=read(prior_analysis/'COMPLETE_ANALYSIS_SUMMARY.json');assert summary['complete']
fit_details=read(prior_analysis/'CALIBRATION_FIT_DETAILS_part1.json')
archive_details=read(prior_analysis/'ARCHIVE_CUSTODY_part1.json')
assert fit_details['total_rows']==len(fit_details['rows'])==33
fit_index={(r['seed'],r['arm']):r for r in fit_details['rows']}
archive_index={(r['seed'],r['arm']):r for r in archive_details['rows']}
assert len(fit_index)==len(archive_index)==33
prior_policy=read(prior_reader/'CALIBRATION_POLICY.json');assert prior_policy==read(H/'CALIBRATION_POLICY.json')
reuse=[]
for bank in banks:
 key=bank['seed'],bank['arm'];rec=fit_index[key];arc=archive_index[key]
 assert arc['path']==bank['archive'] and arc['sha256']==bank['archive_sha256']
 assert rec['status']=='finite_complete_five_fold' and rec['attempted_scalar_fits']==5
 assert len(rec['fits'])==5 and [r['fold'] for r in rec['fits']]==list(range(5))
 assert all(r['updates']==500 and r['parameters']==1 and r['status']=='finite_fixed_endpoint' for r in rec['fits'])
 array=prior_analysis/rec['OOF_archive']['name'];assert array.stat().st_size==rec['OOF_archive']['bytes'] and sha(array)==rec['OOF_archive']['sha256']
 reuse.append(dict(arm=bank['arm'],seed=bank['seed'],raw_archive=arc,endpoint_record=rec,calibrated_archive=str(array),calibrated_sha256=sha(array),prior_fit_seconds=rec['fit_seconds_sum']))
assert len(reuse)==15 and sum(r['endpoint_record']['attempted_scalar_fits'] for r in reuse)==75
reused=dict(UTC=datetime.now(timezone.utc).isoformat(),admitted=True,reused_banks=15,reused_scalar_endpoints=75,new_scalar_endpoints=90,total_endpoints=165,TEST_access=False,new_scalar_steps=45000,no_native_forward=True,no_fit_replay=True,policy_sha256=sha(H/'CALIBRATION_POLICY.json'),prior_policy_sha256=sha(prior_reader/'CALIBRATION_POLICY.json'),prior_reader_end_sha256=sha(prior_reader/'READER_JOB_END_V1.json'),prior_complete_summary_sha256=sha(prior_analysis/'COMPLETE_ANALYSIS_SUMMARY.json'),prior_analysis_source_sha256=sha(P/'SAGE_GNCL_SupCon_2x2_complete_reader_source_20261010_v2/analysis.py'),calibration_operator_sha256=sha(P/'common_wrapper_graph_reliability_source_20261010_v2/operators.py'),fold_assignment_sha256=read(prior_analysis/'COMPLETE_REPORT.json')['calibration_folds']['assignment_sha256'],records=reuse,cost_scope='75 already-paid original fits retained separately; 90 new endpoints only')
rp=H/'CALIBRATION_REUSE_ADMISSION.json'
if rp.exists():
 previous=read(rp);assert {k:v for k,v in previous.items() if k!='UTC'}=={k:v for k,v in reused.items() if k!='UTC'};reused=previous
else:rp.write_text(json.dumps(reused,indent=2)+'\n')
value['calibration_reuse_admission']=reused

a=H/'REFERENCE_ADMISSION.json'
if a.exists():
 prior=read(a);assert {k:v for k,v in prior.items() if k!='UTC'}=={k:v for k,v in value.items() if k!='UTC'};value=prior
else:a.write_text(json.dumps(value,indent=2)+'\n')
print('REFERENCE_JSON='+json.dumps(dict(reference=value,reuse=reused)))'''
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
p=subprocess.run(args,text=True,capture_output=True,timeout=90)
receipt=dict(exit_code=p.returncode,stdout=p.stdout if p.returncode else 'reference receipt collected',stderr=p.stderr)
(H/'REFERENCE_ADMISSION_TRANSPORT_V2.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert p.returncode==0,p.stderr[-3000:]
rows=[x for x in p.stdout.splitlines() if x.startswith('REFERENCE_JSON=')];assert len(rows)==1
envelope=json.loads(rows[0].split('=',1)[1]);value=envelope['reference'];(H/'CALIBRATION_REUSE_ADMISSION.json').write_text(json.dumps(envelope['reuse'],indent=2)+'\n');(H/'REFERENCE_ADMISSION.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(admitted=value['admitted'],banks=value['banks'],selected_fits=value['selected_fits'],no_native_forward=value['no_native_forward'])))

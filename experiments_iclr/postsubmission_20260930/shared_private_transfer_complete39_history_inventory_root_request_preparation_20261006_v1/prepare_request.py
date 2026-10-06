"""Bind actual complete39 metadata to the unchanged, separately disabled history byte inventory."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import shlex
import zlib

HERE=Path(__file__).resolve().parent;P=HERE.parent;NAME=HERE.name
COLLECTOR=P/'shared_private_transfer_allocation_replication_preparation_20261005_v2'
ACTUAL=P/'shared_private_transfer_complete39_collection_operation_20261006_v1'
PREVIOUS=P/'shared_private_transfer_complete39_collection_root_request_preparation_20261006_v1'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
PHASE=REPO+'/experiments_iclr/postsubmission_20260930'
OUT='shared_private_transfer_complete39_history_inventory_execution_root_20261006_v1'
REGISTRY={'path':'shared_private_transfer_complete39_collection_execution_root_20261006_v1/COLLECTION_FREEZE.json','sha256':'b3b318ed1a64bf73a5e453c4665183e3540fc7f8ef114fa41345922cf4860d25'}

def sha(b):return hashlib.sha256(b).hexdigest()
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def put(n,b):
 with (HERE/n).open('xb') as f:f.write(b)
def write(n,v):put(n,encoded(v))
def bind(p):
 b=p.read_bytes();return {'path':str(p.relative_to(P)),'bytes':len(b),'sha256':sha(b)}
def ref(p):return {k:v for k,v in bind(p).items() if k!='bytes'}

registry_bytes=(ACTUAL/'emitted_metadata/COLLECTION_FREEZE.json').read_bytes();assert sha(registry_bytes)==REGISTRY['sha256'] and len(registry_bytes)==294161
registry=json.loads(registry_bytes);assert registry['complete'] is True and registry['full39_terminal_source_artifact_custody_passed'] is True and len(registry['completed'])==39
for k in ('history_bytes_observed','history_JSON_parsed','FREEZE_JSON_parsed','CONFIG_JSON_parsed','selected_prediction_payloads_deserialized','quality_fields_accessed_or_emitted','comparative_scoring_performed'):assert registry[k] is False
collection_bytes=(ACTUAL/'emitted_metadata/COLLECTION_RELEASE.json').read_bytes()
assert sha(collection_bytes)==registry['collection_release_sha256']=='b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e'
assert sha((ACTUAL/'emitted_metadata/ATTEMPT_HISTORY.json').read_bytes())=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
assert sha((COLLECTOR/'MANIFEST.json').read_bytes())==registry['collector_manifest_sha256']=='c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
source=bind(COLLECTOR/'inventory_histories_after39.py');assert source['sha256']=='874b7698703941223d97350cd1e5922e642656ff746f08993dc98f62f571bd20'
helper_refs=json.loads((COLLECTOR/'REUSED_CUSTODY_HELPERS.json').read_text())
sources=[bind(COLLECTOR/n) for n in ('inventory_histories_after39.py','protocol.py','collect39_metadata.py','REUSED_CUSTODY_HELPERS.json','MANIFEST.json')]
for r in helper_refs.values():
 b=(P/r['path']).read_bytes();assert sha(b)==r['sha256'];sources.append(bind(P/r['path']))
review_ref={'path':'shared_private_transfer_allocation_replication_operational_independent_review_20261005_v2/REPORT.md','sha256':'786def5d8d61f59c6f5bf23b9e6271ca715efbef461fe5e961ef9a82635ad871'}
assert sha((P/review_ref['path']).read_bytes())==review_ref['sha256']
write('SOURCE_BINDINGS.json',{'schema':'unchanged_operativeV2_history_inventory_immutable_digest_bindings_v1','files':sources,'source_review_evidence':[review_ref],
 'reviewed_helper_refs':helper_refs,'all_bindings_immutably_pinned_by_SHA256':True,'source_bytes_or_modes_changed':False,'source_imported_or_executed':False})
release=json.loads((COLLECTOR/'HISTORY_INVENTORY_RELEASE_DISABLED.json').read_text())
release['complete39_registry']=REGISTRY;release['inventory_manifest_sha256']=registry['collector_manifest_sha256'];release['source_review_evidence']=[review_ref]
assert release['root_history_inventory_approved'] is False
write('HISTORY_INVENTORY_RELEASE_DISABLED.json',release)
approved=dict(release);approved['root_history_inventory_approved']=True
body=encoded(approved);body_sha=sha(body)
assert [k for k in release if release[k]!=approved[k]]==['root_history_inventory_approved']
actual_collection_ref={'path':str(Path(REGISTRY['path']).parent/'COLLECTION_RELEASE.json'),'sha256':sha(collection_bytes)}
pins=[(REGISTRY['path'],REGISTRY['sha256'],True),(actual_collection_ref['path'],actual_collection_ref['sha256'],True),
      (str(Path(REGISTRY['path']).parent/'ATTEMPT_HISTORY.json'),'e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970',True)]
source_pins=[(r['path'],r['sha256'],False) for r in sources]
stage=f'''from pathlib import Path
import base64,hashlib,json,socket,subprocess,sys,zlib
repo=Path({REPO!r});phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
for rel,expected,readonly in {pins!r}:
 p=phase/rel;assert p.resolve(strict=True).is_relative_to(phase) and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected
 if readonly:assert p.stat().st_mode&0o222==0
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())));assert set(payload)=={{'release_base64'}}
body=base64.b64decode(payload['release_base64']);assert len(body)=={len(body)} and hashlib.sha256(body).hexdigest()=={body_sha!r}
value=json.loads(body);assert value['root_history_inventory_approved'] is True and value['complete39_registry']=={REGISTRY!r}
target=phase/{(NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json')!r};output=phase/{OUT!r}
assert not target.exists() and not output.exists() and target.resolve().is_relative_to(phase)
for p in [target,*target.parents]:
 if p==phase.parent:break
 assert not p.is_symlink()
target.parent.mkdir(parents=True,exist_ok=True)
with target.open('xb') as f:f.write(body)
target.chmod(0o444);assert target.stat().st_mode&0o777==0o444 and hashlib.sha256(target.read_bytes()).hexdigest()=={body_sha!r}
assert not output.exists()
print(json.dumps({{'schema':'exact_history_byte_inventory_release_staging_receipt_v1','path':str(target.relative_to(phase)),'bytes':len(body),'sha256':{body_sha!r},'mode':'0444','actual_registry_bound':True,'fresh_output_absent':True,'history_bytes_read':False,'inventory_invoked':False}}))
'''
ast.parse(stage);put('STAGING_REMOTE_SOURCE.txt',stage.encode())
prior=json.loads((PREVIOUS/'ROOT_APPROVAL_REQUEST.json').read_text());ssh_prefix=prior['remote_invocation_argv'][:-1]
stage_argv=ssh_prefix+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(stage)]
payload=base64.b64encode(zlib.compress(json.dumps({'release_base64':base64.b64encode(body).decode()},sort_keys=True).encode(),9))+b'\n'
write('STAGING_INVENTORY.json',{'schema':'exact_history_inventory_single_release_staging_inventory_v1','root_approved':False,'staged':False,
 'files':[{'local_preapproval_input':str(HERE/'HISTORY_INVENTORY_RELEASE_DISABLED.json'),'remote_phase_relative':NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json',
 'bytes':len(body),'sha256':body_sha,'mode':'0444','exists':False,'root_only_delta':'root_history_inventory_approved false -> true; canonical sorted indent2 JSON plus newline.'}],
 'source_staging_or_changes':0,'argv':stage_argv,'remote_command_characters':len(stage_argv[-1]),'remote_source':bind(HERE/'STAGING_REMOTE_SOURCE.txt'),
 'transport':'Short SSH command with compressed base64 JSON stdin containing only release_base64.',
 'expected_stdin_bytes':len(payload),'expected_stdin_sha256':sha(payload),
 'root_only_stdin_recipe':'base64.b64encode(zlib.compress(json.dumps({release_base64:base64.b64encode(approved_release_bytes).decode()},sort_keys=True).encode(),9)) plus newline',
 'actual_registry':REGISTRY,'immutable_reviewed_source_bindings':bind(HERE/'SOURCE_BINDINGS.json'),'overwrite_retry_or_fallback':False})
argv=['/usr/bin/python3','-S','-B',PHASE+'/'+source['path'],'--release',PHASE+'/'+NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json','--output',PHASE+'/'+OUT]
invoke=f'''from pathlib import Path
import hashlib,json,os,socket,subprocess
repo=Path({REPO!r});phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
for rel,expected,readonly in {[(NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json',body_sha,True),*pins,*source_pins]!r}:
 p=phase/rel;assert p.resolve(strict=True).is_relative_to(phase) and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected
 if readonly:assert p.stat().st_mode&0o222==0
assert not (phase/{OUT!r}).exists()
argv={argv!r}
os.execv('/usr/bin/python3',argv)
'''
ast.parse(invoke);put('ONCE_ONLY_REMOTE_INVOCATION_SOURCE.txt',invoke.encode())
invoke_argv=ssh_prefix+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(invoke)]
d2=json.loads((PREVIOUS/'ROOT_D2_ANALYSIS_RELEASE_PENDING.json').read_text());d2['complete39_registry']=REGISTRY
assert d2['root_analysis_approved'] is False and d2['VALID_history_inventory']['sha256'] is None
write('ROOT_D2_ANALYSIS_RELEASE_PENDING.json',d2)
write('NEXT_D2_BINDINGS.json',{'schema':'actual_registry_final_byte_inventory_to_D2_binding_requirements_v1','actual_registry':REGISTRY,
 'actual_collection_release':actual_collection_ref,'prospective_history_inventory':{'path':OUT+'/VALID_HISTORY_INVENTORY.json','sha256':None},
 'prospective_history_release':{'path':NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json','bytes':len(body),'sha256':body_sha,'exists':False},
 'prospective_emitted_history_release':{'path':OUT+'/HISTORY_INVENTORY_RELEASE.json','sha256':body_sha},
 'pending_D2_release':bind(HERE/'ROOT_D2_ANALYSIS_RELEASE_PENDING.json'),
 'D2_source':{'path':'shared_private_transfer_amended39_d2_activation_preparation_20261006_v1/analyze_d2_accounting_proposal.py','sha256':'0fdd061baaa71bb525c7122ff8bdf98c8dfe450074c41819cc3ab659e4ab648a'},
 'D2_source_manifest_sha256':'3fc153c6f690af54080150725dc5e575e79ede5747d5b7cf989ad949ff0cf3f4',
 'D2_independent_delta_review':{'path':'shared_private_transfer_amended39_d2_activation_independent_delta_review_20261006_v1/FINDINGS.json','sha256':'455069737809b754a35cccd00c645741fec973653e8faac608ed97568bfc8555'},
 'needed_after_actual_inventory':['Actual VALID_HISTORY_INVENTORY SHA','Root review/adoption of actual registry and score-unparsed inventory','Separate exact enabled D2 release/one invocation authority'],
 'no_D2_or_score_semantics_authority_issued':True,'original_selector_and_UNKNOWN_and_excluded20_unchanged':True})
write('ROOT_APPROVAL_REQUEST.json',{'schema':'actual_complete39_history_BYTE_inventory_root_request_v1','status':'PREPARED_AWAITING_ROOT_HISTORY_BYTE_INVENTORY_APPROVAL',
 'root_approval_issued':False,'history_inventory_invocations':0,'staging_performed':False,'invocation_count_limit':1,'retry':False,'resume':False,'fallback':False,
 'disabled_release':bind(HERE/'HISTORY_INVENTORY_RELEASE_DISABLED.json'),
 'proposed_approved_release':{'path':NAME+'/ROOT_HISTORY_INVENTORY_RELEASE.json','bytes':len(body),'sha256':body_sha,'exists':False,'only_delta':'root_history_inventory_approved false -> true'},
 'actual_complete39_registry':REGISTRY,'local_actual_registry':bind(ACTUAL/'emitted_metadata/COLLECTION_FREEZE.json'),
 'actual_collection_release':actual_collection_ref,'actual_collection_operation_seal':bind(ACTUAL/'SEAL.json'),
 'inventory_source':source,'inventory_source_packet':bind(COLLECTOR/'MANIFEST.json'),'immutable_reviewed_helper_bindings':bind(HERE/'SOURCE_BINDINGS.json'),
 'source_review_evidence':[review_ref],'staging_inventory':bind(HERE/'STAGING_INVENTORY.json'),
 'remote_invocation_source':bind(HERE/'ONCE_ONLY_REMOTE_INVOCATION_SOURCE.txt'),'remote_invocation_argv':invoke_argv,'inventory_argv':argv,'output_phase_relative':OUT,
 'scope_to_approve':'Stage one exact immutable release; invoke unchanged operativeV2 history wrapper once; reauthenticate39 before hashing39 raw history files; freeze/gather only two emitted JSON metadata files and record exit/cost.',
 'raw_VALID_history_BYTE_hashes_authorized_in_proposed_operation':True,'raw_history_copies':0,'history_JSON_or_score_values_parsed':False,
 'FREEZE_CONFIG_semantics_or_tensor_deserialization':False,'held_TEST_score_D2_fit_model_or_GPU_job_actions':False,'source_donor_selector_changes':0,
 'original_UNKNOWN_history':{'path':str((COLLECTOR/'ATTEMPT_HISTORY.json').relative_to(P)),'sha256':'e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970','rewrite_authorized':False},
 'excluded_old20_GPU77_preserved':True,'stop_after':'Actual VALID_HISTORY_INVENTORY and terminal exit plus two readonly metadata files; D2 remains separately disabled.',
 'next_D2_bindings':bind(HERE/'NEXT_D2_BINDINGS.json')})
write('STATIC_VERIFICATION.json',{'schema':'exact_history_BYTE_inventory_request_static_verification_v1','actual_registry_digest_and39_metadata_scope_verified':True,
 'actual_collection_release_and_original_UNKNOWN_digest_verified':True,'unchanged_inventory_source_874b7698_verified':True,
 'reviewed8_source_metadata_bindings_verified':len(sources),'exact_flag_only_proposed_release_verified':True,
 'approved_release_not_written':True,'no_history_bytes_scores_tensors_held_or_FREEZE_CONFIG_semantics_read':True,
 'no_prepared_source_imported_or_executed':True,'no_remote_contact_staging_or_invocation':True,'envelope_AST_parse_passed':True,
 'D2_registry_actual_hash_filled_history_hash_pending_no_authority':True})
put('REQUEST.md',f'''# Exact all39 history BYTE inventory: root request

Preparation only; root approval pending. No remote contact, staging, source/helper import or execution, raw VALID history byte read/hash, score values, tensors, held labels or FREEZE/CONFIG semantics were observed here.

## Exact action

Root generates one {len(body)}-byte immutable ROOT_HISTORY_INVENTORY_RELEASE.json, SHA256 `{body_sha}`, by changing only `root_history_inventory_approved` false to true in the sealed disabled release and using sorted indent2 JSON plus newline. It does not exist yet. Stage it with STAGING_INVENTORY.json's bounded command plus compressed stdin, then invoke ROOT_APPROVAL_REQUEST.json's exact argv once. No source is changed or restaged. Output is fresh `{OUT}`. The source is unchanged operativeV2 inventory_histories_after39.py SHA874b7698703941223d97350cd1e5922e642656ff746f08993dc98f62f571bd20; packetc29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6. SOURCE_BINDINGS pins all reused helpers/authority metadata by exact reviewed SHA. The outer guard verifies host/UUID/repo, actual registry/release/ledger and source/helper pins; the unchanged wrapper uses -S -B for its reviewed local imports.

## Actual custody

The input registry is `{REGISTRY['path']}`,294161B/SHA `{REGISTRY['sha256']}`. It is the actual exit0 complete39 collection, not a candidate or synthetic registry. All42 collection metadata files are readonly. Source/terminal/artifact custody passed for fixed original13 plus new26; no previous history/semantic/scoring access is claimed. Registry release SHA b9375929fcc458d21a3309757b4ff515d26eaf37d6edc3ff32a70159a03b065e and original UNKNOWN ledgere3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970 remain exact. Old20 GPU77 remains excluded and all failed-stage evidence remains preserved.

## Byte scope and stopping point

The existing wrapper reauthenticates all39 and crosschecks registry artifact identities before its first history read. It then hashes each raw VALID_HISTORY.jsonl as opaque bytes, records its paired FREEZE byte identity, and emits HISTORY_INVENTORY_RELEASE.json plus VALID_HISTORY_INVENTORY.json. It never parses a JSONL record or score. Root's proposed scope includes freezing only those two emitted JSON metadata files0444 and gathering them with command/exit/cost evidence. No raw history copy, tensor loading, FREEZE/CONFIG or history/score semantics, held/TEST access, D2, fit, model or GPU job actions are admitted. No source, donor, scientific cell, selector, promotion or original UNKNOWN metadata changes are admitted. Any uncertain actual launch preserves the same output/handle; no automatic retry/resumption.

The operation stops after the actual inventory and terminal exit. The pending D2 release now binds the real registry hash and preserves the reviewed accounting source/review/addendum, but its history inventory hash is null until this real emitter succeeds. Root then fills that single actual binding, adopts custody and separately authorizes D2. No D2 or semantic score authority is issued here. No concrete preparation blocker remains.
'''.encode())
files=[p for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')]
write('MANIFEST.json',{'schema':'stdlib_sha256_manifest_v1','status':'SEALED_EXACT_ACTUAL_REGISTRY_HISTORY_BYTE_INVENTORY_ROOT_REQUEST_AWAITING_APPROVAL',
 'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files]})
write('SEAL.json',{'schema':'actual_complete39_history_BYTE_inventory_request_seal_v1','manifest_sha256':sha((HERE/'MANIFEST.json').read_bytes()),
 'root_approval_issued':False,'history_inventory_D2_executions':0,'raw_history_bytes_or_scores_read':False,'actual_registry':REGISTRY,
 'approved_release_sha256':body_sha,'approved_release_exists':False,'source_selectors_UNKNOWN_excluded20_unchanged':True})
for p in HERE.iterdir():
 if p.is_file():p.chmod(0o444)
print(json.dumps({'packet':str(HERE),'manifest_members':len(files),'manifest_sha256':sha((HERE/'MANIFEST.json').read_bytes()),'seal_sha256':sha((HERE/'SEAL.json').read_bytes()),
 'approved_release_bytes':len(body),'approved_release_sha256':body_sha,'stage_command_characters':len(stage_argv[-1]),'stdin_bytes':len(payload)}))

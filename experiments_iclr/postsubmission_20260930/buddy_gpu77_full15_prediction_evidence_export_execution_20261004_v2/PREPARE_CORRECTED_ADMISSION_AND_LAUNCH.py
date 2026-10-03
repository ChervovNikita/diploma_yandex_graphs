"""Prepare a root-reviewable metadata recovery; do not execute or change remote files."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,importlib.util,json,shlex,sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
OLD=PHASE/'buddy_gpu77_full15_prediction_evidence_export_execution_20261004_v1'
PACKET=PHASE/'buddy_gpu77_postfamily_eval_preparation_v4'
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(OLD/'EXECUTION_MANIFEST.json')=='b8fb3ab5b521120690a8ac159c935d2d7a0cea993ea40efb2d980deb1e8143c3'
old_admission=json.loads((OLD/'ROOT_PREDICTION_EXPORT_ADMISSION.json').read_text())
assert sha(OLD/'ROOT_PREDICTION_EXPORT_ADMISSION.json')=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
admission=dict(old_admission,optimizer_fits=24,epochs_per_cell=100,
 closed_family_count_semantics='optimizer_fits=24 and epochs_per_cell=100 describe the already completed and locked family. These fields authorize no new training, fitting, epochs or selection.')
assert all(admission[key]==value for key,value in old_admission.items())
assert set(admission)-set(old_admission)=={'optimizer_fits','epochs_per_cell','closed_family_count_semantics'}
raw=(json.dumps(admission,indent=2)+'\n').encode();admission_sha=hashlib.sha256(raw).hexdigest()
with (HERE/'ROOT_PREDICTION_EXPORT_ADMISSION.json').open('xb') as h:h.write(raw)
precheck=json.loads((OLD/'PRECHECK_RESULT.json').read_text());pins=precheck['identity']
manifest=json.loads((PACKET/'SOURCE_MANIFEST.json').read_text())
assert sha(PACKET/'SOURCE_MANIFEST.json')=='fd77b2f09418070e3e8d358231e4a98388d3e81796d513ba2165c9755ddeab22'
for row in manifest['files']:
 path=PACKET/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
sys.path.insert(0,str(PACKET))
import gate_contract as gates
import runtime_boundary as boundary
assert boundary.STAGE_SCHEMAS['export']=='buddy77-prediction-export-admission-v2'
assert boundary.STAGE_ADMISSIONS['export']=='ROOT_PREDICTION_EXPORT_ADMISSION.json'
# Evaluate the complete literal admission predicate in the actual shared function.
# Host normal-runtime/FD evidence remains a separate fresh delegated gate in the client.
tree=ast.parse((PACKET/'runtime_boundary.py').read_text())
function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='current_execution_boundary')
shared=next(node.value for node in function.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
            and isinstance(node.value.func,ast.Name) and node.value.func.id=='matches')
dictcall=shared.args[1];requirements={}
for keyword in dictcall.keywords:
 if keyword.arg=='schema':value=boundary.STAGE_SCHEMAS['export']
 else:value=ast.literal_eval(keyword.value)
 requirements[keyword.arg]=value
gates.matches(admission,requirements,'Shared prospective stage admission literal predicate')
assert gates.runtime_mode_gate(admission)=='environment_and_explicit_repo_paths_only'
# Check the exact downstream admission predicate and cost decision without Torch.
export_tree=ast.parse((PACKET/'export_predictions77.py').read_text())
export_function=next(node for node in export_tree.body if isinstance(node,ast.FunctionDef) and node.name=='export')
downstream=next(node.value for node in export_function.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
               and isinstance(node.value.func,ast.Name) and node.value.func.id=='matches'
               and len(node.value.args)>2 and isinstance(node.value.args[2],ast.Constant)
               and node.value.args[2].value=='Separate root prediction-export admission')
down_requirements=dict(pins)
for keyword in downstream.args[1].keywords:down_requirements[keyword.arg]=ast.literal_eval(keyword.value)
gates.matches(admission,down_requirements,'Exact downstream export admission predicate')
assert isinstance(admission['root_export_cost_decision'],str) and admission['root_export_cost_decision'].strip()
assert all(admission[key]==value for key,value in pins.items())
static=dict(schema='buddy77_export_corrected_admission_static_gate_review_v1',UTC=datetime.now(timezone.utc).isoformat(),status='PASS',
 corrected_admission_sha256=admission_sha,preserved_failed_admission_sha256=sha(OLD/'ROOT_PREDICTION_EXPORT_ADMISSION.json'),
 preserved_v1_execution_manifest_sha256=sha(OLD/'EXECUTION_MANIFEST.json'),added_fields={key:admission[key] for key in set(admission)-set(old_admission)},
 shared_literal_requirements=requirements,downstream_literal_requirements=down_requirements,
 shared_path_schema_and_normal_mode_checked=True,all_required_admission_fields_pass=True,
 static_scope='Exact source predicates evaluated against corrected in-memory JSON. Does not certify live host/FD/runtime state.',
 fresh_complete_shared_normal_runtime_boundary_delegated_in_prelaunch_client=True,
 source_predicates_sha256={name:sha(PACKET/name) for name in ('gate_contract.py','runtime_boundary.py','export_predictions77.py')},
 scientific_source_or_primary_results_changed=False,remote_executed=False)
with (HERE/'STATIC_ADMISSION_GATE_RESULT.json').open('x') as h:json.dump(static,h,indent=2);h.write('\n')
runner=(OLD/'DETACHED_RUNNER_SOURCE.py').read_text();client=(OLD/'LAUNCH_REMOTE.py.txt').read_text()
assert hashlib.sha256(runner.encode()).hexdigest()=='78943faec665b260d2077123df24c2cfdf6e1665e60a8719e213e22f1f3a225a'
assert hashlib.sha256(client.encode()).hexdigest()=='740e95ca9f4deaae7b34635a50eae6d930dd7e3fbf528dface6a5ccb10f2a898'
payload_node=next(node for node in ast.parse(client).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in node.targets))
old_encoded=ast.literal_eval(payload_node.value.args[0].args[0]);payload=json.loads(base64.b64decode(old_encoded))
assert client.count(old_encoded)==1
runner=runner.replace('root_predictions_supervision_20261004_v1','root_predictions_supervision_20261004_v2')
runner=runner.replace('3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623',admission_sha)
client=client.replace('root_predictions_supervision_20261004_v1','root_predictions_supervision_20261004_v2')
client=client.replace('import os,sys,json,base64,hashlib,subprocess,time,resource','import os,sys,json,base64,hashlib,subprocess,time,resource,importlib.util')
anchor="existing=[p for p in [ADMISSION,HERE/'root_predictions_v1',OUT] if p.exists() or p.is_symlink()]"
assert client.count(anchor)==1
client=client.replace(anchor,"existing=[p for p in [HERE/'root_predictions_v1',OUT] if p.exists() or p.is_symlink()]")
anchor="for entry in payload['source_checks'].values():"
assert client.count(anchor)==1
client=client.replace(anchor,r'''
PREVIOUS=HERE/'root_predictions_supervision_20261004_v1'
assert sha(ADMISSION)=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
assert sha(PREVIOUS/'PHYSICAL_TERMINAL.json')=='50bf21bedbc2644dad7986bba224cc63713602febd2cb2400809f2781d0218ab'
assert sha(PREVIOUS/'CHILD.stderr.log')=='700e28ea76d768d90f51cba69fb3cc2317c00700117b71fca529b6ee79b8fbf6'
assert read(PREVIOUS/'PHYSICAL_TERMINAL.json')['exit_code']==1
for name in ('RUNNER_STARTED.json','CHILD_STARTED.json'):
 handle=read(PREVIOUS/name)['identity'];stat=Path(f"/proc/{handle['pid']}/stat")
 if stat.exists():
  line=stat.read_text();fields=line[line.rfind(')')+2:].split()
  assert int(fields[19])!=handle['start_ticks'] or fields[0]=='Z','Previous owned export process remains live'
'''+anchor)
anchor="with safe(ADMISSION).open('xb') as h:h.write(admission)\nOUT.mkdir(exist_ok=False)"
assert client.count(anchor)==1
replacement=r'''
# Call the full unchanged normal shared boundary with corrected in-memory stage
# JSON, while every family/runtime/FD read delegates to the actual sealed host.
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('sealed_export_recovery_gate_preparation',HERE/'evaluate77.py')
preparation=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation)
launcher=preparation.load_module(preparation.LAUNCHER/'launch77.py','sealed_export_recovery_actual_launcher')
corrected=json.loads(admission)
class ProspectiveAdmission:
 def __getattr__(self,name):return getattr(launcher,name)
 def read_json(self,path):return corrected if Path(path)==ADMISSION else launcher.read_json(path)
 def sha(self,path):return payload['admission_sha256'] if Path(path)==ADMISSION else launcher.sha(path)
fresh_boundary=preparation.current_execution_boundary(ProspectiveAdmission(),ADMISSION,'export',pins['replay_GPU_UUID'],HERE)
preparation.matches(corrected,dict(schema='buddy77-prediction-export-admission-v2',decision='admitted',family_cells=15,
 optimizer_fits=24,epochs_per_cell=100,prior_partial_fits_excluded=True),'Complete shared gate counts')
preparation.matches(corrected,dict(pins,schema='buddy77-prediction-export-admission-v2',decision='admitted',family_cells=15,
 export_once=True,extra_forward_per_cell=1,no_new_training=True,no_new_selection=True,other_jobs_stopped=False,
 prior_partial_fits_excluded=True),'Complete downstream export admission')
assert isinstance(corrected['root_export_cost_decision'],str) and corrected['root_export_cost_decision'].strip()
OUT.mkdir(exist_ok=False)
# The canonical path is fixed by the unchanged exporter. Preserve the original
# failed bytes by a rename into this exclusively created custody directory,
# then install corrected bytes with an exclusive create; no old file is overwritten.
preserved=OUT/'PRESERVED_FAILED_ADMISSION_v1.json'
assert not preserved.exists() and sha(ADMISSION)=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
os.rename(ADMISSION,preserved)
assert sha(preserved)=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
with safe(ADMISSION).open('xb') as h:h.write(admission)
assert sha(ADMISSION)==payload['admission_sha256']
custody=dict(schema='buddy77_failed_export_admission_preserved_metadata_recovery_v2',UTC=datetime.now(timezone.utc).isoformat(),
 original_fixed_path=str(ADMISSION),preserved_failed_admission=descriptor(preserved),corrected_admission=descriptor(ADMISSION),
 prior_physical_terminal=descriptor(PREVIOUS/'PHYSICAL_TERMINAL.json'),prior_stderr=descriptor(PREVIOUS/'CHILD.stderr.log'),
 prior_extra_forwards=0,changed_fields=['optimizer_fits','epochs_per_cell','closed_family_count_semantics'],
 closed_family_counts_authorize_no_new_training=True,fresh_complete_shared_boundary=fresh_boundary,
 scientific_source_or_primary_results_changed=False,prior_supervision_files_modified=False,
 exclusive_new_metadata_and_supervision=True)
with (OUT/'ADMISSION_RECOVERY_CUSTODY.json').open('x') as h:json.dump(custody,h,indent=2);h.write('\n')
'''
client=client.replace(anchor,replacement)
client=client.replace('numerical_retry=False,training_executed=False)',"numerical_retry=False,training_executed=False,\n prior_failed_stage_invocations=1,prior_extra_forwards=0,metadata_recovery_custody=descriptor(OUT/'ADMISSION_RECOVERY_CUSTODY.json'))")
payload.update(admission_base64=base64.b64encode(raw).decode(),admission_sha256=admission_sha,
 runner_base64=base64.b64encode(runner.encode()).decode(),runner_sha256=hashlib.sha256(runner.encode()).hexdigest())
client=client.replace(old_encoded,base64.b64encode(json.dumps(payload).encode()).decode())
compile(runner,'corrected_export_successor_runner','exec');compile(client,'corrected_export_successor_client','exec')
with (HERE/'DETACHED_RUNNER_SOURCE.py').open('x') as h:h.write(runner)
with (HERE/'LAUNCH_REMOTE.py.txt').open('x') as h:h.write(client)
command=PHASE/'gpu77_connection_recovery_v1/buddy_v4_prediction_export_launch_20261004_v2_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+client])+'\n')
receipt=dict(schema='buddy77_corrected_export_metadata_successor_preparation_v2',UTC=datetime.now(timezone.utc).isoformat(),
 corrected_admission_sha256=admission_sha,corrected_admission_bytes=len(raw),
 static_gate_result_sha256=sha(HERE/'STATIC_ADMISSION_GATE_RESULT.json'),
 runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),runner_bytes=len(runner.encode()),
 client_sha256=hashlib.sha256(client.encode()).hexdigest(),client_bytes=len(client.encode()),
 command_sha256=sha(command),command_file=str(command.relative_to(PHASE)),
 v1_sealed_manifest_preserved_sha256=sha(OLD/'EXECUTION_MANIFEST.json'),v1_costs_and_all_logs_preserved=True,
 original_failed_admission_to_be_preserved_before_exclusive_corrected_canonical_install=True,
 shared_and_downstream_admission_predicates_checked=True,fresh_actual_shared_normal_runtime_gate_before_metadata_mutation=True,
 unchanged_exporter_and_runtime_sources=True,root_review_required_before_launch=True,
 remote_executed=False,export_launched=False,large_tensor_payloads_fetched=False)
with (HERE/'LAUNCH_PREPARATION.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(receipt))

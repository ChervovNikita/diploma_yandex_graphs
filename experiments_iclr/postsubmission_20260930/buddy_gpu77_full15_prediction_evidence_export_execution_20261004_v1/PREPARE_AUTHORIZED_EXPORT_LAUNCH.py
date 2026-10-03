"""Minimally adapt the reviewed once-only runner/client to the sealed post-primary exporter."""
from pathlib import Path
from datetime import datetime,timezone
import ast,base64,hashlib,json,shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
OLD=PHASE/'buddy_gpu77_postfamily_heldout_evaluation_execution_20261004_v1'
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
transport_path=PHASE/'gpu77_connection_recovery_v1/commands/buddy_v4_prediction_export_precheck_20261004_v1/RECEIPT.json'
transport=json.loads(transport_path.read_text());assert transport['exit_code']==0
precheck=json.loads(transport['stdout'].strip())
assert precheck['status']=='PASS_FULL15_PRIMARY_NO_EXISTING_EXPORT_ATTEMPT'
assert (precheck['family_cells'],precheck['optimizer_fits'],precheck['total_completed_cell_epochs'])==(15,24,1500)
assert precheck['admission_exact_byte_gate_passed'] is True and precheck['scoring_GPU_available_without_stopping_other_jobs'] is True
assert precheck['TEST_payload_deserialized'] is False and precheck['remote_writes'] is False
with (HERE/'PRECHECK_RESULT.json').open('x') as h:json.dump(precheck,h,indent=2);h.write('\n')
admission=(HERE/'ROOT_PREDICTION_EXPORT_ADMISSION.json').read_bytes()
admission_sha=hashlib.sha256(admission).hexdigest()
assert admission_sha=='3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'
root_review=OLD/'ROOT_COMPLETE_HELDOUT_REVIEW.json'
assert hashlib.sha256(root_review.read_bytes()).hexdigest()=='f9242c4de9e0d0e1a72bad8f76890df63de0316b85d6a98e5ff6b05e79e2e368'
runner=(OLD/'DETACHED_RUNNER_SOURCE.py').read_text()
client=(OLD/'LAUNCH_REMOTE.py.txt').read_text()
assert hashlib.sha256(runner.encode()).hexdigest()=='f74a141b9e4d8302b0d3456aa232255228d9ef29c3e704076c87bd6f3c365d06'
assert hashlib.sha256(client.encode()).hexdigest()=='80dd77b4ac0e4022c8da5f15b8967cdf68c939450cbcbc95e3fd5c4ca338043e'
payload_node=next(node for node in ast.parse(client).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in node.targets))
old_encoded=ast.literal_eval(payload_node.value.args[0].args[0]);assert client.count(old_encoded)==1
for old,new in [('root_eval_supervision_20261004_v1','root_predictions_supervision_20261004_v1'),
 ('ROOT_EVALUATION_ADMISSION','ROOT_PREDICTION_EXPORT_ADMISSION'),("HERE/'root_eval_v1'","HERE/'root_predictions_v1'"),
 ('heldout_evaluation','prediction_export'),('evaluation_once=True','export_once=True'),
 ("stage='evaluate'","stage='export'")]:
 runner=runner.replace(old,new);client=client.replace(old,new)
runner=runner.replace('48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c',admission_sha)
old_argv="str(HERE/'evaluate77.py'),'evaluate','--execute'"
assert runner.count(old_argv)==1
runner=runner.replace(old_argv,"str(HERE/'export_predictions77.py'),'--execute'")
runner=runner.replace('overlapping_internal_evaluation_intervals_not_added','overlapping_internal_export_intervals_not_added')
runner=runner.replace('owned audit descendants','owned export descendants')
client=client.replace('evaluate_launched=True','export_launched=True')
client=client.replace('export_executed=False','export_execution_delegated_to_admitted_stage=True')
client=client.replace("pins['scoring_GPU_UUID']","pins['replay_GPU_UUID']")
anchor="[payload['lock'],payload['lock_audit'],payload['audit_physical_terminal'],*payload['dependency_records'],*payload['cache_checks']]"
assert client.count(anchor)==1
client=client.replace(anchor,"[payload['lock'],payload['lock_audit'],payload['audit_physical_terminal'],payload['primary_evaluation'],payload['primary_admission'],payload['test_manifest'],payload['test_cache'],*payload['primary_results'],*payload['dependency_records'],*payload['cache_checks']]")
anchor="assert not any(p.exists() or p.is_symlink() for p in test_paths)"
assert client.count(anchor)==1
client=client.replace(anchor,"assert all(p.is_file() and not p.is_symlink() for p in test_paths[:3]) and len(test_paths[3:])==15")
payload={key:precheck[key] for key in ('identity','source_checks','ledgers','prior_custody','normal_runtime_boundary',
 'lock','lock_audit','audit_physical_terminal','dependency_records','cache_checks','qualified_runtime_paths',
 'primary_evaluation','primary_admission','primary_results','test_manifest','test_cache')}
payload.update(admission_base64=base64.b64encode(admission).decode(),admission_sha256=admission_sha,
 runner_base64=base64.b64encode(runner.encode()).decode(),runner_sha256=hashlib.sha256(runner.encode()).hexdigest())
client=client.replace(old_encoded,base64.b64encode(json.dumps(payload).encode()).decode())
compile(runner,'prediction_export_once_runner','exec');compile(client,'prediction_export_once_client','exec')
with (HERE/'DETACHED_RUNNER_SOURCE.py').open('x') as h:h.write(runner)
with (HERE/'LAUNCH_REMOTE.py.txt').open('x') as h:h.write(client)
command=PHASE/'gpu77_connection_recovery_v1/buddy_v4_prediction_export_launch_20261004_v1_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+client])+'\n')
receipt=dict(schema='buddy77_full15_export_once_launch_preparation_v1',UTC=datetime.now(timezone.utc).isoformat(),
 admission_sha256=admission_sha,root_complete_heldout_review_sha256=hashlib.sha256(root_review.read_bytes()).hexdigest(),
 runner_sha256=hashlib.sha256(runner.encode()).hexdigest(),runner_bytes=len(runner.encode()),
 client_sha256=hashlib.sha256(client.encode()).hexdigest(),client_bytes=len(client.encode()),
 command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),command_file=str(command.relative_to(PHASE)),
 precheck_transport_sha256=hashlib.sha256(transport_path.read_bytes()).hexdigest(),
 prior_Popen_once_wait4_physical_procedure_preserved=True,normal_host_mode=True,export_launched=False,
 scientific_source_modified=False,no_new_training=True,no_new_selection=True,no_retry=True,
 changes=['sealed exporter argv, export/admission/output/schema bindings',
 'primary15 scalar/evaluation admission/receipt/test-cache byte rechecks; existing TEST artifacts required',
 'inherited peak_RSS_scope prose corrected to owned export descendants'],
 root_source_review_required_before_launch=True)
with (HERE/'LAUNCH_PREPARATION.json').open('x') as h:json.dump(receipt,h,indent=2);h.write('\n')
print(json.dumps(receipt))

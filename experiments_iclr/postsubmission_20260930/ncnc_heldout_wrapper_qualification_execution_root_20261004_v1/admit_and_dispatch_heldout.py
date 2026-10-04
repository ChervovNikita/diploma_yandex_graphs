"""Adopt actual logical/physical QA before the frozen one-time TEST dispatch."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
from stage_and_launch_qa import HERE, PHASE, REPO, REMOTE_PHASE, REMOTE, PREP, SOURCE, REVIEW, run, sha, save


def descriptor(local, remote=None):
    return dict(path=str(remote or (REMOTE_PHASE / local.relative_to(PHASE))),
                bytes=local.stat().st_size, sha256=sha(local))


monitor = HERE / 'owned_monitor01'
qa_path = monitor / 'QUALIFICATION.json'
terminal_path = monitor / 'physical/SUPERVISOR_TERMINAL.json'
qa = json.loads(qa_path.read_text())
terminal = json.loads(terminal_path.read_text())
qa_release_path = HERE / 'ROOT_QA_RELEASE.json'
qa_release = json.loads(qa_release_path.read_text())
review_path = REVIEW / 'SOURCE_REVIEW.json'
review = json.loads(review_path.read_text())
enabled = json.loads((PREP / 'ROOT_HELDOUT_RELEASE_DISABLED_CANDIDATE.json').read_text())
remote_prep = REMOTE_PHASE / PREP.name
source_sha = sha(SOURCE / 'MANIFEST.json')
qa_remote_path = remote_prep / 'qualification/run01/QUALIFICATION.json'
terminal_remote_path = REMOTE / 'physical/run01/SUPERVISOR_TERMINAL.json'
assert qa['status'] == 'PASS' and terminal['status'] == 'PHYSICALLY_COMPLETE'
assert terminal['exit_code'] == 0 and terminal['error'] is None and terminal['bounded_stop_reason'] is None
assert terminal['signal_events'] == [] and terminal['release_and_supervisor_after_sha256_match'] is True
assert qa['root_release_sha256'] == terminal['root_release_sha256'] == sha(qa_release_path)
assert qa['heldout_source_manifest_sha256'] == terminal['heldout_source_manifest_sha256'] == source_sha
assert qa['original_family_identity'] == enabled['identity'] == review['identity']
assert qa['invocation'] == terminal['invocation'] == review['qualification_invocation'] == qa_release['authorized_invocations'][0]
assert qa['policy'] == review['policy'] == enabled['policy']
assert qa['runtime_authority'] == qa_release['runtime_authority'] == enabled['runtime_authority']
assert qa['fabricated_inputs_only'] is True and qa['TEST_opened'] is False
assert qa['study_data_or_selected_checkpoint_accessed'] is False and qa['training_updates'] == 0
assert qa['automatic_retry'] is False
plan = json.loads((SOURCE / 'FABRICATED_QUALIFICATION_PLAN.json').read_text())
assert qa['case_count'] == len(qa['cases']) == len(plan['cases']) == 18
assert {row['case'] for row in qa['cases']} == set(plan['cases'])
assert all(row['status'] == 'PASS' for row in qa['cases'])
assert qa['actual_original_scorer_calls'] == sum(row.get('original_scorer_calls', 0) for row in qa['cases']) == 40
assert qa['actual_stub_scorer_entries'] == sum(row.get('stub_scorer_entries', 0) for row in qa['cases']) == 52
assert terminal['logical_result'] == descriptor(qa_path, qa_remote_path)
assert qa['source_entry'] == terminal['source_entry'] == descriptor(SOURCE / 'heldout_qualification.py')
assert terminal['physical_supervisor'] == descriptor(HERE / 'physical_qa_supervisor.py')
assert review['status'] == 'PASS' and review['heldout_source_manifest_sha256'] == source_sha
assert qa_release['supervisor_source'] == review['supervisor_source']
assert qa_release['dispatcher_source'] == review['dispatcher_source']
bindings = [descriptor(qa_path, qa_remote_path), descriptor(terminal_path, terminal_remote_path),
            descriptor(qa_release_path), descriptor(review_path), descriptor(HERE / 'physical_qa_supervisor.py')]
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');root=Path(' + repr(str(remote_prep)) + ');assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'pins=' + repr(bindings) + '\n'
code += 'for row in pins:\n path=Path(row["path"]);assert path.resolve().is_relative_to(repo) and not path.is_symlink();assert path.stat().st_size==row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"]\n'
code += 'assert not (root/"ONE_TIME_TEST_CLAIM.json").exists() and not (root/"heldout/run01").exists() and not (root/"supervision_run01").exists() and not (root/"MANIFEST.json").exists()\n'
code += 'query=subprocess.run(["nvidia-smi","-i","0","--query-gpu=uuid,memory.free,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15);gpu=[value.strip() for value in query.stdout.strip().split(",")]\n'
code += 'assert gpu[0]=="GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998" and int(gpu[1])>=24576 and int(gpu[2])==81920\n'
code += 'available=next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"));assert available>=16*1024**3\n'
code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="PASS_ROOT_METADATA_AND_RESOURCE_CHECK",pins=pins,GPU=gpu,host_MemAvailable_bytes=available,TEST_file_opened_or_hashed=False,checkpoint_tensors_loaded=False,dispatch_recheck_required=True)))\n'
resource = run('ncnc_heldout_v2_root_QA_binding_resource_check_20261004', code)
save(HERE / 'ROOT_PREHELDOUT_RESOURCE_OBSERVATION.json', resource)
adoption = dict(schema='ncnc-heldout-wrapper-QA-root-adoption-v1', UTC=datetime.now(timezone.utc).isoformat(), status='PASS',
                independent_source_review=descriptor(review_path), heldout_source_manifest_sha256=source_sha,
                actual_enabled_fabricated_QA_release=descriptor(qa_release_path),
                actual_logical_QA=descriptor(qa_path, qa_remote_path),
                actual_successful_physical_QA_terminal=descriptor(terminal_path, terminal_remote_path),
                entry=qa['source_entry'], supervisor=qa_release['supervisor_source'], dispatcher=qa_release['dispatcher_source'],
                physical_QA_supervisor=descriptor(HERE / 'physical_qa_supervisor.py'),
                runtime_authority=qa['runtime_authority'], invocation=qa['invocation'], policy=qa['policy'],
                release_hash_logical_and_physical_match=True, physical_exit0_no_error_bound_signal_or_unreaped_child=True,
                all18_exact_case_identities_PASS=True, actual_original_scorer_calls=40, actual_stub_scorer_entries=52,
                identity=enabled['identity'], sentinel_QA_does_not_establish_production_memory=True,
                reviewed_automatic_gate_R1_requirement_satisfied_by_explicit_root_adoption=True,
                remote_metadata_and_resource_observation=resource,
                no_scientific_criterion_threshold_selection_or_retry_changed=True)
save(HERE / 'ROOT_QA_ADOPTION.json', adoption)
admission = json.loads((PREP / 'RUNTIME_RESOURCE_ADMISSION_DISABLED_CANDIDATE.json').read_text())
admission.update(status='PASS', UTC=datetime.now(timezone.utc).isoformat(),
                 root_QA_adoption=descriptor(HERE / 'ROOT_QA_ADOPTION.json'), actual_resource_observation=resource,
                 source_and_exact_dispatch_reviewed_by_root=True,
                 expanded_TEST_memory_not_guaranteed=True, owned_bounded_stop_and_all25_failure_closure_required=True)
save(HERE / 'ROOT_HELDOUT_RESOURCE_ADMISSION.json', admission)
enabled.update(execution_enabled=True, root_authorization_reference=str(REMOTE / 'ROOT_QA_ADOPTION.json'), TEST_access_authorized=True,
               independent_heldout_source_review=descriptor(review_path),
               runtime_resource_admission=descriptor(HERE / 'ROOT_HELDOUT_RESOURCE_ADMISSION.json'),
               heldout_wrapper_fabricated_qualification=descriptor(qa_path, qa_remote_path),
               heldout_wrapper_qualification_root_release=descriptor(qa_release_path),
               heldout_wrapper_qualification_physical_terminal=descriptor(terminal_path, terminal_remote_path),
               heldout_wrapper_qualification_root_adoption=descriptor(HERE / 'ROOT_QA_ADOPTION.json'))
enabled_path = HERE / 'ROOT_HELDOUT_RELEASE_ENABLED_ROOT_V1.json'
save(enabled_path, enabled)
enabled_remote_path = remote_prep / enabled_path.name
payload = []
for local, remote in ((HERE / 'ROOT_QA_ADOPTION.json', REMOTE / 'ROOT_QA_ADOPTION.json'),
                      (HERE / 'ROOT_HELDOUT_RESOURCE_ADMISSION.json', REMOTE / 'ROOT_HELDOUT_RESOURCE_ADMISSION.json'),
                      (enabled_path, enabled_remote_path)):
    payload.append(dict(path=str(remote), bytes=local.stat().st_size, sha256=sha(local), data=base64.b64encode(local.read_bytes()).decode()))
code = 'from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');phase=Path(' + repr(str(REMOTE_PHASE)) + ');root=Path(' + repr(str(remote_prep)) + ');assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'rows=' + repr(payload) + '\nfor row in rows:\n path=Path(row["path"]);assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]\n with path.open("xb") as stream:stream.write(raw)\n'
gate_code = 'import sys\nfrom pathlib import Path\nsys.path.insert(0,' + repr(str(REMOTE_PHASE / SOURCE.name)) + ')\nfrom heldout_gate import metadata_admission,OUTPUT\nmetadata_admission(' + repr(str(enabled_remote_path)) + ',OUTPUT,expected_sha=' + repr(sha(enabled_path)) + ')\nprint("EXACT_FROZEN_HELDOUT_STDLIB_GATE_PASS")\n'
code += 'environment=dict(os.environ,CUDA_VISIBLE_DEVICES=' + repr(enabled['cuda_visible_devices']) + ')\n'
code += 'gate=subprocess.run(["/usr/bin/python3","-I","-S","-B","-c",' + repr(gate_code) + '],cwd=repo,env=environment,capture_output=True,text=True,timeout=60)\n'
code += 'print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status="PASS" if gate.returncode==0 else "FAILED",exit_code=gate.returncode,stdout=gate.stdout,stderr=gate.stderr,TEST_file_opened_or_hashed=False,root_release_sha256=' + repr(sha(enabled_path)) + ')))\n'
gate = run('ncnc_heldout_v2_complete_stdlib_gate_before_TEST_20261004', code)
save(HERE / 'PRE_TEST_STDLIB_GATE.json', gate)
assert gate['status'] == 'PASS', gate
dispatcher = PREP / 'dispatch_heldout_once.py'
cmd = ['/usr/bin/python3', '-I', '-S', '-B', str(remote_prep / dispatcher.name),
       '--root-release', str(enabled_remote_path), '--release-sha256', sha(enabled_path)]
save(HERE / 'ROOT_FINAL_DISPATCH_REVIEW.json', dict(UTC=datetime.now(timezone.utc).isoformat(), status='APPROVED_ONCE',
     command=cmd, dispatcher=descriptor(dispatcher), root_release=descriptor(enabled_path, enabled_remote_path),
     root_QA_adoption=descriptor(HERE / 'ROOT_QA_ADOPTION.json'), pre_TEST_stdlib_gate_sha256=sha(HERE / 'PRE_TEST_STDLIB_GATE.json'),
     all25_no_subset_selection=True, primary_contrast='private_minus_pooled', new_threshold=None,
     retry_refit_reselection_calibration=False, no_direct_child_invocation=True, training_updates=0))
code = 'from pathlib import Path\nimport hashlib,json,os,subprocess\n'
code += 'repo=Path(' + repr(str(REPO)) + ');root=Path(' + repr(str(remote_prep)) + ');assert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
code += 'assert hashlib.sha256((root/' + repr(enabled_path.name) + ').read_bytes()).hexdigest()==' + repr(sha(enabled_path)) + '\n'
code += 'assert hashlib.sha256((root/"dispatch_heldout_once.py").read_bytes()).hexdigest()==' + repr(sha(dispatcher)) + '\n'
code += 'p=subprocess.run(' + repr(cmd) + ',cwd=repo,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr+p.stdout\nprint(p.stdout)\n'
launch = run('ncnc_heldout_v2_original_once_dispatch_20261004', code)
save(HERE / 'HELDOUT_DETACHED_LAUNCH.json', launch)
print(json.dumps(launch))

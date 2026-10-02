"""Stdlib metadata binding for admitted normal or optional namespace execution."""
import hashlib
import json
import os
from pathlib import Path
import sys

from gate_contract import require, matches, runtime_mode_gate
from runtime_paths import path_environment

CERTIFICATIONS = ['outside_mounts_read_only', 'write_root_bind_writable', 'private_mount_propagation',
                  'private_pid_proc', 'inherited_extra_fds_closed', 'capabilities_zero', 'no_new_privs']
STAGE_SCHEMAS = {'audit-lock':'buddy77-postfamily-lock-audit-admission-v1',
                 'evaluate':'buddy77-postfamily-evaluation-admission-v2',
                 'export':'buddy77-prediction-export-admission-v2'}
STAGE_ADMISSIONS = {'audit-lock':'ROOT_LOCK_AUDIT_ADMISSION.json',
                    'evaluate':'ROOT_EVALUATION_ADMISSION.json',
                    'export':'ROOT_PREDICTION_EXPORT_ADMISSION.json'}


def bound_namespace_proof(launcher, policy_path, proof_path, admission, argv, current):
    policy_path, proof_path = launcher.confined(policy_path), launcher.confined(proof_path)
    policy, proof = launcher.read_json(policy_path), launcher.read_json(proof_path)
    matches(policy, dict(schema='namespace-write-guard77-policy-v1', actual_repo=str(launcher.REPO)),
            'Namespace policy')
    require(launcher.sha(policy_path) == admission.get('external_namespace_policy_sha256'),
            'Prospectively admitted namespace policy changed')
    matches(proof, dict(schema='namespace-write-guard77-execution-proof-v1', mode='execute',
                        actual_repo=str(launcher.REPO), write_root=str(launcher.REPO),
                        policy_path=str(policy_path), policy_sha256=launcher.sha(policy_path),
                        proof_path=str(proof_path), argv=argv), 'Namespace execution proof')
    require(all(proof.get('certification', {}).get(key) is True for key in CERTIFICATIONS),
            'Namespace execution certification incomplete')
    if current:
        require(proof.get('namespaces') == launcher.namespace_identity(),
                'Execution proof is from a different user/mount/PID namespace')
    digest = hashlib.sha256(json.dumps(argv, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    require(proof.get('argv_sha256') == digest, 'Unchanged argv proof hash differs')
    source_manifest = launcher.confined(proof['source_manifest_path'])
    source_sha = admission.get('external_namespace_source_manifest_sha256')
    require(proof.get('source_manifest_sha256') == source_sha and
            launcher.verify_namespace_manifest(source_manifest, source_sha) == source_sha,
            'Admitted namespace source payloads changed')
    qualification = launcher.confined(admission['external_namespace_qualification_receipt'])
    qualification_sha = admission.get('external_namespace_qualification_receipt_sha256')
    require(qualification_sha and launcher.sha(qualification) == qualification_sha and
            proof.get('qualification_path') == str(qualification) and
            proof.get('qualification_sha256') == qualification_sha, 'Namespace qualification changed')
    qualified = launcher.read_json(qualification)
    matches(qualified, dict(schema='namespace-write-guard77-qualification-v1', qualification_passed=True,
                            actual_repo=str(launcher.REPO), policy_sha256=launcher.sha(policy_path),
                            source_manifest_sha256=source_sha), 'Successful namespace qualification')
    return dict(policy=dict(path=str(policy_path), sha256=launcher.sha(policy_path)),
                proof=dict(path=str(proof_path), sha256=launcher.sha(proof_path), prebound_in_admission=False),
                source_manifest=dict(path=str(source_manifest), sha256=source_sha),
                qualification=dict(path=str(qualification), sha256=qualification_sha),
                argv_sha256=digest, certification=proof['certification'],
                namespaces=proof.get('namespaces'), current_namespace_ids_checked=current,
                kernel_write_enforcement_certified_by_this_preparation=False)


def normal_runtime_evidence(launcher, family_admission):
    matches(family_admission, dict(runtime_boundary_mode='environment_and_explicit_repo_paths_only',
                                   user_allows_incidental_runtime_caches_outside_repo=True),
            'Existing user-authorized normal family runtime')
    path = launcher.confined(family_admission['root_normal_runtime_qualification_receipt'])
    digest = family_admission.get('root_normal_runtime_qualification_sha256')
    require(digest and launcher.sha(path) == digest, 'Admitted normal-runtime qualification bytes changed')
    receipt = launcher.read_json(path)
    matches(receipt, dict(schema='buddy77-normal-runtime-qualification-v1', passed=True, torch_version='2.7.1',
                          physical_GPU_UUIDs=list(launcher.UUIDS), scientific_data_opened=False,
                          model_training_executed=False), 'Existing successful normal Torch/CUDA qualification')
    devices = receipt.get('devices', [])
    require(len(devices) == 2 and {row.get('device') for row in devices} == {0, 1} and
            all(row.get('data_free_matrix_check_passed') is True for row in devices),
            'Both intended GPU normal-runtime checks required')
    return dict(mode='environment_and_explicit_repo_paths_only', execution_guarded=False,
                user_allows_incidental_runtime_caches_outside_repo=True,
                normal_runtime_qualification=dict(path=str(path), sha256=digest),
                proof=dict(path=None, sha256=None, required=False),
                current_namespace_ids_checked=False, external_namespace_evidence_used=False,
                kernel_write_enforcement_certified_by_this_preparation=False)


def current_execution_boundary(launcher, admission_path, stage, gpu, here):
    require(stage in STAGE_SCHEMAS, 'Unknown post-family stage')
    admission_path = launcher.confined(admission_path)
    require(admission_path == here / STAGE_ADMISSIONS[stage], 'Use the separate packet-local stage admission')
    admission = launcher.read_json(admission_path)
    matches(admission, dict(schema=STAGE_SCHEMAS[stage], decision='admitted', family_cells=15,
                            optimizer_fits=24, epochs_per_cell=100, prior_partial_fits_excluded=True),
            'Existing prospective stage admission')
    mode = runtime_mode_gate(admission)
    if mode == 'environment_and_explicit_repo_paths_only':
        family_admission_path = launcher.confined(launcher.HERE / 'ROOT_ADMISSION.json')
        evidence = normal_runtime_evidence(launcher, launcher.read_json(family_admission_path))
        evidence.update(stage=stage, stage_admission_sha256=launcher.sha(admission_path), GPU_UUID=gpu,
                        family_admission_sha256=launcher.sha(family_admission_path),
                        inherited_FD_audit=launcher.inherited_fd_audit(launcher.REPO),
                        environment_paths_are_not_kernel_confinement=True)
        return evidence
    require(isinstance(admission.get('root_runtime_boundary_decision'), str) and
            bool(admission['root_runtime_boundary_decision'].strip()), 'Root runtime-boundary decision missing')
    policy = os.environ.get('BUDDY_NAMESPACE_POLICY_PATH')
    proof = os.environ.get('BUDDY_NAMESPACE_PROOF_PATH')
    require(policy and proof, 'Executing every post-family stage requires a qualified external namespace guard')
    script = here / ('export_predictions77.py' if stage == 'export' else 'evaluate77.py')
    argv = list(getattr(sys, 'orig_argv', [str(launcher.PYTHON), '-B', *sys.argv]))
    require(argv[:3] == [str(launcher.PYTHON), '-B', str(script)] and sys.flags.dont_write_bytecode,
            'Use pinned Python -B and the absolute post-family script path')
    evidence = bound_namespace_proof(launcher, policy, proof, admission, argv, current=True)
    evidence.update(mode=mode, execution_guarded=True, stage=stage, stage_admission_sha256=launcher.sha(admission_path), GPU_UUID=gpu,
                    inherited_FD_audit=launcher.inherited_fd_audit(launcher.REPO),
                    environment_paths_are_not_kernel_confinement=True)
    return evidence


def historical_family_boundary(launcher, terminal, runtime_path, admission_path, admission):
    runtime_path = launcher.confined(runtime_path)
    runtime = launcher.read_json(runtime_path)
    runtime_sha = launcher.sha(runtime_path)
    mode = admission.get('runtime_boundary_mode')
    require(mode in {'environment_and_explicit_repo_paths_only', 'external_namespace_profile_bound'},
            'Unknown admitted family runtime mode')
    matches(runtime, dict(schema='buddy77-repo-runtime-paths-v1',
                          operational_admission_sha256=launcher.sha(admission_path),
                          runtime_root=str(launcher.RUNTIME), variables=path_environment(launcher.REPO, launcher.RUNTIME),
                          boundary_mode=mode,
                          kernel_write_enforcement_certified_by_this_launcher=False,
                          environment_paths_are_not_kernel_confinement=True, prior_partial_fits_excluded=True),
            'New-family runtime evidence')
    matches(terminal, dict(schema='buddy77-family-launch-receipt-v2', runtime_environment_sha256=runtime_sha,
                           prior_partial_fits_excluded=True,
                           predecessor_resource_is_runtime_observation_not_confinement_proof=True),
            'New-family terminal runtime binding')
    require(terminal.get('context', {}).get('runtime_environment_sha256') == runtime_sha and
            terminal['context'].get('operational_repair_admission_sha256') == launcher.sha(admission_path),
            'Terminal context does not bind new-family runtime/admission')
    require(runtime.get('inherited_FD_audit', {}).get('outside_writable_regular_file_FDs_observed') is False,
            'New-family inherited writable regular-file evidence incomplete')
    profile = runtime['external_namespace_profile']
    if mode == 'environment_and_explicit_repo_paths_only':
        require(profile == {}, 'Ordinary family runtime contains an unexpected namespace profile')
        return normal_runtime_evidence(launcher, admission)
    expected_argv = [str(launcher.PYTHON), '-B', str(launcher.HERE / 'launch77.py'), 'family', '--execute',
                     '--admission', str(admission_path)]
    evidence = bound_namespace_proof(launcher, profile['policy']['path'], profile['proof']['path'],
                                    admission, expected_argv, current=False)
    for key in ['policy','proof','source_manifest','qualification']:
        matches(profile[key], evidence[key], 'Historical family ' + key + ' evidence')
    require(profile.get('argv_sha256') == evidence['argv_sha256'] and
            profile.get('certification') == evidence['certification'], 'Historical family proof profile changed')
    return evidence

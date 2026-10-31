"""Nested project-only metadata fixtures. No real archive/tensor/Torch/SSH work."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate77 as preparation
from runtime_boundary import bound_namespace_proof, current_execution_boundary, historical_family_boundary, normal_runtime_evidence
from gate_contract import audit_admission_gate
from runtime_paths import PATHS, create_paths, path_environment

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixture_sealed_launcher_v3',
    HERE.parent / 'buddy_gpu77_resource_family_launcher_v3/launch77.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)  # stdlib source only; no main/context execution


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_normal_receipt(f):
    qualification = launcher.HERE / 'normal_runtime/QUALIFICATION.json'
    write(qualification, dict(schema='buddy77-normal-runtime-qualification-v1', passed=True, torch_version='2.7.1',
        physical_GPU_UUIDs=list(launcher.UUIDS), scientific_data_opened=False, model_training_executed=False,
        devices=[dict(device=index,data_free_matrix_check_passed=True) for index in [0,1]]))
    family_admission = dict(runtime_boundary_mode='environment_and_explicit_repo_paths_only',
        user_allows_incidental_runtime_caches_outside_repo=True,
        root_normal_runtime_qualification_receipt=str(qualification),
        root_normal_runtime_qualification_sha256=digest(qualification))
    write(launcher.HERE / 'ROOT_ADMISSION.json', family_admission)
    return family_admission,qualification


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='fixture_boundary_', dir=HERE) as folder:
        repo = Path(folder).resolve() / 'repo'; repo.mkdir()
        packet, launcherdir, guard = (repo / name for name in ['preparation','launcher','guard'])
        for path in [packet, launcherdir, guard]: path.mkdir()
        python = repo / 'python-placeholder'
        policy, proof, qual = guard / 'MOUNT_POLICY.json', repo / 'current_proof.json', repo / 'qualification.json'
        write(policy, dict(schema='namespace-write-guard77-policy-v1', actual_repo=str(repo)))
        manifest = guard / 'SOURCE_MANIFEST.json'
        write(manifest, dict(schema='namespace-write-guard77-source-manifest-v1', actual_repo=str(repo),
                             files={'MOUNT_POLICY.json':digest(policy)}))
        write(qual, dict(schema='namespace-write-guard77-qualification-v1', qualification_passed=True,
            actual_repo=str(repo), policy_sha256=digest(policy), source_manifest_sha256=digest(manifest)))
        fields = dict(external_namespace_policy_sha256=digest(policy),
            external_namespace_source_manifest_sha256=digest(manifest),
            external_namespace_qualification_receipt=str(qual),
            external_namespace_qualification_receipt_sha256=digest(qual),
            runtime_boundary_mode='external_namespace_profile_bound', execution_guarded=True,
            prior_partial_fits_excluded=True, root_runtime_boundary_decision='Fixture only')
        argv = [str(python), '-B', str(packet / 'evaluate77.py'), 'audit-lock', '--execute']
        ns = dict(user='fixture-user', mnt='fixture-mnt', pid='fixture-pid')
        cert = {key:True for key in ['outside_mounts_read_only','write_root_bind_writable',
            'private_mount_propagation','private_pid_proc','inherited_extra_fds_closed','capabilities_zero','no_new_privs']}
        proof_value = dict(schema='namespace-write-guard77-execution-proof-v1', mode='execute', actual_repo=str(repo),
            write_root=str(repo), policy_path=str(policy), policy_sha256=digest(policy), proof_path=str(proof),
            source_manifest_path=str(manifest), source_manifest_sha256=digest(manifest), qualification_path=str(qual),
            qualification_sha256=digest(qual), argv=argv, certification=cert, namespaces=ns,
            argv_sha256=hashlib.sha256(json.dumps(argv,ensure_ascii=True,separators=(',', ':')).encode()).hexdigest())
        write(proof, proof_value)
        admission_path = packet / 'ROOT_LOCK_AUDIT_ADMISSION.json'
        admission = dict(fields, schema='buddy77-postfamily-lock-audit-admission-v1', decision='admitted',
            family_cells=15, optimizer_fits=24, epochs_per_cell=100)
        write(admission_path, admission)
        with patch.multiple(launcher, REPO=repo, HERE=launcherdir, PYTHON=python,
                RUNTIME=launcherdir / 'root_runtime_v1'), \
                patch.object(launcher, 'namespace_identity', return_value=ns), \
                patch.object(launcher, 'inherited_fd_audit', return_value={
                    'outside_writable_regular_file_FDs_observed':False,'scope':'fixture metadata'}), \
                patch.object(sys, 'orig_argv', argv, create=True), patch.dict(os.environ, {
                    'BUDDY_NAMESPACE_POLICY_PATH':str(policy), 'BUDDY_NAMESPACE_PROOF_PATH':str(proof)}, clear=True):
            yield dict(repo=repo, packet=packet, python=python, policy=policy, proof=proof, qual=qual,
                manifest=manifest, fields=fields, argv=argv, ns=ns, cert=cert, proof_value=proof_value,
                admission=admission, admission_path=admission_path)


class BoundaryFixtures(unittest.TestCase):
    def test_authorized_normal_stage_uses_existing_receipt_without_namespace_or_new_admission(self):
        with fixture() as f:
            family_admission,qualification = install_normal_receipt(f)
            admission = dict(schema='buddy77-postfamily-lock-audit-admission-v1',decision='admitted',
                family_cells=15,optimizer_fits=24,epochs_per_cell=100,prior_partial_fits_excluded=True,
                full_terminal_and_all_training_ledgers_reviewed=True,audit_once=True,
                official_test_access_authorized=False,other_jobs_stopped=False,root_audit_cost_decision='Fixture only')
            write(f['admission_path'],admission)
            with patch.dict(os.environ,{},clear=True), \
                    patch.object(launcher,'namespace_identity',side_effect=AssertionError('Namespace read in ordinary mode')), \
                    patch.object(launcher,'verify_namespace_manifest',side_effect=AssertionError('Namespace manifest read in ordinary mode')):
                evidence = current_execution_boundary(launcher,f['admission_path'],'audit-lock','fixture-GPU',f['packet'])
                audit_admission_gate(admission,dict(prior_partial_fits_excluded=True))
            self.assertFalse(evidence['execution_guarded'])
            self.assertFalse(evidence['external_namespace_evidence_used'])
            self.assertFalse(evidence['current_namespace_ids_checked'])
            self.assertIsNone(evidence['proof']['sha256'])
            self.assertEqual(evidence['normal_runtime_qualification']['sha256'],digest(qualification))
            self.assertTrue(evidence['user_allows_incidental_runtime_caches_outside_repo'])

    def test_normal_runtime_receipt_and_user_exception_remain_bound(self):
        with fixture() as f:
            admission,qualification = install_normal_receipt(f)
            normal_runtime_evidence(launcher,admission)
            for changed in [dict(admission,user_allows_incidental_runtime_caches_outside_repo=False),
                            dict(admission,root_normal_runtime_qualification_sha256='0'*64)]:
                with self.assertRaises(RuntimeError):
                    normal_runtime_evidence(launcher,changed)
            receipt = launcher.read_json(qualification);receipt['passed']=False;write(qualification,receipt)
            with self.assertRaises(RuntimeError):
                normal_runtime_evidence(launcher,dict(admission,root_normal_runtime_qualification_sha256=digest(qualification)))

    def test_normal_family_runtime_has_no_namespace_profile(self):
        with fixture() as f:
            admission,qualification = install_normal_receipt(f)
            admission_path = launcher.HERE / 'ROOT_ADMISSION.json'
            runtime_path = launcher.HERE / 'root_family_v2/RUNTIME_ENVIRONMENT.json'
            runtime = dict(schema='buddy77-repo-runtime-paths-v1',operational_admission_sha256=digest(admission_path),
                runtime_root=str(launcher.RUNTIME),variables=path_environment(f['repo'],launcher.RUNTIME),
                boundary_mode='environment_and_explicit_repo_paths_only',kernel_write_enforcement_certified_by_this_launcher=False,
                environment_paths_are_not_kernel_confinement=True,prior_partial_fits_excluded=True,
                inherited_FD_audit=dict(outside_writable_regular_file_FDs_observed=False),external_namespace_profile={})
            write(runtime_path,runtime)
            terminal = dict(schema='buddy77-family-launch-receipt-v2',runtime_environment_sha256=digest(runtime_path),
                prior_partial_fits_excluded=True,predecessor_resource_is_runtime_observation_not_confinement_proof=True,
                context=dict(runtime_environment_sha256=digest(runtime_path),operational_repair_admission_sha256=digest(admission_path)))
            with patch.object(launcher,'namespace_identity',side_effect=AssertionError('Historical namespace read in ordinary mode')):
                evidence = historical_family_boundary(launcher,terminal,runtime_path,admission_path,admission)
            self.assertIsNone(evidence['proof']['sha256'])
            self.assertEqual(evidence['normal_runtime_qualification']['sha256'],digest(qualification))
            runtime['external_namespace_profile']={'unexpected':True};write(runtime_path,runtime)
            terminal['runtime_environment_sha256']=digest(runtime_path)
            terminal['context']['runtime_environment_sha256']=digest(runtime_path)
            with self.assertRaises(RuntimeError):
                historical_family_boundary(launcher,terminal,runtime_path,admission_path,admission)

    def test_current_stage_requires_guard_and_binds_dynamic_proof(self):
        with fixture() as f:
            evidence = current_execution_boundary(launcher, f['admission_path'], 'audit-lock', 'fixture-GPU', f['packet'])
            self.assertEqual(evidence['proof']['sha256'], digest(f['proof']))
            self.assertFalse(evidence['proof']['prebound_in_admission'])
            self.assertTrue(evidence['current_namespace_ids_checked'])
            self.assertFalse(evidence['kernel_write_enforcement_certified_by_this_preparation'])
            for value in [dict(f['proof_value'], namespaces={}), dict(f['proof_value'], mode='qualify'),
                    dict(f['proof_value'], certification=dict(f['cert'], no_new_privs=False)),
                    dict(f['proof_value'], argv=[]), dict(f['proof_value'], write_root=str(f['repo'].parent))]:
                write(f['proof'], value)
                with self.assertRaises(RuntimeError):
                    current_execution_boundary(launcher, f['admission_path'], 'audit-lock', 'fixture-GPU', f['packet'])
            write(f['proof'], f['proof_value'])
            os.environ.pop('BUDDY_NAMESPACE_PROOF_PATH')
            with self.assertRaises(RuntimeError):
                current_execution_boundary(launcher, f['admission_path'], 'audit-lock', 'fixture-GPU', f['packet'])

    def test_unqualified_or_unbound_stage_cannot_preflight(self):
        with fixture() as f:
            for changed in [dict(f['admission'], execution_guarded=False),
                    dict(f['admission'], prior_partial_fits_excluded=False),
                    dict(f['admission'], external_namespace_source_manifest_sha256='0'*64),
                    dict(f['admission'], root_runtime_boundary_decision='')]:
                write(f['admission_path'], changed)
                with self.assertRaises(RuntimeError):
                    current_execution_boundary(launcher, f['admission_path'], 'audit-lock', 'fixture-GPU', f['packet'])
            write(f['admission_path'], f['admission'])
            bad_qual = launcher.read_json(f['qual']); bad_qual['qualification_passed'] = False
            write(f['qual'], bad_qual)
            with self.assertRaises(RuntimeError):
                current_execution_boundary(launcher, f['admission_path'], 'audit-lock', 'fixture-GPU', f['packet'])

    def test_historical_new_family_evidence_stays_separate_from_current_guard(self):
        with fixture() as f:
            admission_path = launcher.HERE / 'ROOT_ADMISSION.json'
            write(admission_path, f['fields'])
            proof_path = launcher.HERE / 'family_proof.json'
            argv = [str(launcher.PYTHON), '-B', str(launcher.HERE / 'launch77.py'), 'family', '--execute',
                    '--admission', str(admission_path)]
            proof = dict(f['proof_value'], argv=argv, proof_path=str(proof_path),
                namespaces=dict(user='old-user',mnt='old-mnt',pid='old-pid'),
                argv_sha256=hashlib.sha256(json.dumps(argv,ensure_ascii=True,separators=(',', ':')).encode()).hexdigest())
            write(proof_path, proof)
            evidence = bound_namespace_proof(launcher, f['policy'], proof_path, f['fields'], argv, current=False)
            profile = {key:evidence[key] for key in ['policy','proof','source_manifest','qualification','argv_sha256','certification']}
            runtime_path = launcher.HERE / 'root_family_v2/RUNTIME_ENVIRONMENT.json'
            runtime = dict(schema='buddy77-repo-runtime-paths-v1', operational_admission_sha256=digest(admission_path),
                runtime_root=str(launcher.RUNTIME), variables=path_environment(f['repo'],launcher.RUNTIME),
                boundary_mode='external_namespace_profile_bound', kernel_write_enforcement_certified_by_this_launcher=False,
                environment_paths_are_not_kernel_confinement=True, prior_partial_fits_excluded=True,
                inherited_FD_audit=dict(outside_writable_regular_file_FDs_observed=False),
                external_namespace_profile=profile)
            write(runtime_path,runtime)
            terminal = dict(schema='buddy77-family-launch-receipt-v2', runtime_environment_sha256=digest(runtime_path),
                prior_partial_fits_excluded=True, predecessor_resource_is_runtime_observation_not_confinement_proof=True,
                context=dict(runtime_environment_sha256=digest(runtime_path),operational_repair_admission_sha256=digest(admission_path)))
            bound = historical_family_boundary(launcher,terminal,runtime_path,admission_path,f['fields'])
            self.assertFalse(bound['current_namespace_ids_checked'])
            self.assertEqual(bound['proof']['sha256'],digest(proof_path))
            for changed in [dict(terminal,schema='buddy77-family-launch-receipt-v1'),
                    dict(terminal,prior_partial_fits_excluded=False), dict(terminal,runtime_environment_sha256='0'*64)]:
                with self.assertRaises(RuntimeError):
                    historical_family_boundary(launcher,changed,runtime_path,admission_path,f['fields'])
            write(proof_path,dict(proof,mode='qualify'))
            with self.assertRaises(RuntimeError):
                historical_family_boundary(launcher,terminal,runtime_path,admission_path,f['fields'])

    def test_repo_runtime_and_scientific_source_are_preserved(self):
        with fixture() as f:
            output = f['packet'] / 'stage'; output.mkdir()
            with patch.object(preparation,'REPO',f['repo']), patch.dict(os.environ,{},clear=True):
                env, values = preparation.owned_runtime_environment(launcher,output)
                self.assertEqual(env['TMP'],values['TMPDIR'])
                self.assertEqual(env['TEMP'],values['TMPDIR'])
                for key in PATHS:
                    self.assertTrue(Path(values[key]).is_dir())
                    self.assertTrue(Path(values[key]).is_relative_to(f['repo']))
        self.assertEqual(preparation.LAUNCHER.name,'buddy_gpu77_resource_family_launcher_v3')
        self.assertEqual(preparation.RUNS.parent.name,'root_family_v2')
        self.assertEqual(preparation.TERMINAL.parent,preparation.RUNS.parent)
        self.assertEqual(preparation.RESOURCE.parent.parent.name,'buddy_gpu77_resource_family_launcher_v2')
        self.assertEqual(preparation.ARCHIVE.name,'collab_stream77_v1.zip')
        previous=HERE.parent/'buddy_gpu77_postfamily_eval_preparation_v2'
        asts=lambda path: {node.name:node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
        old,new=asts(previous/'evaluate77.py'),asts(HERE/'evaluate77.py')
        for name in ['load_module','bootstrap_staging_module','lock_command','score_command','deny_existing_test_artifacts',
                'stage_authentic_test','qualify_official_test','qualify_final_cache']:
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
        scientific=lambda node:next(item for item in node.body if isinstance(item,ast.Try))
        self.assertEqual(ast.dump(scientific(old['audit_lock']),include_attributes=False),
                         ast.dump(scientific(new['audit_lock']),include_attributes=False))
        new_primary=scientific(new['evaluate'])
        added=[item for item in new_primary.body if isinstance(item,ast.Expr) and
            isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Attribute) and
            isinstance(item.value.func.value,ast.Name) and item.value.func.value.id=='env' and
            item.value.func.attr=='update' and len(item.value.args)==1 and
            isinstance(item.value.args[0],ast.Name) and item.value.args[0].id=='runtime_paths']
        self.assertEqual(len(added),1)
        new_primary.body.remove(added[0])
        self.assertEqual(ast.dump(scientific(old['evaluate']),include_attributes=False),
                         ast.dump(new_primary,include_attributes=False))
        old,new=asts(previous/'export_predictions77.py'),asts(HERE/'export_predictions77.py')
        self.assertEqual(ast.dump(scientific(old['export']),include_attributes=False),
                         ast.dump(scientific(new['export']),include_attributes=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)

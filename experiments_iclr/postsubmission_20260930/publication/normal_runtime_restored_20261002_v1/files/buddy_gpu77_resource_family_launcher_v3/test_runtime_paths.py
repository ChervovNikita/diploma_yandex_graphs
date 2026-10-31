"""Project-only stdlib fixtures. No Torch, tensors, GPU, archive or remote work."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import launch77 as launcher
from runtime_paths import PATHS, canonical_inside, create_paths, path_environment, writable_regular_fd_gate

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def fixture():
    # Both the allowed root and its simulated external sibling remain in project.
    with tempfile.TemporaryDirectory(prefix='fixture_runtime_paths_', dir=HERE) as folder:
        base = Path(folder).resolve()
        repo, outside = base / 'repo', base / 'outside'
        repo.mkdir(); outside.mkdir()
        yield repo, outside


class RuntimePathFixtures(unittest.TestCase):
    def test_paths_reused_after_boundary_qualification(self):
        with fixture() as (repo, outside):
            root = repo / 'packet/root_runtime_v1'
            values = create_paths(repo, root)
            self.assertEqual(values, create_paths(repo, root))
            self.assertEqual(set(PATHS), {key for key in values if key in PATHS})
            for name in PATHS:
                path = Path(values[name])
                self.assertTrue(path.is_dir())
                self.assertEqual(path, path.resolve())
                self.assertTrue(path.is_relative_to(repo))
            self.assertEqual(values['PYTHONDONTWRITEBYTECODE'], '1')
            self.assertEqual(values['CUDA_CACHE_DISABLE'], '1')

    def test_external_nested_root_rejected(self):
        with fixture() as (repo, outside):
            with self.assertRaises(RuntimeError):
                path_environment(repo, outside / 'runtime')
            with self.assertRaises(RuntimeError):
                canonical_inside(repo, repo)

    def test_symlink_and_noncanonical_traversal_rejected(self):
        with fixture() as (repo, outside):
            (repo / 'linked').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                create_paths(repo, repo / 'linked/runtime')
            with self.assertRaises(RuntimeError):
                canonical_inside(repo / 'child/../runtime', repo)

    def test_writable_regular_fd_metadata_gate(self):
        with fixture() as (repo, outside):
            bad = dict(regular_file=True, writable=True, target=str(outside / 'simulated.log'))
            with self.assertRaises(RuntimeError):
                writable_regular_fd_gate([bad], repo)
            writable_regular_fd_gate([dict(bad, writable=False), dict(bad, regular_file=False),
                                      dict(bad, target=str(repo / 'inside.log'))], repo)

    def test_fresh_family_is_separate_and_exclusive(self):
        with fixture() as (repo, outside):
            packet = repo / 'new_packet'; packet.mkdir()
            runtime, family = packet / 'root_runtime_v1', packet / 'root_family_v2'
            previous = repo / 'previous/root_family_v1/metadata.json'
            write(previous, dict(preserved=True))
            old_sha = digest(previous)
            create_paths(repo, runtime)  # fixture boundary qualification prepares runtime only
            admission_path = repo / 'admission.json'; write(admission_path, {})
            admission = dict(runtime_boundary_mode='environment_and_explicit_repo_paths_only')
            with patch.multiple(launcher, REPO=repo, HERE=packet, RUNTIME=runtime, FAMILY=family), \
                    patch.object(launcher, 'inherited_fd_audit', return_value=dict(scope='fixture metadata')), \
                    patch.dict(os.environ, {}, clear=True):
                record = launcher.prepare_runtime(admission_path, admission)
                self.assertTrue((family / 'RUNTIME_ENVIRONMENT.json').is_file())
                self.assertFalse(record['kernel_write_enforcement_certified_by_this_launcher'])
                with self.assertRaises(FileExistsError):
                    launcher.prepare_runtime(admission_path, admission)
            self.assertEqual(digest(previous), old_sha)

    def test_environment_only_mode_is_explicit(self):
        with fixture() as (repo, outside), patch.dict(os.environ, {}, clear=True):
            mode, evidence = launcher.external_namespace_evidence(repo / 'admission.json',
                dict(runtime_boundary_mode='environment_and_explicit_repo_paths_only'))
            self.assertEqual(mode, 'environment_and_explicit_repo_paths_only')
            self.assertEqual(evidence, {})
            with self.assertRaises(RuntimeError):
                launcher.external_namespace_evidence(repo / 'admission.json',
                    dict(runtime_boundary_mode='external_namespace_profile_bound'))

    def test_prospective_admission_binds_custody_and_termination(self):
        with fixture() as (repo, outside):
            packet, source, data, qualdir = (repo / name for name in ['packet','source','data/run','qual'])
            for folder in [packet, source, data.parent]:
                write(folder / 'SOURCE_MANIFEST.json', dict(files=[]))
                write(folder / 'SEAL.json', dict(source_manifest_sha256=digest(folder / 'SOURCE_MANIFEST.json')))
            write(packet / 'CONTRACT.json', dict(data_wrapper_source_manifest_sha256=digest(data.parent / 'SOURCE_MANIFEST.json')))
            qual = qualdir / 'CPU_QUALIFICATION.json'; write(qual, {})
            write(qualdir / 'CPU_RUN_RECEIPT.json', {})
            cache = data / 'cache'; write(cache / 'manifest.json', {})
            write(data / 'QUALIFICATION.json', {})
            resource, family = repo / 'resource', packet / 'root_family_v2'
            write(resource / 'RESOURCE_RECEIPT.json', {})
            custody, pause, termination = (repo / (name + '.json') for name in ['custody','pause','termination'])
            write(custody, dict(fixture='custody')); write(pause, dict(fixture='pause'))
            terminated = dict(schema='buddy77-own-family-operational-termination-v1',
                status='all_original_owned_processes_terminated', other_jobs_targeted=False,
                partial_outputs_preserved=True)
            write(termination, terminated)
            source_sha = digest(source / 'SOURCE_MANIFEST.json')
            admission = dict(schema='buddy77-runtime-boundary-repair-admission-v1',
                decision='admitted_fresh_family', source_manifest_sha256=source_sha,
                wrapper_manifest_sha256=digest(packet / 'SOURCE_MANIFEST.json'),
                predecessor_wrapper_manifest_sha256=launcher.V2_SHA,
                cache_manifest_sha256=digest(cache / 'manifest.json'), CPU_qualification_sha256=digest(qual),
                CPU_execution_receipt_sha256=digest(qualdir / 'CPU_RUN_RECEIPT.json'),
                data_qualification_sha256=digest(data / 'QUALIFICATION.json'),
                resource_receipt_sha256=digest(resource / 'RESOURCE_RECEIPT.json'),
                prior_family_namespace=str(repo / 'buddy_gpu77_resource_family_launcher_v2/root_family_v1'),
                fresh_family_namespace=str(family), family_cells=15, optimizer_fits=24, epochs_per_cell=100,
                cpu_threads=4, physical_GPU_UUIDs=list(launcher.UUIDS), exclude_prior_partial_fits=True,
                preserve_prior_cost_and_failure_evidence=True, prior_own_processes_terminated_before_restart=True,
                heldout_metrics_inspected=False, root_runtime_boundary_decision='fixture only',
                prior_partial_preservation_receipt=str(custody), prior_partial_preservation_receipt_sha256=digest(custody),
                prior_pause_receipt=str(pause), prior_pause_receipt_sha256=digest(pause),
                prior_termination_receipt=str(termination), prior_termination_receipt_sha256=digest(termination))
            path = repo / 'admission.json'; write(path, admission)
            with patch.multiple(launcher, REPO=repo, PHASE=repo, HERE=packet, SOURCE=source,
                    SOURCE_SHA=source_sha, DATA_DIR=data, QUAL_DIR=qualdir, QUAL=qual, CACHE=cache,
                    RESOURCE=resource, FAMILY=family, PARTIAL_CUSTODY_SHA=digest(custody),
                    PAUSE_SHA=digest(pause), TERMINATION_SHA=digest(termination)):
                self.assertEqual(launcher.operational_admission(path)[1], admission)
                for changed in [dict(admission, heldout_metrics_inspected=True),
                        dict(admission, prior_own_processes_terminated_before_restart=False),
                        dict(admission, prior_partial_preservation_receipt_sha256='0'*64)]:
                    write(path, changed)
                    with self.assertRaises(RuntimeError):
                        launcher.operational_admission(path)
                write(termination, dict(terminated, other_jobs_targeted=True))
                write(path, dict(admission, prior_termination_receipt_sha256=digest(termination)))
                with patch.object(launcher, 'TERMINATION_SHA', digest(termination)), self.assertRaises(RuntimeError):
                    launcher.operational_admission(path)

    def test_external_profile_dynamic_proof_and_rejections(self):
        with fixture() as (repo, outside):
            packet, guard = repo / 'packet', repo / 'guard'
            packet.mkdir(); guard.mkdir()
            policy_path, proof_path = guard / 'POLICY.json', repo / 'execution/PROOF.json'
            write(policy_path, dict(schema='namespace-write-guard77-policy-v1', actual_repo=str(repo)))
            source = guard / 'SOURCE_MANIFEST.json'
            write(source, dict(schema='namespace-write-guard77-source-manifest-v1', actual_repo=str(repo),
                               files={'POLICY.json':digest(policy_path)}))
            qualification = repo / 'qualification.json'
            write(qualification, dict(schema='namespace-write-guard77-qualification-v1', qualification_passed=True,
                actual_repo=str(repo), policy_sha256=digest(policy_path), source_manifest_sha256=digest(source),
                fixture_only=True))
            admission_path = repo / 'admission.json'; write(admission_path, {})
            python = repo / 'python-placeholder'
            argv = [str(python), '-B', str(packet / 'launch77.py'), 'family', '--execute',
                    '--admission', str(admission_path)]
            cert = {key: True for key in ['outside_mounts_read_only','write_root_bind_writable',
                'private_mount_propagation','private_pid_proc','inherited_extra_fds_closed',
                'capabilities_zero','no_new_privs']}
            namespace_ids = dict(user='fixture-user', mnt='fixture-mnt', pid='fixture-pid')
            proof = dict(schema='namespace-write-guard77-execution-proof-v1', mode='execute',
                actual_repo=str(repo), write_root=str(repo), policy_path=str(policy_path),
                policy_sha256=digest(policy_path), proof_path=str(proof_path), source_manifest_path=str(source),
                source_manifest_sha256=digest(source), qualification_path=str(qualification),
                qualification_sha256=digest(qualification), argv=argv,
                argv_sha256=hashlib.sha256(json.dumps(argv, ensure_ascii=True,
                    separators=(',', ':')).encode('utf-8')).hexdigest(), certification=cert,
                namespaces=namespace_ids)
            write(proof_path, proof)
            admission = dict(runtime_boundary_mode='external_namespace_profile_bound',
                external_namespace_policy_sha256=digest(policy_path),
                external_namespace_source_manifest_sha256=digest(source),
                external_namespace_qualification_receipt=str(qualification),
                external_namespace_qualification_receipt_sha256=digest(qualification))
            with patch.multiple(launcher, REPO=repo, HERE=packet, PYTHON=python), \
                    patch.object(sys, 'orig_argv', argv, create=True), patch.dict(os.environ, {
                    'BUDDY_NAMESPACE_POLICY_PATH':str(policy_path),
                    'BUDDY_NAMESPACE_PROOF_PATH':str(proof_path)}, clear=True), \
                    patch.object(launcher, 'namespace_identity', return_value=namespace_ids):
                mode, evidence = launcher.external_namespace_evidence(admission_path, admission)
                self.assertEqual(mode, 'external_namespace_profile_bound')
                self.assertFalse(evidence['proof']['prebound_in_admission'])
                self.assertEqual(evidence['proof']['sha256'], digest(proof_path))
                for mutated in [dict(proof, mode='qualify'), dict(proof, argv=[]),
                        dict(proof, namespaces={}),
                        dict(proof, certification=dict(cert, private_pid_proc=False)),
                        dict(proof, qualification_sha256='0'*64)]:
                    write(proof_path, mutated)
                    with self.assertRaises(RuntimeError):
                        launcher.external_namespace_evidence(admission_path, admission)
                write(proof_path, proof)
                with self.assertRaises(RuntimeError):
                    launcher.external_namespace_evidence(admission_path,
                        dict(admission, external_namespace_policy_sha256='0'*64))
                os.environ.pop('BUDDY_NAMESPACE_PROOF_PATH')
                with self.assertRaises(RuntimeError):
                    launcher.external_namespace_evidence(admission_path, admission)

    def test_scientific_argv_unchanged_from_v2(self):
        old = ast.parse((HERE.parent / 'buddy_gpu77_resource_family_launcher_v2/launch77.py').read_text())
        new = ast.parse((HERE / 'launch77.py').read_text())
        functions = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                                  for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name in ['command', 'family_command', 'resource_evidence']:
            self.assertEqual(functions(old)[name], functions(new)[name])
        self.assertEqual(launcher.FAMILY.name, 'root_family_v2')
        self.assertEqual(launcher.RUNTIME.name, 'root_runtime_v1')
        self.assertEqual(launcher.RESOURCE.parent.name, 'buddy_gpu77_resource_family_launcher_v2')
        self.assertEqual(launcher.SEEDS, (0, 1, 2))
        self.assertEqual(len(launcher.ARMS), 5)
        contract = json.loads((HERE / 'CONTRACT.json').read_text())
        self.assertEqual((contract['family_cells'], contract['family_optimizer_fits'],
                          contract['family_epochs_per_cell'], contract['cpu_threads']), (15, 24, 100, 4))
        self.assertFalse(contract['resource_observation_is_confinement_proof'])


if __name__ == '__main__':
    unittest.main()

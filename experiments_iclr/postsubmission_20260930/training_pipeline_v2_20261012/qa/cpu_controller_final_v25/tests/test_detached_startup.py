"""Detached lifecycle checks with mocked children and a local Git remote."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from launcher import detach_host, receipt_authentication
from phase_rendezvous import MAX_COMPRESSED_BYTES, MAX_PROOF_BYTES
from spec import fingerprint


class DetachedStartupTests(unittest.TestCase):
    def fixture(self):
        plan = {'output_root': 'outputs', 'phase_rendezvous_capacity': {'all_hosts_fit_fixed_limits': True}, 'phase_rendezvous': {
            'schema': 'phase-git-rendezvous-v1', 'remote_name': 'origin',
            'remote_url_sha256': 'a' * 64,
            'max_compressed_host_bytes': MAX_COMPRESSED_BYTES,
            'max_uncompressed_proof_bytes': MAX_PROOF_BYTES}}
        return plan, fingerprint(plan)

    def test_go_and_remote_authentication_precede_detached_children(self):
        plan, digest = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('launcher.load_plan', side_effect=ValueError('No GO')), \
                 patch('launcher.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'No GO'):
                    detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                spawn.assert_not_called()
                self.assertFalse((root/'outputs').exists())
            with patch('launcher.load_plan', return_value=(plan, digest, {}, {}, {}, {})), \
                 patch('launcher.host_check'), \
                 patch('launcher.receipt_authentication', side_effect=RuntimeError('No authentication')), \
                 patch('launcher.subprocess.Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'No authentication'):
                    detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                spawn.assert_not_called()
                self.assertFalse((root/'outputs').exists())

    def test_sidecar_and_supervisor_are_detached_once_with_exact_pid_custody(self):
        plan, digest = self.fixture()
        children = [Mock(pid=101), Mock(pid=102)]
        for child in children:
            child.poll.return_value = None
        def identity(pid):
            return {'pid': pid, 'start_ticks': pid * 100, 'pgid': pid}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('launcher.load_plan', return_value=(plan, digest, {}, {}, {}, {})), \
                 patch('launcher.host_check'), \
                 patch('launcher.receipt_authentication', return_value=(root, plan['phase_rendezvous'])), \
                 patch('launcher.process_identity', side_effect=identity), \
                 patch('launcher.time.sleep'), \
                 patch('launcher.subprocess.Popen', side_effect=children) as spawn:
                value = detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                self.assertEqual(value['status'], 'started')
                self.assertFalse(value['phase_sync_requires_MacLink'])
                self.assertEqual(spawn.call_count, 2)
                commands = [call.args[0] for call in spawn.call_args_list]
                self.assertIn('--watch', commands[0])
                self.assertTrue(commands[0][3].endswith('phase_rendezvous.py'))
                self.assertTrue(commands[1][3].endswith('launcher.py'))
                for call in spawn.call_args_list:
                    self.assertTrue(call.kwargs['start_new_session'])
                    self.assertEqual(call.kwargs['stdin'], subprocess.DEVNULL)
                again = detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                self.assertEqual(again['status'], 'already_running')
                self.assertEqual(spawn.call_count, 2)
                self.assertEqual(json.loads((root/'outputs/allocation.startup-owner.json').read_text())['supervisor'], identity(102))

    def test_partial_startup_or_one_remaining_process_never_spawns_a_duplicate(self):
        plan, digest = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/'outputs'; output.mkdir()
            owner = {'host': 'allocation', 'plan_hash': digest,
                     'rendezvous': {'pid': 101, 'start_ticks': 10100, 'pgid': 101}}
            path = output/'allocation.startup-owner.json'
            with patch('launcher.load_plan', return_value=(plan, digest, {}, {}, {}, {})), \
                 patch('launcher.host_check'), \
                 patch('launcher.receipt_authentication', return_value=(root, plan['phase_rendezvous'])), \
                 patch('launcher.subprocess.Popen') as spawn:
                path.write_text(json.dumps(owner))
                with self.assertRaisesRegex(RuntimeError, 'Partial startup custody'):
                    detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                owner['supervisor'] = {'pid': 102, 'start_ticks': 10200, 'pgid': 102}
                path.write_text(json.dumps(owner))
                with patch('launcher.live_owned', side_effect=[False, True]):
                    with self.assertRaisesRegex(RuntimeError, 'Only one startup process'):
                        detach_host(root, root/'plan', root/'approval', 'allocation', Path('/python'))
                spawn.assert_not_called()

    @unittest.skipUnless(shutil.which('git'), 'Git is unavailable')
    def test_existing_remote_digest_and_read_access_are_checked_without_push(self):
        plan, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); repo = root/'source'; repo.mkdir()
            remote = root/'remote.git'
            def git(*args):
                subprocess.run(['git', *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git('init', '--bare', str(remote))
            git('-C', str(repo), 'init')
            git('-C', str(repo), 'remote', 'add', 'origin', str(remote))
            plan['phase_rendezvous']['remote_url_sha256'] = hashlib.sha256(str(remote).encode()).hexdigest()
            project = repo/'research'; project.mkdir()
            observed, _ = receipt_authentication(project, plan)
            self.assertEqual(observed, repo.resolve())
            plan['phase_rendezvous']['remote_url_sha256'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'differs from the approved'):
                receipt_authentication(project, plan)
            branches = subprocess.check_output(['git', '--git-dir', str(remote), 'for-each-ref']).strip()
            self.assertEqual(branches, b'')


if __name__ == '__main__':
    unittest.main()

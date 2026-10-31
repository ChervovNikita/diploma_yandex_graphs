"""Stdlib push-helper contract tests; all external commands are mocked."""
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import push_source_successor as helper


class PushTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='.push-test-', dir=Path(__file__).resolve().parents[4])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / '.git').mkdir()
        (self.root / '.git/index').write_bytes(b'unchanged live index')
        self.value = {'branch': helper.BRANCH, 'parent': helper.PARENT, 'live_head': helper.LIVE_HEAD,
                      'commit': 'd' * 40, 'live_head_unchanged': True, 'live_index_unchanged': True,
                      'working_tree_checkout_performed': False, 'publication_only': True,
                      'training_launch_performed': False, 'conditional_full_training_GO': True}
        self.input = self.root / 'COMMIT.json'
        self.output = self.root / 'PUSH.json'
        self.calls = []
        self.loaded = False
        self.changed_after_add = False
        self.bad_remote = False
        self.bad_live = False
        self.remote_queries = 0

    def check_output(self, argv, cwd, env=None, text=True):
        self.assertEqual(cwd, self.root)
        self.calls.append(argv)
        if argv[0] == 'ssh-agent':
            self.assertEqual(argv[1], '-a')
            self.assertEqual(argv[-1], '-s')
            path = Path(argv[2])
            self.assertTrue(path.is_relative_to(self.root))
            self.assertFalse(path.exists())
            self.socket = str(path)
            return 'SSH_AUTH_SOCK=' + str(path) + '; export SSH_AUTH_SOCK;\nSSH_AGENT_PID=123; export SSH_AGENT_PID;\n'
        self.assertEqual(argv[0], 'git')
        args = argv[1:]
        if args == ['symbolic-ref', 'HEAD']:
            return 'refs/heads/live-pilot'
        if args == ['rev-parse', 'HEAD']:
            return helper.PARENT if self.bad_live else helper.LIVE_HEAD
        if args == ['rev-parse', 'refs/heads/' + helper.BRANCH]:
            return 'e' * 40 if self.changed_after_add and self.loaded else self.value['commit']
        if args == ['rev-parse', self.value['commit'] + '^']:
            return helper.PARENT
        if args == ['rev-parse', '--git-path', 'index']:
            return '.git/index'
        if args == ['remote', 'get-url', '--all', 'origin']:
            return helper.ORIGIN
        if args == ['remote', 'get-url', '--push', '--all', 'origin']:
            return 'git@example.invalid:other.git' if self.bad_remote else helper.ORIGIN
        if args == ['ls-remote', '--exit-code', 'origin', 'refs/heads/' + helper.BRANCH]:
            self.remote_queries += 1
            return (helper.PARENT if self.remote_queries == 1 else self.value['commit']) + '\trefs/heads/' + helper.BRANCH
        raise AssertionError('Unexpected external command: ' + repr(argv))

    def mock_run(self, argv, cwd, env=None, **kwargs):
        self.assertEqual(cwd, self.root)
        self.calls.append(argv)
        self.assertEqual(env['SSH_AUTH_SOCK'], self.socket)
        ssh_args = shlex.split(env['GIT_SSH_COMMAND'])
        self.assertIn('IdentityAgent=' + self.socket, ssh_args)
        self.assertIn('StrictHostKeyChecking=yes', ssh_args)
        if argv == ['ssh-add', helper.KEY]:
            self.loaded = True
            self.assertNotIn('capture_output', kwargs)
            self.assertNotIn('input', kwargs)
        elif argv == ['ssh-agent', '-k']:
            pass
        else:
            self.assertEqual(argv, ['git', 'push', '--no-follow-tags', 'origin',
                                    self.value['commit'] + ':refs/heads/' + helper.BRANCH])
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def invoke(self):
        self.input.write_text(json.dumps(self.value))
        with patch.object(helper, 'ROOT', self.root), patch.object(helper, 'verify_route'), \
             patch.object(helper.subprocess, 'check_output', side_effect=self.check_output), \
             patch.object(helper.subprocess, 'run', side_effect=self.mock_run):
            return helper.push_source(self.input, self.output)

    def test_distinct_parent_live_head_exact_push_and_cleanup(self):
        report = self.invoke()
        self.assertNotEqual(report['parent'], report['live_head'])
        self.assertTrue(report['verified'])
        self.assertTrue(report['conditional_full_training_GO'])
        self.assertTrue(report['live_index_unchanged'])
        self.assertFalse(report['training_launch_performed'])
        self.assertFalse(report['passphrase_recorded'])
        self.assertNotIn('full_campaign_GO', report)
        self.assertEqual(self.calls[-1], ['ssh-agent', '-k'])
        self.assertEqual((self.root / '.git/index').read_bytes(), b'unchanged live index')
        self.assertEqual(json.loads(self.output.read_text()), report)
        self.assertNotIn('SSH_AUTH_SOCK', self.output.read_text())
        self.assertFalse(any(a[0] == 'git' and a[1] in ('checkout', 'reset', 'switch', 'add') for a in self.calls))

    def test_old_parent_cannot_substitute_for_live_head(self):
        self.bad_live = True
        with self.assertRaisesRegex(ValueError, 'Live or source branch changed'):
            self.invoke()
        self.assertFalse(any(a[0] == 'ssh-agent' for a in self.calls))

    def test_conditional_go_and_exact_push_origin_required(self):
        self.value['conditional_full_training_GO'] = False
        with self.assertRaisesRegex(ValueError, 'conditional all-fit'):
            self.invoke()
        self.assertEqual(self.calls, [])
        self.value['conditional_full_training_GO'] = True
        self.bad_remote = True
        with self.assertRaisesRegex(ValueError, 'Unexpected Git remote'):
            self.invoke()
        self.assertFalse(any(a[0] == 'ssh-agent' for a in self.calls))

    def test_source_ref_change_during_unlock_is_not_pushed(self):
        self.changed_after_add = True
        with self.assertRaisesRegex(ValueError, 'Live or source branch changed'):
            self.invoke()
        self.assertFalse(any(a[:2] == ['git', 'push'] for a in self.calls))
        self.assertEqual(self.calls[-1], ['ssh-agent', '-k'])
        report = json.loads(self.output.read_text())
        self.assertFalse(report['verified'])
        self.assertTrue(report['conditional_full_training_GO'])


if __name__ == '__main__':
    unittest.main()

"""Exercise the publisher contract with a scalar Git mock; no Git invocation."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import publish_source_branch as publisher


class PublishTests(unittest.TestCase):
    def test_explicit_parent_preserves_distinct_live_head_and_index(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            index = root / '.git/index'
            index.parent.mkdir()
            index.write_bytes(b'unchanged live index')
            source = root / 'staging/AUTHORIZATION.json'
            source.parent.mkdir()
            source.write_text(json.dumps({'training_authorized_after_all_fit_smokes_pass': True, 'expected_training_fits': 6460}))
            target = 'experiments_iclr/postsubmission_20260930/AUTHORIZATION.json'
            raw = source.read_bytes()
            row = {'path': target, 'source_path': 'staging/AUTHORIZATION.json', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            value = {'reviewed': True, 'publication_only': True, 'conditional_full_training_GO': True,
                'expected_parent': '2' * 40, 'expected_live_head': '4' * 40, 'branch': 'codex/new-source',
                'allowed_target_prefix': 'experiments_iclr/postsubmission_20260930', 'files': [row],
                'authorization': {k: row[k] for k in ('path', 'bytes', 'sha256')}, 'message': 'source publication'}
            selection = root / 'SELECTION.json'
            selection.write_text(json.dumps(value))
            receipt = root / 'RECEIPT.json'
            calls = []

            def fake_git(argv, cwd, env=None, input=None, text=True):
                self.assertEqual(cwd, root)
                args = argv[1:]
                calls.append(tuple(args))
                if args == ['rev-parse', 'HEAD']:
                    return '4' * 40
                if args == ['symbolic-ref', 'HEAD']:
                    return 'refs/heads/live-pilot'
                if args == ['rev-parse', '2' * 40 + '^{commit}']:
                    return '2' * 40
                if args == ['rev-parse', '--git-path', 'index']:
                    return '.git/index'
                if args[0] == 'read-tree':
                    self.assertEqual(args[1], '2' * 40)
                    Path(env['GIT_INDEX_FILE']).write_bytes(b'temporary index')
                if args[0] == 'hash-object':
                    self.assertEqual(input, raw)
                    self.assertFalse(text)
                    return ('b' * 40).encode()
                if args[0] == 'write-tree':
                    return 'c' * 40
                if args[0] == 'diff-tree':
                    self.assertEqual(args[-2], '2' * 40)
                    return target + '\x00'
                if args[0] == 'commit-tree':
                    self.assertEqual(args[-2:], ['-p', '2' * 40])
                    self.assertEqual(input, 'source publication\n')
                    return 'd' * 40
                return ''

            with patch.object(publisher.subprocess, 'check_output', side_effect=fake_git), patch.object(publisher.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                result = publisher.publish(root, selection, receipt)
            self.assertEqual(result['parent'], '2' * 40)
            self.assertEqual(result['live_head'], '4' * 40)
            self.assertEqual(index.read_bytes(), b'unchanged live index')
            self.assertEqual(source.read_bytes(), raw)
            self.assertFalse(receipt.with_suffix('.index').exists())
            self.assertTrue(result['conditional_full_training_GO'])
            self.assertFalse(result['training_launch_performed'])
            self.assertFalse(any(call[0] in ('checkout', 'reset', 'switch', 'push', 'add') for call in calls))
            self.assertEqual([call for call in calls if call[0] == 'update-ref'], [('update-ref', 'refs/heads/codex/new-source', 'd' * 40, '0' * 40)])

    def test_escaped_paths_refused(self):
        for value in ('../escape.py', '/absolute.py', 'a/../../escape.py', '.git/config', 'a\nfile.json'):
            with self.subTest(path=value), self.assertRaises(ValueError):
                publisher.safe_relative(value)


if __name__ == '__main__':
    unittest.main()

"""Local plaintext chunk and fixed-proof checks; no remote operations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resume_release as resume


class ResumeTest(unittest.TestCase):
    def test_small_commands_preserve_exact_text_and_fresh_temporary(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'candidate.json'
            source.write_text(''.join('  "fixture-%05d": "plain text",\n' % i for i in range(2000)))
            destination = resume.Path(resume.ROOTS['server51']) / 'fixture/CANDIDATE_GLOBAL_SMOKE_PASS.json'
            transport = resume.SmallTransport(Path(temp) / 'audit')
            commands = []
            def call(host, command, timeout=180):
                commands.append(command)
                return source.read_bytes() if command.startswith('cat -- ') else b''
            with patch.object(transport, 'call', call):
                transport.stage_plaintext('server51', source, destination)
                first_prepare = commands[0]
                first = list(commands)
                commands.clear()
                transport.stage_plaintext('server51', source, destination)
            self.assertNotEqual(first_prepare, commands[0])
            blocks = []
            for command in first:
                self.assertLessEqual(len((resume.route.guard_shell('server51') + command).encode()), 12000)
                if "<<'READABLE_FILE_" in command:
                    body = command.split("<<'", 1)[1]
                    delimiter, body = body.split("'\n", 1)
                    block = body[:-(len(delimiter) + 1)]
                    self.assertLessEqual(len(block.encode()), 8000)
                    blocks.append(block)
            self.assertGreater(len(blocks), 1)
            self.assertEqual(''.join(blocks).encode(), source.read_bytes())
            self.assertEqual(resume.PROOF, resume.frozen.HERE / 'proofs' / resume.BINDINGS_SHA)


if __name__ == '__main__':
    unittest.main()

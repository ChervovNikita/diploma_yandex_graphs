"""One local cosmetic-launch fixture; no native fit or remote operation."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from launch_named import PROCESS_TITLE, run_native


class NamedLaunchTest(unittest.TestCase):
    def test_title_native_arguments_script_directory_and_exit(self):
        title = []
        library = types.SimpleNamespace(setproctitle=lambda value: title.append(value), getproctitle=lambda: title[-1])
        previous_argv, previous_path = list(sys.argv), list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                script, result = folder / 'native.py', folder / 'result.json'
                script.write_text('import json, sys\nfrom pathlib import Path\nimport setproctitle\n'
                                  'Path(sys.argv[1]).write_text(json.dumps({"argv": sys.argv, "script_directory": sys.path[0], '
                                  '"title": setproctitle.getproctitle(), "name": __name__}))\nraise SystemExit(7)\n')
                native = [str(script), str(result), '--original-flag', 'value with spaces']
                with patch.dict(sys.modules, {'setproctitle': library}):
                    with self.assertRaises(SystemExit) as raised:
                        run_native(native)
                self.assertEqual(raised.exception.code, 7)
                self.assertEqual(json.loads(result.read_text()), {'argv': native, 'script_directory': str(folder.resolve()),
                                                                 'title': PROCESS_TITLE, 'name': '__main__'})
        finally:
            sys.argv[:], sys.path[:] = previous_argv, previous_path


if __name__ == '__main__':
    unittest.main()

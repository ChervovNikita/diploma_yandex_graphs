"""Run the retained read-only script on the exact authorized one-GPU route."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=15',
       '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222',
       'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
result = subprocess.run(SSH + ['python3 -'], input=(HERE / 'owned_runtime_observer.py').read_text(),
                        text=True, capture_output=True, timeout=50)
if result.returncode:
    raise RuntimeError('Owned observation failed: ' + result.stderr)
value = json.loads(result.stdout)
(HERE / 'RUNTIME_OBSERVATION.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
print(json.dumps(value, indent=2, sort_keys=True))

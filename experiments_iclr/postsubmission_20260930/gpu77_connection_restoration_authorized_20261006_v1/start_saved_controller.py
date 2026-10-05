"""Reopen the existing paired MacLink forwarding route, with sensitive output omitted."""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path('/Users/alex/Documents/ChatGPT/anogena allocation')
HERE = Path(__file__).resolve().parent
state = Path('/Users/alex/.maclink-reverse')
if not all((state / name).is_file() for name in ('server.json', 'server.crt', 'server.key')):
    raise RuntimeError('Saved controller state missing; no new pairing is authorized here')
command = [str(ROOT / 'reverse_maclink/maclink-env/bin/python'), '-B',
           str(ROOT / 'reverse_maclink/maclink.py'), 'connect']
child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, start_new_session=True)
receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'controller_pid': child.pid,
           'transport': 'Existing seven-GPU allocation SSH account for MacLink forwarding only',
           'saved_pairing_state_reused': True, 'new_pairing_performed': False,
           'credential_values_printed_or_logged': False,
           'target_access_reauthorized': 'Explicit current user authorization on 2026-10-06',
           'seven_GPU_compute_query_or_filesystem_command': False}
(HERE / 'CONTROLLER_LAUNCH.json').write_text(json.dumps(receipt, indent=2) + '\n')
(HERE / 'CONTROLLER_LAUNCH.json').chmod(0o444)
with (HERE / 'controller_sanitized.log').open('a') as log:
    for raw_line in child.stdout:
        if re.search(r'pairing\s+code|password|passwd|token|secret', raw_line, re.IGNORECASE):
            line = '[sensitive connection output omitted]\n'
        else:
            line = raw_line
        log.write(line)
        log.flush()
        print(line, end='', flush=True)
code = child.wait()
print('Saved MacLink controller exited with code', code, flush=True)
sys.exit(code)

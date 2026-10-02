"""Start a separate MacLink controller via the authorized one-GPU relay.

Quote the known-hosts option for the project's space-containing absolute path.
No default/forbidden relay is used and the existing controller is unchanged.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOGIN = 'anogena-2.ai0001053-01174'
if Path.cwd().resolve() != ROOT:
    raise RuntimeError('Run from the allowed local project')
sys.path.insert(0, str(ROOT / 'reverse_maclink'))
import relay_transport
import maclink

original = relay_transport.RelayTunnel.command

def quoted_command(self):
    if self.settings['ssh_user'] != LOGIN or self.settings['client_ssh_user'] != LOGIN:
        raise RuntimeError('Prohibited relay identity')
    command = original(self)
    unquoted = 'UserKnownHostsFile=' + str(self.known_hosts)
    quoted = 'UserKnownHostsFile="' + str(self.known_hosts).replace('\\', '\\\\').replace('"', '\\"') + '"'
    return [quoted if item == unquoted else item for item in command]

relay_transport.RelayTunnel.command = quoted_command
sys.argv = [str(ROOT / 'reverse_maclink/maclink.py'), '--state-dir', str(ROOT / '.ml77'),
            'serve', '--bind', '127.0.0.1', '--port', '8444', '--relay',
            '--relay-config', str(HERE / 'relay-onegpu.json')]
maclink.main()

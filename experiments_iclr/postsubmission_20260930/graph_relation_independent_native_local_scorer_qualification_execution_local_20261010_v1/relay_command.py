"""Run one explicit project command via MacLink; SSH secret never leaves the linked Mac."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXPECT_PREFIX = r'''
set timeout 120
set handle [open /Users/shmelev/Desktop/server-192.168.18.51-password.txt r]
set password [string trim [read $handle]]
close $handle
if {$password eq "" || [string first "\n" $password] >= 0} {puts stderr "Invalid single-line SSH credential"; exit 2}
log_user 0
spawn /usr/bin/ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o PreferredAuthentications=password -o PubkeyAuthentication=no shmelev@192.168.18.77
'''


def tcl_quote(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('[', '\\[') + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--command-file', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id):
        raise RuntimeError('Simple single-use command identity required')
    source = args.command_file.resolve()
    if not source.is_relative_to(HERE):
        raise RuntimeError('Explicit reviewed project command file required')
    command = source.read_text()
    destination = HERE / 'commands' / args.id
    destination.mkdir(parents=True, exist_ok=False)
    # append the single remote command as one Tcl-quoted argv item, not shell text on the linked Mac
    expect = EXPECT_PREFIX.rstrip() + ' ' + tcl_quote(command) + '\n' + r'''
expect {
  -re {(?i)password:} {send -- "$password\r"}
  timeout {puts stderr "SSH authentication prompt timed out"; exit 3}
  eof {puts stderr "SSH exited before password authentication"; exit 4}
}
unset password
log_user 1
expect {
  -re {Permission denied, please try again} {puts stderr "SSH authentication rejected"; exit 5}
  timeout {puts stderr "Remote project command timed out"; exit 6}
  eof {set status [wait]; exit [lindex $status 3]}
}
'''
    argv = [str(WORKSPACE / 'reverse_maclink/maclink-env/bin/python'), '-B',
            str(WORKSPACE / 'reverse_maclink/maclink.py'), 'run', '--', '/usr/bin/expect', '-c', expect]
    result = subprocess.run(argv, cwd=WORKSPACE, capture_output=True, text=True, timeout=150)
    record = dict(UTC=datetime.now(timezone.utc).isoformat(), id=args.id,
                  target='shmelev@192.168.18.77', transport='user-authorized MacLink forwarding relay only',
                  command_sha256=hashlib.sha256(command.encode()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  credential_value_recorded=False, exit_code=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    (destination / 'COMMAND.txt').write_text(command)
    (destination / 'RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('exit_code', 'stdout', 'stderr')}, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

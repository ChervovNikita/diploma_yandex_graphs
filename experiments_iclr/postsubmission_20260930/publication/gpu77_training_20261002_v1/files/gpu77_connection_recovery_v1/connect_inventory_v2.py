"""Use the user-authorized MacLink relay and privately supplied Desktop SSH password."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

WORKSPACE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECT = r'''
set timeout 30
set handle [open /Users/shmelev/Desktop/server-192.168.18.51-password.txt r]
set password [string trim [read $handle]]
close $handle
if {$password eq "" || [string first "\n" $password] >= 0} {puts stderr "Password file must contain one nonempty line"; exit 2}
log_user 0
spawn /usr/bin/ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o PreferredAuthentications=password -o PubkeyAuthentication=no shmelev@192.168.18.77 {cd /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning && nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader && git rev-parse --show-toplevel && git branch --show-current && git rev-parse HEAD && git status --short}
expect {
  -re {(?i)password:} {send -- "$password\r"}
  timeout {puts stderr "SSH authentication prompt timed out"; exit 3}
  eof {puts stderr "SSH exited before password authentication"; exit 4}
}
unset password
log_user 1
expect {
  -re {Permission denied} {puts stderr "SSH authentication rejected"; exit 5}
  timeout {puts stderr "SSH inventory timed out"; exit 6}
  eof {set status [wait]; exit [lindex $status 3]}
}
'''


def main():
    path = HERE / 'GPU77_INVENTORY_ATTEMPT_v2.json'
    if path.exists():
        raise RuntimeError('Inventory identity already attempted')
    command = [str(WORKSPACE / 'reverse_maclink/maclink-env/bin/python'), '-B',
               str(WORKSPACE / 'reverse_maclink/maclink.py'), 'run', '--', '/usr/bin/expect', '-c', EXPECT]
    result = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True, timeout=80)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(),
                   schema='gpu77-MacLink-inventory-v2', transport='existing seven-GPU SSH account used only as MacLink forwarding relay',
                   user_relay_exception='2026-10-02: explicitly authorized forwarding commands to the other Mac',
                   target='shmelev@192.168.18.77', expected_project='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning',
                   credential_source='user-designated Desktop text file on the linked Mac; read only inside the remote authentication process',
                   credential_value_returned_or_recorded=False, seven_GPU_compute_or_filesystem_commands=False,
                   GPU_compute_or_project_writes=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    with path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: receipt[k] for k in ('exit_code', 'stdout', 'stderr')}, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

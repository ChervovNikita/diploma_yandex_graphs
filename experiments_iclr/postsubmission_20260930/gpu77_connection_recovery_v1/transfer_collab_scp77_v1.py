"""Reuse the SHA-bound collab archive via scp through the authorized linked Mac."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
EXPECT = r'''
set timeout 240
set handle [open /Users/shmelev/Desktop/server-192.168.18.51-password.txt r]
set password [string trim [read $handle]]
close $handle
if {$password eq "" || [string first "\n" $password] >= 0} {puts stderr "Invalid single-line credential"; exit 2}
log_user 0
spawn /usr/bin/scp -3 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -i /Users/shmelev/.maclink-reverse/anogena_identity scp://anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/buddy_official_archive_staging_v1/root_transfer_v2/collab.zip scp://shmelev@192.168.18.77:22/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930/buddy_official_archive_staging_v1/root_transfer_v2/collab.zip
expect {
 -re {(?i)password:} {send -- "$password\r"}
 timeout {puts stderr "SCP authentication timed out"; exit 3}
 eof {puts stderr "SCP exited before password authentication"; exit 4}
}
unset password
expect {
 -re {Permission denied, please try again} {puts stderr "SCP authentication rejected"; exit 5}
 timeout {puts stderr "SCP transfer timed out"; exit 6}
 eof {set status [wait]; puts "SCP exit: [lindex $status 3]"; exit [lindex $status 3]}
}
'''


def main():
    out = HERE / 'commands/archive77_scp_transfer_v1'
    out.mkdir(parents=True, exist_ok=False)
    result = subprocess.run([str(WORKSPACE / 'reverse_maclink/maclink-env/bin/python'), '-B',
                             str(WORKSPACE / 'reverse_maclink/maclink.py'), 'run', '--',
                             '/usr/bin/expect', '-c', EXPECT],
                            cwd=WORKSPACE, capture_output=True, text=True, timeout=290)
    record = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode,
                  source_route='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222',
                  target_route='shmelev@192.168.18.77:22', transport='scp -3 through linked Mac',
                  archive_expected_sha256='c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc',
                  archive_expected_bytes=121625147, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  credential_value_recorded=False, linked_Mac_archive_copy_created=False,
                  seven_GPU_allocation_role='MacLink forwarding relay only; archive source is the authorized one-GPU repository',
                  stdout=result.stdout, stderr=result.stderr, target_byte_verification_pending=True)
    (out / 'RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, sort_keys=True))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

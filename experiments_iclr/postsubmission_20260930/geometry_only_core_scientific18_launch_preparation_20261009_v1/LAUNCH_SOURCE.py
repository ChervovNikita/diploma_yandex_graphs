"""Root-only normal-host detached launch; prepare locally, execute after publication."""
import argparse
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published-commit',required=True)
    parser.add_argument('--release-sha256',required=True)
    args=parser.parse_args()
    A=Path(__file__).resolve().parent;P=A.parent;R=P.parents[1]
    assert A.name=='geometry_only_core_scientific18_activation_root_20261009_v1'
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (A/'LAUNCH.json').exists() and not (A/'OWNER.log').exists()
    program=A/'SUPERVISOR.py';release=A/'RELEASE.json'
    assert hashlib.sha256(program.read_bytes()).hexdigest()=='4cf5451da80385f1d55314f7cc3a39ca5d67d71ad963e66b70afe06c6207dbbd'
    assert hashlib.sha256(release.read_bytes()).hexdigest()==args.release_sha256
    cfg=json.loads(release.read_text());assert cfg['enabled'] is True
    assert not Path(cfg['output_directory']).exists()
    head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
    assert head==args.published_commit
    subprocess.run(['git','-C',str(R),'merge-base','--is-ancestor','e80b02c44a926bb29f348af62f4f47f35215bb4e',head],check=True)
    with (A/'OWNER.log').open('xb') as log:
        child=subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(program)],cwd=R,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    raw=Path('/proc',str(child.pid),'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    identity=dict(pid=child.pid,start_ticks=int(fields[19]),group=int(fields[2]),session=int(fields[3]),state=fields[0],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    assert identity['group']==identity['session']==child.pid
    record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=identity,source_commit=head,
        scientific_source_anchor=cfg['execution_source_commit'],supervisor_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),
        release_sha256=args.release_sha256,original_logical_records=18,required_family_records_before_comparison=21,
        normal_host=True,automatic_retry=False,scientific_results_opened=False,comparative_outcomes_opened=False,TEST_truth_accessed=False)
    with (A/'LAUNCH.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps(record))


if __name__=='__main__':main()

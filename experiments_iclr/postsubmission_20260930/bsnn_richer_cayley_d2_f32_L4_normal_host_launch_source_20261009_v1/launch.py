"""Inactive once-only launcher using the existing BSNN detached normal-host owner."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--release',type=Path)
    parser.add_argument('--release-sha256')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(inactive=True,host_or_provider_access=False)))
        return
    assert args.release and args.release_sha256
    spec = importlib.util.spec_from_file_location('_richer_bsnn_original_owner',HERE/'SUPERVISOR.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    cfg,A,_,_,_ = owner.gate(args.release,args.release_sha256,True)
    assert not (A/'LAUNCH.json').exists() and not (A/'OWNER.log').exists()
    program = HERE/'SUPERVISOR.py'
    with (A/'OWNER.log').open('xb') as log:
        child = subprocess.Popen(['/usr/bin/python3','-I','-S','-B',str(program),'--execute',
            '--release',str(args.release.resolve()),'--release-sha256',args.release_sha256],
            cwd=owner.R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    saved = owner.identity(child.pid)
    assert saved and saved['group']==saved['session']==child.pid
    record = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=saved,
        source_commit=cfg['execution_source_commit'],supervisor_sha256=owner.sha(program),
        release_sha256=args.release_sha256,worker_release=cfg['worker_release'],mode=cfg['mode'],
        normal_host=True,automatic_retry=False,scientific_results_opened=False,TEST_truth_accessed=False)
    with (A/'LAUNCH.json').open('x') as f:
        json.dump(record,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(record))


if __name__=='__main__': main()

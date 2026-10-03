"""Own one source-bound qualification child; preserve every terminal and log."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parent
    phase=root.parent
    repo=phase.parents[1]
    assert str(repo)=='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
    assert Path.cwd().resolve()==repo
    run=root/'transport/root_focused_qualification_run01'
    packet=phase/'graph_mixed_block_training_preparation_20261003_v1'
    release=json.loads((root/'QUALIFICATION_RELEASE.json').read_text())
    assert release['qualification_authorized'] is True and release['execution_authorized'] is False
    resource.setrlimit(resource.RLIMIT_AS,(release['address_space_limit_bytes'],release['address_space_limit_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(3600,3660))
    command=[sys.executable,'-B',str(packet/'qualify_policy.py'),'--freeze',str(root/'FROZEN_STUDY.json'),'--admission',str(root/'QUALIFICATION_RELEASE.json'),'--run-name','root_focused_qualification_run01']
    start=datetime.now(timezone.utc).isoformat()
    timed_out=False
    with (run/'child.stdout.log').open('x') as out,(run/'child.stderr.log').open('x') as err:
        child=subprocess.Popen(command,cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        with (run/'CHILD.json').open('x') as stream:json.dump(dict(PID=child.pid,command=command),stream,indent=2);stream.write('\n')
        try:
            code=child.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    receipt=dict(start_UTC=start,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=code,timed_out=timed_out,qualification_only=True,scientific_training_started=False,unrelated_processes_touched=False)
    with (run/'SUPERVISOR_TERMINAL.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))
    return code


if __name__=='__main__':
    raise SystemExit(main())

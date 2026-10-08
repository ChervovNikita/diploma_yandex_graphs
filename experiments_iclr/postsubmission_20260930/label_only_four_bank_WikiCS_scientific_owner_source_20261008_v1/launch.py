"""Disabled once-only detached scientific parent launcher; no remote calls."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from owned import identity,read,require,sha,write

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True);parser.add_argument('--authorized',action='store_true');args=parser.parse_args()
    began=time.monotonic();cfg=read(args.release);pins=read(HERE/'SOURCE_BINDINGS.json');runtime=pins['runtime']
    require(args.authorized and cfg['enabled'] is True and cfg['scientific_execution_authorized'] is True
        and cfg['source_review_approved'] is True and sha(args.release)==args.release_sha256,'Disabled until exact enabled root release')
    activation=Path(runtime['phase'])/cfg['activation_relative'];require(activation.is_dir(),'Separate root activation required')
    receipt=activation/'LAUNCH.json';write(receipt,dict(reserved=True,automatic_retry=False,scores_read=False),True)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');environment.pop('PYTHONHOME',None)
    argv=['/usr/bin/python3','-I','-S','-B',str(HERE/'owned.py'),'--release',str(args.release.resolve()),
        '--release-sha256',args.release_sha256,'--authorized','--launch-started',repr(began)]
    with (activation/'OWNER.log').open('xb') as log:
        require(time.monotonic()-began<30,'Finite launch reserve')
        parent=subprocess.Popen(argv,cwd=runtime['repository'],env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    saved=identity(parent.pid);write(receipt,dict(reserved=True,parent=saved,argv=argv,release_sha256=args.release_sha256,
        parent_poll=parent.poll(),detached=True,automatic_retry=False,scores_read=False))
    require(saved and saved['group']==saved['session']==parent.pid,'Exact detached scientific parent')
    print(str(receipt))


if __name__=='__main__':main()

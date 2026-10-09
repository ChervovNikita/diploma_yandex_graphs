"""Disabled once-only detached normal-host launch for the fixed complete36 queue."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from common import HERE, release_config, require, write
from owned import identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--authorized',action='store_true')
    args=parser.parse_args()
    began=time.monotonic()
    cfg,pins,phase=release_config(args.release,args.release_sha256,args.authorized)
    activation=phase/cfg['activation_relative']
    require(activation.is_dir(),'Separate root activation directory required')
    write(activation/'LAUNCH.json',dict(reserved=True,parent=None,scores_read=False,automatic_retry=False),True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONHOME',None)
    argv=['/usr/bin/python3','-I','-S','-B',str(HERE/'owned.py'),'--release',str(args.release.resolve()),
          '--release-sha256',args.release_sha256,'--authorized','--launch-started',repr(began)]
    # -I omits the script directory, so bootstrap only this exact source directory.
    bootstrap='import runpy,sys;sys.path.insert(0,'+repr(str(HERE))+');runpy.run_path('+repr(str(HERE/'owned.py'))+',run_name="__main__")'
    argv=['/usr/bin/python3','-I','-S','-B','-c',bootstrap,*argv[5:]]
    with (activation/'OWNER.log').open('xb') as log:
        parent=subprocess.Popen(argv,cwd=pins['runtime']['repository'],env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    saved=identity(parent.pid)
    require(saved and saved['group']==saved['session']==parent.pid,'Exact detached owner PID/birth custody')
    write(activation/'LAUNCH.json',dict(reserved=True,parent=saved,argv=argv,release_sha256=args.release_sha256,
        detached=True,parent_poll=parent.poll(),scores_read=False,automatic_retry=False))


if __name__=='__main__': main()

"""Read Git and named credential locations on the authorized allocation only."""
import json
import os
from pathlib import Path
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def run(args):
    result = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True)
    return result.stdout

def main():
    if Path.cwd().resolve() != REPO or os.environ.get('GNNM_SSH_DESTINATION') != LOGIN:
        raise ValueError('Wrong repository or calling route')
    gpu = run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    if gpu != [UUID]:
        raise ValueError('Wrong allocation')
    if Path(run(['git', 'rev-parse', '--show-toplevel']).strip()).resolve() != REPO:
        raise ValueError('Wrong Git root')
    staged = run(['git', 'diff', '--cached', '--name-only', '-z']).split('\0')
    untracked = run(['git', 'ls-files', '--others', '--exclude-standard', '-z']).split('\0')
    names = [p for p in untracked if p]
    credentials = sorted(p for p in names if Path(p).name == 'the_github')
    for relative in run(['git', 'ls-files', '-z']).split('\0'):
        if relative and Path(relative).name == 'the_github':
            credentials.append(relative)
    named_home_key = Path('/home/jovyan/the_github')
    print(json.dumps(dict(
        schema='gnnm-scoped-git-inspection-v2',
        ssh_destination=LOGIN, gpu_uuid=UUID,
        branch=run(['git', 'branch', '--show-current']).strip(),
        head=run(['git', 'rev-parse', 'HEAD']).strip(),
        staged_paths=[p for p in staged if p],
        tracked_status=run(['git', 'status', '--short', '--untracked-files=no']),
        untracked_count=len(names),
        phase_untracked_count=sum(p.startswith('experiments_iclr/postsubmission_20260930/') for p in names),
        credential_filenames_only=credentials,
        named_home_key_exists=named_home_key.is_file(),
        origin=run(['git', 'remote', 'get-url', 'origin']).strip(),
        branch_refs=run(['git', 'for-each-ref', '--format=%(refname:short) %(objectname)', 'refs/heads', 'refs/remotes']),
        recent_log=run(['git', 'log', '-5', '--format=%h %s'])
    ), indent=2))

if __name__ == '__main__':
    main()

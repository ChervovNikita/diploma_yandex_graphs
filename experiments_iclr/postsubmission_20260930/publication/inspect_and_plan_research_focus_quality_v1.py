"""Inspect exact reviewed targets and write a prospective publication plan."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
SNAPSHOT = PHASE / 'publication/research_focus_quality_20261002_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
REMOTE_CODE = '''import hashlib,json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path.cwd().resolve()!=repo or os.environ.get('GNNM_SSH_DESTINATION')!=sys.argv[2]:
 raise ValueError('Wrong repository or route')
def run(args):
 return subprocess.run(args,cwd=repo,capture_output=True,text=True,check=True).stdout
if run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).splitlines()!=[sys.argv[3]]:
 raise ValueError('Wrong allocation')
inventory=json.load(sys.stdin)
branch=run(['git','branch','--show-current']).strip()
head=run(['git','rev-parse','HEAD']).strip()
staged=run(['git','diff','--cached','--name-only'])
if branch!=inventory['branch'] or head!=inventory['expected_head'] or staged:
 raise ValueError('Git state differs from publication review')
rows=[]
for row in inventory['files']:
 rel=pathlib.PurePosixPath(row['target'])
 if rel.is_absolute() or '..' in rel.parts: raise ValueError('Invalid target')
 if str(rel)!='README.md' and not str(rel).startswith('experiments_iclr/postsubmission_20260930/'):
  raise ValueError('Outside reviewed phase')
 p=repo.joinpath(*rel.parts)
 if not p.resolve().is_relative_to(repo): raise ValueError('Escaping target')
 cursor=repo
 for part in rel.parts:
  cursor=cursor/part
  if cursor.is_symlink(): raise ValueError('Symlink target')
 if p.exists() and not p.is_file(): raise ValueError('Non-file target')
 rows.append(dict(target=row['target'],expected_old_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None))
print(json.dumps(dict(branch=branch,head=head,staged_paths=[],files=rows,ssh_destination=sys.argv[2],gpu_uuid=sys.argv[3])))
'''

def main():
    receipt = SNAPSHOT / 'REMOTE_INSPECTION_v2.json'
    plan_path = SNAPSHOT / 'PLAN_v2.json'
    if receipt.exists() or plan_path.exists():
        raise ValueError('Publication inspection identity already used')
    raw = (SNAPSHOT / 'INVENTORY.json').read_bytes()
    inventory = json.loads(raw)
    for row in inventory['files']:
        path = PHASE / row['source_snapshot']
        if not path.resolve().is_relative_to(SNAPSHOT):
            raise ValueError('Not an immutable inventory snapshot')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('Reviewed snapshot changed')
    command = ('cd ' + shlex.quote(REPO) + ' && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh'
               + ' && GNNM_SSH_DESTINATION=' + shlex.quote(LOGIN) + ' .venv/bin/python -c ' + shlex.quote(REMOTE_CODE) + ' ' + shlex.join([REPO, LOGIN, UUID]))
    start = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(SSH + [command], input=raw, capture_output=True, timeout=60)
    value = dict(start_UTC=start, terminal_UTC=datetime.now(timezone.utc).isoformat(),
                 exit_code=result.returncode, inventory_sha256=hashlib.sha256(raw).hexdigest(),
                 stdout=result.stdout.decode(errors='replace'), stderr=result.stderr.decode(errors='replace'))
    with receipt.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
    if result.returncode:
        raise RuntimeError('Authorized target inspection failed; receipt retained')
    remote = json.loads(value['stdout'])
    if [x['target'] for x in remote['files']] != [x['target'] for x in inventory['files']]:
        raise ValueError('Remote inventory order changed')
    files = [dict(row, expected_old_sha256=old['expected_old_sha256'])
             for row, old in zip(inventory['files'], remote['files'])]
    plan = dict(branch=inventory['branch'], expected_head=inventory['expected_head'], files=files,
                message='Prepare complete-cohort evaluation and prioritize predictive quality',
                inventory_sha256=hashlib.sha256(raw).hexdigest(),
                inspection_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                scientific_utility_claim=False, binary_artifacts_included=False)
    with plan_path.open('x') as stream:
        json.dump(plan, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(plan=str(plan_path), files=len(files), head=remote['head'],
                          missing=sum(x['expected_old_sha256'] is None for x in files),
                          changed=sum(x['expected_old_sha256'] != x['sha256'] for x in files))))

if __name__ == '__main__':
    main()

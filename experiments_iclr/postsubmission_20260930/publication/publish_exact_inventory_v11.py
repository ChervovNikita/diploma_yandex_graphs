"""Publish an explicit small inventory, optionally updating the main README.

The inspection and commit are separate operations. Only one obsolete uppercase
ledger path may be removed, after confirming its tracked Git custody. Credentials
and large research artifacts are never accepted as inventory paths.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import zlib

PHASE = Path(__file__).resolve().parents[1]
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BRANCH = 'codex/postsubmission-research-20260930'
PREFIX = 'experiments_iclr/postsubmission_20260930/'
OBSOLETE = PREFIX + 'RESEARCH_LEDGER.json'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]

REMOTE = r'''
import base64,hashlib,json,pathlib,subprocess,sys,zlib
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];operation=sys.argv[3]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
def git(*args):
 r=subprocess.run(['git',*args],cwd=repo,capture_output=True,check=True)
 return r.stdout
def digest(data):return hashlib.sha256(data).hexdigest()
def confined(relative):
 rel=pathlib.PurePosixPath(relative)
 assert not rel.is_absolute() and '..' not in rel.parts
 assert relative=='README.md' or relative.startswith('experiments_iclr/postsubmission_20260930/')
 path=repo.joinpath(*rel.parts)
 assert path.resolve().is_relative_to(repo)
 cur=repo
 for part in rel.parts:
  cur/=part
  assert not cur.is_symlink()
 return path
assert repo.resolve()==repo and pathlib.Path(git('rev-parse','--show-toplevel').decode().strip())==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[uuid]
assert git('branch','--show-current').decode().strip()==payload['branch']
assert git('rev-parse','HEAD').decode().strip()==payload['expected_head']
assert not git('diff','--cached','--name-only')
targets=[r['target'] for r in payload['files']]+[r['target'] for r in payload['remove']]
assert len(targets)==len(set(targets))
assert not payload['remove'] or [r['target'] for r in payload['remove']]==['experiments_iclr/postsubmission_20260930/RESEARCH_LEDGER.json']
observed=[]
for row in payload['files']:
 path=confined(row['target']);assert not path.exists() or path.is_file()
 observed.append(dict(target=row['target'],expected_old_sha256=digest(path.read_bytes()) if path.exists() else None))
deleted=[]
for row in payload['remove']:
 path=confined(row['target']);assert path.is_file()
 tracked=git('show','HEAD:'+row['target'])
 assert path.read_bytes()==tracked,'Obsolete ledger has uncommitted changes'
 deleted.append(dict(target=row['target'],expected_old_sha256=digest(tracked),history_commit=payload['expected_head']))
if operation=='inspect':
 print(json.dumps(dict(files=observed,remove=deleted,head=payload['expected_head'],route_uuid=uuid)))
 sys.exit(0)
assert operation=='commit'
commit_receipt=confined(payload['commit_receipt_target'])
assert not commit_receipt.exists()
assert observed==[{k:r[k] for k in ('target','expected_old_sha256')} for r in payload['files']]
assert deleted==payload['remove']
prepared=[]
for row in payload['files']:
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and digest(data)==row['sha256']
 prepared.append((confined(row['target']),data))
ledger=next((d for p,d in prepared if p.name=='research_ledger.json'),None)
if payload['remove']:
 assert ledger is not None
 old=json.loads(confined(payload['remove'][0]['target']).read_bytes())
 new=json.loads(ledger)
 assert set(old)<=set(new),'Do not lose top-level research history in ledger correction'
 for rel in ('research_ledger.json','RESEARCH_LEDGER.json'):
  p=repo/'experiments_iclr/postsubmission_20260930'/rel
  if p.exists():assert set(json.loads(p.read_bytes()))<=set(new)
for path,data in prepared:
 if not path.exists() or path.read_bytes()!=data:
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
if payload['remove']:git('rm','--',payload['remove'][0]['target'])
git('add','-f','--',*[r['target'] for r in payload['files']])
staged=[v for v in git('diff','--cached','--name-only','-z').decode().split('\0') if v]
assert staged and set(staged)<=set(targets)
for row in payload['files']:
 assert digest(git('show',':'+row['target']))==row['sha256']
for row in payload['remove']:
 assert row['target'] not in git('ls-files').decode().splitlines()
git('commit','-m',payload['message'])
head=git('rev-parse','HEAD').decode().strip()
changed=git('diff-tree','--no-commit-id','--name-only','-r',head).decode().splitlines()
assert set(changed)==set(staged) and not git('diff','--cached','--name-only')
result=dict(commit=head,prior_head=payload['expected_head'],branch=payload['branch'],
            changed_paths=changed,removed_ledger_history=payload['remove'],
            route_uuid=uuid,push_verified=False)
commit_receipt.parent.mkdir(parents=True,exist_ok=True)
with commit_receipt.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
print(json.dumps(result))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Require project-relative path')
    path = PHASE.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(PHASE) or path.is_symlink():
        raise ValueError('Unconfined source')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('inspect', 'commit'))
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    inventory_path = local(args.inventory)
    receipt_path = local(args.receipt)
    assert not receipt_path.exists()
    payload = json.loads(inventory_path.read_text())
    assert payload['branch'] == BRANCH
    assert not payload['remove'] or [r['target'] for r in payload['remove']] == [OBSOLETE]
    for row in payload['files']:
        if row['target'] == 'README.md':
            assert PurePosixPath(row['source']).name == 'README_MAIN.md'
            assert local(row['source']).is_relative_to(PHASE / 'publication')
        else:
            assert row['target'] == PREFIX + row['source']
        source = local(row['source'])
        assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256']
        assert source.suffix in ('.py', '.json', '.jsonl', '.md', '.txt', '.html', '.diff', '.patch', '.log', '.raw', '.sha256', '.csv', '.xml', '.sh', '.yml') or source.name in ('SHA256SUMS', 'NOTE_SHA256SUMS', '.gitignore')
        assert source.stat().st_size < 2_000_000
        if args.operation == 'commit':
            row['data'] = base64.b64encode(source.read_bytes()).decode()
    if args.operation == 'commit':
        payload['commit_receipt_target'] = PREFIX + str(receipt_path.parent.relative_to(PHASE)) + '/COMMIT_RECEIPT.json'
        assert not receipt_path.with_name('COMMIT_RECEIPT.json').exists()
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE, REPO, UUID, args.operation])
    result = subprocess.run([*SSH, command], input=base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode(),
                            capture_output=True, text=True, timeout=120)
    value = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode,
                 ssh_destination=LOGIN, operation=args.operation, transport='zlib-compressed-explicit-inventory',
                 inventory_sha256=sha(inventory_path), executed_source_sha256=sha(Path(__file__)),
                 stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        value['result'] = json.loads(result.stdout)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    if result.returncode:
        print(json.dumps(dict(exit_code=result.returncode, stderr=result.stderr)))
    elif args.operation == 'inspect':
        plan = json.loads(inventory_path.read_text())
        assert len(plan['files']) == len(value['result']['files'])
        for row, observed in zip(plan['files'], value['result']['files']):
            assert row['target'] == observed['target']
            row['expected_old_sha256'] = observed['expected_old_sha256']
        plan['remove'] = value['result']['remove']
        plan_path = inventory_path.with_name(inventory_path.stem + '_PLAN.json')
        with plan_path.open('x') as handle:
            json.dump(plan, handle, indent=2)
            handle.write('\n')
        print(json.dumps(dict(files=len(plan['files']),remove=len(plan['remove']),
                             plan=str(plan_path),head=plan['expected_head'])))
    else:
        with receipt_path.with_name('COMMIT_RECEIPT.json').open('x') as handle:
            json.dump(value['result'], handle, indent=2)
            handle.write('\n')
        print(json.dumps(dict(commit=value['result']['commit'],
                             changed_files=len(value['result']['changed_paths']),push_verified=False)))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

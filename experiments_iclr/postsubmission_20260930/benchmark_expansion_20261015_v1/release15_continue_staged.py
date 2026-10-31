import subprocess,json,hashlib,datetime
from pathlib import Path
root=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');e=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1')
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
v=json.loads((e/'PUBLICATION_SELECTION_15.json').read_text());release=e/v['release_name'];out=e/v['commit_receipt']
assert git('rev-parse','HEAD')==v['expected_parent'] and not out.exists()
assert json.loads(release.read_text())['files']==v['files']
files=[e/r['path'] for r in v['files']]+[root/'README.md',release]
allowed={str(p.relative_to(root)) for p in files};staged=set(git('diff','--cached','--name-only').splitlines());assert staged<=allowed and staged
for row in v['files']:
 p=e/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
assert not git('diff','--name-only','--',*sorted(allowed))
print('Exact staged selection and working files confirmed',flush=True)
exclusions=v['whitespace_data_exclusions'];paths=['.']+[':(exclude)'+str((e/x).relative_to(root)) for x in exclusions]
subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*paths],cwd=root,check=True)
print('Whitespace check complete; committing',flush=True)
subprocess.run(['git','commit','-m',v['message']],cwd=root,check=True)
r={'commit':git('rev-parse','HEAD'),'parent':v['expected_parent'],'branch':git('branch','--show-current'),'source_release_sha256':hashlib.sha256(release.read_bytes()).hexdigest(),'source_release':v['release_name'],'committed_files':len(files),'full_campaign_GO':False,'recovery':'continued already-staged exact selection after original SSH observation timeout; no active publisher remained','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

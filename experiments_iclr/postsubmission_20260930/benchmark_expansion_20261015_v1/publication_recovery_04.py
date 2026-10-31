import hashlib,json,subprocess
from pathlib import Path
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');e=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1');s={'expected_parent': 'eb6ddd1c58789a88261d81976c65323ce425b6a2', 'release_name': 'SOURCE_ADDITION_04.json', 'commit_receipt': 'COMMIT_RECEIPT_04.json', 'message': 'Expand shared GNNM study and qualify task objectives, actual replay and full-epoch recovery'}
s['files']=json.loads((e/s['release_name']).read_text())['files']
prefix=str(e.relative_to(r))+'/'
expected={prefix+x['path'] for x in s['files']}|{'README.md',prefix+s['release_name']}
def git(*a,timeout=180):return subprocess.check_output(['git',*a],cwd=r,text=True,timeout=timeout).strip()
assert git('rev-parse','HEAD')==s['expected_parent']
assert git('branch','--show-current')=='codex/postsubmission-research-20260930'
assert set(git('diff','--cached','--name-only').splitlines())==expected
raw=subprocess.check_output(['git','cat-file','--batch'],cwd=r,input=''.join(':'+x+'\n' for x in sorted(expected)).encode(),timeout=180)
pos=0;lookup={prefix+x['path']:x for x in s['files']}
for name in sorted(expected):
 end=raw.index(b'\n',pos);header=raw[pos:end].split();size=int(header[-1]);data=raw[end+1:end+1+size];pos=end+size+2
 assert hashlib.sha256(data).digest()==hashlib.sha256((r/name).read_bytes()).digest(),name
 if name in lookup:assert hashlib.sha256(data).hexdigest()==lookup[name]['sha256'],name
assert pos==len(raw)
exception=prefix+'runtime_v3/DELTA.diff.txt'
subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',':(exclude)'+exception],cwd=r,check=True,timeout=180)
print(json.dumps({'validated_files':len(expected),'whitespace_exception':exception,'exception_reason':'stored unified diff blank context lines; frozen bytes preserved'}),flush=True)
subprocess.run(['git','commit','-m',s['message']],cwd=r,check=True,timeout=240)
value={'commit':git('rev-parse','HEAD'),'parent':s['expected_parent'],'branch':git('branch','--show-current'),'source_release_sha256':hashlib.sha256((e/s['release_name']).read_bytes()).hexdigest(),'source_release':s['release_name'],'committed_files':len(expected),'full_campaign_GO':False}
assert git('rev-parse','HEAD^')==s['expected_parent']
(e/s['commit_receipt']).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value),flush=True)

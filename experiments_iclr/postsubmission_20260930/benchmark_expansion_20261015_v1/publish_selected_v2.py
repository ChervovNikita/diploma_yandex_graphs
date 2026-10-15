"""Publish one explicitly reviewed source selection on the guarded allocation.

The local selection pins exact bytes. Results, datasets and model states cannot
enter this publisher. It makes one commit and never launches scientific work.
"""
import argparse
import hashlib
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    selection = Path(args.selection).resolve()
    if not selection.is_relative_to(here):
        raise ValueError("Source selection must be inside this project expansion")
    value = json.loads(selection.read_text())
    if value["full_campaign_GO"] is not False or not value["reviewed"]:
        raise ValueError("Reviewed source-only publication required")
    for row in value["files"]:
        path = (here/row["path"]).resolve()
        if not path.is_relative_to(here) or "results" in path.relative_to(here).parts \
                or path.suffix not in (".py", ".md", ".json", ".txt", ".csv"):
            raise ValueError("Only explicit bounded source/notes may be published")
        if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Reviewed local bytes changed: "+row["path"])
    code = '''import json,hashlib,subprocess,ast
from pathlib import Path
root=Path(ROOT);e=Path(EXP);value=VALUE
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==value['expected_parent']
assert git('branch','--show-current')=='codex/postsubmission-research-20260930'
assert not git('diff','--cached','--name-only')
files=[]
for row in value['files']:
 p=(e/row['path']).resolve()
 assert p.is_relative_to(e) and 'results' not in p.relative_to(e).parts
 raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],row['path']
 assert len(raw)<2000000
 assert (b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----') not in raw and (b'-----BEGIN '+b'RSA PRIVATE KEY-----') not in raw
 if p.suffix=='.py':ast.parse(raw.decode(),filename=str(p))
 files.append(p)
allowed={str(p.relative_to(root)) for p in files}|{'README.md'}
changed=set(git('diff','--name-only').splitlines())
assert changed<=allowed,changed-allowed
release=e/value['release_name'];commit_receipt=e/value['commit_receipt']
assert not release.exists() and not commit_receipt.exists()
manifest={'schema':'reviewed-gnnm-source-addition-v3','expected_parent':value['expected_parent'],
          'files':value['files'],'root_README_sha256':hashlib.sha256((root/'README.md').read_bytes()).hexdigest(),
          'raw_datasets_checkpoints_logits_included':False,'full_campaign_GO':False,
          'scientific_result_scope':value.get('scientific_result_scope','source preparation only'),
          'confirmed_prediction_quality_advantage_claim':False}
release.write_text(json.dumps(manifest,indent=2)+'\\n')
files += [root/'README.md',release]
subprocess.run(['git','add','--',*[str(p.relative_to(root)) for p in files]],cwd=root,check=True)
# Unified diffs are byte-pinned data: an unchanged blank context line is a
# literal single space. Check every source file while excluding only explicitly
# declared, structurally verified diff artifacts from whitespace lint.
exclusions=value.get('whitespace_data_exclusions',[])
assert set(exclusions)<=set(row['path'] for row in value['files'])
for relative in exclusions:
 p=e/relative
 assert p.name.endswith('.diff.txt') and p.read_text().startswith('--- ')
 assert '\\n+++ ' in p.read_text() and '\\n@@ ' in p.read_text()
pathspec=['.']+[':(exclude)'+str((e/relative).relative_to(root)) for relative in exclusions]
subprocess.run(['git','-c','core.whitespace=trailing-space,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--',*pathspec],cwd=root,check=True)
subprocess.run(['git','commit','-m',value['message']],cwd=root,check=True)
receipt={'commit':git('rev-parse','HEAD'),'parent':value['expected_parent'],
         'branch':git('branch','--show-current'),'source_release_sha256':hashlib.sha256(release.read_bytes()).hexdigest(),
         'source_release':value['release_name'],'committed_files':len(files),'full_campaign_GO':False}
commit_receipt.write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    code = code.replace("ROOT", repr(REMOTE_ROOT)).replace("EXP", repr(REMOTE_EXPANSION)).replace("VALUE", repr(value))
    result = call(code, timeout=120)
    receipt = json.loads(result.splitlines()[-1])
    path = here/value["commit_receipt"]
    if path.exists():
        raise ValueError("Preserve prior local commit receipt")
    path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

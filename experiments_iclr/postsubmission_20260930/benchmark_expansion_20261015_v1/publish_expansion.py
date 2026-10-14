"""Commit the reviewed expansion source/notes and current README on allocation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from remote_control import call, REMOTE_ROOT, REMOTE_PHASE, REMOTE_EXPANSION

CODE = '''import json,hashlib
from pathlib import Path
root=Path(ROOT_LITERAL);phase=Path(PHASE_LITERAL);exp=Path(EXP_LITERAL)
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
expected='eefa96f027c0a9ca9f1044bdc43f71fd8a1da483'
assert git('rev-parse','HEAD')==expected
assert git('branch','--show-current')=='codex/postsubmission-research-20260930'
assert not git('diff','--cached','--name-only')
assert not git('diff','--name-only','--',':!experiments_iclr/postsubmission_20260930/AGENTS.md')
readme=root/'README.md';original=readme.read_text();block=(exp/'ROOT_README_BLOCK.md').read_text()
assert block.splitlines()[0] not in original
readme.write_text(original.split('\\n',1)[0]+'\\n\\n'+block+'\\n'+original.split('\\n',1)[1].lstrip())
files=[]
for path in sorted(exp.rglob('*')):
 relative=path.relative_to(exp)
 if not path.is_file() or {'results','__pycache__','.cache'} & set(relative.parts):continue
 if path.suffix not in {'.py','.md','.json'}:continue
 if path.name in {'COMMIT_RECEIPT.json','PUSH_RECEIPT.json'}:continue
 assert path.stat().st_size<2000000
 files.append(path)
manifest={'schema':'gnnm-expansion-source-release-v1','expected_parent':expected,
          'files':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
          'raw_datasets_checkpoints_logits_included':False,'full_campaign_GO':False}
release=exp/'SOURCE_RELEASE.json'
with release.open('x') as stream:json.dump(manifest,stream,indent=2)
ledger_path=phase/'research_ledger.json';ledger=json.loads(ledger_path.read_text())
key='task_aware_benchmark_expansion_and_bounded_molecular_pilot_20261015_v1'
assert key not in ledger
ledger[key]={'path':'benchmark_expansion_20261015_v1','separate_source_reference_fits_retained':270,
 'predecessor_proposed_fit_count':2480,'expanded_campaign_count_final':False,
 'pilot':'pilot_20261015_molbace_v1/SPEC.json','pilot_fits':24,'pilot_trials':3,
 'pilot_spec_sha256':'8a1a004369e5c15de2254323a9f4e17d7b5c4e648566e15afaef519fc3bfbb1c',
 'new_prediction_quality_claim':False,'TEST_scoring':False,'full_campaign_GO':False,
 'source_runtime_scope':'complete native node/molecular epochs; link native integration pending',
 'reading_conclusions':'benchmark_expansion_20261015_v1/literature/SCOPED_CONCLUSIONS.md'}
ledger_path.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\\n')
files+=[release,phase/'AGENTS.md',readme,ledger_path]
subprocess.run(['git','add','--',*[str(p.relative_to(root)) for p in files]],cwd=root,check=True)
stat=git('diff','--cached','--stat')
print(stat,flush=True)
subprocess.run(['git','diff','--cached','--check'],cwd=root,check=True)
subprocess.run(['git','commit','-m','Expand task-aware shared-backbone research and launch bounded molecular pilot'],cwd=root,check=True)
value={'commit':git('rev-parse','HEAD'),'parent':expected,'branch':git('branch','--show-current'),
       'source_release_sha256':hashlib.sha256(release.read_bytes()).hexdigest(),'committed_files':len(files)}
with (exp/'COMMIT_RECEIPT.json').open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps({'COMMIT_RECEIPT':value}))
'''

if __name__=='__main__':
 code=CODE.replace('ROOT_LITERAL',repr(REMOTE_ROOT)).replace('PHASE_LITERAL',repr(REMOTE_PHASE)).replace('EXP_LITERAL',repr(REMOTE_EXPANSION))
 output=call(code,timeout=90)
 print(output)
 last=json.loads(output.splitlines()[-1])['COMMIT_RECEIPT']
 (Path(__file__).parent/'COMMIT_RECEIPT.json').write_text(json.dumps(last,indent=2)+'\n')

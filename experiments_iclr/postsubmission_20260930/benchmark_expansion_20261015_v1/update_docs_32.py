"""Update the explicitly selected current docs from their known release31 bytes."""
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

E = Path(__file__).resolve().parent
prior = json.loads((E/'DOC_UPDATE_RECEIPT_31.json').read_text())['docs']
names = ('README.md', 'SESSION_STATE.md', 'BENCHMARK_OVERVIEW.md',
         'research/CURRENT_DIRECTION_20261015.md')
texts = {name: (E/name).read_text() for name in names}
old_block = (E/'ROOT_README_BLOCK_31.md').read_text()
new_block = (E/'ROOT_README_BLOCK_32.md').read_text()
parent = '1ea29c160aa752309ad1e0ba37dbd5327987f617'
code = '''import json,hashlib,subprocess
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@);prior=@PRIOR@;texts=@TEXTS@
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==@PARENT@
assert not (E/'DOC_UPDATE_RECEIPT_32.json').exists()
for relative,pin in prior.items():
 p=R/relative;raw=p.read_bytes()
 assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],relative
root_text=(R/'README.md').read_text();assert root_text.count(@OLD_BLOCK@)==1
updates={R/'README.md':root_text.replace(@OLD_BLOCK@,@NEW_BLOCK@,1),**{E/k:v for k,v in texts.items()}}
for p,value in updates.items():
 temporary=p.with_name(p.name+'.update32');assert not temporary.exists()
 temporary.write_text(value);temporary.replace(p)
receipt={'docs':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in updates},
         'expected_parent':@PARENT@,'native_calls':0}
(E/'DOC_UPDATE_RECEIPT_32.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
for key,value in {'ROOT':REMOTE_ROOT,'EXP':REMOTE_EXPANSION,'PRIOR':prior,
                  'TEXTS':texts,'OLD_BLOCK':old_block,'NEW_BLOCK':new_block,
                  'PARENT':parent}.items():
    code=code.replace('@'+key+'@',repr(value))
receipt=json.loads(call(code))
with (E/'DOC_UPDATE_RECEIPT_32.json').open('x') as handle:
    json.dump(receipt,handle,indent=2);handle.write('\n')
print(json.dumps({'updated_documents':len(receipt['docs']),'native_calls':0}))

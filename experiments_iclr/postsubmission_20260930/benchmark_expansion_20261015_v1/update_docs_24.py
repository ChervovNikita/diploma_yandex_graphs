"""Update only four current research documents from their known prior bytes."""
import hashlib,json
from pathlib import Path
from remote_control import call,REMOTE_EXPANSION,REMOTE_ROOT
E=Path(__file__).resolve().parent
prior=json.loads((E/'DOC_UPDATE_RECEIPT_22.json').read_text())['docs']
texts={name:(E/name).read_text() for name in ['README.md','SESSION_STATE.md','BENCHMARK_OVERVIEW.md']}
old=(E/'ROOT_README_BLOCK_22.md').read_text();new=(E/'ROOT_README_BLOCK_24.md').read_text()
code='''import json,hashlib,subprocess
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@);prior=@PRIOR@;texts=@TEXTS@
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='afd61de7cd4ff4fe889d2ee1d7f1ee91232dcf4a'
assert not (E/'DOC_UPDATE_RECEIPT_24.json').exists()
for relative,pin in prior.items():
 p=R/relative;data=p.read_bytes();assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],relative
root_text=(R/'README.md').read_text();assert root_text.count(@OLD@)==1
updates={R/'README.md':root_text.replace(@OLD@,@NEW@,1),**{E/k:v for k,v in texts.items()}}
for p,txt in updates.items():
 tmp=p.with_name(p.name+'.update24');assert not tmp.exists();tmp.write_text(txt);tmp.replace(p)
receipt={'docs':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in updates},'expected_parent':'afd61de7cd4ff4fe889d2ee1d7f1ee91232dcf4a','native_calls':0}
(E/'DOC_UPDATE_RECEIPT_24.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
for k,v in {'ROOT':REMOTE_ROOT,'EXP':REMOTE_EXPANSION,'PRIOR':prior,'TEXTS':texts,'OLD':old,'NEW':new}.items():code=code.replace('@'+k+'@',repr(v))
r=json.loads(call(code));out=E/'DOC_UPDATE_RECEIPT_24.json'
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'updated_documents':len(r['docs']),'native_calls':0}))

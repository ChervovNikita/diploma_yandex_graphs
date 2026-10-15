"""Update only explicitly selected current documents from their known bytes."""
import json
from pathlib import Path
from remote_control import call, REMOTE_EXPANSION, REMOTE_ROOT

E = Path(__file__).resolve().parent
prior = json.loads((E/'DOC_UPDATE_RECEIPT_24_FINAL.json').read_text())['docs']
texts = {name: (E/name).read_text() for name in
         ('README.md', 'SESSION_STATE.md', 'BENCHMARK_OVERVIEW.md')}
old_block = (E/'ROOT_README_BLOCK_24.md').read_text()
new_block = (E/'ROOT_README_BLOCK_25.md').read_text()
direction_path = 'research/CURRENT_DIRECTION_20261015.md'
new_direction = (E/direction_path).read_text()
old_direction = new_direction.replace(
    'All four CPU lifecycle checks passed. TRAIN-only support is adequate under the native sampler. Proposed 21 fits are not launched.',
    'Source prepared for bounded qualification. Proposed 21 fits are not launched.')
assert old_direction != new_direction

code = '''import json,hashlib,subprocess
from pathlib import Path
R=Path(@ROOT@);E=Path(@EXP@);prior=@PRIOR@;texts=@TEXTS@
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='4a0347d69221c3472e6300dd57443756692d657f'
assert not (E/'DOC_UPDATE_RECEIPT_25.json').exists()
for relative,pin in prior.items():
 p=R/relative;raw=p.read_bytes()
 assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],relative
root_text=(R/'README.md').read_text();assert root_text.count(@OLD_BLOCK@)==1
direction=E/@DIRECTION_PATH@
assert direction.read_text()==@OLD_DIRECTION@
updates={R/'README.md':root_text.replace(@OLD_BLOCK@,@NEW_BLOCK@,1),
         **{E/k:v for k,v in texts.items()},direction:@NEW_DIRECTION@}
for p,value in updates.items():
 temporary=p.with_name(p.name+'.update25');assert not temporary.exists()
 temporary.write_text(value);temporary.replace(p)
receipt={'docs':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in updates},
         'expected_parent':'4a0347d69221c3472e6300dd57443756692d657f','native_calls':0}
(E/'DOC_UPDATE_RECEIPT_25.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
for key,value in {'ROOT':REMOTE_ROOT,'EXP':REMOTE_EXPANSION,'PRIOR':prior,
                  'TEXTS':texts,'OLD_BLOCK':old_block,'NEW_BLOCK':new_block,
                  'DIRECTION_PATH':direction_path,'OLD_DIRECTION':old_direction,
                  'NEW_DIRECTION':new_direction}.items():
    code=code.replace('@'+key+'@',repr(value))
receipt=json.loads(call(code))
with (E/'DOC_UPDATE_RECEIPT_25.json').open('x') as handle:
    json.dump(receipt,handle,indent=2);handle.write('\n')
print(json.dumps({'updated_documents':len(receipt['docs']),'native_calls':0}))

"""Stdlib-only source checks; no scientific/model/provider/array call."""
import ast
import json
from pathlib import Path
import sys
from source import HERE, sha, verify_sources
from reference_plan import roster, forecast, CONDITIONS


def main():
    compiled=[]
    for path in sorted(HERE.glob('*.py')):
        source=path.read_text();ast.parse(source);compile(source,str(path),'exec');compiled.append(path.name)
    verify_sources()
    rows=json.loads((HERE/'ROSTER.json').read_text())
    assert rows == roster() and len(rows)==9
    assert sum(len(r['body_roster']) for r in rows)==18
    assert len({(r['seed'],r['condition']) for r in rows})==9
    disabled=[]
    for path in sorted(HERE.glob('*TEMPLATE_DISABLED.json'))+sorted((HERE/'releases_disabled').glob('*.json')):
        value=json.loads(path.read_text());assert value['enabled'] is False;disabled.append(path.name)
    for row in rows:
        assert row['max_epochs']==2000 and row['patience']==250
        assert not row['TEST_access'] and not row['CORE'] and not row['masking'] and not row['HPO']
        for m,body in enumerate(row['body_roster']):
            assert body['initialization_seed']==row['seed']+1009*m
            assert body['factual_dropout_seeds']==[row['seed']+1009*m+300001+1009*v for v in range(4 if row['condition']=='single_mean4_dropout' else 1)]
    adapter=(HERE/'adapter.py').read_text();train=(HERE/'train.py').read_text()
    assert 'for view in range(4)' in adapter and '(ce/4).backward()' in adapter
    assert adapter.index('optimizer.step()') > adapter.index('(ce/4).backward()')
    assert 'correct > best_correct' in train and 'epoch-best_epoch >=250' in train
    assert 'for epoch in range(1,2001)' in train
    assert 'logits[0,roles[\'VALID\'][0]].max(1)[1]' in train
    assert 'fit_body(spec,body_spec' in train and 'assembled_I4(spec[\'seed\'],tensor,states)' in train
    assert forecast()['maximum_training_fullgraph_forwards']==54000
    assert not any(n in sys.modules for n in ('numpy','torch','scipy','torch_geometric'))
    print(json.dumps(dict(schema='PubMed-strong-reference-stdlib-static-check-v1',passed=True,
          compiled_sources=compiled,disabled_templates=len(disabled),fresh_body_trajectories=18,
          exact_V3_and_existing_dependencies_verified=True,own_selector_and_streams_frozen=True,
          numerical_imports=False,array_reads=False,model_calls=False,server_calls=False,launch=False,
          verifies_runtime_or_accuracy=False),indent=2,sort_keys=True))


if __name__=='__main__':main()

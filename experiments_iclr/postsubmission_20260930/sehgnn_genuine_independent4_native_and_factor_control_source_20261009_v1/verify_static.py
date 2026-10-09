"""Stdlib-only source/AST/exact binding checks. No native/model/provider runs."""
import ast
import hashlib
import json
from pathlib import Path
import sys

from independent4 import Independent4Config, IndependenceError, VARIANTS, SELECTION


def main():
    here = Path(__file__).resolve().parent
    source = (here/'independent4.py').read_text()
    tree = ast.parse(source)
    for path in here.glob('*.py'):
        compile(ast.parse(path.read_text()),str(path),'exec')
    assert not {'torch','numpy','dgl','torch_sparse','sklearn'} & set(sys.modules)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert not any(alias.name.split('.')[0] in {'torch','numpy','dgl','torch_sparse','sklearn'} for alias in node.names)
        if isinstance(node,ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in {'torch','numpy','dgl','torch_sparse','sklearn'}
    assert VARIANTS == ('plain_native','untied_same_six_factors')
    assert 'per_body_strict_native_VALID_BCE' in SELECTION
    assert 'rt[\'native\'].set_random_seed(seed)' in source and 'prototype = engine.make_model(rt, ctx, costs)' in source
    assert 'for m, seed in enumerate(seeds):' in source
    assert 'torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)' in source
    assert 'scalar = torch.cuda.amp.GradScaler()' in source
    assert "self.rt['native'].train(" in source and 'loss/4' not in source and 'loss / 4' not in source
    assert 'engine.observe_update(' in source and 'engine.evaluate(' in source and 'engine.snapshot(' in source
    assert 'for epoch in range(200):' in source and "scores['VALID']['BCE'] < best_loss" in source and 'epoch-best_epoch > 50' in source
    assert 'members=1' in source and 'extend_to_one_untied_factor_body' in source and 'members=4' not in source
    assert 'not ids & seen_ids and not storage & seen_storage' in source
    assert 'optimizer.param_groups' in source and 'separate_optimizers=True,separate_scalers=True' in source
    assert 'self.restore(group)' in source and 'Serve only restored four own-selected states' in source
    assert 'torch.cuda.amp.autocast(enabled=False)' in source and 'torch.stack([torch.sigmoid(value) for value in values],dim=0).mean(dim=0)' in source
    assert 'pool_selection=False' in source and 'source_supply_untied_correction=False' in source
    assert 'torch.load(' not in source and 'torch.save(' not in source  # root owns safe serialized file custody
    try:
        Independent4Config().require_enabled()
    except IndependenceError:
        pass
    else:
        raise AssertionError('Default enabled unexpectedly')
    verified = []
    for row in json.loads((here/'SOURCE_BINDINGS.json').read_text())['files']:
        path = here.parent/row['path']
        assert path.stat().st_size == row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        verified.append(row['path'])
    record = dict(schema='genuine-independent4-source-static-verification-v1',status='passed',
        source_sha256=hashlib.sha256((here/'independent4.py').read_bytes()).hexdigest(),
        exact_dependencies_verified=verified,
        checks=['Python AST compilation; no numerical imports','default disabled','four separate native constructors and seed streams',
                'separate native Adam/scalers; exact own helper calls without loss/4','native own selector/200epochs/patience locators',
                'explicit same-factor M1-per-body extension','cross-body object/storage/optimizer ownership locators',
                'restored own-selected state required for serving','FP32 four-member per-label pooling; no pool selection',
                'no source-supply untied correction or file/launch interface'],
        provider_model_data_checkpoint_outcome_server_or_float_execution=False,
        actual_native_four_body_or_factor_qualification=False,
        limit='Source/AST/hash/default-only. Actual ownership, updates, cost/memory, safe serialization, selected reconstruction and scientific effects remain root-owned qualification.')
    (here/'STATIC_VERIFICATION.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='passed',source_sha256=record['source_sha256'],native_execution=False)))


if __name__ == '__main__':
    main()

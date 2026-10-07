"""NumPy logical-input fixture only; no real archive, graph, model or fit."""
import ast
import copy
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from frozen_targets import validate_logical_arrays, content_sha


def main():
    ids = np.arange(580, dtype=np.int64) + 2000
    labels = np.arange(580, dtype=np.int64) % 10
    rows = np.linspace(0,579,512,dtype=np.float32).astype(np.int64)
    in_panel = np.zeros(580,dtype=bool); in_panel[rows] = True
    masks = np.eye(580,dtype=bool)[None].repeat(4,axis=0)
    permutations = np.arange(580,dtype=np.int64)[None].repeat(4,axis=0)
    rng = np.random.default_rng(714)
    for route in range(4):
        for cls in range(10):
            for present in (False,True):
                bucket = np.flatnonzero((labels==cls)&(in_panel==present))
                for position,row in enumerate(bucket):
                    masks[route,row,bucket[(position+route+1)%len(bucket)]] = True
                permutations[route,bucket] = rng.permutation(bucket)
    permuted = np.stack([masks[m][p[:,None],p[None,:]] for m,p in enumerate(permutations)])
    arrays = dict(train_ids=ids.copy(),train_labels=labels.copy(),panel_rows=rows.copy(),
                  masks=masks,permuted_masks=permuted,permutations=permutations)
    validate_logical_arrays(arrays,ids,labels,rows,rows)
    checks = ['complete structural fixture accepted without scientific data']
    cases = (
        ('TRAIN order change', lambda d: d['train_ids'].__setitem__(slice(None),ids[::-1])),
        ('TRAIN label change', lambda d: d['train_labels'].__setitem__(0,9)),
        ('panel change', lambda d: d['panel_rows'].__setitem__(slice(None),rows[::-1])),
        ('wrong-class positive', lambda d: d['masks'].__setitem__((0,0,1),True)),
        ('missing self positive', lambda d: d['masks'].__setitem__((0,0,0),False)),
        ('nonbijection permutation', lambda d: d['permutations'].__setitem__((0,1),d['permutations'][0,0])),
        ('permuted relation mismatch', lambda d: d['permuted_masks'].__setitem__((0,0,0),False)),
    )
    for name,mutate in cases:
        changed = copy.deepcopy(arrays); mutate(changed)
        try:
            validate_logical_arrays(changed,ids,labels,rows,rows)
        except ValueError:
            checks.append(name+' rejected')
        else:
            raise AssertionError(name+' accepted')
    try:
        validate_logical_arrays(arrays,ids,labels,rows,rows[::-1])
    except ValueError:
        checks.append('actual device panel mismatch rejected')
    else:
        raise AssertionError('device panel mismatch accepted')
    changed = arrays['masks'].copy(); changed[0,0,1] = True
    assert content_sha(changed) != content_sha(arrays['masks'])
    checks.append('canonical logical content hash detects target change')
    dispatch = (HERE/'context_dispatch.py').read_text()
    assert all(name not in dispatch for name in ('graph_contexts(', 'positive_masks(', 'within_class_permuted('))
    assert dispatch.index('frozen_targets.validate_current') < dispatch.index('session = public.Session')
    driver = (HERE/'train_context.py').read_text()
    assert driver.index('frozen_targets = load_frozen_targets') < driver.index('session, facade, policy, preparation = make_session')
    assert 'np.savez' not in driver and 'verified_frozen_targets' in driver
    for name in ('frozen_targets.py','context_dispatch.py','train_context.py'):
        ast.parse((HERE/name).read_text())
    checks.append('consume-only validation precedes native construction; no relation regeneration')
    report = dict(status='NUMPY_LOGICAL_CONTRACT_FIXTURE_ONLY',checks=checks,
        real_frozen_archive_opened=False, scientific_data_access=False,
        native_model_or_CUDA_execution=False, scientific_fits=0)
    (HERE/'FROZEN_TARGET_SOURCE_QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

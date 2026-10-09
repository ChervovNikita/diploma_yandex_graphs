# Use the existing V2 setup and loop

This is an **inactive caller outline**, not a new runner or executable launch script. Current V2 activation templates do not authorize this study. Root must separately bind this source/template, the exact V2 source/runtime/roles and a qualified operator capture before calling it. Default `execute=False` returns without model/data access.

Use the original V2 runner's validated data/arguments, helper imports, immutable topology placement and ordinary subprocess custody. Load the two new modules with exact source origins in that same established namespace. The borrowed `bank`, `support`, adapter/native and helper modules must remain the bound V2 ones. Do not dispatch V2's original outer3-shared/12-independent panel as this initializer study.

```python
# REVIEW OUTLINE ONLY. No code in this block ran during preparation.
# Root's qualified V2 setup supplies np, torch, roc_auc_score, old, common,
# helpers, placement, adapter, data, role_meta, identity and output.
# policy and authorization are separately source/hash-bound root records.

if authorization.get('enabled') is not True:
    raise SystemExit('Inactive source outline: no root initialization release')

cfg = policy['optimizer']
config = policy['configuration']
args = dict(config['native_args'], graph_size=data['x'].shape[0],
            input_dim=data['x'].shape[1], output_dim=2,
            device=str(data['x'].device))
native_factory = placement.make_native_placed_factory(
    torch, adapter, data['cpu_edge_index'], data['edge_index'], args)

def fresh_bank():
    return bound_V2_bank.make_bank(torch, adapter, native_factory)

for seed in policy['seeds']:
    warm = live_action_init.warm_once(
        np, torch, roc_auc_score, bound_original_V2_shared_fit,
        old, common, helpers, fresh_bank, cfg, data, seed, identity,
        output / 'common' / ('seed'+str(seed)) / 'warmup',
        execute=False, authorization=authorization)
    if warm['status'] != 'complete':
        continue  # Defaults are inactive. No incomplete warm state is used.
    screen = live_action_init.candidate_screen(
        np, torch, old, common, helpers, fresh_bank, warm, data, seed,
        output / 'common' / ('seed'+str(seed)) / 'screen',
        execute=False, authorization=authorization)
    if screen['status'] != 'complete':
        continue  # No survivor arms after a common screen failure.
    for arm in policy['arms']:
        initialization = dict(execute=False, authorization=authorization,
                              warm=warm, screen=screen, arm=arm,
                              warm_cost=warm['cost'], screen_cost=screen['cost'])
        shared_fit_initialized.fit(
            np, torch, roc_auc_score, old, common, helpers, placement,
            adapter, cfg, config, data, role_meta, identity,
            output / arm, seed, initialization=initialization)
```

In an enabled successor, the caller first verifies warm/screen `status=='complete'`, retains both records, and stops the complete four-arm seed on any common failure. It checks each branch is terminal with400 completed continuation epochs, exact selected reconstruction and a fresh serving reference before comparison. Partial/interrupted attempts remain, with no replacement. Root's existing hard-kill/resource custody owns process cleanup and partial-record retention.

`shared_fit_initialized.py` is derived directly from the sealed V2 file. `V2_SHARED_FIT_INITIALIZATION.patch` shows every change. `optimizer`, `train_update` and `evaluate` are AST-identical to V2. The fit changes only the inactive root gate, exact warm-start/factor choice, retained common warm-best, epoch101–500 range/no patience truncation, setup cost metadata and whether the selected checkpoint contains an initializer effect. The original source files were not edited.

No source-response loss or engineering-assessor call is present. The statistical/qualification gates remain future work, not source evidence of a numerical pass.

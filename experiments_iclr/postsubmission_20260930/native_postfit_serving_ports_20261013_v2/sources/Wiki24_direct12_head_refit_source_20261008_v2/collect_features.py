"""Actual saved-native head-input collector; gated, no refit or score opening."""
import argparse
from contextlib import nullcontext
import gc
import os
import random
import sys
import time
from common import *
from cohorts import masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg, output = release(args.release, args.release_sha256, 'collect')
    require(cfg.get('trusted_checkpoint_deserialization_authorized') is True
            and cfg.get('owner_and_children_terminal') is True
            and cfg.get('external_hard_bound_confirmed') is True, 'Exact terminal/trusted/finite collection authority')
    pins = read(ROOT / 'SOURCE_BINDINGS.json')
    export = read(bound(pins['metadata_export']))
    gate = export['gate']
    require(export.get('whole24_accounted') is True and export.get('reselected') is False
            and gate.get('passed') is True and gate.get('family_closed') is True
            and gate['reader_manifest_sha256'] == READER_SHA and gate['source_manifest_sha256'] == SUITE_SHA,
            'Authoritative whole24 export only')
    arms = ['single','single_contrastive','independent4','independent4_contrastive',
            'be_unit','be_init','be_unit_contrastive','be_init_contrastive']
    require([(r['arm'],r['seed']) for r in gate['cells']] == [(a,s) for s in SEEDS for a in arms], 'Whole24 fixed roster')
    terminal = read(bound(pins['terminal_evidence']))
    require(terminal.get('owner_and_children_terminal') is True and terminal['closure'] == gate['closure']
            and terminal['owner'] == gate['owner'], 'Exact closure/owner terminal custody')
    extraction = read(bound(pins['metadata_extraction_cost']))
    require(extraction.get('status') == 'complete' and extraction['metadata_export'] == pins['metadata_export'], 'Exact extraction custody')
    seal(PHASE / SUITE, SUITE_SHA)
    helper_root = PHASE / 'internal_BE_Wiki24_selected_prediction_analysis_source_20261007_v1'
    seal(helper_root, pins['original_selected_collector_manifest_sha256'])
    helper = module(helper_root / 'collect.py', 'direct12_original_restore')
    runtime = module(PHASE / SUITE / 'runtime.py', 'direct12_original_runtime')
    runtime.allocation()
    runtime.runtime_versions()
    import numpy as np
    import torch
    factors = module(PHASE / SUITE / 'factors.py', 'direct12_original_factors')
    previous = sys.modules.get('factors')
    try:
        sys.modules['factors'] = factors
        models = module(PHASE / SUITE / 'models.py', 'direct12_original_models')
    finally:
        if previous is None: sys.modules.pop('factors', None)
        else: sys.modules['factors'] = previous
    data = module(PHASE / SUITE / 'data.py', 'direct12_original_data')
    dependency = read(PHASE / SUITE / 'DEPENDENCIES.json')['polynormer']
    native = models.load_source(bound(dependency), dependency['sha256'], 'direct12_native_polynormer')
    train, dev, authority = data.load_projection(PHASE, gate['data_manifest_binding'], 'wikics', True)
    ids = torch.cat((train['ids'], dev['ids'])).to('cuda:0')
    batch = dict(x=train['x'].to('cuda:0'), edge_index=train['edge_index'].to('cuda:0'), ids=ids)
    require(len(train['y']) == 580 and len(dev['y']) == 5274 and ids.numel() == 5854, 'Complete roles')
    metadata = {(r['arm'],r['seed']):r for r in export['cells']}
    states = {(r['arm'],r['seed']):r for r in gate['cells']}
    output.mkdir(mode=0o700)
    records = []
    calls = 0
    deadline = started + 3590
    for pin in pins['selected_custody']:
        arm, seed = pin['arm'], pin['seed']
        state, meta = states[(arm,seed)], metadata[(arm,seed)]
        row = dict(arm=arm, seed=seed, cell=state['cell'], status='failed', attempted_member_forwards=0)
        records.append(row)
        model = None
        try:
            require(time.monotonic() < deadline and state['status'] == 'complete', 'Original complete cell and finite time required')
            require(state['selected_checkpoint'] == pin['selected_checkpoint'], 'Exact closed selected checkpoint')
            for owned in state.get('own_checkpoints', []): bound(owned)
            if 'own_bank_metric_json' in state: bound(state['own_bank_metric_json'])  # Hash only; scores remain unopened.
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model = models.Ensemble('wikics', arm, seed, gate['config']['model'], (native, None)).to('cuda:0')
            saved = torch.load(bound(state['selected_checkpoint']), map_location='cpu', weights_only=False)
            _, modes, streams = helper.restore(torch, model, saved, state, meta, gate['config'])
            require(modes == pin['selected_member_modes'], 'Exact original per-member selected modes')
            del saved
            Hrows, Lrows, weights, biases, rs, ss = [], [], [], [], [], []
            with torch.no_grad():
                for member in range(model.members):
                    require(time.monotonic() < deadline and calls < 27, 'Frozen member-forward/time budget')
                    body = model.models[member if model.independent else 0].body
                    head = body.pred_global if body._global else body.pred_local
                    captured, outputs = [], []
                    def before(module, arguments):
                        if isinstance(module, factors.FactorLinear):
                            require(module.member == member, 'Actual factor member index correspondence')
                        captured.append(arguments[0].detach().clone())
                    hp = head.register_forward_pre_hook(before)
                    ho = head.register_forward_hook(lambda _, arguments, value: outputs.append(value.detach().clone()))
                    calls += 1; row['attempted_member_forwards'] += 1
                    try:
                        context = torch.random.fork_rng(devices=[0]) if streams is not None else nullcontext()
                        with context:
                            if streams is not None:
                                torch.set_rng_state(streams[member]['cpu']); torch.cuda.set_rng_state(streams[member]['cuda'])
                            logits, representation = model.member_forward(batch, member)
                    finally:
                        hp.remove(); ho.remove()
                    require(len(captured) == len(outputs) == 1 and captured[0].shape == (11701,512)
                            and outputs[0].shape == (11701,10), 'One exact native head call and full graph')
                    require(torch.allclose(representation, captured[0][ids], rtol=1e-6, atol=1e-6)
                            and torch.allclose(logits, outputs[0][ids], rtol=1e-6, atol=1e-6), 'Hook/native return ID correspondence')
                    require(bool(torch.isfinite(representation).all()) and bool(torch.isfinite(logits).all()), 'Finite frozen inputs/outputs')
                    Hrows.append(representation.cpu().numpy().copy()); Lrows.append(logits.cpu())
                    weights.append(head.weight.detach().cpu().numpy().copy())
                    require(head.bias is not None, 'Actual native bias required')
                    biases.append(head.bias.detach().cpu().numpy().copy())
                    if isinstance(head, factors.FactorLinear):
                        rs.append(head.r[member].detach().cpu().numpy().copy()); ss.append(head.s[member].detach().cpu().numpy().copy())
            H = np.stack(Hrows); logits = torch.stack(Lrows)
            probability = logits.softmax(-1); pooled = probability.mean(0)
            arrays = dict(H_train=H[:,:580], H_dev=H[:,580:], y_train=train['y'].numpy(), y_dev=dev['y'].numpy(),
                train_ids=train['ids'].numpy(), dev_ids=dev['ids'].numpy(), native_dev_logits=logits[:,580:].numpy(),
                native_dev_probability=probability[:,580:].numpy(), native_dev_pool=pooled[580:].numpy())
            if arm == 'be_unit_contrastive':
                require(len(rs) == 4 and all(np.array_equal(weights[0],v) for v in weights)
                        and all(np.array_equal(biases[0],v) for v in biases), 'Original shared W/common bias')
                arrays.update(W=weights[0], r=np.stack(rs), s=np.stack(ss), bias=biases[0])
            else:
                require(not rs, 'Native ordinary/single heads must be unfactorized')
                arrays.update(A=np.stack(weights), bias=np.stack(biases))
            path = output / (state['cell'] + '.npz'); np.savez(path, **arrays)
            row.update(status='complete', features=binding(path), selected_checkpoint=state['selected_checkpoint'], selected_member_modes=modes)
            if arm == 'be_unit_contrastive':
                cohort = masks(np, arrays['native_dev_logits'], arrays['native_dev_probability'], arrays['native_dev_pool'], arrays['y_dev'])
                cp = output / ('native_cohorts_' + str(seed) + '.npz')
                np.savez(cp, dev_ids=arrays['dev_ids'], y_dev=arrays['y_dev'], **cohort)
                row['native_cohorts'] = binding(cp)  # No cohort counts or comparative scores opened.
            bound(state['selected_checkpoint'])
        except Exception as error:
            row.update(error_type=type(error).__name__, error=str(error))
        finally:
            del model
            gc.collect(); torch.cuda.empty_cache()
    collection = dict(schema='Wiki24-direct12-features-v1', rows=records, unique_saved_banks=9,
        maximum_member_forwards=27, attempted_member_forwards=calls, complete_TRAIN=580, complete_development=5274,
        data_manifest=gate['data_manifest_binding'], metadata_export=pins['metadata_export'],
        native_cohort_authority='untouched be_unit_contrastive native FP32 mean-probability predictions',
        native_original_scores_unchanged=True, refits=0, TEST_access=False, scores_opened=False,
        inclusive_wall_seconds=time.monotonic()-started)
    write(output / 'COLLECTION.json', collection)
    write(output / 'SEAL.json', dict(collection=binding(output/'COLLECTION.json'), closed=True,
                                    source_manifest_sha256=cfg['source_manifest_sha256'], development_comparative_opening=False))


if __name__ == '__main__': main()

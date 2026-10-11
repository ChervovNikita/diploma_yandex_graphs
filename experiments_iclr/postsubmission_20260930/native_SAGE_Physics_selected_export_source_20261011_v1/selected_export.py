"""Inactive selected-state core exporter; no fit, target reader or scorer.

Root passes the admitted complete Physics Family and owns the joined90-state
closure and later60-bank common calibration admission. Import is stdlib-only.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ARMS = {
    'BASE': ('none', 4, 1, 'shared'),
    'E4': ('endpoint', 4, 1, 'shared'),
    'N4': ('node', 4, 1, 'shared'),
    'equipped_M1': ('endpoint', 1, 1, 'single'),
    'equipped_I4': ('endpoint', 4, 4, 'single'),
}


def receipt(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def checked(item):
    if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
        raise ValueError('Exact immutable file receipt required')
    if receipt(item['path']) != item: raise ValueError('File custody differs')
    return Path(item['path'])


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def export_all(family, complete_fit_path, role_blocks, output, root_validate, release):
    """Restore all50 core banks/80 fits; emit only label-free CAL/CONFIRM.

    family: Physics V5 root-admitted Family, scientific safe TRAIN/SELECT
    loader, no active model. complete_fit_path: its COMPLETE_FAMILY.json.
    role_blocks: all ten seed strings, each exact{cal_ids,confirm_ids} receipts.
    release: status ROOT_SELECTED_EXPORT_ADMITTED, joined_selected_states90
    ROOT_VERIFIED_COMPLETE, limits{wall_seconds,cuda_bytes,rss_bytes}.
    root_validate(family.cfg, action='fixed_Physics_selected_core_export',
                  source_receipt=..., complete_fit_receipt=...,
                  role_blocks=..., release=...) must return custody with
    status ROOT_SELECTED_EXPORT_ADMITTED, binding all90 selected states,
    these80 core state receipts, memberships, source/runtime and caps.
    Returns/writes COMPLETE_SELECTED_CORE_PREDICTIONS.json with50 records.
    """
    f = family; output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); records = []
    result = {'terminal': False, 'complete': False, 'records': records,
              'CAL_CONFIRM_truth_access': False, 'fit_updates': 0,
              'automatic_calibration_or_CONFIRM_release': False}
    try:
        limits = release.get('limits', {})
        if (release.get('status') != 'ROOT_SELECTED_EXPORT_ADMITTED'
                or release.get('joined_selected_states90') != 'ROOT_VERIFIED_COMPLETE'
                or not callable(root_validate) or f.training_only or f._fit_model is not None
                or f.cfg['profile']['dataset'] != 'CoauthorPhysics'
                or f.cfg['seeds'] != list(SEEDS)
                or set(limits) != {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
                or any(type(v) is not int or v <= 0 for v in limits.values())
                or set(role_blocks) != {str(s) for s in SEEDS}
                or any(set(b) != {'cal_ids', 'confirm_ids'} for b in role_blocks.values())):
            raise ValueError('Inactive source: entire fixed state closure and root export admission required')
        core_path = Path(complete_fit_path).resolve()
        core = json.loads(core_path.read_text())
        banks = core.get('results', [])
        if (core.get('complete') is not True or core.get('full_same_runtime_stage_complete') is not True
                or core.get('groups') != 50 or core.get('fit_units') != 80
                or core.get('fixed_seed_roster') != list(SEEDS)
                or core.get('fixed_arm_roster') != list(ARMS)
                or len(banks) != 50 or {(r['seed'], r['arm']) for r in banks} != {(s, a) for s in SEEDS for a in ARMS}):
            raise ValueError('Complete original fixed50-bank/80-fit core closure required')
        custody = root_validate(f.cfg, action='fixed_Physics_selected_core_export',
                                source_receipt=receipt(__file__), complete_fit_receipt=receipt(core_path),
                                role_blocks=role_blocks, release=release)
        if not isinstance(custody, dict) or custody.get('status') != 'ROOT_SELECTED_EXPORT_ADMITTED':
            raise ValueError('Root must admit all90 selected states before selected prediction export')
        result['custody'] = custody
        t, np = f.torch, f.np
        def budget():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if time.perf_counter() - started > limits['wall_seconds'] or rss > limits['rss_bytes']:
                raise RuntimeError('Finite selected export wall/RSS cap exceeded')
            if f.device.type == 'cuda' and t.cuda.max_memory_allocated(f.device) > limits['cuda_bytes']:
                raise RuntimeError('Finite selected export CUDA cap exceeded')
        budget(); index = {(r['seed'], r['arm']): r for r in banks}
        for seed in SEEDS:
            f.activate_role_block(seed)
            block = role_blocks[str(seed)]
            cal = np.load(checked(block['cal_ids']), allow_pickle=False)
            confirm = np.load(checked(block['confirm_ids']), allow_pickle=False)
            if cal.dtype != np.int64 or cal.shape != (150,) or confirm.dtype != np.int64 or confirm.shape != (34093,):
                raise ValueError('Exact Physics label-free CAL/CONFIRM memberships required')
            union = np.concatenate((cal, confirm))
            all_ids = np.concatenate((f.train_ids.cpu().numpy(), f.valid_ids.cpu().numpy(), union))
            if not np.array_equal(np.sort(all_ids), np.arange(34493, dtype=np.int64)):
                raise ValueError('Complete unique disjoint Physics role membership required')
            ids = t.from_numpy(union).to(f.device)
            for arm, (operation, members, bundles, architecture) in ARMS.items():
                original = index[seed, arm]; fits = original['fits']
                if len(fits) != bundles or original['native_trajectories_per_forward'] != members:
                    raise ValueError('Frozen core ownership differs')
                folder = output / f'{arm}_seed{seed}'; folder.mkdir()
                f._operation = operation; outputs = []; retained = []
                f.synchronize(); tick = time.perf_counter(); budget()
                if f.device.type == 'cuda': t.cuda.reset_peak_memory_stats(f.device)
                for member, fit in enumerate(fits):
                    checkpoint = checked(fit['selected_state_receipt'])
                    model, optimizer, streams = f.make(seed, member, 'baseline', True,
                                                       1 if bundles == 4 else members, architecture)
                    state = t.load(checkpoint, map_location='cpu', weights_only=True)
                    if state['step'] != fit['selected_step'] or state['selection'] != fit['selection']:
                        raise ValueError('Selected-step/selector custody differs')
                    model.load_state_dict(state['model'], strict=True)
                    optimizer.load_state_dict(state['optimizer']); streams = state['streams']
                    model.eval()
                    with t.no_grad(): logits = f.logits(model, streams, members > 1 and bundles == 1, ids)
                    expected_members = 1 if bundles == 4 else members
                    if logits.shape != (expected_members, 34243, 5) or not t.isfinite(logits).all().item():
                        raise ValueError('Selected full-graph route output differs')
                    outputs.append(logits.cpu()); retained.append({'status': 'SELECTED_COMPLETE',
                        'selected_state_receipt': fit['selected_state_receipt'], 'costs': fit['costs'],
                        'selected_step': fit['selected_step'], 'member': member})
                    f._fit_model = f._single_adapter = None
                    del model, optimizer, streams, state, logits; budget()
                raw = t.cat(outputs); del outputs
                logp = raw.log_softmax(-1)
                member_argmax = raw.argmax(-1)
                pool_argmax = raw.softmax(-1).mean(0).argmax(-1)
                archives = {}
                for role, membership, rows in (('CAL', cal, slice(0, 150)), ('CONFIRM', confirm, slice(150, None))):
                    path = folder / ('selected_' + role + '.npz')
                    np.savez(path, ids=membership, member_log_probs=logp[:, rows].numpy(),
                             native_member_argmax=member_argmax[:, rows].numpy(),
                             native_pool_argmax=pool_argmax[rows].numpy())
                    archives[role] = receipt(path)
                f.synchronize()
                record = {'block': seed, 'seed': seed, 'arm': arm, 'status': 'SELECTED_COMPLETE',
                          'members': members, 'fits': retained,
                          'selected_CAL_archive': archives['CAL'], 'selected_CONFIRM_archive': archives['CONFIRM'],
                          'restored_fit_units': bundles, 'selected_serving_routes': members,
                          'CAL_CONFIRM_truth_access': False, 'representation': 'member_log_probs_float32',
                          'native_pool_authority': 'float32 raw_logits.softmax(-1).mean(0).argmax(-1)',
                          'export_seconds': time.perf_counter() - tick}
                write(folder / 'RESULT.json', record); records.append(record)
                del raw, logp, member_argmax, pool_argmax; budget()
            del ids, union, all_ids, cal, confirm
        if len(records) != 50 or sum(len(r['fits']) for r in records) != 80:
            raise ValueError('Entire50-bank/80-state core export required')
        result.update(terminal=True, complete=True, banks=50, fits=80,
                      selected_serving_routes=sum(r['selected_serving_routes'] for r in records))
    except BaseException as e:
        result.update(terminal=True, failure={'type': type(e).__name__, 'message': str(e)}, retry=False)
        result['entire_export_seconds'] = time.perf_counter() - started
        write(output / 'COMPLETE_SELECTED_CORE_PREDICTIONS.json', result); raise
    result['entire_export_seconds'] = time.perf_counter() - started
    write(output / 'COMPLETE_SELECTED_CORE_PREDICTIONS.json', result)
    return result


if __name__ == '__main__':
    raise SystemExit('Inactive source. Root supplies admitted Family, all90 states and export custody.')

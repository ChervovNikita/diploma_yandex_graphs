#!/usr/bin/env python3
"""Disabled by the accompanying release; reads saved logits only, prints JSON.

No model imports, checkpoint loads, forwards, fits, input split loads, or network.
The prepared source has not been run on prediction payloads.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CELLS = ('E_end_joint', 'E_end_live', 'S_end_joint', 'J4_end_joint')
BLOCKS = ('b0', 'b1', 'b2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def local_file(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Relative artifact path required')
    path = root / rel
    for part in [path, *path.parents]:
        if part == root.parent:
            break
        require(not part.is_symlink(), 'Artifact symlinks are not admitted')
    require(path.is_file() and path.resolve().is_relative_to(root), 'Bound local bank unavailable')
    require(path.name == 'selected_VALID_logits.pt', 'Only selected saved VALID logits admitted')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    require(release.get('enabled') is True and release.get('source_review_approved') is True,
            'Prepared analysis is disabled pending root release and source review')
    require(release.get('scope') == 'retained_selected_VALID_logits_common_negative_analysis'
            and release.get('TEST_access') is False and release.get('model_execution') is False
            and release.get('checkpoint_reselection') is False, 'Scope differs')
    require(release.get('script_sha256') == sha(Path(__file__).resolve()), 'Source bytes changed')
    bindings_path = HERE / 'BINDINGS.json'
    require(release.get('bindings_sha256') == sha(bindings_path), 'Bindings changed')
    bindings = json.loads(bindings_path.read_text())
    root = Path(release['artifact_root']).resolve(strict=True)
    require(root.is_dir(), 'Existing local artifact root required')
    targets = bindings['targets']
    require(len(targets) == 12 and {(r['block'], r['cell']) for r in targets}
            == {(b, c) for b in BLOCKS for c in CELLS}, 'All twelve fixed cells required')
    paths = {}
    for row in targets:
        path = local_file(root, row['logits_relative'])
        require(sha(path) == row['logits_sha256'], 'Saved bank bytes changed: ' + row['cell_id'])
        paths[row['cell_id']] = path
    # No numeric library or tensor deserialization precedes all twelve hashes.
    import torch
    require(str(torch.__version__) == bindings['numeric_runtime']['torch'], 'Declared CPU reader runtime differs')
    require('weights_only' in inspect.signature(torch.load).parameters, 'Safe saved tensor loader required')
    torch.set_num_threads(1)
    bank_metrics, per_query, masks = [], [], {}
    with torch.no_grad():
        for row in targets:
            bank = torch.load(paths[row['cell_id']], map_location='cpu', weights_only=True)
            require(set(bank) == {'member_pos', 'member_neg', 'mean_pos', 'mean_neg',
                                  'selected_cycle', 'inputs', 'checkpoint_sha256'}, 'Bank schema differs')
            require(bank['inputs'] == bindings['input_identities']
                    and bank['selected_cycle'] == row['selected_cycle']
                    and bank['checkpoint_sha256'] == row['checkpoint_sha256'], 'State or input custody differs')
            p, n, pp, pn = [bank[k] for k in ('member_pos', 'member_neg', 'mean_pos', 'mean_neg')]
            count = row['member_count']
            for tensor, shape in [(p, (227, count)), (n, (227, 500, count)),
                                  (pp, (227,)), (pn, (227, 500))]:
                require(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                        and tensor.dtype == torch.float32 and tuple(tensor.shape) == shape
                        and bool(torch.isfinite(tensor).all()), 'Finite native bank geometry differs')
            require(torch.equal(pp, p.mean(-1)) and torch.equal(pn, n.mean(-1)), 'Saved mean logits differ')
            # Float64 subtracts two stored FP32 values without changing comparison signs.
            margins = p.double()[:, None, :] - n.double()
            pooled_margin = pp.double()[:, None] - pn.double()
            wrong = n > p[:, None, :]
            common, union = wrong.all(-1), wrong.any(-1)
            pooled_wrong, pooled_right = pn > pp[:, None], pn < pp[:, None]
            survive = common & pooled_wrong
            masks[(row['block'], row['cell'])] = {'common': common, 'union': union,
                                                 'pooled_wrong': pooled_wrong, 'pooled_right': pooled_right,
                                                 'survive': survive}
            common_count, union_count = common.sum(1), union.sum(1)
            surviving_count, pooled_count = survive.sum(1), pooled_wrong.sum(1)
            defined_fractions = [ratio(int(a), int(b)) for a, b in zip(surviving_count, pooled_count) if int(b)]
            bank_metrics.append({'block': row['block'], 'cell': row['cell'], 'member_count': count,
                'mean_common_strict_count': float(common_count.double().mean()),
                'mean_common_surviving_pool_count': float(surviving_count.double().mean()),
                'mean_union_strict_count': float(union_count.double().mean()),
                'common_over_union_slot_fraction': ratio(int(common.sum()), int(union.sum())),
                'common_share_of_pooled_strict_errors_slot_fraction': ratio(int(survive.sum()), int(pooled_wrong.sum())),
                'mean_query_common_share_given_pooled_strict_error': sum(defined_fractions)/len(defined_fractions) if defined_fractions else None,
                'queries_with_common_strict_negative': int((common_count > 0).sum()),
                'queries_with_at_least_ten_common_surviving_pool_negatives': int((surviving_count >= 10).sum()),
                'common_slots_not_strictly_wrong_in_saved_pool': int((common & ~pooled_wrong).sum()),
                'single_member_common_definition_is_vacuous': count == 1})
            for q in range(227):
                slots = common[q].nonzero(as_tuple=False).flatten().tolist()
                per_query.append({'block': row['block'], 'cell': row['cell'], 'query_slot': q,
                    'member_strict_wrong_counts': wrong[q].sum(0).tolist(),
                    'common_strict_count': int(common_count[q]), 'union_strict_count': int(union_count[q]),
                    'pooled_strict_wrong_count': int(pooled_count[q]),
                    'pooled_tied_count': int((pn[q] == pp[q]).sum()),
                    'common_surviving_pool_count': int(surviving_count[q]),
                    'RR_upper_bound_from_common_surviving_pool_count': 1/(1+int(surviving_count[q])),
                    'common_negatives': [{'negative_slot': j, 'member_signed_margins': margins[q, j].tolist(),
                        'float64_mean_signed_margin': float(margins[q, j].mean()),
                        'saved_pool_signed_margin': float(pooled_margin[q, j])} for j in slots]})
    comparisons = []
    for block in BLOCKS:
        for left, right in [('E_end_joint', 'J4_end_joint'), ('E_end_live', 'E_end_joint')]:
            a, b = masks[(block, left)], masks[(block, right)]
            common_a, common_b = a['common'].sum(1), b['common'].sum(1)
            fraction_a = ratio(int(a['common'].sum()), int(a['union'].sum()))
            fraction_b = ratio(int(b['common'].sum()), int(b['union'].sum()))
            comparisons.append({'block': block, 'left': left, 'right': right,
                'mean_query_common_strict_count_difference': float((common_a-common_b).double().mean()),
                'mean_query_common_surviving_pool_count_difference': float((a['survive'].sum(1)-b['survive'].sum(1)).double().mean()),
                'common_over_union_slot_fraction_difference': fraction_a-fraction_b if fraction_a is not None and fraction_b is not None else None,
                'left_common_slots_strictly_correct_in_right_pool': int((a['common'] & b['pooled_right']).sum()),
                'left_common_slots_strictly_correct_in_right_pool_fraction': ratio(int((a['common'] & b['pooled_right']).sum()), int(a['common'].sum()))})
        for left in ('E_end_joint', 'E_end_live'):
            a, s = masks[(block, left)], masks[(block, 'S_end_joint')]
            comparisons.append({'block': block, 'left': left, 'right': 'S_end_joint',
                'left_common_slots_strictly_correct_in_single': int((a['common'] & s['pooled_right']).sum()),
                'left_common_slots_strictly_correct_in_single_fraction': ratio(int((a['common'] & s['pooled_right']).sum()), int(a['common'].sum()))})
    equal_block_summary = {}
    for cell in CELLS:
        rows = [r for r in bank_metrics if r['cell'] == cell]
        equal_block_summary[cell] = {}
        for field in ('mean_common_strict_count', 'mean_common_surviving_pool_count',
                      'common_over_union_slot_fraction', 'common_share_of_pooled_strict_errors_slot_fraction'):
            values = [r[field] for r in rows]
            equal_block_summary[cell][field] = sum(values)/3 if all(v is not None for v in values) else None
    result = {'schema': 'complete39_fixed12_common_negative_diagnostic_v1',
        'scope': release['scope'], 'bindings_sha256': sha(bindings_path), 'bank_metrics': bank_metrics,
        'equal_block_descriptive_summary': equal_block_summary,
        'paired_block_comparisons': comparisons, 'per_query': per_query,
        'uncertainty': 'Three training blocks share the same graph and 227 queries; no iid query or slot confidence intervals.',
        'interpretation': 'Slot identities are fixed by hashed inputs. Selected VALID association only; no causal graph erasure or novel method conclusion.',
        'TEST_access': False, 'model_execution': False, 'new_fits': 0, 'checkpoint_reselection': False}
    json.dump(result, sys.stdout, allow_nan=False, sort_keys=True)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()

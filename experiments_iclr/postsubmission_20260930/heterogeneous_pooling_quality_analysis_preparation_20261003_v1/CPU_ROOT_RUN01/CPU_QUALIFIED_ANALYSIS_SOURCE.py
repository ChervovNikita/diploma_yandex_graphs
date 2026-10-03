"""Development-only decomposition after complete frozen HGT family closure.

This explains saved predictions; it never trains, chooses a model, opens the
dataset archive, or substitutes diagnostic FP64 scores for recorded results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FROZEN_SHA = '29885a100527226e9d182c54567e748f17f384254d23e009d9089fb18352d352'
SEEDS = [131, 137, 139, 149, 151]
ARMS = ['native_HGT', 'global_BE', 'shared_relation', 'CP', 'unrestricted', 'untied_HGT', 'wider_BE']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def verified(record):
    path = Path(record['path'])
    require(path.is_file() and digest(path) == record['sha256'], 'Input fingerprint differs')
    require('bytes' not in record or path.stat().st_size == record['bytes'], 'Input length differs')
    return path


def closed_family(study):
    """No partial-success summaries or changed denominator may enter analysis."""
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    rows = study['rows']
    require(len(rows) == 35 and {(r['seed'], r['arm']) for r in rows} == expected,
            'Exactly all frozen 35 cases required')
    require(all(r['status'] == 'selected' and r['selected_state_replay'] is True
                and r['final_labels_closed'] is True for r in rows), 'All selected replays required')
    require(study['summary']['status'] == 'complete_development_summary'
            and study['final_labels_closed'] is True
            and study['original_inputs_verified_unchanged'] is True,
            'Complete preserved development study required')
    require(study['admission']['study_freeze_sha256'] == FROZEN_SHA,
            'Different study freeze')
    return {(r['seed'], r['arm']): r for r in rows}


def decomposition(torch, members, labels):
    """Known geometric-pool ambiguity identity, evaluated in FP64."""
    values = members.double()
    log_members = values.log_softmax(-1)
    log_pool = values.mean(0).log_softmax(-1)
    member_nll = -log_members.gather(-1, labels[None, :, None].expand(values.shape[0], -1, 1)).squeeze(-1)
    pool_nll = -log_pool.gather(-1, labels[:, None]).squeeze(-1)
    ambiguity = member_nll.mean(0) - pool_nll
    kl = (log_pool.exp()[None] * (log_pool[None] - log_members)).sum(-1).mean(0)
    residual = float((ambiguity - kl).abs().max())
    require(residual <= 1e-10 and float(ambiguity.min()) >= -1e-10,
            'Loss/pooling-matched ambiguity identity failed')
    return dict(mean_member_NLL=float(member_nll.mean()), pool_NLL_FP64=float(pool_nll.mean()),
                mean_ambiguity=float(ambiguity.mean()), mean_KL_pool_to_member=float(kl.mean()),
                max_identity_residual=residual)


def error_structure(torch, members, labels, served_logits=None):
    """Predictions use original tensor arithmetic, matching served pooling."""
    member_predictions = members.argmax(-1)
    pooled = (members.mean(0) if served_logits is None else served_logits).argmax(-1)
    errors = member_predictions != labels[None]
    pairs = [(a, b) for a in range(len(members)) for b in range(a + 1, len(members))]
    return dict(pooled_error_fraction=float((pooled != labels).double().mean()),
                every_member_wrong_fraction=float(errors.all(0).double().mean()),
                any_member_correct_fraction=float((~errors).any(0).double().mean()),
                mean_pair_disagreement=None if not pairs else sum(float((member_predictions[a] != member_predictions[b]).double().mean()) for a, b in pairs) / len(pairs),
                mean_pair_joint_error=None if not pairs else sum(float((errors[a] & errors[b]).double().mean()) for a, b in pairs) / len(pairs),
                member_coverage_is_not_a_pooled_accuracy_upper_bound=True)


def selection_check(rows, selection):
    require(rows and [r['epoch'] for r in rows] == list(range(1, len(rows) + 1)),
            'Trace must contain all consecutive post-update epochs')
    require(all(math.isfinite(r['validation_NLL']) for r in rows), 'Nonfinite trace')
    minimum = min(r['validation_NLL'] for r in rows)
    latest = max(r['epoch'] for r in rows if r['validation_NLL'] == minimum)
    require(selection['epoch'] == latest and selection['validation_NLL'] == minimum,
            'Native latest-tie validation selection differs')


def run(freeze_path, study_path, output):
    require(digest(freeze_path) == FROZEN_SHA, 'Exact prospective freeze required')
    freeze = json.loads(Path(freeze_path).read_text())
    require(freeze['seeds'] == SEEDS and freeze['arms'] == ARMS and freeze['test_labels_closed'],
            'Frozen denominator or label scope differs')
    original = [descriptor(freeze_path), descriptor(study_path), freeze['development_labels']]
    study = json.loads(Path(study_path).read_text())
    cases = closed_family(study)
    # No checkpoint/logit/label payload is opened before complete closure.
    labels = json.loads(verified(freeze['development_labels']).read_text())
    require(labels['scope'] == 'TRAIN_VAL_ONLY' and labels['archive_sha256'] == freeze['archive']['sha256'],
            'Explicit development labels only')
    require(labels['train_class_schema'] == [0, 1, 2, 3], 'Exact four-class schema')
    label_map = dict(zip(labels['node_ids'], labels['labels']))
    require(len(label_map) == 1217, 'Complete admitted development pool')
    import torch
    require(torch.__version__.split('+')[0] == '2.1.2', 'Pinned analysis runtime')
    torch.set_num_threads(1)
    reports, paired = [], []
    for split_row in freeze['splits']:
        original.append(split_row['descriptor'])
        split = json.loads(verified(split_row['descriptor']).read_text())
        seed = split_row['seed']
        ids = split['validation_ids']
        require(len(ids) == 243 and len(set(ids)) == 243 and all(i in label_map for i in ids),
                'Exact paired validation membership')
        indices = torch.tensor(ids, dtype=torch.long)
        targets = torch.tensor([label_map[i] for i in ids], dtype=torch.long)
        block = {}
        for arm in ARMS:
            case = Path(study_path).parent / f'seed{seed}' / arm
            selection_path = case / 'SELECTION.json'
            trace_path = case / 'TRACE.jsonl'
            logits_path = case / 'selected_member_logits.pt'
            terminal = cases[(seed, arm)]
            for key, path in (('selection_receipt', selection_path), ('training_trace', trace_path),
                              ('selected_logits', logits_path)):
                require(Path(terminal[key]['path']).resolve() == path.resolve(), 'Case output slot differs')
                verified(terminal[key])
            original.extend(descriptor(path) for path in (selection_path, trace_path, logits_path))
            selection = json.loads(selection_path.read_text())
            require(all(key in terminal and terminal[key] == value for key, value in selection.items()),
                    'Study/case terminal mismatch')
            selection_check([json.loads(line) for line in trace_path.read_text().splitlines()], selection['selection'])
            raw = torch.load(logits_path, map_location='cpu', weights_only=True)
            require(isinstance(raw, torch.Tensor) and raw.shape == (1 if arm == 'native_HGT' else 4, 4057, 4)
                    and raw.dtype == torch.float32 and bool(torch.isfinite(raw).all()), 'Exact saved logits shape/type')
            members = raw[:, indices]
            served_logits = raw.mean(0)[indices]
            native_nll = float(torch.nn.functional.cross_entropy(served_logits, targets))
            require(abs(native_nll - selection['selection']['validation_NLL']) <= 1e-7,
                    'Saved prediction/source selected NLL mismatch')
            row = dict(seed=seed, arm=arm, source_selected_NLL=native_nll,
                       diagnostic_FP64_scope='accounting identity only; does not replace source score',
                       **decomposition(torch, members, targets), **error_structure(torch, members, targets, served_logits))
            block[arm] = (row, served_logits.argmax(-1))
            reports.append(row)
        cp, cp_prediction = block['CP']
        for arm in ARMS:
            if arm == 'CP':
                continue
            control, control_prediction = block[arm]
            member_delta = cp['mean_member_NLL'] - control['mean_member_NLL']
            ambiguity_delta = cp['mean_ambiguity'] - control['mean_ambiguity']
            pool_delta = cp['pool_NLL_FP64'] - control['pool_NLL_FP64']
            require(abs(pool_delta - (member_delta - ambiguity_delta)) <= 1e-10,
                    'Paired accounting decomposition failed')
            paired.append(dict(seed=seed, control=arm,
                               source_CP_minus_control_NLL=cp['source_selected_NLL'] - control['source_selected_NLL'],
                               diagnostic_mean_member_NLL_delta=member_delta, diagnostic_ambiguity_delta=ambiguity_delta,
                               diagnostic_pool_NLL_delta=pool_delta,
                               CP_rescues_control_error=int(((cp_prediction == targets) & (control_prediction != targets)).sum()),
                               CP_harms_control_correct=int(((cp_prediction != targets) & (control_prediction == targets)).sum()),
                               validation_nodes=243))
    for record in original:
        verified(record)
    result = dict(scope='complete development only; selected validation observations, no heldout or causal inference',
                  known_identity_attribution='Wood et al., JMLR 2023, loss/pooling-matched diversity decomposition',
                  new_theorem_claim=False, final_labels_opened=False, original_inputs_preserved=True,
                  comparison_denominator=35, rows=reports, paired_CP_minus_control=paired, source_inputs=original)
    output = Path(output)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.freeze, args.study, args.output)
    print(json.dumps(dict(comparison_denominator=result['comparison_denominator'], heldout=False,
                          original_inputs_preserved=result['original_inputs_preserved'])))

"""Selected factual and masked-context diagnostics; no import-time numerics."""
import math
import random


def accuracy_counts(predictions, labels):
    correct = int((predictions == labels).sum().item())
    return dict(correct=correct, count=len(labels), accuracy=correct/len(labels))


def classification(logits, probabilities, ids, labels):
    import torch
    if logits.shape != (4, 19717, 3) or probabilities.shape != (19717, 3):
        raise ValueError('Complete four-member logits and factual probabilities required')
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite selected/evaluated logits')
    values = logits[:, ids]
    log_members = values.log_softmax(-1)
    log_pool = torch.logsumexp(log_members, dim=0)-math.log(4)
    prediction = probabilities[ids].argmax(-1)
    pooled = accuracy_counts(prediction, labels)
    selected = log_pool.gather(1, labels[:, None]).flatten()
    pooled['NLL'] = float(-selected.mean().item())
    members = []
    for member in range(4):
        row = accuracy_counts(values[member].argmax(-1), labels)
        row['NLL'] = float(-log_members[member].gather(1, labels[:, None]).mean().item())
        members.append(row)
    classes = []
    for c in range(3):
        chosen = labels == c
        if not chosen.any(): raise ValueError('Every frozen class must be represented')
        row = dict(class_id=c, **accuracy_counts(prediction[chosen], labels[chosen]))
        row['NLL'] = float(-selected[chosen].mean().item())
        row['members'] = [dict(**accuracy_counts(values[m, chosen].argmax(-1), labels[chosen]),
                               NLL=float(-log_members[m, chosen, c].mean().item())) for m in range(4)]
        classes.append(row)
    return dict(pooled=pooled, members=members,
                mean_member_accuracy=sum(m['accuracy'] for m in members)/4,
                worst_member_accuracy=min(m['accuracy'] for m in members),
                classes=classes, macro_accuracy=sum(c['accuracy'] for c in classes)/3)


def mask_diagnostics(session, method, roles):
    import torch
    from torch.nn import functional as F
    structural = dict(eligible_partition_counts=[len(ids) for ids in session.anchor_ids],
                      partition_seed=190101, ownership='persistent route m owns quarter m',
                      uses_public_feature_targets_only=True, affects_selection=False)
    if session.name != 'shared4_core':
        return dict(available=False, reason='Exact own-only V3 condition has no masked views or decoder', **structural), None
    modules = [module for body in session.bodies for module in body.modules()]+list(session.decoder.modules())
    modes = [(module, module.training) for module in modules]
    try:
        for body in session.bodies: body.eval()
        session.decoder.eval()
        banks, routes = [], []
        with torch.no_grad():
            for member in range(4):
                logits, hidden = session._forward(member, 'masked', member)
                banks.append(logits)
                ids = session.anchor_ids[member]
                reconstruction = session.decoder(hidden[ids])
                generator = random.Random()
                generator.setstate(session.negative_generators[member].getstate())
                negatives = method.negative_indices(len(ids), generator).to(session.device)
                cosine = (F.normalize(reconstruction, dim=-1, eps=1e-8)*F.normalize(session.x[ids], dim=-1, eps=1e-8)).sum(-1)
                role_rows = {}
                for name, (role_ids, role_labels) in roles.items():
                    chosen = torch.isin(role_ids, ids)
                    full = dict(**accuracy_counts(logits[role_ids].argmax(-1), role_labels),
                                NLL=float(F.cross_entropy(logits[role_ids], role_labels).item()))
                    missing = None if not chosen.any() else dict(
                        **accuracy_counts(logits[role_ids[chosen]].argmax(-1), role_labels[chosen]),
                        NLL=float(F.cross_entropy(logits[role_ids[chosen]], role_labels[chosen]).item()))
                    role_rows[name] = dict(all_role_nodes=full, own_missing_feature_nodes=missing)
                routes.append(dict(member=member, quarter=member, anchors=len(ids),
                                   CORE32=float(method.core_loss(reconstruction, session.x[ids], negatives).item()),
                                   reconstruction_target_cosine_mean=float(cosine.mean().item()), roles=role_rows))
            bank = torch.stack(banks)
            probabilities = bank.softmax(-1).mean(0)
            masked_valid = classification(bank, probabilities, *roles['VALID'])
        return dict(available=True, routes=routes, masked_VALID=masked_valid,
                    masked_pool_is_diagnostic_only=True,
                    negative_draw='Clone of selected epoch next-negative-generator states; no training stream is advanced',
                    **structural), bank
    finally:
        for module, mode in modes: module.training = mode

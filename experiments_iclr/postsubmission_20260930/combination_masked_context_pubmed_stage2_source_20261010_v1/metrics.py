"""Selected factual and masked-context diagnostics; no import-time numerics."""
import math
import random


def accuracy_counts(predictions, labels):
    correct = int((predictions == labels).sum().item())
    return dict(correct=correct, count=len(labels), accuracy=correct/len(labels))


def classification(logits, probabilities, ids, labels):
    import torch
    members_count = logits.shape[0]
    if members_count not in (1,4) or logits.shape != (members_count, 19717, 3) or probabilities.shape != (19717, 3):
        raise ValueError('Complete one/four-member factual logits and probabilities required')
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite selected/evaluated logits')
    values = logits[:, ids]
    log_members = values.log_softmax(-1)
    log_pool = torch.logsumexp(log_members, dim=0)-math.log(members_count)
    prediction = probabilities[ids].argmax(-1)
    pooled = accuracy_counts(prediction, labels)
    selected = log_pool.gather(1, labels[:, None]).flatten()
    pooled['NLL'] = float(-selected.mean().item())
    members = []
    for member in range(members_count):
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
                               NLL=float(-log_members[m, chosen, c].mean().item())) for m in range(members_count)]
        classes.append(row)
    return dict(pooled=pooled, members=members,
                mean_member_accuracy=sum(m['accuracy'] for m in members)/members_count,
                worst_member_accuracy=min(m['accuracy'] for m in members),
                classes=classes, macro_accuracy=sum(c['accuracy'] for c in classes)/3)


def mask_diagnostics(session, method, roles, selected_ownership):
    import torch
    from torch.nn import functional as F
    structural = dict(eligible_partition_counts=[len(ids) for ids in session.anchor_ids],
                      partition_seed=190101, selected_training_ownership=selected_ownership,
                      uses_public_feature_targets_only=True, affects_selection=False)
    if not session.spec['masked']:
        return dict(available=False, reason='Exact own-only V3 condition has no masked views or decoder', **structural), None
    if sorted(selected_ownership) != list(range(4)):raise ValueError('Selected four-quarter ownership permutation required')
    modules=[module for body in session.bodies for module in body.modules()]
    if session.decoder is not None:modules+=list(session.decoder.modules())
    modes=[(module,module.training) for module in modules]
    try:
        for body in session.bodies:body.eval()
        if session.decoder is not None:session.decoder.eval()
        banks,routes=[],[]
        with torch.no_grad():
            for slot in range(4):
                member=slot if session.spec['members']==4 else 0
                quarter=selected_ownership[member] if session.spec['members']==4 else slot
                logits,hidden=session._forward(member,'masked',quarter)
                banks.append(logits)
                ids=session.anchor_ids[quarter]
                reconstruction_report=None
                if session.decoder is not None:
                    reconstruction=session.decoder(hidden[ids])
                    generator=random.Random()
                    generator.setstate(session.negative_generators[member if session.spec['members']==4 else quarter].getstate())
                    negatives=method.negative_indices(len(ids),generator).to(session.device)
                    cosine=(F.normalize(reconstruction,dim=-1,eps=1e-8)*F.normalize(session.x[ids],dim=-1,eps=1e-8)).sum(-1)
                    reconstruction_report=dict(CORE32=float(method.core_loss(reconstruction,session.x[ids],negatives).item()),
                        reconstruction_target_cosine_mean=float(cosine.mean().item()))
                role_rows={}
                for name,(role_ids,role_labels) in roles.items():
                    chosen=torch.isin(role_ids,ids)
                    full=dict(**accuracy_counts(logits[role_ids].argmax(-1),role_labels),NLL=float(F.cross_entropy(logits[role_ids],role_labels).item()))
                    missing=None if not chosen.any() else dict(**accuracy_counts(logits[role_ids[chosen]].argmax(-1),role_labels[chosen]),
                        NLL=float(F.cross_entropy(logits[role_ids[chosen]],role_labels[chosen]).item()))
                    role_rows[name]=dict(all_role_nodes=full,own_missing_feature_nodes=missing)
                routes.append(dict(member=member,quarter=quarter,anchors=len(ids),reconstruction=reconstruction_report,roles=role_rows))
            bank=torch.stack(banks)
            masked_valid=classification(bank,bank.softmax(-1).mean(0),*roles['VALID'])
        return dict(available=True,routes=routes,masked_VALID=masked_valid,masked_pool_is_diagnostic_only=True,
            negative_draw='Clone selected next-negative states; no training RNG/ownership is advanced',
            core_available=session.decoder is not None,**structural),bank
    finally:
        for module,mode in modes:module.training=mode

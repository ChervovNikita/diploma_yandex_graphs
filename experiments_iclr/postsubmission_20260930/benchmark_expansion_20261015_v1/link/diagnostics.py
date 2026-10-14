"""Whole-VALID member competence and pooled-ranking effects, beyond geometry."""


def link_bank_diagnostics(torch, logits, groups, evaluator, role='VALID'):
    if role != 'VALID':
        raise ValueError('Pilot bank analysis admits official VALID only')
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if logits.shape[0] != 4 or logits.shape[1] != len(groups):
        raise ValueError('Four aligned native VALID logit rows required')
    labels = torch.as_tensor(groups, dtype=torch.long, device=logits.device)
    if int((labels == 1).sum()) != 60084 or int((labels == 0).sum()) != 100000:
        raise ValueError('Complete official Collab VALID positive/negative pools required')
    evaluator.K = 50
    member_hits, member_flags = [], []
    for row in logits:
        positive, negative = row[labels == 1], row[labels == 0]
        member_hits.append(float(evaluator.eval({'y_pred_pos': positive, 'y_pred_neg': negative})['hits@50']))
        member_flags.append(positive > torch.topk(negative, 50).values[-1])
    pool = logits.mean(0)
    positive, negative = pool[labels == 1], pool[labels == 0]
    pool_hit = positive > torch.topk(negative, 50).values[-1]
    flags = torch.stack(member_flags)
    any_hit, all_hit = flags.any(0), flags.all(0)
    fraction = lambda value: float(value.float().mean())
    return {'role': role, 'whole_official_VALID': True, 'member_hits@50': member_hits,
        'mean_member_hits@50': sum(member_hits)/4,
        'pool_hits@50': float(evaluator.eval({'y_pred_pos': positive, 'y_pred_neg': negative})['hits@50']),
        'member_hit_disagreement_rate': fraction(any_hit & ~all_hit),
        'all_members_miss_rate': fraction(~any_hit),
        'pool_cancels_available_member_hit_rate': fraction(any_hit & ~pool_hit),
        'pool_rescues_common_member_miss_rate': fraction(~any_hit & pool_hit),
        'mean_positive_logit_variance': float(logits[:, labels == 1].var(0, unbiased=False).mean()),
        'mean_negative_logit_variance': float(logits[:, labels == 0].var(0, unbiased=False).mean()),
        'interpretation': 'each member and pool has its own native negative-pool top50 threshold; union is not a pooling upper bound',
        'geometry_is_accuracy_evidence': False, 'TEST_opened': False}

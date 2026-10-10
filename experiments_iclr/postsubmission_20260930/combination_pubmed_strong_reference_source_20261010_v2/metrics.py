"""Full selected native/class/repair readouts; stdlib at import time."""
import math
from source import fingerprint


def counts(prediction, labels):
    correct = int((prediction == labels).sum().item())
    return dict(correct=correct, count=len(labels), accuracy=correct/len(labels))


def predictions(logits, probabilities, ids):
    members = logits[:,ids].max(2)[1]
    # Bind the author's native classifier for a one-body predictor explicitly.
    pooled = members[0] if len(logits) == 1 else probabilities[ids].argmax(-1)
    return pooled,members


def signatures(logits, probabilities, ids, labels):
    pooled,members = predictions(logits,probabilities,ids)
    return dict(pooled_predictions=fingerprint(pooled.detach().cpu().numpy()),
                member_predictions=fingerprint(members.detach().cpu().numpy()),
                pooled_correct_mask=fingerprint((pooled == labels).detach().cpu().numpy()),
                member_correct_masks=fingerprint((members == labels[None,:]).detach().cpu().numpy()))


def classification(logits, probabilities, ids, labels):
    import torch
    if logits.dtype != torch.float32 or logits.shape not in ((1,19717,3),(4,19717,3)):
        raise ValueError('Complete native float32 bank required')
    if probabilities.shape != (19717,3) or not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise FloatingPointError('Complete finite factual predictions required')
    values = logits[:,ids]
    log_members = values.log_softmax(-1)
    log_pool = torch.logsumexp(log_members,dim=0)-math.log(len(logits))
    pooled_pred,member_pred = predictions(logits,probabilities,ids)
    chosen = log_pool.gather(1,labels[:,None]).flatten()
    pooled = dict(**counts(pooled_pred,labels),NLL=float(-chosen.mean().item()))
    members = [dict(**counts(member_pred[m],labels),
                    NLL=float(-log_members[m].gather(1,labels[:,None]).mean().item()))
               for m in range(len(logits))]
    classes=[]
    for c in range(3):
        selected=labels == c
        if not selected.any(): raise ValueError('Every frozen class required')
        classes.append(dict(class_id=c,**counts(pooled_pred[selected],labels[selected]),
                            NLL=float(-chosen[selected].mean().item()),
                            members=[dict(**counts(member_pred[m,selected],labels[selected]),
                                          NLL=float(-log_members[m,selected,c].mean().item()))
                                     for m in range(len(logits))]))
    return dict(pooled=pooled,members=members,classes=classes,
                mean_member_accuracy=sum(r['accuracy'] for r in members)/len(members),
                worst_member_accuracy=min(r['accuracy'] for r in members),
                macro_accuracy=sum(r['accuracy'] for r in classes)/3)


def repair_diagnostics(logits, probabilities, ids, labels):
    import torch
    pooled,members=predictions(logits,probabilities,ids)
    good=members == labels[None,:]; pooled_good=pooled == labels
    union=good.any(0); all_good=good.all(0); all_wrong=~union
    true_values=logits[:,ids].gather(2,labels[None,:,None].expand(len(logits),-1,1))
    rivals=(logits[:,ids] > true_values).all(0)
    rivals.scatter_(1,labels[:,None],False)
    count=lambda mask:int(mask.sum().item())
    return dict(count=len(labels),member_correct=[count(g) for g in good],
                oracle_union_correct=count(union),all_members_correct=count(all_good),
                all_members_wrong=count(all_wrong),pool_correct=count(pooled_good),
                all_wrong_repaired_by_pool=count(all_wrong & pooled_good),
                pool_wrong_despite_some_member_correct=count(union & ~pooled_good),
                member0_errors_repaired_by_pool=count(~good[0] & pooled_good),
                member0_correct_lost_by_pool=count(good[0] & ~pooled_good),
                strict_common_false_rival_nodes=count(rivals.any(1)),
                strict_common_false_rival_is_separate_from_all_wrong=True,
                pair_disagreement=[dict(a=a,b=b,count=count(members[a] != members[b]))
                                   for a in range(len(logits)) for b in range(a)],
                pool_minus_mean_member_accuracy_pp=100*(count(pooled_good)/len(labels)-sum(count(g) for g in good)/(len(logits)*len(labels))),
                affects_selection=False)


def verify_counts(actual, expected):
    """Prediction signatures/counts are decisive; floats are not compared here."""
    if actual['pooled']['correct'] != expected['pooled']['correct'] or actual['pooled']['count'] != expected['pooled']['count']:
        raise ValueError('Selected pooled exact counts changed')
    if [(r['correct'],r['count']) for r in actual['members']] != [(r['correct'],r['count']) for r in expected['members']]:
        raise ValueError('Selected native member exact counts changed')
    for a,e in zip(actual['classes'],expected['classes']):
        if (a['class_id'],a['correct'],a['count']) != (e['class_id'],e['correct'],e['count']):
            raise ValueError('Selected exact class counts changed')


def verify_floats(actual, expected):
    """FP32 repeat tolerance is restore verification only, never an outcome gate."""
    if isinstance(expected,dict):
        if set(actual) != set(expected): raise ValueError('Selected metric fields changed')
        for k in expected: verify_floats(actual[k],expected[k])
    elif isinstance(expected,list):
        if len(actual) != len(expected): raise ValueError('Selected metric shape changed')
        for a,e in zip(actual,expected): verify_floats(a,e)
    elif isinstance(expected,float):
        if not math.isclose(actual,expected,abs_tol=2e-6,rel_tol=2e-6):
            raise ValueError('Selected FP32 metric repeat outside restore tolerance')
    elif actual != expected or type(actual) != type(expected):
        raise ValueError('Selected count/identity changed')

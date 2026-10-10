"""Stopped scoring of a prebound baseline stratum; never supplies learning credit."""
from .caps import CLOSED
from .native import require


def compare_fixed_common_rival(torch, baseline, candidate, caps=CLOSED):
    caps.require('source_bound', 'data', 'runtime', 'scientific')
    require(baseline['ids'] == candidate['ids'] and torch.equal(baseline['truth'], candidate['truth']),
            'One complete fixed VALID role and unchanged truth; no subset shopping')
    y, rival = baseline['truth'], baseline['fixed_rival']
    z = candidate['member_logits']
    selected = baseline['strict_common_rival']
    by = z.gather(2, y[None, :, None].expand(4, -1, 1)).squeeze(2)
    br = z.gather(2, rival[None, :, None].expand(4, -1, 1)).squeeze(2)
    clears = by > br
    member_correct = z.argmax(dim=-1) == y[None]
    old_wrong = baseline['prediction'] != y
    new_wrong = candidate['prediction'] != y
    return dict(fixed_common_rival_nodes=int(selected.sum().item()),
        fixed_common_rival_any_member_clearance=int((selected & clears.any(dim=0)).sum().item()),
        fixed_common_rival_any_member_correct=int((selected & member_correct.any(dim=0)).sum().item()),
        fixed_common_rival_pool_repairs=int((selected & ~new_wrong).sum().item()),
        pool_repairs=int((old_wrong & ~new_wrong).sum().item()),
        pool_new_errors=int((~old_wrong & new_wrong).sum().item()),
        net_pool_repairs=int((old_wrong & ~new_wrong).sum().item()-(~old_wrong & new_wrong).sum().item()),
        baseline_rival_and_population_fixed_before_candidate_results=True,
        no_geometry_or_entropy_surrogate_admission=True)

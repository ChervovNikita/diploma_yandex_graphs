"""Retrospective fixed raw-mean/Borda comparison on twelve saved VALID members.

No data loader, checkpoint, model, fitted aggregation, training or TEST access.
The complete prior cohort and all twelve score identities precede payload reads.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import itertools
import json
import math
import socket
import statistics
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
HERE = PHASE/'citeseer_saved_independent_bank_pooling_screen_20261005_v1'
COHORT = PHASE/'citeseer_endpoint_frame_paired_development_20261005_v1'
PRIOR = PHASE/'citeseer_frame_complete_analysis_20261005_v2'
PLAN_SHA = 'fd9fec451a81d0512cd8431ba2a990c580592ff6a6e8e12d784386a99c5e9daa'
COHORT_SHA = '120f57e53377494a055a994405581fd9fb1fcfe0c5636bdf4446dd8735bd94f8'
PRIOR_SHA = 'd675f6ccd3d4d3eef1da88c445c82bc28c4ba2b664710b057e6630a491127ec6'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while part := stream.read(1048576):
            h.update(part)
    return h.hexdigest()

def checked(path, expected=None, maximum=1048576):
    assert path.is_relative_to(PHASE) and path.resolve(strict=True) == path
    assert path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum
    assert expected is None or sha(path) == expected, str(path)
    return json.loads(path.read_text())

def summary(values):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    mean = statistics.mean(values)
    half = 4.302652729696142*statistics.stdev(values)/math.sqrt(3)
    return dict(block_values=values, mean=mean, sample_SD=statistics.stdev(values),
                descriptive_t95_training_seed_interval=[mean-half, mean+half])

def main():
    started = time.monotonic()
    assert Path.cwd().resolve() == REPO and Path(__file__).resolve().parent == HERE
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                                  text=True, timeout=20).split() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert not (HERE/'RESULTS.json').exists() and not (HERE/'START.json').exists()
    protocol = checked(HERE/'PROTOCOL.json')
    assert sha(Path(__file__)) == protocol['analysis_sha256']
    assert protocol['comparison'] == ['mean_raw_logits','equal_candidate_midrank_Borda']
    assert protocol['TEST_access'] is False and protocol['fit_count'] == 0
    plan = checked(COHORT/'PLAN.json', PLAN_SHA)
    cohort = checked(COHORT/'COHORT_FREEZE.json', COHORT_SHA)
    prior = checked(PRIOR/'RESULTS.json', PRIOR_SHA)
    assert cohort['complete'] is True and cohort['TEST_access'] is False
    assert len(cohort['completed_physical_fits']) == 36
    completed = {r['id']:r for r in cohort['completed_physical_fits']}
    assert set(completed) == {r['id'] for r in plan['physical_fits']}
    audits = {r['id']:r for r in prior['fit_audits']}
    identities = None
    banks, score_files = [], []
    for b in range(3):
        ids = [f'b{b}_native_single_seed{b}'] + [f'b{b}_independent_member_m{m}_seed{b+5*m}' for m in range(1,4)]
        banks.append(ids)
        for member in ids:
            folder = COHORT/'fits'/member
            freeze = checked(folder/'FREEZE.json', completed[member]['freeze_sha256'])
            assert freeze['TEST_access'] is False and freeze['member_count'] == 1
            assert freeze['cohort_plan_sha256'] == PLAN_SHA
            assert freeze['VALID_logits_sha256'] == audits[member]['VALID_logits_sha256']
            assert freeze['checkpoint_sha256'] == audits[member]['checkpoint_sha256']
            assert freeze['selected_epoch'] == audits[member]['selected_epoch']
            if identities is None:
                identities = freeze['input_identities']
            assert freeze['input_identities'] == identities
            scores = folder/'selected_VALID_logits.pt'
            assert scores.resolve(strict=True) == scores and scores.is_file() and not scores.is_symlink()
            assert scores.stat().st_size < 4194304 and sha(scores) == freeze['VALID_logits_sha256']
            score_files.append(dict(member=member, path=scores, freeze=freeze, sha256=sha(scores)))
    assert len(score_files) == 12 and len({r['member'] for r in score_files}) == 12
    # Payload access follows full-cohort metadata and complete twelve-member binding.
    (HERE/'START.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), analysis_sha256=sha(Path(__file__)),
        twelve_saved_members_authenticated=True, TEST_access=False, fit_count=0),indent=2)+'\n')
    import torch
    torch.set_num_threads(2)

    def ranks(scores):
        values, order = torch.sort(scores, dim=-1, descending=True, stable=True)
        positions = torch.arange(scores.shape[-1], dtype=torch.int64).expand_as(order)
        first = torch.ones_like(order, dtype=torch.bool)
        last = torch.ones_like(order, dtype=torch.bool)
        first[...,1:] = values[...,1:] != values[...,:-1]
        last[...,:-1] = values[...,:-1] != values[...,1:]
        starts = torch.where(first, positions, 0).cummax(-1).values
        ends = torch.where(last, positions, scores.shape[-1]-1).flip(-1).cummin(-1).values.flip(-1)
        mid = (starts.to(torch.float64)+ends.to(torch.float64))/2 + 1
        out = torch.empty_like(mid)
        return out.scatter_(-1, order, mid)

    fixture = torch.tensor([[2.,2.,0.,-1.], [0.,3.,1.,1.]], dtype=torch.float32)
    assert torch.equal(ranks(fixture), torch.tensor([[1.5,1.5,3.,4.], [4.,1.,2.5,2.5]],dtype=torch.float64))
    assert torch.equal(ranks(fixture), ranks(3*fixture+7))

    predictions = {}
    for record in score_files:
        saved = torch.load(record['path'], map_location='cpu', weights_only=True)
        freeze = record['freeze']
        assert saved['input_identities'] == identities and saved['selected_epoch'] == freeze['selected_epoch']
        assert saved['checkpoint_sha256'] == freeze['checkpoint_sha256']
        pos, neg = saved['pos'], saved['neg']
        assert pos.shape == (227,) and neg.shape == (227,500) and pos.dtype == neg.dtype == torch.float32
        assert torch.isfinite(pos).all() and torch.isfinite(neg).all()
        predictions[record['member']] = torch.cat([pos[:,None],neg],dim=1)
        assert sha(record['path']) == record['sha256']

    def evaluate(scores):
        pos, neg = scores[:,0], scores[:,1:]
        rank = 1 + .5*((neg>pos[:,None]).sum(1)+(neg>=pos[:,None]).sum(1))
        return (1/rank.float()).mean().item(), rank

    blocks, raw_values, borda_values = [], [], []
    for b, ids in enumerate(banks):
        scores = torch.stack([predictions[member] for member in ids])
        pooled = scores.mean(0)
        borda = -ranks(scores).mean(0)
        raw_mrr, raw_rank = evaluate(pooled)
        borda_mrr, borda_rank = evaluate(borda)
        assert abs(raw_mrr-prior['family_summaries']['ordinary_independent4']['values'][b]) < 1e-7
        # This oracle chooses each query/candidate weight using labels. It is ONLY a ceiling.
        # Every candidate-specific simplex pool lies within its member-score interval.
        positive_max = scores[:,:,0].max(0).values
        negative_min = scores[:,:,1:].min(0).values
        best_rank = 1 + .5*((negative_min>positive_max[:,None]).sum(1)+(negative_min>=positive_max[:,None]).sum(1))
        bound_mrr = (1/best_rank.float()).mean().item()
        assert bool((best_rank <= raw_rank).all()) and bound_mrr >= raw_mrr-1e-7
        raw_wrong = pooled[:,1:] > pooled[:,0,None]
        borda_correct = borda[:,0,None] > borda[:,1:]
        majority_correct = (scores[:,:,0,None] > scores[:,:,1:]).sum(0) >= 3
        raw_correct = pooled[:,0,None] > pooled[:,1:]
        borda_wrong = borda[:,1:] > borda[:,0,None]
        raw_values.append(raw_mrr); borda_values.append(borda_mrr)
        blocks.append(dict(seed_block=b, member_ids=ids, raw_MRR=raw_mrr, Borda_MRR=borda_mrr,
            Borda_minus_raw=borda_mrr-raw_mrr, better_queries=int((borda_rank<raw_rank).sum()),
            worse_queries=int((borda_rank>raw_rank).sum()), equal_queries=int((borda_rank==raw_rank).sum()),
            raw_wrong_Borda_correct_candidate_pairs=int((raw_wrong & borda_correct).sum()),
            majority_correct_scale_dominated_pairs=int((raw_wrong & majority_correct).sum()),
            scale_dominated_pairs_corrected_by_Borda=int((raw_wrong & majority_correct & borda_correct).sum()),
            raw_correct_Borda_wrong_candidate_pairs=int((raw_correct & borda_wrong).sum()),
            label_oracle_candidatewise_simplex_MRR_upper_bound=bound_mrr,
            unavoidable_strict_negative_ahead_pairs=int((negative_min>positive_max[:,None]).sum()),
            oracle_not_a_predictor_or_achievable_generalization_claim=True))
    gaps = [a-b for a,b in zip(borda_values,raw_values)]
    p = sum(abs(statistics.mean([s*d for s,d in zip(signs,gaps)])) >= abs(statistics.mean(gaps))-1e-15
            for signs in itertools.product((-1,1),repeat=3))/8
    result = dict(status='complete_retrospective_saved_bank_pooling_screen', UTC=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(HERE/'PROTOCOL.json'), analysis_sha256=sha(Path(__file__)),
        cohort_freeze_sha256=COHORT_SHA, prior_results_sha256=PRIOR_SHA,
        score_identities=[dict(member=r['member'],sha256=r['sha256']) for r in score_files],
        blocks=blocks, mean_raw_logits=summary(raw_values), equal_midrank_Borda=summary(borda_values),
        paired_difference=summary(gaps), exact_two_sided_seed_signflip_p=p,
        seed_test_assumption='Sign symmetry/independent seed blocks; n=3 cannot establish it and minimum two-sided p is0.25.',
        oracle_scope='Candidate-dependent scalar convex logit pools; label-informed extreme choice, not a learned method.',
        shared_bank_evaluated=False, shared_bank_reason='Prior runner retained pooled scores only; shared member replay not performed.',
        split_scope='227 validation-selected positive links with their complete500fixed negatives; all3plannedseedblocks.',
        retrospective=True, independent_confirmation=False, checkpoint_reselection=False,
        fitted_weights=False, TEST_access=False, model_forwards=0, fit_count=0, optimizer_updates=0,
        unchanged_prior_method_gate=True, original_manuscript_scores_unchanged=True,
        inclusive_seconds=time.monotonic()-started)
    with (HERE/'RESULTS.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],mean_raw=statistics.mean(raw_values),mean_Borda=statistics.mean(borda_values),
        difference=statistics.mean(gaps),shared_bank_evaluated=False,TEST_access=False)))

if __name__ == '__main__':
    main()

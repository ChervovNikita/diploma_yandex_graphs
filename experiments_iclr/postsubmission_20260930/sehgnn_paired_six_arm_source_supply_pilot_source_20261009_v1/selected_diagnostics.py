"""Single full-VALID source-utility implementation; numerical runtime supplied."""
import math

FAMILIES = ('actor', 'director', 'keyword')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def f1_scores(torch, probability, y):
    """Native >.5 decisions, micro/macro multilabel F1, zero denominator=>0."""
    predicted, truth = probability > .5, y.bool()
    tp = (predicted & truth).sum(dim=0).double()
    fp = (predicted & ~truth).sum(dim=0).double()
    fn = (~predicted & truth).sum(dim=0).double()
    denominator = 2 * tp + fp + fn
    per_label = torch.where(denominator > 0, 2 * tp / denominator, torch.zeros_like(denominator))
    total = denominator.sum()
    return dict(micro_F1=float(2 * tp.sum() / total) if float(total) else 0.,
                macro_F1=float(per_label.mean()), per_label_F1=per_label.tolist())


def repair_harm(before, after, y):
    correct0, correct1 = (before > .5) == y.bool(), (after > .5) == y.bool()
    repair, harm = (~correct0) & correct1, correct0 & (~correct1)
    return dict(repaired_per_label=repair.sum(dim=0).tolist(), harmed_per_label=harm.sum(dim=0).tolist(),
                repaired_total=int(repair.sum()), harmed_total=int(harm.sum()),
                repair_mask=repair, harm_mask=harm)


def compare_selected(torch, reference, candidate, costs, reference_name):
    """Deployed full-input pool changes on identical paired VALID events."""
    with costs.measure('paired_candidate_vs_' + reference_name + '_complete_VALID_pool_changes'):
        require(reference['valid_ids'] == candidate['valid_ids'] and torch.equal(reference['targets'], candidate['targets']),
                'Identical paired all-VALID rows and five-label truth')
        require(reference['paired_identity'] == candidate['paired_identity'], 'Same prospective study/base/role/full-input binding')
        events = repair_harm(reference['full_pool_probabilities'], candidate['full_pool_probabilities'], candidate['targets'])
        return dict(reference=reference_name, candidate='assigned_source_supply',
            full_pool_BCE_difference_candidate_minus_reference=candidate['full_pool_BCE'] - reference['full_pool_BCE'],
            reference_member_BCE=reference['full_member_BCE'], candidate_member_BCE=candidate['full_member_BCE'],
            reference_member_F1=reference['full_member_F1'], candidate_member_F1=candidate['full_member_F1'],
            reference_pool_F1=reference['full_pool_F1'], candidate_pool_F1=candidate['full_pool_F1'],
            deployed_full_input_pool_events=events,
            reference_common_wrong_repaired_total=int((reference['common_wrong_events'] & events['repair_mask']).sum()),
            reference_any_correct_member_pool_repaired_total=int((reference['any_correct_member_events'] & events['repair_mask']).sum()),
            VALID_used_for_checkpoint_selection=True, no_independent_confirmation_claim=True,
            no_best_arm_or_subset_selection=True, success_or_competence_admission_assessed=False)


def collect_selected_logits(rt, engine, bank_module, models, member_rng, full_ctx, views, costs):
    """Current restored members, fresh FP32 eval, no TRAIN replay tokens or VJPs."""
    torch = rt['torch']
    ids = full_ctx.valid_index
    caller = engine.capture_rng(rt['numpy'], torch)
    result = {}
    try:
        with costs.measure('selected_full_4_by_3_current_VALID_source_assay', gpu=True), torch.no_grad():
            for source, context in [(None, full_ctx), *[(family, views[family]) for family in FAMILIES]]:
                values = []
                for member, model in enumerate(models):
                    parameters = tuple((id(p), int(p._version)) for p in model.parameters())
                    buffers = [(name, value, value.detach().cpu().clone()) for name, value in model.named_buffers(remove_duplicate=False)]
                    caches = tuple((namespace, id(mapping), tuple((key, id(value), int(value._version)) for key, value in mapping.items()))
                                   for namespace, mapping in [('feature', context.feats), ('label', context.label_feats)])
                    features = {key: value[ids].to(rt['device']) for key, value in context.feats.items()}
                    labels = {key: value[ids].to(rt['device']) for key, value in context.label_feats.items()}
                    with bank_module.scratch_native_state(rt, engine, model, 'eval', member_rng[member]), torch.cuda.amp.autocast(enabled=False):
                        output = model(ids.to(rt['device']), features, labels, None).float()
                    require(output.shape == (len(ids), 5) and torch.isfinite(output).all().item(), 'Complete finite current VALID logits')
                    require(parameters == tuple((id(p), int(p._version)) for p in model.parameters()), 'Diagnostic never updates learned parameters')
                    require([(name, id(value)) for name, value in model.named_buffers(remove_duplicate=False)] == [(name, id(value)) for name, value, _ in buffers]
                            and all(torch.equal(value.detach().cpu(), old) for _, value, old in buffers), 'Diagnostic preserves actual member buffers')
                    require(caches == tuple((namespace, id(mapping), tuple((key, id(value), int(value._version)) for key, value in mapping.items()))
                                            for namespace, mapping in [('feature', context.feats), ('label', context.label_feats)]), 'Canonical source caches remain immutable')
                    values.append(output.detach().cpu().clone())
                    del features, labels, output
                result['full' if source is None else source] = tuple(values)
        return result
    finally:
        engine.restore_rng(rt['numpy'], torch, caller)
        require(engine.exact(torch, engine.capture_rng(rt['numpy'], torch), caller), 'Diagnostic preserves caller streams')


def analyze(torch, logits, targets, valid_ids, costs):
    """Exact U/D/event/repair-harm arithmetic; all rows and negative events retained."""
    with costs.measure('selected_complete_VALID_U_D_event_and_repair_harm_arithmetic'):
        y = targets.detach().cpu().float()
        require(y.shape == (len(valid_ids), 5) and ((y == 0) | (y == 1)).all().item(), 'Complete known VALID Bernoulli truth')
        require(set(logits) == {'full', *FAMILIES} and all(len(values) == 4 for values in logits.values()), 'All4 factual and12 current ablated outputs')
        def log_events(values):
            return [torch.where(y.bool(), torch.nn.functional.logsigmoid(value), torch.nn.functional.logsigmoid(-value)) for value in values]
        def mixed(values):
            logq = torch.logsumexp(torch.stack(values), dim=0) - math.log(4)
            return logq, -logq.mean()
        def probabilities(values):
            return torch.stack([torch.sigmoid(value) for value in values]).mean(dim=0)
        factual = log_events(logits['full'])
        full_logq, full_risk = mixed(factual)
        full_probability = probabilities(logits['full'])
        U, D, detail = [[0.] * 3 for _ in range(4)], [[0.] * 3 for _ in range(4)], {}
        for column, family in enumerate(FAMILIES):
            absent = log_events(logits[family])
            absent_logq, absent_risk = mixed(absent)
            absent_probability = probabilities(logits[family])
            detail[family] = dict(all_absent_BCE=float(absent_risk), member_observed_event_probabilities=torch.stack(absent).exp(), members=[])
            for member in range(4):
                supply_events, remove_events = list(absent), list(factual)
                supply_events[member], remove_events[member] = factual[member], absent[member]
                supply_logq, supply_risk = mixed(supply_events)
                remove_logq, remove_risk = mixed(remove_events)
                U[member][column], D[member][column] = float(absent_risk - supply_risk), float(remove_risk - full_risk)
                supply_logits, remove_logits = list(logits[family]), list(logits['full'])
                supply_logits[member], remove_logits[member] = logits['full'][member], logits[family][member]
                detail[family]['members'].append(dict(member=member, only_member_full_BCE=float(supply_risk), only_member_absent_BCE=float(remove_risk),
                    U_per_query_label=supply_logq - absent_logq, D_per_query_label=full_logq - remove_logq,
                    absent_peer_context=repair_harm(absent_probability, probabilities(supply_logits), y),
                    full_peer_context=repair_harm(probabilities(remove_logits), full_probability, y)))
        def specificity(matrix):
            return sum(matrix[m][m] for m in range(3))/3 - sum(matrix[m][a] for m in range(3) for a in range(3) if a != m)/6
        require(all(math.isfinite(v) for matrix in (U, D) for row in matrix for v in row), 'Finite complete source utility matrices')
        full_correct = (full_probability > .5) == y.bool()
        member_correct = torch.stack([(torch.sigmoid(value) > .5) == y.bool() for value in logits['full']])
        return dict(schema='selected-current-full-VALID-4x3-source-utility-v1', valid_ids=list(valid_ids), targets=y,
            family_order=list(FAMILIES), member_order=list(range(4)), U=U, D=D,
            assigned_specificity=dict(U=specificity(U), D=specificity(D), assignment=['actor', 'director', 'keyword', None]),
            full_pool_BCE=float(full_risk), full_pool_probabilities=full_probability,
            full_member_BCE=[float(-value.mean()) for value in factual],
            full_member_F1=[f1_scores(torch, torch.sigmoid(value), y) for value in logits['full']],
            full_pool_F1=f1_scores(torch, full_probability, y),
            full_member_observed_event_probabilities=torch.stack(factual).exp(),
            member_correct=member_correct, full_pool_correct=full_correct,
            common_wrong_events=(~member_correct).all(dim=0), any_correct_member_events=member_correct.any(dim=0),
            family_details=detail, no_selection_or_fitting_from_diagnostics=True,
            current_eval_peers=True, training_reference_tokens_or_frozen_training_peers_used=False,
            geometry_or_sharing_advantage_established=False)

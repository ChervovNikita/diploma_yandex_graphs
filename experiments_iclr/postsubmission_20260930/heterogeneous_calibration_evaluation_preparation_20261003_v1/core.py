"""Common DBLP/ACM saved-logit calibration and metrics; no file or graph reader."""
import hashlib
import json
import math

SEEDS = [131, 137, 139, 149, 151]
LOWER_SCALE, UPPER_SCALE, ITERATIONS = 0.05, 20.0, 80
CALIBRATION_SCHEMA = 'source_validation_inverse_temperature_golden80_v1'
METRIC_PATHS = ('raw.NLL_FP32', 'raw.NLL_FP64_calibration_reference',
                'calibrated.NLL_FP64', 'raw.micro_F1', 'raw.macro_F1',
                'calibrated.micro_F1', 'calibrated.macro_F1',
                'raw.Brier_FP64', 'raw.ECE15_FP64',
                'calibrated.Brier_FP64', 'calibrated.ECE15_FP64')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def class_schema(classes):
    require(type(classes) in (list, tuple) and len(classes) >= 2
            and all(type(c) is int for c in classes)
            and list(classes) == list(range(len(classes))), 'Explicit contiguous fixed class schema required')
    return list(classes)


def tensor_ids(torch, ids, name):
    require(isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.ndim == 1
            and ids.numel() > 0, name + ' must be a nonempty explicit int64 vector')
    values = ids.detach().cpu().tolist()
    require(min(values) >= 0 and len(values) == len(set(values)), name + ' must contain distinct nonnegative IDs')
    return values


def aggregate_logits(raw_logits, classes):
    """FP32 mean raw member logits, or unchanged FP32 native single logits."""
    import torch
    classes = class_schema(classes)
    require(isinstance(raw_logits, torch.Tensor) and raw_logits.dtype == torch.float32
            and raw_logits.ndim in (2, 3), 'FP32 native[N,C] or members[M,N,C] logits required')
    require(all(size > 0 for size in raw_logits.shape) and raw_logits.shape[-1] == len(classes),
            'Nonempty logits with the explicit fixed class count required')
    require(bool(torch.isfinite(raw_logits).all()), 'Nonfinite raw logits')
    # Conversion to FP64 is deliberately after the same served FP32 mean(0).
    served = raw_logits if raw_logits.ndim == 2 else raw_logits.mean(0)
    require(served.dtype == torch.float32 and bool(torch.isfinite(served).all()), 'Nonfinite served FP32 logits')
    return served


def selected_logits(raw_logits, row_ids, evaluation_ids, labels, classes):
    """Select saved rows by explicit ID mapping; compact labels follow requested IDs."""
    import torch
    classes = class_schema(classes)
    served = aggregate_logits(raw_logits, classes)
    rows = tensor_ids(torch, row_ids, 'Saved-logit row IDs')
    requested = tensor_ids(torch, evaluation_ids, 'Requested evaluation IDs')
    require(len(rows) == len(served), 'Saved-logit rows and explicit row IDs differ')
    lookup = {node: index for index, node in enumerate(rows)}
    require(all(node in lookup for node in requested), 'Requested ID missing from saved logits')
    require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long and labels.ndim == 1
            and len(labels) == len(requested), 'Compact int64 labels aligned to requested IDs required')
    require(bool(((labels >= 0) & (labels < len(classes))).all()), 'Label outside the fixed class schema')
    index = torch.tensor([lookup[node] for node in requested], dtype=torch.long, device=served.device)
    return served[index], labels.to(served.device), requested, classes


def fit_inverse_temperature(raw_logits, row_ids, validation_ids, validation_labels, classes, *, label_scope):
    """Fit source VALIDATION only, with80 fixed iterations and first-minimum ties.

    Candidate order: identity1, global lower bound, global upper bound, final
    left interior, final right interior. Equal interior objectives keep the left
    bracket branch. Any calibration failure retains its reason and falls back1.
    """
    result = dict(schema=CALIBRATION_SCHEMA, status='fallback', inverse_temperature=1.0, temperature=1.0,
                  fit_label_scope=label_scope, iterations=ITERATIONS, bounds=[LOWER_SCALE, UPPER_SCALE],
                  iterations_completed=0,
                  candidate_order=['identity', 'lower_bound', 'upper_bound', 'final_left_interior', 'final_right_interior'],
                  tie_rule='first minimum in candidate order; interior equality keeps left bracket',
                  failure_reason=None, validation_node_count=None, class_schema=None,
                  raw_calibration_reference_NLL_FP64=None, selected_calibration_NLL_FP64=None,
                  candidates=[], selected_candidate_index=None)
    try:
        import torch
        require(label_scope == 'SOURCE_VALIDATION_ONLY', 'Calibration may fit source VALIDATION only')
        logits, labels, ids, fixed_classes = selected_logits(raw_logits, row_ids, validation_ids,
                                                            validation_labels, classes)
        logits = logits.detach().to(torch.float64)
        result.update(validation_node_count=len(ids), class_schema=fixed_classes,
                      validation_ID_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())

        def objective(scale):
            require(math.isfinite(scale) and LOWER_SCALE <= scale <= UPPER_SCALE, 'Invalid calibration scale')
            with torch.no_grad():
                value = float(torch.nn.functional.cross_entropy(logits * scale, labels))
            require(math.isfinite(value), 'Nonfinite FP64 calibration objective')
            return value

        baseline = objective(1.0)
        result['raw_calibration_reference_NLL_FP64'] = baseline
        lower, upper = LOWER_SCALE, UPPER_SCALE
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        left = upper - ratio * (upper - lower)
        right = lower + ratio * (upper - lower)
        loss_left, loss_right = objective(left), objective(right)
        for _ in range(ITERATIONS):
            if loss_left <= loss_right:
                upper, right, loss_right = right, left, loss_left
                left = upper - ratio * (upper - lower)
                loss_left = objective(left)
            else:
                lower, left, loss_left = left, right, loss_right
                right = lower + ratio * (upper - lower)
                loss_right = objective(right)
            result['iterations_completed'] += 1
        scales = [1.0, LOWER_SCALE, UPPER_SCALE, left, right]
        candidates = [dict(name=name, inverse_temperature=scale, NLL_FP64=objective(scale))
                      for name, scale in zip(result['candidate_order'], scales)]
        selected = 0
        for index in range(1, len(candidates)):
            if candidates[index]['NLL_FP64'] < candidates[selected]['NLL_FP64']:
                selected = index
        scale = candidates[selected]['inverse_temperature']
        result.update(status='fitted', inverse_temperature=scale, temperature=1.0 / scale,
                      candidates=candidates, selected_candidate_index=selected,
                      selected_calibration_NLL_FP64=candidates[selected]['NLL_FP64'])
    except Exception as error:
        result.update(status='fallback', inverse_temperature=1.0, temperature=1.0,
                      failure_reason=type(error).__name__ + ': ' + str(error),
                      selected_calibration_NLL_FP64=result['raw_calibration_reference_NLL_FP64'])
    return result


def probability_metrics(torch, logits_fp64, labels, classes):
    """FP64 probabilities; Brier=sum-class squared error, ECE15 equal-width bins."""
    probabilities = torch.softmax(logits_fp64, dim=-1)
    # Preserve the configured predictor's logit argmax. Rounded softmax values
    # can tie even when two saved FP32 logits are distinct.
    predictions = logits_fp64.argmax(-1)
    correct = predictions == labels
    onehot = torch.nn.functional.one_hot(labels, num_classes=len(classes)).to(torch.float64)
    brier = float(((probabilities - onehot).square().sum(-1)).mean())
    confidence = probabilities.max(-1).values
    bins = torch.floor(confidence * 15).to(torch.long).clamp(0, 14)
    ece = 0.0
    for index in range(15):
        mask = bins == index
        count = int(mask.sum())
        if count:
            gap = abs(float(confidence[mask].mean()) - float(correct[mask].to(torch.float64).mean()))
            ece += count / len(labels) * gap
    f1 = []
    for cls in classes:
        tp = int(((predictions == cls) & (labels == cls)).sum())
        fp = int(((predictions == cls) & (labels != cls)).sum())
        fn = int(((predictions != cls) & (labels == cls)).sum())
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return dict(Brier_FP64=brier, ECE15_FP64=ece, micro_F1=float(correct.to(torch.float64).mean()),
                macro_F1=sum(f1) / len(classes))


def evaluate_logits(raw_logits, row_ids, evaluation_ids, labels, classes, *, calibration, evaluation_scope):
    """Raw source-arithmetic NLL plus separate FP64 reference/calibrated metrics.

    Labels are supplied by the root after any required release. This function
    neither reads files nor grants label access, performs inference, or retrains.
    """
    import torch
    require(evaluation_scope in ('source_validation', 'heldout'), 'Explicit evaluation scope required')
    require(calibration['schema'] == CALIBRATION_SCHEMA and calibration['status'] in ('fitted', 'fallback')
            and calibration['fit_label_scope'] == 'SOURCE_VALIDATION_ONLY'
            and calibration['iterations'] == ITERATIONS
            and calibration['bounds'] == [LOWER_SCALE, UPPER_SCALE], 'Frozen source-validation calibration record required')
    fixed_classes = class_schema(classes)
    require(calibration['class_schema'] in (None, fixed_classes), 'Calibration and evaluation class schemas differ')
    scale = calibration['inverse_temperature']
    require(type(scale) in (int, float) and math.isfinite(scale) and LOWER_SCALE <= scale <= UPPER_SCALE,
            'Frozen positive calibration scale required')
    require(calibration['status'] != 'fallback' or (scale == 1.0 and bool(calibration['failure_reason'])),
            'Calibration failure must retain reason and identity fallback')
    logits, compact_labels, ids, fixed_classes = selected_logits(raw_logits, row_ids, evaluation_ids, labels, classes)
    with torch.no_grad():
        # This is the original configured-predictor FP32 CE, without recasting.
        raw_nll = float(torch.nn.functional.cross_entropy(logits, compact_labels))
        fp64 = logits.detach().to(torch.float64)
        reference_nll = float(torch.nn.functional.cross_entropy(fp64, compact_labels))
        calibrated_nll = float(torch.nn.functional.cross_entropy(fp64 * scale, compact_labels))
        require(all(math.isfinite(x) for x in (raw_nll, reference_nll, calibrated_nll)), 'Nonfinite evaluation NLL')
        raw = dict(NLL_FP32=raw_nll, NLL_FP64_calibration_reference=reference_nll,
                   **probability_metrics(torch, fp64, compact_labels, fixed_classes))
        calibrated = dict(NLL_FP64=calibrated_nll,
                          **probability_metrics(torch, fp64 * scale, compact_labels, fixed_classes))
    return dict(schema='fixed_class_saved_logit_metrics_v1', evaluation_scope=evaluation_scope, nodes=len(ids),
                class_schema=fixed_classes, aggregation='FP32 mean raw member logits; native single logits unchanged',
                raw=raw, calibrated=calibrated, calibration=calibration,
                ECE_policy='15 equal-width confidence bins; left-closed/right-open except final bin closed',
                Brier_policy='sum over classes then mean over nodes', probability_arithmetic='FP64 after FP32 aggregation')


def descriptive_five(values):
    require(len(values) == 5 and all(math.isfinite(x) for x in values), 'Five finite paired values required')
    mean = sum(values) / 5
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / 4)
    se = sd / math.sqrt(5)
    return dict(vector=list(values), mean=mean, paired_seed_SD=sd, paired_seed_SE=se,
                illustrative_t95_interval=[mean - 2.7764451051977987 * se, mean + 2.7764451051977987 * se],
                leave_one_block_out_means=[sum(x for j, x in enumerate(values) if j != i) / 4 for i in range(5)],
                negative_values=sum(x < 0 for x in values), positive_values=sum(x > 0 for x in values), ties=sum(x == 0 for x in values))


def paired_summary(rows, family_binding, *, reference_arm):
    """Score only the complete, same-binding family/allfive; no partial summaries.

    Root verifies actual immutable storage/source/predictor custody externally.
    Each supplied row must be complete and repeat the exact bound family record.
    """
    require(family_binding['immutable'] is True and family_binding['dataset'] in ('HGB-DBLP', 'HGB-ACM')
            and family_binding['evaluation_scope'] in ('source_validation', 'heldout')
            and family_binding['seeds'] == SEEDS, 'One immutable graph/scope and all five frozen seeds required')
    arms = family_binding['arms']
    require(type(arms) is list and arms and all(type(arm) is str and arm for arm in arms)
            and len(arms) == len(set(arms)) and reference_arm in arms, 'Full distinct frozen family and reference arm required')
    expected_classes = list(range(4 if family_binding['dataset'] == 'HGB-DBLP' else 3))
    require(family_binding['class_schema'] == expected_classes, 'Frozen graph class schema required')
    for key in ('family_freeze_sha256', 'evaluation_manifest_sha256', 'predictor_lock_sha256'):
        digest = family_binding[key]
        require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                'Exact root custody SHA256 required: ' + key)
    expected = {(arm, seed) for arm in arms for seed in SEEDS}
    incomplete = dict(status='incomplete', successful_subset_scored=False,
                      expected_terminal_count=len(expected), observed_terminal_count=len(rows))
    try:
        pairs = [(row['arm'], row['seed']) for row in rows]
        require(len(pairs) == len(expected) and len(set(pairs)) == len(pairs) and set(pairs) == expected,
                'Complete frozen arm/seed family required')
        lookup = {}
        failures = []
        for row in rows:
            require(row['status'] == 'complete' and row['family_binding'] == family_binding,
                    'Incomplete terminal or changed immutable family binding')
            metrics = row['metrics']
            require(metrics['evaluation_scope'] == family_binding['evaluation_scope']
                    and metrics['class_schema'] == expected_classes, 'Mixed evaluation scope/class schema refused')
            values = {}
            for path in METRIC_PATHS:
                section, name = path.split('.')
                value = metrics[section][name]
                require(type(value) in (int, float) and math.isfinite(value), 'Finite complete metrics required')
                values[path] = float(value)
            lookup[(row['arm'], row['seed'])] = values
            if metrics['calibration']['status'] == 'fallback':
                failures.append(dict(arm=row['arm'], seed=row['seed'], reason=metrics['calibration']['failure_reason']))
        scores, contrasts = {}, {}
        for path in METRIC_PATHS:
            scores[path] = {arm: descriptive_five([lookup[(arm, seed)][path] for seed in SEEDS]) for arm in arms}
            contrasts[path] = {arm: descriptive_five([lookup[(reference_arm, seed)][path] - lookup[(arm, seed)][path]
                                                     for seed in SEEDS]) for arm in arms if arm != reference_arm}
        return dict(status='complete_paired_summary', family_binding=family_binding, reference_arm=reference_arm,
                    paired_seed_order=SEEDS, scores=scores, reference_minus_control=contrasts,
                    calibration_failures=failures, successful_subset_scored=False,
                    uncertainty='Descriptive five paired seeds only; illustrative t interval assumes independent approximately normal deltas; no significance, power or independent-dataset claim',
                    practical_scientific_gate_evaluated=False)
    except (KeyError, TypeError, ValueError) as error:
        return dict(incomplete, reason=type(error).__name__ + ': ' + str(error))

"""One deterministic CPU qualification function; no real data or model inference."""
import json
import sys
sys.dont_write_bytecode = True


def qualify():
    """Check meaningful calibration, metric, aggregation and family-closure cases."""
    try:
        import torch
    except ModuleNotFoundError:
        return dict(status='NOT_RUN', reason='Local Torch runtime absent; root may run on authorized server; do not install')
    import core
    classes = [0, 1, 2]
    ids = torch.tensor([10, 3], dtype=torch.long)
    requested = torch.tensor([3, 10], dtype=torch.long)
    labels = torch.tensor([1, 0], dtype=torch.long)
    members = torch.tensor([[[10., 0., -5.], [0., 10., -5.]],
                            [[0., 4., -5.], [4., 0., -5.]]], dtype=torch.float32)
    flat = torch.zeros((2, 3), dtype=torch.float32)
    tied = core.fit_inverse_temperature(flat, ids, requested, labels, classes, label_scope='SOURCE_VALIDATION_ONLY')
    assert tied['status'] == 'fitted' and tied['inverse_temperature'] == 1.0 and tied['selected_candidate_index'] == 0
    assert tied['iterations'] == tied['iterations_completed'] == 80
    assert [c['inverse_temperature'] for c in tied['candidates'][:3]] == [1., .05, 20.]
    perfect = torch.tensor([[2., 0., -2.], [2., 0., -2.]], dtype=torch.float32)
    bad = torch.tensor([[0., 40., -40.], [0., 40., -40.]], dtype=torch.float32)
    zeros = torch.zeros(2, dtype=torch.long)
    perfect_fit = core.fit_inverse_temperature(perfect, ids, requested, zeros, classes, label_scope='SOURCE_VALIDATION_ONLY')
    bad_fit = core.fit_inverse_temperature(bad, ids, requested, zeros, classes, label_scope='SOURCE_VALIDATION_ONLY')
    assert perfect_fit['status'] == bad_fit['status'] == 'fitted'
    assert perfect_fit['inverse_temperature'] == 20.0 and bad_fit['inverse_temperature'] == .05
    assert bad_fit['selected_calibration_NLL_FP64'] < bad_fit['raw_calibration_reference_NLL_FP64']
    invalid = flat.clone(); invalid[0, 0] = float('nan')
    failed = core.fit_inverse_temperature(invalid, ids, requested, labels, classes, label_scope='SOURCE_VALIDATION_ONLY')
    assert failed['status'] == 'fallback' and failed['inverse_temperature'] == 1.0 and 'Nonfinite raw logits' in failed['failure_reason']
    forbidden = core.fit_inverse_temperature(flat, ids, requested, labels, classes, label_scope='HELDOUT')
    assert forbidden['status'] == 'fallback' and 'source VALIDATION only' in forbidden['failure_reason']
    identity = tied
    metrics = core.evaluate_logits(members, ids, requested, labels, classes, calibration=identity, evaluation_scope='source_validation')
    expected = members.mean(0)[[1, 0]]
    torch.testing.assert_close(core.aggregate_logits(members, classes), members.mean(0), atol=0, rtol=0)
    assert metrics['raw']['NLL_FP32'] == float(torch.nn.functional.cross_entropy(expected, labels))
    assert metrics['raw']['NLL_FP64_calibration_reference'] == float(torch.nn.functional.cross_entropy(expected.double(), labels))
    mean_probability_nll = float(-members.softmax(-1).mean(0)[[1, 0], labels].log().mean())
    assert abs(mean_probability_nll - metrics['raw']['NLL_FP32']) > .1
    tiny = torch.tensor([[0., 1e-20, 0.], [0., 1e-20, 0.]], dtype=torch.float32)
    tiny_metrics = core.evaluate_logits(tiny, ids, requested, torch.ones(2, dtype=torch.long), classes,
                                       calibration=identity, evaluation_scope='source_validation')
    assert tiny_metrics['raw']['micro_F1'] == tiny_metrics['calibrated']['micro_F1'] == 1.0
    confident = torch.tensor([[1e20, -1e20, 0.], [1e20, -1e20, 0.]], dtype=torch.float32)
    correct = core.evaluate_logits(confident, ids, requested, zeros, classes, calibration=identity, evaluation_scope='source_validation')
    incorrect = core.evaluate_logits(confident, ids, requested, torch.ones(2, dtype=torch.long), classes,
                                    calibration=identity, evaluation_scope='source_validation')
    assert correct['raw']['Brier_FP64'] == correct['raw']['ECE15_FP64'] == 0.0
    assert incorrect['raw']['Brier_FP64'] == 2.0 and incorrect['raw']['ECE15_FP64'] == 1.0
    assert abs(correct['raw']['macro_F1'] - 1 / 3) < 1e-15
    binding = dict(dataset='HGB-ACM', evaluation_scope='source_validation', class_schema=classes, immutable=True,
                   arms=['CP', 'global_BE'], seeds=core.SEEDS,
                   family_freeze_sha256='0' * 64, evaluation_manifest_sha256='1' * 64, predictor_lock_sha256='2' * 64)
    rows = [dict(arm=arm, seed=seed, status='complete', family_binding=binding, metrics=metrics)
            for arm in binding['arms'] for seed in core.SEEDS]
    summary = core.paired_summary(rows, binding, reference_arm='CP')
    assert summary['status'] == 'complete_paired_summary'
    delta = summary['reference_minus_control']['raw.NLL_FP32']['global_BE']
    assert delta['vector'] == [0.] * 5 and delta['leave_one_block_out_means'] == [0.] * 5
    incomplete = core.paired_summary(rows[:-1], binding, reference_arm='CP')
    assert incomplete['status'] == 'incomplete' and 'scores' not in incomplete and 'reference_minus_control' not in incomplete
    return dict(status='PASS', Torch_version=torch.__version__, CPU_only=True,
                cases='flat-first-minimum tie; perfect upper/bad lower bounds; failure/forbidden-scope fallback; explicit ID mapping; FP32-mean-vs-probability aggregation; raw-FP32/separate-FP64 NLL; logit argmax despite rounded-softmax tie; exact Brier/ECE endpoint; fixed-class F1; complete/allfive family closure',
                original_data_labels_outcomes_graphs_models_or_remote_execution=False)


if __name__ == '__main__':
    result = qualify()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'PASS' else 2)

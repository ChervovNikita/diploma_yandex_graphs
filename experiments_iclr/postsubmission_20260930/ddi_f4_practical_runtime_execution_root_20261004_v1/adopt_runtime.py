"""Audit complete actual runtime receipts and retain their limited conclusion."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SOURCE = PHASE / 'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2'
EVIDENCE = HERE / 'monitor04'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    terminal = json.loads((EVIDENCE / 'QUEUE_TERMINAL.json').read_text())
    observation = json.loads((EVIDENCE / 'OBSERVATION.json').read_text())
    assert terminal['status'] == 'COMPLETE_PHYSICAL_RUNTIME_FAMILY' and terminal['failure'] is None
    assert observation['supervisor'] is None
    assert terminal['unattempted_arms'] == [] and not terminal['TEST_access'] and not terminal['VALID_scoring']
    assert sha(SOURCE / 'MANIFEST.json') == terminal['source_manifest_sha256']
    spec = importlib.util.spec_from_file_location('exact_runtime_receipt_rules', SOURCE / 'paired_comparison.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    streams, summaries, bindings = {}, {}, []
    for arm in rules.ARMS:
        folder = EVIDENCE / arm
        physical = json.loads((folder / 'PHYSICAL_TERMINAL.json').read_text())
        assert physical['physical_exit_code'] == 0 and physical['direct_child_reaped'] and physical['physical_session_closed']
        assert physical['stop'] is None and not physical['errors'] and not physical['termination_actions']
        summary = json.loads((folder / 'run01/runtime_summary.json').read_text())
        assert summary['status'] == 'COMPLETE_REAL_TRAIN_EPOCH_RUNTIME_ONLY'
        assert summary['completed_native_updates'] == 17 and summary['completed_native_records'] == 1067911
        assert summary['last_batch_records'] == 19335 and summary['seed'] == 0 and summary['arm'] == arm
        assert not summary['VALID_scored'] and not summary['TEST_read'] and not summary['checkpoints_saved']
        assert not summary['runtime_state_reusable_for_fits'] and summary['cuda_allocator_fraction'] == 0.30
        for field in ('epoch_wall_seconds', 'total_wall_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
            assert math.isfinite(summary[field]) and summary[field] > 0
        rows = [json.loads(line) for line in (folder / 'run01/paired_stream.jsonl').read_text().splitlines()]
        assert len(rows) == 17
        for batch, row in enumerate(rows):
            count = 65536 if batch < 16 else 19335
            assert rules.validate_receipt(row, arm, 0, 1, batch, count) == []
        streams[arm], summaries[arm] = rows, summary
        for name in ('PHYSICAL_TERMINAL.json', 'run01/runtime_summary.json', 'run01/paired_stream.jsonl', 'run01/runtime_updates.jsonl'):
            path = folder / name
            bindings.append(dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path)))
    for batch in range(17):
        rows = [streams[arm][batch] for arm in rules.ARMS]
        for field in rules.COMMON:
            assert all(row[field] == rows[0][field] for row in rows[1:]), (batch, field)
        for field in rules.AUXILIARY_COMMON:
            assert rows[1][field] == rows[2][field], (batch, field)
    # A full 500-epoch family is not thereby proved feasible. This TRAIN-only
    # linear scale is recorded solely to justify a prospective development budget.
    cf = sum(summaries[arm]['epoch_wall_seconds'] for arm in rules.ARMS)
    result = dict(UTC=datetime.now(timezone.utc).isoformat(),
        status='ROOT_ADOPTED_THREE_COMPLETE_REAL_TRAIN_EPOCH_RUNTIME_CHECKS',
        source_manifest_sha256=terminal['source_manifest_sha256'],
        root_release_sha256=terminal['root_release_sha256'],
        receipt_stream_alignment='PASS_ALL_17_ROWS_COMMON_AND_JOINT_SEPARATE_AUXILIARY',
        independent_source_review_manifest_sha256=sha(PHASE / 'ddi_f4_runtime_and_receipt_repair_independent_source_review_20261004_v2/MANIFEST.json'),
        arms={arm: {field: summaries[arm][field] for field in ('completed_native_updates', 'completed_native_records',
                 'epoch_wall_seconds', 'total_wall_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes',
                 'host_peak_RSS_bytes', 'support_teacher_aggregates')} for arm in rules.ARMS},
        linear_training_only_cost_scale=dict(original_nine_500epoch_GPU_hours=3*500*cf/3600,
            proposed_nine_F4_100epoch_GPU_hours=3*100*cf/3600,
            native_M1_cost_unmeasured=True, VALID_and_replay_cost_unmeasured=True,
            later_epoch_and_standalone_throughput_unproved=True),
        co_resident_timing=True, VALID_scoring=False, TEST_access=False, prediction_gain_established=False,
        methodological_novelty_established=False, full_budget_feasibility_established=False,
        model_states_discarded=True, original_paper_scores_changed=False, bindings=bindings)
    with (HERE / 'ROOT_ADOPTION.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    lines = ['# Actual DDI TRAIN runtime result', '',
        'Purpose: measure the complete native-sized training workload and the two exact conditional-pattern losses on the actual DDI graph before scheduling a predictive comparison.', '',
        'All three fresh seed-0 arms completed one full epoch: 17 updates, 1,067,911 positive records and the 19,335-record final batch. Physical exit 0, owned-session closure and reap passed in each case. Initial states, all native sampling/RNG receipts and joint/separate support/teacher receipts agree across all 17 updates.', '',
        '| Arm | Epoch seconds | Peak allocated GiB | Peak reserved GiB |', '|---|---:|---:|---:|']
    for arm in rules.ARMS:
        r = summaries[arm]
        lines.append(f"| {arm} | {r['epoch_wall_seconds']:.2f} | {r['peak_cuda_allocated_bytes']/1024**3:.2f} | {r['peak_cuda_reserved_bytes']/1024**3:.2f} |")
    lines += ['',
        f'Linear TRAIN-only scaling of these co-resident measurements gives {3*500*cf/3600:.1f} GPU-hours for the preserved nine-fit 500-epoch proposal, or {3*100*cf/3600:.1f} GPU-hours for nine F4 fits at 100 epochs. Native M1 fits, VALID, replay and later-epoch uncertainty are additional. These are budget scales, not promised completion times or standalone benchmarks.', '',
        'The auxiliary arms contain 544 selected positive and 544 native-negative queries per epoch, with complete supports and deterministic rows retained. Their actual population denominators are stored in ROOT_ADOPTION.json. This is exposure to informative reconstruction targets; it establishes no served ranking improvement.', '',
        'No VALID scores, TEST data, donor state or checkpoint were produced. Runtime states were discarded. The scientific 500-epoch source remains disabled and preserved. A separate prospectively fixed 100-epoch development comparison is being prepared from these TRAIN costs before any DDI predictive score. It must report the shorter budget and use separately fixed competitive confirmation if promising.', '',
        'The first bootstrap used a system Python without hashlib.file_digest and stopped before any remote launch. Its failed transport is preserved; the explicit recovery used the pinned Python 3.12 and the same reviewed release/workload. The original technical review does not cover that bootstrap adapter. No host settings or existing jobs were changed.', '']
    with (HERE / 'RESULTS_SUMMARY.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps({k: result[k] for k in ('status', 'receipt_stream_alignment', 'linear_training_only_cost_scale', 'prediction_gain_established')}, indent=2))


if __name__ == '__main__':
    main()

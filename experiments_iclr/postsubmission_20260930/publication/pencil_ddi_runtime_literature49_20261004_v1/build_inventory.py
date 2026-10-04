"""Build a small explicit publication inventory, excluding raw research payloads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PREFIX = 'experiments_iclr/postsubmission_20260930/'
EXTENSIONS = {'.py', '.json', '.jsonl', '.md', '.txt', '.diff'}


def main():
    paths = {PHASE / n for n in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json')}
    for name in ('pencil_collab_paired_predictive_preparation_20261004_v2',
                 'pencil_collab_paired_predictive_independent_source_review_20261004_v2',
                 'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v1',
                 'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v2',
                 'ddi_f4_exact_cb_independent_source_review_20261004_v1',
                 'ddi_f4_runtime_and_receipt_repair_independent_source_review_20261004_v2',
                 'literature_memory/index_v48', 'literature_memory/index_v49',
                 'status_history/20261004_1513_before_pencil_ddi_cost_literature49_update'):
        root = PHASE / name
        assert root.is_dir()
        paths.update(p for p in root.rglob('*') if p.is_file() and p.suffix in EXTENSIONS
                     and '__pycache__' not in p.parts)
    # Only code, explicit admissions and final physical/runtime receipts from
    # the DDI run; omit repeated monitor copies and staging transport encodings.
    root = PHASE / 'ddi_f4_practical_runtime_execution_root_20261004_v1'
    paths.update(p for p in root.glob('*.py'))
    for name in ('ROOT_AUTHORIZATION.md', 'STAGE_INVENTORY.json', 'STAGE_RECEIPT.json', 'ROOT_RELEASE.json',
                 'DETACHED_LAUNCH.json', 'BOOTSTRAP_RECOVERY_ADMISSION.json', 'ROOT_ADOPTION.json',
                 'RESULTS_SUMMARY.md', 'GPU_WINDOW_01.json',
                 'ddi_f4_practical_launch_20261004_LOCAL_TRANSPORT.json'):
        paths.add(root / name)
    paths.update(p for p in (root / 'monitor04').rglob('*') if p.is_file() and p.suffix in EXTENSIONS)
    pencil = PHASE / 'pencil_collab_paired_predictive_execution_root_20261004_v2'
    paths.update(p for p in pencil.glob('*.py'))
    for name in ('ROOT_AUTHORIZATION.md', 'ROOT_ADMISSION.json', 'ROOT_RELEASE.json', 'PRENUMERICAL_GATE.json',
                 'DEPENDENCY_ADMISSION_RECEIPT.json', 'SOURCE_STAGE_RECEIPT.json', 'QUEUE_SOURCE_STAGE.json',
                 'DETACHED_LAUNCH.json', 'MONITOR_0001.json', 'MONITOR_0002.json', 'MONITOR_0003.json'):
        paths.add(pencil / name)
    amazon = PHASE / 'amazon_polynormer_paired_family_execution_root_20261003_v3/v6_queue_owned_monitoring_20261003_v1'
    paths.update(amazon / ('MONITOR_%04d_RESULT.json' % n) for n in (33, 34))
    collab = PHASE / 'exact_cb_support_bucket_paired_predictive_execution_root_20261004_v1'
    for number in (7, 8):
        paths.update(collab / ('owned_monitor%03d' % number) / name for name in ('SUMMARY.json', 'OBSERVATION.json'))
    # Literature conclusions/scopes point to authenticated primary files. Raw
    # paper text/PDF/pixels stay in research custody and are not Git payloads.
    scout = PHASE / 'target_aligned_private_propagation_diversity_scout_20261004_v1'
    for name in ('REPORT.md', 'PROPOSAL.json', 'READ_SCOPES.json', 'LITERATURE_MEMORY_RECORD.json',
                 'MANIFEST.json', 'SEAL.json'):
        paths.add(scout / name)
    previous = PHASE / 'publication/pencil_launch_failure_ddi_and_filters_20261004_v1'
    paths.update(previous / name for name in ('PUSH_RECEIPT.json', 'ACKNOWLEDGEMENT.json'))
    paths.add(Path(__file__))
    rows = []
    for path in sorted(paths):
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(PHASE)
        assert path.suffix in EXTENSIONS
        raw = path.read_bytes()
        assert len(raw) < 2_000_000
        relative = str(path.relative_to(PHASE))
        rows.append(dict(source=relative, target=PREFIX + relative, bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest()))
    payload = dict(branch='codex/postsubmission-research-20260930',
                   expected_head='392eb51b457e3522aa620c2d82c2895477d29118', files=rows, remove=[],
                   message='Preserve PENCIL repair and actual DDI TRAIN cost; record prospective pilot and literature limits')
    with (HERE / 'INVENTORY.json').open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), files=len(rows), bytes=sum(r['bytes'] for r in rows),
                         raw_primary_files=False, dataset_or_checkpoint_payloads=False, predictive_results_promoted=False)))


if __name__ == '__main__':
    main()

"""Prepare a finite publication list; preserve oversized server evidence by hash."""
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
DEST = PHASE / 'publication/New_data_baselines_resource_closure_and_master_v7_20261012_v1'
PREFIX = 'experiments_iclr/postsubmission_20260930/'


def main():
    DEST.mkdir(exist_ok=True)
    assert not (DEST / 'INVENTORY.json').exists()
    package = PHASE / 'training_pipeline_v2_20261012'
    selected = {
        PHASE / 'RESEARCH_STATE.md',
        PHASE / 'publication/publish_exact_inventory_v33.py',
        Path(__file__).resolve(),
    }
    selected.update(package / name for name in (
        'README.md', 'phase_rendezvous.py',
        'compile_polyformer_practical_matrix.py',
        'compile_polyformer_coauthor_practical_matrix.py',
        'tests/test_polyformer_coauthor_practical.py',
        'preparation/prepare_polyformer_coauthor_practical_allocation.py',
        'preparation/qualify_controller_phase_v4.py',
        'preparation/assemble_extended_master_v7.py',
        'docs/POLYFORMER_COAUTHOR_PRACTICAL_V1.md',
        'docs/INTEGRATED_RUN_ROSTER_20261012_V7.md',
        'docs/COMPREHENSIVE_COVERAGE_AND_PORT_QUEUE_20261012.md',
        'docs/COMPREHENSIVE_COVERAGE_AND_PORT_QUEUE_20261012.json',
        'docs/OGB_BATCHNORM_AUTHOR_REFERENCE_PORT_PLAN_20261012.md',
        'docs/OGB_BATCHNORM_AUTHOR_REFERENCE_PORT_PLAN_20261012.json',
        'docs/NEW_STUDY_ANALYSIS_AND_CONVERGENCE_PROTOCOL_20261012.md',
        'docs/PRELAUNCH_AND_RUNTIME_BOUNDARIES_20261012.md',
    ))
    for relative in (
        'preparation/controller_phase_capacity_20261012_v4',
        'preparation/master_v6_capacity_inspection_20261012_v1',
        'preparation/practical_master_20261012_v7',
        'preparation/private_queue_root_closure_20261012_v1',
        'qa/comprehensive_coverage_port_queue_v1',
    ):
        selected.update(p for p in (package / relative).rglob('*')
                        if p.is_file() and p.suffix in ('.json', '.md', '.py', '.csv'))
    # The remote Coauthor packet is authoritative; mirrored receipts are retained
    # under their local mirror paths, without replacing its frozen source bytes.
    mirror = package / 'preparation/polyformer_coauthor_practical_allocation_20261012_v1'
    selected.update(p for p in mirror.rglob('*') if p.is_file()
                    and p.suffix in ('.json', '.md', '.py')
                    and p.name != 'PREPARATION_TRANSPORT.json')
    staged = PHASE / 'staged_resource_representative_review_20261012_v1/execution_v1'
    selected.add(staged / 'RESOURCE_CLOSURE_v2.json')
    selected.add(PHASE / 'private_GRU_acquisition_root_20261011_v1/OBSERVATION_preparation_resume_20261012_v1.json')

    references, rows = [], []
    for source in sorted(selected):
        assert source.is_file() and source.resolve().is_relative_to(PHASE)
        raw = source.read_bytes()
        row = {'source': str(source.relative_to(PHASE)),
               'target': PREFIX + str(source.relative_to(PHASE)),
               'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if len(raw) >= 2_000_000:
            references.append(row)
        else:
            rows.append(row)
    reference_path = DEST / 'OVERSIZED_METADATA_REFERENCES.json'
    assert not reference_path.exists()
    reference_path.write_text(json.dumps({
        'status': 'FULL_ORIGINALS_PRESERVED; COMPACT_PUBLICATION_ONLY',
        'prior_inventory_build_failed_before_remote_mutation': True,
        'publisher_limit_bytes_exclusive': 2_000_000,
        'omitted_full_metadata': references,
        'scientific_fits': 0, 'GO_created': False,
    }, indent=2) + '\n')
    raw = reference_path.read_bytes()
    relative = str(reference_path.relative_to(PHASE))
    rows.append({'source': relative, 'target': PREFIX + relative,
                 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    inventory = {'branch': 'codex/postsubmission-research-20260930',
                 'expected_head': 'a53c808436880343605661c9b6247267fa5e8932',
                 'message': 'Prepare new-data original GNNM baselines and preserve disabled master v7',
                 'remove': [], 'files': rows}
    (DEST / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                      'oversized_full_metadata_preserved': len(references)}))


if __name__ == '__main__':
    main()

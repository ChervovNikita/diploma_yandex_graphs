"""Record the first finite stage after reading the sealed scientific source."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
SOURCE = PHASE / 'pubmed_private_landmark_memory_execution_source_20261011_v1'
PUBLICATION = PHASE / 'publication/PubMed_LM4_sealed_source_prepare_admission_20261011_v1'

def receipt(path):
    path = Path(path)
    return dict(path=str(REMOTE / path.relative_to(PHASE)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    manifest_path = SOURCE / 'MANIFEST.json'
    assert receipt(manifest_path)['sha256'] == 'bf14d42fe933ac035af21d39dd24476f7245ae3f91ec667b67d80ea4a528850c'
    manifest = json.loads(manifest_path.read_text())
    for rel, row in manifest['files'].items():
        path = SOURCE / rel
        assert path.resolve().is_relative_to(SOURCE)
        assert path.stat().st_size == row['bytes'] and receipt(path)['sha256'] == row['sha256']
    stage = HERE / 'prepare'
    stage.mkdir(exist_ok=False)
    PUBLICATION.mkdir(exist_ok=False)
    decision = dict(
        UTC=datetime.now(timezone.utc).isoformat(),
        status='PREPARE_ADMITTED_AFTER_SEALED_SOURCE_REVIEW',
        admitted_stages=['prepare'],
        source_manifest=receipt(manifest_path),
        nomination_manifest_sha256='84d44d954104d0059db53f2393ecd1a99944784975d0b63dff6e366cb816e0fb',
        review_scope=['core graph-memory law and common/private ownership',
                     'explicit native head/memory Adam and private dropout streams',
                     'fresh source and own selected-state acquisition',
                     'SELECT-only fitting and label-free CAL/DEV exports',
                     'direct finite owner and fixed resource caps'],
        prepare_operations=['project frozen original VALID into separate SELECT/CAL/DEV roles',
                            'construct three label-free landmark/BFS capsules'],
        source_or_stage2_fits_admitted=False, qualifier_admitted=False,
        complete_family_required_before_DEV=True, TEST_closed=True,
        allocation_only=True, no_77_or_MacLink=True,
        original_scores_unchanged=True, quality_or_novelty_certified=False,
        subsequent_stage='One full-input qualifier after successful prepare closure; separate root admission',
    )
    decision_path = stage / 'ROOT_DECISION.json'
    write(decision_path, decision)
    config = json.loads((SOURCE / 'PREPARE_CONFIG.disabled.json').read_text())
    assert config['execution_enabled'] is False and config['stage'] == 'prepare'
    config.update(execution_enabled=True, source_manifest=receipt(manifest_path), root_decision=receipt(decision_path))
    write(stage / 'CONFIG.json', config)

    state_path = PHASE / 'RESEARCH_STATE.md'
    (PUBLICATION / 'PREVIOUS_RESEARCH_STATE.md').write_bytes(state_path.read_bytes())
    state = state_path.read_text()
    state = state.replace('Latest progress observation at18:29UTC: all9new ordinary P1 ancestors and the first B4 decoder-only control completed,10/30newfitgroups and10/48newoptimizer trajectories. G1block10103 was at update154.',
                          'Latest exact observation (OBSERVATION_05): all9new ordinary P1 ancestors plus B4 and G1 completed,11/30newfitgroups and11/48newoptimizer trajectories. GF1block10103 was at update102.')
    state = state.replace('No numerical execution is admitted yet. It will require its own fixed capable-single/genuine-independent comparisons and finite admission before concurrent execution.',
                          'The complete execution source is sealed and root-reviewed (manifest bf14d42fe933ac035af21d39dd24476f7245ae3f91ec667b67d80ea4a528850c). Only the finite preparation stage is admitted: fresh role projection and label-free landmark/BFS geometry. It is awaiting its actual launch. Qualification and all36fitgroups/54trajectories remain unadmitted until successful prior closure. Separate full-family reader preparation continues.')
    state = state.replace('the latest is`OBSERVATION_02.json`.', 'the latest is`OBSERVATION_05.json`.')
    state_path.write_text(state)
    ledger_path = PHASE / 'research_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    key = 'pubmed_LM4_sealed_source_prepare_admission_20261011_v1'
    assert key not in ledger
    ledger[key] = dict(decision=receipt(decision_path), config=receipt(stage/'CONFIG.json'),
                      source_manifest=receipt(manifest_path),
                      scientific_state='Preparation only; no fit/qualifier launched, no predictive result',
                      material_source_review_complete=True, fixed_roster=dict(banks=36,trajectories=54,CAL_endpoints=39),
                      existing_GRU_progress='11/30 fresh groups; all nine new P1 plus B4/G1 closed; no partial outcome opened',
                      allocation_only=True, novelty_and_acceptance_unmet=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True)+'\n')
    folders = [SOURCE, HERE, PHASE/'pubmed_private_landmark_memory_qualification_policy_root_20261011_v1',
               PHASE/'joint_shared_graph_ensemble_new_primary_scout_20261011_v1',
               PHASE/'pubmed_LM4_grouped_attention_equivalence_20261011_v1',
               PHASE/'private_memory_acquisition_interpretation_plan_20261011_v1']
    paths = [p for folder in folders for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md') and '__pycache__' not in p.parts and 'evidence' not in p.parts]
    paths.extend([state_path, ledger_path, PUBLICATION/'PREVIOUS_RESEARCH_STATE.md',
                  PHASE/'private_GRU_acquisition_root_20261011_v1/OBSERVATION_04.json',
                  PHASE/'private_GRU_acquisition_root_20261011_v1/OBSERVATION_05.json',
                  PHASE/'private_GRU_readout_after_whole_acquisition_root_20261011_v1/OBSERVATION_01.json',
                  PHASE/'private_GRU_readout_after_whole_acquisition_root_20261011_v1/OBSERVATION_02.json',
                  PHASE/'publication/LATEST_VERIFIED_20261011_V28.json'])
    paths = sorted(set(paths))
    files = []
    for path in paths:
        rel = path.relative_to(PHASE).as_posix()
        item = receipt(path)
        files.append(dict(source=rel,target='experiments_iclr/postsubmission_20260930/'+rel,
                          bytes=item['bytes'],sha256=item['sha256']))
    write(PUBLICATION/'INVENTORY.json',dict(
        branch='codex/postsubmission-research-20260930',
        expected_head='289fa8f6cc5cafc401d2f9818015f31f3d14f0e0',
        message='Seal PubMed landmark-memory source and admit allocation preparation',files=files,remove=[]))
    print(json.dumps(dict(admitted_stages=['prepare'],files=len(files),source_manifest=receipt(manifest_path))))

if __name__ == '__main__':
    main()

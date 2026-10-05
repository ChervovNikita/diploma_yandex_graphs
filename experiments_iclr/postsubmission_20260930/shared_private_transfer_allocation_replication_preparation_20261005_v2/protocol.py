"""Stdlib metadata authority for one explicitly counted prospective replication."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
REMOTE_PHASE = REMOTE_REPO/'experiments_iclr/postsubmission_20260930'
PROVIDER = 'authorized_one_GPU_allocation'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.suffix == '.json' and path.name not in
            ('FREEZE.json','CONFIG.json','EXTERNAL_ANCHORS.json'), 'Outcome payload is not metadata authority')
    def reject(value):
        raise ValueError('Nonfinite metadata: '+value)
    return json.loads(path.read_text(), parse_constant=reject)


def relative(value):
    path = Path(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts, 'Require phase-relative path')
    return path


def file_in(phase, value):
    phase = Path(phase).resolve(strict=True)
    path = phase/relative(value)
    require(path.resolve(strict=True).is_relative_to(phase), 'Input leaves authorized phase')
    for part in [path, *path.parents]:
        if part == phase.parent:
            break
        require(not part.is_symlink(), 'Symlink custody is not admitted')
    require(path.is_file(), 'Bound input must be a file')
    return path


def binding(phase, ref):
    require(isinstance(ref, dict) and isinstance(ref.get('sha256'), str)
            and len(ref['sha256']) == 64, 'Unresolved binding')
    path = file_in(phase, ref['path'])
    require(sha(path) == ref['sha256'], 'Bound bytes changed: '+ref['path'])
    return path


def reference(phase, path):
    return {'path':str(Path(path).relative_to(phase)), 'sha256':sha(path)}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_packet(expected):
    require(sha(HERE/'MANIFEST.json') == expected, 'Reviewed replication packet changed')
    for row in read(HERE/'MANIFEST.json')['files']:
        path = file_in(HERE, row['path'])
        require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Packet member changed')


def verify_frozen_protocol(phase, release):
    verify_packet(release['operational_manifest_sha256'])
    require(sha(HERE/'PROSPECTIVE_AMENDMENT.json') == release['amendment_sha256']
            and sha(HERE/'ATTEMPT_HISTORY.json') == release['attempt_history_sha256'], 'Prospective amendment/history changed')
    for key in ('execution_authorized','explicit_replication_exception_approved','source_review_approved',
                'all26_selected_before_any_score','full39_collection_contract_approved',
                'b0_terminal_artifact_custody_approved','allocation_serial_custody_approved'):
        require(release.get(key) is True, 'Disabled or unreviewed release: '+key)
    for key in ('TEST_access','retry','scores_read','setup_fallback','resumption'):
        require(release.get(key) is False, 'Prospective replication scope differs: '+key)
    for key in ('root_review_evidence','independent_review_evidence'):
        require(release.get(key), 'Exact operational review evidence required')
        for ref in release[key]:
            binding(phase, ref)
    mapping = read(HERE/'MAPPING_BINDING.json')
    for ref in mapping.values():
        binding(phase, ref)
    a = read(HERE/'PROSPECTIVE_AMENDMENT.json')
    binding(phase, a['original30']); binding(phase, a['companion9'])
    binding(phase, a['original_promotion_release']); binding(phase, a['external_anchors'])
    for history in read(HERE/'ATTEMPT_HISTORY.json')['old77_attempt_registration_and_observation']:
        binding(phase, history['launch_receipt']); binding(phase, history['queue'])
        for key in ('actual_original_donor_registration','actual_original_provider_admission','original_root_release'):
            binding(phase, history[key])
        require(history['latest_terminal_status'] == 'UNKNOWN_AFTER_WITHDRAWAL', 'Do not rewrite old terminal status')
    binding(phase, read(HERE/'ATTEMPT_HISTORY.json')['old77_saved_observation'])
    custody = read(binding(phase, release['b0_custody_binding']))
    require(custody.get('root_authenticated_all13_terminal_and_artifact_bytes') is True
            and custody.get('predetermined_b0_donors') == a['chosen_b0_donors']
            and custody.get('scores_read') is False and custody.get('TEST_access') is False
            and custody.get('both_prior_queue_processes_absent') is True,
            'Actual prior b0 complete13 custody and released queue absence required')
    require(custody.get('evidence'), 'Actual b0 terminal/artifact evidence required')
    for ref in custody['evidence']:
        binding(phase, ref)
    return a


def canonical_plan(phase, amendment, family):
    ref = amendment['family_plans'][family]
    path = binding(phase, ref)
    plan = read(path)
    require(plan['root_adopted_after_TRAIN_cost'] is True and plan['TEST_closed'] is True
            and plan['selection'] == 'first_maximum_complete_VALID_MRR_rounded4'
            and plan['selection_budget_fairness_approved'] is True, 'Existing scientific adoption changed')
    expected = [r['scientific_configuration'] for r in amendment['logical_cells39'] if r['family'] == family]
    require(plan['cells'] == expected, 'All exact canonical scientific cells required')
    metadata = amendment['family_metadata'][family]
    for key in ('complete_cycle_cost_evidence','root_numeric_decision','fit_bounds_by_cell'):
        require(plan[key] == metadata[key], 'Existing scientific cost/bounds authority changed')
    for ref in plan['complete_cycle_cost_evidence']+[plan['root_numeric_decision']]:
        binding(phase, ref)
    return path, plan


def job_from_preview(phase, amendment, row):
    job = read(HERE/'disabled_jobs'/(row['cell_id']+'.json'))
    require(job['fits_authorized'] is False and job['VALID_values_access'] is False,
            'Job preview must remain disabled')
    require(all(job[k] == v for k,v in row['scientific_configuration'].items())
            and job['physical_attempt_id'] == row['physical_attempt_id'], 'Exact preview mapping changed')
    job['fits_authorized'] = True
    job['VALID_values_access'] = True
    # Numerical fields, selected source/gate/runtime and original plan binding are unchanged.
    return job

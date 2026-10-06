#!/usr/bin/env python3
"""Prospective D2 only; disabled amended39 registry adapter, unchanged metrics."""
import argparse
from fractions import Fraction
import hashlib
import inspect
import itertools
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
# The explicit amended collector supplies complete39 custody. This adapter
# remains independently unadmitted; JSON alone cannot unlock this version.
COMPANION_COLLECTION_INTERFACE_REVIEWED = True
COLLECTION_PACKET = {'path': 'shared_private_transfer_allocation_replication_preparation_20261005_v2/MANIFEST.json',
                     'sha256': 'c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'}
COLLECTION_REVIEW = {'path': 'shared_private_transfer_allocation_replication_operational_independent_review_20261005_v2/REPORT.md',
                    'sha256': '786def5d8d61f59c6f5bf23b9e6271ca715efbef461fe5e961ef9a82635ad871'}
AMENDMENT = {'path': 'shared_private_transfer_allocation_replication_preparation_20261005_v2/PROSPECTIVE_AMENDMENT.json',
             'sha256': 'd965a699bc9d150470ea2b4b0fc0c07a4ffd90139230db065e0473b0311d7e2d'}
ATTEMPT_HISTORY = {'path': 'shared_private_transfer_allocation_replication_preparation_20261005_v2/ATTEMPT_HISTORY.json',
                   'sha256': 'e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'}
ORIGINAL_PLAN = {'path': 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/COHORT_PLAN.json',
                 'sha256': 'ae4b0f5c77cf48a86ccdbe51179fc95881157ac225f593c3992a5a5014d5346b'}
ORIGINAL_PROMOTION = {'path': 'shared_private_transfer_paired_pilot_execution_root_20261005_v2/ROOT_RELEASE.json',
                      'sha256': 'f638ee0d768cabb5efb999befa9bc688322c6086b983497fa5c49c8fe44d34e3'}
COMPANION_SPEC = {'path': 'shared_private_transfer_row0_single_companion_preparation_20261005_v1/COMPANION_PLAN_DISABLED.json',
                  'sha256': '774f45e1778ab7f0708aaa5f5a3a7aa6970968c63594218993673541856f72e4'}
SCIENCE = {'path': 'shared_private_transfer_paired_pilot_preparation_20261005_v2/SCIENCE_CONTRACT.json',
           'sha256': 'cdca2fe2ca5e2bfcd29e6fdee515d37570d21aaa246fca870f6a5105534d7445'}
SELECTOR = 'first_maximum_complete_VALID_MRR_rounded4'
MEMBERS = {'shared_f4': 4, 'untied4': 4, 'ordinary_native4': 4,
           'capable_single': 1, 'row0_single': 1}
SOURCE_SHA = {
    'original30': {'authorized_one_GPU_allocation': 'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f',
                   'authorized_18.77': '7f274c09bb317e6976d0c8b8636e779dc3ffcd2f2b76493d00f008b8bd08cebc'},
    'companion9': {'authorized_one_GPU_allocation': '67fab0016a144fdfda639e19cf1cf7feae4d10cbec04f1eaf8187917aa975ea8',
                   'authorized_18.77': '2b069a09de826ef94b2bbda981929f1903c29aa533628855cbd778f137faa7f0'}}
PROGRAM_SHA = {'original30': '6d7e75f9bae93ef88b2873f55f4f449ae52b9a0b6768fa808fae391da50ed524',
               'companion9': 'd2c7518bf1904aa7c7612ea5165a514ac7d5b11ddb0357937a56015cdb161ce6'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def phase_file(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Require a relative file in the project phase')
    path = PHASE/relative
    require(path.resolve(strict=True).is_relative_to(PHASE.resolve()), 'Input leaves the phase')
    for parent in [path, *path.parents]:
        if parent == PHASE.parent:
            break
        require(not parent.is_symlink(), 'Symlink input custody is not admitted')
    require(path.is_file(), 'Bound input must be a file')
    return path


def binding(reference):
    require(isinstance(reference, dict) and isinstance(reference.get('sha256'), str)
            and len(reference['sha256']) == 64, 'Unresolved exact metadata binding')
    path = phase_file(reference['path'])
    require(sha(path) == reference['sha256'], 'Bound file changed: '+reference['path'])
    return path


def packet(release):
    require(sha(HERE/'MANIFEST.json') == release['analysis_source_manifest_sha256'],
            'Root-reviewed D2 source packet changed')
    for row in read(HERE/'MANIFEST.json')['files']:
        path = (HERE/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and not (HERE/row['path']).is_symlink()
                and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'],
                'D2 preparation bytes changed')


def release_scope(release):
    require(release.get('root_analysis_approved') is True
            and release.get('collector30_plus9_custody_reviewed') is True
            and release.get('all_39_required_before_payload_access') is True,
            'Separate enabled completed-family analysis and custody release required')
    for key in ('TEST_access', 'fits_authorized', 'original_scores_recalculation',
                'checkpoint_reselection', 'scientific_model_execution', 'promotion_policy_change'):
        require(release.get(key) is False, 'D2 admits no '+key)
    require(release.get('diagnostics') == ['D2'] and release.get('selection') == SELECTOR,
            'Only fixed D2 at the existing selector is admitted')
    for key, expected in (('original_plan_binding', ORIGINAL_PLAN),
                          ('original_promotion_binding', ORIGINAL_PROMOTION),
                          ('companion_spec_binding', COMPANION_SPEC),
                          ('prospective_amendment_binding', AMENDMENT),
                          ('attempt_history_binding', ATTEMPT_HISTORY)):
        require(release.get(key) == expected, 'Prospective binding changed: '+key)
        binding(expected)
    require(release.get('custody_review_evidence'), 'Independent custody review evidence required')
    for reference in release['custody_review_evidence']:
        binding(reference)
    packet(release)


def existing_selector(freeze, history, cell):
    """Audit the retained source choice; return no alternative checkpoint."""
    require(sha(history) == freeze['history']['VALID_HISTORY.jsonl'], 'Selector history custody changed')
    rows = [read_line(line) for line in history.read_text().splitlines()]
    last = freeze['last_cycle']
    require(type(last) is int and 5 <= last <= cell['schedule']['max_cycles'] and last % 5 == 0,
            'Completed horizon differs')
    require([r['cycle'] for r in rows] == list(range(5, last+1, 5)), 'Complete fixed VALID cadence differs')
    scores = [r['complete_VALID']['MRR'] for r in rows]
    require(scores and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1
                           and round(v, 4) == v for v in scores), 'Native rounded4 selector history differs')
    first = scores.index(max(scores))
    require(rows[first]['cycle'] == freeze['selected_cycle'] and scores[first] == freeze['selected_VALID_MRR'],
            'Frozen serving state is not the original first maximum complete rounded4 VALID choice')
    return rows[first]['complete_VALID']


def read_line(line):
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def populations(release):
    """Retain the explicit amended complete39; never choose alternative donors."""
    original = read(binding(ORIGINAL_PLAN))
    companion = read(binding(COMPANION_SPEC))
    amendment = read(binding(AMENDMENT))
    require(original['selection'] == companion['selection'] == SELECTOR, 'Immutable selector differs')
    require(len(original['cells']) == 30 and len(companion['cells']) == 9, 'Exact30+9 spec required')
    registry_path = binding(release['complete39_registry'])
    registry = read(registry_path)
    require(registry.get('schema') == 'authenticated_complete_prospective_allocation_replication39_v1'
            and registry.get('complete') is True
            and registry.get('full39_terminal_source_artifact_custody_passed') is True
            and registry.get('selected_logical_scientific_fits') == 39
            and registry.get('selected_physical_fits') == 39 and registry.get('new_physical_fits') == 26
            and registry.get('physical_fits_lower_bound_including_known_old77_starts') == 45
            and registry.get('old77_final_physical_fits_and_costs_unknown') is True
            and registry.get('no_old_new_outcome_selection') is True,
            'Actual complete amended39 registry required')
    for key in ('comparative_scoring_performed', 'selected_prediction_payloads_deserialized',
                'quality_fields_accessed_or_emitted', 'TEST_access', 'fits_authorized',
                'history_bytes_observed', 'history_JSON_parsed', 'FREEZE_JSON_parsed',
                'CONFIG_JSON_parsed', 'automatic_retry'):
        require(registry.get(key) is False, 'Registry scope differs: '+key)
    require(registry['collector_manifest_sha256'] == COLLECTION_PACKET['sha256']
            and registry['amendment_sha256'] == AMENDMENT['sha256'], 'Operative registry source differs')
    records = registry['completed']
    require(len(records) == 39 and len({r['cell_id'] for r in records}) == 39
            and [r['cell_id'] for r in records] == amendment['selected39_order'],
            'No subset, duplicate, reordered or alternative cell is admissible')
    joined = []
    for family, spec, size in (('original30', original, 30), ('companion9', companion, 9)):
        rows = [r for r in records if r['family'] == family]
        require(len(rows) == size and [r['cell_id'] for r in rows] == spec['execution_order']
                and [r['scientific_configuration'] for r in amendment['logical_cells39']
                     if r['family'] == family] == spec['cells'],
                'Original full scientific population/selector differs')
        joined.append((family, spec, registry_path, registry, rows))
    common_inputs = registry['input_identities']
    require(set(common_inputs) == {'train_pos.txt', 'valid_pos.txt', 'heart_valid_samples.npy', 'gnn_feature'},
            'Common full fixed input/negative-pool identities differ')
    return joined, common_inputs


def provider_custody(family, row, job, record, registry_path, registry):
    """Consume the actual collector descriptor, never fabricate pre-fit admission."""
    reference = row['provider_custody_binding']
    authority = read(binding(reference))
    donor = record['donor']
    expected = {'schema': 'authenticated_allocation_replication_actual_custody_descriptor_v1',
        'descriptor_is_pre_fit_admission_object': False,
        'collector_authenticated_real_pre_fit_evidence': True,
        'provider': donor['provider'], 'hostname': donor['hostname'], 'GPU_UUID': donor['GPU_UUID'],
        'actual_donor': donor, 'source_manifest_sha256': job['source_manifest_sha256'],
        'program_sha256': job['program_sha256'], 'declared_runtime_versions': job['runtime_versions'],
        'qualified_runtime_versions': record['qualified_runtime_versions'],
        'CONFIG_runtime_observation_deferred': True,
        'retrospective_executable_source_custody': record['source_custody'],
        'chronology': {key: record[key] for key in
                      ('queue_start', 'child_start', 'exit_receipt', 'block_freeze', 'root_release')},
        'cross_host_wall_clock_order_assumed': False,
        'attempt_history_binding': registry['attempt_history_binding']}
    require(authority == expected
            and reference['path'] == str((registry_path.parent/('provider_custody_'+row['cell_id']+'.json')).relative_to(PHASE))
            and all(authority[k] == row[k] for k in ('provider', 'hostname', 'GPU_UUID'))
            and row['runtime_versions'] == job['runtime_versions'] == record['qualified_runtime_versions'],
            'Actual amended collector descriptor/source/runtime identity differs')
    # Source CONFIG is byte-bound now, then parsed only in the later all39
    # selected-state pass. No observation is inferred from a declaration.
    config = phase_file(str((record['output']/'CONFIG.json').relative_to(PHASE)))
    runtime = {'declared_runtime_versions': job['runtime_versions'],
               'qualified_runtime_versions': record['qualified_runtime_versions'],
               'observed_training_runtime_versions': None,
               'training_runtime_observation': {'path': str(config.relative_to(PHASE)), 'sha256': sha(config)},
               'training_runtime_observation_available': True, 'unavailable_reason': None,
               'observation_status': 'CONFIG byte custody only; semantic observation deferred'}
    return authority, reference, runtime


def history_inventory(release, joined):
    """Authenticate the actual amended retrospective inventory without history reads."""
    inventory = read(binding(release['VALID_history_inventory']))
    require(inventory['schema'] == 'retrospective_allocation_replication_full39_history_inventory_v1'
            and inventory['complete'] is True and inventory['selected_logical_scientific_fits'] == 39
            and inventory['new_physical_fits'] == 26
            and inventory['created_after_all39_terminal_source_artifact_checks'] is True
            and inventory['history_or_FREEZE_JSON_parsed'] is False
            and inventory['scores_read'] is False and inventory['pre_fit_authority'] is False,
            'Complete score-unparsed retrospective history byte inventory required')
    require(inventory['complete39_registry'] == release['complete39_registry']
            and inventory['inventory_manifest_sha256'] == COLLECTION_PACKET['sha256'],
            'History inventory registry/source custody differs')
    expected = [row['cell_id'] for _, _, _, _, rows in joined for row in rows]
    records = inventory['records']
    require(len(records) == 39 and [r['cell_id'] for r in records] == expected
            and len({r['cell_id'] for r in records}) == 39,
            'History byte inventory must retain exactly all39 original identities/order')
    return {r['cell_id']: r for r in records}, inventory


def collection_custody(release, joined, inventory):
    """Reuse the operative reviewed authenticate39 before the first history byte."""
    packet_path = binding(COLLECTION_PACKET)
    # Authenticate source members before importing only stdlib custody helpers.
    for member in read(packet_path)['files']:
        path = phase_file(str((packet_path.parent/member['path']).relative_to(PHASE)))
        require(sha(path) == member['sha256'] and path.stat().st_size == member['bytes'],
                'Reviewed operative collector packet changed')
    binding(COLLECTION_REVIEW)
    sys.path.insert(0, str(packet_path.parent))
    import protocol as p
    import collect39_metadata as m
    require(Path(p.__file__).resolve() == packet_path.parent/'protocol.py'
            and Path(m.__file__).resolve() == packet_path.parent/'collect39_metadata.py'
            and p.HERE.resolve() == packet_path.parent and m.PHASE.resolve() == PHASE.resolve(),
            'Require the exact operative local metadata helpers')
    p.verify_packet(COLLECTION_PACKET['sha256'])
    inventory_release = p.read(binding(inventory['release_binding']))
    require(inventory_release['root_history_inventory_approved'] is True
            and inventory_release['all39_required'] is True
            and inventory_release['retrospective_custody_only'] is True
            and inventory_release['source_review_evidence']
            and all(inventory_release[k] is False for k in
                    ('fits_authorized', 'TEST_access', 'scores_read', 'pre_fit_authority'))
            and inventory_release['inventory_manifest_sha256'] == COLLECTION_PACKET['sha256']
            and inventory_release['complete39_registry'] == release['complete39_registry'],
            'Actual separately reviewed retrospective history release required')
    for ref in inventory_release['source_review_evidence']:
        binding(ref)
    registry_path, registry = joined[0][2:4]
    collection_ref = {'path': str((registry_path.parent/'COLLECTION_RELEASE.json').relative_to(PHASE)),
                      'sha256': registry['collection_release_sha256']}
    collection = p.read(binding(collection_ref))
    require(collection['root_collection_approved'] is True
            and collection['full39_custody_before_score_history_parsing'] is True
            and collection['no_old_new_outcome_selection'] is True
            and all(collection[k] is False for k in ('fits_authorized', 'TEST_access', 'scores_read'))
            and collection['collector_manifest_sha256'] == COLLECTION_PACKET['sha256']
            and collection['amendment_sha256'] == AMENDMENT['sha256']
            and collection['attempt_history_sha256'] == ATTEMPT_HISTORY['sha256']
            and collection['source_review_evidence'], 'Actual approved complete39 collection required')
    for ref in collection['source_review_evidence']:
        binding(ref)
    amendment = p.read(binding(AMENDMENT))
    history = p.read(binding(ATTEMPT_HISTORY))
    # Retain actual original77 registrations and unknown costs; no old outcomes.
    for old in history['old77_attempt_registration_and_observation']:
        for key in ('launch_receipt', 'queue', 'actual_original_donor_registration',
                    'actual_original_provider_admission', 'original_root_release'):
            binding(old[key])
        require(old['latest_terminal_status'] == 'UNKNOWN_AFTER_WITHDRAWAL',
                'Do not infer withdrawn attempt outcomes')
    binding(history['old77_saved_observation'])
    require(registry['attempt_history_binding'] == {
                'path': str((registry_path.parent/'ATTEMPT_HISTORY.json').relative_to(PHASE)),
                'sha256': ATTEMPT_HISTORY['sha256']}
            and read(binding(registry['attempt_history_binding'])) == history,
            'Both original77 and selected replication attempt histories must remain exact')
    c, i = m.helpers()
    records, inputs = m.authenticate39(c, i, collection, amendment)
    require([r['cell']['cell_id'] for r in records] ==
            [row['cell_id'] for _, _, _, _, rows in joined for row in rows],
            'Reviewed complete39 closure must retain the exact D2 populations/order')
    return records, inputs


def custody(release, joined, common_inputs):
    """Finish full39 amended source/terminal/artifact custody before history bytes."""
    histories, inventory = history_inventory(release, joined)
    records, observed_inputs = collection_custody(release, joined, inventory)
    require(observed_inputs == common_inputs, 'All39 source fixed input identities differ')
    science = read(binding(SCIENCE))
    by_cell = {r['cell']['cell_id']: r for r in records}
    authorized = []
    for family, spec, registry_path, registry, rows in joined:
        for row in rows:
            record = by_cell[row['cell_id']]
            cell, job, donor, receipt = (record[k] for k in ('cell', 'job', 'donor', 'receipt'))
            require(record['family'] == family
                    and cell == next(c for c in spec['cells'] if c['cell_id'] == row['cell_id'])
                    and job['available_manifest_sha256'] == science['available_manifest_sha256'],
                    'Original scientific cell/source fixed inputs differ')
            output = record['output']
            expected_row = {'cell_id': cell['cell_id'], 'family': family, 'block': donor['block'],
                'provider': donor['provider'], 'hostname': donor['hostname'], 'GPU_UUID': donor['GPU_UUID'],
                'physical_attempt_id': job.get('physical_attempt_id'),
                'physical_attempt_id_was_source_emitted': donor['block'] != 'b0',
                'retained_actual_attempt_identity': {'donor_directory_relative': donor['donor_directory_relative'],
                    'job_sha256': receipt['job_sha256'], 'child_identity': receipt['child_identity']},
                'provider_custody_binding': row['provider_custody_binding'],
                'source_manifest_sha256': job['source_manifest_sha256'], 'program_sha256': job['program_sha256'],
                'runtime_versions': job['runtime_versions'],
                'original_donor_directory_relative': donor['donor_directory_relative'],
                'job_relative': str(record['job_path'].relative_to(PHASE)), 'job_sha256': receipt['job_sha256'],
                'freeze_relative': str((output/'FREEZE.json').relative_to(PHASE)),
                'freeze_sha256': receipt['freeze_sha256'],
                'selected_checkpoint_relative': record['checkpoint']['path'],
                'checkpoint_sha256': record['checkpoint']['sha256'],
                'selected_VALID_logits_relative': record['logits']['path'],
                'VALID_logits_sha256': record['logits']['sha256'], 'donor_execution_receipt': receipt}
            require(row == expected_row, 'Actual registry row must equal the exact collector record')
            admission, provider_reference, runtime = provider_custody(
                family, row, job, record, registry_path, registry)
            require(row['source_manifest_sha256'] == SOURCE_SHA[family].get(row['provider'])
                    and row['program_sha256'] == PROGRAM_SHA[family], 'Qualified source/program differs')
            path = binding({'path': row['freeze_relative'], 'sha256': row['freeze_sha256']})
            logits = binding({'path': row['selected_VALID_logits_relative'], 'sha256': row['VALID_logits_sha256']})
            checkpoint = binding({'path': row['selected_checkpoint_relative'], 'sha256': row['checkpoint_sha256']})
            require(logits.parent == checkpoint.parent == path.parent
                    and path.name == 'FREEZE.json' and logits.name == 'selected_VALID_logits.pt'
                    and checkpoint.name == 'selected_checkpoint.pt', 'Retained state/logit paths differ')
            history_record = histories[row['cell_id']]
            require(history_record['freeze_binding'] == {'path': row['freeze_relative'], 'sha256': row['freeze_sha256']}
                    and history_record['history_binding']['path'] == str((path.parent/'VALID_HISTORY.jsonl').relative_to(PHASE)),
                    'Retrospective inventory must preserve the actual source-fixed history/FREEZE identity')
            authorized.append({'family': family, 'cell': cell, 'row': row, 'freeze_path': path,
                'logits': logits, 'history_reference': history_record['history_binding'],
                'history_sha256': history_record['history_binding']['sha256'],
                'cohort_plan_sha256': job['cohort_plan_sha256'],
                'provider_reference': provider_reference, 'runtime': runtime})
    require(len(authorized) == 39, 'All39 metadata/source/artifact custody must finish before history bytes')
    for item in authorized:
        item['history'] = binding(item.pop('history_reference'))
    return authorized


def selected_state_custody(authorized, common_inputs):
    """Only after complete byte custody, bind every retained FREEZE to its artifacts."""
    require(len(authorized) == 39, 'All39 prior byte custody required')
    states = []
    for item in authorized:
        row, cell, path, runtime = item['row'], item['cell'], item['freeze_path'], item['runtime']
        # This is the first outcome-bearing JSON read. All39 original/companion
        # collection, metadata, source and artifact byte checks already returned.
        value = read(path)
        require(value['scope'] == 'TRAIN_VALID_freeze' and value['complete_VALID_scoring'] is True
                and value['TEST_access'] is False and value['cohort_plan_sha256'] == item['cohort_plan_sha256']
                and value['source_manifest_sha256'] == row['source_manifest_sha256']
                and value['job_sha256'] == row['job_sha256'] and value['input_identities'] == common_inputs
                and all(value[k] == cell[k] for k in ('arm', 'rule', 'geometry', 'seed', 'paired_seed_block')),
                'Selected-state freeze source/input/job identity differs')
        require(value['VALID_logits_sha256'] == row['VALID_logits_sha256']
                and value['checkpoint_sha256'] == row['checkpoint_sha256']
                and value['history']['VALID_HISTORY.jsonl'] == item['history_sha256'],
                'Actual source FREEZE checkpoint/logit/history reference differs from authenticated byte custody')
        if runtime['training_runtime_observation_available'] is True:
            observation = binding(runtime['training_runtime_observation'])
            require(observation.parent == path.parent and observation.name == 'CONFIG.json'
                    and sha(observation) == value['config_sha256']
                    and read(observation)['runtime'] == runtime['declared_runtime_versions']
                    == runtime['qualified_runtime_versions'],
                    'Observed training runtime must be the actual source CONFIG bound by this FREEZE')
        runtime = dict(runtime, observed_training_runtime_versions=read(observation)['runtime'],
                       observation_status='Actual source CONFIG bound by retained FREEZE')
        states.append(dict(item, runtime=runtime, frozen=value))
    require(len(states) == 39, 'All39 FREEZE semantic bindings must finish before history score audits')
    return states


def selector_audits(states):
    """Audit every original retained choice only after all39 byte/semantic custody."""
    require(len(states) == 39, 'Complete retained-state custody required before scores')
    authorized = []
    for item in states:
        selected_quality = existing_selector(item['frozen'], item['history'], item['cell'])
        authorized.append((item['cell'], item['row'], item['frozen'], item['logits'], selected_quality,
                           item['provider_reference'], item['runtime']))
    require(len(authorized) == 39, 'All39 selector audits required before numeric import/logits')
    return authorized


def defined(value):
    require(math.isfinite(value), 'Unexpected nonfinite diagnostic')
    return {'defined': True, 'value': value, 'reason': None}


def undefined(reason):
    return {'defined': False, 'value': None, 'reason': reason}


def error_correlation(torch, a, b):
    a, b = a.double(), b.double()
    if bool((a == a[0]).all()) or bool((b == b[0]).all()):
        return undefined('zero_reciprocal_rank_error_variance')
    da, db = a-a.mean(), b-b.mean()
    va, vb = (da*da).sum(), (db*db).sum()
    if float(va) == 0 or float(vb) == 0:
        return undefined('zero_reciprocal_rank_error_variance')
    return defined(float((da*db).sum()/(va*vb).sqrt()))


def d2(torch, logits, member_count):
    p, n, pp, pn = (logits[k] for k in ('member_pos', 'member_neg', 'mean_pos', 'mean_neg'))
    for value, shape in ((p, (227, member_count)), (n, (227, 500, member_count)),
                         (pp, (227,)), (pn, (227, 500))):
        require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                and value.dtype == torch.float32 and tuple(value.shape) == shape
                and bool(torch.isfinite(value).all()), 'Full finite native saved VALID logits required')
    require(torch.equal(pp, p.mean(-1)) and torch.equal(pn, n.mean(-1)),
            'Saved pooled raw logits disagree with all retained member logits')
    member_twice_rank = 2+(n >= p[:, None, :]).sum(1)+(n > p[:, None, :]).sum(1)
    pooled_twice_rank = 2+(pn >= pp[:, None]).sum(1)+(pn > pp[:, None]).sum(1)
    # These float32 operations reproduce the source rank.float() MRR reduction.
    mr = 1/(member_twice_rank.float()*.5)
    pr = 1/(pooled_twice_rank.float()*.5)
    mh, ph = member_twice_rank <= 20, pooled_twice_rank <= 20
    member_denominators, pooled_denominators = member_twice_rank.tolist(), pooled_twice_rank.tolist()
    gains, relation = [], {'higher': 0, 'equal': 0, 'lower': 0}
    for denominators, pooled in zip(member_denominators, pooled_denominators):
        difference = Fraction(2, pooled)-sum((Fraction(2, d) for d in denominators), Fraction())/member_count
        gains.append(float(difference))
        relation['higher' if difference > 0 else 'lower' if difference < 0 else 'equal'] += 1
    pairs = []
    # Correlations use float64 RR values from exact integer midranks.
    errors = 1-2/member_twice_rank.double()
    for i, j in itertools.combinations(range(member_count), 2):
        ei, ej = ~mh[:, i], ~mh[:, j]
        intersection, union = int((ei & ej).sum()), int((ei | ej).sum())
        pairs.append({'member_indices': [i, j], 'RR_error_Pearson': error_correlation(torch, errors[:, i], errors[:, j]),
                      'Hit10_error_overlap': {'both_error_queries': intersection, 'queries': 227,
                                             'joint_error_rate': intersection/227,
                                             'member_error_queries': [int(ei.sum()), int(ej.sum())],
                                             'error_union_queries': union,
                                             'intersection_over_union': defined(intersection/union) if union else
                                             undefined('no_Hit10_errors_in_either_member')}})
    # Source own_loss uses these exact operators, with positive and negative
    # means separate. The 500 negatives do not downweight positive examples.
    logsigmoid = torch.nn.functional.logsigmoid
    aggregate_pos = float(-logsigmoid(pp.double()).mean())
    aggregate_neg = float(-logsigmoid(-pn.double()).mean())
    own_pos = [float(-logsigmoid(p[:, i].double()).mean()) for i in range(member_count)]
    own_neg = [float(-logsigmoid(-n[:, :, i].double()).mean()) for i in range(member_count)]
    member_pos, member_neg = math.fsum(own_pos)/member_count, math.fsum(own_neg)/member_count
    aggregate, mean_member = aggregate_pos+aggregate_neg, member_pos+member_neg
    return {'members': [{'index': i, 'MRR': float(mr[:, i].mean()), 'Hits10': float(mh[:, i].float().mean()),
                         'positive_BCE': own_pos[i], 'negative_BCE': own_neg[i], 'BCE': own_pos[i]+own_neg[i]}
                        for i in range(member_count)],
            'pooled': {'MRR': float(pr.mean()), 'Hits10': float(ph.float().mean())},
            'pooling': {'mean_query_pooled_minus_mean_member_RR': math.fsum(gains)/227,
                        'query_relation_counts': relation, 'query_relation_fractions': {k: v/227 for k, v in relation.items()},
                        'comparison': 'exact rational RR from integer twice-midranks; no equality tolerance',
                        'applicability': 'single_member_identity' if member_count == 1 else 'multiple_members'},
            'pairs': pairs, 'pair_applicability': 'not_applicable_single_member' if member_count == 1 else 'all_member_pairs',
            'BCE': {'aggregate_positive': aggregate_pos, 'aggregate_negative': aggregate_neg, 'aggregate': aggregate,
                    'mean_member_positive': member_pos, 'mean_member_negative': member_neg, 'mean_member': mean_member,
                    'Jensen_gap_positive': member_pos-aggregate_pos, 'Jensen_gap_negative': member_neg-aggregate_neg,
                    'Jensen_gap': mean_member-aggregate,
                    'source_outer_objective': .5*aggregate+.5*mean_member},
            'query_count': 227, 'negatives_per_query': 500, 'member_count': member_count}


def block_mean(values):
    return math.fsum(values)/3


def block_summaries(records):
    summaries = {}
    for name in sorted({r['cell'] for r in records}):
        rows = [next(r for r in records if r['cell'] == name and r['block'] == b) for b in ('b0', 'b1', 'b2')]
        stats = [r['D2'] for r in rows]
        count = stats[0]['member_count']
        require(all(s['member_count'] == count for s in stats), 'Member shape differs across paired blocks')
        members = [{key: block_mean([s['members'][i][key] for s in stats]) for key in
                    ('MRR', 'Hits10', 'positive_BCE', 'negative_BCE', 'BCE')} for i in range(count)]
        pairs = []
        for index in range(len(stats[0]['pairs'])):
            ps = [s['pairs'][index] for s in stats]
            correlations = [p['RR_error_Pearson'] for p in ps]
            overlaps = [p['Hit10_error_overlap']['intersection_over_union'] for p in ps]
            pairs.append({'member_indices': ps[0]['member_indices'],
                          'RR_error_Pearson': {'block_values': correlations, 'arithmetic_mean':
                                              defined(block_mean([v['value'] for v in correlations])) if all(v['defined'] for v in correlations)
                                              else undefined('at_least_one_block_correlation_undefined')},
                          'Hit10_joint_error_rate': {'block_values': [p['Hit10_error_overlap']['joint_error_rate'] for p in ps],
                                                    'arithmetic_mean': block_mean([p['Hit10_error_overlap']['joint_error_rate'] for p in ps])},
                          'Hit10_error_intersection_over_union': {'block_values': overlaps, 'arithmetic_mean':
                                                                defined(block_mean([v['value'] for v in overlaps])) if all(v['defined'] for v in overlaps)
                                                                else undefined('at_least_one_block_error_union_empty')}})
        summaries[name] = {'block_cell_ids': [r['cell_id'] for r in rows], 'equal_weight_blocks': ['b0', 'b1', 'b2'],
                           'members': [dict(index=i, **member) for i, member in enumerate(members)],
                           'pooled': {key: block_mean([s['pooled'][key] for s in stats]) for key in ('MRR', 'Hits10')},
                           'pooled_minus_mean_member_RR': block_mean([s['pooling']['mean_query_pooled_minus_mean_member_RR'] for s in stats]),
                           'query_relation_fractions': {key: block_mean([s['pooling']['query_relation_fractions'][key] for s in stats]) for key in ('higher', 'equal', 'lower')},
                           'BCE': {key: block_mean([s['BCE'][key] for s in stats]) for key in stats[0]['BCE']},
                           'pairs': pairs, 'pair_applicability': stats[0]['pair_applicability']}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(COMPANION_COLLECTION_INTERFACE_REVIEWED,
            'Execution disabled: explicit amended39 semantic/custody adapter and separate full39 analysis release require root review')
    require(args.release.resolve(strict=True).is_relative_to(PHASE), 'Release leaves the phase')
    release = read(args.release)
    release_scope(release)
    joined, common_inputs = populations(release)
    metadata = custody(release, joined, common_inputs)
    states = selected_state_custody(metadata, common_inputs)
    authorized = selector_audits(states)
    output = args.output.resolve()
    require(output.is_relative_to(PHASE) and not output.exists() and output.parent.is_dir(), 'Fresh phase output required')
    # No numeric library, logit payload or checkpoint deserialization precedes
    # the separate release and metadata custody of the complete exact39 family.
    import torch
    require('weights_only' in inspect.signature(torch.load).parameters, 'Safe retained-logit loading required')
    records = []
    with torch.no_grad():
        for cell, row, frozen, path, selected_quality, provider_reference, runtime in authorized:
            logits = torch.load(path, map_location='cpu', weights_only=True)
            require(set(logits) == {'member_pos', 'member_neg', 'mean_pos', 'mean_neg', 'selected_cycle', 'inputs', 'checkpoint_sha256'},
                    'Retained VALID artifact schema differs')
            require(logits['checkpoint_sha256'] == frozen['checkpoint_sha256']
                    and logits['selected_cycle'] == frozen['selected_cycle']
                    and logits['inputs'] == common_inputs, 'Selected serving state/input binding differs')
            stats = d2(torch, logits, MEMBERS[cell['arm']])
            require(round(stats['pooled']['MRR'], 4) == frozen['selected_VALID_MRR']
                    and round(stats['pooled']['Hits10'], 4) == selected_quality['Hits10'],
                    'Saved full population does not reproduce retained native selected quality')
            records.append({'cell_id': cell['cell_id'], 'cell': cell['cell'], 'block': cell['cell_id'].split('_', 1)[0],
                            'arm': cell['arm'], 'rule': cell['rule'], 'geometry': cell['geometry'],
                            'selected_cycle': frozen['selected_cycle'], 'last_cycle': frozen['last_cycle'],
                            'selected_rounded4_VALID_MRR': frozen['selected_VALID_MRR'],
                            'provider': row['provider'], 'hostname': row['hostname'], 'GPU_UUID': row['GPU_UUID'],
                            'source_manifest_sha256': row['source_manifest_sha256'], 'provider_custody_binding': provider_reference,
                            'runtime_provenance': runtime,
                            'job_sha256': row['job_sha256'], 'freeze_sha256': row['freeze_sha256'],
                            'physical_attempt_id': row['physical_attempt_id'],
                            'physical_attempt_id_was_source_emitted': row['physical_attempt_id_was_source_emitted'],
                            'retained_actual_attempt_identity': row['retained_actual_attempt_identity'],
                            'retained_registry_paths': {key: row[key] for key in
                                                       ('job_relative', 'freeze_relative', 'selected_checkpoint_relative',
                                                        'selected_VALID_logits_relative', 'original_donor_directory_relative')},
                            'checkpoint_sha256': row['checkpoint_sha256'], 'VALID_logits_sha256': row['VALID_logits_sha256'], 'D2': stats})
    require(len(records) == 39, 'D2 must retain all39 records')
    result = {'schema': 'fixed39_D2_selected_VALID_development_diagnostics_v1', 'diagnostics': ['D2'],
              'release_sha256': sha(args.release),
              'registry_bindings': {'complete39': release['complete39_registry']},
              'prospective_amendment_binding': AMENDMENT, 'attempt_history_binding': ATTEMPT_HISTORY,
              'selected_logical_scientific_fits': 39, 'new_allocation_physical_fits': 26,
              'physical_fits_lower_bound_including_known_old77_starts': 45,
              'old77_final_physical_fits_and_costs_unknown': True,
              'original_plan_binding': ORIGINAL_PLAN, 'original_promotion_binding': ORIGINAL_PROMOTION,
              'companion_spec_binding': COMPANION_SPEC, 'selector': SELECTOR, 'input_identities': common_inputs,
              'records': records, 'arithmetic_block_means': block_summaries(records),
              'numeric_runtime': {'torch': torch.__version__, 'device': 'cpu'},
              'numerics': {'quality': 'source native float32 tie-aware midrank and reductions',
                           'RR_query_relations': 'exact rational midranks, with float64 mean signed differences',
                           'correlation': 'float64 Pearson of1-RR; no zero-variance stabilizer',
                           'BCE': 'source negative logsigmoid operators in float64, positive and negative means separately, gap unclamped'},
              'TEST_access': False, 'fits_authorized': False, 'checkpoint_reselection': False,
              'original_scores_recalculation': False, 'scientific_model_execution': False,
              'acceptance_verdict': None, 'specialization_verdict': None, 'historical_update_causality_identified': False,
              'limits': ['Selected-state development diagnostics only; no new quality/promotion gate.',
                         'All39 and all three blocks retained; undefined block statistics are never dropped from means.',
                         'Member errors/pooling do not identify which historical private update caused them.',
                         'Four inner streams are not four F1 members; single-member pair statistics are inapplicable.',
                         'Tiny negative float64 Jensen gaps, if any, are retained as numerical rounding rather than clamped.']}
    output.mkdir()
    with (output/'D2_RESULTS.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print('Completed fixed full39 D2 diagnostics: '+str(output))


if __name__ == '__main__':
    main()

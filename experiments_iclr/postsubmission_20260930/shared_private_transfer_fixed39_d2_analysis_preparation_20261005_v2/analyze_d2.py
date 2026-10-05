#!/usr/bin/env python3
"""Prospective D2 only; execution disabled pending authenticated30+9 custody review."""
import argparse
from fractions import Fraction
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
# No companion collector currently supplies this contract. A separately reviewed
# source revision is required; an enabled JSON alone cannot unlock this version.
COMPANION_COLLECTION_INTERFACE_REVIEWED = False
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
                          ('companion_spec_binding', COMPANION_SPEC)):
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
    """Read two existing registries; this is not a collector or terminal authority."""
    original = read(binding(ORIGINAL_PLAN))
    companion = read(binding(COMPANION_SPEC))
    require(original['selection'] == companion['selection'] == SELECTOR, 'Immutable selector differs')
    require(len(original['cells']) == 30 and len(companion['cells']) == 9, 'Exact30+9 spec required')
    joined = []
    common_inputs = None
    for family, spec, size, schema in (
        ('original30', original, 30, 'authenticated_complete_mixed_provider_paired_pilot_collection_v1'),
        ('companion9', companion, 9, 'authenticated_complete_row0_companion_collection_v1')):
        registry_path = binding(release['registries'][family])
        registry = read(registry_path)
        require(registry.get('schema') == schema and registry.get('complete') is True
                and registry.get('physical_fits') == size, 'Authenticated complete registry required: '+family)
        for key in ('comparative_scoring_performed', 'selected_prediction_payloads_opened',
                    'quality_fields_accessed_or_emitted', 'TEST_access', 'fits_authorized',
                    'original_score_recalculation', 'retry'):
            require(registry.get(key) is False, 'Registry scope differs: '+key)
        records = registry['completed']
        order = spec['execution_order']
        require(len(records) == size and len(set(order)) == size
                and [row['cell_id'] for row in records] == order,
                'No subset, duplicate, reordered or unavailable cell is admissible')
        # Both complete populations are checked before reading any per-fit outcome.
        joined.append((family, spec, registry_path, registry, records))
        if common_inputs is None:
            common_inputs = registry['input_identities']
        require(registry['input_identities'] == common_inputs and set(common_inputs) ==
                {'train_pos.txt', 'valid_pos.txt', 'heart_valid_samples.npy', 'gnn_feature'},
                'Common full fixed input/negative-pool identities differ')
    require(len({r['cell_id'] for _, _, _, _, rows in joined for r in rows}) == 39,
            'Distinct complete fixed39 population required')
    return joined, common_inputs


def provider_custody(family, row, job):
    if family == 'original30':
        reference = {'path': row['provider_admission_relative'], 'sha256': row['provider_admission_sha256']}
        authority = read(binding(reference))
        require(authority['approved'] is True and authority['admitted_before_provider_first_fit'] is True,
                'Original actual pre-fit provider admission differs')
        require(row['runtime_versions'] == job['runtime_versions'] == authority['runtime_versions'],
                'Original actual runtime provenance differs')
        runtime = {'declared_runtime_versions': job['runtime_versions'],
                   'qualified_runtime_versions': authority['runtime_versions'],
                   'observed_training_runtime_versions': None,
                   'observation_status': 'original_registry_retained; no new CONFIG runtime observation inferred'}
    else:
        reference = row['provider_custody_binding']
        authority = read(binding(reference))
        require(authority['schema'] == 'authenticated_row0_companion_provider_custody_descriptor_v1'
                and authority['collector_authenticated_pre_fit_evidence'] is True
                and authority['descriptor_is_pre_fit_admission_object'] is False,
                'Honest derived companion custody descriptor required')
        # The descriptor itself may be derived after completion. The collector
        # must authenticate these actual immutable pre-fit records and their
        # chronology; no legacy filename or fictional earlier object is needed.
        proof = authority['pre_fit_evidence']
        for key in ('root_reviews', 'root_release', 'block_job_freeze_authentication', 'launch_receipt',
                    'source_manifest', 'training_step_gate', 'qualified_runtime', 'complete_cycle_costs'):
            require(isinstance(proof.get(key), list) and proof[key], 'Actual pre-fit evidence missing: '+key)
            for evidence in proof[key]:
                binding(evidence)
        require({'path': job['training_step_gate']['path'], 'sha256': job['training_step_gate']['sha256']}
                in proof['training_step_gate'] and authority['provider_source_manifest'] in proof['source_manifest'],
                'Descriptor must retain the actual job source/gate bindings')
        runtime = authority['runtime_provenance']
        require(runtime['declared_runtime_versions'] == job['runtime_versions'], 'Companion declared runtime changed')
        require(runtime['qualified_runtime_versions'] == job['runtime_versions'],
                'Bound companion qualified runtime differs')
        if runtime['training_runtime_observation_available'] is True:
            binding(runtime['training_runtime_observation'])
            require(runtime['observed_training_runtime_versions'] == job['runtime_versions'],
                    'Observed qualified training runtime differs')
        else:
            require(runtime['training_runtime_observation_available'] is False
                    and runtime['observed_training_runtime_versions'] is None
                    and runtime['unavailable_reason'], 'Missing runtime observation must remain explicit')
    require(all(authority[k] == row[k] for k in ('provider', 'hostname', 'GPU_UUID')),
            'Actual provider custody identity differs')
    return authority, reference, runtime


def history_inventory(release, joined):
    """Authenticate retrospective byte custody, not pre-fit/historical authority."""
    inventory = read(binding(release['VALID_history_inventory']))
    require(inventory['schema'] == 'retrospective_full39_VALID_history_inventory_v1'
            and inventory['complete'] is True and inventory['physical_fits'] == 39
            and inventory['created_after_all39_terminal_and_completeness_checks'] is True
            and inventory['history_or_FREEZE_JSON_parsed'] is False
            and inventory['scores_read'] is False and inventory['pre_fit_authority'] is False,
            'Complete score-unparsed retrospective history byte inventory required')
    require(inventory['registries'] == release['registries'], 'History inventory registry custody differs')
    expected = [row['cell_id'] for _, _, _, _, rows in joined for row in rows]
    records = inventory['records']
    require(len(records) == 39 and [r['cell_id'] for r in records] == expected
            and len({r['cell_id'] for r in records}) == 39,
            'History byte inventory must retain exactly all39 original identities/order')
    return {r['cell_id']: r for r in records}


def custody(release, joined, common_inputs):
    """Complete all39 metadata/source/artifact hash checks; parse no fit outcomes."""
    authorized = []
    science = read(binding(SCIENCE))
    histories = history_inventory(release, joined)
    for family, spec, registry_path, registry, rows in joined:
        root = registry_path.parent
        collection_release_path = phase_file(str((root/'COLLECTION_RELEASE.json').relative_to(PHASE)))
        require(sha(collection_release_path) == registry['collection_release_sha256'],
                'Existing collector release custody changed')
        collector_release = read(collection_release_path)
        require(collector_release.get('root_collection_approved') is True
                and collector_release.get('TEST_access') is False
                and collector_release.get('fits_authorized') is False,
                'Existing collection authority is missing')
        plan_path = phase_file(str((root/'COHORT_PLAN.json').relative_to(PHASE)))
        require(sha(plan_path) == registry['cohort_plan_sha256'], 'Collected full plan custody changed')
        plan = read(plan_path)
        require(len(plan['cells']) == len(spec['cells']) and all(plan[key] == spec[key] for key in
                ('cells', 'blocks', 'execution_order', 'resource_assignment', 'selection')),
                'Collected scientific population/selector differs')
        if family == 'original30':
            require(registry['cohort_plan_sha256'] == ORIGINAL_PLAN['sha256']
                    and registry['science_contract_sha256'] == SCIENCE['sha256']
                    and sha(phase_file(str((root/'EXTERNAL_ANCHORS.json').relative_to(PHASE)))) == registry['external_anchors_sha256'],
                    'Original collector science/plan/verbatim-anchor custody differs')
        else:
            require(registry['companion_spec_sha256'] == COMPANION_SPEC['sha256']
                    and registry['original_plan_binding'] == ORIGINAL_PLAN
                    and registry['original_promotion_binding'] == ORIGINAL_PROMOTION,
                    'Companion collector original/spec custody differs')
        for row in rows:
            # Spec cell listing and execution_order need not have identical order.
            cell = next(c for c in spec['cells'] if c['cell_id'] == row['cell_id'])
            require(cell['schedule'] == dict(max_cycles=60, eval_every_cycles=5, validation_miss_limit=11),
                    'Fixed horizon/cadence/miss schedule changed')
            receipt = row['donor_execution_receipt']
            require(receipt.get('terminal_wait_observed') is True and receipt.get('exit_code') == 0
                    and receipt.get('exit_code_authority') == 'subprocess.Popen.wait/poll'
                    and receipt.get('reason') is None and receipt.get('signals_sent') == []
                    and receipt.get('attempts') == 1 and receipt.get('retry') is False,
                    'Successful owned terminal receipt required for every fit')
            job_path = binding({'path': row['job_relative'], 'sha256': row['job_sha256']})
            job = read(job_path)
            require(all(job[k] == v for k, v in cell.items()) and job['cohort_plan_sha256'] == registry['cohort_plan_sha256'],
                    'Original per-fit scientific job identity differs')
            require(job['purpose'] == 'TRAIN_VALID_prospective_private_transfer_fit'
                    and job['fits_authorized'] is True and job['VALID_values_access'] is True
                    and job['TEST_access'] is False and job['retry'] is False,
                    'Prospective per-fit scope differs')
            require(job['available_manifest_sha256'] == science['available_manifest_sha256'],
                    'Original fixed input manifest binding differs')
            available = read(binding({'path': job['available_manifest_relative'],
                                      'sha256': job['available_manifest_sha256']}))
            require(available['TEST_available_to_loader'] is False and set(available['files']) == set(common_inputs)
                    and {name: {key: available['files'][name][key] for key in ('sha256', 'bytes')}
                         for name in common_inputs} == common_inputs,
                    'Retained logits must bind exactly the source fixed VALID/negative-pool input identities')
            admission, provider_reference, runtime = provider_custody(family, row, job)
            require(row['source_manifest_sha256'] == SOURCE_SHA[family].get(row['provider'])
                    == job['source_manifest_sha256'] == admission['provider_source_manifest']['sha256']
                    and row['program_sha256'] == PROGRAM_SHA[family] == job['program_sha256']
                    == admission['program']['sha256'],
                    'Actual already-qualified source/program binding differs')
            source_manifest = binding(admission['provider_source_manifest'])
            program = binding(admission['program'])
            require(program.parent == source_manifest.parent and program.name == 'run.py',
                    'Actual program must belong to the bound qualified source')
            for source_file in read(source_manifest)['files']:
                source_path = phase_file(str((source_manifest.parent/source_file['path']).relative_to(PHASE)))
                require(sha(source_path) == source_file['sha256'] and source_path.stat().st_size == source_file['bytes'],
                        'Qualified provider source bytes changed')
            path = binding({'path': row['freeze_relative'], 'sha256': row['freeze_sha256']})
            logits = binding({'path': row['selected_VALID_logits_relative'], 'sha256': row['VALID_logits_sha256']})
            checkpoint = binding({'path': row['selected_checkpoint_relative'], 'sha256': row['checkpoint_sha256']})
            require(logits.parent == checkpoint.parent == path.parent
                    and path.name == 'FREEZE.json' and logits.name == 'selected_VALID_logits.pt'
                    and checkpoint.name == 'selected_checkpoint.pt', 'Retained state/logit paths differ')
            history_record = histories[row['cell_id']]
            require(history_record['freeze_binding'] == {'path': row['freeze_relative'], 'sha256': row['freeze_sha256']},
                    'Retrospective history inventory must preserve the original FREEZE identity')
            history = binding(history_record['history_binding'])
            require(history.parent == path.parent and history.name == 'VALID_HISTORY.jsonl',
                    'Source-fixed retained VALID history path differs')
            authorized.append({'family': family, 'cell': cell, 'row': row, 'freeze_path': path,
                               'logits': logits, 'history': history,
                               'history_sha256': history_record['history_binding']['sha256'],
                               'cohort_plan_sha256': registry['cohort_plan_sha256'],
                               'provider_reference': provider_reference, 'runtime': runtime})
    require(len(authorized) == 39, 'All39 metadata/source/artifact custody must finish before outcome JSON reads')
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
        if item['family'] == 'companion9' and runtime['training_runtime_observation_available'] is True:
            observation = binding(runtime['training_runtime_observation'])
            require(observation.parent == path.parent and observation.name == 'CONFIG.json'
                    and sha(observation) == value['config_sha256']
                    and read(observation)['runtime'] == runtime['observed_training_runtime_versions'],
                    'Observed training runtime must be the actual source CONFIG bound by this FREEZE')
        states.append(dict(item, frozen=value))
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
            'Execution disabled: authenticated exact9 companion interface/custody and a separate full39 analysis release require root review')
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
                            'retained_registry_paths': {key: row[key] for key in
                                                       ('job_relative', 'freeze_relative', 'selected_checkpoint_relative',
                                                        'selected_VALID_logits_relative', 'original_donor_directory_relative')},
                            'checkpoint_sha256': row['checkpoint_sha256'], 'VALID_logits_sha256': row['VALID_logits_sha256'], 'D2': stats})
    require(len(records) == 39, 'D2 must retain all39 records')
    result = {'schema': 'fixed39_D2_selected_VALID_development_diagnostics_v1', 'diagnostics': ['D2'],
              'release_sha256': sha(args.release), 'registry_bindings': release['registries'],
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

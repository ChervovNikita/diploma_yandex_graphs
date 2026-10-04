"""Unexecuted preparation: exact reader v1 in a fabricated workspace namespace.

Execution requires a separate root release. No production paths or outcomes are
opened. Namespace and fault-hook substitutions are documented in README.md.
"""
from argparse import ArgumentParser
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


WORKSPACE = Path('/Users/alex/Documents/ChatGPT/anogena allocation')
PHASE = WORKSPACE / 'postsubmission_research_20260930'
SOURCE = PHASE / 'ncnc_pattern_complete_pair_reader_preparation_20261004_v1'
SOURCE_SHA = '34b43a3052fea1b62e7370c59b66fc286ba0ab588b34d5587a43b6df3e16a9cf'
HERE = Path(__file__).resolve().parent
EXECUTION = PHASE / 'ncnc_complete_pair_reader_file_path_qualification_execution_20261004_v1'
RECEIPT = EXECUTION / 'LOCAL_FILE_PATH_QUALIFICATION.json'
DRIVER_SHA = '9fc539b8f92d7d4e883224c3b0aa85ae583b64c70f2a701a4a648fd818aa32a1'
FAMILY = 'ncnc-structural-pattern-pair-20261003-v1'
ROUTES = ('own', 'crossed_cyclic_1', 'crossed_cyclic_2', 'crossed_cyclic_3', 'pooled_clamped_weights')
STRATA = ('all', 'has_synthetic_removal', 'no_synthetic_removal', 'cn0', 'has_common')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def pin(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=digest(raw))


def verify_packet(root, expected):
    raw = (root / 'MANIFEST.json').read_bytes()
    require(digest(raw) == expected, 'Packet manifest differs')
    metadata = json.loads(raw)
    for row in metadata['files']:
        path = root / row['path']
        require(path.resolve().parent == root and not path.is_symlink(), 'Packet payload path escaped')
        payload = path.read_bytes()
        require(len(payload) == row['bytes'] and digest(payload) == row['sha256'], 'Packet payload differs')
    return raw, metadata


def source_row(index):
    zero, one = index + 2, 1
    strata = {}
    for name in STRATA:
        count = 65536 if name == 'all' else 65535 if name == 'cn0' else 1
        active = 2 if name == 'all' else 1
        strata[name] = dict(queries=count, joint_query_sum=active * .1,
                            factorial_query_sum=active * .3,
                            joint_minus_factorial_query_sum=active * -.2,
                            between_component_spread_query_sum=active * .01)
    return dict(queries=65536, active_queries=2, residual_slots=zero + one,
                synthetic_removed_observed_slots=one, source_unobserved_slots=zero,
                component_entropy_query_sum=.4, between_component_spread_query_sum=.02,
                active_responsibility_entropy_sum=.8, active_responsibility_max_sum=1.6,
                normalized_native_q_vs_stable_sigmoid_max_abs=0., strata=strata,
                bit_marginal={'0': dict(slots=zero, nll_sum=zero * index, brier_sum=zero * .1,
                                       predicted_observation_probability_sum=zero * .2),
                              '1': dict(slots=one, nll_sum=.3, brier_sum=.4,
                                       predicted_observation_probability_sum=.5)})


def pair_fixture():
    arms, selections = {}, {}
    for arm, hits in (('J', .55), ('F', .56)):
        selections[arm] = dict(order=25, hits50=hits)
        valid = dict(served_hits50=hits, member_hits50=[.5] * 4,
                     positive_coincident_error_fractions=[.25] * 6, positive_oracle_union=.7,
                     member_balanced_BCE=[.8] * 4, served_balanced_BCE=.7,
                     positive_queries=60084, negative_queries=100000,
                     fixed_bank_counterfactual_routes={route: dict(served_hits50=hits, served_balanced_BCE=.7)
                                                       for route in ROUTES},
                     resource_coverage=dict(fabricated_only=True, route_count=5),
                     score_digests=dict(positive_bank='0' * 64, negative_bank='1' * 64),
                     not_a_selector=True, graph='complete_TRAIN_only', serving_pool='mean_raw_logits')
        arms[arm] = dict(selected_epoch=25, VALID=valid,
                         mask_event_batches=[dict(native_member_main_loss=[.8, .9, 1., 1.1],
                                                  positive=source_row(i), negative=source_row(i))
                                             for i in range(1, 18)],
                         source_pattern_mask_wall_seconds=0., mask_seed=2026100307,
                         TRAIN_stream=dict(fabricated_only=True, full_batches=17),
                         support_digests=[digest(('FABRICATED_SUPPORT_' + str(i)).encode()) for i in range(34)],
                         source_teacher=dict(fabricated_only=True, teacher='complete_TRAIN_observation_membership',
                                             source_zero_semantics='unobserved_in_TRAIN_not_verified_latent_nonlink',
                                             teacher_used_as_predictor_input=False),
                         auxiliary_scorer_backward=False, new_optimization_or_selection=False)
    return dict(schema='ncnc-pattern-pair-results-v1',
                identity=dict(driver_manifest_sha256=DRIVER_SHA, family_id=FAMILY),
                selections=selections, representation_diagnostics=arms,
                selected_served_VALID_Hits50_J_minus_F=.55 - .56,
                all_100_native_streams_RNG_and_actual_supports_match=True, test_file_opened=False,
                scope='single_seed_validation_selected_development_pilot_not_confirmatory',
                source_pattern_scope='TRAIN_observation_incidence_not_latent_link_truth',
                stronger_claim_requires_covariance_aware_competent_single=True,
                fixture_scope='FABRICATED_SCALARS_ONLY_NO_SCIENTIFIC_OBSERVATION')


def repin(fixture):
    write_json(fixture.pair_path, fixture.pair)
    pair_pin = pin(fixture.pair_path)
    fixture.closure['pair_results'] = dict(pair_pin, path='PAIR_RESULTS.json')
    write_json(fixture.closure_path, fixture.closure)
    fixture.terminal['closure_receipt'] = pin(fixture.closure_path)
    fixture.terminal['owned_artifact_receipts'] = {'PAIR_RESULTS.json': pair_pin}
    write_json(fixture.terminal_path, fixture.terminal)
    fixture.release['closure_receipt'] = pin(fixture.closure_path)
    fixture.release['closure_normal_terminal'] = pin(fixture.terminal_path)
    write_json(fixture.release_path, fixture.release)


def make_fixture(root, source_raw, source_metadata):
    source_copy = root / 'reader_source'
    source_copy.mkdir(parents=True)
    (source_copy / 'MANIFEST.json').write_bytes(source_raw)
    for row in source_metadata['files']:
        (source_copy / row['path']).write_bytes((SOURCE / row['path']).read_bytes())
    verify_packet(source_copy, SOURCE_SHA)
    repository = root / 'repository'
    phase = repository / 'experiments_iclr/postsubmission_20260930'
    execution = phase / 'graph_ncNC_structural_pattern_execution_root_20261004_v5'
    repository.mkdir()
    fixture = SimpleNamespace(root=root, source=source_copy, repository=repository, phase=phase,
                              execution=execution, output=execution / 'complete_pair_analysis/run01',
                              release_path=execution / 'FABRICATED_ROOT_RELEASE.json',
                              terminal_path=execution / 'supervision/close/run01/SUPERVISOR_TERMINAL.json',
                              closure_path=execution / 'close/run01/CLOSURE.json',
                              pair_path=execution / 'close/run01/PAIR_RESULTS.json', pair=pair_fixture())
    accounting = dict(fabricated_only=True, attempts=1, failed_or_interrupted_attempts=0,
                      closed_attempt_wall_seconds=0., observed_interrupted_wall_seconds=0.,
                      total_cost_exact=True, terminal_accounting_write_tail_measured=False)
    fixture.closure = dict(schema='ncnc-pattern-pair-closure-v1', status='CLOSED',
                           identity=fixture.pair['identity'], unique_scientific_fits=2,
                           scientific_optimizer_steps=3400, scientific_selector_candidates=200,
                           complete_scientific_VALID_evaluations_including_replay_and_diagnostics=206,
                           matched_pair_streams_RNG_and_supports=True, TEST_supported=False,
                           preclosure_bound_stage_accounting={stage: dict(accounting)
                               for stage in ('fit_J', 'fit_F', 'diagnostics', 'numerical', 'full_graph')},
                           inclusive_accounting=dict(accounting),
                           cost_scope='bound_stage_receipts_and_their_own_ledgers_plus_root_listed_prior_failed_stage_receipts; external_unlisted_attempts_not_certified',
                           fabricated_only=True)
    fixture.terminal = dict(schema='ncnc-pattern-normal-diagnostics-closure-supervision-terminal-v1',
                            status='COMPLETE', stage='close', child_exit_code=0, child_signal=None,
                            cap_violation=None, complete_stage_custody_checks_passed=True,
                            complete_driver_peak_capture_checks_passed=True, TEST_opened=False,
                            fabricated_only=True, ordinary_host_execution=True,
                            terminal_write_tail_measured=False)
    fixture.release = dict(schema='ncnc-complete-pair-reader-root-release-v1', execution_enabled=True,
                           reader_manifest_sha256=SOURCE_SHA, root_authorization_reference='FABRICATED_ONLY',
                           output_directory=str(fixture.output), fabricated_only=True)
    repin(fixture)
    spec = importlib.util.spec_from_file_location('fabricated_exact_reader_v1', source_copy / 'reader.py')
    require(spec is not None and spec.loader is not None, 'Reader loader unavailable')
    reader = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(reader)
    finally:
        sys.dont_write_bytecode = previous
    # Only the fixture module's in-memory namespace is adapted; source bytes stay exact.
    reader.REPO, reader.PHASE, reader.EXECUTION = repository, phase, execution
    reader.HERE = source_copy
    reader.socket = SimpleNamespace(gethostname=lambda: 'peptide')
    fixture.reader = reader
    return fixture


CASES = (
    ('complete_main_success', None),
    ('missing_terminal_pin', 'closure_normal_terminal'),
    ('missing_pin_sha', 'sha256'),
    ('missing_closure_file', 'CLOSURE.json'),
    ('wrong_terminal_byte_length', 'Pinned artifact bytes differ'),
    ('wrong_terminal_sha', 'Pinned artifact bytes differ'),
    ('wrong_terminal_fixed_path', 'Unexpected fixed artifact path'),
    ('terminal_not_complete', 'Normal complete closure is missing'),
    ('terminal_closure_binding', 'Normal-terminal closure binding differs'),
    ('terminal_pair_binding', 'Entire results payload binding differs'),
    ('altered_pair_bytes', 'Pinned artifact bytes differ'),
    ('altered_source_payload', 'Reader source payload changed'),
    ('altered_source_manifest', 'Reader source manifest differs'),
    ('release_outside_execution', 'Path outside admitted root'),
    ('symlink_release', 'Canonical nonsymlink path required'),
    ('wrong_output_path', 'Unexpected analysis output'),
    ('existing_output_directory', 'File exists'),
    ('partial_pair_arm', 'Incomplete/unmatched pair or TEST access'),
    ('partial_pair_batches', 'Fixed diagnostic coverage differs'),
    ('pair_changed_after_assembly', 'Pinned artifact bytes differ'),
    ('source_changed_after_assembly', 'Reader source payload changed'),
    ('release_changed_after_assembly', 'Root release changed'),
)


def mutate(fixture, case):
    if case == 'missing_terminal_pin':
        del fixture.release['closure_normal_terminal']
    elif case == 'missing_pin_sha':
        del fixture.release['closure_normal_terminal']['sha256']
    elif case == 'missing_closure_file':
        fixture.closure_path.unlink()
    elif case == 'wrong_terminal_byte_length':
        fixture.release['closure_normal_terminal']['bytes'] += 1
    elif case == 'wrong_terminal_sha':
        fixture.release['closure_normal_terminal']['sha256'] = '0' * 64
    elif case == 'wrong_terminal_fixed_path':
        other = fixture.execution / 'OTHER_TERMINAL.json'
        other.write_bytes(fixture.terminal_path.read_bytes())
        fixture.release['closure_normal_terminal'] = pin(other)
    elif case == 'terminal_not_complete':
        fixture.terminal['status'] = 'FAILED'
        repin(fixture)
    elif case in ('terminal_closure_binding', 'terminal_pair_binding'):
        target = fixture.terminal['closure_receipt'] if case == 'terminal_closure_binding' else fixture.terminal['owned_artifact_receipts']['PAIR_RESULTS.json']
        target['sha256'] = '0' * 64
        write_json(fixture.terminal_path, fixture.terminal)
        fixture.release['closure_normal_terminal'] = pin(fixture.terminal_path)
    elif case == 'altered_pair_bytes':
        fixture.pair_path.write_bytes(fixture.pair_path.read_bytes() + b'\n')
    elif case == 'altered_source_payload':
        path = fixture.source / 'README.md'
        path.write_bytes(path.read_bytes() + b'\n')
    elif case == 'altered_source_manifest':
        path = fixture.source / 'MANIFEST.json'
        path.write_bytes(path.read_bytes() + b'\n')
    elif case == 'release_outside_execution':
        fixture.release_path = fixture.root / 'OUTSIDE_EXECUTION_RELEASE.json'
    elif case == 'symlink_release':
        link = fixture.execution / 'SYMLINK_RELEASE.json'
        link.symlink_to(fixture.release_path)
        fixture.release_path = link
        return
    elif case == 'wrong_output_path':
        fixture.release['output_directory'] = str(fixture.execution / 'wrong_analysis/run01')
    elif case == 'existing_output_directory':
        fixture.output.mkdir(parents=True)
        (fixture.output / 'SENTINEL.txt').write_text('DO_NOT_OVERWRITE\n', encoding='utf-8')
    elif case in ('partial_pair_arm', 'partial_pair_batches'):
        if case == 'partial_pair_arm':
            del fixture.pair['representation_diagnostics']['F']
        else:
            fixture.pair['representation_diagnostics']['J']['mask_event_batches'].pop()
        repin(fixture)
    elif case.endswith('_changed_after_assembly'):
        original = fixture.reader.assemble
        target = fixture.pair_path if case.startswith('pair_') else fixture.source / 'README.md' if case.startswith('source_') else fixture.release_path
        def assembled_then_change(*args):
            result = original(*args)
            target.write_bytes(target.read_bytes() + b'\n')
            return result
        fixture.reader.assemble = assembled_then_change
    write_json(fixture.release_path, fixture.release)


def invoke(fixture):
    captured = io.StringIO()
    previous_cwd, previous_argv = Path.cwd(), sys.argv
    error = None
    try:
        os.chdir(fixture.repository)
        sys.argv = [str(fixture.source / 'reader.py'), '--root-release', str(fixture.release_path)]
        with redirect_stdout(captured):
            fixture.reader.main()
    except Exception as caught:
        error = caught
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    return captured.getvalue(), error


def qualify(fixture, name, expected_error):
    mutate(fixture, name)
    stdout, error = invoke(fixture)
    marker = 'COMPLETE_PAIR_ANALYSIS_WRITTEN ' + str(fixture.output)
    if expected_error is not None:
        require(error is not None and expected_error in str(error), 'Unexpected refusal for ' + name)
        require(stdout == '', 'Rejected input emitted stdout or a success marker')
        if name == 'existing_output_directory':
            require(sorted(p.name for p in fixture.output.iterdir()) == ['SENTINEL.txt'], 'Existing output was mutated')
            require((fixture.output / 'SENTINEL.txt').read_text() == 'DO_NOT_OVERWRITE\n', 'Sentinel changed')
        else:
            require(not fixture.output.exists(), 'Rejected input created analysis output')
        return dict(case=name, status='PASS', refused_exception=type(error).__name__,
                    refused_condition=str(error), success_marker_emitted=False)
    require(error is None, 'Complete fabricated main failed: ' + str(error))
    require(stdout == marker + '\n', 'Final success marker differs')
    require(sorted(p.name for p in fixture.output.iterdir()) == ['ANALYSIS.json', 'COMPLETE_PAIR_RESULTS.json', 'REPORT.md'], 'Complete output set differs')
    analysis = json.loads((fixture.output / 'ANALYSIS.json').read_text())
    retained = json.loads((fixture.output / 'COMPLETE_PAIR_RESULTS.json').read_text())
    require(retained == fixture.pair, 'Entire fabricated pair was not retained')
    expected_inputs = [fixture.release['closure_normal_terminal'], fixture.release['closure_receipt'], pin(fixture.pair_path)]
    require(analysis['reader_manifest_sha256'] == SOURCE_SHA and analysis['input_receipts'] == expected_inputs
            and analysis['root_release_sha256'] == pin(fixture.release_path)['sha256'], 'Analysis input custody differs')
    require(analysis['primary_J_minus_F_Hits50_pp'] == 100 * (.55 - .56), 'Primary arithmetic differs')
    require(analysis['primary_interpretation'].startswith('No quality support'), 'Adverse primary was rescued')
    require(analysis['seed_count'] == 1 and analysis['confidence_intervals_or_significance_claims'] is False, 'Inference scope differs')
    require(analysis['pattern_targets_are_TRAIN_observation_not_latent_truth'] is True and analysis['TEST_opened'] is False, 'Observation/TEST scope differs')
    require(analysis['final_write_tail_measured'] is False, 'Write-tail scope differs')
    require('No quality support' in (fixture.output / 'REPORT.md').read_text(), 'Human report is incomplete')
    verify_packet(fixture.source, SOURCE_SHA)
    return dict(case=name, status='PASS', success_marker_emitted=True,
                complete_output_files=[pin(fixture.output / filename) for filename in ('ANALYSIS.json', 'REPORT.md', 'COMPLETE_PAIR_RESULTS.json')])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    args = parser.parse_args()
    require(WORKSPACE.resolve() == WORKSPACE and HERE.resolve().is_relative_to(WORKSPACE), 'Canonical workspace required')
    require(EXECUTION.resolve() == EXECUTION, 'Canonical qualification execution root required')
    release_path = args.root_release
    require(release_path == EXECUTION / 'ROOT_RELEASE.json' and release_path.resolve() == release_path
            and not release_path.is_symlink(), 'Fixed qualification root release required')
    release_raw = release_path.read_bytes()
    release = json.loads(release_raw)
    require(release['schema'] == 'ncnc-reader-file-path-qualification-root-release-v1'
            and release['execution_enabled'] is True and release['root_authorization_reference'], 'Separate root authorization required')
    require(release['reader_manifest_sha256'] == SOURCE_SHA and release['fixture_workspace'] == str(WORKSPACE)
            and release['output_receipt'] == str(RECEIPT), 'Qualification scope differs')
    harness_raw, _ = verify_packet(HERE, release['harness_manifest_sha256'])
    source_raw, source_metadata = verify_packet(SOURCE, SOURCE_SHA)
    require(not RECEIPT.exists(), 'Qualification receipt already exists')
    cases = []
    with TemporaryDirectory(prefix='ncnc_reader_filepins_fabricated_', dir=WORKSPACE) as temporary:
        fixture_root = Path(temporary)
        require(fixture_root.resolve().is_relative_to(WORKSPACE), 'Temporary fixture escaped workspace')
        for index, (name, expected_error) in enumerate(CASES, start=1):
            fixture = make_fixture(fixture_root / ('case_' + str(index)), source_raw, source_metadata)
            cases.append(qualify(fixture, name, expected_error))
    verify_packet(SOURCE, SOURCE_SHA)
    verify_packet(HERE, digest(harness_raw))
    require(release_path.read_bytes() == release_raw, 'Qualification root release changed')
    receipt = dict(schema='ncnc-reader-file-path-fabricated-qualification-v1',
                   UTC=datetime.now(timezone.utc).isoformat(), status='PASS', cases=cases,
                   case_count=len(cases), reader_manifest_sha256=SOURCE_SHA,
                   harness_manifest_sha256=digest(harness_raw), root_release_sha256=digest(release_raw),
                   fabricated_inputs_only=True, temporary_fixtures_removed=True,
                   exact_reader_source_executed_in_fixture_namespace=True,
                   namespace_bindings=['REPO', 'PHASE', 'EXECUTION', 'HERE', 'socket.gethostname'],
                   after_assembly_hooks=['fixture_pair_bytes', 'fixture_source_bytes', 'fixture_release_bytes'],
                   sealed_reader_source_or_production_files_modified=False,
                   fixture_module_constants_rebound=True,
                   actual_production_main_namespace_qualified=False,
                   scientific_data_checkpoint_model_TEST_outcome_access=False,
                   SSH_or_remote_compute=False, production_admission_granted=False,
                   scope='Fabricated local file-pin/control-flow qualification only; no real authority, normal execution provenance, production host/path/interpreter admission or scientific completion is established')
    with RECEIPT.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('FABRICATED_READER_FILE_PATH_QUALIFICATION_PASS cases=' + str(len(cases)), flush=True)


if __name__ == '__main__':
    main()

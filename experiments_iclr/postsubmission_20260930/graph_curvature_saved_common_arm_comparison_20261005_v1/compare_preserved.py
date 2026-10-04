"""One bounded CPU comparison of preserved installation witnesses, no models.

Import defines stdlib-only helpers. Explicit run(packet) verifies the authorized
project/route/UUID, then loads only saved tensor dictionaries on CPU. It performs
no forwards, preprocessing, optimizer updates, scoring or qualification.
"""
from pathlib import Path
from types import ModuleType
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback

PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ARMS = ('common_only', 'selected_graph_pair', 'selected_permuted_span', 'selected_random_span')


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(record):
    relative = Path(record['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative project descriptor required')
    path = PHASE / relative
    require(path.resolve() == path.absolute(), 'No unbound symlink input')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    require(size == record['bytes'] and digest.hexdigest() == record['sha256'], 'Bound file differs: ' + str(relative))
    return path


def module_from_source(record, name):
    path = verify(record)
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def compare(left, right, torch, comparator, name):
    try:
        comparator.exact_equal(left, right, torch, path=name)
        return dict(check=name, passed=True)
    except Exception as error:
        row = dict(check=name, passed=False, error_type=type(error).__name__, error=str(error))
        if hasattr(error, 'details'):
            row['difference'] = {key: value for key, value in error.details.items()
                if key not in ('expected_keys', 'actual_keys')}
        return row


def require_cpu(value, torch):
    if torch.is_tensor(value):
        require(value.device.type == 'cpu', 'Comparison must stay on CPU')
    elif isinstance(value, dict):
        for item in value.values():
            require_cpu(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_cpu(item, torch)


def run(packet):
    require(Path.cwd() == PHASE and os.environ.get('GNNM_SSH_DESTINATION') == ROUTE,
            'Authorized one-GPU project process required')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True, timeout=10).stdout.splitlines()
    require([row.strip() for row in rows if row.strip()] == [UUID], 'Sole authorized GPU UUID required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before Torch import')
    packet = Path(packet)
    require(packet.resolve() == packet.absolute() and packet.parent == PHASE, 'Exact exclusive project packet required')
    require(not (packet / 'CPU_RESULT.json').exists(), 'No repeat or result overwrite')
    binding = json.loads((packet / 'BINDINGS.json').read_text())
    wall, cpu = time.perf_counter(), time.process_time()
    report = dict(schema='graph-curvature-saved-common-arm-comparison-v1', status='STARTED',
        graph='Squirrel', seed=17, scope='Preserved installation snapshots and original custody receipts',
        CPU_only=True, model_construction=False, model_forwards=0, model_fits=0,
        optimizer_updates=0, preprocessing_runs=0, labels_or_predictive_scores_read=False,
        RNG_restored_or_consumed_for_scientific_execution=False, qualification_promoted=False,
        full_forward_relevant_equivalence_established=False,
        new_observation_of_original_live_returned_models=False, verified_files=[], checks=[], arms=[])
    try:
        comparator = module_from_source(binding['comparator'], 'saved_common_arm_exact_comparator')
        witness_module = module_from_source(binding['witness_source'], 'saved_common_arm_fingerprint_helpers')
        for record in binding['static_forward_source_files']:
            verify(record)
        report['verified_static_forward_source_files'] = binding['static_forward_source_files']
        files = {}
        for record in binding['saved_files']:
            name = Path(record['path']).name
            require(name not in files, 'Duplicate saved file descriptor')
            files[name] = verify(record)
            report['verified_files'].append(record)
        summary = json.loads((packet / 'TRAIN_ONLY_SELECTION_STRUCTURE.json').read_text())
        verify(summary['receipt'])  # Hash only: do not parse TRAIN loss values.
        for arm in ARMS:
            require(summary['selection'][arm]['pair'] is None
                and summary['selection'][arm]['pre_trial_state_returned'] is True,
                'The four compared arms must be saved common/null starts')
        custody = json.loads(files['SOURCE_CUSTODY_DIAGNOSTICS.json'].read_text())
        require(custody['diagnostic_only'] is True and custody['qualification_promoted'] is False,
                'Original diagnostic-only custody receipt required')
        source_hashes = {Path(record['path']).name: record['sha256']
            for record in binding['static_forward_source_files']}
        require(custody['capture_summary']['selector_source_sha256'] == source_hashes['selector.py']
            and custody['capture_summary']['driver_source_sha256'] == source_hashes['driver.py'],
            'Original live witness receipt must bind the assessed executed selector/driver')
        custody_arms = {row['arm']: row for row in custody['arms']}
        require(set(ARMS) <= set(custody_arms), 'Original custody arms absent')
        for arm in ARMS:
            failed = [row for row in custody_arms[arm]['checks'] if not row['passed']]
            require(len(failed) == 1
                and failed[0]['check'] == 'returned_full_state_equals_own_post_install_witness',
                'Preserve the original incidental container failure only')
            detail = failed[0]['exact_state_difference']
            require(detail['expected_type'] == 'dict' and detail['actual_type'] == 'OrderedDict'
                and detail['expected_keys'] == detail['actual_keys'], 'Original failure differs')
        import torch
        report['runtime'] = dict(python=sys.version, torch=torch.__version__, device='cpu', GPU_UUID=UUID)
        archive = torch.load(files['INSTALL_WITNESSES.pt'], map_location='cpu', weights_only=False)
        warm = torch.load(files['FRESH_NATIVE_WARM.pt'], map_location='cpu', weights_only=False)
        require_cpu(archive, torch)
        require_cpu(warm, torch)
        require(archive['schema'] == 'graph-curvature-install-witness-tensors-v1'
            and warm['schema'] == 'graph-init-native-warm-checkpoint-v1', 'Saved schemas differ')
        frozen = archive['frozen']
        require(frozen['backbone'] == 'polyformer_mono'
            and frozen['optimizer']['schema'] == 'named-aliased-adam-v1', 'Frozen prototype/optimizer differs')
        snapshots = {row['serial']: row for row in archive['installations']}
        require(len(snapshots) == len(archive['installations']), 'Witness serial collision')
        owner = snapshots[custody_arms['common_only']['witness_serial']]
        report['checks'].append(compare(frozen['rng'], warm['rng'], torch, comparator, 'frozen_RNG_equals_saved_native_warm_RNG'))
        warm_fp = json.loads(files['FRESH_WARM_FINGERPRINTS.json'].read_text())
        for name, key in (('model', 'logical_model'), ('optimizer', 'logical_optimizer'), ('rng', 'logical_RNG')):
            actual = witness_module.state_fingerprint(warm[name], torch)['sha256_logical_descriptor']
            report['checks'].append(dict(check='saved_warm_' + name + '_logical_fingerprint',
                passed=actual == warm_fp[key]['sha256_logical_descriptor'], sha256=actual))
        frozen_fp = {name: witness_module.state_fingerprint(frozen[name], torch)['sha256_logical_descriptor']
            for name in ('prototype_state', 'optimizer', 'rng', 'modes')}
        report['frozen_snapshot_logical_fingerprints'] = frozen_fp
        report['optimizer_saved_scope'] = 'One actual frozen named-Adam snapshot; per-arm returned optimizer equality is original live-custody evidence, not four independently saved optimizer objects'
        report['RNG_saved_scope'] = 'One actual selector-input RNG snapshot; original per-arm returns equal it in the preserved live-custody receipt'
        aliases = frozen['optimizer']['aliases']
        groups = frozen['optimizer']['groups']
        names = [name for group in groups for name in group['names']]
        require(len(names) == len(set(names)) == len(aliases)
            and set(names) == {row['name'] for row in aliases}
            == set(frozen['optimizer']['state']), 'Frozen optimizer coverage/aliases/groups differ')
        for row in aliases:
            require(row['name'] in row['aliases'] and isinstance(row['requires_grad'], bool), 'Invalid alias inventory')
            for alias in row['aliases']:
                require(alias in owner['installed_model_state']
                    and list(owner['installed_model_state'][alias].shape) == row['shape'], 'Saved alias shape/name absent')
                report['checks'].append(compare(owner['installed_model_state'][row['name']],
                    owner['installed_model_state'][alias], torch, comparator, 'registered_alias_value:' + alias))
            for field, value in frozen['optimizer']['state'][row['name']].items():
                require(field in ('step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'), 'Unexpected saved Adam field')
                if field != 'step':
                    require(torch.is_tensor(value) and list(value.shape) == row['shape'], 'Saved moment shape differs')
        report['optimizer_structure'] = dict(canonical_parameters=len(aliases), alias_names=sum(len(row['aliases']) for row in aliases),
            groups=len(groups), named_state_entries=len(frozen['optimizer']['state']),
            per_arm_live_parameter_links_saved=False, parameter_requires_grad_in_inventory=True)
        relevant = ('installed_head_equals_intended_source_slices', 'returned_head_equals_own_intended_source_slices',
            'non_head_state_equals_actual_frozen_prototype', 'optimizer_equals_actual_frozen_selector_optimizer',
            'RNG_equals_actual_selector_input', 'RNG_equals_original_warm_checkpoint',
            'modes_equal_actual_frozen_prototype', 'storage_disjoint_from_native_and_checkpoint',
            'storage_disjoint_from_every_previous_arm')
        for arm in ARMS:
            original = custody_arms[arm]
            installed = snapshots[original['witness_serial']]
            row = dict(arm=arm, witness_serial=installed['serial'],
                selector_reason=summary['selection'][arm]['reason'], checks=[], original_live_custody_checks={})
            report['arms'].append(row)
            require(installed['head'] == owner['head'] == 'head.R'
                and installed['backbone'] == 'polyformer_mono'
                and installed['matched_returned_arm'] in (None, arm), 'Saved arm identity differs')
            row['checks'].append(compare(dict(owner['installed_model_state']), dict(installed['installed_model_state']),
                torch, comparator, arm + ':stored_full_model_equals_common'))
            row['checks'].append(compare(owner['intended_slices'], installed['intended_slices'],
                torch, comparator, arm + ':stored_intended_head_equals_common'))
            row['checks'].append(compare(installed['intended_slices'], installed['installed_model_state']['head.R'],
                torch, comparator, arm + ':stored_install_equals_intended_head'))
            row['checks'].append(compare(frozen['modes'], installed['installed_modes'],
                torch, comparator, arm + ':stored_modes_equal_actual_frozen_prototype'))
            original_checks = {item['check']: item['passed'] for item in original['checks']}
            require(all(original_checks.get(name) is True for name in relevant), 'Original required live custody check failed')
            row['original_live_custody_checks'] = {name: original_checks[name] for name in relevant}
            row['passed'] = all(item['passed'] for item in row['checks'])
        inp = json.loads(files['INPUT_PREPROCESSING_FINGERPRINTS.json'].read_text())
        snapshot = json.loads(files['PREPROCESSING_SNAPSHOT_RECEIPT.json'].read_text())
        warm_input = warm_fp['input_preprocessing_fingerprints']
        descriptor = next(row for row in binding['saved_files'] if Path(row['path']).name == 'ACTUAL_PREPROCESSED_INPUT.pt')
        require(snapshot['sha256_file'] == descriptor['sha256'] and snapshot['storage_bytes'] == descriptor['bytes']
            and snapshot['teacher_input'] == inp['teacher_input'] == warm_input['teacher_input']
            and snapshot['teacher_edge_index'] == inp['teacher_edge_index'] == warm_input['teacher_edge_index'],
            'Exact captured input hash/logical-reference chain differs')
        report['captured_input_reference'] = dict(file=descriptor,
            teacher_input=inp['teacher_input'], teacher_edge_index=inp['teacher_edge_index'],
            tensor_payload_loaded=False, whole_saved_file_hash_verified=True,
            logical_tensor_fingerprints_recomputed=False,
            capture_before_original_warm=snapshot['actual_input_saved_before_prescribed_warm'])
        require(all(item['passed'] for item in report['checks']) and all(row['passed'] for row in report['arms']),
                'A preserved snapshot comparison failed')
        report.update(status='FOUR_PRESERVED_COMMON_START_SNAPSHOTS_EQUAL_LIMITED_SCOPE',
            stored_registered_state_equal=True, intended_head_intervention_absent=True,
            frozen_optimizer_RNG_modes_bound=True, full_trajectory_equivalence_claim=False,
            selector_version_closure_advice='Close this frozen selector as no intended selected-head intervention in required Squirrel17 block; do not spend thirty fits to rescue the all-seed mechanism screen',
            empirical_NLL_or_accuracy_contrast_claim=False)
    except Exception as error:
        report.update(status='FAILED_OR_UNRESOLVED_PRESERVED_SNAPSHOT_COMPARISON',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report.update(total_wall_seconds=time.perf_counter() - wall,
            total_process_cpu_seconds=time.process_time() - cpu,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        # Alias value checks are kept on the server but summarized for compact transport.
        alias_checks = [row for row in report['checks'] if row['check'].startswith('registered_alias_value:')]
        report['registered_alias_value_checks'] = dict(count=len(alias_checks),
            all_passed=all(row['passed'] for row in alias_checks))
        report['checks'] = [row for row in report['checks'] if not row['check'].startswith('registered_alias_value:')]
        with (packet / 'CPU_RESULT.json').open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    return report

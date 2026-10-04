"""Bounded saved-evidence regression; import/CLI load no tensors or models.

Explicit future callable: run_saved_regression(output). It admits only the
authorized project process, CPU tensor comparisons, and two preserved archives.
It cannot acquire a warm state, construct a native model, or promote qualification.
"""
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
import hashlib
import json
import os
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ARMS = ('common_only', 'fixed_first_graph_pair', 'selected_graph_pair',
        'selected_permuted_span', 'selected_random_span')


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_file(record):
    relative = Path(record['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Phase-relative bound input required')
    path = PHASE / relative
    require(path.resolve() == path.absolute(), 'Symlink input is not bound')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            digest.update(chunk)
    require(size == record['bytes'] and digest.hexdigest() == record['sha256'],
            'Bound file differs: ' + str(relative))
    return path


def verified_json(record):
    return json.loads(verify_file(record).read_text())


def load_comparator(record):
    path = verify_file(record)
    name = 'curvature_v3_comparator_for_saved_v4_regression'
    require(name not in sys.modules, 'Fresh bound comparator module required')
    module = ModuleType(name)
    module.__file__ = str(path)
    # The verified predecessor executes only stdlib imports and definitions here.
    # Its numerical/native entrypoint and __main__ status path are never called.
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def outcome(function):
    try:
        function()
        return dict(passed=True)
    except Exception as error:
        result = dict(passed=False, error_type=type(error).__name__, error=str(error))
        if hasattr(error, 'details'):
            result['exact_state_difference'] = error.details
        return result


def assert_cpu(value, torch):
    if torch.is_tensor(value):
        require(value.device.type == 'cpu', 'Saved regression must remain CPU-only')
    elif isinstance(value, dict):
        for part in value.values():
            assert_cpu(part, torch)
    elif isinstance(value, (list, tuple)):
        for part in value:
            assert_cpu(part, torch)


def status():
    return dict(status='SOURCE_ONLY_SAVED_EVIDENCE_REGRESSION_UNEXECUTED',
        callable='repair_regression.run_saved_regression',
        numerical_execution=False, native_model_construction=False,
        fresh_warm_acquisition=False, qualification_promoted=False,
        new_observation_of_original_live_returned_models=False)


def run_saved_regression(output):
    """One exclusive CPU-only saved comparison report; no retry or training."""
    source = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    records = json.loads((HERE / 'SAVED_EVIDENCE_BINDINGS.json').read_text())
    failure_summary = json.loads((HERE / 'V3_FAILURE_SUMMARY.json').read_text())
    comparator = load_comparator(source['comparison_runner'])
    # Same authorized route/sole UUID check as the bound predecessor. No remote
    # connection is opened by this source; the parent chooses where to call it.
    comparator.verify_runtime_location()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Saved regression must hide CUDA before importing Torch')
    out = Path(output).resolve()
    require(out.is_relative_to(PHASE) and out != PHASE,
            'Exclusive new output must remain inside project phase')
    out.mkdir(exist_ok=False)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    report = dict(schema='graph-curvature-container-repair-saved-regression-v1',
        status='STARTED', diagnostic_only=True, qualification_promoted=False,
        comparison_scope='Evidence-derived returns versus saved installation witnesses',
        new_observation_of_original_live_returned_models=False,
        native_model_construction=False, new_forward_or_optimizer_update=False,
        preprocessing_recomputed=False, selector_called=False,
        fresh_warm_acquisition=False, predictive_scores_read=False,
        saved_tensor_archives_loaded=[],
        verified_sources=[], verified_evidence=[], descriptor_only_evidence=[], arms=[])
    try:
        for name in ('original_witness', 'corrected_witness', 'predecessor_manifest', 'predecessor_seal'):
            verify_file(source[name])
            report['verified_sources'].append(dict(name=name, **source[name]))
        report['verified_sources'].append(dict(name='comparison_runner', **source['comparison_runner']))
        # This local custody receipt describes how source preparation derived its
        # compact summary. Runtime compares the directly hashed original server
        # receipts below; it need not copy the local monitor bundle to the server.
        report['descriptor_only_evidence'].append(failure_summary['observation'])
        chosen = {}
        selected_names = ('INSTALL_WITNESSES.pt', 'INDEPENDENT_RECONSTRUCTION_HEADS.pt',
                          'SOURCE_CUSTODY_DIAGNOSTICS.json', 'INDEPENDENT_RECONSTRUCTION_DIAGNOSTICS.json')
        for record in records['files']:
            name = Path(record['path']).name
            if name in selected_names:
                require(name not in chosen, 'Duplicate selected evidence file')
                chosen[name] = verify_file(record)
                report['verified_evidence'].append(record)
            else:
                # Warm/input files are bound descriptors, not loaded or rehashed
                # for a mapping-representation regression.
                report['descriptor_only_evidence'].append(record)
        require(set(chosen) == set(selected_names), 'Four exact saved comparison files required')
        custody = json.loads(chosen['SOURCE_CUSTODY_DIAGNOSTICS.json'].read_text())
        independent_receipt = json.loads(chosen['INDEPENDENT_RECONSTRUCTION_DIAGNOSTICS.json'].read_text())
        require(tuple(row['arm'] for row in independent_receipt['arms']) == ARMS
            and independent_receipt['passed'] is False
            and independent_receipt['qualification_promoted'] is False,
            'Original independent reconstruction receipt differs')
        require(tuple(row['arm'] for row in custody['arms']) == ARMS
            and custody['passed'] is False and custody['qualification_promoted'] is False
            and all(row['passed'] for row in custody['enumeration']),
            'Original five-arm custody failure receipt differs')
        serials = {}
        for original in custody['arms']:
            failures = [check for check in original['checks'] if not check['passed']]
            require(len(failures) == 1
                and failures[0]['check'] == 'returned_full_state_equals_own_post_install_witness',
                'Exactly the incidental container failure must be preserved')
            detail = failures[0]['exact_state_difference']
            require(detail['reason'] == 'state dictionary type or keys differ'
                and detail['expected_type'] == 'dict' and detail['actual_type'] == 'OrderedDict'
                and detail['expected_keys'] == detail['actual_keys'],
                'Original diagnostic must reject equal keys only on container representation')
            serials[original['arm']] = original['witness_serial']
        require(serials == {row['arm']: row['witness_serial'] for row in failure_summary['arms']},
                'Original custody serial mapping differs')
        import torch
        report['runtime'] = dict(python=sys.version, torch=torch.__version__,
            device='cpu', cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        archive = torch.load(chosen['INSTALL_WITNESSES.pt'], map_location='cpu', weights_only=False)
        heads = torch.load(chosen['INDEPENDENT_RECONSTRUCTION_HEADS.pt'], map_location='cpu', weights_only=False)
        report['saved_tensor_archives_loaded'] = ['INSTALL_WITNESSES.pt', 'INDEPENDENT_RECONSTRUCTION_HEADS.pt']
        assert_cpu(archive, torch)
        assert_cpu(heads, torch)
        require(archive['schema'] == 'graph-curvature-install-witness-tensors-v1'
            and heads['schema'] == 'graph-curvature-independent-heads-v1'
            and set(heads['heads']) == set(ARMS), 'Saved archive schema/arms differ')
        witnesses = {row['serial']: row for row in archive['installations']}
        require(len(witnesses) == len(archive['installations']), 'Witness serial collision')
        original_independent = {row['arm']: row for row in independent_receipt['arms']}
        for arm in ARMS:
            witness = witnesses[serials[arm]]
            head = witness['head']
            installed = witness['installed_model_state']
            intended = witness['intended_slices']
            require(type(installed) is dict and head in installed
                and witness['matched_returned_arm'] in (None, arm), 'Saved witness container or head differs')
            reconstructed = OrderedDict(archive['frozen']['prototype_state'])
            reconstructed[head] = intended
            row = dict(arm=arm, witness_serial=serials[arm], head=head,
                returned_state_kind='Evidence-derived OrderedDict; never original live object',
                negative_controls={})
            report['arms'].append(row)
            row['old_container_predicate'] = outcome(lambda: comparator.exact_equal(
                installed, reconstructed, torch, path='saved[' + repr(arm) + '].old'))
            require(row['old_container_predicate']['passed'] is False
                and row['old_container_predicate']['exact_state_difference']['reason']
                    == 'state dictionary type or keys differ', 'Original predicate must reject representation')
            row['normalized_saved_state'] = outcome(lambda: comparator.exact_equal(
                dict(installed), dict(reconstructed), torch, path='saved[' + repr(arm) + '].normalized'))
            require(row['normalized_saved_state']['passed'], 'Normalized saved state differs')
            require(intended.dtype == torch.float32 and intended.numel() > 1
                and bool(torch.isfinite(intended).all()), 'Finite nonempty saved FP32 head required')
            missing = dict(reconstructed)
            del missing[head]
            dtype = dict(reconstructed)
            dtype[head] = intended.to(torch.float64)
            shape = dict(reconstructed)
            shape[head] = intended.reshape(-1)[:-1].clone()
            ulp = dict(reconstructed)
            ulp[head] = intended.clone()
            flat = ulp[head].reshape(-1)
            flat[0] = torch.nextafter(flat[0], torch.full_like(flat[0], float('inf')))
            require(not torch.equal(ulp[head], intended) and bool(torch.isfinite(ulp[head]).all()),
                    'One-ULP negative control must be finite and different')
            for name, changed in (('missing_key', missing), ('head_dtype', dtype),
                                  ('head_shape', shape), ('head_one_FP32_ULP', ulp)):
                result = outcome(lambda changed=changed: comparator.exact_equal(
                    dict(installed), dict(changed), torch, path='saved[' + repr(arm) + '].' + name))
                row['negative_controls'][name] = result
                require(result['passed'] is False and result['error_type'] == 'ExactStateDifference',
                        'Normalization must retain negative control rejection: ' + name)
            row['independent_head_reconstruction'] = outcome(lambda: comparator.exact_equal(
                heads['heads'][arm], intended, torch, path='saved[' + repr(arm) + '].independent_head'))
            original_failures = [check for check in original_independent[arm]['checks'] if not check['passed']]
            if arm == 'fixed_first_graph_pair':
                result = row['independent_head_reconstruction']
                require(len(original_failures) == 1
                    and original_failures[0]['check'] == 'original_exact_model_tensor:' + head
                    and result['passed'] is False
                    and result['exact_state_difference']['reason'] == 'tensor values fail torch.equal'
                    and result['exact_state_difference']['max_absolute_difference']
                        == original_failures[0]['exact_state_difference']['max_absolute_difference'],
                    'Fixed-pair independent reconstruction failure must remain unchanged')
            else:
                require(not original_failures and row['independent_head_reconstruction']['passed'],
                        'Other independent reconstruction outcomes must remain unchanged')
        report.update(status='SAVED_REPRESENTATION_REGRESSION_PASSED_OTHER_FAILURE_PRESERVED',
            representation_regression_passed=True, five_normalized_saved_states_passed=True,
            independent_reconstruction_qualification=False,
            preserved_independent_failure='fixed_first_graph_pair',
            original_live_custody_receipt_changed=False,
            mean_logit_guard_executed=False, live_factor_probe_executed=False)
    except Exception as error:
        report.update(status='FAILED_OR_UNRESOLVED_SAVED_REPRESENTATION_REGRESSION',
            representation_regression_passed=False, error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc())
    finally:
        report.update(total_wall_seconds=time.perf_counter() - started_wall,
            total_process_cpu_seconds=time.process_time() - started_cpu,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == 'darwin' else 1024))
        with (out / 'SAVED_REPRESENTATION_REGRESSION.json').open('x') as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    return report


if __name__ == '__main__':
    print(json.dumps(status(), sort_keys=True))

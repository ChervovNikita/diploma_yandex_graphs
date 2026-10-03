"""Stdlib-only source custody/compile/import check; never admit evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    evaluator = load('prepared_closed_full40_evaluator_static_check', HERE/'evaluate.py')
    provenance, manifest_sha = evaluator.source_guard()
    compiled = []
    for path in [HERE/'evaluate.py', HERE/'verify_source.py'] + [PHASE/r['path'] for r in provenance['source_records'] if r['path'].endswith('.py')]:
        compile(path.read_text(), str(path), 'exec')
        compiled.append(str(path.relative_to(PHASE)))
    for record in provenance['optional_source_records']:
        evaluator.verify(record)
    # These modules have stdlib-only module bodies. No data/runtime/model/fit,
    # fixture, trace, selected state, calibration or evaluator function is called.
    common = load('prepared_closed_full40_common_static_check', PHASE/provenance['modules']['common'])
    sys.modules['common'] = common
    blocks = load('block_policy', PHASE/provenance['modules']['block_policy'])
    trainer = load('prepared_closed_full40_trainer_static_check', PHASE/provenance['modules']['trainer'])
    driver = load('prepared_closed_full40_driver_static_check', PHASE/provenance['modules']['driver'])
    core = load('prepared_closed_full40_core_static_check', PHASE/provenance['modules']['calibration_core'])
    decomposition = load('prepared_closed_full40_decomposition_static_check', PHASE/provenance['modules']['decomposition'])
    assert trainer.c is common and trainer.block_policy is blocks
    assert callable(driver.NativeEarlyStop) and callable(core.paired_summary) and callable(decomposition.decomposition)
    forbidden = sorted(name for name in sys.modules if name.split('.')[0] in ('torch', 'dgl', 'numpy'))
    assert not forbidden, 'Numerical runtime imported during static verification: '+str(forbidden)
    receipt = dict(schema='closed_full40_evaluation_static_source_receipt_v1', status='static_source_checks_passed',
        evaluation_manifest_sha256=manifest_sha, sources_compiled=compiled, source_hashes_verified=True,
        bare_common_and_block_policy_import_wiring_verified=True, calibration_core_unchanged=True,
        numerical_runtime_imported=False, numerical_or_fixture_execution=False,
        dataset_or_label_or_fitted_state_or_live_outcome_reads=False, remote_calls=False,
        evaluation_execution_authorized=False, numerical_evaluator_qualification=False)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

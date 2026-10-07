"""Disabled one-cell callable; delegates the sealed allocation and full driver."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLS_NAME = 'public_internal_be_allocation_controls_20261007_v1'
CONTROLS_SHA = 'e45c7c74a4e47864867275b41818d4b3a8f2ccf169288d18e0580b8deb4b43dd'
CONTROLS_PROGRAM_SHA = '5fb90f0cc6e40acc0308b0a851258fc0f9904cbb933f021d786e7c6e9649c473'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


method = _module(HERE / 'method.py', '_message_own_source_method')


def _verify(root, expected=None):
    root = Path(root).resolve(strict=True)
    if expected is not None:
        method.require(_sha(root / 'MANIFEST.json') == expected, 'Exact sealed delegate manifest required')
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        rel = Path(row['path'])
        method.require(not rel.is_absolute() and '..' not in rel.parts, 'Source path leaves sealed root')
        path = (root / rel).resolve(strict=True)
        method.require(path.is_relative_to(root) and path.is_file() and _sha(path) == row['sha256']
            and path.stat().st_size == row['bytes'], 'Sealed source changed: ' + row['path'])
    return root


def source_identity(admission=None):
    """Source-only description; does not admit execution or load numerical code."""
    return {'control_id': method.METHOD_ID, 'mode': 'message_own_control', 'policy': 'I', 'parent_policy': 'I',
        'base_constructor_arm': 'be_init', 'task': 'molhiv', 'lambda': .5, 'lambda_used_by_supervision': True,
        'lambda_learned': False, 'strength_adopted_by_source': False,
        'supervision_by_role': dict(method.SUPERVISION),
        'alignment_gradient_permission': 'unchanged .05 A on same audited internal phi INCLUDING message factors',
        'charged_reverse_calls_per_update': 3, 'additional_reverse_calls_over_I': 1,
        'extra_reverse_calls_over_original_shared_update': 2,
        'full_member_view_forwards_per_update': 8, 'Adam_transitions_per_update': 1,
        'full_source_training_updates': 25800, 'full_source_training_member_forwards': 206400,
        'full_source_training_reverse_collections': 77400, 'full_I_training_reverse_collections': 51600,
        'additional_full_training_reverse_collections_over_I': 25800,
        'full_source_training_Adam_transitions': 25800,
        'reverse_contract': 'Two exact I collections with second graph retained, then own+.05A on audited message tensor; one native Adam',
        'auxiliary_log_semantics': 'exact .05 A; scalar J+.05A is a diagnostic, not a global objective for this allocation',
        'selector': 'unchanged original full complete-VALID strict-first joint selector',
        'allocation_controls_manifest_sha256': CONTROLS_SHA,
        'message_control_method_sha256': _sha(HERE / 'method.py'), 'message_control_interface_sha256': _sha(__file__),
        'source_preparation_adopts_execution': False, 'runtime_verified_at_preparation': False,
        'exact_resume_supported': False, 'TEST_scoring': False, 'automatic_campaign': False,
        'cost_padding': False, 'equal_compute_with_I_claimed': False,
        'admission': None if admission is None else dict(admission),
        'admission_is_caller_assertion_not_verified_fixed18_evidence': True}


def run_complete(train, valid, output, *, admission=None, seed=6101, device='cpu'):
    """Disabled by default; one admitted original full MolHIV be_init cell only.

    Admission and source checks precede all data/runtime/output access. Like the
    unchanged full driver, explicit admitted calls must be serial per process.
    """
    admission = method.require_admission(admission)
    method.require(type(seed) is int, 'Integer seed required')
    _verify(HERE)
    root = _verify(HERE.parent / CONTROLS_NAME, CONTROLS_SHA)
    method.require(_sha(root / 'train.py') == CONTROLS_PROGRAM_SHA, 'Exact allocation full-driver delegate required')
    controls = _module(root / 'train.py', '_message_own_sealed_allocation_interface')
    original_adapter, original_identity = controls.method.PolicyAdapter, controls.control_identity

    def identity(policy, base_arm, lambda_value):
        method.require(policy == 'I' and base_arm == 'be_init' and lambda_value == .5, 'One fixed message-own cell required')
        base = original_identity(policy, base_arm, lambda_value)
        base.pop('G_scope', None)
        return {**base, **source_identity(admission), 'message_control_manifest_sha256': _sha(HERE / 'MANIFEST.json')}

    def adapter(session, auditor, *, policy, lambda_value):
        return method.MessageOwnAdapter(session, auditor, original_adapter,
            policy=policy, lambda_value=lambda_value, admission=admission)

    # In-memory changes affect only this freshly loaded delegate, never sealed
    # files or the active family. Its actual work check now requires THREE calls.
    controls.method.PolicyAdapter = adapter
    controls.method.REVERSE_CALLS = {**controls.method.REVERSE_CALLS, 'I': 3}
    controls.control_identity = identity
    return controls.run_complete('molhiv', 'I', 'be_init', train, valid, output, .5, seed=seed, device=device)

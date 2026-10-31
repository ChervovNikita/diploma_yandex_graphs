"""Small stdlib false-draft/request generator; does not execute the diagnostic."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import diagnostic_entry as entry

BASE = Path(__file__).resolve().parent
PHASE = BASE.parent


def request(h, admission_path, decision_path, admitted=False):
    root = entry.REMOTE/'photo_fd_arithmetic_diagnostic_v1'
    return {'schema': 'photo-fd-arithmetic-root-request-v1', 'root_admitted': admitted,
        'action': 'photo_fd_arithmetic_diagnostic', 'full_fit_admitted': False,
        'heldout_scoring_admitted': False, 'whole_cap_seconds': 3600,
        'entry_script': str(root/'diagnostic_entry.py'), 'diagnostic_admission': h.descriptor(admission_path),
        'root_decision': h.descriptor(decision_path),
        'allocation_route': h.descriptor(PHASE/'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'),
        'output': str(root/'diagnostic_v1'), 'receipt_directory': str(root/'root_receipts/diagnostic_v1'),
        'supervisor_directory': str(root/'supervision/diagnostic_v1_inner'),
        'outer_supervisor_directory': str(root/'supervision/diagnostic_v1_outer'),
        'local_launch_receipt': str(root/'supervision/diagnostic_v1_LAUNCH.json'), 'automatic_retry': False}


def prepare(output):
    h = entry.helpers()
    h.verify_sources()
    out = h.mirror(output); out.mkdir(parents=True, exist_ok=False)
    registry_path = PHASE/'graph_init_execution_root_v1/study_v1/GRAPH_INIT_ATTEMPT_REGISTRY.json'
    registry = h.read(registry_path)
    rows = [r for r in registry['attempts'] if r['phase'] == 'qualify' and r['output'].endswith('/Photo_seed17')]
    h.require(len(rows) == 1, 'Exact failed Photo17 registered attempt required')
    row = rows[0]; anchor = Path(registry['anchor_directory'])
    terminal_path = anchor/'terminals'/(row['key']+'.json')
    terminal = h.read(terminal_path)
    claim_path = h.bound(terminal['claim']); claim = h.read(claim_path)
    original_path = h.bound(claim['admission']); original = h.read(original_path)
    failure_path = Path(row['output'])/'FAILED_ATTEMPT.json'
    trace_path = Path(row['output'])/'interface_trace.jsonl'
    failure = h.read(failure_path)
    h.require(terminal['completed'] is False and failure['message'] == 'Actual predictive-function AD qualification failed' and
        claim['attempt'] == row and claim['attempt_registry'] == h.descriptor(registry_path) and
        original['context']['graph'] == 'Photo' and original['context']['seed'] == 17, 'Exact failed-state source identity required')
    previous = [h.descriptor(registry_path), h.descriptor(terminal_path), h.descriptor(claim_path),
        h.descriptor(original_path), h.descriptor(failure_path), h.descriptor(trace_path),
        h.descriptor(Path(row['output'])/'cost_trace.jsonl'),
        h.descriptor(PHASE/'graph_init_execution_continuation_v1/coordinator_run_v1/FAILED.json'),
        h.descriptor(PHASE/'graph_init_execution_root_v1/MANIFEST.json'),
        h.descriptor(PHASE/'graph_init_execution_root_v1/SEAL.json'),
        h.descriptor(PHASE/'graph_init_execution_root_v1/admission_support.py'),
        h.descriptor(PHASE/'protocols/bounded_run_v1.py'), h.descriptor(PHASE/'protocols/run_authorized_v2.py'),
        h.descriptor(PHASE/'protocols/run_logged.py'), h.descriptor(PHASE/'protocols/repo_env.sh')]
    ancestry = h.read(PHASE/h.R17_REL/'SOURCE_BINDINGS.json')['unchanged_method']
    previous.append({'path': str(entry.REMOTE/ancestry['path']), 'sha256': ancestry['sha256']})
    for record in previous:
        h.bound(record)
    admission = {'schema': 'photo-fd-arithmetic-diagnostic-admission-v1', 'execution_authorized': False,
        'context': original['context'], 'original_failed_admission': h.descriptor(original_path),
        'prior_failure': h.descriptor(failure_path), 'prior_attempt_terminal': h.descriptor(terminal_path),
        'prior_interface_trace': h.descriptor(trace_path), 'protected_metadata': previous,
        'modes': list(entry.MODES), 'directions': 3, 'direction_seed': 90017, 'epsilons': list(entry.EPSILONS),
        'thresholds': {'dual_rtol': 2e-4, 'logits_FD_rtol': .05, 'CE_FD_rtol': .05, 'CE_denominator_floor': 1e-3},
        'full_FP64_model_authorized': False, 'original_gate_stays_failed': True, 'retry_authorized': False,
        'source_scope': 'Exact original failed Photo17cfg0 full graph; TRAIN only; two native disposable updates; unchanged Adam/K1/K4 prefix and original failed gate required.',
        'interpretation': 'A reproduces original FP32 scalar cancellation. B isolates per-example FP32 CE followed by FP64 mean/subtraction. C also changes CE softmax/logsumexp arithmetic on identical FP32 logits. Report all modes; no best-mode gate reversal or tolerance/epsilon/seed search.'}
    h.write(out/'ADMISSION_DRAFT.json', admission)
    root = entry.REMOTE/'photo_fd_arithmetic_diagnostic_v1'
    decision = {'schema': 'photo-fd-arithmetic-root-decision-v1', 'approved': False,
        'approved_by': '', 'approved_utc': None, 'diagnostic_execution_authorized': False,
        'source_review_accepted': False, 'source_review': {'path': None, 'sha256': None},
        'source_manifest': {'path': str(root/'MANIFEST.json'), 'sha256': None},
        'source_seal': {'path': str(root/'SEAL.json'), 'sha256': None},
        'admission_draft': h.descriptor(out/'ADMISSION_DRAFT.json'),
        'retry_or_registry_repair_authorized': False, 'fit_or_final_labels_authorized': False,
        'whole_cap_seconds': 3600, 'scope': 'One separately bounded diagnostic only; no original gate reversal, registry retry, useful fit, final labels or precision amendment approval.'}
    if (BASE/'SEAL.json').is_file():
        decision['source_manifest'] = h.descriptor(BASE/'MANIFEST.json')
        decision['source_seal'] = h.descriptor(BASE/'SEAL.json')
    h.write(out/'ROOT_DECISION_TEMPLATE.json', decision)
    h.write(out/'ROOT_REQUEST_DRAFT.json', request(h, out/'ADMISSION_DRAFT.json', out/'ROOT_DECISION_TEMPLATE.json'))
    return out


def make_request(decision_path, output):
    h = entry.helpers()
    decision = h.read(decision_path)
    h.require(decision['approved'] is True and decision['diagnostic_execution_authorized'] is True and
        decision['source_review_accepted'] is True and decision['approved_by'].strip() and decision['approved_utc'] and
        decision['whole_cap_seconds'] == 3600 and decision['retry_or_registry_repair_authorized'] is False and
        decision['fit_or_final_labels_authorized'] is False, 'Separate root diagnostic approval required')
    entry.own_guard(h, decision['source_manifest'], decision['source_seal']); h.bound(decision['source_review'])
    draft = h.read(h.bound(decision['admission_draft']))
    h.require(draft['execution_authorized'] is False, 'Original unapproved diagnostic draft required')
    admission = copy.deepcopy(draft); admission['execution_authorized'] = True
    out = h.mirror(output); out.mkdir(parents=True, exist_ok=False)
    h.write(out/'ADMISSION.json', admission)
    h.write(out/'ROOT_REQUEST.json', request(h, out/'ADMISSION.json', decision_path, True))
    print(str(out/'ROOT_REQUEST.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('make-request'); p.add_argument('--decision', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(str(prepare(args.output)))
    else:
        make_request(args.decision, args.output)


if __name__ == '__main__':
    main()

"""Add GPU-specific admission to the existing single-supervisor host launcher.

This changes scheduling and startup only. Apply to an owned runtime copy;
never alter an already frozen plan's runtime in place.
"""
from pathlib import Path

HELPERS = '''
def enabled_workers(output, host, digest, observed):
    """Read atomic GPU requests; an absent file retains legacy host startup."""
    available = {host + '/gpu' + str(row['index']) for row in observed}
    path = output / (host + '.gpu-admission.json')
    if not path.exists():
        return available
    request = json.loads(path.read_text())
    if request.get('host') != host or request.get('plan_hash') != digest:
        raise ValueError('GPU admission belongs to a different host or plan')
    enabled = request.get('enabled_workers')
    if (not isinstance(enabled, list) or not enabled or len(enabled) != len(set(enabled))
            or any(not isinstance(worker, str) or worker not in available for worker in enabled)):
        raise ValueError('GPU admission requests unavailable hardware')
    return set(enabled)


def request_gpu_admission(output, host, digest, observed, gpu_index):
    """Enable requested GPUs under the caller's exclusive startup lock."""
    available = {host + '/gpu' + str(row['index']) for row in observed}
    requested = available if gpu_index is None else {host + '/gpu' + str(gpu_index)}
    if not requested <= available:
        raise ValueError('Requested GPU index is absent from verified hardware')
    path = output / (host + '.gpu-admission.json')
    # A fresh single-GPU request starts only that GPU. Existing requests accumulate.
    enabled = enabled_workers(output, host, digest, observed) if path.exists() else set()
    enabled |= requested
    request = {'schema': 'go-bound-host-gpu-admission-v1', 'host': host,
               'plan_hash': digest, 'enabled_workers': sorted(enabled), 'slots_per_gpu': 2}
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        stream.write(canonical(request) + '\\n')
    temporary.replace(path)
    return sorted(enabled)


'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected startup source; do not silently patch: ' + old[:100])
    return source.replace(old, new)


def patched_launcher(source):
    source = replace_once(source, 'def schedule_ready(', HELPERS + 'def schedule_ready(')
    source = replace_once(source,
        'if worker.split("/")[0] != host or states[key] != "registered" or key in active_keys:',
        'if worker.split("/")[0] != host or worker not in capacities or states[key] != "registered" or key in active_keys:')
    source = replace_once(source,
        '                                                 owned_gpu_usage(host, current, active))',
        '                                                 owned_gpu_usage(host, current, active))\n'
        '                enabled = enabled_workers(output, host, digest, current)\n'
        '                capacities = {worker: value for worker, value in capacities.items() if worker in enabled}')
    source = replace_once(source,
        'def detach_host(root, plan_path, approval_path, host, python):',
        'def detach_host(root, plan_path, approval_path, host, python, *, gpu_index=None):')
    source = replace_once(source,
        '    host_check(host, plan)\n    repository, policy = receipt_authentication(root, plan)',
        '    observed = host_check(host, plan)\n'
        '    if gpu_index is not None and (type(gpu_index) is not int or gpu_index not in {r["index"] for r in observed}):\n'
        '        raise ValueError("Requested GPU index is absent from verified hardware")\n'
        '    repository, policy = receipt_authentication(root, plan)')
    source = replace_once(source,
        "            if alive['supervisor'] and not alive['rendezvous'] and all_global_gates_closed(root, plan, digest):\n",
        "            if alive['supervisor'] and not alive['rendezvous'] and all_global_gates_closed(root, plan, digest):\n"
        '                request_gpu_admission(output, host, digest, observed, gpu_index)\n')
    source = replace_once(source,
        "            if all(alive.values()):\n                return {'status': 'already_running', **previous,",
        "            if all(alive.values()):\n"
        '                request_gpu_admission(output, host, digest, observed, gpu_index)\n'
        "                return {'status': 'already_running', **previous,")
    source = replace_once(source,
        '        # Check the actual runtime scripts before either detached process starts.',
        '        request_gpu_admission(output, host, digest, observed, gpu_index)\n'
        '        # Check the actual runtime scripts before either detached process starts.')
    source = replace_once(source,
        '    parser.add_argument("--python")',
        '    parser.add_argument("--python")\n'
        '    parser.add_argument("--gpu-index", type=int, help="Enable one GPU in the single host supervisor")')
    source = replace_once(source,
        '    args = parser.parse_args()\n    if args.host',
        '    args = parser.parse_args()\n'
        '    if args.gpu_index is not None and not args.detach:\n'
        '        parser.error("--gpu-index requires --detach")\n'
        '    if args.host')
    source = replace_once(source,
        'detach_host(root, args.plan, args.approval, args.host, python)',
        'detach_host(root, args.plan, args.approval, args.host, python, gpu_index=args.gpu_index)')
    return source


def patched_shell(source):
    source = replace_once(source,
        'gnnm_approval=${3:-"$gnnm_package/configs/APPROVAL.json"}',
        'gnnm_approval=${3:-"$gnnm_package/configs/APPROVAL.json"}\n'
        'gnnm_gpu_args=()\n'
        'if [[ $# -ge 4 ]]; then gnnm_gpu_args=(--gpu-index "$4"); fi')
    return replace_once(source,
        '--host "$gnnm_host" --python "$gnnm_python" --detach',
        '--host "$gnnm_host" --python "$gnnm_python" --detach "${gnnm_gpu_args[@]}"')


def apply(runtime):
    runtime = Path(runtime)
    launcher = runtime / 'launcher.py'
    shell = runtime / 'start_massive.sh'
    launcher.write_text(patched_launcher(launcher.read_text()))
    shell.write_text(patched_shell(shell.read_text()))

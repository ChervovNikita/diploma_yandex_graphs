"""Finite sequential root coordinator. Source-only until separately signed and run.

Uses the unchanged launcher and whole-process supervisors. Uploads only explicitly
enumerated text source/metadata. Fetches text receipts only, never model/array files.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import traceback

sys.dont_write_bytecode = True
from continuation_support import (HERE, PHASE, REMOTE_PHASE, REMOTE_REPO, LOGIN, TEXT_EXTENSIONS,
    require, confined, mirror, remote, read, write, descriptor, bound, source_descriptors,
    verify_sources, own_sources, verify_own_sources, deployment_ancillary, registry_guard,
    decision_guard, scan_state, completed_phase)
from build_continuation import make_admission, next_row

SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
SCP = ['scp', '-P', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes']


def utc():
    return datetime.now(timezone.utc).isoformat()


def ensure_upload(records):
    """Explicit files only; existing remote files must be byte-identical; no deletion."""
    for record in records:
        bound(record)
    code = '''import hashlib,json,pathlib,sys
base=pathlib.Path(sys.argv[1]); rows=json.load(sys.stdin); missing=[]
for row in rows:
 p=pathlib.Path(row['path'])
 if not p.is_absolute() or not p.is_relative_to(base) or '..' in p.parts: raise ValueError('confined source/metadata required')
 cursor=base
 for part in p.relative_to(base).parts:
  cursor/=part
  if cursor.is_symlink(): raise ValueError('symlink forbidden')
 if p.exists():
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']: raise ValueError('existing immutable remote file differs')
 else:
  p.parent.mkdir(parents=True,exist_ok=True); missing.append(str(p))
print(json.dumps(missing))'''
    command = shlex.join([str(REMOTE_REPO/'.venv/bin/python'), '-c', code, str(REMOTE_PHASE)])
    result = subprocess.run([*SSH, command], input=json.dumps(records), text=True, capture_output=True,
                            check=True, timeout=60)
    missing = json.loads(result.stdout)
    require(set(missing) <= {r['path'] for r in records}, 'Unexpected upload identity')
    for path in missing:
        subprocess.run([*SCP, str(mirror(path)), LOGIN+':'+path], capture_output=True, check=True, timeout=60)
    verify = subprocess.run([*SSH, command], input=json.dumps(records), text=True, capture_output=True,
                            check=True, timeout=60)
    require(json.loads(verify.stdout) == [], 'Remote deployment verification incomplete')
    return {'explicit_files': records, 'new_files_uploaded': missing, 'existing_files_unchanged': True}


def prepare_launch_parents(request):
    """Create local receipt and remote outer/inner parents before any launcher call."""
    local_receipt = mirror(request['local_launch_receipt'])
    require(not local_receipt.exists(), 'Single-use local launcher receipt required')
    local_receipt.parent.mkdir(parents=True, exist_ok=True)
    parents = sorted({str(Path(request[k]).parent) for k in
        ('supervisor_directory', 'outer_supervisor_directory', 'local_launch_receipt')})
    require(all(Path(p).is_relative_to(REMOTE_PHASE/HERE.name) for p in parents),
            'Remote launch parents must belong to current wrapper')
    code = "import pathlib,sys; base=pathlib.Path(sys.argv[1]); paths=[pathlib.Path(x) for x in sys.argv[2:]]; "+\
           "assert all(p.is_absolute() and p.is_relative_to(base) and '..' not in p.parts for p in paths); "+\
           "assert all(not q.is_symlink() for p in paths for q in [p,*p.parents] if q.is_relative_to(base)); "+\
           "[p.mkdir(parents=True,exist_ok=True) for p in paths]"
    command = shlex.join([str(REMOTE_REPO/'.venv/bin/python'), '-c', code, str(REMOTE_PHASE/HERE.name), *parents])
    subprocess.run([*SSH, command], capture_output=True, text=True, check=True, timeout=60)
    return {'UTC': utc(), 'local_receipt_parent': str(local_receipt.parent),
            'remote_supervision_parents': parents, 'scientific_execution': False,
            'original_diagnostic_receipt_parent_failure_preserved': True}


def bootstrap(receipt):
    verify_sources()
    manifest, seal = own_sources()
    verify_own_sources(manifest, seal)
    from lineage_support import preflight_remote_lineage
    preflight = preflight_remote_lineage()
    records = [manifest, seal]
    for item in read(bound(manifest))['payload']:
        records.append(descriptor(HERE/item['path']))
    # Deploy only explicit sealed text/source dependencies and protected lineage.
    for source_manifest, source_seal in zip(source_descriptors()[::2], source_descriptors()[1::2]):
        records += [source_manifest, source_seal]
        source_root = bound(source_manifest).parent
        for item in read(source_root/'MANIFEST.json')['payload']:
            records.append(descriptor(source_root/item['path']))
    from build_continuation import protected as protected_metadata
    records += protected_metadata()
    records += deployment_ancillary()
    unique = {r['path']: r for r in records}
    value = ensure_upload([unique[k] for k in sorted(unique)])
    mirror(receipt).parent.mkdir(parents=True, exist_ok=True)
    write(mirror(receipt), {'schema': 'graph-init-continuation-explicit-deployment-receipt-v1',
        'UTC': utc(), 'scientific_execution': False, 'lineage_preflight':preflight, **value})


def export_metadata(registry_path, run_root=None):
    """Remote metadata stream, including failures. Does not evaluate scientific code."""
    require(PHASE == REMOTE_PHASE, 'Metadata exporter runs only in authorized remote phase')
    record = descriptor(registry_path)
    registry = registry_guard(record)
    names = set()
    def add(path):
        path = confined(path)
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            names.add(path)
    add(registry_path)
    add(bound(registry['request'])); add(bound(registry['lineage_authorization']))
    anchor = Path(registry['anchor_directory'])
    for row in registry['attempts']:
        add(anchor/'claims'/(row['key']+'.json')); add(anchor/'terminals'/(row['key']+'.json'))
        out = confined(row['output'])
        if out.is_dir():
            for path in out.iterdir():
                add(path)  # Only .json/.jsonl/etc; arrays/models remain remote.
            freeze_path = out/'FREEZE.json'
            if freeze_path.is_file():
                freeze = read(freeze_path)
                add(bound(freeze['admission']))
                for item in freeze['payload']:
                    rel = Path(item['path'])
                    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe phase receipt')
                    if rel.suffix in TEXT_EXTENSIONS:
                        add(out/rel)
    if run_root:
        root = confined(run_root)
        require(root.is_relative_to(PHASE/HERE.name) and
                not root.is_relative_to(anchor), 'Exact continuation receipt root required')
        if root.is_dir():
            for path in root.rglob('*.json'):
                add(path)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for path in sorted(names):
            archive.add(path, arcname=str(path.relative_to(PHASE)), recursive=False)


def sync_metadata(registry_path, run_root=None):
    argv = [str(REMOTE_REPO/'.venv/bin/python'), str(REMOTE_PHASE/HERE.name/'finite_coordinator.py'),
            'export-metadata', '--registry', remote(registry_path)]
    if run_root:
        argv += ['--run-root', remote(run_root)]
    command = 'cd '+shlex.quote(str(REMOTE_REPO))+' && '+shlex.join(argv)
    result = subprocess.run([*SSH, command], capture_output=True, check=True, timeout=60)
    imported = []
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for member in archive:
            rel = Path(member.name)
            require(member.isfile() and not rel.is_absolute() and '..' not in rel.parts and
                    rel.suffix in TEXT_EXTENSIONS, 'Only confined text metadata may be received')
            path = confined(PHASE/rel)
            data = archive.extractfile(member).read()
            if path.exists():
                require(path.read_bytes() == data, 'Existing immutable local metadata differs: '+str(path))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
            imported.append({'path': str(REMOTE_PHASE/rel), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    return imported


def run(decision_path, output):
    require(PHASE != REMOTE_PHASE, 'Coordinator runs locally; remote scientific children use the entry')
    decision, registry, plan = decision_guard(decision_path)
    out = mirror(output)
    require(remote(out) == decision['coordinator_run_root'], 'Exact signed coordinator run identity required')
    out.mkdir(parents=True, exist_ok=False)
    (out/'requests').mkdir(); (out/'supervision').mkdir()
    started = time.monotonic()
    write(out/'START.json', {'schema': 'graph-init-finite-coordinator-start-v1', 'UTC': utc(),
        'root_decision': descriptor(decision_path), 'attempt_registry': decision['attempt_registry'],
        'finite_plan_sha256': decision['finite_plan_sha256'], 'attempt_count': 72, 'automatic_retry': False})
    write(out/'PLAN.json', plan)
    try:
        # Root reviews and explicitly deploys the source packet before run. Verify remote decision/audits byte identity.
        deployment = ensure_upload([descriptor(decision_path), decision['independent_R17_source_audit'],
                                    decision['continuation_source_review']])
        write(out/'DECISION_DEPLOYMENT.json', deployment)
        sync_metadata(decision['attempt_registry']['path'], decision['coordinator_run_root'])
        for planned in plan:
            decision, registry, current_plan = decision_guard(decision_path)
            next_attempt = next_row(decision, registry, current_plan)
            require(next_attempt == planned, 'Only exact next signed attempt may launch')
            key = planned['key']
            request_dir = out/'requests'/key
            request = make_admission(decision_path, key, request_dir)
            transfer = ensure_upload([descriptor(request_dir/'ADMISSION.json'), descriptor(request_dir/'REQUEST.json')])
            write(out/(key+'_UPLOAD.json'), transfer)
            # Write a local exclusive launch claim before SSH. A transport failure is retained and never retried.
            write(out/(key+'_LAUNCH_CLAIM.json'), {'UTC': utc(), 'attempt': planned,
                'request': descriptor(request_dir/'REQUEST.json'), 'automatic_retry': False})
            argv = [sys.executable, str(PHASE/'protocols/launch_modern_root_v1.py'),
                '--request', str(request_dir.relative_to(PHASE)/'REQUEST.json'),
                '--outer', str(Path(request['outer_supervisor_directory']).relative_to(REMOTE_PHASE)),
                '--inner', str(Path(request['supervisor_directory']).relative_to(REMOTE_PHASE)),
                '--receipt', str(Path(request['local_launch_receipt']).relative_to(REMOTE_PHASE))]
            write(out/(key+'_LAUNCH_PARENTS.json'), prepare_launch_parents(request))
            phase_start = time.monotonic()
            child = subprocess.run(argv, capture_output=True, text=True, check=False,
                                   timeout=planned['whole_cap_seconds']+180)
            write(out/(key+'_LAUNCH_RESULT.json'), {'UTC': utc(), 'attempt': planned, 'exit_code': child.returncode,
                'local_launch_seconds': time.monotonic()-phase_start, 'stdout': child.stdout, 'stderr': child.stderr,
                'automatic_retry': False})
            # Fetch failed evidence too. No numerical retry occurs if retrieval fails.
            imported = sync_metadata(decision['attempt_registry']['path'], decision['coordinator_run_root'])
            write(out/(key+'_METADATA_FETCH.json'), {'UTC': utc(), 'files': imported})
            require(child.returncode == 0, 'Whole supervised phase failed; finite study blocked')
            outer = read(mirror(Path(request['outer_supervisor_directory'])/'TERMINAL.json'))
            root = read(mirror(Path(request['receipt_directory'])/'TERMINAL.json'))
            require(outer['complete'] is True and outer['within_whole_cap'] is True and
                    outer['root_request_unchanged'] is True and root['completed'] is True,
                    'Whole/root successful terminals required before next admission')
            canonical = [r for r in registry['attempts'] if r['key'] == key][0]
            _, terminal, _ = completed_phase(decision['attempt_registry'], registry, canonical)
            write(out/(key+'_COMPLETED.json'), {'UTC': utc(), 'phase_terminal': terminal,
                'whole_terminal': descriptor(mirror(Path(request['outer_supervisor_directory'])/'TERMINAL.json')),
                'whole_supervised_seconds': outer['whole_supervised_seconds'], 'whole_cap_seconds': planned['whole_cap_seconds'],
                'whole_process_costs_charged': True})
        decision, registry, _ = decision_guard(decision_path)
        completed = scan_state(decision['attempt_registry'], registry)
        require(len(completed) == 72 and next_row(decision, registry, plan) is None, 'All72source attempts must close')
        write(out/'COMPLETED.json', {'schema': 'graph-init-finite-coordinator-terminal-v1', 'UTC': utc(),
            'completed': True, 'remaining_attempts': 72, 'registered_successful_terminals': 72,
            'seconds': time.monotonic()-started, 'compare_report_or_final_labels_executed': False,
            'automatic_retry': False, 'scientific_result_assessment_still_root_required': True})
    except BaseException as error:
        write(out/'FAILED.json', {'schema': 'graph-init-finite-coordinator-failure-v1', 'UTC': utc(),
            'error_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
            'seconds': time.monotonic()-started, 'automatic_retry': False,
            'this_coordinator_run_blocked': True, 'closure_blocked': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bootstrap')
    p.add_argument('--receipt', type=Path, required=True)
    for command in ('sync-metadata', 'export-metadata'):
        p = sub.add_parser(command)
        p.add_argument('--registry', type=Path, required=True)
        p.add_argument('--run-root', type=Path)
    p = sub.add_parser('run')
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'bootstrap':
        bootstrap(args.receipt)
    elif args.command == 'export-metadata':
        export_metadata(args.registry, args.run_root)
    elif args.command == 'sync-metadata':
        value = sync_metadata(args.registry, args.run_root)
        print(json.dumps({'metadata_files_received': len(value)}))
    else:
        run(args.decision, args.output)


if __name__ == '__main__':
    main()

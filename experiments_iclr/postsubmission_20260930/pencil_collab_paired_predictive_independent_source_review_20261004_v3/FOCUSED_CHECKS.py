"""Independent source comparison and synthetic sampler checks; stdlib only.

This script imports no candidate module and reads no live proc or payload data.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
CANDIDATE = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v3'
PARENT = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_packet(root, expected_manifest, expected_seal):
    must(digest(root / 'MANIFEST.json') == expected_manifest, 'Manifest binding')
    must(digest(root / 'SEAL.json') == expected_seal, 'Seal binding')
    manifest = load(root / 'MANIFEST.json')
    must(load(root / 'SEAL.json')['manifest_sha256'] == expected_manifest, 'Seal manifest link')
    rows = manifest['files']
    must(len(rows) == len({r['path'] for r in rows}), 'Duplicate manifest path')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    must(actual == {r['path'] for r in rows} | {'MANIFEST.json', 'SEAL.json'}, 'Exact packet file set')
    for row in rows:
        path = root / row['path']
        must(path.resolve().is_relative_to(root) and not path.is_symlink(), 'Packet path escaped')
        must(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Payload binding ' + row['path'])
    return rows


def stat_record(pid, state='R', group=500, session=500, start=9001, pages=0,
                comm='loader (nested) with spaces', page_fields=52):
    # Linux stat field 3 is list index 0; starttime 22 -> 19; rss 24 -> 21.
    fields = ['0'] * (page_fields - 2)
    fields[0] = state
    fields[2], fields[3], fields[19], fields[21] = map(str, (group, session, start, pages))
    return f'{pid} ({comm}) ' + ' '.join(fields)


def synthetic_sampler_checks():
    tree = ast.parse((CANDIDATE / 'supervise.py').read_text())
    names = {'parse_proc_stat_rss', 'process_identity_with_rss', 'members_of_session'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    must(len(functions) == 3, 'Exact sampler helpers')
    snapshots, reads = {}, []

    class SyntheticPath:
        def __init__(self, value): self.value = str(value)
        @property
        def name(self): return self.value.rsplit('/', 1)[-1]
        def __truediv__(self, suffix): return SyntheticPath(self.value + '/' + str(suffix))
        def iterdir(self):
            must(self.value == '/proc', 'Only synthetic proc inventory')
            return [SyntheticPath('/proc/' + name) for name in ('500', '501', '502', '503', '504', 'self')]
        def read_text(self):
            must(self.value.endswith('/stat'), 'No separate status observation')
            reads.append(self.value)
            if self.value not in snapshots:
                raise FileNotFoundError(self.value)
            return snapshots[self.value]

    class SyntheticOS:
        @staticmethod
        def sysconf(name):
            must(name == 'SC_PAGE_SIZE', 'Runtime page size key')
            return 4096

    namespace = {'require': must, 'Path': SyntheticPath, 'os': SyntheticOS}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<review synthetic sampler>', 'exec'), namespace)
    parse = namespace['parse_proc_stat_rss']
    accepted = []
    for state, pages, page_size in [('R', 0, 4096), ('S', 0, 4096), ('Z', 0, 4096),
                                    ('S', 17, 4096), ('D', 23, 65536)]:
        row = parse(stat_record(501, state=state, pages=pages), 501, page_size)
        must(row == dict(pid=501, start_ticks=9001, session=500, group=500,
                         state=state, RSS_bytes=pages * page_size), 'Identity and RSS indexing')
        accepted.append(dict(state=state, rss_pages=pages, page_size=page_size, RSS_bytes=row['RSS_bytes']))
    # A comm containing a newline/parentheses still ends at the final closing paren.
    must(parse(stat_record(501, comm='weird ) ( name\n'), 501, 4096)['start_ticks'] == 9001,
         'Linux comm envelope')
    rejected = []
    cases = [
        ('negative_rss', stat_record(501, pages=-1), 501, 4096),
        ('mismatched_pid', stat_record(501), 502, 4096),
        ('truncated', '501 (loader) R 0 500', 501, 4096),
        ('invalid_state', stat_record(501, state='?'), 501, 4096),
        ('missing_comm', '501 R 0 500', 501, 4096),
        ('invalid_page_size', stat_record(501), 501, 0),
        ('nonnumeric_rss', stat_record(501), 501, 4096),
    ]
    # Explicitly corrupt field24 in the last case, rather than an unused tail field.
    prefix, tail = cases[-1][1].rsplit(') ', 1)
    fields = tail.split(); fields[21] = 'unknown'
    cases[-1] = ('nonnumeric_rss', prefix + ') ' + ' '.join(fields), 501, 4096)
    for label, raw, pid, page_size in cases:
        try:
            parse(raw, pid, page_size)
        except (AssertionError, ValueError):
            rejected.append(label)
        else:
            raise AssertionError('Malformed case accepted ' + label)
    snapshots.update({
        '/proc/500/stat': stat_record(500, state='Z'),
        '/proc/501/stat': stat_record(501, state='R'),
        '/proc/502/stat': stat_record(502, state='S', pages=17),
        '/proc/503/stat': stat_record(503, state='S', group=600, session=600, pages=3),
    })
    members = namespace['members_of_session'](500)
    must([x['pid'] for x in members] == [500, 501, 502], 'Same-session membership/disappearance')
    must(sum(x['RSS_bytes'] for x in members) == 17 * 4096, 'Session sum includes RSS0')
    must(reads == [f'/proc/{pid}/stat' for pid in (500, 501, 502, 503, 504)], 'One stat read per listed task')
    snapshots['/proc/502/stat'] = stat_record(502, group=600, pages=17)
    changed = namespace['members_of_session'](500)
    must(any(x['pid'] == 502 and x['group'] == 600 for x in changed), 'Changed owned group remains visible')
    return dict(accepted=accepted, malformed_rejected=rejected, unusual_comm_parsed=True,
                disappearance_omitted=True, one_stat_read_per_listed_task=True,
                session_filter_preserved=True, changed_owned_group_visible=True,
                live_proc_read=False, processes_created=False, signals_sent=False)


def run():
    candidate_rows = verify_packet(CANDIDATE,
        '45f84078f57fb788e14302aaedc37a98388ca7a120a8ff52d1a7c0d70a624cea',
        'a33f3c51e7b539900708435b711b3f89589ecfef78d1d49111b4d4a95a988bd1')
    parent_rows = verify_packet(PARENT,
        'c937462c776f48ef3ce84d428ede3f3d5ccc4837c4874d54bd8961638a74bbc4',
        '6fcf6d2992f51c638b57e7d1ec462cc6b2256ca941d9f212ec2016fd323029c4')
    changed = sorted(r['path'] for r in parent_rows
                     if (PARENT / r['path']).read_bytes() != (CANDIDATE / r['path']).read_bytes())
    expected = sorted(['AUTHOR_SOURCE_CHECK.json', 'INPUT_BINDINGS.json', 'PLAN.json',
                       'PROPOSED_COMMAND.json', 'README.md', 'ROOT_RELEASE_TEMPLATE.json',
                       'static_check.py', 'supervise.py'])
    must(changed == expected, 'Declared inherited payload delta')
    before = ast.parse((PARENT / 'supervise.py').read_text())
    after = ast.parse((CANDIDATE / 'supervise.py').read_text())
    old_functions = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    must(set(new_functions) - set(old_functions) == {'parse_proc_stat_rss', 'process_identity_with_rss'}, 'Only new sampler helpers')
    changed_functions = sorted(name for name in old_functions
                               if ast.dump(old_functions[name]) != ast.dump(new_functions[name]))
    must(changed_functions == ['members_of_session'], 'Only existing sampler function changes')
    normalized = copy.deepcopy(after)
    normalized.body = [copy.deepcopy(old_functions[n.name]) if isinstance(n, ast.FunctionDef) and n.name == 'members_of_session' else n
                       for n in normalized.body if not (isinstance(n, ast.FunctionDef) and n.name in {'parse_proc_stat_rss', 'process_identity_with_rss'})]
    must(ast.dump(normalized) == ast.dump(before), 'Complete supervisor AST apart from sampler delta')
    old_plan, new_plan = load(PARENT / 'PLAN.json'), load(CANDIDATE / 'PLAN.json')
    must(set(new_plan) - set(old_plan) == {'v2_failure_metadata', 'supervisor_RSS_sampling'}, 'Only declared plan additions')
    normalized_plan = copy.deepcopy(new_plan)
    for key in ('v2_failure_metadata', 'supervisor_RSS_sampling'): normalized_plan.pop(key)
    normalized_plan['execution_directory'] = old_plan['execution_directory']
    normalized_plan['protocol_differences'][-1] = old_plan['protocol_differences'][-1]
    must(normalized_plan == old_plan, 'Complete science/caps/control recipe preserved')
    preserved = ['worker.py', 'common.py', 'data_adapter.py', 'OFFICIAL_CONFIG_REFERENCE.yaml',
                 'metadata/config.json', 'metadata/DATA_AUTHORITY.json', 'metadata/RUNTIME_AUTHORITY.json',
                 'ACTUAL_DEPENDENCY_BINDING.json', 'NATIVE_SOURCE_BINDINGS.json']
    for name in preserved:
        must(digest(CANDIDATE / name) == digest(PARENT / name), 'Preserved source ' + name)
    native = load(CANDIDATE / 'NATIVE_SOURCE_BINDINGS.json')['files']
    must(len(native) == 20, '20 native source files')
    for row in native:
        must(digest(CANDIDATE / row['path']) == row['sha256'] == digest(PARENT / row['path']), 'Native source binding')
    release = load(CANDIDATE / 'ROOT_RELEASE_DISABLED_CANDIDATE.json')
    must(release == load(CANDIDATE / 'ROOT_RELEASE_TEMPLATE.json'), 'Identical disabled proposal')
    must(release['status'] == 'DISABLED_TEMPLATE_NOT_AUTHORIZATION' and release['root_authorization_reference'] is None, 'No authorization')
    must(release['source_manifest_sha256'] == 'ROOT_BIND_EXACT_MANIFEST' and release['plan_sha256'] == digest(CANDIDATE / 'PLAN.json'), 'Disabled candidate binding')
    old_release = load(PARENT / 'ROOT_RELEASE_TEMPLATE.json')
    must(dict(release, plan_sha256=digest(PARENT / 'PLAN.json')) == old_release, 'Only release plan digest changed')
    command = load(CANDIDATE / 'PROPOSED_COMMAND.json')
    must(command['numeric_launch_authorized'] is False and command['supplied_client'] is False, 'No launch/client')
    must(command['environment'] == load(PARENT / 'PROPOSED_COMMAND.json')['environment'], 'Environment policy preserved')
    must(command['maximum_concurrent_fits'] == 1 and command['execution_order'] == [0, 1, 2], 'Serialized paired workload')
    inputs = load(CANDIDATE / 'INPUT_BINDINGS.json')
    must(inputs['inputs'] == load(PARENT / 'INPUT_BINDINGS.json')['inputs'], 'Inherited pins preserved')
    for row in inputs['v3_repair_inputs']:
        path = PHASE / row['path']
        must(path.resolve().is_relative_to(PHASE) and not path.is_symlink(), 'Repair input escaped')
        must(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Repair input binding ' + row['path'])
    copies = [('evidence/V2_FAILURE_DIAGNOSIS.json', 'pencil_v2_failure_diagnosis_20261004_v1/DIAGNOSIS.json'),
              ('evidence/V2_FAILURE_RESOURCE_RECEIPTS.json', 'pencil_v2_failure_diagnosis_20261004_v1/FAILURE_RESOURCE_RECEIPTS.json'),
              ('evidence/V2_ROOT_MONITOR_0004.json', 'pencil_collab_paired_predictive_execution_root_20261004_v2/MONITOR_0004.json')]
    for local, pinned in copies:
        must(digest(CANDIDATE / local) == digest(PHASE / pinned), 'Pinned failure copy ' + local)
    receipts = load(CANDIDATE / 'evidence/V2_FAILURE_RESOURCE_RECEIPTS.json')
    physical = next(r['metadata'] for r in receipts['receipts'] if r['path'].endswith('PHYSICAL_TERMINAL.json'))
    monitor = load(CANDIDATE / 'evidence/V2_ROOT_MONITOR_0004.json')
    diagnosis = load(CANDIDATE / 'evidence/V2_FAILURE_DIAGNOSIS.json')
    release_sha = '9fe8f9dbac65aeb5ced77d83b3934dd7ee831a1563ddb24decfcef4749ee1e5f'
    must(physical['source_manifest_sha256'] == digest(PARENT / 'MANIFEST.json') and physical['release_sha256'] == release_sha, 'Failure source/release identity')
    must(diagnosis['release_sha256'] == release_sha and diagnosis['source_manifest_sha256'] == digest(PARENT / 'MANIFEST.json'), 'Diagnosis identity')
    expected_worker = monitor['seeds'][0]['worker']['expected']
    must(expected_worker['PID'] == physical['identity']['pid'] and expected_worker['start_ticks'] == physical['identity']['start_ticks'], 'Worker closure identity')
    must(physical['physical_session_closed'] and physical['direct_child_reaped'] and not physical['unresolved_cleanup'], 'Recorded v2 closure')
    must(physical['errors'] == ['RuntimeError: Owned live RSS observation missing'] and physical['physical_exit_code'] == -9, 'Recorded failure branch')
    must(physical['peak_observed_session_RSS_bytes'] < new_plan['caps']['host_RSS_bytes'] and physical['wall_seconds'] < new_plan['caps']['wall_seconds'], 'No recorded RSS/wall breach')
    parsed = 0
    for path in CANDIDATE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path)); parsed += 1
    must(all(p.stat().st_mode & 0o777 == 0o444 for p in CANDIDATE.rglob('*') if p.is_file()), 'Immutable source file modes')
    must(all(p.stat().st_mode & 0o777 == 0o555 for p in [CANDIDATE] + list(CANDIDATE.rglob('*')) if p.is_dir()), 'Immutable source directory modes')
    synthetic = synthetic_sampler_checks()
    # Final hash verification confirms that focused checks did not mutate the source.
    verify_packet(CANDIDATE,
        '45f84078f57fb788e14302aaedc37a98388ca7a120a8ff52d1a7c0d70a624cea',
        'a33f3c51e7b539900708435b711b3f89589ecfef78d1d49111b4d4a95a988bd1')
    return dict(status='PASS_FOCUSED_INDEPENDENT_SOURCE_CHECKS', candidate_payloads=len(candidate_rows),
                parent_payloads=len(parent_rows), changed_inherited_payloads=changed,
                changed_existing_supervisor_functions=changed_functions,
                complete_remaining_supervisor_AST_identical=True, native_files_identical=20,
                scientific_plan_workload_caps_allocator_and_environment_preserved=True,
                repair_input_pins_verified=len(inputs['v3_repair_inputs']), failure_copies_verified=len(copies),
                v2_closure_identity_consistent=True, python_sources_AST_parsed=parsed,
                source_unchanged_after_checks=True, synthetic_sampler=synthetic,
                candidate_module_imports=False, numerical_imports=False, real_proc_access=False,
                server_actions=False, scientific_execution=False, payload_data_reads=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))

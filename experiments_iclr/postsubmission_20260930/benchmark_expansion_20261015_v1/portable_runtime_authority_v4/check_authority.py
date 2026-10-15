"""Local source/JSON checks only; no guarded remote reads, model imports or launches."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from descriptor_binding import audit_runtime_paths, rebind_job, digest, allowed, is_provenance, RUNTIME_FIELDS, PROVENANCE
from host_binding import ALLOCATION_ROOT, PHASE_RELATIVE, pin, verify_parent_compiler_authority
from ncn_repair_authority import validate_prior_ncn_report, validate_repair_admission

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent


def main():
    checks = []
    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, function):
        try:
            function()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError(name)
    original = EXPANSION / 'qualification/portablev3_link_queued_lifecycle_v1'
    queued = EXPANSION / 'qualification/portablev3_link_queued_lifecycle_cacheaware_v1'
    provenance = json.loads((queued / 'SOURCE_JOB_PROVENANCE.json').read_text())
    source_closure = json.loads((queued / 'SOURCE_CLOSURE.json').read_text())['source_metadata_receipts']
    jobs = {}
    for row in provenance['rows']:
        family = row['case_id']
        path = queued / ('JOB_' + family + '_CUDA.json')
        job = jobs[family] = json.loads(path.read_text())
        check(family + ': original queued JOB bytes retained', path.read_bytes() == (original / path.name).read_bytes())
        audit = audit_runtime_paths(job, ALLOCATION_ROOT, PHASE_RELATIVE)
        check(family + ': all actual engineering runtime fields classified', any(x['pointer'] == '/parent_compiler_descriptor/path' for x in audit))
        check(family + ': exact parent file and canonical identities retained', job['parent_compiler_descriptor'] == row['source_compiler_row']['descriptor']
              and job['parent_runtime_job_hash'] == row['source_compiler_row']['runtime_job_hash'])
        bound, receipt = rebind_job(job, ALLOCATION_ROOT, PHASE_RELATIVE, ALLOCATION_ROOT, PHASE_RELATIVE)
        check(family + ': allocation rebind is identity', bound == job and receipt['field_changes'] == [])
        relocated, receipt = rebind_job(job, ALLOCATION_ROOT, PHASE_RELATIVE, '/explicit/future/project', 'phase')
        check(family + ': future explicit parent-path binding preserves bytes SHA and canonical hash',
              relocated['parent_compiler_descriptor']['path'] != job['parent_compiler_descriptor']['path']
              and all(relocated['parent_compiler_descriptor'][k] == job['parent_compiler_descriptor'][k] for k in ('bytes', 'sha256'))
              and relocated['parent_runtime_job_hash'] == job['parent_runtime_job_hash']
              and all(relocated.get(k) == job.get(k) for k in PROVENANCE))
        check(family + ': predecessor closure omission documented', job['parent_compiler_descriptor'] not in source_closure)
    job = jobs['NCN']
    variants = {
        'outside root': lambda j: j['parent_compiler_descriptor'].update(path='/tmp/parent.json'),
        'traversal': lambda j: j['parent_compiler_descriptor'].update(path=ALLOCATION_ROOT + '/x/../parent.json'),
        'non-JSON authority': lambda j: j['parent_compiler_descriptor'].update(path=ALLOCATION_ROOT + '/parent.pt'),
        'extra receipt field': lambda j: j['parent_compiler_descriptor'].update(arbitrary=ALLOCATION_ROOT + '/extra.json'),
        'missing byte pin': lambda j: j['parent_compiler_descriptor'].pop('bytes'),
        'boolean byte count': lambda j: j['parent_compiler_descriptor'].update(bytes=True),
        'missing canonical hash': lambda j: j.pop('parent_runtime_job_hash'),
        'malformed file hash': lambda j: j['parent_compiler_descriptor'].update(sha256='anything'),
        'nonengineering parent authority': lambda j: j.update(engineering_only=False),
        'orphan canonical hash': lambda j: j.pop('parent_compiler_descriptor'),
        'unclassified absolute field': lambda j: j.update(arbitrary_authority=ALLOCATION_ROOT + '/extra.json'),
    }
    for name, mutate in variants.items():
        bad = copy.deepcopy(job); mutate(bad)
        reject('reject ' + name, lambda: audit_runtime_paths(bad, ALLOCATION_ROOT, PHASE_RELATIVE))
    with tempfile.TemporaryDirectory(prefix='authority-json-check-') as folder:
        root = Path(folder).resolve(); parent = root / 'parent.json'
        metadata = {'schema': 'native-full-epoch-fit-v1', 'id': 'metadata-only-receipt-check'}
        parent.write_text(json.dumps(metadata) + '\n')
        candidate = {'schema': 'native-full-epoch-fit-v1', 'engineering_only': True,
                     'parent_compiler_descriptor': pin(parent), 'parent_runtime_job_hash': digest(metadata)}
        verify_parent_compiler_authority(candidate, root)
        checks.append('metadata-only parent JSON byte SHA and canonical identity accepted')
        bad = copy.deepcopy(candidate); bad['parent_runtime_job_hash'] = '0' * 64
        reject('reject wrong parent canonical identity', lambda: verify_parent_compiler_authority(bad, root))
        parent.write_text(json.dumps({**metadata, 'id': 'changed'}) + '\n')
        reject('reject changed parent JSON bytes', lambda: verify_parent_compiler_authority(candidate, root))
    audit = json.loads((EXPANSION / 'portable_dispatcher_v3/FULL_SCHEMA_AUDIT.json').read_text())
    pointers = audit['absolute_runtime_and_provenance_pointers']
    check('recorded complete canonical6460 inventory accepted', audit['fits_scanned'] == 6460
          and audit['all_descriptor_file_job_and_science_hashes_verified'] is True
          and audit['runtime_path_normalization_or_confinement_violations'] == []
          and all(is_provenance(p) or allowed(p) for p in pointers))
    check('engineering parent field absent from recorded canonical6460 inventory', '/parent_compiler_descriptor/path' not in pointers)
    projection = json.loads((EXPANSION / 'PORTABLE_PROJECTION_RUN_13.json').read_text())
    check('successful actual full prewrite projection evidence retained', projection['returncode'] == 0
          and projection['actual_output_pins']['PREWRITE_DESCRIPTOR_AUDIT.json']['sha256'] == '62400f6221df89011a5cf7cd1abe2636c101e284577364aea6535ab325a198cf')
    previous = ast.parse((EXPANSION / 'portable_dispatcher_v3/descriptor_binding.py').read_text())
    old_fields = next(ast.literal_eval(n.value) for n in previous.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'RUNTIME_FIELDS' for t in n.targets))
    check('only one exact runtime pointer added', set(RUNTIME_FIELDS) - set(old_fields) == {'/parent_compiler_descriptor/path'}
          and set(old_fields).issubset(RUNTIME_FIELDS) and 'parent_compiler_descriptor' not in PROVENANCE)
    copied = list(HERE.glob('runtime_link/*.py')) + list(HERE.glob('hosts/*.json'))
    check('eight link runtime modules and three host configs byte-identical', len(copied) == 11
          and all(p.read_bytes() == (EXPANSION / 'portable_dispatcher_v3' / p.relative_to(HERE)).read_bytes() for p in copied))
    contract = json.loads((HERE / 'NCN_REPAIR_CONTRACT.json').read_text())
    supervisor = json.loads((EXPANSION / 'NCN_LIFECYCLE_OBSERVATION_17.json').read_text())['supervisor']
    child = supervisor['last_child_report']
    summary = validate_prior_ncn_report(supervisor, child, contract)
    check('actual saved NCN started-child failure has exactly zero native work', summary['prior_child_started'] is True and summary['prior_native_work'] == 0)
    for name, mutate in {
        'nonzero native work': lambda c: c.update(updates_attempted=1),
        'different error': lambda c: c['failure'].update(message='different failure'),
        'different source': lambda c: c['script'].update(sha256='0' * 64),
        'Torch imported': lambda c: c['interpreter'].update(Torch_imported=True),
    }.items():
        bad = copy.deepcopy(child); mutate(bad)
        changed_supervisor = copy.deepcopy(supervisor); changed_supervisor['last_child_report'] = bad
        reject('repair rejects ' + name, lambda: validate_prior_ncn_report(changed_supervisor, bad, contract))
    admission = {'case_id': 'NCN', 'repair_attempt_id': contract['repair_attempt_id'],
                 'prior_failed_attempt': contract['admission_prior_failed_attempt'], 'output': contract['repair_output'],
                 'attempt_claim': contract['repair_attempt_claim'], 'job_sha256': contract['prior_job']['sha256'],
                 'limits': contract['limits'], 'maximum_attempts': 1, 'automatic_retries': 0}
    validate_repair_admission(admission, contract)
    check('explicit finite repair identity accepted', admission['attempt_claim'] != contract['prior_attempt_claim']['path'])
    for key in ('repair_attempt_id', 'prior_failed_attempt', 'attempt_claim'):
        bad = dict(admission); bad.pop(key)
        reject('repair rejects missing ' + key, lambda: validate_repair_admission(bad, contract))
    bad = copy.deepcopy(admission); bad['limits']['updates_attempted'] += 1
    reject('repair rejects expanded native work bound', lambda: validate_repair_admission(bad, contract))
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    checks.append('all Python sources parse without scientific imports')
    result = {'schema': 'runtime-parent-authority-local-source-checks-v1', 'status': 'PASS', 'checks': checks,
              'check_count': len(checks), 'native_jobs_started': 0, 'tensor_or_dataset_files_opened': 0,
              'server77_accessed': False, 'fresh_remote_canonical6460_reread': False,
              'canonical6460_evidence': 'complete recorded pointer inventory plus root successful full prewrite receipt',
              'guarded_actual_parent_and_prior_terminal_verification': 'required at root runtime before Torch; not performed locally'}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

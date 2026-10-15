"""Focused local checks for the concrete finite driver; never executes a worker."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parents[1]
PREVIOUS = HERE.parent / 'portablev3_link_queued_lifecycle_cacheaware_v1'
RUNTIME = EXPANSION / 'portable_runtime_authority_v4'
sys.path.insert(0, str(RUNTIME))
from ncn_repair_authority import validate_repair_admission
from descriptor_binding import audit_runtime_paths
from host_binding import ALLOCATION_ROOT, PHASE_RELATIVE


def main():
    checks = []
    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    unchanged = ('CASES.json', 'CACHE_POLICY.json', 'JOB_NCN_CUDA.json', 'JOB_BUDDY_CUDA.json', 'SOURCE_JOB_PROVENANCE.json')
    check('native cases JOB recipes policies bounds and cache policy byte-identical',
          all((HERE / n).read_bytes() == (PREVIOUS / n).read_bytes() for n in unchanged))
    sources = {n: (HERE / n).read_text() for n in ('support.py', 'qualify.py', 'supervise.py')}
    def function(folder, file, name):
        tree = ast.parse((folder / file).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    check('cache-aware preflight and two-worker resource ceiling unchanged',
          function(HERE, 'supervise.py', 'preflight') == function(PREVIOUS, 'supervise.py', 'preflight'))
    check('native lifecycle body after Torch import unchanged',
          sources['qualify.py'].split('        import torch\n', 1)[1] == (PREVIOUS / 'qualify.py').read_text().split('        import torch\n', 1)[1])
    check('strict parent/job validation before child launch and before Torch',
          'host_binding.validate_host_job(json.loads(job.read_text()),ROOT)' in sources['support.py']
          and sources['supervise.py'].index('load_admission(args.admission,here)') < sources['supervise.py'].index('subprocess.Popen(')
          and sources['qualify.py'].index('common.validate_host_job(job,ROOT)') < sources['qualify.py'].index('        import torch\n'))
    check('exact prior failure verified before repair admission returns',
          sources['support.py'].index('verify_prior_ncn_failure(ROOT)') < sources['support.py'].index('validate_repair_admission(value,contract)')
          < sources['support.py'].index('    return value,job,case'))
    closure = json.loads((HERE / 'SOURCE_CLOSURE.json').read_text())
    contract = json.loads((RUNTIME / 'NCN_REPAIR_CONTRACT.json').read_text())
    retarget = json.loads((HERE / 'PRELAUNCH_RETARGET.json').read_text())
    runtime_sha = hashlib.sha256((RUNTIME / 'MANIFEST.json').read_bytes()).hexdigest()
    check('concrete runtime manifest pinned consistently', runtime_sha in sources['support.py']
          and retarget['target_portable_manifest']['sha256'] == runtime_sha
          and closure['current_runtime_manifest']['sha256'] == runtime_sha)
    check('no predecessor source custody removed', all(r in closure['source_metadata_receipts']
          for r in json.loads((PREVIOUS / 'SOURCE_CLOSURE.json').read_text())['source_metadata_receipts']))
    for family in ('NCN', 'BUDDY'):
        job = json.loads((HERE / ('JOB_' + family + '_CUDA.json')).read_text())
        audit_runtime_paths(job, ALLOCATION_ROOT, PHASE_RELATIVE)
        check(family + ': parent descriptor is now a direct closure receipt', job['parent_compiler_descriptor'] in closure['source_metadata_receipts'])
        admission = json.loads((HERE / ('ADMISSION_TEMPLATE_' + family + '.json')).read_text())
        check(family + ': template remains unadmitted and pins same JOB', admission['template_is_not_admission'] is True
              and admission['published'] is False and admission['publication_commit'] is None
              and admission['job_sha256'] == hashlib.sha256((HERE / ('JOB_' + family + '_CUDA.json')).read_bytes()).hexdigest()
              and admission['runtime_manifest_sha256'] == runtime_sha)
        if family == 'NCN':
            validate_repair_admission(admission, contract)
            check('NCN repair identity cannot reuse failed claim/output', admission['attempt_claim'] != contract['prior_attempt_claim']['path']
                  and admission['output'] not in contract['prior_child_result']['path'])
        else:
            old_admission = json.loads((PREVIOUS / 'ADMISSION_TEMPLATE_BUDDY.json').read_text())
            check('BUDDY original claim output logical case retained', admission['output'] == old_admission['output']
                  and admission['attempt_claim'].endswith('/portablev3_link_queued_BUDDY_ATTEMPT_CLAIM.json')
                  and 'repair_attempt_id' not in admission and 'prior_failed_attempt' not in admission)
    check('permanent fixed claim exclusive-create retained', "claim=inside(admission['attempt_claim'])" in sources['supervise.py']
          and "with claim.open('x')" in sources['supervise.py']
          and all(x not in sources['supervise.py'] for x in ('.unlink(', '.rmdir(', 'shutil.rmtree(')))
    check('repair lineage recorded in supervisor child and claim',
          sources['supervise.py'].count('verified_prior_failure') >= 4 and 'verified_prior_failure' in sources['qualify.py'])
    check('closed TEST no extra case or expanded map bounds',
          set(json.loads((HERE / 'CASES.json').read_text())['cases']) == {'NCN', 'BUDDY'}
          and json.loads((HERE / 'CASES.json').read_text())['cases']['NCN']['limits']['optimizer_attempts'] == 34
          and json.loads((HERE / 'CASES.json').read_text())['cases']['BUDDY']['limits']['optimizer_attempts'] == 3170)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    checks.append('all driver Python parses without runtime imports or launch')
    print(json.dumps({'schema': 'finite-queued-link-retarget-source-checks-v1', 'status': 'PASS', 'checks': checks,
                      'check_count': len(checks), 'native_jobs_started': 0, 'tensor_or_dataset_files_opened': 0,
                      'remote_access': False, 'actual_allocation_admission_and_execution': 'not performed'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

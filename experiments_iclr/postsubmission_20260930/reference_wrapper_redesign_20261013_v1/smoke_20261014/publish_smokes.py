"""Publish reviewed smoke source and compact receipts; never publish tensors."""
import argparse
import gzip
import hashlib
import json
import shutil
import socket
import subprocess
from pathlib import Path

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
DESIGN = REPO / 'experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1'
SMOKE = DESIGN / 'smoke_20261014'
BASE = '765c42fffbf60fa4ff6a5ed05747d12d0da4f5a1'
BRANCH = 'codex/postsubmission-research-20260930'
EXCLUDED = {'results', 'working_data', 'transport', '__pycache__', 'publication'}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).splitlines() == [
        'GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = Path(args.report).resolve()
    assert report.is_relative_to(SMOKE / 'results')
    data = json.loads((report / 'RESULTS.json').read_text())
    assert data['counts']['expected_matrix_cases'] == 167
    assert data['counts']['missing'] == 0 and not data['parse_errors']
    assert data['full_campaign_GO'] is False
    assert git('branch', '--show-current') == BRANCH
    assert git('rev-parse', 'HEAD') == BASE
    assert git('remote', 'get-url', 'origin') == 'git@github.com:ChervovNikita/diploma_yandex_graphs.git'
    assert not git('diff', '--cached', '--name-only')
    publication = SMOKE / 'publication'
    assert not publication.exists()
    publication.mkdir()
    for name in ['REPORT.md', 'CASE_CHECKS.csv', 'ATTEMPTS.csv',
                 'STEP_WORK.csv', 'PAIR_MEMORY.csv']:
        shutil.copy2(report / name, publication / name)
    (publication / 'RESULTS.json.gz').write_bytes(gzip.compress(
        (report / 'RESULTS.json').read_bytes(), mtime=0))
    (SMOKE / 'MANIFEST.json.gz').write_bytes(gzip.compress(
        (SMOKE / 'MANIFEST.json').read_bytes(), mtime=0))
    if (SMOKE / 'NONNODE_MANIFEST.json').exists():
        (SMOKE / 'NONNODE_MANIFEST.json.gz').write_bytes(gzip.compress(
            (SMOKE / 'NONNODE_MANIFEST.json').read_bytes(), mtime=0))
    counts = data['counts']
    functional = data['functional_completion_counts']
    block = f'''## Real-data smoke checks — 14 October 2026

Completed **167 distinct configuration checks** across the redesigned study:
56 node, 96 molecular and 15 link cases. Each uses real source TRAIN data,
two initial optimizer updates, complete VALID evaluation and checkpoint
save/reload. A third update is run and replayed; all four callbacks and all
four independently owned I4 optimizers are charged. TEST is not scored.

The latest seeded attempts report **{counts['passed']} strict passes and
{counts['failed']} failures**. The separate functional diagnostic confirms
{functional['selected_observed_functional_completion']} cases; incomplete
legacy telemetry remains unknown. Floating replay failures stay failed in
the reports, even when initialization, updates and exact restoration succeed.
These checks establish execution evidence, not prediction quality or convergence.

The source audit corrected link bias scaling and limited NCNC output factors
to the top-level task scorer, retaining its native recursive completion.
Corrected link cases were rerun. All earlier outcomes and repair history remain.
'''
    passed_pairs = [p for p in data['pair_memory'] if p['status'] == 'passed']
    if passed_pairs:
        tested = passed_pairs[-1]
        block += (f"\nA concurrent Physics Mono/private-ensemble-credit + NCNC/I4 probe passed. "
                  f"Its maximum sampled own GPU usage was "
                  f"{tested['maximum_sampled_own_GPU_usage_mib'] / 1024:.1f} GiB, "
                  f"with at least {tested['minimum_sampled_device_free_mib'] / 1024:.1f} GiB "
                  "device memory free. This qualifies the tested allocation pairing, "
                  "not every combination or the 18.77 runtime.\n")
    block += '''
**Full scientific launch remains disabled.** Complete epoch handling,
source/main selectors, independent I4 stopping/checkpoint ownership, selected
TEST serving and the executable three-GPU launcher still require integration.
The old v29 launcher remains retired; no full-campaign GO has been received.

- [Smoke methodology and limits](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/README.md)
- [All configuration checks and retained failures](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/publication/REPORT.md)
- [Remaining full-run implementation](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/controller/FULL_RUN_GAPS.md)

'''
    readme = REPO / 'README.md'
    original = readme.read_text()
    assert '## Real-data smoke checks — 14 October 2026' not in original
    history = SMOKE / 'history'
    history.mkdir(exist_ok=True)
    (history / 'ROOT_README_BEFORE_SMOKES.md').write_text(original)
    title, rest = original.split('\n', 1)
    readme.write_text(title + '\n\n' + block + rest.lstrip('\n'))
    (SMOKE / 'ROOT_README_BLOCK.md').write_text(block)
    names = [p for p in sorted(SMOKE.rglob('*')) if p.is_file()
             and not (EXCLUDED & set(p.relative_to(SMOKE).parts))
             and p.suffix in {'.py', '.md', '.json', '.csv', '.gz', '.txt', '.sh'}
             and p not in {SMOKE / 'MANIFEST.json', SMOKE / 'NONNODE_MANIFEST.json'}]
    names += sorted(p for p in publication.iterdir() if p.is_file())
    release = {'schema': 'smoke-source-release-v1', 'client_date': '2026-10-14',
               'scientific_GO': False, 'accuracy_results': 0,
               'matrix_counts': data['counts'],
               'source_files': [{'path': str(p.relative_to(REPO)),
                                 'bytes': p.stat().st_size,
                                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in names],
               'server_evidence_report': str(report),
               'report_sha256': hashlib.sha256((report / 'RESULTS.json').read_bytes()).hexdigest()}
    release_path = publication / 'SOURCE_RELEASE.json'
    release_path.write_text(json.dumps(release, indent=2) + '\n')
    names += [release_path, DESIGN / 'README.md', REPO / 'README.md']
    assert all(p.stat().st_size < 5_000_000 for p in names)
    assert sum(p.stat().st_size for p in names) < 20_000_000
    relative = [str(p.relative_to(REPO)) for p in names]
    git('add', '-f', '--', *relative)
    staged = git('diff', '--cached', '--name-only').splitlines()
    assert set(staged) <= set(relative)
    git('-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof',
        'diff', '--cached', '--check')
    print(git('diff', '--cached', '--stat'), flush=True)
    git('commit', '-m', 'Add real-data smoke checks, source-port fixes and preserved qualification receipts')
    receipt = {'schema': 'real-data-smoke-commit-v1', 'branch': BRANCH,
               'commit': git('rev-parse', 'HEAD'), 'base': BASE,
               'committed_paths': staged, 'matrix_counts': data['counts'],
               'full_campaign_GO': False, 'scientific_accuracy_results': 0}
    (SMOKE / 'COMMIT_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'committed_paths'}))


if __name__ == '__main__':
    main()

"""Prepare a narrow reviewed publication packet; inspect only, never commit or push."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import base64
import difflib
import hashlib
import json
import runpy
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
PUB = 'publication/v6_original15_v4_numerical_literature_20261004_v1'
BASE = 'db0302db9991bbb5e167a94948ff6fe5f4199551'
PREFIX = 'experiments_iclr/postsubmission_20260930/'
AMAZON = 'amazon_polynormer_paired_family_execution_root_20261003_v3'
HELPER = 'publication/publish_exact_inventory_v10.py'
SUFFIXES = {'.py', '.json', '.jsonl', '.md', '.txt', '.html', '.diff', '.patch',
            '.log', '.raw', '.sha256', '.csv', '.xml', '.sh'}
ROOTS = [
    AMAZON + '/v6_runtime_cpu_v1',
    AMAZON + '/v6_runtime_cuda0_v1',
    AMAZON + '/v6_runtime_execution_receipts_20261003_v1',
    AMAZON + '/v6_qualification_block0_cuda0_v1',
    AMAZON + '/v6_qualification_execution_receipts_20261003_v1',
    AMAZON + '/v6_resource_filesystem_benchmark_20261003_v1',
    AMAZON + '/v6_measured_resource_admission_candidate_20261003_v1',
    AMAZON + '/v6_admitted_resource_fit_release_preparation_20261003_v1',
    AMAZON + '/v6_admitted_resource_publication_receipts_20261003_v1',
    AMAZON + '/v6_releases',
    AMAZON + '/releases/fits',
    AMAZON + '/v6_full_schedule_v1',
    AMAZON + '/v6_queue_detached_launch_20261003_v1',
    AMAZON + '/v6_queue_launch_execution_receipts_20261003_v1',
    AMAZON + '/v6_queue_launch_execution_receipts_20261003_v2',
    AMAZON + '/v6_queue_owned_monitoring_20261003_v1',
    'graph_ncNC_structural_pattern_pilot_preparation_20261003_v4',
    'graph_ncNC_structural_pattern_v4_independent_source_review_20261003_v1',
    'graph_ncNC_structural_pattern_v4_numerical_execution_metadata_preparation_20261003_v1',
    'graph_ncNC_structural_pattern_v4_numerical_helper_root_review_20261003_v1',
    'graph_ncNC_structural_pattern_numerical_execution_root_20261003_v4',
    'literature_memory/index_v41',
    'graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1',
    'ncnc_base_complete_family_analysis_source_report_20261003_v1',
]
EXPLICIT = {
    'PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
    AMAZON + '/V6_CONSUMER_RELEASE_v1.json',
    AMAZON + '/V6_RESOURCE_ADMISSION_v1.json',
    AMAZON + '/fits/split0_gnnm_boundary_4_seed17/LAUNCH.json',
}

REMOTE_BASELINE = r'''
import base64,hashlib,json,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2]
payload=json.loads(sys.stdin.read())
def git(*args,input=None):
 return subprocess.run(['git',*args],cwd=repo,input=input,capture_output=True,check=True).stdout
assert repo.resolve()==repo and pathlib.Path(git('rev-parse','--show-toplevel').decode().strip())==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[uuid]
assert git('branch','--show-current').decode().strip()==payload['branch']
assert git('rev-parse','HEAD').decode().strip()==payload['expected_head']
assert git('remote','get-url','origin').decode().strip()=='git@github.com:ChervovNikita/diploma_yandex_graphs.git'
assert not git('diff','--cached','--name-only')
paths=payload['targets']
assert len(paths)==len(set(paths))
for value in paths:
 rel=pathlib.PurePosixPath(value)
 assert not rel.is_absolute() and '..' not in rel.parts
 assert value=='README.md' or value.startswith('experiments_iclr/postsubmission_20260930/')
rows=[]
for line in git('ls-tree','-rz',payload['expected_head'],'--',*paths).split(b'\0'):
 if not line:continue
 header,path=line.split(b'\t',1);mode,kind,oid=header.split()
 assert kind==b'blob' and mode in (b'100644',b'100755')
 rows.append((path.decode(),oid.decode()))
oids=[oid for _,oid in rows]
data=git('cat-file','--batch',input=('\n'.join(oids)+'\n').encode()) if oids else b''
offset=0;result=[];readme=None
for path,oid in rows:
 end=data.index(b'\n',offset);header=data[offset:end].split();offset=end+1
 assert header[0].decode()==oid and header[1]==b'blob'
 size=int(header[2]);blob=data[offset:offset+size];offset+=size
 assert data[offset:offset+1]==b'\n';offset+=1
 result.append(dict(target=path,git_blob=oid,bytes=size,sha256=hashlib.sha256(blob).hexdigest()))
 if path=='README.md':readme=base64.b64encode(blob).decode()
assert offset==len(data) and readme is not None
print(json.dumps(dict(head=payload['expected_head'],branch=payload['branch'],route_uuid=uuid,
                     no_staged_edits=True,origin_verified=True,files=result,README_base64=readme)))
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def local(name):
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    path = PHASE.joinpath(*rel.parts)
    assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
    for ancestor in path.parents:
        if ancestor == PHASE:
            break
        assert not ancestor.is_symlink()
    return path


def row(name, target=None):
    data = local(name).read_bytes()
    return dict(source=name, target=target or PREFIX + name, bytes=len(data), sha256=digest(data))


def validate_text(path):
    data = path.read_bytes()
    assert len(data) < 2_000_000, str(path)
    assert b'\0' not in data, str(path)
    text = data.decode('utf-8')
    assert '-----BEGIN PRIVATE KEY-----' not in text
    assert '-----BEGIN OPENSSH PRIVATE KEY-----' not in text
    if path.suffix == '.json':
        json.loads(text)
    elif path.suffix == '.jsonl':
        for line in text.splitlines():
            if line.strip():
                json.loads(line)


def verify_manifest(path):
    value = json.loads(path.read_text())
    entries = value.get('payload', value.get('files', value.get('payloads', [])))
    if not isinstance(entries, list):
        return dict(path=str(path.relative_to(PHASE)), format='nonlist_payload', checked=0)
    checked = 0
    for item in entries:
        assert isinstance(item, dict)
        descriptor = item.get('descriptor', item)
        relative = item.get('relative') or descriptor.get('path') or descriptor.get('file')
        if not relative or 'sha256' not in descriptor or 'bytes' not in descriptor:
            continue
        if item.get('relative'):
            candidate = path.parent / relative
        elif Path(relative).is_absolute():
            candidate = Path(relative)
        elif (path.parent / relative).is_file():
            candidate = path.parent / relative
        else:
            candidate = PHASE / relative
        assert candidate.resolve().is_relative_to(PHASE) and not candidate.is_symlink()
        data = candidate.read_bytes()
        assert len(data) == descriptor['bytes'] and digest(data) == descriptor['sha256'], str(candidate)
        checked += 1
    return dict(path=str(path.relative_to(PHASE)), checked=checked)


def main():
    output = local(PUB)
    output.mkdir(exist_ok=False)
    helpers = runpy.run_path(str(local('publication/publish_exact_inventory_v9.py')),
                             run_name='publication_helper_definitions_only')
    old_helper = local('publication/publish_exact_inventory_v9.py').read_text()
    old_allow = "source.suffix in ('.py', '.json', '.md',"
    new_allow = "source.suffix in ('.py', '.json', '.jsonl', '.md',"
    assert old_helper.count(old_allow) == 1
    new_helper = old_helper.replace(old_allow, new_allow)
    assert not local(HELPER).exists()
    with local(HELPER).open('x') as handle:
        handle.write(new_helper)
    helper_diff = ''.join(difflib.unified_diff(old_helper.splitlines(True), new_helper.splitlines(True),
                         fromfile='publish_exact_inventory_v9.py', tofile='publish_exact_inventory_v10.py'))
    (output / 'HELPER_ALLOWLIST.diff').write_text(helper_diff)
    assert helpers['REMOTE'] == runpy.run_path(str(local(HELPER)), run_name='successor_helper_definitions_only')['REMOTE']
    sources = set(EXPLICIT)
    excluded = []
    for name in ROOTS:
        root = local(name)
        assert root.is_dir(), name
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = str(path.relative_to(PHASE))
            assert not path.is_symlink()
            if '__pycache__' in path.parts or path.suffix not in SUFFIXES or path.stat().st_size >= 2_000_000:
                excluded.append(dict(path=relative, bytes=path.stat().st_size, reason='binary_cache_or_over_text_limit'))
                continue
            sources.add(relative)
    for name in sources:
        validate_text(local(name))
    candidates = [row(name) for name in sorted(sources)]
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE_BASELINE,
                          helpers['REPO'], helpers['UUID']])
    result = subprocess.run(helpers['SSH'] + [command], input=json.dumps(dict(
        expected_head=BASE, branch=helpers['BRANCH'], targets=['README.md'] + [r['target'] for r in candidates])),
        capture_output=True, text=True, timeout=60)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), operation='read_only_baseline_git_metadata',
                   exit_code=result.returncode, ssh_destination=helpers['LOGIN'],
                   executed_source_sha256=digest(REMOTE_BASELINE.encode()), stdout=result.stdout, stderr=result.stderr)
    write_json(output / 'BASELINE_GIT_READ_RECEIPT.json', receipt)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    baseline = {r['target']: r for r in observed['files']}
    write_json(output / 'BASELINE_TARGET_PINS.json', {k:v for k,v in observed.items() if k != 'README_base64'})
    before = base64.b64decode(observed['README_base64'])
    assert digest(before) == baseline['README.md']['sha256']
    (output / 'README_MAIN_BEFORE.md').write_bytes(before)
    text = before.decode()
    start = text.index('The six native Amazon Ratings recipe fits')
    end = text.index('\nThe complete shared/private HGT objective study', start)
    text = text[:start] + '''The six native Amazon Ratings recipe fits and replay remain complete. The fixed authored Polynormer-r comparison preserves raw features, 200 local and 2500 global epochs, three paired blocks and 15 distinct fits. Fresh V6 CPU/GPU runtime captures and five-form numerical qualification passed, including ten bitwise next-step replays and five state-retirement checks. Measured resource admission then released the original 15-fit queue under ordinary detached supervision. At 20:59:56 UTC on 3 October, the first GNNM fit had completed 56 of 2700 updates; no predictive comparison is available. The measured forecast is 51.96 hours, or 77.95 hours with a 50% margin. Quota and allocation expiry remain unknown.
''' + text[end:]
    start = text.index('The frozen NCNC hypothesis compares')
    end = text.index("\nFoRDE's explicit streamer", start)
    text = text[:start] + '''The frozen NCNC hypothesis compares reconstruction of whole TRAIN observation patterns with matched marginal supervision. The likelihood has GRAN ancestry. Both complete-graph qualification attempts failed memory caps before a completed J update, with all costs retained. V4 preserves the complete scientific work through activation recomputation. Independent model/helper source reviews and the fresh small-input J/F numerical pair passed: 14 engineering updates checked gradients, model/Adam state, RNG, flags and serialized replay. Full-graph parity, feasibility and predictive J/F results remain pending. No further stage was launched. The count-aware structural single has a sealed implementation, with review and runtime pending.

BUDDY and NCNC base comparisons require complete family closure before scoring. The [NCNC base source report](experiments_iclr/postsubmission_20260930/ncnc_base_complete_family_analysis_source_report_20261003_v1/REPORT.md) reuses the existing frozen 35-fit/25-cell saved-VALID analysis and records the missing selected-checkpoint numerical replay pass. Its minimal replay proposal is unreleased; no fitted outcome was accessed for that report. Checkpoints and runtime records remain on the authorized servers.
''' + text[end:]
    start = text.index('Two full-graph NCNC pattern qualification attempts exceeded memory caps')
    end = text.index('\nOriginal paper scores, unsuccessful experiments', start)
    text = text[:start] + '''Amazon V6 runtime, five-form numerical qualification and measured resource admission are complete; the original 15-fit queue is running. NCNC V4 independent source reviews and the small-input numerical pair passed. The two earlier full-graph failures remain preserved, and full-graph parity and feasibility remain pending. These engineering results establish no predictive winner.

Canonical [literature index_v41](experiments_iclr/postsubmission_20260930/literature_memory/index_v41/INTEGRATION.json) now preserves 180 scoped conclusions across 131 paper identities and two software identities. It integrates the first coherent-reconstruction scout and the [second structural-specialization scout](experiments_iclr/postsubmission_20260930/graph_conditioned_low_rank_structural_specialization_literature_scout_20261003_v1/REPORT.md): twelve new method scopes and zero whole-paper certifications. Relevant prior work is retained. Any contribution still requires capable controls, prospective paired replication, heldout confirmation and fresh independent manuscript review.
''' + text[end:]
    (output / 'README_MAIN.md').write_text(text)
    readme_diff = ''.join(difflib.unified_diff(before.decode().splitlines(True), text.splitlines(True),
                       fromfile='README.md@' + BASE, tofile='README.md@proposed'))
    (output / 'README_AMENDMENT.diff').write_text(readme_diff)
    (output / 'README_AMENDMENT.md').write_text('''# Proposed README amendment

Record the completed Amazon V6 runtime, five-form qualification and measured resource admission, followed by the original 15-fit detached queue launch and the 20:59:56 UTC progress observation. Preserve the unknown quota/expiry and absence of a predictive comparison.

Record NCNC V4 independent model/helper reviews and the single small-input numerical PASS, while preserving both earlier full-graph failures and pending full-graph parity/feasibility. Describe the existing NCNC base analysis and its missing, unreleased selected-checkpoint replay pass.

Update the canonical literature count to index_v41: 180 scoped conclusions, 131 paper identities and two software identities. Link the second scout and retain the twelve new method scopes / zero whole-paper-certification boundary. Preserve the complete negative Mixed40 decision and all existing paper results.
''')
    unchanged = []
    selected = []
    for candidate in candidates:
        previous = baseline.get(candidate['target'])
        if previous and previous['sha256'] == candidate['sha256']:
            unchanged.append(dict(**candidate, reason='identical_to_verified_baseline_commit', git_blob=previous['git_blob']))
        else:
            selected.append(candidate)
    manifests = [verify_manifest(local(name)) for name in sources if PurePosixPath(name).name == 'MANIFEST.json']
    write_json(output / 'EXCLUSIONS.json', dict(nontext_or_large=excluded, already_published_unchanged=unchanged))
    extra = {HELPER, str(Path(__file__).resolve().relative_to(PHASE))}
    extra |= {PUB + '/' + name for name in ('BASELINE_GIT_READ_RECEIPT.json', 'BASELINE_TARGET_PINS.json',
              'README_MAIN_BEFORE.md', 'README_AMENDMENT.md', 'README_AMENDMENT.diff', 'HELPER_ALLOWLIST.diff', 'EXCLUSIONS.json')}
    selected += [row(name) for name in sorted(extra)]
    files = [row(PUB + '/README_MAIN.md', 'README.md')] + selected
    assert len(files) == len({r['target'] for r in files})
    for item in files:
        validate_text(local(item['source']))
    state_pins = [r for r in files if r['source'] in {'PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json'}]
    assert len(state_pins) == 3
    assert 'fresh V4 small-input numerical pair passed once' in local('PUBLIC_STATUS.md').read_text()
    assert 'V4 small-input value/gradient/RNG/update parity passed once' in local('RESEARCH_STATE.md').read_text()
    helper_validation = dict(original_sha256=digest(old_helper.encode()), successor_sha256=digest(new_helper.encode()),
                             change='Add .jsonl to the local text suffix allowlist only', remote_body_byte_identical=True,
                             original_source_unchanged=local('publication/publish_exact_inventory_v9.py').read_text()==old_helper)
    validation = dict(schema='narrow_new_records_publication_validation_v1', UTC=datetime.now(timezone.utc).isoformat(),
                      expected_head=BASE, selected_records=len(files), selected_bytes=sum(r['bytes'] for r in files),
                      already_published_unchanged=len(unchanged), excluded_nontext_or_large=len(excluded),
                      UTF8_text_and_JSON_JSONL_parsing=True, no_duplicate_targets=True, path_confinement=True,
                      maximum_file_bytes=max(r['bytes'] for r in files), helper=helper_validation,
                      payload_manifest_checks=manifests, state_pins=state_pins,
                      scope=dict(roots=ROOTS, explicit=sorted(EXPLICIT)),
                      actions=dict(remote_read_only=True, numerical_work=False, process_monitoring=False,
                                   commit=False, push=False, source_overwrite=False, checkpoint_or_data_access=False))
    write_json(output / 'VALIDATION.json', validation)
    files.append(row(PUB + '/VALIDATION.json'))
    inventory = dict(schema='exact_research_publication_inventory_v1', expected_head=BASE,
                     branch=helpers['BRANCH'], message='Record Amazon original15 launch, NCNC V4 numerical PASS and literature integration',
                     files=files, remove=[], excludes=excluded,
                     already_published_unchanged=len(unchanged), preparation_only=True)
    write_json(output / 'INVENTORY.json', inventory)
    print(json.dumps(dict(inventory=PUB + '/INVENTORY.json', expected_head=BASE, files=len(files),
                         bytes=sum(r['bytes'] for r in files), already_published_unchanged=len(unchanged),
                         nontext_or_large_excluded=len(excluded), manifests_checked=len(manifests))))


if __name__ == '__main__':
    main()

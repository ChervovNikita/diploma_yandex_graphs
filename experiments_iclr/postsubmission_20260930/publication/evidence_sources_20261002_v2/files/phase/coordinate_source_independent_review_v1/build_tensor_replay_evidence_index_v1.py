"""Build the immutable Stage1 evidence index using JSON metadata and byte hashes only.

Does not import the assessment/replay/model packages or deserialize tensor payloads.
Writes new files exclusively and never alters evidence or preserved sources.
"""

import hashlib
import json
import stat
from collections import Counter
from pathlib import Path


PHASE = Path(__file__).resolve().parent.parent
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
ROOT = 'coordinate_ensemble_execution_root_v1'
ASSESS = 'coordinate_stage1_assessment_source_v2'
REPLAY = 'coordinate_stage1_tensor_audit_source_v1'
REVIEW = 'coordinate_source_independent_review_v1'
INDEX = ROOT + '/TENSOR_REPLAY_EVIDENCE_INDEX_v1.json'
LAUNCHER = ROOT + '/stage1_serial_queue_run01'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    p = Path(value)
    require('..' not in p.parts, 'Traversal: ' + str(value))
    if p.is_absolute():
        p = p.relative_to(REMOTE)
    require(str(p) != '.' and not p.is_absolute(), 'Invalid relative path')
    return p.as_posix()


def safe(value):
    rel = relative(value)
    p = PHASE
    for part in Path(rel).parts:
        p = p / part
        require(not p.is_symlink(), 'Symlink: ' + rel)
    require(p.resolve().is_relative_to(PHASE), 'Path escapes phase: ' + rel)
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


entries = {}
groups = {}
binding_checks = []
metadata_read = []


def add(value, group, expected_sha=None, expected_bytes=None):
    rel, p = relative(value), safe(value)
    require(stat.S_ISREG(p.stat().st_mode), 'Not a regular file: ' + rel)
    before = p.stat()
    item = {'path': rel, 'sha256': digest(p), 'bytes': before.st_size}
    after = p.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed while hashing: ' + rel)
    if expected_sha is not None:
        require(item['sha256'] == expected_sha, 'Bound SHA256 mismatch: ' + rel)
    if expected_bytes is not None:
        require(item['bytes'] == expected_bytes, 'Bound length mismatch: ' + rel)
    if rel in entries:
        require(entries[rel] == item, 'Inconsistent duplicate: ' + rel)
    else:
        entries[rel], groups[rel] = item, group
    return item


def bind(binding, origin):
    require(len(binding['sha256']) == 64, 'Unfrozen binding: ' + origin)
    item = add(binding['path'], 'source_control', binding['sha256'], binding.get('bytes'))
    binding_checks.append({'origin': origin, **item})
    return item


def read_metadata(value):
    rel = relative(value)
    metadata_read.append(rel)
    return json.loads(safe(value).read_text())


def bindings(value, origin):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            bind(value, origin)
        else:
            for key, child in value.items():
                bindings(child, origin + '/' + key)
    elif isinstance(value, list):
        for number, child in enumerate(value):
            bindings(child, origin + '/' + str(number))


def add_tree(value, group, expected_count):
    root = safe(value)
    require(root.is_dir(), 'Missing evidence directory: ' + str(value))
    files = []
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(), 'Symlink evidence: ' + str(p))
        if p.is_file():
            files.append(p)
        else:
            require(p.is_dir(), 'Non-file evidence node: ' + str(p))
    require(len(files) == expected_count, 'Evidence directory count differs: ' + str(value))
    for p in files:
        add(p.relative_to(PHASE).as_posix(), group)


def write_new(value, payload):
    p = safe(value)
    with p.open('x') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    return {'path': relative(value), 'sha256': digest(p), 'bytes': p.stat().st_size}


# Anchor controls and sources were frozen by the root before outcome inspection.
anchors = {
    ROOT + '/STAGE1_COMPARATIVE_PROTOCOL_v1.json': '7817d3b818ee9deec11dae6bb2867ff91d150bcea350f5428d23c95bee241444',
    ROOT + '/STAGE1_SERIAL_QUEUE_MANIFEST_v2.json': 'b3f68705410e3e302f4c0bbd5225dc6ec5cd45844c6eda9cefc2e2c05204e734',
    ROOT + '/CS_ACQUISITION_SOURCES_v1.json': 'e8545d00399bfb471e7f9cc6dc4f6a984659edaa784aab7a4de01a7bda64796c',
    ASSESS + '/SPECIFICATION_v2.json': '6da6a44bbc2b55118296b8e0de007330fd5d83983b57fb3d17fdd093e0cef44d',
    ASSESS + '/ROOT_ADOPTION_v2.json': '0f3f6c4e8dac4f695e8291d3b9eb65897f3984f9f8c09879dc46cb42d8bfad17',
    ASSESS + '/validate_stage1_v2.py': 'e5f870f62a0d9635113517bc80a8ff1fe8b858b9b2c8d4e00b22854770fc4bab',
    ASSESS + '/SHA256SUMS_v2.json': 'b2c8a3230ed73952ff27a37cf75c058181610b11515d66c839925c8f64b929d4',
    REPLAY + '/replay_selected_v2.py': '0d6d33923415bd0af45022ccf8ebd0648897ecf90f26789a4017684ac384cc45',
    REPLAY + '/REQUEST_TEMPLATE_v2.json': '2279cc4ba1d57efc6a65225f02c02c72f72fad5acdd7338ce38cbd8417a9cf60',
    REPLAY + '/SHA256SUMS_v2.json': 'c61a8f7073e267cb80407a18582da12bb7201bf4567bd56326de2c6e8b7f9fef',
    ROOT + '/TENSOR_REPLAY_ROOT_ADOPTION_v1.json': '0721fca0fffbc52311265205e1b009fee7da0c787f58b67d023f764167a4a3bc',
    ROOT + '/TENSOR_REPLAY_ALLOCATION_v1.json': '49f279b2bb02b518322b8367f60965ce3a9ab6883275a6da6e9c4235c31624fe',
}
for rel, expected in anchors.items():
    add(rel, 'source_control', expected)

protocol = read_metadata(ROOT + '/STAGE1_COMPARATIVE_PROTOCOL_v1.json')
queue = read_metadata(ROOT + '/STAGE1_SERIAL_QUEUE_MANIFEST_v2.json')
sources = read_metadata(ROOT + '/CS_ACQUISITION_SOURCES_v1.json')
spec = read_metadata(ASSESS + '/SPECIFICATION_v2.json')
adoption = read_metadata(ASSESS + '/ROOT_ADOPTION_v2.json')
template = read_metadata(REPLAY + '/REQUEST_TEMPLATE_v2.json')
replay_adoption = read_metadata(ROOT + '/TENSOR_REPLAY_ROOT_ADOPTION_v1.json')
allocation = read_metadata(ROOT + '/TENSOR_REPLAY_ALLOCATION_v1.json')
require(Path(queue['remote_phase']) == REMOTE and Path(spec['remote_phase']) == REMOTE,
        'Remote phase differs')
require(protocol['cells'] == spec['expected_cells'] and protocol['recipes'] == spec['expected_recipes'],
        'Protocol/specification cell or recipe freeze differs')
require(adoption['specification_sha256'] == anchors[ASSESS + '/SPECIFICATION_v2.json'] and
        adoption['validator_sha256'] == anchors[ASSESS + '/validate_stage1_v2.py'], 'Assessment adoption differs')
require(replay_adoption['source_sha256'] == anchors[REPLAY + '/replay_selected_v2.py'] and
        replay_adoption['request_template_sha256'] == anchors[REPLAY + '/REQUEST_TEMPLATE_v2.json'] and
        replay_adoption['tolerances'] == template['tolerances'], 'Replay source/template adoption differs')

# Exact current queue references and superseded Stage1 controls are retained.
bindings(queue, 'queue_v2')
prior_queue = read_metadata(queue['prior_unexecuted_queue']['path'])
bindings(prior_queue, 'prior_queue_v1')
for b in queue['batches']:
    request = read_metadata(b['request']['path'])
    bindings(request, 'batch_v2_' + str(b['number']))
for b in prior_queue['batches']:
    request = read_metadata(b['request']['path'])
    bindings(request, 'prior_batch_v1_' + str(b['number']))
superseded = read_metadata(queue['superseded_unexecuted_request']['path'])
bindings(superseded, 'superseded_stage1_request')
for key in ['allocation_amendment', 'forecast_adoption', 'admission_precedence']:
    bindings(read_metadata(queue[key]['path']), key)
bindings(allocation['prior_allocation'], 'replay_prior_stage1_allocation')
for key in ['protocol', 'queue', 'runner', 'expected_sources', 'launcher', 'staged_adoption',
            'inherited_design_reference']:
    bind(spec[key], 'assessment_spec/' + key)
for item in sources['files']:
    bind(item, 'frozen_source_manifest')
for item in template['protected_files']:
    bind(item, 'effective_replay_protected')
for key in ['protocol', 'sources', 'stdlib_specification', 'stdlib_adoption']:
    bind(template[key], 'replay_template/' + key)

# Include effective package documentation and immutable hash manifests; do not
# expand historical calibration/qualification output references within them.
for package in [ASSESS, REPLAY]:
    hashes = read_metadata(package + '/SHA256SUMS_v2.json')
    for item in hashes['files']:
        add(package + '/' + item['path'], 'source_control', item['sha256'], item['bytes'])

requests = [read_metadata(b['request']['path']) for b in queue['batches']]
operations = [operation for request in requests for operation in request['operations']]
require(len(operations) == 54 and [(o['dataset'], *[o['cell'][k] for k in ['split', 'seed', 'arm', 'recipe']]) for o in operations] ==
        [(c['dataset'], *[c[k] for k in ['split', 'seed', 'arm', 'recipe']]) for c in protocol['cells']],
        'Exact 54 operation identities/order differ')
identity = {(c['dataset'], c['arm'], c['seed'], c['split']) for c in protocol['cells']}
expected_identity = {(d, a, s, 'core0') for d in ['AmazonPhoto', 'CoauthorCS']
                     for a in ['original', 'factor', 'permutation', 'bias_only', 'coordinate',
                               'single', 'untied', 'heads', 'gt_sep_single'] for s in [17, 29, 43]}
require(len(protocol['cells']) == 54 and identity == expected_identity, 'Frozen 54-cell coverage differs')
fit_root = relative(protocol['phases']['fit']['output_root'])
suffixes = {f'{c["dataset"]}__core0__seed{c["seed"]}__{c["arm"]}__{c["recipe"]}' for c in protocol['cells']}
actual_dirs = list(safe(fit_root).iterdir())
require(all(p.is_dir() and not p.is_symlink() for p in actual_dirs) and
        {p.name for p in actual_dirs} == suffixes, 'Fetched fit directory coverage differs')
snapshots = [f'verified_sources/{i:03d}_{Path(item["path"]).name}' for i, item in enumerate(sources['files'])]
expected_payloads = {'cell_report.json', 'training_curve.jsonl', 'invocation.json', 'model_storage.json',
                     'frozen_protocol.json', 'frozen_sources.json', 'frozen_data_manifest.json',
                     'source_snapshots.json', 'initial_model_state.pt', 'selected_state.pt', 'final_state.pt',
                     'selected_validation_member_logits.pt', 'final_validation_member_logits.pt',
                     'selected_deployment_state.pt', 'wall_clock_receipt.json', *snapshots}
for n, suffix in enumerate(sorted(suffixes), 1):
    cell_root = fit_root + '/' + suffix
    manifest = read_metadata(cell_root + '/artifacts.json')
    payloads = manifest['files']
    require(manifest['schema_version'] == 1 and len(payloads) == 22 and
            len({i['path'] for i in payloads}) == 22 and {i['path'] for i in payloads} == expected_payloads,
            'Cell artifact inventory differs: ' + suffix)
    physical = []
    for p in safe(cell_root).rglob('*'):
        require(not p.is_symlink(), 'Symlink cell payload: ' + str(p))
        if p.is_file():
            physical.append(p.relative_to(safe(cell_root)).as_posix())
        else:
            require(p.is_dir(), 'Non-file cell payload: ' + str(p))
    require(set(physical) == expected_payloads | {'artifacts.json'}, 'Extra/missing cell payload: ' + suffix)
    add(cell_root + '/artifacts.json', 'fit_payload')
    for item in payloads:
        require(relative(item['path']) == item['path'], 'Noncanonical artifact path')
        add(cell_root + '/' + item['path'], 'fit_payload', item['sha256'], item['bytes'])
    for filename, expected_sha in [('frozen_protocol.json', anchors[ROOT + '/STAGE1_COMPARATIVE_PROTOCOL_v1.json']),
                                   ('frozen_sources.json', anchors[ROOT + '/CS_ACQUISITION_SOURCES_v1.json']),
                                   ('frozen_data_manifest.json', protocol['data_bindings'][suffix.split('__')[0]]['data_manifest_sha256'])]:
        require(entries[cell_root + '/' + filename]['sha256'] == expected_sha, 'Cell frozen binding differs')
    for source, snapshot in zip(sources['files'], snapshots):
        require(entries[cell_root + '/' + snapshot]['sha256'] == source['sha256'], 'Source snapshot differs')
    print(f'Hashed fit artifacts: {n}/54', flush=True) if n % 9 == 0 else None

for b in queue['batches']:
    for key, count in [('remote_bound', 3), ('remote_supervisor', 5), ('remote_bridge', 74)]:
        add_tree(b[key], 'batch_supervision', count)
add_tree(LAUNCHER, 'local_queue_receipts', 11)

# Only these four public tensors are eligible. The manifest's primitive train/test
# declarations remain in the public manifest, but none of those payloads are opened.
public_tensors = []
for dataset in ['AmazonPhoto', 'CoauthorCS']:
    public_root = ROOT + '/acquire/public/' + dataset
    manifest_rel = public_root + '/data_manifest.json'
    add(manifest_rel, 'public_manifest', protocol['data_bindings'][dataset]['data_manifest_sha256'])
    manifest = read_metadata(manifest_rel)
    for item in [manifest['graph'], manifest['splits']['core0']['validation']]:
        require(item['path'] in ['graph.pt', 'core0/validation.pt'], 'Unexpected public replay tensor')
        public_tensors.append(add(public_root + '/' + item['path'], 'public_replay_tensor', item['sha256'], item['bytes']))

require(len(public_tensors) == 4, 'Public tensor count differs')
public_prefix = ROOT + '/acquire/'
allowed_public = {i['path'] for i in public_tensors} | {
    ROOT + '/acquire/public/' + d + '/data_manifest.json' for d in ['AmazonPhoto', 'CoauthorCS']}
require({rel for rel in entries if rel.startswith(public_prefix)} == allowed_public, 'Unadmitted public input indexed')
require(not any('/sealed/' in rel or '/processed/' in rel or '/raw/' in rel for rel in entries), 'Unadmitted input indexed')
require(INDEX not in entries, 'Index cannot include itself')
index = {'schema': 'coordinate-stage1-evidence-index-v1', 'queue_launcher_receipt_root': LAUNCHER,
         'files': [entries[rel] for rel in sorted(entries)]}
index_record = write_new(INDEX, index)

group_counts = Counter(groups.values())
group_bytes = {group: sum(item['bytes'] for rel, item in entries.items() if groups[rel] == group)
               for group in sorted(group_counts)}
review = {
    'schema': 'coordinate-stage1-evidence-index-build-review-v1', 'evidence_index': index_record,
    'indexed_file_count': len(entries), 'indexed_total_bytes': sum(i['bytes'] for i in entries.values()),
    'coverage': {'frozen_cells': 54, 'payloads_per_cell_including_artifacts_manifest': 23,
                 'batch_bound_roots': 3, 'batch_inner_supervisor_roots': 3, 'batch_bridge_roots': 3,
                 'public_replay_tensors': public_tensors},
    'group_file_counts': dict(sorted(group_counts.items())), 'group_total_bytes': group_bytes,
    'bound_hash_and_length_checks': binding_checks,
    'metadata_json_files_parsed': sorted(set(metadata_read)),
    'consumer_compatibility': {'stdlib_validator_v2': 'same schema and phase-relative paths accepted statically',
                               'native_replay_v2': 'same schema and phase-relative paths accepted statically',
                               'consumer_executed': False},
    'inspection_scope': 'JSON control/artifact/public manifest metadata and streaming file hashes only',
    'comparative_reports_curves_or_metrics_parsed': False, 'tensor_deserialization': False,
    'model_import_or_execution': False, 'validator_execution_or_compilation': False, 'ssh_used': False,
    'historical_reference_expansion': 'Historical calibration/qualification outputs, prior unrelated allocation, preserved superseded native source and previous validator package are not expanded into this execution index.',
    'excluded_payload_classes': ['raw', 'processed PyG', 'sealed', 'training targets/indices',
                                 'test targets/indices', 'core1/core2 inputs', 'historical calibration/qualification payloads'],
}
review_record = write_new(REVIEW + '/EVIDENCE_INDEX_BUILD_REVIEW_v1.json', review)
upload_entries = [entries[rel] for rel in sorted(entries) if groups[rel] in ['source_control', 'local_queue_receipts']]
upload = {'schema': 'coordinate-stage1-evidence-source-upload-inventory-v1', 'remote_phase': str(REMOTE),
          'evidence_index': index_record,
          'files': [index_record, *upload_entries],
          'file_count': len(upload_entries) + 1,
          'total_bytes': sum(i['bytes'] for i in upload_entries) + index_record['bytes'],
          'policy': 'These are the exact indexed source/control files and local launcher receipts to materialize at the same phase-relative remote paths. Remote-origin fit, batch supervision and public input payloads are already present remotely and are excluded from this small upload list.',
          'concrete_replay_request': 'Not created by this metadata build; root must bind index hash and allocation into a new immutable admitted request using effective REQUEST_TEMPLATE_v2.json.'}
upload_record = write_new(REVIEW + '/EVIDENCE_SOURCE_UPLOAD_INVENTORY_v1.json', upload)
print(json.dumps({'index': index_record, 'review': review_record, 'upload_inventory': upload_record,
                  'indexed_files': len(entries), 'indexed_total_bytes': review['indexed_total_bytes'],
                  'group_file_counts': review['group_file_counts'], 'group_total_bytes': group_bytes,
                  'upload_files': upload['file_count'], 'upload_total_bytes': upload['total_bytes']}, indent=2))

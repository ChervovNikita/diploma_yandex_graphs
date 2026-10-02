"""Allowed metadata-only checker. Never imports/compiles scientific drafts.

Reads only bound source and JSON/Markdown receipt bytes. No dataset archive,
arrays, weights, model/provider execution, network, SSH or installation.
Run without flags for the stable pre-seal checks, or --verify-seal for read-only
payload/seal verification. JSON goes to stdout; this file writes nothing.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = '3b9b115490249cc777227c846babfb55f35bd8c4'
DGL_COMMIT = 'c6c874bf7ea085beb04ea1487cfd216a0bacd6c1'
ARCHIVE_SHA = '90c38ab363dd57ce67ec62e1501532a5189403119c9cf4791a91f127321a3ad6'


def digest(path):
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    if sys.argv[1:] not in ([], ['--verify-seal']):
        raise ValueError('Only optional --verify-seal is admitted')
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        if not condition:
            raise ValueError(name)

    bindings = load('SOURCE_BINDINGS.json')
    inputs = bindings['external_inputs'] + bindings['local_draft_sources']
    for item in inputs:
        path = Path(item['path'])
        check('source/metadata suffix: ' + path.name,
              path.suffix in ('.py', '.json', '.md', '.toml'))
        sha, size = digest(path)
        check('bound bytes: ' + path.name, sha == item['sha256'] and size == item['bytes'])
    external = {Path(item['path']).name: Path(item['path'])
                for item in bindings['external_inputs']}
    tree = json.loads(external['graphpfn_tree.json'].read_text())
    tree_blobs = {row['path']: row['sha'] for row in tree['tree']}
    old_receipts = json.loads(external['SOURCE_RECEIPTS.json'].read_text())['author_source']
    for name in ('graphpfn_paper_gnn.py', 'graphpfn_paper_gat_tolokers2_tuning.toml',
                 'graphpfn_paper_pyproject.toml', 'graphpfn_paper_ft10_tolokers2_evaluation.toml'):
        item = next(row for row in old_receipts if Path(row['descriptor']['path']).name == name)
        data = external[name].read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        check('earlier native-source receipt: ' + name,
              item['commit'] == COMMIT and item['imported_or_executed'] is False
              and item['matches_saved_pinned_tree_blob'] is True
              and hashlib.sha256(data).hexdigest() == item['descriptor']['sha256']
              and len(data) == item['descriptor']['bytes']
              and blob == item['git_blob_sha1'] == tree_blobs[item['remote_path']])
    sources = load('SOURCE_RETRIEVALS.json')
    check('four delegated native sources', len(sources) == 4)
    for item in sources:
        path = ROOT / item['path']
        data = path.read_bytes()
        git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        check('native source receipt: ' + item['remote_path'],
              item['status'] == 200 and item['commit'] == COMMIT
              and len(data) == item['bytes']
              and hashlib.sha256(data).hexdigest() == item['sha256']
              and git_blob == item['actual_git_blob_sha1'] == item['tree_git_blob_sha1']
              == tree_blobs[item['remote_path']] and item['matches_pinned_tree'] is True)
    dgl = load('DGL_SOURCE_RECEIPTS.json')
    check('DGL release receipts retained', len(dgl) == 4)
    for item in dgl:
        sha, size = digest(ROOT / item['path'])
        check('DGL receipt bytes: ' + item['path'], sha == item['sha256'] and size == item['bytes'])
        if item['path'].endswith('.py'):
            check('DGL source pin: ' + item['path'],
                  item['status'] == 200 and item['commit'] == DGL_COMMIT and item['release'] == 'v2.4.0')
    check('failed bare tag retained', dgl[0]['status'] == 404)
    tag = load('sources/dgl_tag_v2_4_0.json')
    check('DGL v2.4.0 commit tag', tag['object'] == {
        'sha': DGL_COMMIT, 'type': 'commit',
        'url': 'https://api.github.com/repos/dmlc/dgl/git/commits/' + DGL_COMMIT})
    archive = json.loads(external['ARCHIVE_RECEIPT.json'].read_text())
    check('archive receipt identity only', archive['dataset'] == 'tolokers-2'
          and archive['bytes'] == 3361114 and archive['sha256'] == ARCHIVE_SHA
          and archive['verified_provider_md5'] is True
          and archive['arrays_or_labels_decoded'] is False
          and archive['extracted'] is False and archive['trained'] is False)
    check('archive inventory public/target members', {
        'tolokers-2/info.yaml', 'tolokers-2/features.csv', 'tolokers-2/edgelist.csv',
        'tolokers-2/targets.csv', 'tolokers-2/split_masks_RL.csv'
    } <= {row['name'] for row in archive['inventory']})

    r17 = next(Path(row['path']).parent for row in bindings['external_inputs']
               if row['path'].endswith('round17_graph_init_driver_integration_v2/MANIFEST.json'))
    r17_manifest = json.loads((r17 / 'MANIFEST.json').read_text())
    for row in r17_manifest['payload']:
        sha, size = digest(r17 / row['path'])
        check('unchanged R17 payload: ' + row['path'], sha == row['sha256'] and size == row['bytes'])
    parent = (r17 / 'prototype/graph_band_route_initializer.py').read_text()
    child = (ROOT / 'route_initializer.py').read_text()
    lineage = load('INITIALIZER_LINEAGE.json')
    check('unchanged R17 initializer pin', digest(r17 / 'prototype/graph_band_route_initializer.py')[0]
          == lineage['parent_sha256'] == '1a8036c7bbf9f2b831747f99d3aa2dfdabc31cb636cd41ccad9457206a70cbcf')
    for name in ('symmetric_normalized_adjacency', 'bernstein_cubic_bands',
                 'permute_topology_nodes', 'initialize_four_routes', 'qualify_gradient_interface'):
        pattern = r'^def ' + name + r'\([^\n]*.*?(?=^def |\Z)'
        p = re.search(pattern, parent, re.M | re.S)
        c = re.search(pattern, child, re.M | re.S)
        check('borrowed function bytes: ' + name, p is not None and c is not None and p.group() == c.group())
    parent_constants = re.findall(r'^[A-Z_]+ = .+$', parent, re.M)
    check('borrowed constants unchanged', parent_constants == re.findall(r'^[A-Z_]+ = .+$', child, re.M))

    protocol, matrix, policy = load('PROTOCOL.json'), load('CONFIG_MATRIX.json'), load('ROLE_POLICY.json')
    check('closed unexecuted protocol', protocol['status'] == 'PREPARED_UNEXECUTED'
          and protocol['execution_authorized_by_packet'] is False
          and protocol['new_primary_papers_read'] == 0 and policy['test_label_state'] == 'CLOSED')
    check('exact seeds/config order', matrix['seeds'] == [17, 29, 43]
          and matrix['configuration_order'] == ['c0', 'c1', 'c2']
          and len(matrix['configurations']) == 3)
    expected_cells = [(c, s) for c in ['c0', 'c1', 'c2'] for s in [17, 29, 43]]
    check('nine-cell fit matrix', [(row['config'], row['seed']) for row in matrix['fit_cells']] == expected_cells
          and matrix['maximum_native_fits'] == 9 and matrix['maximum_native_updates'] == 9000)
    for row, lr, frac in zip(matrix['configurations'], [3e-4, 1e-3, 3e-4], ['none', 'none', 'quantile-normal']):
        config = row['native_config']
        backbone = config['model']['backbone']
        check('frozen config: ' + row['id'],
              config['n_steps'] == 1000 and config['patience'] == 100 and config['amp_dtype'] is None
              and config['optimizer']['type'] == 'AdamW' and config['optimizer']['lr'] == lr
              and config['optimizer']['weight_decay'] == 0
              and config['transform']['features'] == {'seed': 0, 'cat_policy': 'one-hot',
                  'num_policy': 'quantile-normal', 'frac_policy': frac}
              and config['model']['n_classes'] == 2 and config['model']['num_features'] is None
              and config['model']['pearl'] is None and config['model']['inp_out_type'] == 'mlp'
              and backbone == {'type': 'BaseGraphBackbone', 'conv_name': 'gat', 'norm_name': 'layer',
                  'residual': True, 'n_layers': 3, 'd_hidden': 512, 'dropout': 0.1, 'activation': 'gelu'}
              and config['data']['graph_self_loops'] == 'none_source_faithful')
    extension = protocol['extension']
    check('bounded extension metadata', extension['arms'] == ['graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy']
          and extension['seeds'] == matrix['seeds'] and extension['continuation_cells'] == 15
          and extension['warm_updates'] == 50 and extension['max_continuation_updates_per_cell'] == 950
          and extension['seed_slices'] == ['stem.S', 'head.R'] and extension['slice_dimensions'] == [512, 512])
    check('useful update ceiling', protocol['budget']['maximum_useful_optimizer_updates']
          == 9 * 1000 + 15 * 950 + 3 * 950 == 26100)
    gates = load('QUALIFICATION_GATES.json')
    check('explicit unresolved gate registry', [row['id'] for row in gates['gates']]
          == ['G' + str(i) for i in range(11)] and gates['status'] == 'ALL_RUNTIME_GATES_UNEXECUTED')
    custodian = (ROOT / 'closed_label_preparation.py').read_text()
    stage2 = custodian[custodian.index('def derive_compact_trainval('):]
    check('isolation guard precedes target I/O and output',
          stage2.index("if admission.get('worker_mount_excludes_raw_archive_and_targets') is not True:")
          < stage2.index('out=Path(output_dir)') < stage2.index('with _archive(archive)'))
    check('test semantics skipped before float conversion', stage2.index('if role is None:continue') < stage2.index('y=float(raw)'))
    check('no scientific artifacts in source packet', not any(
        f.suffix.lower() in {'.zip', '.tar', '.gz', '.pt', '.pth', '.npy', '.npz', '.pkl'}
        for f in ROOT.rglob('*') if f.is_file()))
    if sys.argv[1:] == ['--verify-seal']:
        manifest = load('MANIFEST.json')
        seal = load('SEAL.json')
        check('manifest hash in seal', digest(ROOT / 'MANIFEST.json')[0] == seal['manifest_sha256'])
        check('closed source seal', seal['status'] == 'SEALED_PREPARED_UNEXECUTED'
              and seal['execution_authorized'] is False)
        expected = {row['path'] for row in manifest['payload']}
        actual = {str(f.relative_to(ROOT)) for f in ROOT.rglob('*') if f.is_file()
                  and f.name not in {'MANIFEST.json', 'SEAL.json'}}
        check('exact sealed payload inventory', expected == actual and manifest['payload_files'] == len(expected))
        for row in manifest['payload']:
            sha, size = digest(ROOT / row['path'])
            check('sealed payload bytes: ' + row['path'], sha == row['sha256'] and size == row['bytes'])
    return {'schema': 'tolokers2-source-metadata-checks-v1', 'passed': True,
            'check_count': len(checks), 'checks': checks,
            'scientific_drafts_imported_or_executed': False, 'compilation_performed': False,
            'archive_bytes_opened': False, 'data_or_label_arrays_read': False,
            'runtime_qualification_performed': False,
            'scope_limit': 'Hash/receipt/protocol/source-text consistency only; no syntax or numerical correctness claim.'}


if __name__ == '__main__':
    print(json.dumps(main(), indent=2) + '\n')

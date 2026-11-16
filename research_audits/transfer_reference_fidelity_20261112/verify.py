"""Verify reference sources, official data, and native model identity without training."""
import copy
import hashlib
import io
import json
import subprocess
import unittest
import zipfile
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

protected = [ROOT / name for name in (
    'ablation/campaign/DESIGN.md',
    'ablation/campaign/plan.py',
    'ablation/campaign/plans/rebuttal.json',
    'ablation/campaign/plans/rebuttal.counts.json',
    'ablation/p09_molecular/models.py',
    'ablation/p09_molecular/protocol.py',
    'ablation/p11_collab_buddy/models.py',
    'ablation/p11_collab_buddy/protocol.py',
)]
before = {str(p.relative_to(ROOT)): digest_file(p) for p in protected}
report = {'scope': 'Reference fidelity audit; no optimizer updates or scientific metric scoring',
          'protected_before': before}

mol_root = ROOT / 'external_sources/ogb_molecular'
mol_manifest = json.loads((mol_root / 'manifest.json').read_text())
fresh_author_hashes = {
    'conv.py': 'a592513e7fb0f6ed308ab036a961fe99382d3b09bd1fe3a503d5d8e65ff2e936',
    'gnn.py': 'eeac03fd138e49f0799a6757fc06f024f97d6774f814cba4ddb7ae5ff333f8b9',
    'main_pyg.py': '26125c957250accc8b933ab274ba6fa5ec5e21f91cd61df785266852fd1a156f',
}
for name, expected in mol_manifest['local_sha256'].items():
    if digest_file(mol_root / name) != expected:
        raise RuntimeError('Molecular source differs from pin: ' + name)
for name, expected in fresh_author_hashes.items():
    local = (mol_root / name).read_bytes()
    original = local.replace(b'from .conv import', b'from conv import') if name == 'gnn.py' else local
    if digest_bytes(original) != expected or mol_manifest['original_sha256'][name] != expected:
        raise RuntimeError('Molecular source differs from freshly retrieved author source: ' + name)
report['molecular_source'] = {'commit': mol_manifest['commit'], 'local_files_verified': len(mol_manifest['local_sha256']),
                            'fresh_author_files_verified': list(fresh_author_hashes),
                            'only_gnn_patch': 'Relative package import for conv'}

buddy_manifest = json.loads((ROOT / 'ablation/p11_collab_buddy/upstream_manifest.json').read_text())
for name, expected in buddy_manifest['files'].items():
    local = ROOT / 'external_sources/buddy' / name
    author_blob = subprocess.check_output([
        '/usr/bin/git', '--git-dir=' + str(ROOT / '.transfer_setup_20261110/upstream_buddy.git'),
        'show', buddy_manifest['commit'] + ':' + name])
    if digest_file(local) != expected or digest_bytes(author_blob) != expected:
        raise RuntimeError('BUDDY source differs from pinned author commit: ' + name)
report['buddy_source'] = {'commit': buddy_manifest['commit'],
                          'author_files_verified': len(buddy_manifest['files']),
                          'patches_to_vendored_source': 0}

data_manifest = json.loads((ROOT / 'data/ogb_transfer_manifest.json').read_text())
data_checks = {}
for name, directory in [('ogbg-molbace', 'ogbg_molbace'), ('ogbl-collab', 'ogbl_collab')]:
    spec = data_manifest['datasets'][name]
    archive_path = ROOT / 'data' / spec['archive']
    if digest_file(archive_path) != spec['archive_sha256']:
        raise RuntimeError('Dataset archive differs from acquisition receipt: ' + name)
    checked = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = archive.namelist()
        for relative, expected in spec['raw_and_split_files'].items():
            local_path = ROOT / 'data' / directory / relative
            names = [candidate for candidate in archive_names if candidate.endswith('/' + relative)]
            if len(names) != 1:
                raise RuntimeError('Ambiguous official archive member: ' + relative)
            if digest_file(local_path) != expected or digest_bytes(archive.read(names[0])) != expected:
                raise RuntimeError('Raw data or split differs from official archive: ' + relative)
            checked.append(relative)
    data_checks[name] = {'archive_sha256': spec['archive_sha256'],
                          'official_download_url': spec['download_url'],
                          'raw_and_split_files_verified': checked,
                          'manifest_split_counts': spec['split_counts']}
report['data'] = data_checks
print('Pinned sources and official archive contents verified', flush=True)

from ablation.p09_molecular.training import load_dataset
from ablation.p09_molecular.models import build_model as build_molecular
from ablation.p09_molecular.protocol import REFERENCE as MOLECULAR_RECIPE
from ablation.p11_collab_buddy.models import BUDDY, GNNMBuddy
from ablation.p11_collab_buddy.protocol import namespace, RECIPE as BUDDY_RECIPE
from ablation.p11_collab_buddy.data import load_official
from ogb.linkproppred import PygLinkPropPredDataset
# The full legacy loader imports optional logging and obsolete PyG paths.
# Run its small filter function in isolation, as literal readable source.
# AST equality verifies that this function body is the pinned author's body.
import ast
from torch_geometric.utils import to_undirected

def filter_by_year(data, split_edge, year):
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge

author_tree = ast.parse((ROOT / 'external_sources/buddy/src/data.py').read_text())
audit_tree = ast.parse(Path(__file__).read_text())
author_function = next(node for node in author_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == 'filter_by_year')
audit_function = next(node for node in audit_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'filter_by_year')
if (isinstance(author_function.body[0], ast.Expr)
        and isinstance(author_function.body[0].value, ast.Constant)
        and isinstance(author_function.body[0].value.value, str)):
    author_function.body.pop(0)
if ast.dump(author_function, include_attributes=False) != ast.dump(audit_function, include_attributes=False):
    raise RuntimeError('Isolated author filter function differs structurally from pinned source')
report['author_preprocessing_comparison'] = {
    'function': 'filter_by_year',
    'literal_function_ast_matches_pinned_source': True,
    'full_legacy_loader_imported': False,
    'reason': 'Full loader imports an obsolete PyG namespace and the absent optional wandb logger',
    'environment_or_author_files_changed': False}

dataset, splits, evaluator, binding = load_dataset('ogbg-molbace', ROOT / 'data')
if binding['split_counts'] != data_manifest['datasets']['ogbg-molbace']['split_counts']:
    raise RuntimeError('Loaded molecular split differs from official manifest')
report['molecular_loaded_data'] = binding
batch = Batch.from_data_list([dataset[int(i)] for i in splits['train'][:4]])
mol_parity = {}
for architecture in ('gcn', 'gin'):
    native_config = dict(MOLECULAR_RECIPE, model=architecture, method='ens_m0')
    wrapped_config = dict(MOLECULAR_RECIPE, model=architecture, method='gnnm')
    torch.manual_seed(19)
    native = build_molecular(native_config, 1).eval()
    torch.manual_seed(19)
    wrapped = build_molecular(wrapped_config, 1).eval()
    differences = []
    with torch.no_grad():
        for block in (wrapped.input_be_block, wrapped.output_be_block):
            block.R.fill_(1)
            block.S.fill_(1)
        expected = native(batch)
        for route in range(4):
            actual = wrapped(batch, member=route)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            differences.append(float((actual - expected).abs().max()))
    mol_parity[architecture] = {'layers': 5, 'width': 300, 'routes': 4,
                                'input': 'Four real official TRAIN molecules',
                                'max_absolute_logit_difference': max(differences),
                                'native_function_preserved_at_identity_factors': True}
report['molecular_full_architecture_identity'] = mol_parity
print('Full molecular architecture identity verified', flush=True)

collab = PygLinkPropPredDataset(name='ogbl-collab', root=str(ROOT / 'data'))
official_graph = collab[0]
official_splits = collab.get_edge_split()
counts = {'train_positive': len(official_splits['train']['edge']),
          'valid_positive': len(official_splits['valid']['edge']),
          'valid_negative': len(official_splits['valid']['edge_neg']),
          'test_positive': len(official_splits['test']['edge']),
          'test_negative': len(official_splits['test']['edge_neg'])}
if counts != data_manifest['datasets']['ogbl-collab']['split_counts']:
    raise RuntimeError('Loaded Collab split differs from official manifest')
x, positive, weights, valid = load_official(ROOT / 'data', None)
reference_graph = Data(x=official_graph.x, num_nodes=official_graph.num_nodes)
reference_graph, reference_splits = filter_by_year(reference_graph, copy.deepcopy(official_splits), 2007)
torch.testing.assert_close(positive, reference_splits['train']['edge'], rtol=0, atol=0)
torch.testing.assert_close(weights, reference_splits['train']['weight'].reshape(-1), rtol=0, atol=0)
torch.testing.assert_close(x, official_graph.x, rtol=0, atol=0)
for key in ('edge', 'edge_neg', 'weight'):
    torch.testing.assert_close(valid[key], official_splits['valid'][key], rtol=0, atol=0)
report['collab_loaded_data'] = {'official_split_counts': counts, 'nodes': int(x.size(0)),
                               'features': int(x.size(1)), 'year_cutoff': 2007,
                               'filtered_train_positive': len(positive),
                               'year_filter_matches_author_function': True,
                               'validation_queries_unchanged': True}

torch.manual_seed(19)
native_buddy = BUDDY(namespace(), 128).eval()
torch.manual_seed(19)
wrapped_buddy = GNNMBuddy(128).eval()
inputs = (torch.rand(11, 8), torch.rand(11, 2, 128),
          torch.arange(11).float(), torch.ones(11))
with torch.no_grad():
    for block in (wrapped_buddy.label_lin_layer, wrapped_buddy.lin_feat, wrapped_buddy.lin):
        block.R.fill_(1)
        block.S.fill_(1)
    expected = native_buddy(*inputs)
    for route in range(4):
        torch.testing.assert_close(wrapped_buddy(*inputs, route=route), expected, rtol=0, atol=0)
report['buddy_full_architecture_identity'] = {
    'hidden_width': 1024, 'node_features': 128, 'routes': 4,
    'exact_logit_equality_at_identity_factors': True,
    'input': 'Synthetic eleven-query input with official feature dimension'}
print('Full BUDDY architecture and official Collab partition verified', flush=True)

tests = [
    'ablation.p09_molecular.test_p09.MolecularTests.test_native_identity_boundary_parity',
    'ablation.p09_molecular.test_p09.MolecularTests.test_ogb_split_metadata_interface',
    'ablation.p09_molecular.test_p09.MolecularTests.test_ensemble_uses_mean_logits_and_member_zero_base',
    'ablation.p11_collab_buddy.test_p11.BuddyTests.test_original_boundary_identity_matches_native',
    'ablation.p11_collab_buddy.test_p11.BuddyTests.test_bounded_hashing_is_exact_native_hashing',
    'ablation.p11_collab_buddy.test_p11.BuddyTests.test_ensemble_averages_selected_scores_before_official_metric',
]
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromNames(tests))
(OUT / 'cpu_checks.log').write_text(stream.getvalue())
report['focused_cpu_checks'] = {'tests': tests, 'ran': result.testsRun,
                                'failures': len(result.failures), 'errors': len(result.errors),
                                'passed': result.wasSuccessful()}
if not result.wasSuccessful():
    raise RuntimeError('Reference parity checks failed; see cpu_checks.log')

report['reference_and_campaign_differences'] = {
    'molecular_repeat_count': {'paper': 10, 'prepared_cohorts': 10,
                               'exact_author_seed_ids': 'Not specified in inspected paper'},
    'buddy_seed_policy': {'author_runner': 'set_seed(rep), rep ranges from zero to reps minus one',
                         'prepared_native': '1000 * cohort + member',
                         'prepared_gnnm': '1000 * cohort',
                         'identical_seed_sequence': False},
    'buddy_paper_repeat_count': 'Not established by inspected paper text; prepared five is not certified as paper-identical',
    'buddy_recipe': {'author_code_hidden_width': 1024, 'author_code_lr': 0.02,
                     'paper_search_width_range': [64, 512], 'paper_search_lr_range': [0.0001, 0.01],
                     'repository_labels_itself_reimplementation': True,
                     'exact_paper_recipe_reproduction': False},
    'training_horizon': {'author_reference_epochs': 100,
                         'campaign': 'minimum 100, patience 50, initial cap 1000, continuation cap 2000',
                         'published_100_epoch_readout_retained_separately': True,
                         'extended_readout_is_exact_reference_horizon': False},
    'training_rng': 'Adapters use resumable shuffle streams; trajectories are not certified bit-for-bit author-run reproductions',
    'wrapper': 'Input and output private factors are intentional changes; underlying native backbone function is preserved at identity factors',
    'new_six_arm_plan': 'Molecular contrastive arms and BUDDY contrastive-only remain to be implemented and qualified',
}
after = {str(p.relative_to(ROOT)): digest_file(p) for p in protected}
if before != after:
    raise RuntimeError('A protected model or campaign source changed during audit')
report['protected_files_unchanged'] = True
report['verdict'] = 'Pinned author-code and official-data fidelity verified; exact paper-protocol replication not yet established'
(OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'verdict': report['verdict'], 'cpu_checks': report['focused_cpu_checks'],
                  'molecular_identity': mol_parity, 'collab': report['collab_loaded_data'],
                  'report': str(OUT / 'audit.json')}, indent=2))

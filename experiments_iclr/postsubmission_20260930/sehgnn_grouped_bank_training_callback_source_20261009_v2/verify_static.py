"""Stdlib AST/source contracts only; no numerical or scientific execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

from bank_training import BankConfig, BankContractError, FAMILIES, TRIAL_SCALES, plain_metadata


def main():
    here = Path(__file__).resolve().parent
    source_path = here/'bank_training.py'
    source = source_path.read_text()
    tree = ast.parse(source)
    for path in here.glob('*.py'):
        compile(ast.parse(path.read_text()), str(path), 'exec')
    assert not {'torch', 'numpy', 'dgl', 'torch_sparse'} & set(sys.modules)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(getattr(node, 'module', '') in ('torch', 'numpy', 'dgl', 'torch_sparse')
                   or any(name.name.split('.')[0] in ('torch', 'numpy', 'dgl', 'torch_sparse') for name in node.names) for node in imports)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {node.name for node in classes['BankSession'].body if isinstance(node, ast.FunctionDef)}
    assert {'train_epoch', 'correct', 'native_forward', 'evaluate', 'snapshot', 'restore_selected', 'verify_optimizer'} <= methods
    assert set(FAMILIES) == {'director', 'actor', 'keyword'} and TRIAL_SCALES == (1.0,.5,.25,.125)
    assert 'torch.cuda.amp.autocast(enabled=False)' in source
    assert 'with torch.cuda.amp.autocast():' in source and 'self.scalar.scale(loss/4).backward()' in source
    assert 'self.optimizer = torch.optim.Adam' in source and source.count("'weight_decay': 0") == 2
    assert source.index('adapter.install_member_bank(') < source.index('self.optimizer = torch.optim.Adam')
    assert "score_mode='bernoulli_marginal_logits'" in source and "target_role='TRAIN'" in source
    assert '.make_credit_plan(' in source and '.accumulate_replay(' in source and '.build_private_direction(' in source and '.try_private_step(' in source
    scratch = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'scratch_native_state')
    scratch_source = ast.get_source_segment(source, scratch)
    assert '_buffers.update(buffers)' in scratch_source and '.copy_(' not in scratch_source and 'state_dict' not in scratch_source
    assert 'value[ids].to(self.rt' in source and '_active_cache = before_cache' in source
    assert 'self.rows[\'TRAIN\']' in source and "role == 'TRAIN' or (source is None and mode == 'eval')" in source
    assert 'self.bank.verify_shared_state_dict(saved' in source and 'self.bank.load_state_dict(saved' in source
    assert source.index('self.bank.verify_shared_state_dict(saved') < source.index('self.bank.load_state_dict(saved')
    assert 'for epoch in range(200):' in source and 'epoch-best_epoch > 50' in source
    assert "improved = scores['VALID']['BCE'] < best_loss" in source
    assert 'full_input_probability_pool' not in source and '.serve_full_input(' in source
    # Metadata-only regression witness, using a non-provider str subclass.
    class VersionText(str):
        def __str__(self):
            return self
    normalized = plain_metadata({'versions': {VersionText('torch'): VersionText('2.1.2')}})
    assert type(next(iter(normalized['versions']))) is str
    assert type(normalized['versions']['torch']) is str
    class SplitKey(str):
        def __hash__(self):
            return id(self)
        def __eq__(self, other):
            return self is other
    try:
        plain_metadata({SplitKey('same'): None, SplitKey('same'): None})
    except BankContractError:
        pass
    else:
        raise AssertionError('Normalized duplicate keys were not rejected')
    assert 'identity = plain_metadata(identity)' in source
    try:
        BankConfig().require_enabled()
    except BankContractError:
        pass
    else:
        raise AssertionError('Inactive default did not reject')
    sources_path = here/'SOURCE_BINDINGS.json'
    verified = []
    if sources_path.exists():
        for row in json.loads(sources_path.read_text())['files']:
            path = here.parent/row['path']
            assert path.stat().st_size == row['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
            verified.append(row['path'])
    record = dict(schema='SeHGNN-M4-bank-static-source-verification-v1', status='passed',
                  source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  checks=['AST compilation without execution', 'no numerical/provider imports', 'inactive default',
                          'concrete own/correction/eval/selected APIs', 'deduplicated Adam after install with zero decay',
                          'native own AMP and scaled mean-BCE backward', 'FP32 autocast-disabled source callbacks',
                          'all unchanged helper stages called', 'scratch buffer references restored without copying parameters',
                          'complete TRAIN row/canonical cache isolation locators', 'shared slow preflight before selected load',
                          'literal epoch/VALID selector/patience locators', 'factual helper serving only',
                          'plain checkpoint metadata subclass normalization witness'],
                  exact_dependency_hashes_verified=verified, model_provider_or_scientific_execution=False,
                  numerical_or_floating_tolerance_tests=False,
                  limit='Source/AST/inactive-default checks only. Native state/backward correctness, training, qualification, costs, competence and scientific effects remain unverified.')
    (here/'STATIC_VERIFICATION.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status='passed', source_sha256=record['source_sha256'], model_provider_or_scientific_execution=False)))


if __name__ == '__main__':
    main()

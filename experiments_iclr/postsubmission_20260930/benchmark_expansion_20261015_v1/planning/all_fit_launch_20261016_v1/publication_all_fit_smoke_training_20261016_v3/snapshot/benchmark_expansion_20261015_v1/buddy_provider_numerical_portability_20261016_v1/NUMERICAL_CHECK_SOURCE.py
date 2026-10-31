"""CPU-only validator tests against retained real cards and integer registers."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

assert socket.gethostname() == 'olig'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [
    'GPU-9c6c20fa-a620-479d-f72f-163ac74deef2', 'GPU-30283973-140d-9e7b-e780-0fcc487dec54']
root = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
e = root / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
folder = e / 'planning/all_fit_launch_20261016_v1'
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
import torch
torch.set_num_threads(2); torch.set_num_interop_threads(1)
validator = load(e/'buddy_native_provider_v4/static_binding.py', '_actual_static_validator')
source = load(validator.SOURCE, '_numeric_test_native_hashing')
hasher = source.ElphHashes(SimpleNamespace(max_hash_hops=2, floor_sf=False, minhash_num_perm=128, hll_p=8, use_zero_one=False))
saved = torch.load(folder/'BUDDY_STATIC_SEED2_CONSTRUCTOR.HOST_ONLY.pt', map_location='cpu', weights_only=False)
record = json.loads((folder/'BUDDY_STATIC_SEED2_CONSTRUCTOR.json').read_text())
actual = saved['cards']; original = actual.clone(); original_rng = torch.get_rng_state().clone()
native = SimpleNamespace(cards=actual, hashes=hasher, hash_tables={k: {'hll': v} for k,v in saved['hll'].items()})
result = validator.validate_static_inputs(torch, native, record['observed'], record['expected'])
assert torch.equal(actual, original) and torch.equal(original_rng, torch.get_rng_state())
tests = ['actual_native_cards_PASS', 'actual_cards_not_mutated', 'torch_RNG_unchanged']
canonical = torch.load(folder/'CANONICAL_BUDDY_CARDS.pt', map_location='cpu', weights_only=False)
native.cards = canonical; observed = copy.deepcopy(record['observed']); observed['cards'] = validator.tensor(canonical)
validator.validate_static_inputs(torch, native, observed, record['expected']); tests.append('canonical_float32_rounding_PASS')
def reject(name, value, observed):
    native.cards = value
    try: validator.validate_static_inputs(torch, native, observed, record['expected'])
    except ValueError: tests.append(name); return
    raise AssertionError('Expected rejection: ' + name)
bad = actual.clone(); bad[0,0] += .01
observed = copy.deepcopy(record['observed']); observed['cards'] = validator.tensor(bad)
reject('material_card_change_rejected', bad, observed)
bad = actual.clone(); bad[0,0] = float('nan'); observed['cards'] = validator.tensor(bad)
reject('nonfinite_card_rejected', bad, observed)
bad = actual.double(); observed['cards'] = validator.tensor(bad)
reject('changed_card_dtype_rejected', bad, observed)
observed = copy.deepcopy(record['observed']); observed['hash_tables']['1']['hll']['sha256'] = '0'*64
reject('changed_integer_hash_lineage_rejected', actual, observed)
out = {'schema': 'BUDDY-floating-cardinality-real-CPU-checks-v1', 'status': 'PASS', 'tests': tests,
       'source_sha256': hashlib.sha256((e/'buddy_native_provider_v4/static_binding.py').read_bytes()).hexdigest(),
       'actual_reference_validation': result, 'TEST_accessed': False, 'optimizer_updates': 0, 'model_forwards': 0}
path = folder/'BUDDY_STATIC_VALIDATOR_CHECKS.json'
with path.open('x') as f: json.dump(out, f, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(out))

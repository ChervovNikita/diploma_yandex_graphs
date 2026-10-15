"""Stdlib-only source, typed-descriptor and actual bridge protocol checks."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import runtime_bridge as bridge
import descriptor_binding as descriptor


def refuses(function):
    try: function()
    except (ValueError, RuntimeError): return
    raise AssertionError('Expected explicit refusal')


def main():
    root = HERE.parents[2]
    bridge.verify_sources(root)
    files = sorted(HERE.rglob('*.py'))
    for path in files: ast.parse(path.read_text(), filename=str(path))
    config_root = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
    phase = 'experiments_iclr/postsubmission_20260930'
    pin = {'path': config_root + '/parent.json', 'bytes': 3, 'sha256': 'a'*64}
    declaration = {'schema': 'native-route-BN-composition-v1',
        'policy': 'private_native_running_state_shared_affine_v1', 'matched_control_method_id': 'gnnm_private_bn_state_v3',
        'source_manifest_sha256': 'b'*64, 'banks_sha256': 'c'*64, 'parent_descriptor': pin, 'parent_job_sha256': 'd'*64}
    job = {'schema': 'native-full-epoch-fit-v1', 'family': 'ogb', 'TEST_access': False,
        'callback_module': config_root+'/callback.py', 'data_root': config_root+'/data',
        'recipe': {'_repository_root': config_root, '_redesign_root': config_root+'/'+phase+'/reference_wrapper_redesign_20261013_v1'},
        'normalization_composition': declaration}
    assert descriptor.normalization_authority(job, config_root) == pin
    rows = descriptor.audit_runtime_paths(job, config_root, phase)
    assert any(row['pointer'] == '/normalization_composition/parent_descriptor/path' for row in rows)
    for mutation in ('extra', 'outside', 'traversal', 'bad_hash', 'bad_bytes'):
        bad = copy.deepcopy(job)
        if mutation == 'extra': bad['normalization_composition']['unclassified'] = config_root+'/secret'
        elif mutation == 'outside': bad['normalization_composition']['parent_descriptor']['path'] = '/elsewhere/parent.json'
        elif mutation == 'traversal': bad['normalization_composition']['parent_descriptor']['path'] = config_root+'/x/../parent.json'
        elif mutation == 'bad_hash': bad['normalization_composition']['parent_job_sha256'] = 'g'*64
        else: bad['normalization_composition']['parent_descriptor']['bytes'] = True
        refuses(lambda bad=bad: descriptor.audit_runtime_paths(bad, config_root, phase))
    refuses(lambda: descriptor.rebind_job(job,config_root,phase,config_root,phase))

    # Exercise the original real parent source check, not a substitute claim.
    path = HERE.parent / 'portable_dispatcher_v3/runtime_link/adapters.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LinkSameCallMeanAdapter')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_check_runtime'))
    env = {'Path': Path, '__file__': str(path), 'receipt': bridge.receipt,
           'inside': lambda path, root: Path(path).resolve() if Path(path).resolve().is_relative_to(root) else (_ for _ in ()).throw(ValueError('outside'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), env)

    class Idle:
        def idle(self): pass
    class Normalized:
        def __init__(self):
            self.binding = {}; self.owner = Idle(); self.forward_calls = 0; self.root = root
            self.callback_receipt = bridge.receipt(HERE/'runtime_bridge.py')
            self.extension_receipt = bridge.receipt(HERE/'qualification_policy.py')
            self.policy_receipt = bridge.receipt(HERE/'qualification_policy.py')
            self.provider_manifest = bridge.receipt(HERE.parent/'portable_dispatcher_v3/MANIFEST.json')
            self.dependency_receipts = []
            self.parent_bind = self.parent_export = 0
        _check_runtime = env['_check_runtime']
        def bind_loaded_selection(self, payload, **context):
            self._check_runtime(context['runtime_binding'])
            assert payload['runtime_binding'] == context['runtime_binding']
            assert payload['adapter'] == {'full_composed_envelope': True}
            self.parent_bind += 1
        def selected_mean_export(self, actual, **context):
            self._check_runtime(context['runtime_binding']); self.parent_export += 1
            return {'runtime_binding': context['runtime_binding'], 'extra_forward_calls': 0}
        def load_state(self, value): self.loaded = copy.deepcopy(value)
        def train_epoch(self): return {'fixture': 'no native work'}
        def train_update(self): return {'fixture': 'no native work'}
    inner = Normalized()
    owner = bridge.ProducerAdapter(inner, {'family':'link'}, root, bridge.receipt(HERE/'MANIFEST.json'))
    public = bridge.build_runtime_binding(owner, root)
    assert public['composition_producer']['unchanged_helper_runtime'] == owner.helper_runtime
    context = {'selected_checkpoint': {'fixture':'checkpoint receipt'}, 'job_hash':'fixture-job-hash', 'runtime_binding':public}
    payload = {'runtime_binding': public, 'adapter': {'full_composed_envelope':True}, 'selector': {'selected_epoch':1}}
    original = copy.deepcopy(payload)
    owner.bind_loaded_selection(payload, **context)
    result = owner.selected_mean_export({}, selected_epoch=1, **context)
    assert result['runtime_binding'] == public and result['unchanged_helper_runtime_binding'] == owner.helper_runtime
    assert payload == original and inner.parent_bind == inner.parent_export == 1 and inner.forward_calls == 0
    bad = copy.deepcopy(public); bad['runtime_sha256']['run_fit.py'] = '0'*64
    refuses(lambda: owner.selected_mean_export({}, selected_epoch=1, **{**context,'runtime_binding':bad}))
    refuses(lambda: owner.bind_loaded_selection({**payload,'runtime_binding':owner.helper_runtime}, **context))
    owner.load_state({'fixture':'restore'})
    refuses(lambda: owner.selected_mean_export({}, selected_epoch=1, **context))
    owner.bind_loaded_selection(payload, **context); owner.train_update()
    refuses(lambda: owner.selected_mean_export({}, selected_epoch=1, **context))
    assert 'torch' not in sys.modules
    result = {'status':'PASS', 'python_source_files':len(files),
        'typed_normalization_parent_pointer_passed':True, 'malformed_or_extra_runtime_pointer_refused':True,
        'unsafe_receipt_relocation_refused':True, 'actual_parent_runtime_check_exercised':True,
        'outer_and_helper_runtime_views_both_preserved':True, 'caller_payload_unchanged':True,
        'changed_runtime_and_stale_selected_binding_refused':True, 'bridge_export_added_native_forwards':0,
        'Torch_imported':False,'native_qualification_performed':False,'remote_calls':0,'roster_regeneration':False}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__': main()

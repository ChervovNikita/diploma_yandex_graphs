"""Audit the exact finite link dispatch/export additions against frozen runtimev4."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'runtime_v4'

def main():
    declaration=json.loads((HERE/'PARENT_SOURCE.json').read_text())
    assert hashlib.sha256((PARENT/'MANIFEST.json').read_bytes()).hexdigest()==declaration['manifest_sha256']
    for row in declaration['files']:
        b=(PARENT/row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    unchanged=('bookkeeping.py','common.py','controllers.py','source_rng.py')
    assert all((HERE/name).read_bytes()==(PARENT/name).read_bytes() for name in unchanged)
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    adapter=(HERE/'adapters.py').read_text()
    a=adapter.index('class LinkSameCallMeanAdapter:');b=adapter.index('def create_adapter(torch, job, policy, root):',a)
    reverted=adapter[:a]+adapter[b:]
    new='''        adapter = _link_source_dependencies(adapter, pin, root)
        if adapter.extension_receipt is not None:
            provider, _ = load_module(adapter.extension_receipt['path'], root, adapter.extension_receipt['sha256'])
            return LinkSameCallMeanAdapter(torch, job, policy, root, adapter, provider)
        return adapter'''
    old='''        adapter.callback_receipt, adapter.extension_receipt = pin, None
        meta = adapter.metadata()["native_integration"]
        native = meta["native_binding"]
        adapter.dependency_receipts = [*meta["integration_sources"], *meta["finalized_helper_sources"],
            *native["adapter_source_receipts"].values(), native["adapter_manifest_receipt"]]
        return adapter'''
    assert new in reverted;reverted=reverted.replace(new,old)
    assert reverted==(PARENT/'adapters.py').read_text()
    run=(HERE/'run_fit.py').read_text()
    run=run.replace('def __init__(self, torch, owners, previous=None, maximum_completed=None):','def __init__(self, torch, owners, previous=None):',1)
    run=run.replace('        self.maximum_completed = maximum_completed\n','',1)
    run=run.replace('''                    if self.maximum_completed is not None and sum(self.completed) >= self.maximum_completed:
                        raise ValueError("Finite link CPU optimizer budget exhausted before another transition")
''','',1)
    a=run.index('    elif mode == "cpu_link_engineering_v1":');b=run.index('    elif mode != "production_gpu"',a);run=run[:a]+run[b:]
    a=run.index('        maximum_CPU_updates = None');b=run.index('        calls_at_invocation_start',a)
    run=run[:a]+'        observer = OptimizerObserver(torch, adapter.optimizers, previous_calls)\n'+run[b:]
    a=run.index('        bind_selection = getattr(adapter, "bind_loaded_selection", None)');b=run.index('        final_valid = adapter.validate()',a);run=run[:a]+run[b:]
    run=run.replace('"scientific_scope": "bounded engineering qualification; no scientific accuracy claim" if job.get("engineering_only") is True or device == "cpu" else',
                    '"scientific_scope": "bounded CPU engineering qualification; no scientific accuracy claim" if device == "cpu" else')
    assert run==(PARENT/'run_fit.py').read_text()
    return {'schema':'runtime-link-v5-static-contract-v1','status':'passed',
        'parent_manifest_sha256':declaration['manifest_sha256'],
        'unchanged_producers_byte_identical':list(unchanged),
        'adapters_entire_parent_text_exact_after_removing_declared_link_wrapper_and_source_receipt_dispatch':True,
        'run_fit_entire_parent_text_exact_after_removing_declared_CPU_guard_observer_cap_selected_binding_and_engineering_label':True,
        'Node_OGB_adapter_and_existing_mean_export_code_unchanged':True,
        'frozen_link_policy_and_method_sources_unchanged':True,
        'native_models':0,'native_forward_calls':0,'native_optimizer_updates':0,'TEST_accessed':False}

if __name__=='__main__':print(json.dumps(main(),indent=2))

"""Stdlib-only sealed source verification; no native audit or fixture execution."""
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    auditor = load('native15_prepared_auditor_static_check', HERE/'audit.py')
    provenance, manifest_sha = auditor.source_guard()
    compiled = []
    for path in [HERE/'audit.py', HERE/'verify_source.py'] + [PHASE/r['path'] for r in provenance['source_records'] if r['path'].endswith('.py')]:
        compile(path.read_text(), str(path), 'exec'); compiled.append(str(path.relative_to(PHASE)))
    for key in ('driver', 'inputs', 'loader', 'supervisor'):
        load('native15_original_'+key+'_static_check', PHASE/provenance['modules'][key])
    forbidden = sorted(name for name in sys.modules if name.split('.')[0] in ('torch', 'numpy', 'scipy', 'dgl', 'torch_geometric'))
    assert not forbidden, 'Numerical runtime imported: '+str(forbidden)
    result = dict(schema='native15_closed_auditor_static_source_receipt_v1', status='static_source_checks_passed',
        auditor_manifest_sha256=manifest_sha, source_hash_custody_verified=True, sources_compiled=compiled,
        stdlib_only_original_module_imports_verified=True, numerical_runtime_imported=False,
        audit_or_fixture_or_fit_execution=False, dataset_or_label_or_tensor_or_live_outcome_reads=False,
        server_access=False, execution_authorized=False, independent_source_critic_required=True,
        numerical_replay_auditor_qualified=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

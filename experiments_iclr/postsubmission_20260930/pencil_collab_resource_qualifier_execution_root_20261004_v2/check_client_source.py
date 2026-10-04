"""Local stdlib metadata/hash/AST checks; never connects or executes generated code."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    client = load('pencil_root_client_source_check', 'stage_admit_resource.py')
    monitor = load('pencil_root_monitor_source_check', 'monitor_owned_resource.py')
    review = client.PHASE / client.REVIEW_DIRECTORY
    plan, binding = client.reviewed_plan(review, client.REVIEW_SEAL, 'REVIEW.json')
    source_files = client.packet(client.SOURCE, client.SOURCE_SEAL, client.SOURCE_MANIFEST)[0]
    review_files = client.packet(review, client.REVIEW_SEAL, client.REVIEW_MANIFEST)[0]
    external = json.loads((client.SOURCE / 'INPUT_BINDINGS.json').read_text())['inputs']
    for row in external:
        client.pin(client.PHASE / row['path'], row)
    for row in client.TRANSPORT_PINS:
        client.pin(client.PHASE / row['path'], row)
    rejections = []
    for seal in ('0' * 64, ''):
        try:
            client.reviewed_plan(review, seal, 'REVIEW.json')
        except RuntimeError:
            rejections.append('wrong_or_absent_review_seal_rejected')
        else:
            raise RuntimeError('Review seal negative guard failed')
    expected = dict(source_manifest_sha256=client.SOURCE_MANIFEST, release_sha256='0' * 64,
                    binding=binding, supervisor_PID=1000, supervisor_start_ticks=12345,
                    supervisor_session=1000, supervisor_group=1000,
                    command=['NONEXECUTED_AST_PLACEHOLDER'])
    snippets = dict(context=client.context(),authentication=client.authenticated_remote(binding),
                    staging_join=client.stage_join_code(binding, '0' * 64, 1),
                    dependency_admission=client.admission_code(binding),
                    gate=client.context() + client.remote_gate('0' * 64),
                    launch=client.launch_code(binding, '0' * 64, '1' * 64),
                    monitor=monitor.monitor_code(expected),
                    fresh_stdlib_gate=client.gate_command('0' * 64)[-1])
    rows = []
    for name, text in snippets.items():
        ast.parse(text, filename=name)
        rows.append(dict(name=name,bytes=len(text.encode()),AST='PASS',executed=False))
    for name in ('stage_admit_resource.py','monitor_owned_resource.py','check_client_source.py'):
        ast.parse((HERE / name).read_text(), filename=name)
    client.require(not any(name in sys.modules for name in ('torch','numpy','transformers','torch_geometric')),
                   'Numerical import occurred during local source-only check')
    value = dict(status='PASS_LOCAL_AUTHOR_CLIENT_METADATA_HASH_AND_AST_ONLY',binding=binding,
                 source_payloads=63,source_packet_files=len(source_files),review_packet_files=len(review_files),
                 bound_external_inputs=len(external),total_exact_stage_files=len(set(source_files + review_files + [client.PHASE / r['path'] for r in external])),
                 transport_pins_verified=len(client.TRANSPORT_PINS),remote_snippets=rows,negative_review_guards=rejections,
                 separate_stage_admit_launch_modes=True,source_modified=False,numerical_import=False,
                 generated_code_executed=False,connection=False,staging=False,launch=False,
                 independent_client_review_or_runtime_PASS_claimed=False)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__ == '__main__':
    main()

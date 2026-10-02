"""Static AST/JSON/hash checker. Never imports the builder or custody/science."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
PACKET = PHASE/'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
EXPECTED = 'e2fd767c4d08e93ca7ec9c9c526440c4cc5b8f4e7d8bb345de90308e9c57302b'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):
    require(path.suffix == '.json','JSON metadata only')
    return json.loads(path.read_text())


def sha(path):
    require(path.suffix in {'.json','.py','.md','.diff','.sh','.txt','.log','.jsonl'},'No scientific payload access')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(record):
    remote = Path(record['path']);require(remote.is_relative_to(REMOTE),'Confined original descriptor required')
    local = (PHASE/remote.relative_to(REMOTE)).resolve()
    require(local.is_relative_to(PHASE) and sha(local) == record['sha256'],'Metadata descriptor differs')
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True);args = parser.parse_args()
    prepared = Path(args.prepared).resolve();output = Path(args.output).resolve()
    require(prepared.is_relative_to(HERE) and output.is_relative_to(HERE),'Checker outputs confined to certification_v3')
    tree = ast.parse((HERE/'build_certificates.py').read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module)
    require(set(imports) <= {'__future__','argparse','copy','hashlib','importlib.util','json','pathlib','sys'},'Non-stdlib builder import')
    issue = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'issue')
    text = ast.get_source_segment((HERE/'build_certificates.py').read_text(),issue)
    require(text.index("approval['approved'] is True") < text.index('spec.loader.exec_module(custody)') < text.index('custody.certify('),
        'Root approval must precede stdlib custody loading/issuance')
    require(sha(PACKET/'MANIFEST.json') == EXPECTED,'Sealed modern v3 source manifest changed')
    for row in read(PACKET/'MANIFEST.json')['payload']:
        require(sha(PACKET/row['path']) == row['sha256'],'Sealed modern v3 payload changed')
    index = read(prepared/'INDEX.json');require(len(index['rows']) == 18 and index['certificate_count'] == 0,'Only eighteen ineligible drafts expected')
    require(not (prepared/'certificates').exists(),'No eligible certificates may exist in preparation')
    expected = {(b,f,s) for b in ('polyformer_mono','polynormer_r') for f in
        ('single_author','gnnm_boundary_4','independent_author_4_same_width') for s in (17,29,43)}
    require({(r['backbone'],r['family'],r['seed']) for r in index['rows']} == expected,'Wrong/duplicate prepared target set')
    audit_paths = set();coverage_paths = set()
    for row in index['rows']:
        request = read(bound(row['request']));audit_path = bound(row['external_audit']);audit = read(audit_path)
        coverage_path = bound(row['coverage']);coverage = read(coverage_path);feasibility = read(bound(row['feasibility_template']))
        audit_paths.add(audit_path);coverage_paths.add(coverage_path)
        require(request['target']['execution_authorized'] is False and request['target']['root_signature'] is None and
            request['root_approval'] is None and request['target']['config'] == row['config'] == 0 and
            request['target']['implementation_sha256'] == read(PACKET/'IMPLEMENTATION_BINDINGS.json'),'Target draft cannot authorize execution')
        require(audit['schema'] == 'modern-external-runtime-audit-v2' and audit['root_evidence_accepted'] is False and
            audit['root_signature'] is None and audit['numerical_tested_seeds'] == [17] and audit['numerical_tested_configurations'] == [0] and
            audit['partial_run_binding']['schema'] == 'modern-legacy-qualification-run-binding-v3' and
            audit['partial_receipt'] == request['partial_receipt'] and audit['coverage_authorization'] == row['coverage'],
            'Legacy audit draft linkage/scope/root status differs')
        for value in audit['partial_run_binding'].values():
            if isinstance(value,dict):bound(value)
        require(coverage['schema'] == 'modern-exact-role-label-coverage-v2' and coverage['root_coverage_authorized'] is False and
            coverage['root_signature'] is None and coverage['numerical_tests_on_untested_seeds'] is False and
            coverage['numerical_tested_configurations'] == [0] and coverage['tested_seed17_input']['seed'] == 17 and
            [r['seed'] for r in coverage['authorized_target_inputs']] == [17,29,43] and
            row['input_identity'] in coverage['authorized_target_inputs'],'Coverage scope/target differs')
        require(feasibility['schema'] == 'modern-full-schedule-feasibility-v2' and feasibility['approved'] is False and
            feasibility['root_signature'] is None and feasibility['root_resource_accounting'] is None and
            feasibility['input_identity'] == row['input_identity'] and feasibility['short_qualification_is_not_full_fit_evidence'] is True,
            'Feasibility template cannot assert approval/completion')
    require(len(audit_paths) == len(coverage_paths) == 6,'Six family audit/coverage drafts required')
    history = read(bound(index['history']))
    for record in history['records']:bound(record)
    failures = sum(Path(r['path']).name == 'FAILED_ATTEMPT.json' for r in history['records'])
    require(failures >= 8,'Original failed attempts must remain in history')
    approval = read(HERE/'ROOT_CERTIFICATION_APPROVAL_TEMPLATE.json')
    require(approval['approved'] is False and approval['root_signature'] is None and
        approval['prepared_index']['sha256'] == sha(prepared/'INDEX.json'),'Root approval template must remain unsigned')
    audit_review = read(bound(approval['independent_source_engineering_audit']))
    require(audit_review['remaining_concrete_blockers_in_scoped_repair_audit'] == [] and
        audit_review['expected_and_observed_manifest_sha256'] == EXPECTED,'Exact passed scoped repair source audit required')
    receipt = dict(schema='modern-certification-builder-static-receipt-v3',status='PASS_SOURCE_AND_JSON_METADATA_ONLY',
        source_AST_parsed=True,stdlib_imports_only=True,prepared_target_requests=18,legacy_audit_drafts=6,exact_coverage_drafts=6,
        unapproved_feasibility_templates=18,original_history_JSON_descriptors=len(history['records']),retained_failure_JSONs=failures,
        eligible_certificates=0,root_signature_filled=False,sealed_modern_v3_payloads_unchanged=True,
        source_engineering_audit=approval['independent_source_engineering_audit'],builder_sha256=sha(HERE/'build_certificates.py'),
        prepared_index_sha256=sha(prepared/'INDEX.json'),builder_prepare_was_executed='stdlib JSON/source metadata only',
        builder_issue_executed=False,custody_or_scientific_modules_imported=False,scientific_arrays_labels_checkpoints_opened=False,
        numerical_tests_performed=False,ssh_gpu_training_calls=0,
        limits='AST/hash/JSON checks and projection of existing evidence only; eligible issuer and numerical science not executed')
    with output.open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],prepared_target_requests=18,eligible_certificates=0,retained_failure_JSONs=failures)))


if __name__ == '__main__':main()

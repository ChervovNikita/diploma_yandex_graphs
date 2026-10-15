"""Source/preflight checks only; no tensor library, native program or TEST payload."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def run():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('_prepared_link_serving_static', here/'backend.py')
    backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
    root = backend.PHASE.parent.resolve()
    def rejects(function, error=ValueError):
        try: function()
        except error: return
        raise AssertionError('Incompatible serving request or TEST access admitted')
    def row(path): return {'path': str(path), 'bytes': 1, 'sha256': '0'*64}
    def request(family, role):
        files = {key: row(here/'not_materialized'/str(key+'.pt')) for key in ('graph', 'train', 'valid')}
        if role == 'TEST': files['test'] = row(here/'not_materialized'/'test.pt')
        r = {'schema': 'own-selected-link-serving-v1', 'role': role, 'native_family': family,
            'dataset': 'ogbl-collab', 'recipe_id': {'NCN':'collab-ncn','NCNC':'collab-ncnc','BUDDY':'collab-buddy'}[family],
            'backend_sources': backend.backend_sources(), 'data_files': files,
            'query_population': {'positive': 60084 if role == 'VALID' else 1, 'negative': 100000,
                                 'provider_evidence': row(here/'not_materialized'/'provider.json')},
            'evaluator': {'name':'ogbl-collab','metric':'hits@50','K':50},
            'topology': ('weighted_year2007_TRAIN_plus_weighted_VALID' if role == 'TEST' else 'weighted_year2007_TRAIN') if family == 'BUDDY' else ('unweighted_TRAIN_plus_VALID' if role == 'TEST' else 'unweighted_TRAIN'),
            'inference_batch_size': 1000000 if family == 'BUDDY' else 131072,
            'serving_rule': 'own_selected_raw_logits', 'selected': [{'job':row(here/'not_materialized'/'JOB.json'),
                  'checkpoint':row(here/'not_materialized'/'selected.pt'), 'runtime_binding':{}}],
            'rtol': 0.0001, 'atol': 0.000001}
        if family == 'BUDDY': r.update(buddy_cache_mode='fresh_original_HashDataset',cache_directory=str(here/'not_materialized'/'fresh_cache'))
        if role == 'TEST': r['VALID_qualification'] = row(here/'not_materialized'/'qualification.json')
        return r
    for family in ('NCN','NCNC','BUDDY'):
        for role in ('VALID','TEST'):
            r = request(family, role)
            result = backend.validate_request(r,root)
            assert result['TEST_payload_opened'] is False and result['request_sha256'] == backend.digest(r)
    valid, test = request('NCN','VALID'), request('NCN','TEST')
    rejects(lambda: backend.qualify_VALID(None,test,root,admit_runtime=lambda:None),PermissionError)
    rejects(lambda: backend.authorize_TEST(test,None,root),PermissionError)
    rejects(lambda: backend.serve_TEST(None,test,{},root,admit_runtime=lambda:None),PermissionError)
    rejects(lambda: backend.qualify_VALID(None,valid,root,admit_runtime=None))
    wrong = copy.deepcopy(valid); wrong['data_files']['test'] = test['data_files']['test']
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(test); wrong['data_files'].pop('test')
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(test); wrong['topology'] = 'unweighted_TRAIN'
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(valid); wrong['query_population']['positive'] = 1
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(test); wrong['selected'][0]['checkpoint']['path'] = str(here/'recovery.pt')
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(test); wrong['data_files']['test']['path'] = str(root.parent/'outside.pt')
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = request('BUDDY','TEST'); wrong['buddy_cache_mode'] = 'reuse_TRAIN_hashes'
    rejects(lambda: backend.validate_request(wrong,root))
    wrong = copy.deepcopy(test); wrong['serving_rule'] = 'fit_posthoc_private_gate'
    rejects(lambda: backend.validate_request(wrong,root))
    checks = here/'.checks'; checks.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=checks) as temporary:
        p = Path(temporary)/'qualification.json'
        p.write_text(json.dumps({'schema':'selected-link-VALID-qualification-v1','role':'VALID',
                                'passed':False,'TEST_accessed':False,'binding':backend._qualification_binding(test)}))
        test['VALID_qualification'] = {'path':str(p),'bytes':p.stat().st_size,'sha256':backend.receipt(p)['sha256']}
        go = {'action':'score_final_selected_TEST','approved_by':'human','authorized':True,
              'selection_frozen':True,'request_sha256':backend.digest(test),'authorization_reference':'static rejection fixture; not real GO'}
        rejects(lambda:backend.authorize_TEST(test,go,root))
        go['request_sha256'] = '0'*64
        rejects(lambda:backend.authorize_TEST(test,go,root),PermissionError)
    source = (here/'primary/buddy_src_runners_inference.py').read_text()
    tree = ast.parse(source)
    f = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'get_buddy_preds')
    segment = ast.get_source_segment(source,f)
    assert 'data = loader.dataset' in segment and 'shuffle=False' in segment and 'args.eval_batch_size' in segment
    assert 'sigmoid' not in segment and 'subgraph_features = data.subgraph_features[indices]' in segment
    for path in (here/'backend.py',here/'check_static.py',*sorted((here/'primary').glob('*.py'))):
        compile(path.read_bytes(),str(path),'exec')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'source_and_frozen_integration_policy_receipts':'passed',
            'three_families_VALID_TEST_metadata_preflight_without_payload_reads':'passed',
            'absent_or_changed_GO_rejected_before_TEST_read':'passed',
            'failed_VALID_qualification_rejected_before_TEST_read':'passed',
            'VALID_API_rejects_TEST_role':'passed','full_population_and_source_topology_guards':'passed',
            'own_selected_checkpoint_and_raw_logit_rule_guards':'passed',
            'BUDDY_fresh_role_cache_guard':'passed','project_root_containment':'passed',
            'pinned_BUDDY_original_combined_sequential_raw_logit_inference_audit':'passed',
            'syntax_and_import_without_Torch_NumPy':'passed',
            'native_source_programs_executed':False,'native_model_constructions':0,'optimizer_maps':0,
            'VALID_executions':0,'TEST_payload_opened':False,'TEST_scored':False,
            'scientific_server_operations':0,'public_source_files_acquired':3,
            'actual_native_and_VALID_parity_qualification':'pending'}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))

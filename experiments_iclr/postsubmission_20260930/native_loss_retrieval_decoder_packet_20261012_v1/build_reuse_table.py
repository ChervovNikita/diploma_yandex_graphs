"""Hash full control protocols; candidates only, no artifact/acquisition reuse."""
import ast,gzip,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.path.insert(0,str(P/'runtime'))
from adapters import common_native_wrapper as common
from adapters.native_combinations import bindings_common,CONTROLS,CONTROL_I4
from spec import fingerprint

def main():
    b=bindings_common(json.loads((P/'SOURCE_BINDINGS.json').read_text()))
    prior=ROOT/'common_native_wrapper_packet_20261012_v2'
    assert all(row['sha256']==json.loads((prior/'SOURCE_BINDINGS.json').read_text())[key]['sha256'] for key,row in b.items())
    def studies(folder):return [json.loads(gzip.decompress(p.read_bytes())) for p in sorted((folder/'studies').glob('*.json.gz'))]
    old={(s['dataset'],s['study_id'].split('-')[-1].upper()):s for s in studies(prior/'matrix')}
    rows=[]
    def protocol(s,arm,is_new):
        model=[r for r in s['runs'] if r['kind']=='model_fit' and r['config'].get('combination_arm' if is_new else 'wrapper_arm')==arm]
        # Include source M1 body0 and the four independent fresh I4-body rows;
        # paid identities may differ, source/data/fit protocols must not.
        config=model[0]['config'];native_ast=ast.parse(common.bound(ROOT,b['native']).read_text())
        constructor_cls='TABMModel' if arm=='original_gnnm' else 'Model'
        cls=next(n for n in native_ast.body if isinstance(n,ast.ClassDef) and n.name==constructor_cls)
        rendered=common.render_sources(ROOT,b,config['provider_graph']['features'],config['classes'])
        sections={
          'source':{k:v['sha256'] for k,v in b.items()},
          'constructor':{'native_class_AST':common.ast_hash(cls),'family_make_AST':rendered['bound_method_AST_hashes']['make'],'backbone':config['backbone'],'depth':2,'width':128,'features':config['provider_graph']['features'],'classes':config['classes'],'rank':16,'members':config['members'],'arm':arm},
          'initialization':{'seeds_AST':rendered['source_method_AST_hashes']['seeds'],'source_set_seed':b['driver']['sha256'],'factor_site':'stem r CPU Rademacher only; later r/s one; shared_bias outside s','original_fast_draws_device':'declared device for literal TABM','fits':sorted(set((r['config']['stage'],r['split'],r['seed'],r['body']) for r in model))},
          'optimizer':{'AdamW':'one object after complete construction; exact parameter ownership','weight_decay':0,'candidate_LR':[.0005,.001,.002]},
          'TRAIN':{'base_primitive':rendered['train_primitive_AST_hashes'],'packed_branch_AST':rendered['packed_branch_AST_hash'],'arm':arm,'serving':config['fusion']},
          'data':{'preprocessing':s['data']['preprocessing'],'roles':s['data']['roles'],'provider_graph':config['provider_graph'],'prior_exposure':s['data']['prior_exposure']},
          'HPO':{'domain':[{'learning_rate':lr,'dropout':d} for lr in (.0005,.001,.002) for d in (.1,.2,.4)],'search_split':0,'search_seeds':[88101,88103],'comparative_seeds':[88201,88203,88207],'ranking':'mean SELECT accuracy, mean pooled NLL, config hash','I4':'component M1 HPO; four fresh own-selected comparative bodies','search_code_sha':hashlib.sha256((P/'runtime/adapters/search.py').read_bytes()).hexdigest()},
          'selector_budget':{'fit':sorted(set((r['config']['stage'],r['split'],r['seed'],r['body'],fingerprint(r['budget']),fingerprint(r['selection']),fingerprint({k:r['config'][k] for k in ('learning_rate','dropout')})) for r in model)),'checkpoint0':False,'early_termination':False}}
        return {'sections':sections,'hashes':{k:fingerprint(v) for k,v in sections.items()},'full_protocol_hash':fingerprint(sections),'fit_groups':len(model)}
    for study in studies(P/'matrix_reuse_v2' if (P/'matrix_reuse_v2').exists() else P/'local_preview'):
        backbone=study['study_id'].split('-')[-1].upper();prior_study=old[(study['dataset'],backbone)]
        for arm in CONTROLS:
            new=protocol(study,arm,True);previous=protocol(prior_study,arm,False)
            assert new['full_protocol_hash']==previous['full_protocol_hash']
            rows.append({'study_id':study['study_id'],'control_arm':arm,'source_model_fit_groups':new['fit_groups'],'common_v2_model_fit_groups':previous['fit_groups'],'paid_multiplicity_status':'source trajectory set identical; standalone/I4 body0 duplicates removed in this successor; common-v2 prior duplicate count retained for root coalescing','source_optimizer_starts':new['fit_groups'],
                'common_v2_study_id':prior_study['study_id'],'all_sections_match_common_v2':True,'reuse_candidate':True,
                'full_protocol_hash':new['full_protocol_hash'],'section_hashes':new['hashes'],
                'independent_control_banks':[k for k,v in CONTROL_I4.items() if v==arm],
                'reuse_status':'not coalesced; all current starts remain paid; root must rebind provenance/gates before final roster'})
    groups={}
    for row in rows:groups.setdefault(row['full_protocol_hash'],[]).append(row['study_id']+'/'+row['control_arm'])
    out={'schema':'exact-control-protocol-reuse-candidates-v2','rows':rows,'cross_family_groups':groups,
        'sections':['source','constructor','initialization','optimizer','TRAIN','data','HPO','selector_budget'],
        'paired_seed_rationale':'same fixed common-v2 seed block preserves paired controls and permits exact prospective coalescing; seeds are not changed to create method distinctions',
        'actual_artifact_reuse':False,'actual_acquisitions_coalesced':0,'prospective_internal_body0_references':1890,'protocol_hash_uses_unique_complete_trajectory_set':True,'paid_execution_multiplicity_separately_reported':True,'counts_status':'all self-contained paid counts retained until root final roster',
        'scope':'source protocol metadata only, no result/benchmark/checkpoint payloads','real_tensor_reads':0,'GO':False}
    (P/'CONTROL_REUSE_CANDIDATES_V2.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(len(rows),'exact control-family comparisons')
if __name__=='__main__':main()

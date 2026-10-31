"""Retain passed Physics preparation and correct the exact WikiCS path."""
import ast,base64,hashlib,json
from transport import OUT,request
CODE=r'''
stage=phase/sys.argv[1];attempt=json.loads((stage/'DATA_PREPARATION_ATTEMPT2.json').read_text());assert attempt['exit_code']==1 and 'data/wikics/graph.pt' in attempt['stderr']
oldbinding=json.loads((stage/'DATA_INPUT_BINDINGS.json').read_text());records=oldbinding['physics']
def receipt(p):
 raw=p.read_bytes();return {'path':str(p.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
splits=[]
for split,block in enumerate((109001,109003,109007)):
 d=stage/'data/physics'/('split_'+str(split));role_rows={r:receipt(d/(r+'.pt')) for r in ('train','select','cal','evaluate')}
 endpoints={r+'_ids':receipt(d/(r+'_ids.pt')) for r in ('cal','evaluate')};endpoints.update({r+'_targets':role_rows[r] for r in ('cal','evaluate')})
 splits.append({'split_id':split,'source_block':block,'role_seed':block+5000011+211000003,'counts':{'train':100,'select':150,'cal':150,'evaluate':34093},'roles':role_rows,'node_inputs':{'graph':records['graph'],'train':role_rows['train'],'select':role_rows['select']},'endpoint_inputs':endpoints})
value={'dataset_id':'physics','graph':records['graph'],'raw':records['raw'],'geometry':records['geometry'],'splits':splits,'graph_features_and_support_equal_pinned_raw':True,'new_graph_data_copies':0,'immutable_prepared_edge_order_retained':True,'Physics_feature_rule':'pinned_PyG_float32_dense_positive_entries_binarized;not_feature_normalization','evaluation_scope':'encountered_development;no_unused_or_protected_confirmation_claim','official_raw_masks_used':False,'historical_target_archives_or_scores_read':False,'successful_checks_inherited_from_failed_later_WikiCS_path_attempt':receipt(stage/'DATA_PREPARATION_ATTEMPT2.json')}
p=stage/'DATASET_physics_CUSTODY.json';assert not p.exists();p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');p.chmod(0o444)
for row in json.loads(base64.b64decode(sys.argv[2])):
 p=stage/row['path'];old=p.read_bytes();q=p.with_name('attempt2_'+p.name);assert not q.exists();q.write_bytes(old);q.chmod(0o444);raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];p.write_bytes(raw)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');command=[str(repo/'.venv/bin/python'),'-B',str(stage/'preparation/prepare_data.py'),'--project-root',str(phase),'--package',sys.argv[1],'--datasets','wikics'];start=time.monotonic();child=subprocess.run(command,cwd=stage/'preparation',env=env,capture_output=True,text=True,timeout=600)
result={'command':command,'exit_code':child.returncode,'wall_seconds':time.monotonic()-start,'stdout':child.stdout,'stderr':child.stderr,'passed_Physics_checks_repeated':False,'actual_input_models':0,'scientific_fits':0,'GO':False};p=stage/'DATA_PREPARATION_ATTEMPT3.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result));sys.exit(0 if child.returncode==0 else 1)
'''
def main():
    ast.parse(CODE)
    files=[{'path':n,'data':base64.b64encode((OUT/n).read_bytes()).decode(),'sha256':hashlib.sha256((OUT/n).read_bytes()).hexdigest()} for n in ('DATA_INPUT_BINDINGS.json','preparation/prepare_data.py')]
    v=request(CODE,[OUT.name,base64.b64encode(json.dumps(files).encode()).decode()],'WIKICS_EXACT_GRAPH_PATH_AND_PHYSICS_INHERITANCE_RETRY1',700)
    print(json.dumps(v,indent=2))
if __name__=='__main__':main()

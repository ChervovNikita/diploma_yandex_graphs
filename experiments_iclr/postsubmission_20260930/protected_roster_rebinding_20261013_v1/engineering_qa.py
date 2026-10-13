"""Actual compiler-resource transform over synthetic metadata, no tensors."""
from pathlib import Path
import copy,json,sys,hashlib
phase=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir()
sys.path.insert(0,str(phase/'master_controller_source_20261013_v27/runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_master_plan import apply_resources
from rebind_protected_roster import rebind,write_successor
predecessor=json.loads((phase/'protected_roster_rebinding_20261013_v1/PREDECESSOR_PROTECTED76.json').read_text())
amendment=json.loads((phase/'four_native_namespace_import_successors_20261013_v1/AMENDMENT.json').read_text())
studies={}
for row in predecessor['rows']:
    study=studies.setdefault(row['study_id'],{'study_id':row['study_id'],'resources':{'profile':{'scope':'synthetic metadata-only resource-transform fixture'}},'runs':{}})
    study['runs'][row['run_id']]=copy.deepcopy(row['original_run'])
for row in predecessor['rows']:
    study=studies[row['study_id']]
    for child in row['direct_children']:
        if child['run_id'] not in study['runs']:
            study['runs'][child['run_id']]={'run_id':child['run_id'],'adapter':child['adapter'],'kind':child['kind'],'parents':child['parents'],
                    'config':{'stage':child['stage'],'reservation_mib':1,'estimated_seconds':1}}
profiles={}
for study in studies.values():
    study['runs']=list(study['runs'].values());change={}
    for row in predecessor['rows']:
        if row['study_id']==study['study_id']:
            config=row['original_run']['config'];change[row['run_id']]={'reservation_mib':config['reservation_mib']+16,
                                                                   'estimated_seconds':config['estimated_seconds']+17}
    profiles[study['study_id']]={'memory':{'kind':'unknown','scope':'synthetic metadata fixture only'},
                              'time':{'kind':'provisional','scope':'synthetic metadata fixture only'},'runs':change}
priced=[apply_resources(s,profiles) for s in studies.values()]
result=rebind(priced,predecessor,amendment,project_root=phase);assert len(result['roster'])==76
written=write_successor(result,out/'REVIEWED_SYNTHETIC_SUCCESSOR');cases=[{'case':'actual_apply_resources_registered_overrides_preserve76complete_numerical_contracts','passed':True}]
mutations=[('learning_rate',lambda r:r['config'].__setitem__('learning_rate',r['config'].get('learning_rate',.001)+.031)),
           ('budget',lambda r:r['budget'].__setitem__('maximum',r['budget']['maximum']+1)),
           ('parents',lambda r:r['parents'].append('SYNTHETIC_UNREGISTERED_PARENT')),
           ('unregistered_resource_change',lambda r:r['config'].__setitem__('estimated_seconds',r['config']['estimated_seconds']+1))]
key=predecessor['rows'][0];failures=[]
for name,mutate in mutations:
    modified=copy.deepcopy(priced);study=next(s for s in modified if s['study_id']==key['study_id']);run=next(r for r in study['runs'] if r['run_id']==key['run_id']);mutate(run)
    try:rebind(modified,predecessor,amendment,project_root=phase);raise AssertionError('Invalid mutation accepted: '+name)
    except ValueError as ex:failures.append({'case':name,'expected_rejection':str(ex)})
    cases.append({'case':'reject_'+name,'passed':True})
assert 'torch' not in sys.modules
value={'status':'passed','cases':cases,'expected_rejections_preserved':failures,'protected76':len(result['roster']),
       'real_original_full_contracts':True,'actual_compiler_apply_resources':True,'metadata_fixture_only':True,
       'source_arrays_decoded':0,'Torch_imported':False,'scientific_fits':0,'TRAIN_callbacks':0,'optimizer_maps':0,'GO':False}
(out/'RESULT.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps(value))

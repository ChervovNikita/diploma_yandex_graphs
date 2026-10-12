"""Audit new-data baseline records without tensor or score access."""
import json
from pathlib import Path
from prepare_molecular_packets_allocation import GUARD, invoke

PACKAGE=Path(__file__).resolve().parents[1]
LABEL='v8_new_data_baseline_audit_root_20261012_v2'

REMOTE=GUARD+r'''
import collections,gzip,importlib
sys.path.insert(0,str(phase/'master_controller_source_20261012_v5/runtime'))
from spec import fingerprint,normalized_id,ORIGINAL_PAPER_DATASETS
master_path=phase/'massive_practical_master_20261012_v8/PLAN.json'
plan=json.loads(master_path.read_text())
assert fingerprint(plan)=='f38437b3b2e2faa9e5a7d6400a3c5c8e4091646b7b71ea3f0be34e25163b28df'
assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True
studies=[];original=[];violations=[];by_dataset=collections.Counter();baseline_studies=set()
def original_reference(run):
 c=run['config']
 if run['kind']!='model_fit':return False
 if c.get('is_original_gnnm') is True:return True
 # The unchanged fixed-key packet identifies its original source reference
 # through the native architecture rather than the optional later boolean.
 return (run['method']=='original_boundary_GNNM' and c.get('architecture')=='original_gnnm'
         and c.get('members')==4)
for row in plan['studies']:
 path=phase/row['path'];raw=path.read_bytes();value=json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
 assert fingerprint(value)==row['spec_hash']
 author=all(r['kind']=='author_cli_group' for r in value['runs'])
 if normalized_id(value['dataset']) in ORIGINAL_PAPER_DATASETS:violations.append(value['study_id'])
 for r in value['runs']:
  if normalized_id(r['config'].get('dataset','')) in ORIGINAL_PAPER_DATASETS:violations.append(value['study_id']+'/'+r['run_id'])
  if original_reference(r):
   original.append({'study_id':value['study_id'],'dataset':value['dataset'],'run_id':r['run_id'],
     'stage':r['config']['stage'],'split':r['split'],'seed':r['seed'],'config_hash':fingerprint(r['config'])})
   baseline_studies.add(value['study_id']);by_dataset[value['dataset']]+=1
 studies.append({'study_id':value['study_id'],'dataset':value['dataset'],'author_track':author,
                 'new_comparison':not author,'declared_original_GNNM_fit_records':sum(original_reference(r) for r in value['runs'])})
assert not violations and len(studies)==36 and len(baseline_studies)==34 and len(original)==1822
expected={'wikics','amazon-photo','citeseer','pubmed','coauthor-cs','coauthor-physics','ogbg-molhiv','ogbg-molbace','ogbl-collab'}
assert {r['dataset'] for r in studies if r['new_comparison']}==expected
proof={'schema':'v8-new-data-original-GNNM-roster-audit-v1','plan_hash':fingerprint(plan),
 'studies':studies,'original_GNNM_records':original,'original_GNNM_by_dataset':dict(by_dataset),
 'standardized_studies':34,'separate_author_studies':2,'datasets':sorted(expected),
 'original_GNNM_planned_fit_records':1822,'all_standardized_studies_retain_original_GNNM':True,
 'original_five_excluded_for_every_method':True,'original_dataset_violations':violations,
 'no_new_fits_or_GO':True,'arrays_or_logits_or_scores_decoded':False,'original_reported_scores_changed':False,
 'coverage_is_roster_inclusion;source_resource_execution_are_separate_scopes':True,
 'first_audit_marker_only_failure_retained;no_method_or_scores_changed':True}
stage=phase/'v8_new_data_baseline_audit_root_20261012_v2';assert not stage.exists();stage.mkdir()
path=stage/'PROOF.json';path.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
print(json.dumps(proof))
'''


def main():
    out=PACKAGE/'preparation'/LABEL;out.mkdir(exist_ok=False)
    result=invoke(REMOTE,[],out,'AUDIT',120)
    assert result['exit_code']==0,result['stdout']
    value=json.loads(result['stdout']);(out/'PROOF.json').write_text(json.dumps(value,indent=2)+'\n')
    summary={k:v for k,v in value.items() if k not in ('studies','original_GNNM_records')}
    (out/'COMPACT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'new_datasets':len(value['datasets']),'standardized_studies':34,
        'original_GNNM_planned_fit_records':value['original_GNNM_planned_fit_records'],'original_five_violations':0,
        'fits_started':0,'GO_created':False}))


if __name__=='__main__':main()

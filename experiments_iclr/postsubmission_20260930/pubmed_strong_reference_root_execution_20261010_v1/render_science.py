"""Render exact9 references after all3 actual engineering owners close."""
import json
from render_common import HERE,SOURCE,SOURCE_SHA,arguments,context,bound,read_bound,write,sha,release,plan


def main():
    args=arguments(__doc__);ctx=context(args)
    owner_plan=read_bound(bound(HERE/'ENGINEERING_OWNER_PLAN.json'))
    conditions=('single_native','single_mean4_dropout','independent4_own')
    expected=['seed9101__'+c for c in conditions]
    if owner_plan.get('purpose')!='engineering' or [r['record_id'] for r in owner_plan['records']]!=expected or owner_plan.get('source_manifest_sha256')!=SOURCE_SHA or owner_plan.get('owner_sha256')!=sha(HERE/'queue.py'):
        raise ValueError('Exact frozen engineering owner plan required')
    terminals={};completions={}
    # Close every actual terminal before reading a numerical engineering report.
    for condition,row in zip(conditions,owner_plan['records']):
        path=HERE/'owners'/row['owner_id']/'RAW_OWNER_TERMINAL.json'
        descriptor=bound(path);terminal=read_bound(descriptor)
        absent=terminal.get('owned_absence',{})
        if terminal.get('complete') is not True or terminal.get('directly_waited') is not True or terminal.get('child_exit_code')!=0 or terminal.get('cap_or_owner_failure') is not None:
            raise ValueError('All3 engineering owners must be actually waited successfully')
        if absent.get('owned_process_absence_verified') is not True or absent.get('owned_CUDA_absence_verified') is not True:
            raise ValueError('Actual engineering owned absence required')
        if row['release_sha256'] not in terminal.get('argv',[]):raise ValueError('Exact engineering release must occur in waited argv')
        terminals[condition]=descriptor
    for condition,row in zip(conditions,owner_plan['records']):
        path=HERE/'engineering'/'cells'/row['record_id']/'COMPLETE.json'
        descriptor=bound(path);result=read_bound(descriptor)
        if result.get('schema')!='PubMed-strong-reference-TRAIN-only-engineering-v1' or result.get('complete') is not True or result.get('condition')!=condition or result.get('seed')!=9101 or result.get('source_manifest_sha256')!=SOURCE_SHA or result.get('engineering_release_sha256')!=row['release_sha256']:
            raise ValueError('Exact new-interface full-TRAIN completion required')
        if any(result.get(k) is not False for k in ('VALID_access','TEST_access','science_enabled')):raise ValueError('Engineering remains TRAIN-only')
        expected=4 if condition=='independent4_own' else 1
        if len(result.get('bodies',[]))!=expected or any(b.get('updates')!=1 or b.get('TRAIN_label_count')!=11829 or b.get('complete_graph_nodes')!=19717 for b in result['bodies']):
            raise ValueError('Every complete full-TRAIN update required')
        completions[condition]=descriptor
    adoption=json.loads((SOURCE/'ROOT_ADOPTION_TEMPLATE_DISABLED.json').read_text())
    adoption.update(enabled=True,source_review_approved=True,source_delta_assessment_approved=True,
                    source_manifest_sha256=SOURCE_SHA,roster_sha256=sha(SOURCE/'ROSTER.json'),protocol_sha256=sha(SOURCE/'PROTOCOL.json'),
                    anchors_sha256=sha(SOURCE/'ANCHORS.json'),source_review=ctx['review'],source_delta_assessment=ctx['delta'],
                    qualifications=completions,qualification_terminals=terminals)
    adoption_path=HERE/'ROOT_ADOPTION.json';write(adoption_path,adoption)
    rows=[release(r['record_id']+'.json','science',ctx,bound(adoption_path))
          for r in json.loads((SOURCE/'ROSTER.json').read_text())]
    result=plan('science',rows,ctx)
    print(json.dumps(dict(new_scientific_groups=9,fresh_native_trajectories=18,owner_plan=result,science_not_started=True)))


if __name__=='__main__':main()

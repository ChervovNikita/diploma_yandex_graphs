"""Separate exact200/patience50 recipe reference metadata; no models or fits."""
import argparse
import json
from pathlib import Path

from source_utils import bind_source,bound,load,receipt,RAW_FILES
from prepare_node_data import verify_allocation
from spec import fingerprint,confined

SEEDS=(84301,84303,84307)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('project-root','runtime-package','output','custody'):p.add_argument('--'+n,required=True)
    a=p.parse_args();verify_allocation();root=Path(a.project_root).resolve();out=confined(root,a.output)
    if out.exists():raise ValueError('Fresh source-recipe reference packet required')
    source=bind_source(root);loader=load(root,source['modules']['role_loader'],'reference_role_loader');raw=root/'hgb_IMDB_public_acquisition_20261009_v1/development_inputs';bindings={n:receipt(root,raw/n) for n in RAW_FILES}
    if any(bindings[n]['bytes']!=v[0] or bindings[n]['sha256']!=v[1] for n,v in RAW_FILES.items()):raise ValueError('Exact three allowed raw inputs required')
    labels=loader.development_labels(raw/'label.dat',4932,5);out.mkdir();roles=[]
    import numpy as np
    for seed in SEEDS:
        ids=np.asarray(sorted(labels),dtype=np.int64);np.random.RandomState(seed).shuffle(ids);n=len(ids)//5
        role={'schema':loader.ROLE_SCHEMA,'dataset':'HGB IMDB','seed':seed,'split_rule':loader.SPLIT_RULE,'class_dim':5,'train_ids':sorted(ids[n:].tolist()),'valid_ids':sorted(ids[:n].tolist()),'input_files':{k:{f:v[f] for f in ('bytes','sha256')} for k,v in bindings.items()}}
        loader.check_role(role,labels,5,role['input_files']);path=out/('ROLE_'+str(seed)+'.json');path.write_text(json.dumps(role,indent=2,sort_keys=True)+'\n');roles.append({'seed':seed,'role':receipt(root,path),'TRAIN':len(role['train_ids']),'VALID':len(role['valid_ids'])})
    cp=confined(root,a.custody);custody=json.loads(cp.read_text());full=next(f for f in custody['files'] if f['path']==custody['graph_feature_caches']['full']);full={**full,'path':str((cp.parent/full['path']).relative_to(root))};bound(root,full)
    runtime=confined(root,a.runtime_package);runtime_files=[receipt(root,f) for f in sorted(runtime.rglob('*.py')) if '__pycache__' not in f.parts]
    packet={'schema':'native-SeHGNN200-patience50-source-recipe-reference-v1','scientific_launch_enabled':False,'human_GO_required':True,'source_recipe_not_published_score_replication':True,
            'seeds':list(SEEDS),'ordered_seed_loop_retains_one_native_master_GradScaler':True,'runs':roles,'raw_input_dir':str(raw.relative_to(root)),'raw_files':bindings,'source':source,'runtime_files':runtime_files,
            'new_paid_graph_only_full_feature_cache':full,'cache_cost_paid_by':receipt(root,cp),'reuse_has_no_fitted_teacher_or_old_results':True,
            'geometry':{'nodes':4932,'feature_channels':25,'label_channels':12,'embedding':512,'hidden':512,'classes':5,'parameters':83659532},
            'fit_recipe':{'original_entrypoint':'engine.run_one(qualify=False)','epoch_loop':'range(200), native zero-based epochs0..199','maximum_complete_native_epochs':200,'patience_stop':'epoch-best_epoch > 50','selector':'strict-first complete native VALID BCE','learning_rate':.001,'dropout':.5,'weight_decay':0.,'TRAIN_batch_size':10000,'TRAIN_batches_per_epoch':1,'AMP':'literal native CUDA AMP','TRAIN_label_returns':'fresh per seed; exact native sparse products and diagonal removal'},
            'optimizer_owners':3,'maximum_Adam_calls':600,'minimum_actual_epochs_not_assumed':True,'selected_native_state_and_history_preserved':True,'TEST_access':False,'historical_authorization_or_outcomes_inherited':False,
            'comparison_scope':'Descriptive source-recipe reference on fresh80/20development roles; not comparable to primary50/20/10/20roles or original official-test paper scores',
            'status':'DISABLED_OWN_NATIVE_ENTRYPOINT_RESOURCE_ADMISSION_AND_EXPLICIT_GO_REQUIRED','fits':0,'CUDA':False,'GO':False}
    path=out/'REFERENCE_PACKET.json';path.write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n');(out/'IDENTITY.json').write_text(json.dumps({'packet':receipt(root,path),'packet_hash':fingerprint(packet),'GO':False},indent=2)+'\n');print(json.dumps({'status':'disabled_reference_prepared','native_fit_groups':3,'maximum_epochs_per_fit':200,'patience':50,'fits_executed':0,'GO':False}))


if __name__=='__main__':main()

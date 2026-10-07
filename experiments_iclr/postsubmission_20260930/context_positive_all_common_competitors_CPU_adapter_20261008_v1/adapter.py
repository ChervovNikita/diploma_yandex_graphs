"""CPU ordinal diagnostics over the existing complete9 prediction collection."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SEEDS=(8101,8203,8307)
CONDITIONS=('shared_common','shared_route','shared_route_permuted')
ARCHIVE_SHA='80d30ed946a1408f861a773952b174713c487765c75e95b8e25456b30478eae6'
REQUIRED=('valid_ids','truth','member_logits','member_prediction','pool_prediction')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def all_common_classes(np,logits,truth):
    require(logits.shape==(4,len(truth),10),'Original M4/ten-class axes')
    true_logits=np.take_along_axis(logits,truth[None,:,None],axis=2)
    return (logits>true_logits).all(axis=0)


def diagnose(np,before,after,common_classes):
    """Rank relations use all COMMON rivals; predictions retain native argmax."""
    truth=before['truth'];n=len(truth)
    require(np.array_equal(truth,after['truth'])
        and np.array_equal(before['valid_ids'],after['valid_ids']), 'Paired original labels/IDs')
    require(common_classes.shape==(n,10) and common_classes.dtype==bool
        and np.array_equal(common_classes,all_common_classes(np,before['member_logits'],truth)),
        'All original COMMON qualifying classes must remain fixed')
    eligible=common_classes.any(axis=1)
    true_candidate=np.take_along_axis(after['member_logits'],truth[None,:,None],axis=2)
    beats=true_candidate>after['member_logits']
    # Quantifier order is deliberate: EXISTS one member FOR ALL old rivals.
    same_member=(~common_classes[None]|beats).all(axis=2)&eligible[None]
    same_member_any=same_member.any(axis=0)
    # Weaker: FOR EACH old rival there EXISTS some potentially different member.
    per_class=common_classes&beats.any(axis=0)
    every_class_some_member=(~common_classes|per_class).all(axis=1)&eligible
    any_class_some_member=per_class.any(axis=1)
    member_correct=after['member_prediction']==truth[None]
    baseline_member_correct=before['member_prediction']==truth[None]
    any_correct=member_correct.any(axis=0)
    old_pool_correct=before['pool_prediction']==truth
    new_pool_correct=after['pool_prediction']==truth
    repairs=~old_pool_correct&new_pool_correct
    harms=old_pool_correct&~new_pool_correct
    candidate_common=all_common_classes(np,after['member_logits'],truth)
    newly_common=candidate_common&~common_classes
    def count(mask):
        return int(np.count_nonzero(mask))
    return dict(full_population=dict(objects=n,baseline_pool_correct=count(old_pool_correct),
        candidate_pool_correct=count(new_pool_correct),served_repairs=count(repairs),
        introduced_served_errors=count(harms),net_repairs=count(repairs)-count(harms),
        baseline_any_member_correct=count((before['member_prediction']==truth[None]).any(axis=0)),
        candidate_any_member_correct=count(any_correct)),
        COMMON_obstruction_population=dict(objects=count(eligible),
            qualifying_class_object_pairs=count(common_classes),
            baseline_common_competitor_and_pool_correct=count(eligible&old_pool_correct),
            any_old_class_reversed_by_some_member=count(eligible&any_class_some_member),
            old_class_object_pairs_reversed_by_some_member=count(per_class),
            every_old_class_reversed_by_some_potentially_different_member=count(every_class_some_member),
            one_same_member_strictly_beats_ALL_old_classes=count(same_member_any),
            weaker_all_classes_reversed_but_no_same_member=count(every_class_some_member&~same_member_any),
            any_candidate_member_actually_correct=count(eligible&any_correct),
            same_member_ALL_but_no_candidate_member_correct=count(same_member_any&~any_correct),
            same_member_ALL_and_served_repair=count(same_member_any&repairs),
            same_member_ALL_but_pool_still_wrong=count(same_member_any&~new_pool_correct),
            served_repair_without_same_member_ALL=count(eligible&repairs&~same_member_any),
            served_repair_with_no_candidate_member_correct=count(eligible&repairs&~any_correct),
            newly_common_competitor_objects=count(eligible&newly_common.any(axis=1)),
            newly_common_class_object_pairs=count(eligible[:,None]&newly_common),
            same_member_ALL_with_a_new_common_competitor=count(same_member_any&newly_common.any(axis=1)),
            served_repairs=count(eligible&repairs),introduced_served_errors=count(eligible&harms)),
        by_member=[dict(member_index0=m,strictly_beats_all_old_classes=count(same_member[m]),
            actually_correct_on_COMMON_obstruction=count(eligible&member_correct[m]),
            beats_all_old_but_this_member_incorrect=count(same_member[m]&~member_correct[m]),
            full_population_member_repairs=count(~baseline_member_correct[m]&member_correct[m]),
            full_population_member_introduced_errors=count(baseline_member_correct[m]&~member_correct[m]))
            for m in range(4)])


def validate(np,arrays):
    require(set(arrays)==set(REQUIRED),'Only original existing ordinal prediction fields')
    shapes={'valid_ids':(5274,),'truth':(5274,),'member_logits':(4,5274,10),
            'member_prediction':(4,5274),'pool_prediction':(5274,)}
    for key,value in arrays.items():
        dtype=np.float32 if key=='member_logits' else np.int64
        require(value.shape==shapes[key] and value.dtype==dtype and np.isfinite(value).all(),
                'Complete original5274/M4/ten-class field: '+key)
    require(len(np.unique(arrays['valid_ids']))==5274,'Original unique complete IDs')
    for key in ('truth','member_prediction','pool_prediction'):
        require(np.all((arrays[key]>=0)&(arrays[key]<10)),'Original ten-class labels/predictions')
    require(np.array_equal(arrays['member_prediction'],arrays['member_logits'].argmax(axis=2)),
            'Native member argmax labels must correspond to the existing logits')


def run(output,collection):
    """No prediction archive opens until all9 and complete collection are verified."""
    require(collection.get('whole9_complete_before_opening') is True
        and collection.get('owner_and_children_terminal_before_opening') is True
        and collection.get('frozen_target_archive_sha256')==ARCHIVE_SHA
        and collection.get('status')=='complete','Whole9 terminal and COMPLETE collection only')
    cells=collection['cells']
    require(len(cells)==9 and {(r['seed'],r['condition']) for r in cells}
        =={(s,c) for s in SEEDS for c in CONDITIONS},'All fixed9 cells, without survivor filtering')
    require(all(r['family_status']=='complete' and r['collection_status']=='complete'
                for r in cells),'No incomplete fit/prediction collection opens')
    output=Path(output).resolve()
    require(output.is_relative_to(PHASE) and output!=PHASE,'Project research custody')
    rows={(r['seed'],r['condition']):r for r in cells};paths={}
    # Verify every original raw artifact before the first semantic array read.
    for key,record in rows.items():
        binding=record['raw_archive'];relative=Path(binding['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Phase-relative archive custody')
        path=(PHASE/relative).resolve()
        require(path.is_relative_to(PHASE) and path.is_file() and sha(path)==binding['sha256'],
                'Existing raw prediction archive binding')
        if 'bytes' in binding:
            require(path.stat().st_size==binding['bytes'],'Existing raw archive size')
        paths[key]=path
    import numpy as np
    def load(key):
        with np.load(paths[key],allow_pickle=False) as archive:
            require(set(REQUIRED).issubset(archive.files),'Existing original prediction fields')
            arrays={name:archive[name].copy() for name in REQUIRED}
        validate(np,arrays);return arrays
    baselines={};masks={};role=None;mask_bindings=[]
    # Define ALL rival identities from COMMON alone, before this adapter reads candidates.
    for seed in SEEDS:
        arrays=load((seed,'shared_common'))
        if role is None:
            role=(arrays['valid_ids'].copy(),arrays['truth'].copy())
        require(np.array_equal(arrays['valid_ids'],role[0]) and np.array_equal(arrays['truth'],role[1]),
                'Identical complete role across all seeds')
        baselines[seed]=arrays;masks[seed]=all_common_classes(np,arrays['member_logits'],arrays['truth'])
        masks[seed].setflags(write=False)
        mask_bindings.append(dict(seed=seed,COMMON_raw_archive=rows[(seed,'shared_common')]['raw_archive'],
            all_qualifying_class_mask_sha256=hashlib.sha256(masks[seed].tobytes()).hexdigest(),
            defined_from_COMMON_only=True,frozen_before_this_adapter_reads_candidate_arrays=True))
    comparisons=[]
    for seed in SEEDS:
        for condition in ('shared_route','shared_route_permuted'):
            after=load((seed,condition))
            comparisons.append(dict(seed=seed,candidate_condition=condition,
                diagnostic=diagnose(np,baselines[seed],after,masks[seed])))
            del after
    result=dict(schema='all_COMMON_competitors_CPU_diagnostic-v1',comparisons=comparisons,
        COMMON_mask_bindings=mask_bindings,complete_objects_per_cell=5274,classes=10,members=4,
        fixed_seeds=list(SEEDS),all9_complete_collection_before_opening=True,
        model_constructions=0,model_forward_calls=0,TRAIN_updates=0,additional_member_inference_calls=0,
        frozen_gate_changed=False,scientific_execution_authorized=False,methodological_novelty_claimed=False,
        limits=['Beating all baseline common competitors is insufficient for correctness if a new rival blocks truth.',
                'One same member beating all baseline rivals is not necessary for the convex pool to repair: different members can reverse different rivals.',
                'Strict reversal differs from native argmax at ties. Exact-arithmetic convex-pool identities are not microscopic floating-point rejection gates.',
                'Newly common means a rival becomes higher than truth in all candidate members; it need not have been absent from every baseline member.',
                'Counts are descriptive on one selected development graph; class-object pairs and seed-node repetitions are not independent observations.'])
    directory=output/'compact';directory.mkdir(exist_ok=True)
    path=directory/'ALL_COMMON_COMPETITOR_DIAGNOSTICS.json'
    require(not path.exists(),'Preserve an existing diagnostic rather than overwrite')
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result

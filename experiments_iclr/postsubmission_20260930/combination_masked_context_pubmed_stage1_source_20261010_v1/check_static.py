"""Source/admission/closure fixtures only. No numerical imports or data reads."""
import ast
import copy
import json
from pathlib import Path
import sys
from admission import admit
from compare import summarize
from source import HERE, load_v3, sha, verify_sources
from stage_plan import CONDITIONS, SEEDS, SPLIT, description, roster
from train import verify_selected_metrics


def fixture(class_correct, nll=1.):
    classes=[]
    for c,(correct,count) in enumerate(zip(class_correct,(820,1547,1575))):
        row=dict(class_id=c,correct=correct,count=count,accuracy=correct/count,NLL=nll)
        row['members']=[{k:v for k,v in row.items() if k != 'class_id'} for _ in range(4)]
        classes.append(row)
    correct=sum(class_correct)
    pooled=dict(correct=correct,count=3942,accuracy=correct/3942,NLL=nll)
    return dict(pooled=pooled,members=[dict(pooled) for _ in range(4)],classes=classes,
                mean_member_accuracy=correct/3942,worst_member_accuracy=correct/3942,
                macro_accuracy=sum(r['accuracy'] for r in classes)/3)


def main():
    verify_sources()
    parsed={f.name:ast.parse(f.read_text(),filename=str(f)) for f in HERE.glob('*.py')}
    banned={'numpy','torch','scipy','torch_geometric'}
    for tree in parsed.values():
        for node in tree.body:
            if isinstance(node,ast.Import): assert not any(alias.name.split('.')[0] in banned for alias in node.names)
            if isinstance(node,ast.ImportFrom): assert (node.module or '').split('.')[0] not in banned
    bindings=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    assert bindings['exact_V3_method']['sha256'] == bindings['V2_method']['sha256']
    assert bindings['exact_V3_native']['sha256'] == bindings['V2_native']['sha256']
    v3=load_v3()
    assert v3['plan'].SEEDS == SEEDS and v3['plan'].STAGE1 == CONDITIONS
    assert SPLIT['TRAIN_class_counts'] == [2461,4643,4725] and SPLIT['VALID_class_counts'] == [820,1547,1575]
    assert len(roster()) == 9 and not description()['enabled']
    disabled=sorted((HERE/'releases_disabled').glob('*.json'))
    assert len(disabled) == 9
    admitted=0
    for path in disabled:
        release=json.loads(path.read_text())
        assert release['enabled'] is False and release['runtime_qualified'] is False
        assert release['root_science_authorized'] is False and release['TEST_access'] is False
        assert release['max_epochs'] == 2000 and release['patience'] == 250
        assert release['TRAIN_class_counts'] == [2461,4643,4725] and release['VALID_class_counts'] == [820,1547,1575]
        assert release['source_manifest_sha256'] is None
        try: admit(path,sha(path))
        except ValueError as error: assert str(error).startswith('Inactive root admission:')
        else: admitted += 1
    assert admitted == 0
    records={}
    for seed in SEEDS:
        for name,correct,nll in (('shared4_own',[400,900,1300],1.),('shared4_core',[404,904,1304],.99),('independent4_native',[408,908,1308],.98)):
            records[seed,name]=dict(complete=True,selected_VALID=fixture(correct,nll),selected_epoch=17)
    assert summarize(records)['continuation_eligible']
    incomplete=dict(records);incomplete.pop((9307,'shared4_core'))
    try: summarize(incomplete)
    except ValueError as error: assert str(error).startswith('No partial-family comparison:')
    else: raise AssertionError('Partial family compared')
    failed=copy.deepcopy(records);failed[9101,'shared4_core']['complete']=False
    try: summarize(failed)
    except ValueError: pass
    else: raise AssertionError('Failed record compared')
    negative=copy.deepcopy(records);negative[9307,'shared4_core']['selected_VALID']=fixture([399,899,1299],.99)
    assert not summarize(negative)['gates']['accuracy_gain_positive_all_three_seeds']
    macro=copy.deepcopy(records)
    for seed in SEEDS: macro[seed,'shared4_core']['selected_VALID']=fixture([360,920,1340],.99)
    result=summarize(macro)
    assert result['gates']['mean_accuracy_gain_at_least_0_20pp']
    assert not result['gates']['mean_macro_accuracy_no_worse_than_own']
    expected=fixture([400,900,1300])
    repeated=copy.deepcopy(expected);repeated['pooled']['NLL'] += 1e-7
    verify_selected_metrics(repeated,expected)
    changed=copy.deepcopy(expected);changed['pooled']['correct'] += 1
    try: verify_selected_metrics(changed,expected)
    except ValueError: pass
    else: raise AssertionError('Changed exact prediction count tolerated')
    changed=copy.deepcopy(expected);changed['pooled']['NLL'] += .01
    try: verify_selected_metrics(changed,expected)
    except ValueError: pass
    else: raise AssertionError('Material restored metric change tolerated')
    source=(HERE/'train.py').read_text()
    assert 'correct > best_correct' in source and 'epoch-best_epoch >= 250' in source
    assert 'selected_signatures != best_signatures' in source
    assert 'member_correct_masks=' in source and 'pooled_correct_mask=' in source
    assert "method.Session(spec['condition'], spec['seed'], tensor['x'], tensor['edge_index'], tensor['train_ids'], tensor['train_y']" in source
    assert 'session.train_step(audit=False)' in source
    assert 'load_v3(numerical=True)' in source
    assert all(name not in sys.modules for name in ('torch','numpy','scipy','torch_geometric'))
    print(json.dumps(dict(schema='masked-context-stage1-stdlib-static-checks-v1',complete=True,
        parsed_python_files=sorted(parsed),disabled_releases_rejected=9,
        exact_V3_method_and_provider_unchanged=True,authoritative_class_order=True,
        complete_nine_and_failed_record_gate_fixture_pass=True,macro_gate_fixture_pass=True,
        tiny_float_repeat_tolerated=True,changed_integer_count_rejected=True,
        exact_prediction_and_correct_mask_source_guard=True,
        numerical_modules_imported=False,models_constructed=False,raw_arrays_or_checkpoints_read=False,
        server_operations=False,qualification_or_scientific_calls=0,
        static_checks_prove_runtime_or_competence=False),indent=2,sort_keys=True))


if __name__ == '__main__': main()

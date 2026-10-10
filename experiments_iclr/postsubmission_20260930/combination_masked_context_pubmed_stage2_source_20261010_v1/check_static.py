"""Source/hash/disabled-admission checks only; no numeric routines or arrays."""
import ast
import json
import sys
from source import HERE,sha,verify_sources,load_v3
from stage_plan import CONDITIONS,STAGE1,ALL_CONDITIONS,SEEDS,roster,REWIRE,spec
from admission import admit
from generate_rewire import admit as admit_graph
from family import read_complete27


def main():
    bindings=verify_sources()
    parsed={f.name:ast.parse(f.read_text(),filename=str(f)) for f in HERE.glob('*.py')}
    numerical={'torch','numpy','scipy','torch_geometric'}
    for tree in parsed.values():
        for node in tree.body:
            if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in numerical for a in node.names)
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in numerical
    v3=load_v3()['plan']
    assert tuple(c for c in v3.CONDITIONS if c not in v3.STAGE1)==CONDITIONS
    assert len(CONDITIONS)==6 and len(SEEDS)==3 and len(roster())==18 and len(ALL_CONDITIONS)==9
    assert set(ALL_CONDITIONS)==set(CONDITIONS)|set(STAGE1)
    assert bindings['exact_V3_method']['sha256']==bindings['V2_method']['sha256']
    assert bindings['exact_V3_native']['sha256']==bindings['V2_native']['sha256']
    releases=list((HERE/'releases_disabled').glob('*.json'))
    assert len(releases)==18
    for file in releases:
        value=json.loads(file.read_text())
        assert value['enabled'] is False and value['root_science_authorized'] is False
        assert value['complete_nine_positive_stage1_verified'] is False
        assert value['seed'] in SEEDS and value['condition'] in CONDITIONS
        assert value['TRAIN_class_counts']==[2461,4643,4725] and value['VALID_class_counts']==[820,1547,1575]
        try:admit(file,sha(file))
        except ValueError as error:assert str(error).startswith('Inactive root admission:')
        else:raise AssertionError('Disabled control admitted')
    for name,function in (('REWIRE_GENERATION_RELEASE_TEMPLATE_DISABLED.json',admit_graph),('COMPLETE27_READOUT_RELEASE_TEMPLATE_DISABLED.json',read_complete27)):
        file=HERE/name
        try:function(file,sha(file))
        except ValueError as error:assert 'disabled' in str(error)
        else:raise AssertionError('Disabled auxiliary/readout template admitted')
    method=(HERE.parent/bindings['prototype_folder']/'method.py').read_text()
    assert "scale = 1./self.spec['members'] if not self.spec['untied'] else 1." in method
    assert 'total = (.5*ce+.1*core)/4' in method
    assert "decoder_gradient_divisor = self.spec['members'] if self.spec['untied'] and self.decoder is not None else 1" in method
    assert 'parameter.grad.div_(decoder_gradient_divisor)' in method
    assert 'self._view(masked_x, aux)' in method
    assert spec('single_four_view_core')['backwards']==5 and spec('untied4_shared_decoder_core')['optimizer_steps']==5
    train=(HERE/'train.py').read_text();metrics=(HERE/'metrics.py').read_text();rewire=(HERE/'rewire.py').read_text()
    assert 'session.train_step(audit=False)' in train and 'selected_last_ownership' in train
    assert "expected_decoder_divisor=4 if spec['condition']=='untied4_shared_decoder_core' else 1" in train
    assert 'selected_ownership[member]' in metrics and "session.spec['members']==4 else slot" in metrics
    assert "if session.decoder is not None:" in metrics and 'members_count not in (1,4)' in metrics
    assert 'before!=degree_after' in rewire and 'self_edges!=self_after' in rewire
    assert 'len({a,b,c,d})!=4' in rewire and 'first in edges or second in edges' in rewire
    assert REWIRE['double_edge_swap_seed']==190121 and REWIRE['accepted_swaps_per_nonself_undirected_edge']==10
    assert 'candidate' not in rewire and 'label' not in rewire.split('def swap_graph')[1].split('def admit_auxiliary')[0].replace('labels_used=False','')
    assert all(name not in sys.modules for name in numerical)
    print(json.dumps(dict(schema='masked-context-stage2-source-static-checks-v1',complete=True,
        parsed_python_files=sorted(parsed),disabled_control_releases_rejected=18,
        disabled_graph_and_readout_releases_rejected=True,exact_V3_method_native_preserved=True,
        sealed_Stage1_hashes_verified=True,exact18_plus9_roster_verified=True,
        control_aware_member_view_backward_optimizer_counts_declared=True,
        V3_decoder_normalization_and_recomputed_rewire_banks_source_verified=True,
        rewire_invariants_source_checked=True,rewire_algorithm_executed=False,
        numerical_modules_imported=False,numerical_routines_or_models_or_arrays_executed=False,
        current_scientific_outcomes_or_TEST_read=False,server_operations=False,
        runtime_competence_or_graph_mixing_proven=False),indent=2,sort_keys=True))


if __name__=='__main__':main()

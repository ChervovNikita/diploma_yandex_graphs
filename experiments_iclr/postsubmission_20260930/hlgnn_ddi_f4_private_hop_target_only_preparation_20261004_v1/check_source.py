"""Local stdlib checks only; no numerical imports, model construction or artifact loads."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def check():
    files = sorted(HERE.rglob('*.py'))
    trees = {str(p.relative_to(HERE)):ast.parse(p.read_text(),filename=str(p)) for p in files}
    bindings = json.loads((HERE/'INPUT_BINDINGS.json').read_text())['inputs']
    for row in bindings:
        source = PHASE/row['path']
        value = source.read_bytes()
        assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256']
        if 'copy' in row:assert value==(HERE/row['copy']).read_bytes()
    base = PHASE/'hlgnn_ddi_fresh_training_source_preparation_20261004_v1'
    shared = PHASE/'hlgnn_shared_propagation_member_filter_preparation_20261004_v1'
    for packet in (base,shared):
        for row in json.loads((packet/'MANIFEST.json').read_text())['files']:
            value = (packet/row['path']).read_bytes()
            assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256']
    config = json.loads((HERE/'config.json').read_text())
    native_config = json.loads((HERE/'BASELINE_CONFIG.json').read_text())
    assert config['release_enabled'] is False and native_config['release_enabled'] is False
    assert config['recipe']==native_config['recipe'] and config['seeds']==[0,1,2]
    assert config['selection_metric']=='Hits@20' and config['artifact_contract']['train_weight_present'] is False
    assert config['artifact_contract']['path'] is None and config['artifact_contract']['sha256'] is None
    assert config['recipe']['epochs']==500 and config['recipe']['batch_size']==65536 and config['recipe']['num_neg']==3
    assert config['recipe']['eval_steps']==5 and config['recipe']['gnn_num_layers']==15
    assert config['variant']['members']==4 and config['variant']['private_alpha'] is True
    assert config['variant']['conditional_auxiliary'] is False and config['variant']['counts_in_served_scores'] is False
    entry=trees['train_f4_ddi.py']
    variant = next(n for n in entry.body if isinstance(n,ast.Assign) and n.targets[0].id=='VARIANT')
    assert ast.literal_eval(variant.value)==config['variant']
    assert 'baseline.read_release_config' in ast.unparse(entry) and 'baseline.load_train_valid' in ast.unparse(entry)
    assert 'baseline.run_seed' in ast.unparse(entry) and 'F4PrivateHopTargetModel' in ast.unparse(entry)
    main = next(n for n in entry.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    gate = next(n for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='baseline.read_release_config')
    for node in ast.walk(main):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            assert node.lineno>gate.end_lineno
    model=trees['f4_model.py'];text=(HERE/'f4_model.py').read_text()
    cls=next(n for n in model.body if isinstance(n,ast.ClassDef))
    assert cls.name=='F4PrivateHopTargetModel' and ast.unparse(cls.bases[0])=='BaseModel'
    assert "members=self.MEMBERS,private_alpha=True" in text and 'MEMBERS = 4' in text
    assert 'margin=None' in text and 'torch.stack(route_losses).mean()' in text
    assert "set(split_edge['train']) != {'edge'}" in text and "storage='aggregates'" in text
    assert 'self.calculate_loss' in text and 'def calculate_loss' not in text
    assert 'torch.optim.Adam(self.para_list,lr=lr)' in text
    assert 'get_pos_neg_edges' in text and 'loss.backward()' in text
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in ('load','load_state_dict','detach','get_edge_split','save') for n in ast.walk(model))
    assert 'torch.stack(routes,dim=0).mean(dim=0)' in text
    baseline=(HERE/'baseline_train_ddi.py').read_text()
    assert 'if metric > best_metric:' in baseline and 'metric = results["Hits@20"]' in baseline
    assert 'for epoch in range(1, recipe["epochs"] + 1)' in baseline
    evidence=json.loads((HERE/'evidence/EQUIVALENCE_SUMMARY.json').read_text())
    assert evidence['status']=='PASS_OUTPUT_AND_FULL_GRADIENT_EQUIVALENCE'
    assert evidence['comparison_summary']['assertions']==109 and evidence['CUDA_initialized'] is False
    command=json.loads((HERE/'PROPOSED_COMMAND.json').read_text())
    assert command['launch_authorized'] is False and command['execution_client_prepared'] is False
    return dict(status='LOCAL_SOURCE_CHECKS_PASS_ONLY',AST_parsed_files=len(files),bound_inputs=len(bindings),
        native_modules_and_baseline_driver_unchanged=True,verified_factorization_source_unchanged=True,
        native_missing_weight_AUC_dispatch_inherited=True,full_native_recipe_preserved=True,
        F4_private_hop_target_only=True,shared_embedding_context=True,four_private_native_predictor_heads=True,
        mean_native_route_loss_and_mean_raw_serving=True,fresh_initialization_and_first_tie_VALID_selection=True,
        new_numerical_imports=False,extra_equivalence_runs=False,artifact_payload_reads=False,
        fitting_or_tuning=False,heldout_reads=False,execution_authorized=False,original_sources_rehashed_unchanged=True,
        limitation='Source-only F4 integration; the earlier encoder check does not validate this trainer or DDI GPU behavior.')


if __name__=='__main__':
    print(json.dumps(check(),indent=2,sort_keys=True))

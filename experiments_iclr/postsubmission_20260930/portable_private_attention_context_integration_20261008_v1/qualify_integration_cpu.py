"""Engineering-only constructor/replay/reconstruction on fabricated CPU inputs.

One synthetic replay/Adam transition with580 fake rows and a small-width native
body. No scientific arrays, scores, checkpoint files, CUDA or full training.
"""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType


def module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def run(phase):
    import numpy as np
    import torch
    import torch_geometric
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent
    adapter = module(folder/'constructor_adapter.py','_private_attention_constructor_fixture')
    successor = module(folder/'train.py','_private_attention_CLI_fixture')
    context = successor.load_driver(phase/'portable_context_steering_public_interface_20261008_v1')
    public = context.load_public_interface(phase/'portable_internal_be_public_interface_20261007_v2')
    native_path = phase/'wikics_native_polynormer_r_donor_preparation_20261007_v1/vendor/native_polynormer.py'
    hook_root = phase/'portable_private_local_attention_20261008_v1'

    # Explicitly synthetic constructor provider, isolated from the public module.
    # Scientific CLI never exposes or installs this reduced recipe.
    fixture_public = ModuleType('_private_attention_small_fixture_public')
    fixture_public.__dict__.update(vars(public))
    original_recipe = public.recipe
    def small_recipe(task):
        value = original_recipe(task)
        value['model'].update(in_channels=5,hidden_channels=8,local_layers=2,global_layers=1)
        return value
    fixture_public.recipe = small_recipe
    tree = ast.parse(Path(public.__file__).read_text())
    native_class = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Session')
    init = copy.deepcopy(next(n for n in native_class.body
                              if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
    namespace = dict(vars(fixture_public))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[init],type_ignores=[])),
                 '<unchanged-shared-engineering-constructor>','exec'),namespace)
    shared_class = type('SmallSharedEngineeringSession',(public.Session,),
                        {'__init__':namespace['__init__']})
    private_public = adapter.adapted_public(fixture_public,hook_root)
    shared = shared_class('wikics','be_unit_contrastive',9311,'cpu',native_path)
    private = private_public.Session('wikics','be_unit_contrastive',9311,'cpu',native_path)
    assert not hasattr(shared,'private_local_attention')
    assert len(private.optimizers)==1 and len(private.optimizers[0].state)==0
    optimized = [p for group in private.optimizers[0].param_groups for p in group['params']]
    assert len(optimized)==len(set(map(id,optimized)))
    assert set(map(id,optimized))==set(map(id,private.model.parameters()))
    banks = [getattr(c.parametrizations,n).original
             for c in private.model.models[0].body.local_convs for n in ('att_src','att_dst')]
    assert set(map(id,banks))<=set(map(id,optimized))
    for a,b in zip(shared.streams,private.streams):
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    assert public.Session is fixture_public.Session
    checks = {'isolated_constructor_before_fresh_Adam_exact_membership':True,
              'unchanged_shared_class_and_initial_streams':True}

    generator = torch.Generator(device='cpu').manual_seed(411)
    x = torch.randn(580,5,generator=generator)
    ids = torch.arange(580)
    edges = torch.stack((torch.cat((ids,(ids+7)%580)),torch.cat(((ids+1)%580,(ids+13)%580))))
    labels = ids%10
    train = {'x':x,'edge_index':edges,'ids':ids,'y':labels}
    batch = {'x':x,'edge_index':edges,'ids':ids}
    shared.model.eval();private.model.eval()
    before_shared = shared.forward(batch)
    before_private = private.forward(batch)
    assert all(torch.equal(a,b) for a,b in zip(before_shared,before_private))
    checks['initial_original_Session_shadow_forward_equivalence']=True

    # Existing public target generator used only for the fabricated fixture.
    targets = context.prepare_targets(train,torch,torch.device('cpu'))
    with tempfile.TemporaryDirectory(prefix='_private_attention_CPU_',dir=folder) as temporary:
        path = Path(temporary)/'fabricated_targets.npz'
        np.savez(path,**targets.arrays)
        digest = successor.sha(path)
        consumed = successor.consume_targets(context,path,digest,train,torch,torch.device('cpu'))
        assert consumed.array_hashes==targets.array_hashes
        assert consumed.metadata['target_relations_regenerated'] is False
        try:
            successor.consume_targets(context,path,'0'*64,train,torch,torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError('Changed target archive binding accepted')
        checks['caller_targets_consumed_unchanged_and_wrong_hash_rejected']=True

        # Actual dispatcher, constructor wrapper, target facade and replay engine.
        successor.configure_driver(context,hook_root,path,digest)
        session,facade,policy,_ = context.make_session(fixture_public,'shared_route',
            9413,torch.device('cpu'),native_path,train,consumed)
        assert policy.own_selected_four is False and policy.underlying_arm=='be_unit_contrastive'
        assert session.config['training']==original_recipe('wikics')['training']
        assert session.config['contrastive']['residual_weight']==0.
        context.install_replay(session,diagnostics=True)
        observed = {f'{i}.{name}':[] for i,name,_ in session.private_local_attention.sites}
        handles=[]
        for i,name,selector in session.private_local_attention.sites:
            def capture(module,args,key=f'{i}.{name}'):
                observed[key].append(module.member)
            handles.append(selector.register_forward_pre_hook(capture))
        session.train_step(batch,labels)
        for handle in handles:handle.remove()
        expected_rows=[m for _ in range(4) for m in range(4)]
        assert all(rows==expected_rows for rows in observed.values())
        assert session.execution_totals==dict(shadow_member_forwards=8,replay_member_forwards=8,
            output_cotangent_collections=1,member_reverse_collections=8,
            optimizer_bank_updates=1,exact_member_RNG_endpoint_checks=1)
        assert session.steps==1 and session.last_replay_diagnostics['exact_member_RNG_endpoint']
        assert all(selector.member is None for _,_,selector in session.private_local_attention.sites)
        active_rows=[0.,0.,0.,0.]
        for conv in session.model.models[0].body.local_convs:
            for name in ('att_src','att_dst'):
                parameter=getattr(conv.parametrizations,name).original
                moments=session.optimizers[0].state[parameter]['exp_avg']
                for m in range(4):active_rows[m]+=float(moments[m].abs().sum())
        assert all(v>0 for v in active_rows)
        checks['all_shadow_replay_members_select_correct_scorer_rows']=True
        checks['one_complete_synthetic_replay_VJP_Adam_transition_and_RNG_endpoint']=True
        checks['every_private_scorer_route_received_native_update']=True

        # In-memory selected-state semantics only; no scientific snapshot files.
        session.model.set_global(True);session.model.eval()
        run={'arm':'public_private_attention_context__shared_route',
             'underlying_session_arm':'be_unit_contrastive','seed':9413}
        snapshot=context.joint_snapshot(session,1,0.,[0.,0.,0.,0.],run)
        assert snapshot['private_local_attention']==adapter.descriptor(session)
        restored=adapter.reconstruct_selected(fixture_public,hook_root,snapshot,'cpu',native_path)
        assert restored.model.models[0].body._global is True
        assert len(restored.optimizers[0].state)==0 and restored.steps==0
        try:
            restored.train_step(batch,labels)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Serving-only reconstruction accepted a training step')
        native_calls=[];restored_calls=[]
        with torch.no_grad():
            for m in range(4):
                native_calls.append(session.model.member_forward(batch,m))
                restored_calls.append(restored.model.member_forward(batch,m))
        assert all(torch.equal(a,b) for old,new in zip(native_calls,restored_calls)
                   for a,b in zip(old,new))
        bad=copy.copy(snapshot);bad['private_local_attention']=dict(snapshot['private_local_attention'])
        bad['private_local_attention']['hook_source_sha256']='0'*64
        try:
            adapter.reconstruct_selected(fixture_public,hook_root,bad,'cpu',native_path)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong selected scorer source accepted')
        checks['joint_snapshot_metadata_and_selected_global_reconstruction']=True
        checks['selected_source_mismatch_rejected_no_optimizer_history_inherited']=True

    assert public.recipe is original_recipe
    return {'schema':'private-attention-context-integration-CPU-fixture-v1',
        'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,
        'device':'cpu','fabricated_TRAIN_rows':580,'fabricated_input_features':5,
        'engineering_hidden_width':8,'engineering_local_layers':2,
        'scientific_recipe_overridden_by_CLI':False,'public_module_mutated':False,
        'checks':checks,'passed_checks':len(checks),'synthetic_optimizer_steps':1,
        'scientific_training_updates':0,'scientific_data_score_or_checkpoint_access':False,
        'GPU_calls':0,'performance_evidence':False,'author_family_admitted':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',type=Path,default=Path(__file__).resolve().parent.parent)
    args=parser.parse_args()
    print(json.dumps(run(args.phase),indent=2))

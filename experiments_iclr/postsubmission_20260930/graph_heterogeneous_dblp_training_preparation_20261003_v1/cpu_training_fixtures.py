"""Later root synthetic CPU checks of added families and complete checkpoint RNG.

No ZIP, real labels, data acquisition or driver main. Three synthetic optimizer
steps exercise saved model/AdamW/OneCycle/member-RNG transport and replay.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def main():
    packet = Path(__file__).resolve().parent
    provenance = json.loads((packet/'PROVENANCE.json').read_text())
    for row in provenance['inputs']:
        data = (packet.parent/row['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes'],row['path']
    import torch
    assert torch.__version__.split('+')[0]=='2.1.2'
    torch.set_num_threads(1)
    core = load('qualified_HGB_cpu_fixture',packet.parent/provenance['implementation_source'])
    families = load('prepared_HGB_families_cpu_fixture',packet/'families.py')
    ids = torch.tensor([0,1,2],dtype=torch.long)
    graph = core.HeteroGraph({'0':3,'1':3},[
        core.Relation(0,'0','1',ids,ids),core.Relation(1,'1','0',ids,ids)])
    features = {t:torch.eye(3) for t in graph.ntypes}
    labels = torch.tensor([0,1,0])
    models = families.build(core,graph,{'0':3,'1':3},2,seed=131,factor_seed=900132,
                           device='cpu',requested=['native_HGT','global_BE','CP','unrestricted','shared_relation','untied_HGT','wider_BE'])
    checks = []
    for arm,model in models.items():
        model.eval(); value = model(graph,features,'0')
        assert value.shape==((1 if arm=='native_HGT' else 4),3,2),arm
    checks.append('all_seven_runnable_family_shapes')
    with torch.no_grad():
        torch.testing.assert_close(models['CP'](graph,features,'0'),models['unrestricted'](graph,features,'0'),atol=0,rtol=0)
    checks.append('CP_free_table_initial_function_matches_in_FP32')
    factors = models['shared_relation'].factors
    for layer in factors:
        residual = [layer.output(m,0)-layer.b[m] for m in range(4)]
        for value in residual[1:]:
            torch.testing.assert_close(value,residual[0],atol=2e-7,rtol=2e-7)
    checks.append('shared_relation_control_has_common_residual_no_member_c')
    model = models['CP'].train()
    streams = core.MemberStreams([7,11,13,17])
    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,total_steps=300,max_lr=1e-3,pct_start=.05)
    for step in range(3):
        optimizer.zero_grad(); loss = core.mean_member_ce(model(graph,features,'0',streams),ids,labels)
        loss.backward(); optimizer.step(); scheduler.step(step+1)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'synthetic_complete.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        streams=streams.state_dict(),CPU_RNG=torch.get_rng_state()),path)
        expected = model(graph,features,'0',streams).detach()
        with torch.no_grad():
            model.core.out.weight.add_(1)
        saved = torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); streams.load_state_dict(saved['streams'])
        torch.set_rng_state(saved['CPU_RNG'])
        torch.testing.assert_close(model(graph,features,'0',streams).detach(),expected,atol=0,rtol=0)
        assert scheduler.last_epoch==3 and all(float(value['step'])==3 for value in optimizer.state.values())
    checks.append('serialized_model_AdamW_OneCycle_and_private_dropout_states_replay')
    print(json.dumps(dict(status='PASS',checks=checks,synthetic_CPU_only=True,optimizer_steps=3,
                         driver_main_launched=False,real_data_labels_or_GPU_execution=False,
                         remote_invocation_scope='root receipt must record transport separately')))


if __name__=='__main__':
    main()

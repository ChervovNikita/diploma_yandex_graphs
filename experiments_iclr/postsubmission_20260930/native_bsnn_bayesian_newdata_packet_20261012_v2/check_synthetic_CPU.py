"""Tiny full-native Bayesian/KL/SciPy/checkpoint checks, not scientific fits."""
import copy
import io
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
sys.path.insert(0,str(HERE/'runtime'))
from adapters import bsnn_bayesian as port


def main():
    import torch
    import numpy as np
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    B=json.loads((HERE/'BSNN_BAYESIAN_SOURCE_BINDINGS.json').read_text())
    n=12;ids=torch.arange(n);torch.manual_seed(331)
    edges=torch.stack((ids.repeat(2),torch.cat(((ids+1)%n,(ids-1)%n))))
    edges=edges[:,(edges[0]*n+edges[1]).argsort()]
    graph={'x':torch.randn(n,6),'edge_index':edges}
    train={'ids':ids[:8],'y':ids[:8]%2};select={'ids':ids[8:],'y':ids[8:]%2}
    rows=[]
    for variant in port.VARIANTS:
        for arm in ('native_M1','original_GNNM'):
            model,opt,h,s,args=port.construct(torch,PHASE,B,graph,variant,arm,88201,'cpu')
            a=port.attach(torch,model,opt,h,s,args,graph,train,select,arm,88201,{'synthetic_variant':variant,'arm':arm})
            owned=[p for g in opt.param_groups for p in g['params']]
            assert len({id(p) for p in owned})==len(owned)==len(list(model.parameters()))
            assert all(group['lr']==.01 and group['weight_decay']==.0005 for group in opt.param_groups)
            start_rng=copy.deepcopy(a.capture_state())
            result=a.train_step(1)
            assert result['operation_counts']['optimizer_updates']==1
            assert result['operation_counts']['native_KL_evaluations']==a.members
            assert result['operation_counts']['native_posterior_layer_samples']==a.members*args['layers']
            assert result['regularization']>0 and result['objective']>result['ce']
            body=model.body if arm=='original_GNNM' else model
            before=copy.deepcopy(s.capture(np,torch,h,body))
            score=a.validate(1)
            assert s.exact(torch,before,s.capture(np,torch,h,body))
            assert score['operation_counts']['model_forwards']==4*a.members
            assert a.observe_validation(1,score)['save_selected']
            state=copy.deepcopy(a.capture_state());m=copy.deepcopy(model.state_dict());o=copy.deepcopy(opt.state_dict())
            buffer=io.BytesIO();torch.save({'model':m,'optimizer':o,'adapter':state},buffer)
            buffer.seek(0);saved=torch.load(buffer,weights_only=True)
            assert s.exact(torch,saved['adapter'],state)
            a.train_step(2);post_m=copy.deepcopy(model.state_dict());post_o=copy.deepcopy(opt.state_dict());post_state=copy.deepcopy(a.capture_state())
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);a.restore_state(saved['adapter'])
            assert s.exact(torch,state,a.capture_state())
            a.train_step(2)
            assert s.exact(torch,post_m,model.state_dict()) and s.exact(torch,post_o,opt.state_dict())
            assert s.exact(torch,post_state,a.capture_state()), 'Next-step SciPy/posterior/private/global RNG replay exact'
            model.eval();rng=copy.deepcopy(a.capture_state())
            with torch.no_grad():
                four=a.serving_draws(1,4)
                if arm=='native_M1':
                    sixteen=a.serving_draws(1,16)
                    assert torch.equal(four,sixteen[:4]), 'MC16 first4 exactly sourceMC4 with fixed same seed'
                    assert not torch.equal(sixteen[0],sixteen[1]), 'Posterior draws must not be copies'
            assert s.exact(torch,rng,a.capture_state()), 'Serving restores every owned stream/plain flag'
            rows.append({'variant':variant,'arm':arm,'synthetic_optimizer_maps':3,'source_returned_KL_preserved':True,
                         'native_grouped_Adam_exact':True,'source_four_draw_SELECT_RNG_exact':True,
                         'complete_native_plain_tensor_aliases_flags_and_SciPy_checkpoint':True,
                         'next_native_Adam_map_after_restore_exact':True,'MC16_first4_source_control_exact':arm=='native_M1',
                         'native_layers':args['layers'],'SELECT_native_draws':4*a.members,
                         'registered_parameter_scalars':sum(p.numel() for p in model.parameters()),
                         'serialized_complete_checkpoint_bytes':len(buffer.getvalue()),
                         'source_beta_epoch1':result['native_beta'],'actual_TRAIN_arrays_decoded':False,
                         'scientific_fits':0,'GO':False})
    # Exercise source patience/key without500 numerical toy fits.
    a._source_key=None;a._source_stale=0;a._last_source_selection=0
    for epoch in range(1,202):
        decision=a.observe_validation(epoch,{'auroc':.5,'nll':1.})
        assert decision['stop_requested']==(epoch==201)
    report={'schema':'native-Bayesian-BSNN-synthetic-CPU-QA-v1','status':'PASS','cases':rows,
            'synthetic_optimizer_maps':sum(r['synthetic_optimizer_maps'] for r in rows),
            'patience_stop201_verified_without_extra_fits':True,'actual_input_qualification':False,
            'source_CUDA_resources_qualified':False,'actual_TRAIN_arrays_decoded':False,
            'scientific_fits':0,'CUDA_initialized':False,'GO':False}
    (HERE/'SYNTHETIC_CPU_QA.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()

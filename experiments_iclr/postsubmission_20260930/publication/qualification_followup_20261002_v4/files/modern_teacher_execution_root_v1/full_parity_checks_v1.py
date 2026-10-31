"""Extra full-graph ownership, bias and gradient checks at fixed tolerances."""
def qualification_parity(torch,adapter,spec,graph,train):
    from backbone_boundary_adapter import set_boundary_identity_
    device=graph.teacher_input.device
    single=adapter.TeacherFamily(adapter.specification(spec['backbone'],'single_author',spec['config'],spec['seed'])).to(device).eval()
    shared=adapter.TeacherFamily(adapter.specification(spec['backbone'],'gnnm_boundary_4',spec['config'],spec['seed'])).to(device).eval()
    set_boundary_identity_(shared.boundary)
    pairs=[('stem','lin1'),('head','lin3')] if spec['backbone']=='polyformer_mono' else [('stem','lin_in'),('local_head','pred_local'),('global_head','pred_global')]
    for boundary,native_name in pairs:
        if not isinstance(getattr(shared.boundary.core,native_name),torch.nn.Identity):
            raise ValueError('Dormant boundary owner remains trainable')
        block=getattr(shared.boundary,boundary);native=getattr(single.models[0],native_name)
        torch.testing.assert_close(block.weight,native.weight,rtol=0,atol=0)
        torch.testing.assert_close(block.B,native.bias[None].expand_as(block.B),rtol=0,atol=0)
    rows=[]
    for stage in ((False,True) if spec['backbone']=='polynormer_r' else (False,)):
        single.set_global_stage(stage);shared.set_global_stage(stage)
        with torch.no_grad():a,b=single(graph),shared(graph)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError('Nonfinite complete-graph identity logits')
        for m in range(4):torch.testing.assert_close(a[0],b[m],rtol=1e-5,atol=1e-6)
        single.zero_grad(set_to_none=True);shared.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(single(graph)[0,train.nodes],train.labels).backward()
        torch.nn.functional.cross_entropy(shared(graph)[:,train.nodes].reshape(-1,spec['native']['classes']),train.labels.repeat(4)).backward()
        for model in (single,shared):
            for name,p in model.named_parameters():
                if not bool(torch.isfinite(p).all()) or (p.grad is not None and not bool(torch.isfinite(p.grad).all())):
                    raise ValueError('Nonfinite parameter/gradient in parity: '+name)
        native_parameters=dict(single.models[0].named_parameters());compared=0;bias_rows=0
        for name,p in shared.boundary.core.named_parameters():
            expected=native_parameters[name].grad
            if expected is not None or p.grad is not None:
                if expected is None or p.grad is None:raise ValueError('Body gradient availability mismatch')
                torch.testing.assert_close(expected,p.grad,rtol=1e-4,atol=1e-6);compared+=1
        for boundary,native_name in pairs:
            block=getattr(shared.boundary,boundary)
            expected=native_parameters[native_name+'.weight'].grad
            if expected is not None or block.weight.grad is not None:
                if expected is None or block.weight.grad is None:raise ValueError('Weight gradient availability mismatch')
                torch.testing.assert_close(expected,block.weight.grad,rtol=1e-4,atol=1e-6);compared+=1
            expected_bias=native_parameters[native_name+'.bias'].grad
            if expected_bias is not None or block.B.grad is not None:
                if expected_bias is None or block.B.grad is None:raise ValueError('Bias gradient availability mismatch')
                for m in range(4):torch.testing.assert_close(expected_bias/4,block.B.grad[m],rtol=1e-4,atol=1e-6);bias_rows+=1
                torch.testing.assert_close(expected_bias,block.B.grad.sum(0),rtol=1e-4,atol=1e-6)
        rows.append(dict(global_stage=stage,max_absolute_logit_difference=float((b-a).abs().max()),
                         common_body_gradient_tensors_compared=compared,private_bias_gradient_rows_compared=bias_rows,
                         all_graph_logits_and_active_gradients_finite=True,dormant_owner_removal_verified=True,full_graph=True))
    return dict(rtol_logits=1e-5,atol_logits=1e-6,rtol_gradients=1e-4,atol_gradients=1e-6,
                checks=rows,native_vs_identity_boundary_passed=True,shortened_fit_report_eligible=False)

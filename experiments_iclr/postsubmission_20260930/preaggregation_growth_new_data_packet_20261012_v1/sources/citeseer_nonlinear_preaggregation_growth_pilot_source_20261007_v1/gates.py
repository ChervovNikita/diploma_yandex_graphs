"""Actual copied-native gates, evaluated only inside adopted native runtime."""

GRAD_RTOL, GRAD_ATOL = 1e-6, 1e-7


def loss(torch, positive, negative, pooled):
    def own(p,n):
        return -torch.nn.functional.logsigmoid(p).mean()-torch.nn.functional.logsigmoid(-n).mean()
    return .5*own(positive.mean(1),negative.mean(1))+.5*own(positive,negative) if pooled else own(positive,negative)


def paired(torch, steps, candidate, reference, warm, features, support, queries,
           seed, device, *, training, route, independent=False):
    """Compare all pre-growth leaves on identical actual native loss and RNG.

    FP32 logits must be exactly equal. Old gradients must satisfy the declared
    FP32 tolerance; literal equality is separately reported, never inferred.
    The native no-growth reference includes every existing BE/LN/bias parameter.
    """
    candidate_names = dict(candidate.named_parameters())
    reference_names = dict(reference.named_parameters())
    old = list(reference_names)
    if not set(old).issubset(candidate_names): raise ValueError("Pre-growth names changed")
    if any(not torch.equal(candidate_names[n],reference_names[n]) for n in old):
        raise ValueError("Common warm pre-growth bytes differ")
    stream_c = steps.DropoutStreams(torch,device,seed+200000)
    stream_r = steps.DropoutStreams(torch,device,seed+200000)
    index = 0 if route is None else route
    pc,nc = steps.functional_forward(torch,candidate,candidate_names,features,support,queries,
        training=training,stream=(stream_c,index),route=route,advance=False)
    pr,nr = steps.functional_forward(torch,reference,reference_names,features,support,queries,
        training=training,stream=(stream_r,index),route=route,advance=False)
    if not torch.equal(pc,pr) or not torch.equal(nc,nr):
        raise ValueError("Copied-native FP32 growth-zero logits fail exact gate")
    objective_c,objective_r = loss(torch,pc,nc,not independent),loss(torch,pr,nr,not independent)
    names_c = list(candidate_names)
    gc = torch.autograd.grad(objective_c,tuple(candidate_names[n] for n in names_c),allow_unused=True)
    gr = torch.autograd.grad(objective_r,tuple(reference_names[n] for n in old),allow_unused=True)
    gc = dict(zip(names_c,gc)); gr = dict(zip(old,gr))
    records = []
    for name in old:
        a,b = gc[name],gr[name]
        if a is None or b is None or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("Old native gradient disconnected/nonfinite: "+name)
        difference = float((a-b).abs().max())
        agrees = torch.allclose(a,b,rtol=GRAD_RTOL,atol=GRAD_ATOL)
        records.append({"name":name,"exact":torch.equal(a,b),"max_abs_difference":difference,
            "reference_max_abs":float(b.abs().max()),"within_declared_tolerance":agrees})
        if not agrees: raise ValueError("Old native gradient failed FP32 tolerance: "+name)
    growth_records = []
    for name in names_c:
        if name in reference_names: continue
        gradient = gc[name]
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise ValueError("New growth gradient disconnected/nonfinite: "+name)
        if name.endswith(".B"):
            if bool((gradient != 0).any()): raise ValueError("Zero-incoming growth must have exact grad_B=0")
            growth_records.append({"name":name,"gradient_exact_zero":True})
        elif name.endswith(".V"):
            active = list(range(len(gradient))) if route is None or independent else [route]
            norms = [float(gradient[m].norm()) for m in range(len(gradient))]
            if any(norms[m] <= 0 for m in active):
                raise ValueError("Incoming growth gradient is not live for active member")
            growth_records.append({"name":name,"member_gradient_norms":norms,"active_members":active})
        else: raise ValueError("Unexpected new parameter at growth gate: "+name)
    donor_difference = None
    if not training:
        with torch.no_grad():
            pd,nd = steps.functional_forward(torch,warm,dict(warm.named_parameters()),features,support,queries,training=False)
        if not torch.equal(pc,pd.expand_as(pc)) or not torch.equal(nc,nd.expand_as(nc)):
            raise ValueError("Warm donor logits differ from copied-native growth-zero bank")
        donor_difference = 0.
    return {"training":training,"route":route,"exact_positive_negative_logits":True,
        "maximum_donor_logit_difference":donor_difference,"all_old_gradients_exact":all(r["exact"] for r in records),
        "old_gradient_rtol":GRAD_RTOL,"old_gradient_atol":GRAD_ATOL,"old_gradients":records,
        "growth_gradients":growth_records,"all_parameters_trainable":all(v.requires_grad for v in candidate.parameters()),
        "optimizer_updates":0,"dropout_masks_replayed":training,"dropout_stream_advanced":False}


def qualify(torch,steps,model,reference,warm,features,support,queries,seed,device):
    """Pooled deterministic inclusion and every live dropout route gate."""
    results = []
    if hasattr(model,"routes"):
        for member,(candidate,old) in enumerate(zip(model.routes,reference.routes)):
            for training in (False,True):
                result = paired(torch,steps,candidate,old,warm,features,support,queries,
                    seed+5*member,device,training=training,route=0,independent=True)
                result["independent_member"] = member; results.append(result)
        identities = [{id(p) for p in route.parameters()} for route in model.routes]
        if any(identities[a]&identities[b] for a in range(4) for b in range(a+1,4)):
            raise ValueError("Independent native controls share parameters")
    else:
        results.append(paired(torch,steps,model,reference,warm,features,support,queries,
            seed,device,training=False,route=None))
        for member in range(model.member_count):
            results.append(paired(torch,steps,model,reference,warm,features,support,queries,
                seed,device,training=True,route=member))
    return {"actual_native_models":True,"actual_TRAIN_calibration_episode":True,
        "copied_native_forward_gradient_gate_passed":True,"cases":results,
        "FP32_bitwise_old_gradient_claim":all(r["all_old_gradients_exact"] for r in results),
        "VALID_TEST_loaded":False,"no_parameter_or_optimizer_update":True}

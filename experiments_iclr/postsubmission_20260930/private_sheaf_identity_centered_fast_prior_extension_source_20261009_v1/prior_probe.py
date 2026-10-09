"""Root engineering utility only; no model/provider import or CLI execution."""
from support import require


def probe(np,torch,bank,cfg,fit,old,helpers):
    """On a fresh full-input identity bank, probe one factor and restore exactly."""
    before=old.cpu_tree(torch,bank.state_dict())
    before_rng=helpers.capture_rng(np,torch)
    parameters=list(bank.parameters())
    private_names={path+suffix for path in bank.factor_paths for suffix in ('.r','.s')}
    factors=[value for member in bank.members for name,value in member.named_parameters() if name in private_names]
    require(all(torch.equal(value,torch.ones_like(value)) for value in factors),'Fresh full-input identity factors for analytic prior probe')
    def total():return sum(fit.member_prior(torch,bank,member,cfg['sheaf_decay']) for member in bank.members)
    at_identity=total()
    require(float(at_identity.item())==0.0,'Identity-centered prior exactly zero at initialization')
    first=factors[0]
    try:
        with torch.no_grad():first.view(-1)[0].add_(0.125)
        away=total()
        require(torch.isfinite(away).item() and float(away.item())>0,'Prior nonzero for declared engineering displacement from identity')
        gradients=torch.autograd.grad(away,parameters,allow_unused=True)
        require(all(value is None or torch.isfinite(value).all().item() for value in gradients),'Finite prior-only gradients')
        fast_ids={id(value) for value in factors}
        require(all(gradient is None for value,gradient in zip(parameters,gradients) if id(value) not in fast_ids),'Prior has no slow-native gradient')
        first_gradient=gradients[next(i for i,value in enumerate(parameters) if value is first)]
        require(first_gradient is not None and float(first_gradient.view(-1)[0].item())>0,'Centered prior gradient points back toward identity')
        result=dict(centered_prior_probe_passed=True,prior_at_identity=0.0,prior_away_from_identity=float(away.item()),
                    displaced_private_gradient=float(first_gradient.view(-1)[0].item()),engineering_displacement=0.125,
                    coefficient=cfg['sheaf_decay'],scientific_initializer_changed=False,scientific_metric_access=False)
    finally:
        bank.load_state_dict(before,strict=True)
        require(old.exact(torch,old.cpu_tree(torch,bank.state_dict()),before),'Probe restores exact fresh scientific parameter/buffer values')
        helpers.restore_rng(np,torch,before_rng)
        require(old.exact(torch,helpers.capture_rng(np,torch),before_rng),'Probe restores exact owned RNG')
        bank.assert_ownership()
    return result

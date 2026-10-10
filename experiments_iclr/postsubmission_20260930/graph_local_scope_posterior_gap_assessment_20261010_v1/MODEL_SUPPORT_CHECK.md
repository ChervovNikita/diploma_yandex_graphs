# Manual support and likelihood check

This is a prospective model check, not numerical evidence, a novelty result or an author-code audit.

BNA prints a Bernoulli prior conditional and a ConcreteBernoulli variational conditional. For a literal discrete Bernoulli measure p, the open interval(0,1) has p-measure zero. A finite-temperature Concrete variable lies in that interval with probability one. Therefore q is not absolutely continuous with respect to p and KL(q||p) is infinite. A relaxed implementation may use a different common-support model; it has not been read here.

The prospective option explicitly uses Beta priors p(v), Beta variational q(v), and an identical continuous conditional p(z|v)=q(z|v). By the chain rule, the joint KL is then KL(q(v)||p(v)); no continuous-to-discrete KL is introduced. The likelihood is the actual continuous attenuation predictor, so this is not an exact discrete-depth model.

Four learned private predictors are distinct likelihood functions. Specify a uniform latent route index for each labeled node and fix its variational distribution to that prior. Jensen gives mean_m log p_m(y|z) <= log mean_m p_m(y|z); the route-index KL is zero. Thus the mean own likelihood plus Beta KL is an explicit lower-bound objective rather than equality to served mixture log likelihood. The source must preserve the original1/M normalization and charge all latent sampling and posterior work.

No posterior calibration, exact inference, graph-specific accuracy, variational optimality or generalization guarantee follows from this construction. Dense weights are point trained. Prospective alpha/beta, temperature, estimator, checkpoint-serving draws and roles are still unadopted; no tensor or source integration was performed.

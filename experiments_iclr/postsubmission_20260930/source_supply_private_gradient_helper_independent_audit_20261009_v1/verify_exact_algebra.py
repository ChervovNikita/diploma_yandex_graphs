"""Exact rational analytical fixtures; no float, model, Torch or scientific data."""
from fractions import Fraction as F
import json
from pathlib import Path


def observed_pool_gradient(p, peer_probabilities, outcome, labels=1):
    members=1+len(peer_probabilities)
    q=(p+sum(peer_probabilities,F(0)))/members
    observed=p if outcome else 1-p
    peer_observed=[value if outcome else 1-value for value in peer_probabilities]
    responsibility=observed/(observed+sum(peer_observed,F(0)))
    responsible_ce=responsibility*(p-outcome)/labels
    direct_chain=(q-outcome)*p*(1-p)/(members*q*(1-q)*labels)
    assert responsible_ce==direct_chain
    return responsibility,responsible_ce


def main():
    peers=[F(1,4),F(1,2),F(3,4)]
    # Binary categorical target-class logit is sufficient to check the exact
    # source responsibility factor. Off-target derivative has the opposite sign.
    rho_f,cat_f=observed_pool_gradient(F(1,2),peers,1)
    rho_a,cat_a=observed_pool_gradient(F(1,4),peers,1)
    assert (rho_f,cat_f,rho_a,-cat_a)==(F(1,4),F(-1,8),F(1,7),F(3,28))
    # Bernoulli labels remain separate. The second label includes the NEGATIVE
    # observation, hence uses (1-p)/(sum observed 1-p), not p/sum p.
    bern_r0,bern_g0=observed_pool_gradient(F(1,2),peers,1,labels=2)
    bern_r1,bern_g1=observed_pool_gradient(F(3,4),peers,0,labels=2)
    assert (bern_r0,bern_g0,bern_r1,bern_g1)==(F(1,4),F(-1,16),F(1,7),F(3,56))
    _,bern_probe0=observed_pool_gradient(F(1,4),peers,1,labels=2)
    _,bern_probe1=observed_pool_gradient(F(1,4),peers,0,labels=2)
    assert (-bern_probe0,-bern_probe1)==(F(3,56),F(-1,24))
    # Exact finite-guard implication: if J and absent both do not increase,
    # supply=J+absent cannot increase. J alone does not establish supply gain.
    reference_j,reference_absent=F(-1,10),F(4,5)
    trial_j,trial_absent=F(-1,5),F(79,100)
    assert trial_j<=reference_j and trial_absent<=reference_absent
    assert trial_j+trial_absent<reference_j+reference_absent
    gaming_j,gaming_absent=F(-1,5),F(9,10)
    assert gaming_j<reference_j and gaming_absent>reference_absent
    assert gaming_j+gaming_absent==reference_j+reference_absent
    # Positive rescaling cannot change this cone. The exact projection of
    # [1,-1] onto a*d0<=0, for every positive a, is [0,-1]. Float norm
    # overflow/underflow may not turn the constraint into an empty constraint.
    expected=(F(0),F(-1))
    for scale in (F(1),F(10**200),F(1,10**200)):
        assert scale>0 and scale*expected[0]<=0
        assert (F(1)-expected[0])**2+(F(-1)-expected[1])**2==1
    result={
        "schema":"independent-source-supply-exact-rational-fixtures-v1",
        "status":"passed",
        "categorical_target_logit_factual_gradient":str(cat_f),
        "categorical_target_logit_subtracted_probe_gradient":str(-cat_a),
        "bernoulli_factual_gradient_per_two_labels":[str(bern_g0),str(bern_g1)],
        "bernoulli_subtracted_probe_gradient_per_two_labels":[str(-bern_probe0),str(-bern_probe1)],
        "cone_scale_invariance_expected_projection":[str(value) for value in expected],
        "checks":["exact categorical responsibility-weighted CE equals direct pool chain rule",
                  "positive and negative Bernoulli responsibilities and independent label mean",
                  "exact J-plus-absent supply guard implication and denominator-gaming counterexample",
                  "exact positive-rescaling invariance of the analytical cone"],
        "floating_point_tests":False,"torch_or_models_executed":False,
        "scientific_data_or_outcomes_accessed":False,
        "qualification_limit":"Exact algebra examples only. Not execution of helper Torch/autograd/replay, native source removal, training, prediction quality, robustness, novelty or scientific utility."
    }
    Path(__file__).with_name("EXACT_ALGEBRA_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"floating_point_tests":False,"torch_or_models_executed":False}))


if __name__=="__main__":main()

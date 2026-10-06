"""Small algebra illustrations only; no model, data, labels or scientific fits."""
from fractions import Fraction as F
import json,math
from pathlib import Path

# Balanced weights preserve a common loss/gradient with fixed-count reduction.
z1=[F(1,2),F(-1,2),F(1,5),F(-1,5)]
z2=[F(3,10),F(1,10),F(-3,10),F(-1,10)]
weights=[[1+s*F(1,2)*z for z in field] for field,s in [(z1,1),(z1,-1),(z2,1),(z2,-1)]]
assert all(sum(w[v] for w in weights)/4==1 for v in range(4))
assert all(F(1,2)<=x<=F(3,2) for w in weights for x in w)
g=[F(1),F(2),F(-1),F(3)]
ordinary=sum(g)/4
balanced=sum(sum(a*b for a,b in zip(w,g))/4 for w in weights)/4
assert ordinary==balanced
# Minibatch per-route weight-sum normalization breaks the identity.
normalized=sum(sum(w[i]*g[i] for i in range(2))/sum(w[:2]) for w in weights)/4
ordinary_batch=sum(g[:2])/2
assert normalized!=ordinary_batch
# Arbitrary route-dependent derivatives likewise break the samplewise identity.
route_g=[g,[F(2),F(1),F(-1),F(3)],g,g]
route_weighted=sum(sum(a*b for a,b in zip(w,v))/4 for w,v in zip(weights,route_g))/4
route_unweighted=sum(sum(v)/4 for v in route_g)/4
assert route_weighted!=route_unweighted

# Function-preserving positive affine gauge, homogeneous activation and fixed mask.
y=[.7,-.2,1.1];d=[2.,.5,1.5];r=[.8,1.2,.4];mask=[1.,0.,1.]
base=sum(max(a,0)*b*c for a,b,c in zip(y,r,mask))
gauged=sum(max(a*t,0)*(b/t)*c for a,t,b,c in zip(y,d,r,mask))
assert abs(base-gauged)<1e-14
# The original-coordinate Adam step equals LR eta/t and epsilon t*eps.
def adam_step(g,m,v,iteration,lr,eps):
 m=.9*m+.1*g;v=.999*v+.001*g*g
 step=lr*(m/(1-.9**iteration))/(math.sqrt(v/(1-.999**iteration))+eps)
 return m,v,step
for t in [2.,.5,1.7]:
 mz=vz=mo=vo=0.;a=[];b=[]
 for iteration,g0 in enumerate([.3,-.17,.11],1):
  mz,vz,sz=adam_step(g0/t,mz,vz,iteration,.001,1e-8)
  mo,vo,so=adam_step(g0,mo,vo,iteration,.001/t,t*1e-8)
  a.append(sz/t);b.append(so)
  assert abs(sz/t-so)<1e-14

# Zero-incoming nonlinear growth leaves output unchanged and has live input gradients.
h=[0.,2.];outgoing=1.;incoming=0.
residual=sum(math.tanh(v*incoming) for v in h)/2*outgoing
assert residual==0
analytic_input_derivative=sum(h)/2*outgoing
step=1e-6
finite_difference=sum(math.tanh(v*step) for v in h)/2*outgoing/step
assert abs(finite_difference-analytic_input_derivative)<1e-10
# Nonlinear messages can distinguish equal linear means (restricted interface).
h2=[1.,1.];linear_mean=sum(h)/2;linear_mean2=sum(h2)/2
nonlinear_mean=sum(math.tanh(v) for v in h)/2
nonlinear_mean2=sum(math.tanh(v) for v in h2)/2
assert linear_mean==linear_mean2 and nonlinear_mean!=nonlinear_mean2
# A Stiefel output basis picks top-r gradient singular modes; pure Frob budget can collapse.
stiefel_top2=3**2+2**2;stiefel_other2=2**2+1**2
frobenius_equal_top2=F(stiefel_top2,2);frobenius_rank1=F(9)
assert stiefel_top2>stiefel_other2 and frobenius_rank1>frobenius_equal_top2

def serial(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,list):return [serial(v) for v in x]
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 return x
output={'scope':'Finite stdlib arithmetic only, no statistical/model experiment','all_assertions_passed':True,'weights':weights,'common_gradient_ordinary':ordinary,'common_gradient_balanced':balanced,'per_route_normalized_batch_mean':normalized,'ordinary_batch_mean':ordinary_batch,'different_route_gradients_weighted':route_weighted,'different_route_gradients_unweighted':route_unweighted,'gauge_base_output':base,'gauge_compensated_output':gauged,'Adam_coordinate_equivalence_checked_steps':3,'growth_initial_residual':residual,'growth_input_derivative':analytic_input_derivative,'equal_linear_mean':linear_mean,'preaggregation_tanh_values':[nonlinear_mean,nonlinear_mean2],'stiefel_top2_gradient_energy':stiefel_top2,'stiefel_other2_gradient_energy':stiefel_other2,'Frobenius_budget_equal_top2_energy':frobenius_equal_top2,'Frobenius_budget_rank1_energy':frobenius_rank1}
Path(__file__).with_name('ANALYTIC_CHECKS.json').write_text(json.dumps(serial(output),indent=2)+'\n')
print('All finite algebra checks passed')

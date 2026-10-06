"""Exact finite arithmetic illustrations only; no graph/model/data payloads."""
from fractions import Fraction as F
import json, math
from pathlib import Path


def add(x,y): return [[a+b for a,b in zip(r,s)] for r,s in zip(x,y)]
def scale(c,x): return [[c*a for a in r] for r in x]
def mm(x,y): return [[sum(a*b for a,b in zip(r,col)) for col in zip(*y)] for r in x]
def mv(x,y): return [sum(a*b for a,b in zip(r,y)) for r in x]
def mean(x,y): return scale(F(1,2),add(x,y))
def rows(x): return [[a/sum(r) for a in r] for r in x]
def serial(x):
 if isinstance(x,list):return [serial(a) for a in x]
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 return str(x) if isinstance(x,F) else x

p=[[F(1,2),F(1,2)],[F(1,2),F(1,2)]]
d=[[F(1,4),F(0)],[F(0),F(0)]]
plus=add(p,d);minus=add(p,scale(-1,d))
assert mean(plus,minus)==p
renormalized=mean(rows(plus),rows(minus))
assert renormalized==[[F(7,15),F(8,15)],[F(1,2),F(1,2)]] and renormalized!=p
h=[F(1),F(-1)]
rp=[max(a,0) for a in mv(plus,h)];rm=[max(a,0) for a in mv(minus,h)]
relu_mean=[(a+b)/2 for a,b in zip(rp,rm)]
assert relu_mean==[F(1,8),F(0)] and mv(p,h)==[F(0),F(0)]
depth_mean=mean(mm(plus,plus),mm(minus,minus))
assert depth_mean==add(mm(p,p),mm(d,d)) and depth_mean!=mm(p,p)
identity=[[F(1),F(0)],[F(0),F(1)]]
rank_one_score=[[F(1,2),F(1,2)],[F(1,2),F(1,2)]]
masked_delta=[[a*b for a,b in zip(r,s)] for r,s in zip(identity,rank_one_score)]
score_det=rank_one_score[0][0]*rank_one_score[1][1]-rank_one_score[0][1]*rank_one_score[1][0]
delta_det=masked_delta[0][0]*masked_delta[1][1]-masked_delta[0][1]*masked_delta[1][0]
assert score_det==0 and delta_det==F(1,4)
# P=I, a=ones, c=+/-1/2: all eigenvalues of P_plus are3/2.
assert scale(F(3,2),identity)[0][0]>1
# Common-input qualification, same centered operators, different member inputs.
hplus=[F(1),F(0)];hminus=[F(0),F(0)]
actual=[(a+b)/2 for a,b in zip(mv(plus,hplus),mv(minus,hminus))]
common_average=mv(p,[(a+b)/2 for a,b in zip(hplus,hminus)])
assert actual!=common_average
# Separate symmetric degree normalization of weighted P (base degrees are1).
def symnorm(x):
 degree=[sum(r) for r in x]
 return [[float(a)/math.sqrt(float(degree[i]*degree[j])) for j,a in enumerate(r)] for i,r in enumerate(x)]
splus=symnorm(plus);sminus=symnorm(minus)
symmean=[[.5*(a+b) for a,b in zip(r,s)] for r,s in zip(splus,sminus)]
assert abs(symmean[0][0]-7/15)<1e-14 and abs(symmean[0][1]-.5)>1e-4
output={'scope':'stdlib finite arithmetic, no scientific model/data execution','all_assertions_passed':True,'P':p,'Delta':d,'P_plus':plus,'P_minus':minus,'exact_mean':mean(plus,minus),'mean_row_normalized':renormalized,'mean_symmetric_normalized':symmean,'relu_mean':relu_mean,'baseline_linear_and_relu_output':mv(p,h),'mean_two_layer_operator':depth_mean,'baseline_two_layer_operator':mm(p,p),'rank_one_score_determinant':score_det,'masked_delta_determinant':delta_det,'max_eigenvalue_constant_gate_example':'3/2','mean_different_input_outputs':actual,'P_times_mean_inputs':common_average}
Path(__file__).with_name('ALGEBRA_CHECK.json').write_text(json.dumps(serial(output),indent=2)+'\n')
print('All finite arithmetic assertions passed')

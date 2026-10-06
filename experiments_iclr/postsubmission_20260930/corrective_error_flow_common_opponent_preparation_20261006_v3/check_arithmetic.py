"""Artificial arithmetic only; no model, dataset, label file or scoring call."""
from pathlib import Path
import ast,hashlib,importlib.util,json,math
D=Path(__file__).resolve().parent
P=D.parent
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def state(rows):
 probs=[]
 for member in rows:
  probs.append([[x/math.fsum(row) for x in row] for row in member])
 return {"A_ids":[0,1,2],"native_FP32_logits":[[[math.log(x) for x in row] for row in member] for member in probs],"served_FP32_member_probabilities":probs,"served_FP32_pool_probabilities":[[math.fsum(probs[m][i][c] for m in range(4))/4 for c in range(5)] for i in range(3)]}
def five(a,b,c):return [a,b,c,1e-8,1e-8]
old=load(P/"corrective_error_flow_source_preparation_20261006_v2/error_flow.py","predecessor")
new=load(D/"error_flow.py","candidate")
initial=state([[five(.2,.5,.3),five(.35,.4,.25),five(.2,.7,.1)],[five(.2,.5,.3),five(.35,.25,.4),five(.2,.1,.7)],[five(.2,.5,.3),five(.35,.4,.25),five(.2,.7,.1)],[five(.2,.5,.3),five(.35,.25,.4),five(.2,.1,.7)]])
endpoint=state([[five(.34,.4,.26),five(.35,.4,.25),five(.2,.7,.1)],[five(.34,.26,.4),five(.35,.25,.4),five(.2,.1,.7)],[five(.34,.4,.26),five(.35,.4,.25),five(.2,.7,.1)],[five(.34,.26,.4),five(.35,.25,.4),five(.2,.1,.7)]])
args=(initial,endpoint,[0,0,0],[0,1,2],[0,3,5])
a=old._analyze(*args);b=new._analyze(*args)
checks=[]
for key in a:
 if b[key]!=a[key]:raise RuntimeError("Original calculation changed: "+key)
checks.append("every_original_V2_report_field_exactly_preserved")
r=new._state_rows(initial,[0,0,0]);e=new._state_rows(endpoint,[0,0,0])
if not (r[1]["all_members_wrong"] and r[1]["correct"] and not r[1]["shared_strict_opponents"]):raise RuntimeError("All-wrong averaging counterexample failed")
checks.append("all_members_wrong_can_pool_correctly")
if not (r[0]["shared_strict_opponents"]==(1,2) and not r[0]["correct"]):raise RuntimeError("Shared rival classification failed")
checks.append("strict_shared_rival_detected")
if not (e[0]["correct"] and e[0]["correct_members"]==0 and not e[0]["shared_strict_opponents"]):raise RuntimeError("No-correct-member repair failed")
checks.append("repair_need_not_create_a_correct_member")
if b["baseline_shared_strict_opponent_pool_errors"]["n"]!=1 or b["shared_opponent_repair_details"]["repaired_without_a_correct_member"]!=1:raise RuntimeError("Fixed cohort/repair count failed")
checks.append("fixed_baseline_shared_opponent_cohort_accounted")
if r[2]["shared_strict_opponents"] or r[2]["correct"]:raise RuntimeError("No-common-rival failure counterexample failed")
# Any mix retains p_y=.2/normalizer; p1+p2=.8/normalizer > 2*p_y.
if not (.8>2*.2):raise RuntimeError("Elementary convex obstruction failed")
checks.append("no_shared_rival_does_not_prove_convex_repair_feasible")
try:new.analyze(*args)
except RuntimeError:checks.append("public_scientific_entry_remains_disabled")
else:raise RuntimeError("Scientific entry enabled")
oldtree=ast.parse((P/"corrective_error_flow_source_preparation_20261006_v2/error_flow.py").read_text());newtree=ast.parse((D/"error_flow.py").read_text())
oldf={x.name:ast.dump(x,include_attributes=False) for x in oldtree.body if isinstance(x,ast.FunctionDef)};newf={x.name:ast.dump(x,include_attributes=False) for x in newtree.body if isinstance(x,ast.FunctionDef)}
unchanged=[name for name in oldf if oldf[name]==newf[name]]
if "_summarize" not in unchanged or "_logsumexp" not in unchanged or "analyze" not in unchanged:raise RuntimeError("Original metric/release function changed")
checks.append("original_summarizer_NLL_helper_public_guard_AST_preserved")
v={"status":"PASS_ARTIFICIAL_COMMON_OPPONENT_ARITHMETIC_ONLY","checks":checks,"new_dataset_experiments":0,"scientific_model_or_label_files_read":False,"original_function_ASTs_preserved":unchanged,"candidate_sha256":hashlib.sha256((D/"error_flow.py").read_bytes()).hexdigest(),"NLL_probability_dtype_note":"Artificial Python floats; no native FP32 execution claim."}
with (D/"ARITHMETIC_RESULT.json").open("x") as f:json.dump(v,f,indent=2);f.write("\n")
print(json.dumps({"status":v["status"],"checks":len(checks),"scientific_scoring":False}))

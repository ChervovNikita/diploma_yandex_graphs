"""Composable anchor-checked HGT300 source stopping against complete root v16."""
import ast,hashlib,json,difflib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]

def once(text,old,new):
 if text.count(old)!=1:raise ValueError('HGT amendment anchor differs: '+old[:100])
 return text.replace(old,new)

def amend_engine(text):
 text=once(text,'if policy not in {"full_budget", "native_selector_stop"}:','if policy not in {"full_budget", "native_selector_stop", "hgt_native_selector_stop"}:')
 text=once(text,'    if policy == "native_selector_stop":',
 '''    if policy == "hgt_native_selector_stop":
        from hgt_stopping import validate_hgt_run
        validate_hgt_run(spec["adapter_spec"]["run"])
        if (spec["budget_unit"] != "optimizer_updates" or spec["budget_steps"] != 300
                or spec["validation_interval"] != 1 or spec["validate_initial"] or spec["prefixes"]):
            raise ValueError("HGT native stopping must keep300 cap/p30/no0/every-update SELECT")
    if policy == "native_selector_stop":''')
 old='spec.get("completion_policy", "full_budget") == "native_selector_stop"'
 if text.count(old)!=5:raise ValueError('Exactly five existing native stopping dispatch sites required')
 text=text.replace(old,'spec.get("completion_policy", "full_budget") in {"native_selector_stop", "hgt_native_selector_stop"}')
 text=once(text,'result["completion_reason"] = "native_patience50"','result["completion_reason"] = "native_patience30" if spec.get("completion_policy") == "hgt_native_selector_stop" else "native_patience50"')
 text=once(text,'result.setdefault("completion_reason", "native_patience50" if last_decision["stop_requested"] else "native_cap100000")',
 '''result.setdefault("completion_reason", ("native_patience30" if last_decision["stop_requested"] else "native_cap300")
                                  if spec.get("completion_policy") == "hgt_native_selector_stop"
                                  else ("native_patience50" if last_decision["stop_requested"] else "native_cap100000"))''')
 return text

def amend_spec(text):
 text=once(text,'{"full_budget", "minimum_then_patience", "native_selector_stop"}','{"full_budget", "minimum_then_patience", "native_selector_stop", "hgt_native_selector_stop"}')
 text=once(text,'            if budget["policy"] == "native_selector_stop":',
 '''            if budget["policy"] == "hgt_native_selector_stop":
                from hgt_stopping import validate_hgt_run
                validate_hgt_run(run)
            if budget["policy"] == "native_selector_stop":''')
 return text

def amend_run(text):
 text=once(text,'    require_completion(run)',
 '''    if run["budget"]["policy"] == "hgt_native_selector_stop":
        from hgt_stopping import validate_hgt_run
        validate_hgt_run(run)
    else:
        require_completion(run)''')
 text=once(text,'        if run["budget"]["policy"] == "native_selector_stop":\n            engine_spec["completion_policy"] = "native_selector_stop"',
 '''        if run["budget"]["policy"] in {"native_selector_stop", "hgt_native_selector_stop"}:
            engine_spec["completion_policy"] = run["budget"]["policy"]''')
 return text

def amend_compiler(text):
 return once(text,"    if budget['policy'] == 'native_selector_stop':",
 "    if budget['policy'] == 'hgt_native_selector_stop':\n        from hgt_stopping import validate_hgt_run\n        return validate_hgt_run(run)\n    if budget['policy'] == 'native_selector_stop':")

if __name__=='__main__':
 methods={'engine.py':amend_engine,'spec.py':amend_spec,'run_study.py':amend_run,'compile_master_plan.py':amend_compiler};rows=[]
 for name,fn in methods.items():
  old=(OUT/'controller_v16_base'/name).read_text();new=fn(old);ast.parse(new);(OUT/'runtime'/name).write_text(new)
  (OUT/'preparation'/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v16/'+name,tofile='HGT300/'+name)))
  before={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(old).body};after={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in ast.parse(new).body}
  rows.append({'file':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'successor_sha256':hashlib.sha256(new.encode()).hexdigest(),'changed_AST_functions':[n for n,v in before.items() if v!=after[n]]})
 (OUT/'ROOT_V16_COMPOSITION.json').write_text(json.dumps({'schema':'native-HGT-v16-narrow-composition-v1','files':rows,'new_helper':'runtime/hgt_stopping.py','base':'master_controller_source_20261012_v16/runtime','global_root_edited':False,'GO':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps(rows))

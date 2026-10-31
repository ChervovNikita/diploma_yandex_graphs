"""Composable anchor-checked function amendments against the exact frozen v13.

Root can apply these independently to its composed text. Untargeted functions
and all source-specific typing retain the input bytes. Full-budget semantics
remain the default for prior packets. No global controller is edited here.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]

def once(text,old,new):
 if text.count(old)!=1:raise ValueError('Amendment anchor differs: '+old[:100])
 return text.replace(old,new)

def amend_engine(text):
 next_function='_training_counts' if 'def _training_counts(' in text else '_metrics'
 text=once(text,'    return spec\n\n\ndef '+next_function,
 '''    policy = spec.get("completion_policy", "full_budget")
    if policy not in {"full_budget", "native_selector_stop"}:
        raise ValueError("Unsupported trajectory completion policy")
    if policy == "native_selector_stop":
        from native_stopping import validate_native_run
        validate_native_run(spec["adapter_spec"]["run"])
        if (spec["budget_unit"] != "optimizer_updates" or spec["budget_steps"] != 100000
                or spec["validation_interval"] != 1 or spec["validate_initial"] or spec["prefixes"]):
            raise ValueError("Native source stopping has no checkpoint0, virtual prefixes or skipped validation")
    return spec


def '''+next_function)
 text=once(text,'            for step in range(start_step + 1, spec["budget_steps"] + 1):',
 '''            native_already_stopped = (spec.get("completion_policy", "full_budget") == "native_selector_stop"
                                      and last_decision is not None and last_decision["stop_requested"])
            trajectory_end = start_step if native_already_stopped else spec["budget_steps"]
            for step in range(start_step + 1, trajectory_end + 1):''')
 text=once(text,'                        or any(p.get("closure_step") == step for p in result["prefixes"])):',
 '''                        or any(p.get("closure_step") == step for p in result["prefixes"])
                        or (spec.get("completion_policy", "full_budget") == "native_selector_stop"
                            and last_decision["stop_requested"])):''')
 text=once(text,'                resource("after_step")\n            phase = "checkpoint"',
 '''                resource("after_step")
                if (spec.get("completion_policy", "full_budget") == "native_selector_stop"
                        and last_decision["stop_requested"]):
                    result["completion_reason"] = "native_patience50"
                    break
            if spec.get("completion_policy", "full_budget") == "native_selector_stop":
                result.setdefault("completion_reason", "native_patience50" if last_decision["stop_requested"] else "native_cap100000")
                result["declared_optimizer_update_cap"] = spec["budget_steps"]
                result["actual_completed_optimizer_updates"] = result["completed_steps"]
            phase = "checkpoint"''')
 text=once(text,'result["trajectory_operation_counts"]["optimizer_updates"] != spec["budget_steps"]',
  'result["trajectory_operation_counts"]["optimizer_updates"] != (result["completed_steps"] if spec.get("completion_policy", "full_budget") == "native_selector_stop" else spec["budget_steps"])')
 text=once(text,'"Reported optimizer updates differ from the declared update budget"','"Reported optimizer updates differ from the declared trajectory completion"')
 return text

def amend_spec(text):
 text=once(text,'{"full_budget", "minimum_then_patience"}','{"full_budget", "minimum_then_patience", "native_selector_stop"}')
 text=once(text,'            if budget["policy"] == "minimum_then_patience":',
 '''            if budget["policy"] == "native_selector_stop":
                from native_stopping import validate_native_run
                validate_native_run(run)
            if budget["policy"] == "minimum_then_patience":''')
 return text

def amend_run(text):
 text=once(text,'    if run["budget"]["policy"] != "full_budget":\n        raise ValueError("The current engine implements full-budget trajectories")',
 '    from native_stopping import require_completion\n    require_completion(run)')
 text=once(text,'        def guard(phase, step, result):',
 '        if run["budget"]["policy"] == "native_selector_stop":\n            engine_spec["completion_policy"] = "native_selector_stop"\n        def guard(phase, step, result):')
 return text

def amend_compiler(text):
 text=once(text,"    if budget['policy'] != 'full_budget':\n        raise ValueError('Master only supports the registered full-budget controller')",
 "    if budget['policy'] == 'native_selector_stop':\n        from native_stopping import validate_native_run\n        return validate_native_run(run)\n    if budget['policy'] != 'full_budget':\n        raise ValueError('Master only supports registered typed completion policies')")
 text=once(text,'    bounded_updates = []\n', '    bounded_updates = []\n    native_stopping_updates = []\n')
 branch="if semantics.get('known_sehgnn_AMP_bounded_commit_fit_groups'):" if "if semantics.get('known_sehgnn_AMP_bounded_commit_fit_groups'):" in text else "if 'bounded_optimizer_step_calls_maximum' in semantics:"
 text=once(text,"                quantities['registered_' + run['budget']['unit']] = units\n                "+branch,
 """                if 'native_stopping_updates_maximum' in semantics:
                    from native_stopping import native_count_quantities
                    quantities.update(native_count_quantities(semantics))
                    native_stopping_updates.append(study['study_id']+'/'+run['run_id'])
                else:
                    quantities['registered_' + run['budget']['unit']] = units
                if 'native_stopping_updates_maximum' in semantics:
                    pass
                el"""+branch)
 text=once(text,"'model_optimizer_step_call_total_fully_determined': not unknown_updates and not bounded_updates,",
 "'model_optimizer_step_call_total_fully_determined': not unknown_updates and not bounded_updates and not native_stopping_updates,")
 text=once(text,"'model_optimizer_step_call_total': totals['known_registered_optimizer_step_calls'] if not unknown_updates and not bounded_updates else None,",
 "'model_optimizer_step_call_total': totals['known_registered_optimizer_step_calls'] if not unknown_updates and not bounded_updates and not native_stopping_updates else None,")
 text=once(text,"totals['known_registered_optimizer_step_calls']+totals['bounded_optimizer_step_calls_minimum'] if not unknown_updates else None,",
 "totals['known_registered_optimizer_step_calls']+totals['bounded_optimizer_step_calls_minimum']+totals['native_optimizer_updates_minimum_sum'] if not unknown_updates else None,")
 text=once(text,"totals['known_registered_optimizer_step_calls']+totals['bounded_optimizer_step_calls_maximum'] if not unknown_updates else None,",
 "totals['known_registered_optimizer_step_calls']+totals['bounded_optimizer_step_calls_maximum']+totals['native_optimizer_update_cap_sum'] if not unknown_updates else None,")
 anchor="'finite_bounded_optimizer_groups':len(bounded_updates)," if "'finite_bounded_optimizer_groups':len(bounded_updates)," in text else "'finite_bounded_trial_groups':len(bounded_updates),"
 text=once(text,anchor,"'native_stopping_fit_groups':len(native_stopping_updates),'native_stopping_run_keys':native_stopping_updates,\n        "+anchor)
 return text

def main():
 methods={'engine.py':amend_engine,'spec.py':amend_spec,'run_study.py':amend_run,'compile_master_plan.py':amend_compiler};records=[]
 for name,fn in methods.items():
  path=OUT/'controller_base'/name;old=path.read_text();new=fn(old);ast.parse(new)
  (OUT/'runtime'/name).write_text(new)
  patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v13/'+name,tofile='native_stop/'+name))
  (OUT/'preparation'/(name+'.patch')).write_text(patch)
  oldtree=ast.parse(old);newtree=ast.parse(new);oldnodes={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in oldtree.body};newnodes={getattr(n,'name',None):ast.dump(n,include_attributes=False) for n in newtree.body}
  changed=[n for n in oldnodes if oldnodes[n]!=newnodes[n]]
  records.append({'file':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'successor_sha256':hashlib.sha256(new.encode()).hexdigest(),'changed_top_level_nodes':changed})
 (OUT/'CONTROLLER_TYPE_AMENDMENT.json').write_text(json.dumps({'schema':'native-meansage-v13-stopping-cost-amendment-v1','base':'master_controller_source_20261012_v13/runtime','files':records,
  'composition':'apply anchor-checked functions from preparation/amend_controller.py to root-composed text; copy runtime/native_stopping.py',
  'native_costs':'bounded51..100000 updates per fit; executed counts unknown until source stopping completes; caps are never exact executed updates',
  'existing_full_budget_default_unchanged':True,'global_controller_edited':False,'GO':False},indent=2,sort_keys=True)+'\n')
 print(json.dumps(records))

if __name__=='__main__':main()

"""Composable narrow TABM stopping and OIPG type amendments; no root edits."""
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
    text=text.replace('{"native_selector_stop", "hgt_native_selector_stop"}','{"native_selector_stop", "hgt_native_selector_stop", "tabm_native_selector_stop"}')
    text=once(text,'{"full_budget", "native_selector_stop", "hgt_native_selector_stop"}',
              '{"full_budget", "native_selector_stop", "hgt_native_selector_stop", "tabm_native_selector_stop"}')
    text=once(text,'    if policy == "hgt_native_selector_stop":',
        '''    if policy == "tabm_native_selector_stop":
        from tabm_stopping import validate_tabm_run
        validate_tabm_run(spec["adapter_spec"]["run"])
        run=spec["adapter_spec"]["run"]
        if (spec["budget_unit"] != run["budget"]["unit"] or spec["budget_steps"] != run["budget"]["maximum"]
                or spec["validation_interval"] != 1 or spec["validate_initial"] or spec["prefixes"]):
            raise ValueError("Literal TABM stopping/cadence and own coupled round budget required")
    if policy == "hgt_native_selector_stop":''')
    text=once(text,'                result["declared_optimizer_update_cap"] = spec["budget_steps"]\n                result["actual_completed_optimizer_updates"] = result["completed_steps"]',
        '''                owners=spec["adapter_spec"]["run"]["config"].get("source_optimizer_count",1) if spec.get("completion_policy")=="tabm_native_selector_stop" else 1
                result["declared_optimizer_update_cap"] = spec["budget_steps"]*owners
                result["actual_completed_optimizer_updates"] = result["trajectory_operation_counts"]["optimizer_updates"]
                if spec.get("completion_policy")=="tabm_native_selector_stop":
                    result["source_outer_rounds_completed"] = result["completed_steps"]
                    result["source_own_completion"] = "four_native_own_stopping_states" if owners==4 else "native_own_stopping_state"''')
    return text
def amend_spec(text):
    text=once(text,'{"full_budget", "minimum_then_patience", "native_selector_stop", "hgt_native_selector_stop"}',
              '{"full_budget", "minimum_then_patience", "native_selector_stop", "hgt_native_selector_stop", "tabm_native_selector_stop"}')
    text=once(text,'            if budget["policy"] == "hgt_native_selector_stop":',
              '            if budget["policy"] == "tabm_native_selector_stop":\n                from tabm_stopping import validate_tabm_run\n                validate_tabm_run(run)\n            if budget["policy"] == "hgt_native_selector_stop":')
    return text
def amend_run(text):
    text=once(text,'    if run["budget"]["policy"] == "hgt_native_selector_stop":',
              '    if run["budget"]["policy"] == "tabm_native_selector_stop":\n        from tabm_stopping import validate_tabm_run\n        validate_tabm_run(run)\n    elif run["budget"]["policy"] == "hgt_native_selector_stop":')
    text=once(text,'{"native_selector_stop", "hgt_native_selector_stop"}',
              '{"native_selector_stop", "hgt_native_selector_stop", "tabm_native_selector_stop"}')
    return text
def amend_compiler(text):
    text=once(text,"    if budget['policy'] == 'hgt_native_selector_stop':",
              "    if budget['policy'] == 'tabm_native_selector_stop':\n        from tabm_stopping import validate_tabm_run\n        return validate_tabm_run(run)\n    if budget['policy'] == 'hgt_native_selector_stop':")
    text=once(text,"    if 'sehgnn_arm' in config:\n        return sehgnn_semantics(run)",
              "    if 'oipg_policy' in config:\n        from oipg_typing import validate_oipg_run\n        return validate_oipg_run(run)\n    if 'sehgnn_arm' in config:\n        return sehgnn_semantics(run)")
    text=once(text,"                    quantities.update(native_count_quantities(semantics))",
              "                    quantities.update(native_count_quantities(semantics))\n                    if 'source_detached_prefix_forward_calls_upper_bound' in semantics:\n                        quantities['source_detached_prefix_forward_calls_upper_bound']=semantics['source_detached_prefix_forward_calls_upper_bound']")
    return text
def main():
    methods={'engine.py':amend_engine,'spec.py':amend_spec,'run_study.py':amend_run,'compile_master_plan.py':amend_compiler};records=[]
    for name,method in methods.items():
        old=(OUT/'controller_base'/name).read_text();new=method(old);ast.parse(new);(OUT/'runtime'/name).write_text(new)
        (OUT/'preparation'/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='rootv17/'+name,tofile='PhysicsOIPG/'+name)))
        records.append({'file':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'successor_sha256':hashlib.sha256(new.encode()).hexdigest()})
    (OUT/'CONTROLLER_TYPE_AMENDMENT.json').write_text(json.dumps({'base':'master_controller_source_20261012_v17/runtime','files':records,
        'composition':'apply anchor-checked preparation/amend_controller.py functions to newer root text; add tabm_stopping.py and oipg_typing.py',
        'global_root_controller_edited':False,'existing_HGT_FiLM_Rank1_AMP_retention_paths_preserved':True,'GO':False},indent=2)+'\n')
    print(json.dumps(records))
if __name__=='__main__':main()

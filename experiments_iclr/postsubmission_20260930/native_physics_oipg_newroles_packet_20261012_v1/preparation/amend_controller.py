"""Composable narrow TABM stopping and OIPG type amendments; no root edits."""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Amendment anchor differs: '+old[:100])
    return text.replace(old,new)
def add_policy_sets(text):
    def replace(match):
        old=match.group(0)
        if '"hgt_native_selector_stop"' not in old or '"tabm_native_selector_stop"' in old:return old
        return old[:-1]+', "tabm_native_selector_stop"}'
    return re.sub(r'\{[^{}\n]*"hgt_native_selector_stop"[^{}\n]*\}',replace,text)
def amend_engine(text):
    text=add_policy_sets(text)
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
    text=add_policy_sets(text)
    text=once(text,'            if budget["policy"] == "hgt_native_selector_stop":',
              '            if budget["policy"] == "tabm_native_selector_stop":\n                from tabm_stopping import validate_tabm_run\n                validate_tabm_run(run)\n            if budget["policy"] == "hgt_native_selector_stop":')
    return text
def amend_run(text):
    text=once(text,'    if run["budget"]["policy"] == "hgt_native_selector_stop":',
              '    if run["budget"]["policy"] == "tabm_native_selector_stop":\n        from tabm_stopping import validate_tabm_run\n        validate_tabm_run(run)\n    if run["budget"]["policy"] == "hgt_native_selector_stop":')
    text=add_policy_sets(text)
    return text
def amend_compiler(text):
    tree=ast.parse(text)
    declarations=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROLLER_FILES' for t in n.targets)]
    if len(declarations)!=1 or not isinstance(declarations[0].value,ast.Tuple):raise ValueError('One literal CONTROLLER_FILES tuple required')
    node=declarations[0].value;names=ast.literal_eval(node);missing=[v for v in ('tabm_stopping.py','oipg_typing.py') if v not in names]
    if missing:
        lines=text.splitlines(True);offset=sum(len(v) for v in lines[:node.end_lineno-1])+node.end_col_offset-1
        if text[offset]!=')':raise ValueError('Controller tuple boundary differs')
        text=text[:offset]+', '+', '.join(repr(v) for v in missing)+text[offset:]
    text=once(text,"    if budget['policy'] == 'hgt_native_selector_stop':",
              "    if budget['policy'] == 'tabm_native_selector_stop':\n        from tabm_stopping import validate_tabm_run\n        return validate_tabm_run(run)\n    if budget['policy'] == 'hgt_native_selector_stop':")
    text=once(text,"    if 'sehgnn_arm' in config:\n        return sehgnn_semantics(run)",
              "    if 'oipg_policy' in config:\n        from oipg_typing import validate_oipg_run\n        return validate_oipg_run(run)\n    if 'sehgnn_arm' in config:\n        return sehgnn_semantics(run)")
    text=once(text,"                    quantities.update(native_count_quantities(semantics))",
              "                    quantities.update(native_count_quantities(semantics))\n                    if 'source_detached_prefix_forward_calls_upper_bound' in semantics:\n                        quantities['source_detached_prefix_forward_calls_upper_bound']=semantics['source_detached_prefix_forward_calls_upper_bound']")
    text=once(text, "                count.update(quantities)",
        "                if 'oipg_policy' in run['config']:\n                    quantities.update({k:semantics[k] for k in ('source_complete_body_constructors','source_discarded_body_constructors','source_optimizer_constructors')})\n                    quantities['registered_OIPG_source_old_state_gradient_collections']=semantics['known_source_old_state_gradient_collections']\n                count.update(quantities)")
    return text
def main():
    methods={'engine.py':amend_engine,'spec.py':amend_spec,'run_study.py':amend_run,'compile_master_plan.py':amend_compiler};records=[]
    for name,method in methods.items():
        old=(OUT/'controller_base'/name).read_text();new=method(old);ast.parse(new);(OUT/'runtime'/name).write_text(new)
        (OUT/'preparation'/(name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='rootv17/'+name,tofile='PhysicsOIPG/'+name)))
        records.append({'file':name,'base_sha256':hashlib.sha256(old.encode()).hexdigest(),'successor_sha256':hashlib.sha256(new.encode()).hexdigest()})
    (OUT/'CONTROLLER_TYPE_AMENDMENT.json').write_text(json.dumps({'base':'master_controller_source_20261012_v17/runtime','files':records,
        'composition':'apply anchor-checked preparation/amend_controller.py functions to newer root text; copy tabm_stopping.py and oipg_typing.py beside compiler; policy sets preserve BSNN and original HGT if anchor, either order',
        'inherited_numerical_source_proof':'SYNTHETIC_CPU_PROOF.json; no model or training adapter numerical changes',
        'global_root_controller_edited':False,'existing_HGT_FiLM_Rank1_AMP_retention_paths_preserved':True,'GO':False},indent=2)+'\n')
    print(json.dumps(records))
if __name__=='__main__':main()

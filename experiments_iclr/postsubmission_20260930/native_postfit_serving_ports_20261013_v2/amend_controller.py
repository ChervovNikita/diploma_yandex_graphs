"""Narrow additions against the actual v22 controller, preserving prior branches."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
def amend_native(text):
 anchor=" c=run['config']\n";new=" c=run['config']\n if 'postfit_source_contract' in c:\n  from postfit_semantics import validate_postfit_run\n  return validate_postfit_run(run)\n"
 return text if "'postfit_source_contract' in c" in text else text.replace(anchor,new,1)
def amend_compiler(text):
 if "'postfit_source_contract' in config" not in text:
  anchor="    if budget['policy'] == 'tabm_native_selector_stop':";assert anchor in text;text=text.replace(anchor,"    if 'postfit_source_contract' in config:\n        from postfit_semantics import validate_postfit_run\n        return validate_postfit_run(run)\n"+anchor,1)
  anchor="        fits = row['config'].get('fit_parents')";assert anchor in text;text=text.replace(anchor,anchor+"\n        if row['config'].get('postfit_fold_bank') or row['config'].get('postfit_affine_I4_bank'):\n            from postfit_semantics import validate_special_bank\n            banks.append(validate_special_bank(row,runs))\n            continue",1)
  anchor="            family = staged_family(run['config'])";assert anchor in text;text=text.replace(anchor,"            if run['config'].get('lifecycle',{}).get('schema')=='paid-frozen-native-feature-cohort-v1':\n                from postfit_semantics import require_cohort_parent,validate_donor_binding\n                for identifier in paid:\n                    require_cohort_parent(records[row['study']+'/'+identifier]['run'],run)\n                validate_donor_binding(run,{i:records[row['study']+'/'+i]['run'] for i in paid})\n"+anchor,1)
  anchor="    excluded = {'initialization_seed', 'dropout_seed', 'shuffle_seed', 'resolved_selection',";assert anchor in text
  # Cohort IDs do not vary per body. Source member0/body identity is encoded in existing init seed.
  anchor="                elif semantics['known_optimizer_step_calls'] is None:";assert anchor in text
  text=text.replace(anchor,"                elif 'known_postfit_LBFGS_trajectories' in semantics:\n                    quantities['known_registered_optimizer_step_calls']=1\n                    quantities['postfit_LBFGS_trajectories']=1\n                    quantities['postfit_LBFGS_internal_iteration_cap']=1000\n                    quantities['postfit_LBFGS_closure_cap']=50000\n"+anchor,1)
  anchor="            elif paid:\n                raise ValueError('Only an explicitly typed staged continuation lifecycle is admitted')";assert anchor in text
  text=text.replace(anchor,"            elif run['config'].get('lifecycle',{}).get('schema')=='paid-frozen-native-feature-cohort-v1':\n                pass\n"+anchor,1)
 if "'postfit_semantics.py'" not in text:
  anchor=next(line for line in text.splitlines(keepends=True) if line.startswith('CONTROLLER_FILES = '));text=text.replace(anchor,anchor+"CONTROLLER_FILES = CONTROLLER_FILES + ('postfit_semantics.py',)\n",1)
 return text
def amend_run_study(text):
 if "'paid-frozen-native-feature-cohort-v1'" not in text:
  anchor='    lifecycle = run["config"].get("lifecycle")';assert anchor in text
  text=text.replace(anchor,anchor+"\n    if isinstance(lifecycle,dict) and lifecycle.get('schema')=='paid-frozen-native-feature-cohort-v1':\n        from postfit_semantics import validate_lifecycle\n        return validate_lifecycle(run)",1)
  anchor='    identifiers = paid_parent_ids(run)';assert anchor in text
  text=text.replace(anchor,"    if run['config'].get('lifecycle',{}).get('schema')=='paid-frozen-native-feature-cohort-v1':\n        from postfit_semantics import paid_feature_parents\n        return paid_feature_parents(root,spec,run,registry,output_root)\n"+anchor,1)
 if "adapter.finalize_result(destination,result)" not in text:
  anchor='        from retention_lifecycle import close_fit';assert anchor in text
  text=text.replace(anchor,"        if hasattr(adapter,'finalize_result'):\n            result=adapter.finalize_result(destination,result)\n            from engine import _atomic_json\n            _atomic_json(destination/'result.json',result)\n"+anchor,1)
 return text
def install(runtime):
 root=Path(runtime);result={}
 for name,fun in [('native_stopping.py',amend_native),('compile_master_plan.py',amend_compiler),('run_study.py',amend_run_study)]:
  q=root/name;old=q.read_text();new=fun(old);assert old==new or 'postfit' in new;q.write_text(new);result[name]=hashlib.sha256(new.encode()).hexdigest()
 (root/'postfit_semantics.py').write_text((HERE/'runtime/postfit_semantics.py').read_text())
 return result
if __name__=='__main__':print(install(HERE/'runtime'))

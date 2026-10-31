"""Narrow additive typed-context200 native epoch/AMP compiler and engine patch."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
POLICY='typed_context_native_selector_stop'
def amend_compiler(text):
 if 'typed_context_source_contract' not in text:
  anchor="    if budget['policy'] == 'hgt_native_selector_stop':";assert anchor in text
  text=text.replace(anchor,"    if 'typed_context_source_contract' in config:\n        from typed_context_semantics import validate_typed_context_run\n        return validate_typed_context_run(run)\n"+anchor,1)
  anchor="            family = staged_family(run['config'])";assert anchor in text
  block="            if 'typed_context_source_contract' in run['config']:\n                from typed_context_semantics import validate_typed_context_run\n                validate_typed_context_run(run)\n                previous=run['config'].get('master_scaler_parent_fit_id')\n                if previous is not None:\n                    parent=records[row['study']+'/'+previous]['run']\n                    if (row['study']+'/'+previous not in barriers.get(key,[]) or parent['kind']!='model_fit' or parent['config'].get('typed_context_arm')!=run['config']['typed_context_arm'] or parent['seed']!=run['seed']-1 or parent['body']!=run['body']):\n                        raise ValueError('Exact preceding samecondition/body terminalmaster scaler barrier required')\n"
  text=text.replace(anchor,block+anchor,1)
  anchor="                elif semantics['known_optimizer_step_calls'] is None:";assert anchor in text
  text=text.replace(anchor,"                elif 'typed_context_epoch_maximum' in semantics:\n                    from typed_context_semantics import count_quantities\n                    quantities.update(count_quantities(semantics))\n                    quantities['bounded_optimizer_step_calls_minimum']=semantics['typed_context_AMP_commit_minimum']\n                    quantities['bounded_optimizer_step_calls_maximum']=semantics['typed_context_AMP_commit_maximum']\n                    bounded_updates.append(study['study_id']+'/'+run['run_id'])\n                    amp_bounded_updates.append(study['study_id']+'/'+run['run_id'])\n"+anchor,1)
 text=text.replace("            rng_excluded = set(excluded)","            rng_excluded = set(excluded)\n            if 'typed_context_source_contract' in parents[0]['config']:\n                rng_excluded.add('master_scaler_parent_fit_id')",1) if "rng_excluded.add('master_scaler_parent_fit_id')" not in text else text
 return text

def amend_engine(text):
 if POLICY in text:return text
 text=text.replace('policy not in {"full_budget", "native_selector_stop", "hgt_native_selector_stop"}', 'policy not in {"full_budget", "native_selector_stop", "hgt_native_selector_stop", "'+POLICY+'"}')
 anchor='    if policy == "hgt_native_selector_stop":'
 block="    if policy == '"+POLICY+"':\n        from typed_context_semantics import validate_typed_context_run\n        validate_typed_context_run(spec['adapter_spec']['run'])\n        if spec['budget_unit']!='epochs' or spec['budget_steps']!=200 or spec['validation_interval']!=1 or spec['validate_initial'] or spec['prefixes']:\n            raise ValueError('Exact typedcontext source200epochs/p50/no0/everyepoch required')\n"
 text=text.replace(anchor,block+anchor,1)
 anchor="    if policy is not None:\n        run=spec['adapter_spec'].get('run',{});c=run.get('config',{})"
 repl="    if policy is not None and policy.get('schema')=='typed-context-AMP-complete-cycle-v1':\n        from typed_context_semantics import validate_typed_context_run\n        validate_typed_context_run(spec['adapter_spec']['run'])\n        if policy!={'schema':'typed-context-AMP-complete-cycle-v1','optimizer_owner_attempts':1} or spec['budget_unit']!='epochs' or spec['budget_steps']!=200:\n            raise ValueError('Exact typedcontext native epoch AMP law required')\n    elif policy is not None:\n        run=spec['adapter_spec'].get('run',{});c=run.get('config',{})"
 assert anchor in text;text=text.replace(anchor,repl,1)
 text=text.replace('{"native_selector_stop", "hgt_native_selector_stop"}', '{"native_selector_stop", "hgt_native_selector_stop", "'+POLICY+'"}')
 anchor='                result["actual_completed_optimizer_updates"] = result["completed_steps"]'
 text=text.replace(anchor,anchor+"\n                if spec.get('completion_policy')=='"+POLICY+"':\n                    result['declared_source_epoch_cap']=result.pop('declared_optimizer_update_cap')\n                    result['actual_completed_source_epochs']=result.pop('actual_completed_optimizer_updates')",1)
 anchor="            result['final_training_checkpoint'] = checkpoint_receipt(output / 'final_training.pt', fit_id=spec['fit_id'], step=step)"
 assert anchor in text
 text=text.replace(anchor,anchor+"\n            if hasattr(adapter,'export_terminal_master_scaler'):\n                result['terminal_master_scaler']=adapter.export_terminal_master_scaler(step)",1)
 return text

def amend_run_study(text):
 if POLICY in text:return text
 anchor="            engine_spec['native_amp_cycle_policy']={'schema':'native-SeHGNN-AMP-complete-cycle-v1','optimizer_owner_attempts':1}"
 assert anchor in text;text=text.replace(anchor,anchor+"\n            if 'typed_context_source_contract' in run['config']:\n                engine_spec['native_amp_cycle_policy']={'schema':'typed-context-AMP-complete-cycle-v1','optimizer_owner_attempts':1}",1)
 text=text.replace('{"native_selector_stop", "hgt_native_selector_stop"}', '{"native_selector_stop", "hgt_native_selector_stop", "'+POLICY+'"}')
 return text

def amend_spec(text):
 if POLICY in text:return text
 text=text.replace('{"full_budget", "minimum_then_patience", "native_selector_stop", "hgt_native_selector_stop"}', '{"full_budget", "minimum_then_patience", "native_selector_stop", "hgt_native_selector_stop", "'+POLICY+'"}')
 anchor='            if budget["policy"] == "hgt_native_selector_stop":'
 assert anchor in text
 text=text.replace(anchor,"            if budget['policy']=='"+POLICY+"':\n                from typed_context_semantics import validate_typed_context_run\n                validate_typed_context_run(run)\n"+anchor,1)
 return text

def semantic_module():
 code=(HERE/'runtime/typed_context_semantics.py').read_text();tree=ast.parse((HERE/'runtime/adapters/typed_context.py').read_text());values={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if isinstance(t,ast.Name) and t.id in ('CONTRACT','SEEDS','CONDITIONS'):values[t.id]=ast.literal_eval(n.value)
 return code.replace('from adapters.typed_context import CONTRACT,SEEDS,CONDITIONS','\n'.join(k+'='+repr(values[k]) for k in ('CONTRACT','SEEDS','CONDITIONS')))
def install(runtime_dir):
 root=Path(runtime_dir);changed={}
 for name,fun in [('compile_master_plan.py',amend_compiler),('engine.py',amend_engine),('run_study.py',amend_run_study),('spec.py',amend_spec)]:
  q=root/name;old=q.read_text();new=fun(old);q.write_text(new);changed[name]=hashlib.sha256(new.encode()).hexdigest()
 (root/'typed_context_semantics.py').write_text(semantic_module());return changed

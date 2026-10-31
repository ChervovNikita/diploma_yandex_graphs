"""Execute-only completion dispatch correction after all policy amendments."""
import ast
def amend(text):
 old='    else:\n        require_completion(run)\n    registry = Registry(registry_path)'
 new='    if run["budget"]["policy"] not in {"hgt_native_selector_stop", "tabm_native_selector_stop", "bsnn_native_selector_stop"}:\n        require_completion(run)\n    registry = Registry(registry_path)'
 if text.count(old)!=1:raise ValueError('Exact execute fallback/registry boundary required')
 for policy in ('hgt_native_selector_stop','tabm_native_selector_stop','bsnn_native_selector_stop'):
  if 'if run["budget"]["policy"] == "'+policy+'":' not in text:raise ValueError('Separate literal validator anchor missing: '+policy)
 result=text.replace(old,new);ast.parse(result)
 before=ast.parse(text);after=ast.parse(result)
 def functions(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a,b=functions(before),functions(after)
 if [name for name in a if a[name]!=b[name]]!=['execute']:raise ValueError('Only execute may change')
 return result

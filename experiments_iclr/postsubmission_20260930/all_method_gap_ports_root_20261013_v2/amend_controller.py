"""Narrow additive configured-relation and complete-hop controller support."""
import ast
from pathlib import Path

POLICY='hop_complete_native_selector_stop'

def extend_sets(text):
    lines=text.splitlines(True);edits=[]
    for n in ast.walk(ast.parse(text)):
        if not isinstance(n,ast.Set):continue
        values={v.value for v in n.elts if isinstance(v,ast.Constant) and isinstance(v.value,str)}
        if 'hgt_native_selector_stop' in values and POLICY not in values:
            assert n.lineno==n.end_lineno
            edits.append((n.lineno-1,n.end_col_offset-1))
    for row,col in sorted(edits,reverse=True):lines[row]=lines[row][:col]+', '+repr(POLICY)+lines[row][col:]
    return ''.join(lines)

def insert_before(path,anchor,addition):
    text=path.read_text();assert text.count(anchor)==1
    path.write_text(text.replace(anchor,addition+anchor,1))

def install(runtime):
    runtime=Path(runtime);here=Path(__file__).resolve().parent
    for name in ('relation_semantics.py','hop_complete_semantics.py'):
        assert not (runtime/name).exists();(runtime/name).write_bytes((here/'runtime'/name).read_bytes())
    for name in ('engine.py','spec.py','run_study.py'):
        p=runtime/name;p.write_text(extend_sets(p.read_text()))
    insert_before(runtime/'compile_master_plan.py',"    if budget['policy'] == 'tabm_native_selector_stop':\n",
        "    if 'configured_relation_source_contract' in config:\n"
        "        from relation_semantics import validate_relation_run\n"
        "        return validate_relation_run(run)\n"
        "    if budget['policy']=='hop_complete_native_selector_stop':\n"
        "        from hop_complete_semantics import validate_hop_run\n"
        "        return validate_hop_run(run)\n")
    p=runtime/'compile_master_plan.py';text=p.read_text();anchor="('known_sehgnn_', 'known_response_', 'known_label_corrector_')";assert text.count(anchor)==1
    p.write_text(text.replace(anchor,"('known_sehgnn_', 'known_response_', 'known_label_corrector_', 'known_relation_')",1))
    insert_before(runtime/'spec.py',"            if budget['policy']=='typed_context_native_selector_stop':\n",
        "            if 'configured_relation_source_contract' in run['config']:\n"
        "                from relation_semantics import validate_relation_run\n"
        "                validate_relation_run(run)\n"
        "            if budget['policy']=='hop_complete_native_selector_stop':\n"
        "                from hop_complete_semantics import validate_hop_run\n"
        "                validate_hop_run(run)\n")
    insert_before(runtime/'run_study.py',"    from native_stopping import require_completion\n",
        "    if 'configured_relation_source_contract' in run['config']:\n"
        "        from relation_semantics import validate_relation_run\n"
        "        validate_relation_run(run)\n"
        "    if run['budget']['policy']=='hop_complete_native_selector_stop':\n"
        "        from hop_complete_semantics import validate_hop_run\n"
        "        validate_hop_run(run)\n")
    insert_before(runtime/'engine.py',"    if policy == \"tabm_native_selector_stop\":\n",
        "    if policy=='hop_complete_native_selector_stop':\n"
        "        from hop_complete_semantics import validate_hop_run\n"
        "        validate_hop_run(spec['adapter_spec']['run'])\n"
        "        if spec['budget_unit']!='optimizer_updates' or spec['budget_steps']!=2000 or spec['validation_interval']!=1 or spec['validate_initial'] or spec['prefixes']:\n"
        "            raise ValueError('Complete source hop2000/p250/no0/everyupdate required')\n")
    # run_study dispatch validates the new policy; retain all previous policies.
    p=runtime/'native_stopping.py';text=p.read_text();anchor=" elif run['budget']['policy']!='full_budget':raise ValueError('Unsupported completion policy')\n";assert text.count(anchor)==1
    p.write_text(text.replace(anchor," elif run['budget']['policy']=='hop_complete_native_selector_stop':\n  from hop_complete_semantics import validate_hop_run\n  validate_hop_run(run)\n"+anchor,1))
    p=runtime/'compile_master_plan.py';text=p.read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONTROLLER_FILES' for t in n.targets))
    names=list(ast.literal_eval(node.value));names.extend(n for n in ('relation_semantics.py','hop_complete_semantics.py') if n not in names)
    lines=text.splitlines(True);lines[node.lineno-1:node.end_lineno]=['CONTROLLER_FILES = '+repr(tuple(names))+'\n'];p.write_text(''.join(lines))
    for p in runtime.glob('*.py'):compile(p.read_text(),str(p),'exec')

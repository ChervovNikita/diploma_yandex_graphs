"""Source/metadata checks only; no Torch/SciPy/original-array execution."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    packet = Path(__file__).resolve().parent; parent = packet.parent
    provenance = json.loads((packet/'PROVENANCE.json').read_text()); checks = []
    def check(name,value):
        assert value,name; checks.append(name)
    for row in provenance['inputs']:
        raw = (parent/row['path']).read_bytes()
        check('bound_'+row['path'],len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'])
    raw = (parent/provenance['HGEN_model_source']).read_text()
    try:
        ast.parse(raw); raise AssertionError('Original broken source unexpectedly parses')
    except SyntaxError as error:
        check('private_author_TabError_preserved',type(error).__name__=='TabError' and error.lineno==355)
    prefix = ''.join(raw.splitlines(keepends=True)[:354]); tree = ast.parse(prefix)
    check('mechanical_orphan_tail_repair_parses_unchanged_classes',set(n.name for n in tree.body if isinstance(n,ast.ClassDef))=={'GCN','GAT','GraphSAGE','AttentionH','MultiGCN','GCN_embed'})
    paired = json.loads((packet/'PAIRED_HGB_PROPOSAL.json').read_text())
    freeze = json.loads((parent/paired['current_native_freeze']['path']).read_text())
    for key in ('dataset','scope','archive','development_labels','seeds','splits','members','member_sha256','source_label_member_sha256','expected_node_counts'):
        check('exact_current_paired_binding_'+key,paired[key]==freeze[key])
    check('all_five_fixed_seeds',paired['seeds']==[131,137,139,149,151])
    check('HGEN_not_adopted_by_native15_adoption',paired['HGEN_execution_adopted'] is False and paired['root_native15_adoption_is_not_HGEN_adoption'] is True)
    check('no_training_driver_supplied',not (packet/'train_hgen.py').exists() and paired['training_or_execution_driver_supplied'] is False)
    model = ast.parse((packet/'hgen_adapter.py').read_text())
    profile = next(n for n in model.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROFILES' for t in n.targets))
    check('only_code_literal_profile_settled',ast.literal_eval(profile.value)==('code_literal',))
    objective = next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='objective')
    check('only_source_default_lambda0_exposed',ast.literal_eval(objective.args.defaults[-1])==0.)
    parameter_count = 9*(334*64+64+2*(64*64+64)+64*64+64)+3*((64*8+8)+3*(64*8+8)+(3*8)*3+3)+3*(64*4+4)
    check('derived_parameter_count_includes_unused_source_maps',parameter_count==provenance['derived_parameter_counts']['code_defaults_HGB334_classes4']==312525)
    check('unused_parameter_count_honest',9*(64*64+64)+3*(64*8+8)==provenance['derived_parameter_counts']['unused']==39000)
    for path in packet.glob('*.py'):
        ast.parse(path.read_text(),str(path)); checks.append('own_syntax_'+path.name)
    print(json.dumps(dict(status='PASS',checks=checks,Torch_SciPy_original_arrays_labels_remote_GPU_or_training_execution=False)))


if __name__=='__main__': main()

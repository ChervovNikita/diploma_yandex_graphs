"""Three existing finite jobs retargeted at real portable source; metadata only."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable
import sys

HERE=Path(__file__).resolve().parent
E=HERE.parents[1]
LINK=E/'qualification/portablev3_link_queued_lifecycle_v1'
MOL=E/'qualification/portablev3_molhiv_gin_vn_resource_v1'


def main():
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    old_link=E/'qualification/runtime_link_v9_combined_lifecycle_v1';old_mol=E/'qualification/molhiv_gin_vn_cuda_resource_v1'
    for family in ('NCN','BUDDY'):
        assert (LINK/('JOB_'+family+'_CUDA.json')).read_bytes()==(old_link/('JOB_'+family+'_CUDA.json')).read_bytes()
    assert (LINK/'CASES.json').read_bytes()==(old_link/'CASES.json').read_bytes()
    assert (MOL/'JOB.json').read_bytes()==(old_mol/'JOB.json').read_bytes()
    old=json.loads((old_mol/'INPUTS.json').read_text());new=json.loads((MOL/'INPUTS.json').read_text())
    for name in ['bounds','native_session_config','resource_envelope','required_absent_paths','non_dominance_limits']:
        assert old[name]==new[name],name
    free={}
    for path in list(LINK.glob('*.py'))+list(MOL.glob('*.py'))+[HERE/'check_source.py']:
        text=path.read_text();tree=ast.parse(text,filename=str(path));table=symtable.symtable(text,str(path),'exec')
        names={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
        allowed=names|set(dir(builtins))|{'__file__','__name__','__package__'};missing=[]
        def visit(scope):
            for symbol in scope.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in allowed: missing.append(symbol.get_name())
            for child in scope.get_children():visit(child)
        visit(table);assert not missing,(path,missing);free[str(path.relative_to(E))]=missing
        for node in tree.body:
            if isinstance(node,ast.Import):assert all(not n.name.startswith(('torch','numpy','scipy','ogb')) for n in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('torch','numpy','scipy','ogb'))
    for folder,target in [(LINK,'portable_dispatcher_v3/runtime_link'),(MOL,'portable_dispatcher_v3/runtime_node_molecular')]:
        text=(folder/'qualify.py').read_text();tree=ast.parse(text)
        imports=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Import) and any(i.name=='torch' for i in n.names)]
        guards={n.func.attr:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='common' and n.func.attr in ('require_allocation','bound_project_root','validate_host_job')}
        assert len(imports)==1 and set(guards)=={'require_allocation','bound_project_root','validate_host_job'}
        assert all(line<imports[0] for line in guards.values())
        assert target in text
    portable=E/'portable_dispatcher_v3'
    manifest=json.loads((portable/'MANIFEST.json').read_text())
    for row in manifest['files']:
        blob=(portable/row['path']).read_bytes();assert len(blob)==row['bytes'] and hashlib.sha256(blob).hexdigest()==row['sha256']
    cases=json.loads((LINK/'CASES.json').read_text())['cases']
    counts={'NCN':cases['NCN']['limits']['optimizer_attempts'],'BUDDY':cases['BUDDY']['limits']['optimizer_attempts'],'MolHIV_GIN_VN':new['bounds']['optimizer_transitions']}
    assert counts=={'NCN':34,'BUDDY':3170,'MolHIV_GIN_VN':1029}
    result={'schema':'three-queued-portablev3-retarget-source-checks-v1','status':'PASS',
        'case_count':3,'original_JOB_bytes_unchanged':True,'native_recipe_method_seed_policy_and_limits_unchanged':True,
        'portable_host_root_and_descriptor_guards_before_Torch_import':True,'runtime_module_source_parent_checked':True,
        'optimizer_maps_already_queued':counts,'total_already_queued_optimizer_maps':sum(counts.values()),
        'additional_fit_or_method_grid':False,'new_portable_generation_or_checkpoint_binding_rewrite':False,
        'unbound_global_names':free,'native_or_model_data_checkpoint_TEST_execution':False,
        'actual_root_resource_admission_required':True}
    for folder in [HERE,LINK,MOL]:(folder/'SOURCE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Compose only the Bayesian stopping law onto a newer current controller.

Preserves source families added after rootv17. No scientific job/GO is created.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ALLOWED={'engine.py':{'_validate_spec','run_trajectory'},'spec.py':{'validate'},
         'run_study.py':{'execute'},'compile_master_plan.py':{'fit_semantics'}}


def add_once(source, anchor, clause):
    if source.count(anchor)!=1:
        raise ValueError('Expected exactly one current-source amendment anchor')
    return source.replace(anchor,clause+anchor,1)


def amend(name, source):
    if 'bsnn_native_selector_stop' in source:
        raise ValueError('Use an unamended current base; do not partially compose twice')
    if name in ('engine.py','spec.py','run_study.py'):
        anchor='"native_selector_stop", "hgt_native_selector_stop"'
        if anchor not in source:
            raise ValueError('Current source completion sets need independent review')
        source=source.replace(anchor,anchor+', "bsnn_native_selector_stop"')
    if name=='engine.py':
        source=add_once(source,'    if policy == "hgt_native_selector_stop":\n',
            '    if policy == "bsnn_native_selector_stop":\n'
            '        from bsnn_stopping import validate_bsnn_run\n'
            '        validate_bsnn_run(spec["adapter_spec"]["run"])\n'
            '        if (spec["budget_unit"] != "optimizer_updates" or spec["budget_steps"] != 500\n'
            '                or spec["validation_interval"] != 1 or spec["validate_initial"] or spec["prefixes"]):\n'
            '            raise ValueError("Bayesian BSNN keeps cap500/p200/no0/full four posterior SELECT draws")\n')
    elif name=='spec.py':
        source=add_once(source,'            if budget["policy"] == "hgt_native_selector_stop":\n',
            '            if budget["policy"] == "bsnn_native_selector_stop":\n'
            '                from bsnn_stopping import validate_bsnn_run\n'
            '                validate_bsnn_run(run)\n')
    elif name=='run_study.py':
        source=add_once(source,'    if run["budget"]["policy"] == "hgt_native_selector_stop":\n',
            '    if run["budget"]["policy"] == "bsnn_native_selector_stop":\n'
            '        from bsnn_stopping import validate_bsnn_run\n'
            '        validate_bsnn_run(run)\n')
    else:
        source=add_once(source,"    if budget['policy'] == 'hgt_native_selector_stop':\n",
            "    if budget['policy'] == 'bsnn_native_selector_stop':\n"
            "        from bsnn_stopping import validate_bsnn_run\n"
            "        return validate_bsnn_run(run)\n")
    ast.parse(source)
    return source


def definitions(source):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def compose(base, destination, stopping_module):
    base=Path(base).resolve();destination=Path(destination).resolve();stopping_module=Path(stopping_module).resolve()
    if destination.exists() or destination.is_relative_to(base):
        raise ValueError('Fresh distinct current-source successor required')
    shutil.copytree(base,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    rows={};patch=[]
    for name,allowed in ALLOWED.items():
        original=(base/name).read_text();revised=amend(name,original)
        before,after=definitions(original),definitions(revised)
        changed={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
        if changed!=allowed:
            raise ValueError('Amendment changed unexpected current-source definitions')
        (destination/name).chmod(0o644)
        (destination/name).write_text(revised)
        rows[name]={'base_sha256':hashlib.sha256(original.encode()).hexdigest(),
                    'successor_sha256':hashlib.sha256(revised.encode()).hexdigest(),
                    'changed_AST_definitions':sorted(changed)}
        patch.extend(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),
                     fromfile='current/'+name,tofile='successor/'+name))
    shutil.copyfile(stopping_module,destination/'bsnn_stopping.py')
    others=[p for p in base.rglob('*.py') if '__pycache__' not in p.parts and p.name not in ALLOWED]
    if not all(p.read_bytes()==(destination/p.relative_to(base)).read_bytes() for p in others):
        raise ValueError('Another current source family/retention file changed')
    return {'schema':'Bayesian-BSNN-cumulative-controller-composition-v1','current_base':str(base),
            'destination':str(destination),'changed_core_definitions':rows,'other_current_Python_files_preserved':len(others),
            'stopping_sha256':hashlib.sha256(stopping_module.read_bytes()).hexdigest(),
            'source_family_additions_after_v17_preserved':True,'patch':''.join(patch),'scientific_fits':0,'GO':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True)
    parser.add_argument('--destination',required=True);parser.add_argument('--stopping-module',required=True)
    args=parser.parse_args();result=compose(args.base,args.destination,args.stopping_module)
    path=Path(args.destination).parent/'BSNN_COMPOSITION.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='patch'}))

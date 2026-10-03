"""AST/hash/config verification only: no Torch/data/tensor/training imports."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import shlex
import sys
sys.dont_write_bytecode = True
import common as c


def check():
    source=c.verify_sources();design=c.read(c.PACKET/'DESIGN.json')
    c.require(design['cases']==[list(x) for x in c.EXPECTED_CASES] and set(design['recipes'])==set(c.RECIPES),
              'Two recipes and all three original blocks required')
    tree=ast.parse((c.AUTHOR/'training.py').read_text())
    defaults={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args:
            name=ast.literal_eval(node.args[0]).removeprefix('--')
            for keyword in node.keywords:
                if keyword.arg=='default':defaults[name]=ast.literal_eval(keyword.value)
    fields=('hidden','d_ffn','K','nlayer','n_head','q','multi','dropout','dprate','base','lr','weight_decay','attn_lr','attn_wd')
    c.require(design['recipes']['source_defaults']=={k:defaults[k] for k in fields},'Source-default recipe changed')
    commands=[]
    for line in (c.AUTHOR/'node_classification.sh').read_text().splitlines():
        if line.strip().startswith('python '):
            args=shlex.split(line);options={args[i][2:]:args[i+1] for i in range(len(args)-1) if args[i].startswith('--')}
            if options.get('dataset')=='roman-empire' and options.get('base')=='mono':commands.append(options)
    c.require(len(commands)==1,'Exact pinned Roman mono recipe missing')
    roman={k:(commands[0][k] if k=='base' else type(defaults[k])(commands[0][k])) for k in fields}
    c.require(design['recipes']['roman_mono']==roman,'Pinned Roman mono recipe changed')
    c.require(defaults['epochs']==2000 and defaults['early_stopping']==250 and
              design['max_epochs']==defaults['epochs'] and design['patience']==defaults['early_stopping'] and
              design['initial_best_validation_accuracy']==0 and design['selector']=='strict_validation_accuracy_earliest_tie',
              'Source epoch/patience/selector initialization changed')
    modules=[]
    for p in sorted(c.PACKET.glob('*.py')):
        value=p.read_text();ast.parse(value,filename=str(p));compile(value,str(p),'exec')
        modules.append(p.name)
    # Import the complete executable wiring with stdlib only. Main functions do
    # not run, and all numerical imports stay inside explicitly admitted calls.
    for name in ('common','byte_identity','native_training','capture_runtime','prepare_data','prepare_resource','train_study','evaluate_study'):
        importlib.import_module(name)
    c.require(not any(name=='torch' or name.startswith(('torch.','numpy','scipy','torch_geometric')) for name in sys.modules),
              'Source-only verification unexpectedly imported numerical libraries')
    c.require(design['scientific_competence_gate']['status']=='unresolved' and
              design['TEST_label_use_or_scoring_authorized'] is False and design['both_recipes_retained'] is True,
              'Scientific/TEST/family boundary changed')
    templates=sorted(c.PACKET.glob('*_RELEASE_TEMPLATE.json'))
    c.require(len(templates)==4 and all(c.read(p)['execution_authorized'] is False and
              c.read(p)['root_observed_source_review'] is False and c.read(p)['test_labels_authorized'] is False
              and c.read(p)['packet_manifest']['path']==c.PACKET.name+'/MANIFEST.json'
              for p in templates), 'Disabled separate root release templates required')
    predecessor=c.PHASE/'amazon_ratings_native_warm_study_preparation_20261003_v1'
    old_design=c.read(predecessor/'DESIGN.json')
    c.require({k:v for k,v in design.items() if k!='checkpoint_scope'} ==
              {k:v for k,v in old_design.items() if k!='checkpoint_scope'},
              'Narrow byte/state-scope successor cannot change scientific design fields')
    return {'schema':'amazon_native_warm_source_verification_v2','status':'passed','UTC':c.utc(),'packet_manifest':source,
            'stdlib_verification_interpreter':{'path':sys.executable,'version':sys.version},
            'prepared_Python_ASTs':modules,'both_exact_source_recipes_verified':True,
            'stdlib_executable_import_wiring_passed':True,'native8_metadata_binding_passed':True,
            'numerical_imports_or_calls':False,'data_labels_checkpoints_or_outcomes_opened':False,
            'resource_training_evaluation_or_cuda_passage_claimed':False,'automatic_job_dispatch':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True)
    args=p.parse_args();receipt=c.confined(args.receipt)
    c.require(not receipt.is_relative_to(c.PACKET) and not receipt.is_relative_to(c.V4),
              'Verification receipt is external to sealed sources')
    value=check();c.write(receipt,value);print(json.dumps(value))


if __name__=='__main__':
    main()

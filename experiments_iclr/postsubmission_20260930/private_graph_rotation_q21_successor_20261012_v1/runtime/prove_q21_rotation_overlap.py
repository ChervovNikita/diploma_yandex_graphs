"""Static finite-unit mapping proof against the actual frozen Q21 factory."""
import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path

from adapters.private_rotation import SOURCES,receipt
from spec import fingerprint

FACTORY_SHA='e45397a0d2f13002acad4cc80e9b980665957433d7f973ff1a36f7b47bc08a0f'
MAPPING={
 'private_graph_rotation_shared4_coherent':'private_bootstrap_control_shared4_coherent',
 'private_graph_rotation_shared4_paired_graph':'private_bootstrap_control_shared4_paired_graph',
 'private_graph_rotation_shared4_rank1_lora_graph':'private_bootstrap_control_shared4_rank1_lora_graph'}


def function(tree,name):
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:raise ValueError('One exact AST function required: '+name)
    return nodes[0]


def prove(root,packet):
    root=Path(root).resolve();packet=Path(packet).resolve()
    bindings=json.loads((packet/'SOURCE_BINDINGS.json').read_text())
    literal=ast.parse((root/bindings['family']['path']).read_bytes())
    q21_binding=json.loads((packet/'lineage/prior_q21/SOURCE_BINDINGS.json').read_text())
    prior=ast.parse((root/q21_binding['family']['path']).read_bytes())
    factory_path=root/'training_pipeline_v2_20261012/adapters/private_bootstrap_source.py'
    factory=receipt(root,str(factory_path.relative_to(root)))
    if factory['sha256']!=FACTORY_SHA:raise ValueError('Actual reviewed Q21 factory changed')
    tree=ast.parse(factory_path.read_bytes());methods={}
    for name in ('seeds','make','scope','logits'):
        left=ast.dump(function(literal,name));right=ast.dump(function(prior,name))
        if left!=right:raise ValueError('Literal constructor/stream method differs: '+name)
        methods[name]={'identical':True,'AST_sha256':hashlib.sha256(left.encode()).hexdigest()}
    for name in ('corrections','native','routes','factors','seedlaw','base_train'):
        if bindings[name]['sha256']!=q21_binding[name]['sha256']:raise ValueError('Shared source pin differs')
    source_unit=next(n for n in ast.walk(function(literal,'fit')) if isinstance(n,ast.If) and ast.unparse(n.test)=='bank')
    # The weighted branch also has a logits assignment. Resolve only the final
    # unweighted four-route else body, never the weighted block derivative.
    chain=next(n for n in function(tree,'custom_train').body if isinstance(n,ast.If) and ast.unparse(n.test)=='weighted')
    while len(chain.orelse)==1 and isinstance(chain.orelse[0],ast.If):chain=chain.orelse[0]
    q21_nodes=chain.orelse
    def core(nodes):
        chosen=[]
        for node in nodes:
            if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in ('logits','loss'):chosen.append(copy.deepcopy(node))
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('loss.backward','optimizer.step'):chosen.append(copy.deepcopy(node))
        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self,n):
                if ast.unparse(n) in ('self.F','torch.nn.functional'):return ast.Name(id='F',ctx=ast.Load())
                if ast.unparse(n)=='self.train_ids':return ast.Name(id='ids',ctx=ast.Load())
                return self.generic_visit(n)
            def visit_Subscript(self,n):
                if ast.unparse(n)=='self.data.y[self.train_ids]':return ast.Name(id='y',ctx=ast.Load())
                return self.generic_visit(n)
            def visit_Name(self,n):
                if n.id=='source':return ast.Name(id='self',ctx=n.ctx)
                return ast.Constant(4) if n.id=='members' else n
        unit=ast.Module(body=chosen,type_ignores=[])
        return ast.dump(Normalize().visit(unit))
    left=core(source_unit.body);right=core(q21_nodes)
    if left!=right:raise ValueError('Four-route finite loss/backward/step differs')
    prior_plan=json.loads((packet/'lineage/prior_q21/matrix/PLAN.json').read_text());cohorts=[]
    for row in prior_plan['studies']:
        study=json.loads(gzip.decompress((packet/'lineage/prior_q21/matrix/studies'/Path(row['path']).name).read_bytes()))
        if fingerprint(study)!=row['spec_hash']:raise ValueError('Prior Q21 study changed')
        expected={(split,seed) for split in range(20 if study['dataset']=='wikics' else 10) for seed in (77201,77203,77207)}
        for canonical,method in MAPPING.items():
            fits=[r for r in study['runs'] if r['kind']=='model_fit' and r['method']==method]
            if any(r['adapter']!='massive_private_bootstrap_packet_20261012_v1/runtime_v3/adapters/private_bootstrap_source.py:make_adapter'
                   or r['config']['members']!=4 or 'bootstrap_weights' in r['config'] or r['config']['weight_decay']!=0
                   or r['budget']!={'unit':'optimizer_updates','policy':'full_budget','maximum':5000}
                   or r['selection']['ties']!='first' or r['selection']['checkpoint0'] is not False for r in fits):
                raise ValueError('Actual Q21 mapped fit recipe differs')
            exports=[r for r in study['runs'] if r['kind']=='ensemble_assembly' and r['method']==method]
            if {(r['split'],r['seed']) for r in exports}!=expected or any(r['config']['fusion']!='mean_probabilities' for r in exports):
                raise ValueError('Complete mapped cohort/serving roster required')
            cohorts.append({'study_id':study['study_id'],'canonical_ID':canonical,'existing_method':method,
              'existing_fit_groups':len(fits),'existing_comparative_banks':len(exports),'additional_fit_groups':0,
              'fit_run_ids':[r['run_id'] for r in fits],'export_run_ids':[r['run_id'] for r in exports]})
    result={'schema':'private-rotation-Q21-finite-unit-equivalence-v1','actual_frozen_factory':factory,
      'actual_remote_factory_path':'massive_private_bootstrap_packet_20261012_v1/runtime_v3/adapters/private_bootstrap_source.py',
      'literal_source_family':bindings['family'],'Q21_source_family':q21_binding['family'],
      'constructor_and_RNG_method_proof':methods,'normalized_unweighted_bank_core_AST_identical':True,
      'normalized_core_AST_sha256':hashlib.sha256(left.encode()).hexdigest(),
      'normalization':'self.F/torch.nn.functional->F;source->self;self.train_ids->ids;self.data.y[self.train_ids]->y;members->4; only logits/loss/backward/step',
      'law':'CE(flatten(logits[4,T,C]),y.repeat(4)),mean over4T;shared=sum private-route contributions/4;each private route receives1/4;one backward;one AdamW;no private rescaling',
      'optimizer':'one exact Family.make AdamW group;all native/factor/correction parameters once;lr/weight_decay0/default betas(.9,.999)/eps1e-8',
      'canonical_to_existing_methods':MAPPING,'cohorts':cohorts,'new_bank_fits_for_mapped_three':0,
      'finite_unit_learning_equivalence':True,'whole_legacy_trajectory_equivalence':False,'failure_behavior_equivalence':False,
      'limits':['Rotation source guards finite bank loss before backward;Q21 checks parameters/gradients after step.',
        'Source M1 uses private scope(stream0);existing Q21 M1 uses global dropout state. Existing Q21 M1/I4 controls remain competent references with actual RNG provenance.',
        'Source historical max1000/patience300/old role dimensions differ from approved full5000/all20-or10 masks/firstSELECT/commonCAL.',
        'Existing coherent factorized_allmap_M1 is distinct from first-input-Rademacher M1. The added first-input-Rademacher M1/I4 is an explicit equipment/RNG contrast.'],
      'old_weighted_procedures_unchanged':True,'scientific_outcomes_read':False,'CUDA':0,'scientific_fits':0,'GO':False}
    (packet/'Q21_FINITE_BANK_EQUIVALENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--packet',required=True)
    a=p.parse_args();v=prove(a.project_root,a.packet);print(json.dumps({'mapped_bank_methods':len(MAPPING),'cohort_mappings':len(v['cohorts']),'additional_mapped_bank_fits':0}))

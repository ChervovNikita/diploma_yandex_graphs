"""Write fixed disabled configs and reproducible source seal; never fit/train."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARMS=['single','single_contrastive','independent4','independent4_contrastive',
      'be_unit','be_init','be_unit_contrastive','be_init_contrastive']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    deps={'polynormer':{'path':'wikics_native_polynormer_r_donor_preparation_20261007_v1/vendor/native_polynormer.py',
        'sha256':'9b4e533f46ae7f91a23a552f88bbb47a01359e224b53996cf1a68c10a865f6a8',
        'author_repository':'https://github.com/cornell-zhang/Polynormer',
        'author_commit':'fc8c276c9c5dfbd616d83f65338a3392188a5e08'},
        'ncn':{'author_repository':'https://github.com/GraphPKU/NeuralCommonNeighbor',
        'author_commit':'11d597013750da17ce7468e344bec756a7af39a4',
        'license_status':'No LICENSE/COPYING in retained pinned tree; redistribution unresolved; existing private research dependency only',
        'model':{'path':'graph_ncNC_member_completion_qualification_preparation_20261003_v1/private_evidence/native/model.py',
        'sha256':'5b6cfe26074c83d57900f589b37a4ef57fc7b8eb6dff3179fc52e3f6943aa5e3'},
        'utils':{'path':'graph_ncNC_member_completion_qualification_preparation_20261003_v1/private_evidence/native/utils.py',
        'sha256':'29df3b9cf82ae5885ece0fac010b81383f1fb5391068649c88b7d2bc98ad86be'}}}
    write(ROOT/'DEPENDENCIES.json',deps)
    tasks={
      'wikics':{'model':dict(in_channels=300,hidden_channels=512,out_channels=10,local_layers=7,global_layers=2,
          in_dropout=.5,dropout=.5,global_dropout=.5,heads=1,beta=-1,pre_ln=False),
          'training':dict(epochs=1100,local_epochs=100,lr=.001,batch_size=580,eval_batch_size=5274),
          'metric':'official_split0_accuracy','pool':'mean_probability','cell_soft_seconds':28800,'cell_hard_seconds':32400},
      'collab':{'model':dict(features=128,hidden=64),
          'training':dict(epochs=100,lr=None,encoder_lr=.0082,decoder_lr=.0037,batch_size=65536,eval_batch_size=131072),
          'metric':'official_Hits50','pool':'mean_raw_logit','cell_soft_seconds':43200,'cell_hard_seconds':46800},
      'molhiv':{'model':dict(hidden=256,layers=5,dropout=.5),
          'training':dict(epochs=100,lr=.001,batch_size=128,eval_batch_size=128),
          'metric':'official_scaffold_ROC_AUC','pool':'mean_raw_logit','cell_soft_seconds':28800,'cell_hard_seconds':32400}}
    for task,values in tasks.items():
        budget={k:values.pop(k) for k in ('cell_soft_seconds','cell_hard_seconds')}
        budget.update(max_cells=24,max_pilot_epochs=24*values['training']['epochs'],
                      estimates_measured=False,resource_qualification_required=True,no_shortening_for_cap=True)
        config=dict(schema='internal-be-fixed-representative-pilot-v1',task=task,
            arms=ARMS,pilot_seeds=[6101,6203,6307],confirmation_seeds=[7109,7211,7309,7411,7517],
            budget=budget,contrastive=dict(alignment_weight=.05,residual_weight=.05,temperature=.2,max_objects=512),
            normalization='stateless_shared_LayerNorm_per_object; no_BatchNorm',
            initialization='native_reset; first_r_Rademacher_seed+900001_only_for_be_init; other_factors_one',
            label_opportunity='two_full_own_loss_views_every_arm; all_TRAIN_labels_every_member',
            checkpoint='strict_first_max_complete_VALID; native_local_transition; no_early_stop',
            root_adopted=False,tranche_order=['wikics','collab','molhiv'],automatic_next_tranche=False,**values)
        write(ROOT/'configs'/f'{task}.json',config)
        write(ROOT/'jobs'/f'{task}_TEMPLATE_DISABLED.json',dict(schema='internal-be-predictive-cell-v1',task=task,
            arm='be_init_contrastive',seed=6101,root_execution_authorized=False,source_review_approved=False,
            fixed_protocol_adopted=False,
            TEST_access=False,automatic_retry=False,source_manifest_sha256=None,
            config={'path':ROOT.name+'/configs/'+task+'.json','sha256':None},
            data_manifest={'path':None,'sha256':None},source_review_evidence=[],resource_qualification_evidence=[],
            data_export_review={'path':None,'sha256':None},supervisor_receipt_path=None,
            output_directory=None,external_hard_bound_confirmed=False,
            soft_seconds=budget['cell_soft_seconds'],hard_seconds=budget['cell_hard_seconds']))
    for task in tasks:
        write(ROOT/'jobs'/f'{task}_EXPORT_TEMPLATE_DISABLED.json',dict(task=task,data_export_authorized=False,source_review_approved=False,TEST_access=False,source_manifest_sha256=None,source_review={'path':None,'sha256':None},output_directory=None))
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    exclude={'MANIFEST.json','CPU_CHECK.json','STAGE_RECEIPT.json','STATIC_CHECK.json','READ_SCOPES.json','PREPARATION_HANDOFF.json','CPU_FAILURE.json'}
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name not in exclude and '__pycache__' not in path.parts:
            files.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    write(ROOT/'MANIFEST.json',dict(schema='internal-be-source-seal-v1',files=files,
        training_enabled=False,scientific_results_present=False))
    write(ROOT/'STATIC_CHECK.json',dict(passed=True,python_AST_files=len(list(ROOT.glob('*.py'))),
        source_manifest_sha256=hashlib.sha256((ROOT/'MANIFEST.json').read_bytes()).hexdigest()))
    print(json.dumps({'source_sealed':True,'files':len(files),'training_enabled':False}))


if __name__=='__main__':main()

"""Bind closed preparation/qualification into one new finite stage config."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REMOTE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
SOURCE=PHASE/'pubmed_private_landmark_memory_execution_source_20261011_v1'

def receipt(path):
    return dict(path=str(REMOTE/path.relative_to(PHASE)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')

def closed(folder,name):
    owner=json.loads((folder/'OWNER_END.json').read_text())
    report=json.loads((folder/name).read_text())
    item=json.loads((folder/(name+'.REMOTE_RECEIPT.json')).read_text())
    assert owner['status']=='COMPLETE_EXIT_0' and owner['exit_code']==0 and not owner['timed_out']
    assert owner['direct_child_wait'] and owner['child_pid_absent'] and owner['owned_cuda_pid_absent']
    assert owner['produced_artifact']==item and report['complete']
    assert owner['source_manifest']==receipt(SOURCE/'MANIFEST.json')==report['source_manifest']
    return report,item,json.loads((folder/'OWNER_END.json.REMOTE_RECEIPT.json').read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('qualify','acquire'));parser.add_argument('expected_head');args=parser.parse_args()
    prepared,custody,prepare_end=closed(HERE/'prepare_recovery01','CUSTODY.json')
    assert prepared['TEST_read'] is False and prepared['original_VALID_exact_disjoint_union']
    stage=HERE/args.stage;stage.mkdir(exist_ok=False)
    config=json.loads((SOURCE/(args.stage.upper()+'_CONFIG.disabled.json')).read_text())
    config.update(execution_enabled=True,source_manifest=receipt(SOURCE/'MANIFEST.json'),prepared_custody=custody,prepare_owner_end=prepare_end)
    decision=dict(UTC=datetime.now(timezone.utc).isoformat(),admitted_stages=[args.stage],
        source_manifest=config['source_manifest'],prepared_custody=custody,prepare_owner_end=prepare_end,
        allocation_only=True,original_scores_unchanged=True,TEST_closed=True,
        scientific_recipe_unchanged=True,partial_predictive_outcomes_used=False,
        numerical_setup_is_not_quality_evidence=True)
    if args.stage=='qualify':
        decision.update(status='ONE_MATERIAL_FULL_INPUT_QUALIFIER_ADMITTED',
            checks=['4 discarded native-source full TRAIN updates','26 stage2 updates across all8 interfaces',
                    'zero readout source containment and step2 finite nonzero memory gradients',
                    'exact model/Adam/modes/global/private-RNG restore','genuine disjoint equipped I4 and full serving'],
            qualifier_labels='TRAIN only',acquisition_admitted=False,optional_check_expansion=False)
    else:
        qualified,qualifier,qualifier_end=closed(HERE/'qualify','QUALIFICATION.json')
        assert qualified['status']=='pass' and qualified['interfaces']==8 and qualified['source_updates']==4 and qualified['stage2_updates']==26
        assert qualified['checkpoint0_empty_moments_and_full_restore'] and qualified['all_memory_second_step_CE_gradients_verified']
        assert qualified['prepared_custody']==custody
        config.update(qualification=qualifier,qualification_owner_end=qualifier_end)
        decision.update(status='COMPLETE_FIXED_ACQUISITION_ADMITTED_AFTER_MATERIAL_QUALIFICATION',
            qualification=qualifier,qualification_owner_end=qualifier_end,
            counts=dict(fresh_banks=36,fresh_trajectories=54,serving_endpoints=39),
            source_ancestors='12 fresh factorized M1s; candidate uses one per seed and other bodies are reference costs',
            fits_access='TRAIN and SELECT only; CAL/DEV exports contain no targets',
            utility='All declared equipped singles and genuine I4 references; whole family before scoring',
            no_scientific_retry_or_rescue_grid=True,unused_confirmation_or_novelty_established=False)
    write(stage/'ROOT_DECISION.json',decision);config['root_decision']=receipt(stage/'ROOT_DECISION.json');write(stage/'CONFIG.json',config)
    publication=PHASE/('publication/PubMed_LM4_'+args.stage+'_admission_20261011_v1');publication.mkdir(exist_ok=False)
    state=PHASE/'RESEARCH_STATE.md';(publication/'PREVIOUS_RESEARCH_STATE.md').write_bytes(state.read_bytes())
    state.write_text(state.read_text()+'\n## PubMed LM4 '+args.stage+' admission\n\n'+
        ('Exact missing metadata was restored. Preparation closed successfully with frozen SELECT/CAL/DEV roles and three label-free landmark/BFS capsules. One material full-input qualifier is admitted; its launch and result remain to verify. No model acquisition is admitted yet.' if args.stage=='qualify' else
         'The single material full-input qualifier passed all8 interfaces,4 discarded source/26 stage2 updates, second-step memory gradients, exact state restoration and full equipped I4 serving within finite caps. The unchanged complete36-bank/54-trajectory comparison is admitted; actual launch remains to verify. No partial development outcomes were used and no quality/novelty conclusion follows from qualification.')+'\n')
    ledger_path=PHASE/'research_ledger.json';ledger=json.loads(ledger_path.read_text());key='pubmed_LM4_'+args.stage+'_admission_20261011_v1';assert key not in ledger
    ledger[key]=decision;ledger_path.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
    folders=[HERE]
    if args.stage=='qualify':folders.append(PHASE/'pubmed_private_landmark_memory_readout_source_20261011_v1')
    paths=[p for folder in folders for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md') and '__pycache__' not in p.parts]
    paths.extend([state,ledger_path,publication/'PREVIOUS_RESEARCH_STATE.md'])
    rows=[]
    for path in sorted(set(paths)):
        rel=path.relative_to(PHASE).as_posix();item=receipt(path);assert item['bytes']<2000000
        rows.append(dict(source=rel,target='experiments_iclr/postsubmission_20260930/'+rel,bytes=item['bytes'],sha256=item['sha256']))
    write(publication/'INVENTORY.json',dict(branch='codex/postsubmission-research-20260930',expected_head=args.expected_head,
        message='Admit PubMed landmark-memory '+args.stage+' after exact prior closure',files=rows,remove=[]))
    print(json.dumps(dict(stage=args.stage,files=len(rows),limits=config['limits'])))

if __name__=='__main__':main()

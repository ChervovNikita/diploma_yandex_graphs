"""Disabled complete27 metadata readout; no model/array imports or forwards."""
import argparse
import json
import math
from pathlib import Path
import socket
from source import HERE,PHASE,bind,inside,sha,verify_manifest,write
from closure import positive_nine,require_27
from stage_plan import SEEDS,STAGE1,ALL_CONDITIONS,SERVER_HOST,SERVER_PHASE,spec as condition_spec

CONTRASTS=(('shared4_core','shared4_own'),('shared4_masked_ce','shared4_own'),
    ('shared4_core','shared4_masked_ce'),('shared4_core','shared4_core_shuffled'),
    ('shared4_core','shared4_core_rewired'),('shared4_core','single_native'),
    ('single_four_view_core','single_native'),('shared4_core','single_four_view_core'),
    ('untied4_shared_decoder_core','independent4_native'),('shared4_core','untied4_shared_decoder_core'),
    ('shared4_own','single_native'),('shared4_core','independent4_native'))


def validate_metrics(value,members):
    def checked(row,count):
        if row.get('count')!=count or type(row.get('correct')) is not int or not 0<=row['correct']<=count:raise ValueError('Exact role correct counts required')
        if row.get('accuracy')!=row['correct']/count or not math.isfinite(row['NLL']) or row['NLL']<0:raise ValueError('Exact count accuracy and finite NLL required')
    checked(value['pooled'],3942)
    if len(value['members'])!=members or len(value['classes'])!=3:raise ValueError('All factual members/classes required')
    for row in value['members']:checked(row,3942)
    for c,count in enumerate((820,1547,1575)):
        row=value['classes'][c]
        if row['class_id']!=c or len(row['members'])!=members:raise ValueError('Fixed class order and member dimension')
        checked(row,count)
        for member in row['members']:checked(member,count)
    if sum(c['correct'] for c in value['classes'])!=value['pooled']['correct']:raise ValueError('Class/pooled count mismatch')
    for m in range(members):
        if sum(c['members'][m]['correct'] for c in value['classes'])!=value['members'][m]['correct']:raise ValueError('Member/class count mismatch')


def read_complete27(release,digest):
    path=inside(release)
    if sha(path)!=digest:raise ValueError('Exact separate root complete27 readout release required')
    spec=json.loads(path.read_text())
    if spec.get('schema')!='masked-context-stage2-complete27-readout-release-v1':raise ValueError('Separate complete27 readout release')
    for key in ('enabled','root_readout_authorized','whole27_owned_completion_verified'):
        if spec.get(key) is not True:raise ValueError('Complete27 readout remains disabled: '+key)
    if spec.get('TEST_access') is not False or spec.get('automatic_retry') is not False:raise ValueError('TEST/retry closed')
    bindings=verify_manifest(spec['source_manifest_sha256'])
    positive_nine(spec['stage1_comparison'],bindings)
    expected={f'seed{seed}__{name}':(seed,name) for seed in SEEDS for name in ALL_CONDITIONS}
    if set(spec.get('records',{}))!=set(expected):raise ValueError('Exactly all original27 records required')
    if socket.gethostname()!=SERVER_HOST or str(PHASE)!=SERVER_PHASE:raise ValueError('Server-only metadata readout')
    # Every owned terminal is checked before the first completion metric is opened.
    terminals={}
    for identity,rows in spec['records'].items():
        terminal=json.loads(bind(rows['terminal_custody']).read_text())
        if terminal.get('schema')!='masked-context-stage1-owned-terminal-custody-v1' or terminal.get('record_id')!=identity:raise ValueError('Exact owned terminal identity')
        if terminal.get('directly_waited') is not True or terminal.get('child_exit_code')!=0 or terminal.get('cap_or_owner_failure') is not None:raise ValueError('Every actual successful direct wait required')
        if terminal.get('owned_process_absence_verified') is not True or terminal.get('owned_CUDA_absence_verified') is not True:raise ValueError('Actual owned absence required')
        raw=json.loads(bind(terminal['raw_owner_terminal']).read_text());bind(terminal['owned_absence_evidence'])
        if raw.get('directly_waited') is not True or raw.get('child_exit_code')!=0 or raw.get('cap_or_owner_failure') is not None or terminal['release_sha256'] not in raw.get('argv',[]):raise ValueError('Raw owner direct-wait/release evidence required')
        if terminal.get('complete_sha256')!=rows['complete']['sha256']:raise ValueError('Terminal/completion binding mismatch')
        terminals[identity]=terminal
    records={}
    for identity,rows in spec['records'].items():
        f=bind(rows['complete']);r=json.loads(f.read_text());seed,name=expected[identity]
        manifest=bindings['stage1_manifest']['sha256'] if name in STAGE1 else spec['source_manifest_sha256']
        schema='masked-context-PubMed-stage1-complete-v1' if name in STAGE1 else 'masked-context-PubMed-stage2-complete-v1'
        if r.get('schema')!=schema or r.get('complete') is not True or (r.get('seed'),r.get('condition'),r.get('record_id'))!=(seed,name,identity):raise ValueError('Immutable exact complete record identity')
        if r.get('source_manifest_sha256')!=manifest or terminals[identity]['source_manifest_sha256']!=manifest or r['release_sha256']!=terminals[identity]['release_sha256']:raise ValueError('Original source/release lineage must be preserved')
        if r.get('max_epochs')!=2000 or r.get('patience')!=250 or not 1<=r['selected_epoch']<=r['epochs_executed']<=2000:raise ValueError('Complete full declared trajectory')
        if r.get('stopped_by')=='patience250':
            if r['epochs_executed']-r['selected_epoch']!=250:raise ValueError('Exact patience closure')
        elif r.get('stopped_by')!='max2000' or r['epochs_executed']!=2000:raise ValueError('No shortened control completion')
        if r.get('selected_mask_diagnostics_epoch')!=r['selected_epoch'] or r.get('TEST_scored') is not False or r.get('TEST_labels_or_id_inputs') is not False:raise ValueError('Same selected epoch and TEST closure')
        if r.get('split_seed')!=190111 or r.get('split_identity')!='PubMed-class-stratified-floor60-20-20':raise ValueError('Unchanged representative split')
        for key in ('complete_saved_member_logits','selected_predictor'):
            payload=r[key];target=(f.parent/payload['path']).resolve(strict=True)
            if not target.is_relative_to(f.parent) or payload.get('server_only') is not True or sha(target)!=payload['sha256'] or target.stat().st_size!=payload['bytes']:raise ValueError('Complete saved server-only selected payload')
        validate_metrics(r['selected_VALID'],condition_spec(name)['members'])
        records[seed,name]=r
    require_27(records)
    if len({(r['train_bundle_sha256'],r['valid_bundle_sha256']) for r in records.values()})!=1:raise ValueError('Exact same TRAIN/VALID arrays for all27')
    output=inside(spec['output'],existing=False)
    if output.exists() or output.is_relative_to(HERE):raise ValueError('Fresh separate readout output')
    return spec,records,output


def summarize(records):
    require_27(records)
    mean=lambda values:sum(values)/len(values)
    contrasts=[]
    for a,b in CONTRASTS:
        rows=[]
        for seed in SEEDS:
            x,y=records[seed,a]['selected_VALID'],records[seed,b]['selected_VALID']
            rows.append(dict(seed=seed,accuracy_pp=100*(x['pooled']['accuracy']-y['pooled']['accuracy']),
                NLL=x['pooled']['NLL']-y['pooled']['NLL'],macro_pp=100*(x['macro_accuracy']-y['macro_accuracy']),
                mean_member_pp=100*(x['mean_member_accuracy']-y['mean_member_accuracy']),
                worst_member_pp=100*(x['worst_member_accuracy']-y['worst_member_accuracy'])))
        contrasts.append(dict(a=a,b=b,paired=rows,means={k:mean([r[k] for r in rows]) for k in ('accuracy_pp','NLL','macro_pp','mean_member_pp','worst_member_pp')}))
    interaction=[]
    for seed in SEEDS:
        acc=lambda name:records[seed,name]['selected_VALID']['pooled']['accuracy']
        interaction.append(dict(seed=seed,accuracy_interaction_pp=100*(acc('shared4_core')-acc('shared4_own')-acc('single_four_view_core')+acc('single_native'))))
    return dict(schema='masked-context-complete27-mechanism-readout-v1',complete27=True,contrasts=contrasts,
        route_bank_by_auxiliary_package_interaction=interaction,
        mean_accuracy_interaction_pp=mean([r['accuracy_interaction_pp'] for r in interaction]),
        actual_repair_sets_not_inferred_from_aggregate_scores=True,
        positive_interaction_does_not_override_required_utility=True,
        own_selected_independent_reference_still_required=True,
        claim_limit='Descriptive one-split mechanism screen with disclosed body/parameter/factual-view/selector differences; no universal superiority, causal isolation, novelty or unused confirmation.',
        later_fit_activation=False,TEST_access=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('describe','readout'),default='describe')
    parser.add_argument('--release',type=Path);parser.add_argument('--release-sha256')
    args=parser.parse_args()
    if args.mode=='describe':print(json.dumps(dict(enabled=False,required_records=27,contrasts=CONTRASTS,TEST_access=False)));return
    if args.release is None or args.release_sha256 is None:parser.error('Separate exact root complete27 release required')
    spec,records,output=read_complete27(args.release,args.release_sha256)
    output.mkdir(parents=True,exist_ok=False);result=summarize(records)
    result['record_custody']=spec['records'];result['readout_release_sha256']=args.release_sha256
    write(output/'COMPLETE27_READOUT.json',result)
    print(json.dumps(dict(complete27=True,output=str(output),later_fit_activation=False)))


if __name__=='__main__':main()

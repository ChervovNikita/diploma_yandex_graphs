"""Closed local-source setting roster. No numerical imports or remote access."""
import csv
import hashlib
import json
import shlex
from pathlib import Path

OUT=Path(__file__).resolve().parent
PHASE=OUT.parents[1]
S=PHASE/'coordinate_source_independent_review_v1/strong_backbones_v1/sources'
OLD={'roman-empire','amazon-ratings','minesweeper','questions','tolokers'}
rows=[]
base={'primary_arms':'native_M1; original_boundary_GNNM; genuine_ordinary_I4',
      'HPO':False,'scientific_execution_enabled':False,'new_comparison_seeds':10,
      'reference_mean':None,'reference_std':None,'reference_other_dispersion':None,
      'reference_dispersion_type':'unverified','reference_metric':'',
      'reference_score_null_reason':'No exact numeric table cell verified in retained sources',
      'reported_secondary_metrics':[],'warmup':0,'patience':None}
def add(**row):
    assert row['dataset'] not in OLD
    rows.append(base|row)
def parse_flags(tokens):
    out={};i=0
    while i<len(tokens):
        if tokens[i].startswith('--'):
            k=tokens[i][2:]
            if '=' in k:k,v=k.split('=',1);out[k]=v;i+=1
            elif i+1<len(tokens) and not tokens[i+1].startswith('--'):
                out[k]=tokens[i+1];i+=2
            else:out[k]=True;i+=1
        else:i+=1
    return out

poly_scores={
 'mono':[(82.37,.65),(96.49,.09),(91.01,.41),(98.42,.16)],
 'cheb':[(81.80,.76),(96.49,.17),(90.68,.31),(98.08,.27)]}
aliases={'citeseer':'citeseer','cs':'coauthor-cs','pubmed':'pubmed','physics':'coauthor-physics'}
order={'citeseer':0,'cs':1,'pubmed':2,'physics':3}
for line_no,line in enumerate((S/'polyformer_code/node_classification/node_classification.sh').read_text().splitlines(),1):
    if not line.startswith('python'):continue
    p=parse_flags(shlex.split(line))
    if p['dataset'] not in aliases:continue
    dataset=aliases[p['dataset']];basis=p['base'];mean,ci=poly_scores[basis][order[p['dataset']]]
    settings={k:p[k] for k in ('lr','weight_decay','attn_lr','attn_wd','hidden','dropout','dprate','n_head','d_ffn','q','multi','K','nlayer')}
    add(cell_id='polyformer-'+dataset+'-'+basis,dataset=dataset,architecture='PolyFormer',
        preset=basis,roster_role='primary_fixed_setting',status='preparation_pending',
        setting_evidence_class='paper_task_and_author_release_fixed_setting',settings=settings,
        optimizer='Adam; ordinary/attention parameter groups',epochs=2000,patience=250,
        seed_policy='Source ten advancing-RNG repeats from global seed42; ten literal split seeds. Independent paired optimizer reseeding is a labeled comparison adaptation, not published seed-protocol equality.',
        split_policy='Source random 60/20/20, all ten literal split seeds; no SELECT/CAL subdivision',
        reference_metric='accuracy_percent',reference_mean=mean,reference_other_dispersion=ci,
        reference_dispersion_type='95_percent_confidence_interval_over_10_runs',reference_score_null_reason='',
        score_location='polyformer_paper.txt Table5 '+('Mono' if basis=='mono' else 'Cheb')+' / '+p['dataset'],
        source_script=str((S/'polyformer_code/node_classification/node_classification.sh').relative_to(PHASE))+':'+str(line_no),
        source_program=str((S/'polyformer_code/node_classification/training.py').relative_to(PHASE)),
        comparability_limits='Source-task/native split preservation is required. Changed runtime and independently seeded wrapper cohorts are declared adaptations; no achieved score claimed.',
        published_vs_release_schedule='Both source maximum2000/patience250; fixed native recipe, no HPO')

ogb_scores={'ogbg-molhiv':{'gcn':(76.06,.97),'gcn-virtual':(75.99,1.19),'gin':(75.58,1.40),'gin-virtual':(77.07,1.49)},
            'ogbg-molbace':{'gcn':(79.15,1.44),'gcn-virtual':(68.93,6.95),'gin':(72.97,4.00),'gin-virtual':(73.46,5.24)}}
for dataset in ('ogbg-molhiv','ogbg-molbace'):
    for arch in ('gcn','gcn-virtual','gin','gin-virtual'):
      for dropout in (0.0,0.5):
        mean,std=ogb_scores[dataset][arch]
        add(cell_id='ogb-'+dataset+'-'+arch+'-drop'+str(dropout),dataset=dataset,architecture=arch,preset='full_features_fixed_dropout_'+str(dropout),
            roster_role='primary_fixed_setting',status='preparation_pending',
            setting_evidence_class='two_named_paper_declared_dropout_presets; exact_paper_winner_unknown',
            settings={'hidden':300,'layers':5,'dropout':dropout,'lr':.001,'weight_decay':0,'batch_size':32,
                'feature':'full','residual':False,'JK':'last','graph_pooling':'mean','num_workers':0},
            optimizer='Adam, native defaults',epochs=100,
            seed_policy='No seed/repeat option or native seed call. New comparison uses ten default seeds0..9, set before imports; explicitly declared repetition adaptation.',
            split_policy='Official scaffold TRAIN / whole VALID / TEST; original categorical atom and bond features; no CAL subdivision',
            reference_metric='rocauc_percent',reference_mean=mean,reference_std=std,
            reference_dispersion_type='standard_deviation_over_10_runs',reference_score_null_reason='',
            reference_exact_configuration_match='unknown_selected_paper_dropout_between_0_and_0.5',
            score_location='reference_wrapper_redesign_20261013_v1/references/ogb_paper.html Table'+('15/fullHIV' if dataset=='ogbg-molhiv' else '23/fullBACE')+'; primarytext1805 std10 and7194 tuned dropout0/.5',
            source_script='main_pyg.py --dataset '+dataset+' --gnn '+arch+' --drop_ratio '+str(dropout)+' --num_layer5 --emb_dim300 --batch_size32 --epochs100 --feature full',
            source_program='training_pipeline_v2_20261012/preparation/ogb_native_source_readonly_root_20261012_v1/source/main_pyg.py',
            comparability_limits='Paper tuned dropout0/.5; exact selected value per numeric cell unknown. Both presets are predeclared fixed experiments, no HPO/winner selection or claim of exact paper-winning config. Whole VALID selects first argmax ROC-AUC; native source has no selected-model checkpoint. Source train/eval singleton skipping and advancing shuffled TRAIN eval must remain.',
            published_vs_release_schedule='100 complete epochs; no native early stopping or warmup')

polynormer_scores={'amazon-photo':{'pure':(96.11,.23),'relu':(96.46,.26)},
                   'wikics':{'pure':(79.53,.83),'relu':(80.10,.67)}}
for line_no,line in enumerate((S/'polynormer_code/run.sh').read_text().splitlines(),1):
    if not line.startswith('python'):continue
    p=parse_flags(shlex.split(line))
    if p['dataset'] not in polynormer_scores:continue
    dataset=p['dataset'];local=int(p['local_epochs']);release_global=int(p['global_epochs'])
    for variant in ('pure','relu'):
        mean,std=polynormer_scores[dataset][variant]
        # Paper explicitly states effectiveH512/eight heads; release has per-head
        # constructor width64x8 on Photo and512x1 on WikiCS. Keep presets separate.
        paper_settings={'effective_hidden':512,'attention_heads':8,'lr':.001,
            'weight_decay':float(p['weight_decay']),'dropout':float(p['dropout']),
            'input_dropout_from_release':float(p['in_dropout']),
            'global_dropout_from_release':float(p.get('global_dropout',p['dropout'])),
            'local_layers':int(p['local_layers']),'global_layers':int(p['global_layers']),
            'activation_variant':'no_ReLU' if variant=='pure' else 'ReLU',
            'beta_from_release':-1,'qk_shared_from_release':True}
        total=1000 if dataset=='amazon-photo' else 500
        split=('Source archived random60/20/20 split[0] reused for all runs; bind exact archive before activation'
            if dataset=='amazon-photo' else 'Enumerate all official WikiCS train-mask columns; VALID=val_mask OR stopping_mask; common official TEST; mask/run mapping explicit')
        common={'dataset':dataset,'architecture':'Polynormer' if variant=='pure' else 'Polynormer-r',
            'status':'preparation_pending','optimizer':'Adam, native defaults','warmup':local,
            'seed_policy':'Source defaultseed42 with runs1 command; paper averages10runs. New ten paired seed cohorts labeled where overriding advancing source RNG. Preserve official mask axis independently.',
            'split_policy':split,'reference_metric':'accuracy_percent','reference_mean':mean,'reference_std':std,
            'reference_dispersion_type':'standard_deviation_over_10_runs','reference_score_null_reason':'',
            'score_location':'polynormer_paper.txt Table1 '+('Polynormer' if variant=='pure' else 'Polynormer-r')+' / '+dataset,
            'source_script':str((S/'polynormer_code/run.sh').relative_to(PHASE))+':'+str(line_no),
            'source_program':str((S/'polynormer_code/main.py').relative_to(PHASE)),
            'comparability_limits':'No-ReLU is the documented paper variant; not a new ELU substitution. Release/paper horizon and WikiCS head choices differ. Paper input/global dropout and WD details use named release defaults and are labeled; no exact full-protocol score equality until resolved.',
            'published_vs_release_schedule':'PaperTable5 warmup'+str(local)+'/total'+str(total)+'; release local'+str(local)+'+global'+str(release_global)+'='+str(local+release_global)}
        add(**common,cell_id='polynormer-'+dataset+'-'+variant+'-paper',preset='paper_stated',
            roster_role='primary_fixed_setting',setting_evidence_class='paper_stated_core_settings; named_release_auxiliary_settings',
            settings=paper_settings,epochs=total)
        release_settings=paper_settings|{'hidden_channels_per_head':int(p['hidden_channels']),
            'attention_heads':int(p['num_heads']),'effective_hidden':int(p['hidden_channels'])*int(p['num_heads'])}
        add(**common,cell_id='polynormer-'+dataset+'-'+variant+'-release',preset='author_release_native_horizon',
            roster_role='separate_source_horizon_reference',setting_evidence_class='literal_author_release; not_paper_horizon_equivalent',
            settings=release_settings,epochs=local+release_global)

for arch,predictor in [('NCN','cn1'),('NCNC','incn1cn1')]:
    add(cell_id='collab-'+arch.lower(),dataset='ogbl-collab',architecture=arch,preset='author_collab',
        roster_role='primary_fixed_setting',status='preparation_pending',
        setting_evidence_class='author_release_native_benchmark_command',
        settings={'predictor':predictor,'encoder':'gcn','hidden':64,'encoder_layers':1,
            'encoder_lr':.0082,'predictor_lr':.0037,'weight_decay':0,'batch_size':65536,'test_batch_size':131072,
            'xdropout':.25,'tail_dropout':.05,'encoder_edge_dropout':.25,'predictor_edge_dropout':0,
            'predictor_dropout':.3,'encoder_dropout':.1,'pt':.1,'probscale':2.5,'proboffset':6.,'alpha':1.05,
            'ln':True,'lnnn':True,'residual':True,'maskinput':True,'use_xlin':True,'tailact':True},
        optimizer='Adam; separate encoder/predictor parameter groups',epochs=100,
        seed_policy='Literal --runs10 with set_seed(run):0..9; preserve rather than replace',
        split_policy='Official complete temporalTRAIN/VALID/TEST; unweighted encoderTRAIN topology; --use_valedges_as_input adds VALID for TEST; source whole-minibatch target removal and dropped tail',
        reference_metric='hits_at_50',score_location='NCN paper2302.00890 linked in local ncn README; no numeric paper table retained in bounded source record',
        source_script='graph_link_competing_family_source_20261003_v1/excerpts/ncn__README.md.txt:52',
        source_program='graph_link_competing_family_source_20261003_v1/source/ncn__NeighborOverlap.py',
        comparability_limits='Current training-only-test variants and HeaRT CiteSeer tasks are not this original Collab protocol. Preserve original predictor choices, all official evaluation records and native training tail policy. TEST-only-after-selection monitoring can preserve VALID decision rule but is a disclosed source access adaptation.',
        published_vs_release_schedule='Native100epochs/runs10; no warmup/patience specified in author command')

add(cell_id='collab-buddy',dataset='ogbl-collab',architecture='BUDDY',preset='author_year2007',
    roster_role='primary_fixed_setting',status='preparation_pending',
    setting_evidence_class='author_release_native_benchmark_command',
    settings={'hidden':1024,'lr':.02,'weight_decay':0,'batch_size':1024,'feature_dropout':.05,
        'label_dropout':.1,'year':2007,'K':50,'add_normed_features':True,'cache_subgraph_features':True,
        'eval_batch_size':1000000,'subgraph_feature_batch_size':11000000,'max_hash_hops':2,
        'hll_p':8,'minhash_num_perm':128,'sign_k':0,'num_negs':1,'use_RA':False,'use_zero_one':False,'floor_sf':False},
    optimizer='Adam, native defaults',epochs=100,
    seed_policy='README --reps10 and run.py set_seed(rep):0..9',
    split_policy='Native year>=2007 TRAIN filtering; native TEST topology TRAIN+VALID; official temporal VALID/TEST queries and Hits@50',
    reference_metric='hits_at_50',score_location='BUDDYpaper openreview:m1oqEOAozQU linked in retained README; no numeric paper cell verified locally',
    source_script='link_covariance_idea_v1/sources/buddy_readme.md:Experiments/CollabBUDDY',
    source_program='cached_structure_link_ensemble_feasibility_v1/sources/buddy_src_runners_run.py',
    comparability_limits='README calls repository a reimplementation with evolving defaults; freeze pinned commit. Existing year0/full-node/training-only-test/cache batch65536 packet is an explicit adaptation, not this source preset. Any resource batch reduction must be separately labeled.',
    published_vs_release_schedule='Native100epochs; no early stopping/warmup documented; native default batches differ from preparation adaptations')

add(cell_id='hgb-dblp-hgt-evidence',dataset='HGB-DBLP',architecture='HGT',preset='HGB_author',
    roster_role='excluded_hidden_online_test_evidence',status='not_primary',
    setting_evidence_class='native_source_and_secondary_paper_baseline_reference',
    settings={'hidden':64,'layers':3,'heads':8,'effective_dropout':.2,'feats_type':2,'use_norm':True,
        'weight_decay':.0001,'schedule_max_lr':.001,'schedule_total_steps':300,'schedule_pct_start':.05},
    optimizer='AdamW+OneCycleLR; parsed lr.005/dropout.5 are not passed to optimizer/model',epochs=300,patience=30,
    seed_policy='Source command repeat1; no source seed law established; ten new seeds would be an adaptation',
    split_policy='HGB official public local30% / hidden onlineTEST70%; source localTRAIN/VALID; no inner local holdout replaces officialTEST',
    reference_metric='micro_F1_percent',reference_mean=93.49,reference_other_dispersion=.25,
    reference_dispersion_type='reported_plus_minus_type_not_verified_in_bounded_passage',reference_score_null_reason='',
    reported_secondary_metrics=[{'metric':'macro_F1_percent','mean':93.01,'reported_plus_minus':.23}],
    score_location='SeHGNN saved primarypaper Table3 HGT row/DBLP columns; secondary reference toHGBbaseline, not primaryHGTpaper',
    source_script='graph_heterogeneous_private_modulation_source_20261003_v1/source/hgb__NC__benchmark__methods__HGT__run_dblp.sh',
    source_program='graph_heterogeneous_private_modulation_source_20261003_v1/source/hgb__NC__benchmark__methods__HGT__train_hgt.py',
    comparability_limits='Excluded from primary head-to-head until official hidden evaluation is obtainable; no submission authorized or performed',published_vs_release_schedule='Sourcecap300/patience30; scheduler warm-startpct.05')
add(cell_id='hgb-imdb-sehgnn-evidence',dataset='HGB-IMDB',architecture='SeHGNN',preset='HGB_author',
    roster_role='excluded_hidden_online_test_evidence',status='not_primary',
    setting_evidence_class='primary_paper_core_settings_and_native_source_protocol',
    settings={'hidden':512,'embed':512,'lr':.001,'weight_decay':0,'dropout':.5,'input_dropout':0,'attention_dropout':0,
        'batch_size':10000,'feature_hops':4,'label_hops':4,'feature_channels':25,'label_channels':12,
        'feature_projection_layers':2,'task_layers':4,'transformer_heads':1},
    optimizer='Adam/nativeAMP',epochs=200,patience='epoch-best_epoch>50',
    seed_policy='Paper average5localpartitions; native local sourceorderedseeds1..5. Preserve partition axis; ten new optimizerseed comparisons are labeled extension',
    split_policy='Paper HGB30%publiclocal /70%hiddenonlineTEST; local24%TRAIN/6%VALID; native multitargetIMDB5labels',
    reference_metric='micro_F1_percent',reference_mean=69.17,reference_other_dispersion=.43,
    reference_dispersion_type='reported_plus_minus_type_not_verified_in_bounded_passage',reference_score_null_reason='',
    reported_secondary_metrics=[{'metric':'macro_F1_percent','mean':67.11,'reported_plus_minus':.25}],
    score_location='heldout_confirmation_two_new_tasks_recommendation_20261009_v1/retrieved/sehgnn_paper_correct_html_normalized_text.txt:367–375/Table3',
    source_script='sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/PROTOCOL.json',
    source_program='NativeSeHGNN sourcecommit e92bd37d0b803457339555684f139b4c8f3e160d',
    comparability_limits='Local SELECT/CAL or five-class custom role outcomes do not match hidden official score; excluded primary until full official evaluation available',published_vs_release_schedule='Source200epochs/stale>50; paper confirmsH512/Adam.001/WD0/channels25+12')

primary=[r for r in rows if r['roster_role']=='primary_fixed_setting']
value={'schema':'closed-paper-author-setting-roster-v1','old_paper_datasets_excluded':sorted(OLD),
       'primary_arms':['native_M1','original_boundary_GNNM','genuine_ordinary_I4'],
       'primary_fixed_setting_cells':len(primary),'separate_source_horizon_cells':4,
       'hidden_online_evidence_cells':2,'cells':rows,'HPO':False,'scientific_execution_enabled':False,
       'scope':'Setting/protocol audit only; preparation_pending is not an executable/source-parity certificate'}
(OUT/'paper_setting_roster.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
with (OUT/'paper_setting_roster.csv').open('w',newline='') as stream:
    fields=sorted({k for r in rows for k in r});writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for r in rows:
        writer.writerow({k:json.dumps(v,separators=(',',':'),ensure_ascii=False) if isinstance(v,(dict,list)) else '' if v is None else v for k,v in r.items()})
sources=[S/'polyformer_code/node_classification/node_classification.sh',S/'polyformer_code/node_classification/training.py',
         S/'polyformer_paper.txt',S/'polynormer_code/run.sh',S/'polynormer_code/main.py',S/'polynormer_code/data_utils.py',S/'polynormer_paper.txt',
         PHASE/'hyperparameter_explanation_20261013_v1/AUTHOR_PROTOCOLS.json',
         PHASE/'training_pipeline_v2_20261012/preparation/ogb_native_source_readonly_root_20261012_v1/source/README.md',
         PHASE/'graph_link_competing_family_source_20261003_v1/excerpts/ncn__README.md.txt',
         PHASE/'link_covariance_idea_v1/sources/buddy_readme.md',
         PHASE/'cached_structure_link_ensemble_feasibility_v1/sources/buddy_src_runners_run.py',
         PHASE/'cached_structure_link_ensemble_feasibility_v1/sources/buddy_src_data.py',
         PHASE/'heldout_confirmation_two_new_tasks_recommendation_20261009_v1/retrieved/sehgnn_paper_correct_html_normalized_text.txt',
         PHASE/'sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/PROTOCOL.json']
sources.append(PHASE/'reference_wrapper_redesign_20261013_v1/references/ogb_paper.html')
receipts=[]
for p in sources:
    raw=p.read_bytes();receipts.append({'path':str(p.relative_to(PHASE)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
(OUT/'SOURCE_AUDIT_RECEIPT.json').write_text(json.dumps({'sources':receipts,'paper_score_cells_not_new_outcomes':True,
    'primary_count':len(primary),'total_rows':len(rows),'remote_access':False,'fits':0,'models':0,'HPO':False},indent=2,sort_keys=True)+'\n')
print(json.dumps({'primary':len(primary),'source_horizon_reference':4,'excluded_hidden_online':2,'rows':len(rows)}))

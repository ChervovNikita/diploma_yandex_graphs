"""Build a source-only, bounded literature evidence packet from saved receipts."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

R = Path(__file__).resolve().parent
B = R.parent

def write(name, value):
    (R / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

bindings = json.loads((R / 'INPUT_BINDINGS.json').read_text())
for rel in ['continuous_method_gap_search_v1/round15_graph_route_initialization/REPORT.md',
            'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2/REPORT.md']:
    p = B / rel
    if not any(x['path'] == str(p.resolve()) for x in bindings['input_files']):
        bindings['input_files'].append({'path': str(p.resolve()), 'sha256': digest(p),
            'bytes': p.stat().st_size, 'reuse_role': 'Frozen candidate operation/custody description; not a primary-paper reread'})
write('INPUT_BINDINGS.json', bindings)

old_morgan = B / 'coordinate_source_independent_review_v1/morgan_primary_resolution_v1'
old_fagel = B / 'coordinate_source_independent_review_v1/graph_tangent_initialization_review_v1'
m = json.loads((old_morgan / 'PAPER_CONCLUSIONS.json').read_text())['entries'][0]
f = next(x for x in json.loads((old_fagel / 'PAPER_CONCLUSIONS.json').read_text())['entries'] if x['key'] == 'fagel')
candidate = {
    'warm_state': 'Identical supervised warm predictor with identity active route-private factors; one shared checkpoint/optimizer/RNG branch source.',
    'error_signal': 'Masked TRAIN CE logit cotangent s.',
    'graph_operation': 'Fixed degree-3 Bernstein graph bank H_b, acting on graph-injected cotangents rather than input features; h_b = J^T M B^T H_b B s.',
    'parameter_operation': 'VJPs into the active route-private R/S slice; project h_b orthogonally to common gradient g, center across routes, apply common Frobenius cap; d_b = -g - lambda*t_b.',
    'safeguards': 'Qualified VJP/JVP/functional/null checks and bounded finite every-route/pooled TRAIN CE Armijo safeguards; common-only or unchanged warm fallback retained.',
    'continuation_pooling': 'Ordinary per-member CE continuation; fixed mean raw-logit pooling.',
    'arms': ['graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy'],
    'scope': 'Frozen operation comparison only; no numerical execution, utility/originality claim, or protocol amendment.'}
write('REUSED_CONCLUSIONS.json', {'status': 'saved_scoped_conclusions_reused_no_unchanged_primary_or_code_reread',
    'entries': [m, f], 'frozen_candidate_operation': candidate,
    'other_ancestry': 'Index_v10/round15 already attributes PreGS warm supervised transfer, BernNet spectral bank, C&S graph-correlated supervised errors, TabM private factors/member training, StarSSE warm-copy branching, and graph mixture prior work. These unchanged primaries were not reread.'})

batch_names = ['retrieval_batch1', 'retrieval_batch2', 'adjacent_discovery',
               'retrieval_batch3_identified_asset_and_adjacent', 'adjacent_primary_retrieval']
records = [x for name in batch_names for x in json.loads((R / 'discovery' / (name + '.json')).read_text())]
write('RETRIEVAL_LOG.json', {'schema_version': 1, 'new_request_count': len(records),
    'automatic_retries_per_request': 0, 'per_request_timeout_seconds': 18,
    'target_request_count_including_metadata_author_asset': 14,
    'target_full_paper_access_result': 'Both unresolved after bounded public retrieval; no further target retry in this packet.',
    'prior_failure_logs_preserved_by_reference': [str((old_morgan/'RETRIEVAL_LOG.json').resolve()), str((old_fagel/'RETRIEVAL_LOG.json').resolve())],
    'prior_failures_reused': ['MORGAN canonical OJS article/download RemoteDisconnected; OJS galley/OAI RemoteDisconnected; OpenAlex content PDF/XML HTTP401.',
                            'FAGEL full chapter previously unavailable beyond subscription preview; narrow prior FAGEL preprint discovery did not establish an ID.'],
    'records': records,
    'response_interpretations': [
        {'request_id':'morgan_pdf_fresh','classification':'transport failure','finding':'RemoteDisconnected; no PDF bytes retrieved.'},
        {'request_id':'fagel_pdf_fresh','classification':'HTTP200 subscription HTML, not PDF','finding':'HTML title matches FAGEL, citation PDF points back to same route, hasAccess=N/Open Access=N, Access this chapter purchase heading.'},
        {'request_id':'fagel_registered_pdf','classification':'HTTP200 subscription HTML, not PDF','finding':'Crossref registered PDF route also yields subscription preview; no access controls bypassed.'},
        {'request_id':'morgan_tree_latest','classification':'unchanged author inventory','finding':'HEAD a31af5d9ac4c838cb1dfbeb4131852bafe86e6c6, complete tree, no paper PDF; code/data not opened.'},
        {'request_id':'fagel_tree_latest','classification':'unchanged author inventory','finding':'HEAD 02bdc6dc745a84bc7f947edaa98336adc2e63e28; identified FAGEL_framework.pdf is standalone one-page diagram, not chapter.'},
        {'request_id':'morgan_publications_fresh','classification':'unchanged author publication HTML','finding':'SHA256 matches saved page: 63c0d8232d4d03c924a88bf4206f90eec8b27cc8538877d9f4e883377a00d220. MORGAN entry has no identified paper link; unchanged prose not reread.'},
        {'request_id':'fagel_arxiv_exact_title','classification':'narrow metadata discovery','finding':'Exact title search reports no results. No exhaustive absence claim or verified arXiv ID.'},
        {'request_id':'morgan_openalex','classification':'OA metadata only','finding':'Only identified OA PDF is inaccessible canonical OJS route; no repository full text reported.'},
        {'request_id':'fagel_openalex','classification':'OA metadata only','finding':'Closed chapter and no OA location reported; this is an index result, not global absence proof.'},
        {'request_id':'morgan_releases','classification':'public release inventory','finding':'Empty list.'},
        {'request_id':'fagel_releases','classification':'public release inventory','finding':'Empty list.'},
    ]})

pages = json.loads((R / 'sources/local_boost_pages.json').read_text())
passages = []
def excerpt(pid, page, section, start, end, finding):
    t = pages[page-1]['text']
    a = t.index(start)
    z = t.index(end, a) if end else len(t)
    passages.append({'id':pid, 'canonical_id':'DOI:10.1007/s10994-026-07041-x',
        'source':'sources/local_boost.pdf', 'source_sha256':digest(R/'sources/local_boost.pdf'),
        'pdf_page':page, 'printed_page':page, 'section':section,
        'extraction':'pypdf exact text; linebreak/hyphenation artifacts retained',
        'text_start_offset':a, 'text_end_offset':z, 'exact_text':t[a:z], 'supports':finding})
excerpt('b3f_local_training',5,'2.3 Gated Linear Networks','Each neuron in a GLN solves','Given a neuron','Local independently trained neuron problems; distinct training organization.')
excerpt('b3f_prototype_gating',5,'2.3 Gated Linear Networks','In this work, we rely','Recently, a similar','Nearest training-context prototype gating; no graph-spectrum routing claimed.')
excerpt('b3f_class_modules',6,'3 B3F-GNN','Each layer consists','In the following,','Class-specific one-vs-rest units; error information flows to next BU; local classifier committee.')
excerpt('b3f_concatenation',7,'3 B3F-GNN','The outputs of the various','Gating Mechanism','Learned within-BU linear combination; outputs concatenated into next layer; distinct from fixed pooled ensemble logits.')
excerpt('b3f_weighting_prose',8,'3.1 Input Weighting by Boosting','The boosting strategies','Algorithm 1','Sequential observation weighting, initialized uniformly over TRAIN samples.')
excerpt('b3f_prototypes_prose',8,'3.2 Error-based Prototypes Selection','The second proposed','1 3','Top-loss TRAIN sample contexts bias prototypes; first BU prototypes uniformly random.')
excerpt('b3f_incremental_claim',15,'5 Boosting Analysis continuation','Moreover, the architecture','From this perspective,','Author claims dynamic BU addition without retraining; operational semantics not source-qualified here.')
passages += [
    {'id':'b3f_algorithm1', 'canonical_id':'DOI:10.1007/s10994-026-07041-x', 'source':'sources/local_boost.pdf', 'pdf_page':8,
     'section':'Algorithm1', 'evidence_image':'evidence/local_boost-08.png',
     'extraction':'Manual transcription of mathematical algorithm from visual page; pypdf did not extract algorithm body',
     'transcribed_operation':['Initialize beta_j^(1,l)=1/n_s.', 'For i=1..n_q: train BU phi_(i,l)^(c)(beta^(i,l) .* L_(l-1)^(c), Z).',
       'err_(i,l)^(c) = sum_j beta_j^(i,l) I(y_j != phi_(i,j)^(c)(l_j,z_j)) / sum_j beta_j^(i,l).',
       'alpha_(i,l)^(c) = log((1-err_(i,l)^(c))/err_(i,l)^(c)).',
       'beta_j^(i+1,l) = beta_j^(i,l) * exp(alpha_(i,l)^(c) I(y_j != phi_(i,j)^(c)(l_j,z_j))).'],
     'bounded_finding':'Printed training call scales input representation beta .* L; the exact native loss-weighting/normalization/clipping policy is unresolved without author code. No native reproduction qualified.'},
    {'id':'b3f_algorithm2','canonical_id':'DOI:10.1007/s10994-026-07041-x','source':'sources/local_boost.pdf','pdf_page':9,
     'section':'Algorithm2','evidence_image':'evidence/local_boost-09.png',
     'extraction':'Manual transcription of mathematical algorithm from visual page; pypdf did not extract algorithm body',
     'transcribed_operation':['For i=1..n_q: train BU phi_(i,l)^(c)(L_(l-1)^(c), Z) (uniform weights if reweighting unused).',
       'Compute per-sample loss ell_r = Loss(y_r, phi_(i,r)^(c)(l_r,z_r)).',
       'Take I_(i,l) as indices of top n_k samples by ell_r; define pi on I_(i,l) by normalizing ell_r.',
       'For each neuron j=1..m_c: draw k contexts z_h ~ pi; set P_(j,i,l)^(c) = [z_1,...,z_k].'],
     'bounded_finding':'Prose/Fig1 describe previous-BU error guiding later BUs. Printed Algorithm2 updates the same i-index prototype matrix after training i. The receiving-module index and re-training/order semantics require a native-source check.'},
    {'id':'fagel_author_diagram','canonical_id':'DOI:10.1007/978-3-032-37657-2_35','source':'sources/fagel_framework.pdf',
     'source_sha256':digest(R/'sources/fagel_framework.pdf'),'pdf_page':1,'source_role':'Pinned official author standalone framework diagram, not full primary chapter',
     'evidence_image':'evidence/fagel_framework.png','exact_visible_labels':['Stage 1 GNN','Stage 2 GNN','Batch i','(s=1,k=1)','(s=1,k=2)','Validation Accuracy','Selected GNN_1','Selected GNN_2','Selected GNN_K','Ensemble','Cross-Entropy Loss'],
     'supports':'Diagram shows stage1 fixed sampling, stage2 changing k, validation-selected GNN snapshots and ensemble CE. It supplies no chapter proof/derivation, spectral expert initialization or cotangent VJP text.'}
]
write('evidence/PRIMARY_PASSAGES.json', {'attribution':'Pasa, Frazzetto, Navarin, Sperduti, Machine Learning115:111 (2026), DOI10.1007/s10994-026-07041-x; publisher metadata indicates CC BY4.0. FAGEL diagram from official author repository at pinned commit.', 'passages':passages})

write('READ_SCOPES.json', {'new_scoped_full_paper_reads':1, 'new_scoped_author_diagram_reads':1,
    'conservative_total_new_primary_source_reads':2, 'authorized_maximum':2,
    'unchanged_primary_or_author_implementation_rereads':0,
    'entries':[
      {'key':'morgan','read_status':'full_primary_not_retrieved','new_method_passages_read':[], 'reuse':'Bound saved primary-unavailable/scoped author-source conclusions; fresh metadata/file inventories only.'},
      {'key':'fagel','read_status':'full_primary_not_retrieved_new_author_diagram_read','pdf_pages':[1], 'scope':'Complete visually inspected standalone framework figure; no extracted text available; identity from pinned official author asset path. Publisher no-access marker inspected, old abstract/code conclusions reused.', 'limitations':'No full chapter method/proof/results read.'},
      {'key':'local_boost','read_status':'new_scoped_accessible_primary_read','canonical_id':'DOI:10.1007/s10994-026-07041-x',
       'full_text_retrieved':True,'pdf_pages_count':20,'complete_text_pages_read':[4,5,6,7,8],
       'selected_text_scopes':[{'page':1,'scope':'title/authors/DOI/date/abstract and start of introduction'}, {'page':9,'scope':'Algorithm2; visual context incidentally displays experimental setup/table but no performance conclusion inferred'}, {'page':15,'scope':'any-time incremental BU-addition author claim; not empirical/cost validation'}],
       'visual_pages_inspected':[1,5,6,7,8,9], 'page_headers_inventoried':'All20 header prefixes only to locate relevant sections; not read as complete pages.',
       'exact_method_scope':'Sections2.3/2.4 and3/3.1/3.2, Eq1/2, Fig1, Algorithms1/2. Background definitions on page4; no full results/proof/related-work/code read.',
       'author_implementation_read':False,'native_reproduction_qualified':False,
       'extraction_note':'FullPDF and page text saved; retrieval/extraction do not imply full-paper/every-page reading. Raster algorithm bodies read visually.'}],
    'discovery_only': ['Crossref/OpenAlex canonical/OA metadata', 'GitHub HEAD file inventories/releases', 'MORGAN unchanged publications-page link check', 'FAGEL exact-title preprint search', 'Boosting-title search results other than selected B3F-GNN']})

conclusions = [
 {'key':'morgan','canonical_id':m['canonical_id'],'canonical_doi':m['canonical_doi'],'verified_title':m['verified_title'], 'verified_authors':m['verified_authors'],
  'read_status':'full_primary_unavailable_saved_scoped_source_conclusion_reused','publication_date':'2026-03-14','primary_url':m['primary_url'], 'registered_pdf_url':m['registered_primary_pdf_url'], 'new_full_primary_read':False,
  'saved_takeaway':m['saved_takeaway'],'mechanism_comparison':m['relationship_to_GNNM_and_extension'],
  'durable_conclusion':'Bounded full-primary access closure unsuccessful. No exact full-paper expert initialization/training/spectral-specialization closure. Preserve existing source delta as scoped implementation evidence; missing access supplies no novelty evidence.',
  'access_evidence':'Canonical PDF fresh failure; current unchanged author HEAD/tree and unchanged publications page; empty releases; OA index points only to inaccessible publisher.',
  'decisive_next_step':'Resume only on an identified accessible full-primary/version/author manuscript or named newly available passage; no repeated bounded URL loop.'},
 {'key':'fagel','canonical_id':f['canonical_id'],'canonical_doi':f['canonical_doi'],'verified_title':f['verified_title'],'verified_authors':['Jiajun Shen','Yufei Jin','Xingquan Zhu'],
  'read_status':'full_primary_unavailable_saved_code_conclusion_reused_new_official_diagram_read','first_online_date':'2026-09-08','bibliographic_year_discrepancy':f['bibliographic_year_discrepancy'], 'primary_url':f['primary_url'],'new_full_primary_read':False,
  'saved_takeaway':f['saved_takeaway'],'new_supplementary_finding':'Pinned one-page author diagram confirms staged fixed/changed sampling, validation-selected full GNN snapshots and ensemble CE; it contains no full chapter or theorem/derivation.',
  'mechanism_comparison':f['relationship_to_GNNM_and_extension'], 'durable_conclusion':'Full-primary overlap and prose-code discrepancy unresolved. The author diagram strengthens stage/snapshot attribution only; it does not establish absence of spectral or gradient initialization in the chapter.',
  'decisive_next_step':'Resume only on accessible chapter/manuscript or named new source. Any native FAGEL control needs separate versioned paper/code reconciliation; no control or pooling grid added here.'},
 {'key':'local_boost','canonical_id':'DOI:10.1007/s10994-026-07041-x','canonical_doi':'10.1007/s10994-026-07041-x',
  'verified_title':'Local Learning with Boosting-based Backpropagation-Free Graph Neural Networks','verified_authors':['Luca Pasa','Paolo Frazzetto','Nicolò Navarin','Alessandro Sperduti'],
  'version':'Published version of record, Machine Learning (2026)115:111; PDF DOI/title/authors verified','publication_date_primary':'2026-05-02','metadata_date_discrepancy':'Crossref/OpenAlex give May1; primary page1 explicitly Published online2May2026.',
  'primary_url':'https://link.springer.com/content/pdf/10.1007/s10994-026-07041-x.pdf','source_sha256':digest(R/'sources/local_boost.pdf'),
  'read_status':'new_scoped_accessible_primary_read','new_full_text_available_scoped_paper_read':True,'exact_read_scope':'READ_SCOPES.json local_boost entry',
  'saved_takeaway':'B3F-GNN uses local gated graph neurons grouped in class-specific one-vs-rest BoostingUnits. BU classifiers use learned linear within-unit combination; layer output concatenates units. Error-guided sequential observation reweighting and top-loss prototype contexts specialize later modules. The first prototype set is uniform random; nearest-prototype gates select neuron parameters.',
  'relationship_to_frozen_candidate':'Distinct training-time/local architecture lead: sequential error-weighted module learning and prototype-gate placement, rather than one-time fixed graph-band CE cotangent VJPs into identity private factors of one warm predictor. No spectral bank, common-descent orthogonal tangent/cap, functional/Armijo certificate, or fixed mean raw-logit reducer is specified in inspected B3F-GNN method passages.',
  'bounded_absence_scope':'This statement is limited to inspected method passages and is not an exhaustive full-paper novelty search.',
  'attributed_prior_vs_open_gap':'Error-conditioned graph-module specialization is established ancestry. A separately derived GNNM training-time sequential error weighting/prototype specialization operation would need its own definition, native-source reconciliation, fair paid-cost/selection contract and held-out tests; it cannot inherit this paper efficacy claims or rescue the frozen initializer.',
  'distinctness_evidence':'No matching canonical ID/title in index_v10; Eq1/2, Fig1 and Algorithms1/2 directly establish different representation/training/gating operations. This is an actual scoped primary read after discovery, not search-result inference.',
  'limitations':['Native author code not inspected/executed.', 'Algorithm1 prints beta .* L input scaling; weighted-loss and clipping details unresolved.', 'Algorithm2 same-i prototype assignment after train conflicts with later-BU narrative; receiving-module index/order unresolved.', 'Dynamic BU-addition is an author claim, not an implementation/cost/certificate verified here.', 'No reported benchmark benefit or efficiency transferred to GNNM/admitted graphs.'],
  'decisive_next_step':'Retain as one distinct adjacent source-only lead. If pursued later, inspect pinned native training/source to settle weighting/prototype timing and final classifier construction before creating a separate methodological packet.'}
]
write('PAPER_CONCLUSIONS.json', {'schema_version':1,'purpose':'Bounded unresolved closest-primary access attempt plus one genuinely distinct accessible adjacent methodological lead.',
    'new_scoped_paper_reads':1,'new_scoped_author_diagram_reads':1,'conservative_primary_source_reads':2,
    'entries':conclusions, 'overall_status':'Both closest full primaries remain open; one distinct adjacent method verified within scoped reading. Utility/originality remain unestablished; no novelty from missing access.'})

inventory=[]
for x in records:
    path=x.get('saved_path') or x.get('failure_body_path')
    if path:
        inventory.append({'retrieval_id':x['id'],'path':path,'sha256':digest(R/path),'bytes':(R/path).stat().st_size,
                          'requested_url':x['requested_url'],'final_url':x.get('final_url'),'http_status':x.get('http_status'),'pdf_magic':x.get('pdf_magic',False),
                          'role':'discovery metadata/response' if path.startswith('discovery/') else ('primary_scoped_read' if path=='sources/local_boost.pdf' else 'author_diagram' if path=='sources/fagel_framework.pdf' else 'non_pdf_publisher_access_response')})
write('SOURCE_INVENTORY.json', {'retrieved_response_files':inventory,'derived_extractions':['sources/fagel_framework.txt','sources/local_boost.txt','sources/local_boost_pages.json'], 'scope_note':'Full retrieval/extraction is distinct from actual scoped reading; read scope governs conclusions.'})

matrix = [
 ['identical supervised warm predictor / identity private factors','Frozen candidate operation','MORGAN saved source: fresh expert MLP/gate; full primary unresolved','FAGEL saved source: ordinary full GCN trajectory; full primary unresolved','B3F: local class-specific BF neurons/units; no same-warm identity-factor step in inspected method'],
 ['supervised error signal','masked TRAIN CE logit cotangent','MORGAN saved code filters X, not an explicit TRAIN residual VJP','FAGEL CE/snapshot training, exact chapter signal unresolved','B3F: misclassification observation weights and per-sample top losses'],
 ['graph/specialization operation','fixed cubic Bernstein bank on cotangent','MORGAN learned eigenvector bands on node features; gates mean eigenvalues','FAGEL varied hop/fanout samplers / selected snapshots','B3F: graph convolution + nearest TRAIN-context prototype gate; error-biased prototype sampling'],
 ['parameter expansion','active private R/S VJP; centered g-orthogonal/capped tangents plus common descent','No such initializer in saved inspected constructors/training; no full-paper absence claim','No such initializer in saved inspected source; no full-paper absence claim','Sequential module training / prototype placement rather than one-time factor tangent'],
 ['finite safeguard','VJP/JVP/null/functional checks + bounded route/pooled Armijo fallback','Primary unavailable; do not infer','Primary unavailable; do not infer','None in inspected B3F method; native code not qualified'],
 ['continuation / combination','ordinary member CE; fixed mean raw logits','MORGAN saved code continuous band/filter/gate and fused prediction','FAGEL learned weighted residual raw-logit aggregation; author diagram ensemble CE','B3F learned within-BU linear combination; layer output concatenation'],
 ['claim boundary','utility/originality unestablished','full paper open; implementation-scoped delta only','full paper open; implementation/diagram-scoped delta only','distinct adjacent verified method; transfer utility/originality unestablished'],
]
with (R/'MECHANISM_COMPARISON.csv').open('w',newline='') as stream:
    writer=csv.writer(stream); writer.writerow(['operation','frozen_candidate','morgan','fagel','adjacent_b3f']);writer.writerows(matrix)

report = '''# Closest graph-initialization primary closure v1

## Result

Both requested full primaries remain inaccessible after a bounded legitimate public retrieval pass. Their exact full-paper expert initialization, training, spectral specialization and possible cotangent/VJP overlap remain unresolved. Missing access is not novelty evidence.

One accessible adjacent primary was actually scoped-read: **Local Learning with Boosting-based Backpropagation-Free Graph Neural Networks (B³F-GNN)**, Pasa, Frazzetto, Navarin and Sperduti, DOI **10.1007/s10994-026-07041-x**, Machine Learning115:111 (2026). Its training-time local boosting and prototype gating are a genuinely different operation from the frozen one-time graph-band private-factor initializer. This identifies one future source-only methodological lead; it does not establish GNNM benefit or rescue the frozen initializer.

## Inputs and bounded access

Index_v10 and the saved MORGAN/FAGEL scope, conclusions and retrieval failures were consulted first and hash-bound in INPUT_BINDINGS.json. Unchanged papers and author implementation files were not reread. The frozen comparison is bound to round15 and round17v2 descriptions; no manuscript/science/model/GPU/SSH work was performed.

Nineteen new public requests were retained with URL, time, status, content type, bytes and hash or transport error. Fourteen concern the targets, including metadata, public author inventories and one identified author asset; five concern adjacent discovery/retrieval. Every request has an18-second timeout and zero automatic retries. The target retrieval stops here.

**MORGAN** — canonical DOI10.1609/aaai.v40i28.39553; Lihui Liu and Yuchen Yan; version of record published14March2026. The registered OJS PDF again returned RemoteDisconnected. OpenAlex identifies only that PDF as OA. Author releases are empty; the complete current repo tree remains at saved commit a31af5d9ac4c838cb1dfbeb4131852bafe86e6c6 with no paper PDF. The author publications HTML is byte-identical to the saved page and provides no identified linked MORGAN manuscript. Prior OJS and OpenAlex401 failures remain preserved through their bound receipts.

**FAGEL** — canonical DOI10.1007/978-3-032-37657-2_35; Jiajun Shen, Yufei Jin and Xingquan Zhu; first online8September2026, while publisher citation/copyright year2027 remains a preserved discrepancy. Both PDF routes returned HTTP200 subscription HTML with hasAccess=N, not PDF bytes. OpenAlex reports no OA location; the narrow exact-title arXiv query reports no result, without proving global absence. Public releases are empty; repo HEAD remains saved commit02bdc6dc745a84bc7f947edaa98336adc2e63e28.

The identified pinned FAGEL_framework.pdf is **one standalone diagram**, not the chapter. Its entire page was visually inspected. It shows fixed stage1 sampling, changed stage2 sampling, validation-selected GNN snapshots and ensemble cross-entropy. This strengthens the existing stage/snapshot attribution but supplies no method derivation, proof or full-paper spectral/gradient closure. The image is supplementary author evidence, and is conservatively counted as a new primary-source read.

## Frozen operation comparison

The candidate branches from one identical supervised warm function with identity private factors. Fixed cubic Bernstein operators act on masked TRAIN CE logit cotangents; parameter VJPs enter active route-private R/S factors. Directions are centered, orthogonal to the common gradient and Frobenius capped, with a common descent component, functional/JVP/null qualification and bounded finite route/pooled Armijo safeguards. Ordinary member CE continuation and fixed mean raw-logit pooling follow. The five arms remain graph/common_only/random_tangent/topology_permuted/warm_copy.

Saved MORGAN source evidence instead learns eigenvector frequency-band operators, gates mean eigenvalues, filters input features X and trains a fused predictor. Saved FAGEL source evidence instead trains a full GCN trajectory under changed neighborhood sampling, keeps validation-selected snapshots and applies learned weighted residual fusion. Neither saved inspected source explicitly supplies the complete candidate initializer. These are **bounded implementation findings**, not absence conclusions about inaccessible primary papers. The saved FAGEL prose/code discrepancies remain unresolved.

MECHANISM_COMPARISON.csv and REUSED_CONCLUSIONS.json retain the operation-by-operation comparison and attributed ancestry. PreGS, BernNet, C&S, TabM, StarSSE and other already-read graph mixtures remain acknowledged through existing conclusions; no unchanged primary reread enlarges this bibliography.

## Distinct accessible adjacent lead

Discovery initially used title/DOI metadata; reading was admitted only after the publisher returned an actual20-page PDF, and page1 verified title, all authors and DOI. Crossref/OpenAlex indicate May1, while the primary explicitly states **Published online2May2026**; both are preserved. The canonical ID/title has no matching paper record in index_v10.

The actual scoped reading covers the gated-local learning background, BF-GCN/BF-GraphConv equations, the BoostingUnit architecture, nearest-prototype gating, Figure1, and Algorithms1/2. A selected page15 paragraph about incremental unit addition was read as an author claim. Primary passages have exact text offsets/page/section provenance, and raster algorithms were read from saved page images. Full extraction or page-header inventory does not imply a full-paper read. The budget is one new scoped paper plus one author diagram, conservatively two new primary-source reads.

B³F-GNN trains local neurons independently for one-vs-rest class tasks, groups them in learned linear classifier units, and concatenates unit outputs between graph layers. Two error-conditioned operations specialize sequential units:

1. Observation weights begin uniformly; weighted misclassification rates set log-odds coefficients and multiply weights for subsequent units.
2. Prototypes begin uniformly at random; subsequent selection draws training contexts from normalized losses among the highest-loss samples. Nearest-context prototype gates select neuron parameter regions.

This changes local module training and gate geometry throughout a layered architecture. It does not implement the frozen one-time fixed spectral cotangent pullback/private-factor tangent operation in the inspected passages. The published error-conditioned specialization must be attributed if it informs a future proposal.

Two native-operation questions remain explicit. Algorithm1 prints a train call with **beta .* L input scaling**, so a weighted-CE objective must not be inferred without code. Algorithm2 prints prototype assignment to the **same i-index unit after training i**, while the prose/diagram describe previous-unit errors steering later units. Receiving-unit index, re-training/order and clipping/normalization details need pinned native-source reconciliation. The incremental-addition paragraph is not an implementation, runtime or downstream-dimension certificate. Author code was neither inspected nor run; no native reproduction is qualified.

## Durable conclusion and handoff

The closest full-primary overlap questions stay open. A named accessible manuscript/version or newly available exact passage is the reason to revisit them; repeated URL retries are not. Retain B³F-GNN as the single distinct adjacent verified reading. Pursuing training-time sequential error weighting/prototype specialization would require a separately defined method and custody/paid-cost/selection contract, followed by native-source reconciliation and prospective tests. No benefit, efficiency or originality claim follows from this source packet.

Canonical identities and precise limits: PAPER_CONCLUSIONS.json. Exact read boundaries: READ_SCOPES.json. Inspected passages: evidence/PRIMARY_PASSAGES.json. Reused conclusions: REUSED_CONCLUSIONS.json. New/prior access failures: RETRIEVAL_LOG.json plus INPUT_BINDINGS.json. Source bytes: SOURCE_INVENTORY.json. File integrity: MANIFEST.sha256.
'''
(R/'REPORT.md').write_text(report)
write('PACKET_SUMMARY.json', {'created_utc':datetime.now(timezone.utc).isoformat(), 'status':'bounded_access_attempt_complete_both_full_primaries_open_distinct_adjacent_scoped_read_complete',
    'target_full_primary_reads':0,'new_adjacent_scoped_paper_reads':1,'new_author_diagram_reads':1,'conservative_total_primary_source_reads':2,
    'retrieval_count':len(records),'new_adjacent_canonical_id':'DOI:10.1007/s10994-026-07041-x',
    'no_unchanged_primary_reread':True,'no_unchanged_author_code_reread':True,'no_scientific_execution':True,'no_manuscript_edit':True,'no_memory_index_mutation':True,
    'utility_originality':'unestablished', 'next_action':'Preserve targets as full-primary open; one distinct adjacent local-boosting/prototype-gating lead retained with native algorithm ambiguities.'})

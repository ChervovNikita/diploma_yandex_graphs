"""Build source-literature metadata only. Does not import or run any model."""
from pathlib import Path
import ast, datetime, hashlib, json, re, xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
ROOT = P.parent
UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
NS = {'a': 'http://www.w3.org/2005/Atom'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ref(path, base=None):
    return {'path': str(path.relative_to(base)) if base else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}

def save(name, obj):
    (P / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

indexdir = ROOT / 'literature_memory/index_v40'
idx = json.loads((indexdir / 'LITERATURE_INDEX.json').read_text())
groups = idx['canonical_identifier_normalization']['groups']
known = {g['normalized_identifier'] for g in groups}
index_ref = ref(indexdir / 'LITERATURE_INDEX.json', ROOT)
assert index_ref['sha256'] == '9a451fedaaa11d5533f4255c061f49404e667f52d9c7e94629d6541861d4730e'
frozen_dirs = {
    'index_v40': indexdir,
    'JF_V3': ROOT / 'graph_ncNC_structural_pattern_pilot_preparation_20261003_v3',
    'C64_source_plan': ROOT / 'ncnc_cardinality_single_comparator_source_plan_20261003_v1',
}
expected = {
    'index_v40': 'a7313f1b99dd266cf2b65198cff698d69a9a2131174b01a72e75eb7a88832afb',
    'JF_V3': 'fa7b2a7a2c6ec83362f3c820fb4f7ad5288e5cc9fb0ee5139614d6690f3f7f89',
    'C64_source_plan': 'aa70f5a4d89a394ef059a431c78d333af95e6ab1c0c4e8c8dbbd028ee4b8d2f9',
}
bindings = {}
for key, directory in frozen_dirs.items():
    assert sha(directory / 'MANIFEST.json') == expected[key]
    bindings[key] = {name: ref(directory / name, ROOT) for name in ['MANIFEST.json', 'SEAL.json']}
bindings['index_v40']['index'] = index_ref
bindings['JF_V3']['comparison_contract'] = ref(frozen_dirs['JF_V3'] / 'PILOT_PLAN.json', ROOT)
bindings['C64_source_plan']['comparison_contract'] = ref(frozen_dirs['C64_source_plan'] / 'DESIGN_CONTRACT.json', ROOT)
save('INPUT_BINDINGS.json', {'schema': 'recent_coherent_completion_scout_inputs_v1', 'UTC': UTC,
    'policy': 'Consult index conclusions first; only new identities/unresolved scopes extended.',
    'paper_records': len(idx['paper_records']),
    'normalized_paper_identities': sum(g['kind'] == 'paper' for g in groups),
    'normalized_software_identities': sum(g['kind'] != 'paper' for g in groups),
    'bindings': bindings, 'canonical_or_index_mutation': False})

versions = {'tgsbm': '2601.20646v1', 'sdg': '2601.23233v1', 'flex': '2507.11710v1',
            'lp_pretrain': '2508.04645v1', 'sagmm': '2511.13062v2', 'moe_ml': '2501.17557v1'}
all_entries, queries = {}, []
for f in sorted((P / 'discovery').glob('*.xml')):
    doc = ET.fromstring(f.read_bytes())
    nodes = doc.findall('a:entry', NS)
    total = doc.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
    receipt = json.loads((f.with_name(f.stem + '_RECEIPT.json')).read_text())
    queries.append({'key': f.stem, 'query': receipt['query'], 'returned': len(nodes),
                    'totalResults': int(total.text) if total is not None else None,
                    'retrieval': ref(f, P), 'receipt': ref(f.with_name(f.stem + '_RECEIPT.json'), P)})
    for e in nodes:
        version = e.findtext('a:id', namespaces=NS).rsplit('/', 1)[-1]
        aid = re.sub(r'v\d+$', '', version)
        ent = all_entries.setdefault(aid, {'normalized_id': 'arxiv:' + aid, 'versioned_id': version,
            'title': ' '.join(e.findtext('a:title', namespaces=NS).split()),
            'authors': [a.findtext('a:name', namespaces=NS) for a in e.findall('a:author', NS)],
            'published': e.findtext('a:published', namespaces=NS),
            'updated': e.findtext('a:updated', namespaces=NS),
            'DOI': e.findtext('{http://arxiv.org/schemas/atom}doi'),
            'abstract': ' '.join(e.findtext('a:summary', namespaces=NS).split()),
            'discovered_in': []})
        ent['discovered_in'].append(f.stem)

citations = []
for key, version in versions.items():
    aid = re.sub(r'v\d+$', '', version)
    ent = all_entries[aid]
    assert ent['normalized_id'] not in known
    assert ent['versioned_id'] == version
    source_name = key + ('.pdf' if key == 'moe_ml' else '.html')
    citations.append({'key': key, 'normalized_id': ent['normalized_id'], 'versioned_id': version,
        'title': ent['title'], 'authors': ent['authors'], 'first_submission_UTC': ent['published'],
        'version_update_UTC': ent['updated'], 'DOI_from_discovery': ent['DOI'],
        'primary_URL': 'https://arxiv.org/' + ('pdf/' if key == 'moe_ml' else 'html/') + version,
        'primary_source': ref(P / 'primary' / source_name, P),
        'metadata_sources': ['discovery/' + q + '.xml' for q in ent['discovered_in']],
        'absent_from_index_v40_normalized_groups': True})
save('CITATIONS.json', {'schema': 'scoped_primary_citations_v1', 'UTC': UTC, 'papers': citations})

ranges = {'tgsbm': [[31,43],[62,89],[98,101]], 'sdg': [[37,77],[130,138],[179,179]],
          'flex': [[19,66]], 'lp_pretrain': [[40,79]],
          'sagmm': [[16,17],[21,66],[80,80],[144,147],[153,171]]}
scopes, navs = [], []
for key, spans in ranges.items():
    blocks = json.loads((P / 'primary' / (key + '.blocks.json')).read_text())
    math = json.loads((P / 'primary' / (key + '.equations.json')).read_text())
    picked = [b for b in blocks if any(lo <= b['index'] <= hi for lo,hi in spans)]
    ordinals = sorted({o for b in picked for o in b['math_ordinals']})
    algorithms = json.loads((P / 'primary' / (key + '.algorithms.json')).read_text()) if key != 'lp_pretrain' else []
    scopes.append({'key': key, 'source': ref(P / 'primary' / (key + '.html'), P),
        'substantive_inclusive_block_ranges': spans,
        'substantive_blocks': [{'index': b['index'], 'tag': b['tag'], 'id': b['id'],
                                'heading_locator': b['heading_locator'], 'math_ordinals': b['math_ordinals']} for b in picked],
        'substantive_math_nodes': [math[o] for o in ordinals],
        'algorithms_read': [{'id': a['id'], 'scope': 'complete extracted algorithm text'} for a in algorithms],
        'algorithm_artifact': ref(P / 'primary' / (key + '.algorithms.json'), P) if algorithms else None,
        'incidental_navigation': 'All heading text and first130 characters of every displayed equation table were displayed before scope selection. Targeted keyword snippets also occurred. This is broader exposure than the substantive ranges.',
        'full_paper_read': False, 'proof_audit': False, 'source_code_read_or_reproduced': False})
    nav = {'key': key, 'heading_and_equation_navigation': [{'index': b['index'], 'id': b['id'],
         'text': b['text'] if b['tag'].startswith('h') else b['text'][:130]} for b in blocks if b['tag'].startswith('h') or b['tag'] == 'table']}
    terms = {'tgsbm':['inference','algorithm','test','mean'], 'sdg':['inference','algorithm','sampling'],
             'sagmm':['binary','loss','prediction']}.get(key, [])
    nav['targeted_keyword_snippets'] = [{'index': b['index'], 'id': b['id'], 'text': b['text'][:300]} for b in blocks if any(t in b['text'].lower() for t in terms)]
    navs.append(nav)
scopes.append({'key':'moe_ml', 'source':ref(P / 'primary/moe_ml.pdf', P),
    'substantive_pdf_pages': [3,4],
    'selected_page5_sections_read': ['Assessment Criteria and Goals','Experimental setting','Parameters setting','Baselines and Competing Methods'],
    'page5_incidental_exposure':'Tables1-2 and beginning of Results were displayed during whole-page text extraction, without a results audit.',
    'pdf_visual_pages': [3,4],
    'visual_artifacts': [ref(P / 'primary/moe_ml_method-03.png', P), ref(P / 'primary/moe_ml_method-04.png', P)],
    'previous_navigation':'Earlier keyword snippets and page/section locators across the extracted PDF were displayed before method selection; no uninspected page is certified.',
    'all_pages_extracted':14, 'full_paper_read':False, 'proof_audit':False, 'source_code_read_or_reproduced':False})
save('READ_SCOPES.json', {'schema':'exact_primary_read_scopes_v1','UTC':UTC,'papers':scopes})
save('NAVIGATION_SCOPES.json', {'schema':'incidental_html_navigation_exposure_v1','papers':navs,
    'note':'Deterministically records retained heading/equation and targeted snippets; not substantive-read certification. Raw HTML rg navigation was also displayed, sometimes truncated.'})

conclusions = [
 {'key':'tgsbm','classification':'shared_latent_structured_single',
  'verified_objective':'Eq15 conditional product of Bernoulli edge likelihoods over evaluated positives and sampled negatives; Eq16 negative ELBO with shared stick/membership/strength KL terms and optional features.',
  'verified_inference':'Training reparameterization established. Algorithm1 is training-only; actual posterior predictive readout unresolved.',
  'scientific_implication':'Shared node/community latents can induce marginal dependence although the conditional decoder factorizes. A structured single alternative remains relevant.',
  'overlap_limit':'No verified native NCNC residual support, TRAIN observation teacher, uniform independently served pattern heads, or frozen J/F readout.',
  'unresolved':['posterior mean versus sample averaging at evaluation','one-Gumbel printed binary relaxation','Eq9 posterior family versus Eq11 prior-adjusted membership logits','source implementation and matched benchmarking']},
 {'key':'sdg','classification':'conditional_sequence_diffusion_shared_single',
  'verified_objective':'Shifted full destination sequence reconstruction using squared cosine error plus last-position and intermediate-position ranking supervision; BCE and disclosed dataset-specific BPR settings.',
  'verified_inference':'Algorithm1 initializes one L-position Gaussian embedding sequence, runs reverse diffusion, and scores candidates; no separate-sequence averaging loop specified.',
  'scientific_implication':'Coherent context/sequence supervision is an alternative to multiple independently parameterized predictive heads.',
  'overlap_limit':'Embedding-sequence denoising is not a verified Bernoulli incidence-pattern likelihood. Temporal information and recurrence protocol are unmatched to static Collab.',
  'unresolved':['ELBO-variant theorem not proof-audited','printed dot/elementwise score terminology and reverse-step coefficients not implementation-reconciled','no calibration or multiple-sample prediction claim']},
 {'key':'flex','classification':'link_conditioned_structural_training_augmentation',
  'verified_objective':'Pretraining reconstruction/KL Eq2; generative target-centered KL penalty Eq7; Eq8 link BCE and generative min/max co-training. Original link label constrains generated subgraphs.',
  'verified_inference':'Algorithms1-2 specify SIG-VAE sampling for synthetic training subgraphs and node-aware decode. No explicit repeated generated-hypothesis predictive aggregation was established in this read.',
  'scientific_implication':'Whole link-conditioned subgraph generation and structurally varied supervision are existing operations.',
  'overlap_limit':'Generation/thresholding/changed training examples do not establish a matched native residual mixture serving contract.',
  'unresolved':['printed loss signs/equation references need native-source reconciliation','OOD-generation validity assertion not proof-audited','test-time readout details not fully scoped']},
 {'key':'lp_pretrain','classification':'pretrained_target_scoring_experts_and_logit_fusion',
  'verified_objective':'Separate node and structural branches use sampled-edge BCE Eq9; cluster-distance Gumbel-Softmax routing Eqs10-14. Frozen downstream experts are combined by graph-wide fitted weights in sigmoid(weighted logits), Eq16.',
  'verified_inference':'Zero-shot Eq15 nested sigmoid fusion; adapted Eq16 soft expert logit fusion using one graph-wide weight vector.',
  'scientific_implication':'Scoring-head specialization and cooperative expert reuse are established prior operations.',
  'overlap_limit':'No complete residual-pattern law or multiple graph completions were established.',
  'unresolved':['exact expert-output aggregation during pretraining less explicit than routing equations','source implementation and unmatched data/pretraining resource budget']},
 {'key':'sagmm','classification':'topology_context_gated_representation_or_projected_output_experts',
  'verified_objective':'Generic task loss on gated expert projections in Algorithm1; named importance/diversity auxiliaries. Exact LP loss and negative sampler were not established.',
  'verified_inference':'Thresholded SGA gate, heterogeneous GNN experts, weighted representation combination/output layer described in AppendixD; adaptive pruning. Exact LP pair decoder unresolved.',
  'scientific_implication':'Context/capacity and model routing can improve predictions without an established graph-pattern mixture.',
  'overlap_limit':'Architecture embedding fusion versus Algorithm1 projected-output sum not fully reconciled; node-classification BCE theorem cannot establish LP objective. No residual-pattern density.',
  'unresolved':['LP pair readout and task loss','LP negative sampling and complete official-role equivalence','aggregation order across architecture/algorithm','source implementation and protocol']},
 {'key':'moe_ml','classification':'heuristic_gated_frozen_target_score_experts',
  'verified_objective':'Eq5 target-triple BCE of Eq3 sigmoid(gate-weighted expert outputs). Experts pretrained independently and frozen while gate learns.',
  'verified_inference':'Reference dense softmax gate over heuristic vector; optional Eq4 top-K hard routing. No joint graph-incidence predictive density established.',
  'scientific_implication':'Heuristic-informed target-specific specialization and weighted score combination are prior.',
  'overlap_limit':'Multilayer context and cross-validation/equal random-negative protocol differ from frozen homogeneous Collab roles; sigmoid after score sum is not normalized whole-pattern averaging.',
  'unresolved':['expert-output scale/calibration without source code','uninspected expert implementations and cost appendix']},
]
save('PAPER_CONCLUSIONS.json', {'schema':'recent_coherent_completion_scoped_conclusions_v1','UTC':UTC,
    'papers':conclusions,'citations':'CITATIONS.json','scopes':'READ_SCOPES.json',
    'novelty_guarantee':False,'acceptance_prediction':False,'results_or_runtime_adopted':False})

controls = [
 {'id':'count_explanation','action':'Use prepared C64-D4 under same frozen support/teacher and official target-role comparison; separate density and observed TRAIN mask count diagnostics from target ranking.',
  'falsifier':'Qualified count-aware single matches/exceeds J target-ranking benefit under declared uncertainty; retained multiple heads are not required for that comparison.',
  'limit':'C64 is additive within count, not universal same-count structure.'},
 {'id':'capable_context_single','action':'If count control leaves a gap, require a capable shared-context conditional pattern single with same-count interactions, matched labels/support/information/decoder capacity and disclosed approximation/paid work.',
  'falsifier':'Capable single retains target benefit without distinct independently served completion heads.',
  'limit':'Scientific gap only; no new implementation or run is specified/adopted/authorized here.'},
 {'id':'fixed_bank_serving','action':'Keep J/F own/crossed/pooled soft-weight association diagnostics separate from C64-D4 versus actual-marginal M4 dependence intervention; paired finite draws and fixed bank.',
  'falsifier':'No supported target-ranking/score gain from association or D4 versus M4 under declared paired uncertainty.',
  'limit':'Neither diagnostic alone isolates auxiliary training effects; finite draw noise is not exact expected-score comparison.'},
]
save('CONCLUSIONS.json', {'schema':'recent_coherent_completion_scout_synthesis_v1','UTC':UTC,
    'attribution':'Mixture likelihood has retained GRAN ancestry; task-specific support/teacher/gradient/serving placement is the candidate delta.',
    'J_F_training_gradient_scope':'J responsibility is query-pattern-wide; F responsibility is member/slot-specific. Teacher-conditioned training assignment is absent at serving.',
    'teacher_semantics':'TRAIN observation membership only; zero is unobserved, not verified latent nonlink.',
    'actionable_controls':controls,'control_count':len(controls),
    'stronger_claims_unestablished':['latent graph ambiguity','calibration','necessity of multiple retained hypotheses','novel mixture principle','matched paper benchmark superiority'],
    'unresolved_metadata_leads_retained':idx['unresolved_primary_metadata_leads'],
    'new_access_attempt_for_prior_403_leads':False,
    'canonical_or_index_mutation':False,'training_or_model_execution':False})

selected = {re.sub(r'v\d+$','',v):k for k,v in versions.items()}
dispositions=[]
for aid,ent in sorted(all_entries.items()):
    row={k:ent[k] for k in ['normalized_id','versioned_id','title','discovered_in']}
    if aid in selected:
        row.update({'status':'new_identity_scoped_primary_method_read','key':selected[aid],'scope_reference':'READ_SCOPES.json'})
    elif ent['normalized_id'] in known:
        row.update({'status':'existing_index_conclusion_retained_no_primary_reread','mechanism_reexclusion':False})
    else:
        row.update({'status':'discovery_metadata_locator_only_no_method_read','mechanism_excluded':False,
                    'note':'Selection based on scope relevance/abstract locator only; title and abstract do not verify objective, inference, overlap, or absence.'})
    dispositions.append(row)
save('DISCOVERY_DISPOSITIONS.json', {'schema':'bounded_primary_discovery_dispositions_v1','UTC':UTC,
    'queries':queries,'returned_entry_occurrences':sum(q['returned'] for q in queries),
    'unique_discovered_arxiv_ids':len(all_entries),'dispositions':dispositions,
    'prior_locator_only_exclusions_retained_without_method_reread':idx.get('locator_only_exclusions'),
    'unresolved_primary_metadata_leads':idx['unresolved_primary_metadata_leads']})
save('SEARCH_LIMITS.json', {'schema':'bounded_literature_search_limits_v1','UTC':UTC,
    'provider':'Primary arXiv API plus primary arXiv HTML/PDF','queries':queries,
    'max_results_per_query':25,'start':0,'sort':'submittedDate descending',
    'scope_dates':'2025-01-01 through 2026-10-03; one 2025-only focused query',
    'pagination':'No result pagination; several totalResults exceed returned count.',
    'selected_new_method_reads':6,'systematic_review':False,'absence_proof':False,
    'limits':['arXiv-only fresh discovery misses publisher-only and other unindexed literature','all-field query matches may be incidental','latest metadata abstract is a locator, not objective evidence','prior blocked publisher/SSRN leads remain unresolved','source implementations and full proof/results audits unperformed']})
save('READ_ACCOUNTING.json', {'schema':'honest_read_and_execution_accounting_v1','UTC':UTC,
    'index_conclusions_consulted_before_new_primary_reads':True,
    'prior_index_paper_records':168,'prior_normalized_paper_identities':119,
    'new_paper_identities_with_scoped_primary_method_read':6,'HTML_method_reads':5,'PDF_method_reads':1,
    'PDF_pages_visually_inspected':[3,4],'full_papers_certified':0,'proof_audits':0,
    'author_implementations_inspected':0,'model_training_or_numerical_prediction_executed':False,
    'operations_performed':['stdlib primary metadata/HTML/PDF retrieval','deterministic extraction','bundled pypdf text extraction','Poppler rendering of source pages3-4','metadata hashing and JSON/AST validation'],
    'retrieval_is_not_read':True,'extraction_is_not_read':True,'navigation_is_not_full_method_read':True,
    'truncation_handling':'Combined read display was truncated; LP pretraining scope and SAGMM opening were then displayed separately. PDF page5 partial results exposure is incidental.',
    'executed_downloaded_content':False,'canonical_or_index_mutation':False,
    'source_skill':'PDF skill used for read-only source equation fidelity; no authored PDF artifact.',
    'scopes':'READ_SCOPES.json','incidental_HTML_navigation':'NAVIGATION_SCOPES.json'})

# Remove only extraction-generated bytecode in this fresh scout directory.
cache=P/'__pycache__'
if cache.exists():
    for f in cache.glob('extract_primary_html.cpython-*.pyc'):
        f.unlink()
    if not any(cache.iterdir()): cache.rmdir()

json_paths=sorted(f for f in P.rglob('*.json') if f.name not in ['MANIFEST.json','SEAL.json'])
for f in json_paths: json.loads(f.read_text())
scripts=sorted(P.glob('*.py'))
for f in scripts: ast.parse(f.read_text(),filename=str(f))
assert sha(indexdir/'LITERATURE_INDEX.json') == index_ref['sha256']
save('VERIFICATION.json', {'schema':'literature_metadata_custody_verification_v1','UTC':UTC,
    'JSON_payloads_parsed_before_verification':len(json_paths),'source_scripts_AST_parsed':len(scripts),
    'new_ids_absent_from_119_normalized_paper_groups':len(citations),
    'input_manifest_pins_checked':len(expected),'index_sha256_unchanged':index_ref['sha256'],
    'actionable_controls':len(controls),'scientific_execution':False,
    'verification_does_not_certify':['scientific correctness of uninspected proofs/code','matched performance','comprehensive search','novelty','runtime qualification']})
files=sorted(f for f in P.rglob('*') if f.is_file() and f.name not in ['MANIFEST.json','SEAL.json'])
payload=[ref(f,P) for f in files]
save('MANIFEST.json', {'schema':'recent_coherent_completion_scout_manifest_v1','UTC':UTC,
    'status':'SEALED_SCOPED_PRIMARY_READ_SOURCE_ONLY_NO_INDEX_INTEGRATION',
    'payload':payload,'payload_count':len(payload),'payload_bytes':sum(f['bytes'] for f in payload),
    'inputs':'INPUT_BINDINGS.json','read_scopes':'READ_SCOPES.json','report':'REPORT.md'})
for item in payload:
    f=P/item['path'];assert sha(f)==item['sha256'] and f.stat().st_size==item['bytes']
save('SEAL.json', {'schema':'recent_coherent_completion_scout_seal_v1','UTC':UTC,
    'status':'SCOPED_PRIMARY_SOURCE_PACKET_SEALED', 'manifest':ref(P/'MANIFEST.json',P),
    'payload_count':len(payload),'payload_bytes':sum(f['bytes'] for f in payload),
    'index_v40_sha256_unchanged':index_ref['sha256'],'scoped_new_papers':6,'actionable_controls':3,
    'training_or_model_execution':False,'canonical_or_index_edit':False,
    'report':ref(P/'REPORT.md',P),'read_scopes':ref(P/'READ_SCOPES.json',P),
    'verification':ref(P/'VERIFICATION.json',P)})
print(json.dumps({'payload_count':len(payload),'payload_bytes':sum(f['bytes'] for f in payload),
    'manifest_sha256':sha(P/'MANIFEST.json'),'seal_sha256':sha(P/'SEAL.json'),
    'report_sha256':sha(P/'REPORT.md'),'unique_discovery_ids':len(all_entries)},indent=2))

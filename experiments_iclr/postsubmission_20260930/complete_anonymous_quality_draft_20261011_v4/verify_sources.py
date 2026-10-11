"""Static source/custody audit only. No compilation or scientific execution."""
from pathlib import Path
import collections,csv,hashlib,json,re,sys
P=Path(__file__).resolve().parent
R=P.parent
V3=R/'complete_anonymous_quality_draft_20261011_v3'
A=P/'code_evidence/v3_archive'
checks=[]
def digest(p):
 b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def check(name,ok,detail=None):checks.append({'check':name,'pass':bool(ok),'detail':detail})
def uncomment(s):
 out=[]
 for line in s.splitlines(keepends=True):
  for i,c in enumerate(line):
   if c=='%' and (len(line[:i])-len(line[:i].rstrip('\\')))%2==0:
    line=line[:i]+'\n';break
  out.append(line)
 return ''.join(out)
def escaped(s,i):return (len(s[:i])-len(s[:i].rstrip('\\')))%2==1
def braces(s):
 n=0
 for i,c in enumerate(s):
  if c in '{}' and not escaped(s,i):
   n+=1 if c=='{' else -1
   if n<0:return False
 return n==0
m=json.loads((V3/'MANIFEST.json').read_text())
parent_ok=digest(V3/'MANIFEST.json')['sha256']=='022c051928b9bf585dacef3fd4c27aa23b55ba6b3d0816220e85d5c398ba272c' and all(digest(V3/r['path'])=={k:r[k] for k in ['bytes','sha256']} for r in m['files']) and json.loads((V3/'SEAL.json').read_text())['manifest_sha256']==digest(V3/'MANIFEST.json')['sha256']
check('sealed_V3_parent_unchanged',parent_ok,{'manifest':digest(V3/'MANIFEST.json'),'seal':digest(V3/'SEAL.json')})
oldfiles=sorted(f.relative_to(V3).as_posix() for f in V3.rglob('*') if f.is_file())
archivefiles=sorted(f.relative_to(A).as_posix() for f in A.rglob('*') if f.is_file())
archive_ok=oldfiles==archivefiles and all((A/f).read_bytes()==(V3/f).read_bytes() for f in oldfiles)
check('complete_V3_archive_byte_identical',archive_ok,{'files':len(oldfiles)})
texts={};inputs=[]
def expand(name,chain=()):
 if name in chain:raise ValueError('cycle '+name)
 s=uncomment((P/name).read_text());texts[name]=s
 def sub(m):
  target=m[1] if m[1].endswith('.tex') else m[1]+'.tex'
  inputs.append({'from':name,'input':m[1],'resolved':target})
  return expand(target,chain+(name,))
 return re.sub(r'\\input\{([^}]+)\}',sub,s)
expanded=expand('main.tex');main=(P/'main.tex').read_text();main_only=expanded.split('\\appendix',1)[0]
def expand_raw(name):
 return re.sub(r'\\input\{([^}]+)\}',lambda m:expand_raw(m[1] if m[1].endswith('.tex') else m[1]+'.tex'),(P/name).read_text())
raw_expanded=expand_raw('main.tex')
labels=re.findall(r'\\label\{([^}]+)\}',expanded);refs=re.findall(r'\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}',expanded)
check('recursive_local_inputs_resolve',True,inputs)
check('unique_active_labels',len(labels)==len(set(labels)),[k for k,n in collections.Counter(labels).items() if n>1])
check('all_active_references_resolve',not set(refs)-set(labels),sorted(set(refs)-set(labels)))
check('comment_aware_braces_balanced',all(braces(s) for s in texts.values()))
stack=[];errors=[]
for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}',expanded):
 if kind=='begin':stack.append(env)
 elif not stack or stack.pop()!=env:errors.append(env)
check('environment_nesting_balanced',not stack and not errors,{'unclosed':stack,'mismatched':errors})
check('math_delimiters_paired',all(sum(c=='$' and not escaped(s,i) for i,c in enumerate(s))%2==0 for s in texts.values()) and expanded.count('\\[')==expanded.count('\\]') and expanded.count('\\(')==expanded.count('\\)'))
check('anonymous_scientific_source', '\\author{Anonymous Authors}' in main and 'pdfauthor={Anonymous Authors}' in main and not re.search(r'\\iclrfinalcopy\b',uncomment(main)) and not re.search(r'/Users/|/home/|alex|shmelev|Physics|Penn|COMPLETE_ASSOCIATION',expanded,re.I))
check('no_new_main_formulas_or_rhetorical_questions',main_only.count('\\begin{equation}')==1 and '\\label{eq:be}' in main_only and '?' not in main_only)
graphics=[]
for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',expanded):
 actual=next(f for f in [P/target,P/'figures'/target] if f.is_file())
 graphics.append({'logical_reference':target,'resolved':str(actual.relative_to(P)),**digest(actual)})
check('three_figure_bytes_preserved',len(graphics)==3 and all((P/r['resolved']).read_bytes()==(V3/r['resolved']).read_bytes() for r in graphics),graphics)
bib=(P/'references.bib').read_text();keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib);cites=set()
for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}',expanded):cites.update(k.strip() for k in group.split(','))
check('bibliography_unchanged_and_citations_resolve',(P/'references.bib').read_bytes()==(V3/'references.bib').read_bytes() and len(keys)==len(set(keys)) and not cites-set(keys) and braces(uncomment(bib)),{'used_keys':sorted(cites),'unresolved':sorted(cites-set(keys))})
check('conference_styles_unchanged',all((P/f).read_bytes()==(V3/f).read_bytes() for f in ['iclr2027_conference.sty','iclr2027_conference.bst']))
paths=re.findall(r'\\path\{([^}]+)\}',expanded)
check('all_readable_supplement_paths_exist',all((P/f).exists() for f in paths),[f for f in paths if not (P/f).exists()])
check('supplements_archive_not_PDF_inputs',not any(r['resolved'].startswith(('code_evidence/','evidence/')) for r in inputs))
extraction=json.loads((P/'code_evidence/EXTRACTION_MAP.json').read_text());exact=True
for item in extraction['blocks']:
 lines=(P/item['source']).read_text().splitlines(keepends=True);dest=(P/item['destination']).read_text()
 for r in item['verbatim_ranges']:
  a,b=r['lines'];piece=''.join(lines[a-1:b]);exact &= piece in dest and len(piece.encode())==r['bytes'] and hashlib.sha256(piece.encode()).hexdigest()==r['sha256']
check('all_extracted_source_ranges_verbatim',exact,{'files':len(extraction['blocks'])})
supp={str(f.relative_to(P)):f.read_text() for f in (P/'code_evidence').rglob('*.tex.txt')}
tables=[];pattern=r'\\begin\{table\}.*?\\end\{table\}'
for f in ['main.tex','historical_appendix.tex','sections/F_absolute_results_appendix.tex','sections/conditional_value_and_packed_results.tex','sections/conditional_value_E_PE_appendix.tex']:
 for table in re.findall(pattern,(V3/f).read_text(),re.S):
  label=re.search(r'\\label\{([^}]+)\}',table)[1]
  tables.append({'label':label,'source':f,'paper_verbatim':table in raw_expanded,'supplement_verbatim':[n for n,s in supp.items() if table in s],'sha256':hashlib.sha256(table.encode()).hexdigest()})
check('every_V3_numerical_table_verbatim',all(t['paper_verbatim'] or t['supplement_verbatim'] for t in tables),tables)
required={'tab:benchmark-absolute','tab:resource-main','tab:component-main','tab:conditional-seed-intervals','tab:optimizer-default','tab:gat-pair','tab:ogb-pilot','tab:collab-hits','tab:paired','tab:paired-auc','tab:wikics-matched-budget','tab:cora-matched-budget','tab:internal-sage-F','tab:E-PE-main','tab:E-PE-raw','tab:E-calibrated','tab:E-paired','tab:PE-contrasts','tab:E-PE-errors','tab:E-PE-costs'}
check('all_primary_absolute_error_cost_tables_in_paper',required<={t['label'] for t in tables} and all(t['paper_verbatim'] for t in tables if t['label'] in required))
figs=re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',(V3/'main.tex').read_text()+(V3/'sections/F_protocol_diagnostics_appendix.tex').read_text(),re.S)
check('all_original_figure_environments_verbatim',len(figs)==3 and all(f in expanded for f in figs))
hlines=(V3/'historical_appendix.tex').read_text().splitlines(keepends=True)
check('fixed_mask_and_matched_budget_vectors_verbatim',all(''.join(hlines[a-1:b]) in expanded for a,b in [(350,356),(725,731),(781,801)]))
check('F_absolute_table_and_failed_NLL_rule_preserved',(P/'sections/F_absolute_results_appendix.tex').read_bytes()==(V3/'sections/F_absolute_results_appendix.tex').read_bytes() and '0.021448383545404' in expanded and 'mean-NLL allowance fails against I4' in main_only)
appfiles=['historical_appendix.tex','sections/sharing_scope_appendix.tex','sections/F_absolute_results_appendix.tex','sections/F_protocol_diagnostics_appendix.tex','sections/conditional_value_E_PE_appendix.tex','sections/auxiliary_explorations_appendix.tex']
def appsections(base):return re.findall(r'\\section\{([^}]+)\}\\label\{([^}]+)\}',''.join((base/f).read_text() for f in appfiles))
sections=appsections(P)
check('numeric_appendix_section_sequence_preserved','\\renewcommand{\\thesection}{A\\arabic{section}}' in main and '\\renewcommand{\\theHsection}{appendix.\\arabic{section}}' in main and sections==appsections(V3),[{'number':'A'+str(i+1),'title':t,'label':l} for i,(t,l) in enumerate(sections)])
esm=json.loads((P/'EVIDENCE_SOURCE_MANIFEST.json').read_text())
check('all_closed_evidence_copies_match_sources',all(digest(P/r['local_copy'])==digest(R/r['repo_relative_source'])=={k:r[k] for k in ['bytes','sha256']} for r in esm['sources']),{'copies':len(esm['sources'])})
head=json.loads((P/'evidence/closed_E_head_SUMMARY.json').read_text());rows=list(csv.DictReader((P/'evidence/closed_E_head_EQUAL_SEED_SUMMARY.csv').open()));app=(P/'sections/conditional_value_E_PE_appendix.tex').read_text()
ht=next(t for t in re.findall(pattern,app,re.S) if '\\label{tab:E-head-absolute}' in t)
condition={'original':'untouched','BE_head_continuation':'BE continuation','common_affine_residual':'common affine','private_affine_residual':'private affine','affine_residual':'affine'}
head_ok=True
for r in rows:
 name={'E4':'E4','equipped_M1':'M1','equipped_I4':'I4'}[r['arm']]+' '+condition[r['condition']]
 line=next(l for l in ht.splitlines() if l.startswith(name+' &'))
 actual=[v.strip().rstrip('\\').strip() for v in line.split('&')[1:]]
 expected=[f"{float(r['equal_seed_mean_served_accuracy_pct']):.6f}",f"{float(r['equal_seed_mean_mean_member_accuracy_pct']):.6f}",f"{float(r['equal_seed_mean_pooled_nll']):.9f}",r['strict_rival_dependent_readout_sum']]
 head_ok &= actual==expected
check('all_ten_head_absolute_rows_match_complete_summary',len(rows)==10 and head_ok)
check('head_negative_vectors_and_NLL_gap_preserved',head['status']=='COMPLETE_RECORD_ONLY_NEGATIVE_INTERPRETATION' and not head['all_requested_member_NLL_evaluable'] and all(t in app for t in ['$+3,-1,+3$','$+3,-3,+2$','$+2,-3,0$','Every accuracy clause fails','Per-member SELECT NLL was not exported','serves zero repairs','no unused whole-pipeline confirmation']))
counts=json.loads((P/'evidence/closed_E_strata_COUNTS.json').read_text());count_ok=True
for ref,expected in [('BASE',[316,79,451,2064,1929]),('equipped_I4',[436,122,361,2064,2139])]:
 rs=[r for r in counts['records'] if r['reference']==ref and r['view']=='sum_three_dependent_readouts' and r['dimension']=='class']
 count_ok &= [sum(r[k] for r in rs) for k in ['strict_rival_removed','reference_strict_rival_repaired','new_strict_rival_created','candidate_strict_rival','reference_strict_rival']]==expected
check('rival_counts_and_served_repair_distinction_match',count_ok and 'rather than a shared wrong argmax' in app and 'Predicate removal alone is not served repair' in app and 'nonnegative scalar probability average' in main_only)
check('E_uncertainty_adverse_scores_and_cost_scopes_present',all(t in expanded for t in ['I4 interval includes zero','NLL and Brier are worse','noninferiority','74\\%','overlapped other work on the same GPU','not isolated latency benchmarks','does not isolate private builder factors']))
oldapp=''.join((V3/f).read_text() for f in appfiles);newapp=''.join((P/f).read_text() for f in appfiles)
metrics={'V3_active_appendix_source_bytes':len(oldapp.encode()),'V4_active_appendix_source_bytes':len(newapp.encode()),'source_byte_reduction':len(oldapp.encode())-len(newapp.encode()),'source_byte_reduction_fraction':1-len(newapp.encode())/len(oldapp.encode()),'scope':'Source text only including unchanged tables and added head evidence; no page-count, layout or rendering claim.'}
check('active_appendix_source_materially_shorter',metrics['source_byte_reduction_fraction']>0.15,metrics)
failed=[c['check'] for c in checks if not c['pass']]
notdone=['compilation','editor/PDF opening','rendering','Desktop access','installation','new engineering/model tests','scientific execution','reviewer invocation','source-association outcome access','current Physics/Penn outcomes']
result={'schema':'anonymous_quality_draft_static_checks_v4','status':'PASS' if not failed else 'FAIL','scope':'Standard-library static source/dependency, numerical-transcription and custody audit only; not compilation, layout or scientific validity.','checks':checks,'failed_checks':failed,'not_performed':notdone}
(P/'STATIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
dep={**json.loads((V3/'DEPENDENCY_MAP.json').read_text()),'schema':'complete_anonymous_quality_draft_dependency_map_v4','writing_parent':{'repo_relative_path':V3.name,'manifest':digest(V3/'MANIFEST.json'),'seal':digest(V3/'SEAL.json'),'verified_unchanged':parent_ok,'preserved_archive':'code_evidence/v3_archive'},'resolved_inputs':inputs,'graphics':graphics,'bibliography':{'database':'references.bib','style':'iclr2027_conference.bst','used_keys':sorted(cites),'all_keys_resolved':not cites-set(keys)},'supplement_paths':sorted(set(paths)),'supplement_extraction_map':'code_evidence/EXTRACTION_MAP.json','table_locations':tables,'appendix_section_labels':[{'number':'A'+str(i+1),'title':t,'label':l} for i,(t,l) in enumerate(sections)],'syntax_scope':result['scope'],'not_performed':notdone}
(P/'DEPENDENCY_MAP.json').write_text(json.dumps(dep,indent=2)+'\n')
(P/'CHANGE_AUDIT.json').write_text(json.dumps({'schema_version':1,'status':result['status'],'V3_preserved':parent_ok,'V3_archive_identical':archive_ok,'numerical_tables':tables,'source_reductions':metrics,'supplements':extraction['blocks'],'checks_file':'STATIC_CHECKS.json','no_page_count_claim':True,'scientific_additions':['Completed old-WikiCS error/common-rival diagnostic','Completed negative anchored fixed-feature head diagnostic'],'excluded_results':['Source-context association','Physics/Penn','Unused whole-pipeline confirmation'],'not_performed':notdone},indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'failed':failed,'labels':len(labels),'citation_keys':len(cites),'original_tables':len(tables),'paper_verbatim_tables':sum(t['paper_verbatim'] for t in tables),'appendix_source':metrics},indent=2))
sys.exit(1 if failed else 0)

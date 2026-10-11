"""Stdlib source/custody audit only. Does not compile or run scientific code."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import sys

V3 = Path(__file__).resolve().parent
ROOT = V3.parent
V2 = ROOT / 'complete_anonymous_quality_draft_20261011_v2'
checks = []

def digest(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def check(name, passed, detail=None):
    item = {'check': name, 'pass': bool(passed)}
    if detail is not None:
        item['detail'] = detail
    checks.append(item)

def uncomment(s):
    out = []
    for line in s.splitlines(keepends=True):
        for i, char in enumerate(line):
            if char == '%' and len(line[:i]) - len(line[:i].rstrip('\\')) & 1 == 0:
                line = line[:i] + '\n'
                break
        out.append(line)
    return ''.join(out)

def escaped(s, i):
    return (len(s[:i]) - len(s[:i].rstrip('\\'))) % 2 == 1

def braces(s):
    depth = 0
    for i, char in enumerate(s):
        if char in '{}' and not escaped(s, i):
            depth += 1 if char == '{' else -1
            if depth < 0:
                return False
    return depth == 0

def verify_packet(folder, expected_manifest=None):
    manifest = folder / 'MANIFEST.json'
    data = json.loads(manifest.read_text())
    records = data['files']
    valid = all(digest(folder / r['path']) == {k:r[k] for k in ['bytes','sha256']} for r in records)
    if expected_manifest:
        valid = valid and digest(manifest)['sha256'] == expected_manifest
    seal = json.loads((folder / 'SEAL.json').read_text())
    valid = valid and seal.get('manifest_sha256') == digest(manifest)['sha256']
    return valid, len(records)

v2_valid, v2_count = verify_packet(V2, '230ad5559ea996e0657e89dbbbb3a38f286399fb3771d14edf4c3cf6668588b2')
check('sealed_v2_parent_unchanged', v2_valid, {'files': v2_count, 'manifest': digest(V2/'MANIFEST.json'), 'seal': digest(V2/'SEAL.json')})
olddep = json.loads((V2/'DEPENDENCY_MAP.json').read_text())
upstream = olddep['source_inputs'] + olddep['evidence_report_bindings']
check('inherited_upstream_source_and_closed_report_hashes', all(digest(ROOT/r['repo_relative_path']) == {k:r[k] for k in ['bytes','sha256']} for r in upstream), {'bindings':len(upstream)})

main = (V3/'main.tex').read_text()
oldmain = (V2/'main.tex').read_text()
source_text = {}
resolved_inputs = []

def expand(name, ancestors=()):
    if name in ancestors:
        raise ValueError('Input cycle: ' + name)
    path = V3/name
    s = uncomment(path.read_text())
    source_text[name] = s
    def replace(match):
        target = match.group(1)
        resolved = target if target.endswith('.tex') else target+'.tex'
        if not (V3/resolved).is_file():
            raise FileNotFoundError(resolved)
        resolved_inputs.append({'from':name, 'input':target, 'resolved':resolved})
        return expand(resolved, ancestors+(name,))
    return re.sub(r'\\input\{([^}]+)\}', replace, s)

expanded = expand('main.tex')
main_expanded = expanded.split('\\appendix', 1)[0]
check('all_recursive_tex_inputs_local', True, {'included_files':len(source_text), 'inputs':len(resolved_inputs)})
labels = re.findall(r'\\label\{([^}]+)\}', expanded)
refs = re.findall(r'\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}', expanded)
duplicate_labels = [k for k,v in collections.Counter(labels).items() if v>1]
check('unique_labels', not duplicate_labels, {'labels':len(labels), 'duplicates':duplicate_labels})
check('all_cross_references_resolve', not set(refs)-set(labels), {'occurrences':len(refs), 'unresolved':sorted(set(refs)-set(labels))})
bib = (V3/'references.bib').read_text()
bibkeys = re.findall(r'@\w+\s*\{\s*([^,\s]+)', bib)
cites = set()
for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', expanded):
    cites.update(k.strip() for k in group.split(','))
check('bibliography_keys_unique_and_citations_resolve', len(bibkeys)==len(set(bibkeys)) and not cites-set(bibkeys), {'used_keys':len(cites),'unresolved':sorted(cites-set(bibkeys))})
check('bib_braces_balanced', braces(uncomment(bib)))
check('tex_braces_balanced_comment_aware', all(braces(s) for s in source_text.values()))
stack = []
env_errors = []
for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}', expanded):
    if kind=='begin':
        stack.append(env)
    elif not stack or stack.pop()!=env:
        env_errors.append(env)
check('document_and_environment_nesting_balanced', not stack and not env_errors, {'unclosed':stack,'mismatched':env_errors})
check('inline_dollar_math_paired', all(sum(c=='$' and not escaped(s,i) for i,c in enumerate(s)) % 2 == 0 for s in source_text.values()))
check('display_math_delimiters_paired', expanded.count('\\[')==expanded.count('\\]') and expanded.count('\\(')==expanded.count('\\)'))
graphics = []
for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', expanded):
    candidates = [V3/target, V3/'figures'/target]
    actual = next((p for p in candidates if p.is_file()), None)
    if actual is None:
        raise FileNotFoundError(target)
    graphics.append({'logical_reference':target,'resolved':str(actual.relative_to(V3)), **digest(actual)})
check('graphics_and_bibliography_style_local', len(graphics)==3 and (V3/'iclr2027_conference.bst').is_file() and (V3/'iclr2027_conference.sty').is_file() and '\\bibliography{references}' in main, {'graphics':graphics})
check('standard_anonymous_title_author_and_pdf_metadata', '\\author{Anonymous Authors}' in main and 'pdfauthor={Anonymous Authors}' in main and not re.search(r'\\iclrfinalcopy\b', uncomment(main)) and not re.search(r'\\(?:fancyhead|makeatletter|renewcommand\{\\maketitle)', uncomment(main)))
check('main_only_necessary_BE_display_equation', main_expanded.count('\\begin{equation}')==1 and '\\label{eq:be}' in main_expanded and '\\label{eq:E-paired-maps}' not in main_expanded)
check('no_pending_Penn_or_confirmation_results_or_missing_plot', not re.search(r'Penn|pending|awaiting|E_PE_quality_v3', expanded, re.I))

def block(s, start, end):
    return s[s.index(start):s.index(end)]

start = '\\section{Five-dataset benchmark}'
end = '\\input{sections/internal_sage_methods_results}'
protected = block(main,start,end)
check('five_dataset_resource_and_Roman_block_verbatim_from_v2', protected==block(oldmain,start,end), {'sha256':hashlib.sha256(protected.encode()).hexdigest()})
budget = 'An ordinary ensemble can also store fewer weights'
check('Roman_matched_budget_paragraph_verbatim', main[main.index(budget):].split('\n\n')[0] == oldmain[oldmain.index(budget):].split('\n\n')[0])
preserved = ['historical_appendix.tex','iclr2027_conference.sty','iclr2027_conference.bst','sections/F_absolute_results_appendix.tex','sections/sharing_scope_appendix.tex','sections/auxiliary_explorations_appendix.tex','figures/architecture_sharing_compact.png','figures/gnnm_vs_ens_heatmap.png','figures/F_member_coverage_v2.png']
check('all_inherited_tables_negative_results_and_figure_bytes_preserved', all((V3/f).read_bytes()==(V2/f).read_bytes() for f in preserved), {'files':preserved})
oldF = (V2/'sections/internal_sage_methods_results.tex').read_text()
movedF = (V3/'sections/F_protocol_diagnostics_appendix.tex').read_text()
check('entire_old_F_detail_body_relocated_verbatim', oldF.split('\n',1)[1] == movedF.split('\n',1)[1])
check('historical_appendix_and_F_figure_each_included_once', sum(r['resolved']=='historical_appendix.tex' for r in resolved_inputs)==1 and expanded.count('{F_member_coverage_v2.png}')==1)
check('original_bibliography_exact_prefix_and_four_only_new_keys', (V3/'references.bib').read_bytes().startswith((V2/'references.bib').read_bytes()) and set(bibkeys)-set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', (V2/'references.bib').read_text())) == {'simonovsky2017ecc','brockschmidt2020film','bodnar2022sheaf','bourgerie2026deepnsd'})
check('new_citation_accents_and_preprint_status_preserved', "Li{\\`o}, Pietro" in bib and "R{\\'e}mi" in bib and '{\\v S}ar{\\=u}nas' in bib and 'arXiv preprint arXiv:2605.19021v1' in bib)

esm = json.loads((V3/'EVIDENCE_SOURCE_MANIFEST.json').read_text())
check('all_six_closed_evidence_copies_byte_identical_to_sources', len(esm['sources'])==6 and all(digest(V3/r['local_copy'])==digest(ROOT/r['repo_relative_source'])=={k:r[k] for k in ['bytes','sha256']} for r in esm['sources']))
joined = json.loads((V3/'evidence/closed_joined_PE_REPORT.json').read_text())
summary = json.loads((V3/'evidence/closed_joined_PE_SUMMARY.json').read_text())
E = json.loads((V3/'evidence/closed_E_PAIRED_MAIN_CONTRASTS.json').read_text())
with (V3/'evidence/closed_E_ARM_METRICS.csv').open(newline='') as stream:
    csvrows = {(r['readout'],r['arm']):r for r in csv.DictReader(stream)}
aggregate = {r['arm']:r for r in joined['arm_aggregates']}
app = (V3/'sections/conditional_value_E_PE_appendix.tex').read_text()
newmain = (V3/'sections/conditional_value_and_packed_results.tex').read_text()
display_names = {'BASE':'BASE (fresh shared4)','E4':'E4 (endpoints)','N4':'N4 (receiver)','equipped_M1':'Equipped M1','equipped_I4':'Equipped I4','PE4_gamma1_F':'PE4'}

def table(s, label):
    return next(t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',s,re.S) if '\\label{'+label+'}' in t)

def row(s, label, prefix):
    lines = [l.strip() for l in table(s,label).splitlines() if l.strip().startswith(prefix+' ') and '&' in l]
    if len(lines)!=1:
        raise ValueError((label,prefix,lines))
    return [v.strip().rstrip('\\').strip() for v in lines[0].split('&')[1:]]

def same(fields, values, precisions):
    if len(fields)!=len(values):
        return False
    for f,v,p in zip(fields,values,precisions):
        if isinstance(v,(list,tuple)):
            actual = re.findall(r'-?\d+(?:\.\d+)?',f)
            if actual != [format(x,'.'+str(p)+'f') for x in v]:
                return False
        else:
            f = f.replace('$','').replace(',','').lstrip('+')
            if f != (str(int(v)) if p==0 else format(v,'.'+str(p)+'f')):
                return False
    return True

raw_ok = cost_ok = True
for arm,a in aggregate.items():
    q = a['quality']
    params = a['costs']['selected_parameters_per_seed'][0]['parameters']
    accuracy,nll,brier,member = [q[k]['mean'] for k in ['pooled_accuracy_pct','pooled_nll','pooled_brier','mean_member_accuracy_pct']]
    raw_ok &= same(row(newmain,'tab:E-PE-main',display_names[arm]), [accuracy,nll,params],[4,4,0])
    raw_ok &= same(row(app,'tab:E-PE-raw',display_names[arm]), [accuracy,member,nll,brier,params],[4,4,6,6,0])
    c = a['costs']
    cost_ok &= same(row(app,'tab:E-PE-costs',display_names[arm]), [c['summed_group_costs']['acquisition_seconds'],c['summed_group_costs']['selected_serving_readout_seconds'],c['maximum_recorded_peaks']['cuda_peak_allocated_bytes']/1e9],[3,6,3])
check('all_six_raw_main_and_appendix_arms_match_closed_evidence', raw_ok)
check('all_six_resource_rows_match_closed_scopes', cost_ok)
cal_ok = True
for arm in ['BASE','E4','N4','equipped_M1','equipped_I4']:
    r = csvrows['calibrated',arm]
    cal_ok &= same(row(app,'tab:E-calibrated',display_names[arm]), [float(r[k]) for k in ['accuracy_pct','raw_or_calibrated_NLL','Brier']],[4,6,6])
check('all_five_calibrated_E_rows_match_closed_evidence', cal_ok)
pair_ok = err_ok = pe_ok = True
for key,c in E.items():
    arm = key.removeprefix('E4_minus_')
    q = c['quality_deltas']['pooled_accuracy_pct']
    caldelta = float(csvrows['calibrated','E4']['raw_or_calibrated_NLL'])-float(csvrows['calibrated',arm]['raw_or_calibrated_NLL'])
    ref = {'BASE':'BASE','N4':'N4','equipped_I4':'Equipped I4','equipped_M1':'Equipped M1'}[arm]
    pair_ok &= same(row(app,'tab:E-paired',ref),[q['mean'],q['descriptive_95pct_t_interval_df2'],caldelta],[4,4,6])
    d = c['readout_diagnostics']; delta=d['count_decomposition_delta']
    prefix = 'E4--'+{'BASE':'BASE','N4':'N4','equipped_I4':'equipped I4','equipped_M1':'equipped M1'}[arm]
    fields=row(app,'tab:E-PE-errors',prefix)
    err_ok &= same(fields[:4],[delta[k] for k in ['coverage','lost_correct_alternatives','aggregation_only_correct','pooled_correct']],[0]*4) and fields[4] == str(d['repairs'])+'/'+str(d['harms'])
for key,c in joined['raw_contrasts'].items():
    arm = c['candidate'];q=c['quality_deltas']
    pe_ok &= same(row(app,'tab:PE-contrasts',display_names[arm]), [q['pooled_accuracy_pct']['mean'],q['pooled_accuracy_pct']['descriptive_95pct_t_interval_df2'],q['pooled_nll']['mean'],q['pooled_brier']['mean']],[4,4,6,6])
    d=c['summed_three_seed_readout_diagnostics'];delta=d['count_decomposition_delta']
    prefix={'BASE':'BASE','E4':'E4','N4':'N4','equipped_I4':'Equipped I4','equipped_M1':'Equipped M1'}[arm]+'--PE4'
    fields=row(app,'tab:E-PE-errors',prefix)
    err_ok &= same(fields[:4],[delta[k] for k in ['coverage','lost_correct_alternatives','aggregation_only_correct','pooled_correct']],[0]*4) and fields[4]==str(d['repairs'])+'/'+str(d['harms'])
check('all_four_E_paired_rows_and_intervals_match_evidence', pair_ok)
check('all_five_PE_paired_rows_and_intervals_match_evidence', pe_ok)
check('all_nine_error_accounting_rows_match_evidence', err_ok)
p = joined['protocol']
check('complete_closed_protocol_counts_no_PE_calibration', summary['complete'] and len(joined['arm_details'])==18 and len(aggregate)==6 and p['joined_banks']==18 and p['joined_bundles']==27 and p['joined_body_records']==36 and not p['new_calibration'] and not p['TEST_access'] and not p['model_forwards_or_selected_state_deserialize'] and 'All 75 fits and 37,500 scalar updates closed' in app and 'No PE temperature or calibrated PE endpoint was fitted' in app)
ci = E['E4_minus_equipped_I4']['quality_deltas']['pooled_accuracy_pct']['descriptive_95pct_t_interval_df2']
check('I4_uncertainty_NLL_harm_and_N4_majority_present', ci[0]<0<ci[1] and 'I4 interval includes zero' in newmain and 'NLL and Brier are worse' in newmain and '74\\%' in newmain and '0.0160' in newmain and 'noninferiority' in newmain)
check('development_reuse_and_timing_overlap_limits_present', 'same development nodes informed native selection before calibration' in newmain and 'three dependent readouts' in app and 'overlapped other work on the same GPU' in app and 'not isolated latency benchmarks' in app and 'does not isolate the benefit of private builder factors' in app)
check('closed_F_NLL_failure_and_intervals_retained', '0.021448383545404' in expanded and '0.02' in expanded and 'mean-NLL allowance fails against I4' in main_expanded and '[-0.2653,1.3398]' in main_expanded)

failed = [c['check'] for c in checks if not c['pass']]
result = {'schema':'anonymous_quality_draft_static_checks_v3','status':'PASS' if not failed else 'FAIL','scope':'stdlib local source/custody and numerical transcription checks only; heuristic syntax checks do not establish TeX compilation, layout or scientific validity','checks':checks,'failed_checks':failed,'not_performed':['compilation','editor/PDF opening','rendering','installation','Desktop access','reviewer invocation','data/model/scientific remote execution']}
(V3/'STATIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
dependency = {'schema':'complete_anonymous_quality_draft_dependency_map_v3','entrypoint':'main.tex','source_status':'anonymous uncompiled working source; historical round18 source not verified submitted','writing_parent':{'repo_relative_path':V2.name,'manifest':digest(V2/'MANIFEST.json'),'seal':digest(V2/'SEAL.json'),'verified_unchanged':v2_valid},'inherited_source_inputs':olddep['source_inputs'],'inherited_closed_report_bindings':olddep['evidence_report_bindings'],'resolved_inputs':resolved_inputs,'graphics':graphics,'bibliography':{'database':'references.bib','style':'iclr2027_conference.bst','used_keys':sorted(cites),'all_keys_resolved':not cites-set(bibkeys)},'conference_style':'iclr2027_conference.sty','closed_evidence_manifest':'EVIDENCE_SOURCE_MANIFEST.json','syntax_scope':result['scope'],'not_performed':result['not_performed'],'no_layout_or_submission_ready_claim':True}
dependency['citation_carriers'] = esm.get('citation_carriers', [])
dependency['sealed_related_packets'] = esm.get('sealed_related_packets', [])
dependency['writing_constraints_source'] = esm.get('writing_constraints_source', {})
(V3/'DEPENDENCY_MAP.json').write_text(json.dumps(dependency,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'failed':failed,'included_tex_files':len(source_text),'labels':len(labels),'citation_keys':len(cites),'v2_unchanged':v2_valid},indent=2))
sys.exit(1 if failed else 0)

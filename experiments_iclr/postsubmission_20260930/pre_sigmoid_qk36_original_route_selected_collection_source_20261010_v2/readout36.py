"""Thin Q/K36 configuration over the sealed relation M1/M4 readout."""
import ast
import copy
import hashlib
from pathlib import Path

SEEDS = (6101, 6203, 6307)
KINDS = ('single', 'be_init', 'independent4')
OPERATORS = ('native_tied', 'active_reversible_exp', 'pre_sigmoid_split', 'full_qk')
METRICS = ('served_accuracy', 'served_nll', 'served_brier', 'mean_member_accuracy', 'worst_member_accuracy',
    'mean_member_nll', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')
LIMITS = dict(exploratory=True, independent_TEST_evidence=False, novelty_or_confirmation_claimed=False,
    selection_population_caveat='All original5274 development nodes selected these states and provide this readout.',
    replication='Three fixed optimizer-seed blocks on one graph; no node/member independence or graph-population interval.',
    cohorts='Each family/seed native_tied freezes its own cohorts; all9 baseline banks precede every candidate. Cross-family comparisons use the shared be_init/native_tied cohorts.',
    member_pairing='Each actual M1/M4 bank summarized separately; pooled flows, no arbitrary cross-bank member-index pairing.',
    operator='New-panel be_init Rademacher stem and public shared native local attention; no relationJ/private-local-scorer substitution.',
    causal_claim='Quality does not establish that nonreversibility or negative modes caused a gain.',
    numerical_parity_claimed=False, reselection_or_calibration=False)


def comparisons():
    rows = []
    for kind in ('be_init', 'single', 'independent4'):
        right = kind + '/pre_sigmoid_split'
        for operator in ('native_tied', 'active_reversible_exp', 'full_qk'):
            left = kind + '/' + operator; rows.append((right + '-' + left, left, right))
    for operator in ('pre_sigmoid_split', 'native_tied', 'active_reversible_exp', 'full_qk'):
        right = 'be_init/' + operator
        for kind in ('single', 'independent4'):
            left = kind + '/' + operator; rows.append((right + '-' + left, left, right))
    return tuple(rows)


def arrays_tree(source):
    tree = ast.parse(source); fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'arrays_for'))
    changed = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "record['historical_reference']":
            node.test = ast.parse("'member_brier' not in arrays").body[0].value; changed += 1
    if changed != 1:
        raise ValueError('One original probability-only Brier derivation boundary')
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def report_tree(source):
    tree = ast.parse(source); fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    baseline_index = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == 'seed')
    fn.body[baseline_index:baseline_index+1] = ast.parse("""
for kind in KINDS:
    for seed in SEEDS:
        cohort=collection['cohorts'][kind][str(seed)]
        if cohort.get('available'):
            masks=contract.load(np,g.bound(cohort['archive']),cohort['archive'])
            baseline_record=next(r for r in collection['cells'] if (r['condition'],r['seed'])==(kind+'/native_tied',seed))
            baseline=arrays_for(np,g,baseline_record,contract)
            expected=contract.baseline_cohorts(np,baseline)
            g.require(set(masks)==set(expected) and all(np.array_equal(masks[k],expected[k]) for k in expected),'Immutable family-native-tied cohort/rival seal')
            frozen[(seed,kind)]=masks
            del baseline
""").body
    class Adapt(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'whole18':
                node.id = 'whole36'
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            if isinstance(node.left,ast.Name) and node.left.id == 'seed' and any(isinstance(c,ast.Name) and c.id == 'frozen' for c in node.comparators):
                node.left=ast.parse("(seed,right.split('/')[0])").body[0].value
            return node
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Name) and node.value.id=='frozen' and isinstance(node.slice,ast.Name) and node.slice.id=='seed':
                node.slice=ast.parse("(seed,right.split('/')[0])").body[0].value
            return node
        def visit_Constant(self, node):
            if node.value=='relation_protocol': node.value='pilot_protocol'
            if node.value=='UNCHANGED_SCREEN.json': node.value='PILOT_DECISION.json'
            return node
    fn=Adapt().visit(fn)
    complete=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='whole36' for t in n.targets))
    complete.value=ast.parse("len(summaries)==36 and len(frozen)==9 and all(p.get('available') for p in pairs)").body[0].value
    report_start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lines' for t in n.targets))
    report_end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and 'REPORT.md' in ast.unparse(n))
    fn.body[report_start:report_end+1]=ast.parse('_report(compact,contrasts,screen,collection)').body
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))


def configure(g, pins):
    base=g.module(g.bound(pins['reuse']['relation_readout']), '_qk36_reused_actual_M1_M4')
    analysis,contract,pair_hash=base.configure(g,pins)
    ns=dict(vars(base)); ns.update(SEEDS=SEEDS,KINDS=KINDS,COMPARISONS=comparisons(),METRICS=METRICS,LIMITS=LIMITS,
        gate_result=pilot_decision,_report=report)
    arrays_ast=arrays_tree(Path(base.__file__).read_text())
    exec(compile(arrays_ast,base.__file__+':existing-Brier-derivation','exec'),ns)
    base_pair=base.pair
    def pair(np,before,after,frozen,seed,label,analysis_arg,contract_arg):
        result=base_pair(np,before,after,frozen,seed,label,analysis_arg,contract_arg)
        def rename(value):
            if isinstance(value,dict): return {k.replace('alphaF','native_tied'):rename(v) for k,v in value.items()}
            if isinstance(value,list): return [rename(v) for v in value]
            return value
        return rename(result)
    ns['pair']=pair
    run_ast=report_tree(Path(base.__file__).read_text())
    exec(compile(run_ast,base.__file__+':QK36-configuration','exec'),ns)
    return analysis,contract,ns['arrays_for'],ns['run'],dict(
        arrays_AST_sha256=hashlib.sha256(ast.dump(arrays_ast,include_attributes=False).encode()).hexdigest(),
        analysis_run_AST_sha256=hashlib.sha256(ast.dump(run_ast,include_attributes=False).encode()).hexdigest(),
        pooled_pair_AST_sha256=pair_hash)


def pilot_decision(contrasts, whole36, protocol):
    rule=protocol['prospective_decisions']['quality_and_member_gate']
    expected=dict(mean_accuracy_gain_pp_at_least=.2,mean_over_seed_mean_member_NLL_delta_at_most=.01,
        mean_over_seed_mean_member_accuracy_delta_pp_at_least=-.1,mean_over_seed_worst_member_NLL_delta_at_most=.02,
        mean_over_seed_worst_member_accuracy_delta_pp_at_least=-.2,mean_pool_NLL_delta_at_most=0,positive_every_paired_seed=True)
    if rule!=expected:
        raise ValueError('Exact frozen PILOT_PROTOCOL quality/member gate')
    rows={r['comparison']:r for r in contrasts}
    def gate(label):
        metrics=rows[label]['metrics']
        required=('served_accuracy','served_nll','mean_member_accuracy','worst_member_accuracy','mean_member_nll','worst_member_nll')
        available=all(metrics[m]['available_seeds']==3 for m in required)
        checks=dict(positive_every_paired_seed=available and all(x>0 for x in metrics['served_accuracy']['values']),
            mean_accuracy_gain_pp_at_least=available and metrics['served_accuracy']['mean']>=rule['mean_accuracy_gain_pp_at_least'],
            mean_pool_NLL_delta_at_most=available and metrics['served_nll']['mean']<=rule['mean_pool_NLL_delta_at_most'],
            mean_over_seed_mean_member_accuracy_delta_pp_at_least=available and metrics['mean_member_accuracy']['mean']>=rule['mean_over_seed_mean_member_accuracy_delta_pp_at_least'],
            mean_over_seed_worst_member_accuracy_delta_pp_at_least=available and metrics['worst_member_accuracy']['mean']>=rule['mean_over_seed_worst_member_accuracy_delta_pp_at_least'],
            mean_over_seed_mean_member_NLL_delta_at_most=available and metrics['mean_member_nll']['mean']<=rule['mean_over_seed_mean_member_NLL_delta_at_most'],
            mean_over_seed_worst_member_NLL_delta_at_most=available and metrics['worst_member_nll']['mean']<=rule['mean_over_seed_worst_member_NLL_delta_at_most'])
        return dict(comparison=label,available=available,checks=checks,passed=available and all(checks.values()))
    primary=[gate('be_init/pre_sigmoid_split-be_init/'+operator) for operator in ('native_tied','active_reversible_exp')]
    sharing=[gate('be_init/pre_sigmoid_split-'+kind+'/pre_sigmoid_split') for kind in ('single','independent4')]
    primary_available=all(r['available'] for r in primary)
    primary_pass=all(r['passed'] for r in primary)
    return dict(whole36_readout_available=whole36,original_fixed_rule=rule,operator_co_primary=primary,
        operator_utility_screen_available=primary_available,operator_utility_screen_passed=primary_pass,
        operator_utility_rejected_if_any_co_primary_fails=(not primary_pass if primary_available else None),
        same_operator_sharing_comparisons=sharing,sharing_quality_advantage_supported=primary_pass and all(r['passed'] for r in sharing),
        full_qk_same_family_comparisons=[rows[kind+'/pre_sigmoid_split-'+kind+'/full_qk'] for kind in KINDS],
        original_full_qk_scope=protocol['prospective_decisions']['full_qk_comparisons'],
        full_qk_superiority_automatically_cleared=False,scales_or_asymmetry_can_rescue_failed_operator_gate=False,
        unused_confirmation=False,novelty_cleared=False,further_execution_authorized=False)


def report(compact, contrasts, screen, collection):
    lines=['# Q/K36 selected-development readout','',
        'Four fixed operators × single/shared be_init/genuine independent4 × three paired optimizer seeds. Original selected states and all5274 consumed development nodes; exploratory, no TEST or unused-confirmation evidence.','',
        '| Candidate − reference | Accuracy mean (pp) | SD | Descriptive df2 95% interval | NLL mean | Brier mean |',
        '|---|---:|---:|---|---:|---:|']
    number=lambda x:'unavailable' if x is None else format(x,'.6g')
    for row in contrasts:
        m=row['metrics']; a=m['served_accuracy']; interval=a['exploratory_95pct_df2_interval']
        ci='unavailable' if interval is None else '['+number(interval[0])+', '+number(interval[1])+']'
        lines.append('| '+row['comparison']+' | '+number(a['mean'])+' | '+number(a['SD'])+' | '+ci+' | '+number(m['served_nll']['mean'])+' | '+number(m['served_brier']['mean'])+' |')
    lines.extend(['','Both operator co-primary gates passed: '+str(screen['operator_utility_screen_passed'])+'. Same-operator sharing quality gate supported: '+str(screen['sharing_quality_advantage_supported'])+'.','',
        'Every operator/family/seed, selected epoch and saved body mode, each actual member and mean/worst competence, paired repairs/introduced errors, coverage, pool rescues/harms and native-tied rival diagnostics are retained in compact JSON. Raw arrays stay server-only. Missing seed slots have no survivor-only interval.','',
        'Mandatory full-Q/K comparisons and all training/storage/serving costs remain descriptive. Lower cost, learned scales or raw asymmetry cannot rescue a failed co-primary gate. Novelty, mechanism causality, graph-population generalization and unused confirmation are not established.',''])
    (compact/'REPORT.md').write_text('\n'.join(lines))

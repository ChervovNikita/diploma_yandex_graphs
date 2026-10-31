"""Materialize explicit, inspectable wrapper revisions; no runtime monkeypatching."""
from pathlib import Path
import hashlib
import json
import difflib

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OLD_ROOT = PHASE/'graph_init_execution_root_v1'
OLD_CONT = PHASE/'graph_init_execution_continuation_v1'
NEW_NAME = HERE.name
V3 = 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def replace_function(text, name, replacement):
    start = text.index('def '+name+'(')
    next_def = text.find('\n\ndef ', start+1)
    if next_def == -1:
        next_def = text.index('\n\nif __name__', start)
    return text[:start]+replacement.rstrip()+text[next_def:]

def basic(text):
    replacements = {
      'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2':V3,
      '2a4208581e67e44f8941a43b696fd8ee141b2eec6533893862c1f0dfbe8c0345':'5e8250d9fbe219593bb2aa074665e30cc5686d6ed227fc549f83809dcf89f830',
      'd21fc53935ef20e7f9278039a9a4a26fea4f3248c3a75157401fb0d95c345203':'457cf3bd3275cad315e3d57b5036683577929369f03ea346a718b4f8ffad10f2',
      'c93535404c115c0cf68e64b1f39411f0fd799ecb61f142077e2c2f14981ed713':sha(PHASE/V3/'SOURCE_BINDINGS.json'),
      '44765e8358fac6dd9dc211a13a32f260d304bed7b334c9ea4547d95b2d16856a':sha(PHASE/V3/'PROTOCOL.json'),
      'graph_init_execution_root_v1':NEW_NAME,
      'graph_init_execution_continuation_v1':NEW_NAME,
      'graph_init_cfg0_outcome_aware_v1':'graph_init_cfg0_outcome_aware_precision_v1',
    }
    for a,b in replacements.items():
        text=text.replace(a,b)
    return text

origins=[]
diffs=[]
def emit(old, text):
    original=old.read_text()
    target=HERE/old.name
    if target.exists():
        raise ValueError('No overwrite: '+str(target))
    target.write_text(text)
    origins.append({'target':old.name,'original_path':str(old.relative_to(PHASE)),
                    'original_sha256':sha(old),'revised_sha256':sha(target)})
    diffs.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
                 fromfile=str(old.relative_to(PHASE)),tofile=str(target.relative_to(PHASE))))

for name in ['admission_support.py','continuation_support.py']:
    old=(OLD_ROOT if name=='admission_support.py' else OLD_CONT)/name
    text=basic(old.read_text())
    start=text.index('DISCLOSURE = (')
    end=text.index('\n\n\ndef require',start)
    disclosure=('This precision qualification amendment was designed after the known Photo17 v2 '
        'finite-difference failure and an exactly replayed arithmetic diagnostic. Photo and Squirrel '
        'families were previously exposed; this remains exploratory and outcome-aware. '
        'independent_of_stage1_outcomes describes source-pack custody only: no reuse, binding or '
        'selection of Stage1 fitted outputs or labels as this study evidence. It does not describe '
        'independence of idea/cohort choice or an unseen confirmatory cohort. The failed v2 registry, '
        'all old attempts and all diagnostic lineage remain preserved. No old checkpoint, phase output, '
        'qualification or optimizer history is inherited. All six cold and every actual-warm state '
        'require new v3 qualification with retained original FP32 diagnostics and unchanged thresholds.')
    text=text[:start]+'DISCLOSURE = '+repr(disclosure)+text[end:]
    if name=='continuation_support.py':
        text=text.replace("                if phase == 'qualify' and (graph, seed) == FIRST_CELL:\n                    continue\n",'')
        text=text.replace('len(rows) == 71','len(rows) == 72').replace("== 71, 'Exactly71 remaining attempts required'","== 72, 'Exactly72 fresh attempts required'")
        text=text.replace('Remaining qualifications first','All six fresh qualifications first')
        a=text.index("    first = row_for(registry",text.index('def decision_guard'))
        z=text.index('    return decision, registry, plan',a)
        text=text[:a]+'''    from lineage_support import verify_lineage, prior_registries
    verify_lineage()
    require(registry['prior_attempt_registries'] == prior_registries(), 'All failed v2 registries must be preserved')
    launch = read(bound(decision['registration_launch_request']))
    whole = read(bound(decision['registration_whole_supervision_terminal']))
    root = read(bound(decision['registration_root_terminal']))
    require(launch['action'] == 'register' and launch['root_admitted'] is True and
            launch['phase_payload'] == registry['request'] and launch['output'] == decision['attempt_registry']['path'] and
            launch['source_seals'] == source_descriptors() and
            whole['complete'] is True and whole['within_whole_cap'] is True and
            whole['root_request_unchanged'] is True and whole['whole_cap_seconds'] == 600 and
            root['completed'] is True and root['child_exit_code'] == 0 and root['action'] == 'register' and
            root['output'] == decision['attempt_registry']['path'] and
            root['request_sha256'] == decision['registration_launch_request']['sha256'],
            'Fresh successful registration root/whole terminals required')
''' +text[z:]
        text=text.replace('Signed exact71attempt schedule/caps required','Signed exact72attempt schedule/caps required')
        # Require this precise study anchor, not any moved or copied source registry.
        text=text.replace("    contexts = registry['contexts']\n", "    require(registry['study_id'] == 'graph_init_cfg0_outcome_aware_precision_v1' and\n            registry['anchor_directory'] == str(REMOTE_PHASE/'"+NEW_NAME+"/study_v1'),\n            'Exact new precision study/anchor required')\n    contexts = registry['contexts']\n",1)
    emit(old,text)

old=OLD_ROOT/'build_admissions.py'
text=basic(old.read_text())
text=text.replace('Outputs are exclusive. Initial allowed actions are register and Squirrel17 qualify.',
                  'Outputs are exclusive. Registration is metadata-only; one finite decision admits all72 fresh phases.')
text=text.replace('import sys\n','import sys\nfrom lineage_support import verify_lineage, prior_registries, lineage_descriptors\n',1)
text=text.replace("    records = source_descriptors()\n", "    verify_lineage()\n    records = source_descriptors()+lineage_descriptors()\n",1)
text=replace_function(text,'launch_request', '''def launch_request(action, payload, decision, outdir, authorized=False):
    require(action == 'register', 'Only metadata registration uses this entry')
    base = REMOTE_PHASE/HERE.name
    return {'schema': 'graph-init-root-launch-request-v1', 'root_admitted': authorized, 'action': action,
        'full_fit_admitted': False, 'heldout_scoring_admitted': False,
        'source_packet': str(REMOTE_PHASE/R17_REL), 'source_manifest': source_descriptors()[0],
        'source_seals': source_descriptors(), 'protected_files': protected_metadata(),
        'entry_script': str(base/'graph_init_root_entry.py'),
        'driver_script': str(REMOTE_PHASE/R17_REL/'prototype/graph_init_driver.py'),
        'allocation_route': descriptor(PHASE/'protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json'),
        'root_decision': descriptor(decision), 'phase_payload': descriptor(payload),
        'output': str(Path(ANCHOR)/'GRAPH_INIT_ATTEMPT_REGISTRY.json'),
        'receipt_directory': str(base/'root_receipts/register_v1'),
        'supervisor_directory': str(base/'supervision/register_v1_inner'),
        'outer_supervisor_directory': str(base/'supervision/register_v1_outer'),
        'local_launch_receipt': str(base/'supervision/register_v1_LAUNCH.json'),
        'whole_cap_seconds': 600, 'deterministic': True,
        'heldout_release_requires_all72_completed_and_separate_admission': True}''')
text=text.replace("'prior_attempt_registries': [],","'prior_attempt_registries': prior_registries(),")
# Retained previous source evidence lives at its original v2 location.
text=text.replace("descriptor(PHASE/R17_REL/'review_evidence/FINDINGS.json')", "descriptor(PHASE/'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2/review_evidence/FINDINGS.json')")
start=text.index("    history = {'schema': 'graph-init-source-only-predecessor-history-v1'")
end=text.index("    write(out/'PREDECESSOR_HISTORY_DRAFT.json', history)",start)
text=text[:start]+'''    history = {'schema': 'graph-init-precision-predecessor-history-v1', 'disclosure': DISCLOSURE,
        'prior_R17_numerical_attempt_registries': prior_registries(),
        'lineage_inventory': descriptor(HERE/'LINEAGE_BINDINGS.json'),
        'all_prior_text_receipts_protected': True, 'old_registry_remains_failed': True,
        'old_checkpoint_phase_output_qualification_optimizer_history_inherited': False,
        'six_cold_and_every_actual_warm_requalification_required': True,
        'predecessor_history_complete': False, 'root_independent_lineage_review_required': True,
        'evidence': lineage_descriptors()}
''' +text[end:]
text=text.replace('Fresh independent sealed-R17-v2','Fresh independent sealed-R17-v3 precision')
# Remove initial-qualifier admission/decision from prepare: all six belong to single finite72 decision.
start=text.index('    first = contexts[0]\n')
end=text.index("    write(out/'PREPARE_RECEIPT.json'",start)
text=text[:start]+'''    write(out/'ROOT_REGISTER_REQUEST_DRAFT.json', launch_request('register', out/'REGISTRY_REQUEST_DRAFT.json',
          out/'ROOT_REGISTRATION_DECISION_TEMPLATE.json', out))
''' +text[end:]
start=text.index('    # This fresh template has the admitted context hash; registry SHA remains pending.')
end=text.index('    return out',start)
text=text[:start]+text[end:]
# Delete obsolete first-cold action and function entirely.
start=text.index('def make_qualify(')
end=text.index('\n\ndef main(',start)
text=text[:start]+text[end:]
text=text.replace("for command in ('finalize-registration', 'make-qualify'):","for command in ('finalize-registration',):")
text=text.replace("    else:\n        out = make_qualify(args.decision, args.output)\n",'')
emit(old,text)

old=OLD_ROOT/'graph_init_root_entry.py'
text=basic(old.read_text()).replace('only registry metadata or first cold qualification','only fresh registry metadata')
text=text.replace("request['action'] in ('register', 'qualify')", "request['action'] == 'register'")
text=text.replace("        require(payload['prior_attempt_registries'] == [], 'Initial source-only lineage has no numerical predecessor')",
 "        from lineage_support import verify_lineage, prior_registries\n        verify_lineage()\n        require(payload['prior_attempt_registries'] == prior_registries(), 'Complete failed v2 registry lineage required')")
text=text.replace("lineage['prior_attempt_registries'] == []", "lineage['prior_attempt_registries'] == prior_registries()")
start=text.index('    else:\n',text.index("    if action == 'register':"))
end=text.index('    # Root receipt belongs',start)
text=text[:start]+text[end:]
text=text.replace("anchor = Path(payload['anchor_directory'] if action == 'register' else registry['anchor_directory'])", "anchor = Path(payload['anchor_directory'])")
emit(old,text)

old=OLD_CONT/'build_continuation.py'
text=basic(old.read_text())
text=replace_function(text,'protected', '''def protected():
    from build_admissions import protected_metadata
    records = protected_metadata()+deployment_ancillary()
    for name in ('build_admissions.py', 'build_continuation.py', 'continuation_support.py',
                 'continuation_entry.py', 'finite_coordinator.py', 'lineage_support.py', 'launch_registration.py'):
        records.append(descriptor(HERE/name))
    unique = {r['path']: r for r in records}
    return [unique[key] for key in sorted(unique)]''')
start=text.index("        'initial_qualification_freeze':")
end=text.index("        'coordinator_run_root':",start)
text=text[:start]+'''        'finite_plan': [], 'finite_plan_sha256': None,
        'registration_launch_request': {'path': str(REMOTE_PHASE/HERE.name/'admitted_v1/ROOT_REGISTER_REQUEST.json'), 'sha256': None},
        'registration_whole_supervision_terminal': {'path': str(REMOTE_PHASE/HERE.name/'supervision/register_v1_outer/TERMINAL.json'), 'sha256': None},
        'registration_root_terminal': {'path': str(REMOTE_PHASE/HERE.name/'root_receipts/register_v1/TERMINAL.json'), 'sha256': None},
''' +text[end:]
text=text.replace("'qualify': 5", "'qualify': 6").replace('All remaining cold qualifiers','All six fresh cold qualifiers')
start=text.index("        first = row_for(registry",text.index('def template'))
end=text.index('    return decision',start)
text=text[:start]+'''        decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
            'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
            'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
        for name in ('registration_launch_request', 'registration_whole_supervision_terminal', 'registration_root_terminal'):
            if mirror(decision[name]['path']).is_file():
                decision[name] = descriptor(decision[name]['path'])
''' +text[end:]
start=text.index("    first = row_for(registry",text.index('def prepare_decision'))
end=text.index('    write(mirror(output), decision)',start)
text=text[:start]+'''    require(completed == {}, 'Fresh precision registry must contain zero previous phase attempts')
    plan = finite_plan(registry)
    decision.update({'attempt_registry': record, 'study_id': registry['study_id'],
        'anchor_directory': registry['anchor_directory'], 'contexts_sha256': object_hash(registry['contexts']),
        'finite_plan': plan, 'finite_plan_sha256': object_hash(plan)})
    for name in ('registration_launch_request', 'registration_whole_supervision_terminal', 'registration_root_terminal'):
        decision[name] = descriptor(decision[name]['path'])
    whole = read(decision['registration_whole_supervision_terminal']['path'])
    root = read(decision['registration_root_terminal']['path'])
    require(whole['complete'] is True and whole['within_whole_cap'] is True and
            whole['root_request_unchanged'] is True and root['completed'] is True,
            'Successful new registration whole/root supervision required')
''' +text[end:]
text=text.replace("    first = row_for(registry, [c for c in registry['contexts'] if (c['graph'], c['seed']) == FIRST_CELL][0], 'qualify')\n",'')
text=text.replace("set(completed) == {first['key']} | set(done)","set(completed) == set(done)")
emit(old,text)

emit(OLD_CONT/'continuation_entry.py',basic((OLD_CONT/'continuation_entry.py').read_text()))
old=OLD_CONT/'finite_coordinator.py'
text=basic(old.read_text()).replace("'attempt_count': 71", "'attempt_count': 72").replace("'remaining_attempts': 71", "'remaining_attempts': 72")
text=text.replace("            phase_start = time.monotonic()", "            write(out/(key+'_LAUNCH_PARENTS.json'), prepare_launch_parents(request))\n            phase_start = time.monotonic()")
start=text.index('def bootstrap(')
text=text[:start]+'''def prepare_launch_parents(request):
    """Create local receipt and remote outer/inner parents before any launcher call."""
    local_receipt = mirror(request['local_launch_receipt'])
    require(not local_receipt.exists(), 'Single-use local launcher receipt required')
    local_receipt.parent.mkdir(parents=True, exist_ok=True)
    parents = sorted({str(Path(request[k]).parent) for k in
        ('supervisor_directory', 'outer_supervisor_directory', 'local_launch_receipt')})
    require(all(Path(p).is_relative_to(REMOTE_PHASE/HERE.name) for p in parents),
            'Remote launch parents must belong to current wrapper')
    code = "import pathlib,sys; base=pathlib.Path(sys.argv[1]); paths=[pathlib.Path(x) for x in sys.argv[2:]]; "+\\
           "assert all(p.is_absolute() and p.is_relative_to(base) and '..' not in p.parts for p in paths); "+\\
           "assert all(not q.is_symlink() for p in paths for q in [p,*p.parents] if q.is_relative_to(base)); "+\\
           "[p.mkdir(parents=True,exist_ok=True) for p in paths]"
    command = shlex.join([str(REMOTE_REPO/'.venv/bin/python'), '-c', code, str(REMOTE_PHASE/HERE.name), *parents])
    subprocess.run([*SSH, command], capture_output=True, text=True, check=True, timeout=60)
    return {'UTC': utc(), 'local_receipt_parent': str(local_receipt.parent),
            'remote_supervision_parents': parents, 'scientific_execution': False,
            'original_diagnostic_receipt_parent_failure_preserved': True}


''' +text[start:]
emit(old,text)
(HERE/'ASSEMBLY_ORIGINS.json').write_text(json.dumps({'source_only':True,'runtime_monkeypatching':False,
    'policy':'All revisions are materialized source files with explicit unified diff; original wrappers remain unchanged.',
    'files':origins},indent=2)+'\n')
(HERE/'REUSE_DIFF.patch').write_text(''.join(diffs))
print('Materialized',len(origins),'explicit wrapper revisions')

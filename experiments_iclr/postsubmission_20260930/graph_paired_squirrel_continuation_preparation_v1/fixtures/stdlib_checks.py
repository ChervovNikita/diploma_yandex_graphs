"""Focused stdlib/source/mock checks; no native, remote, GPU or numeric originals."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

packet = Path(__file__).resolve().parents[1]
source_path = packet/'prototype/continue_squirrel.py'
source = source_path.read_bytes()
tree = ast.parse(source)
frozen = json.loads((packet/'FROZEN_STUDY.json').read_text())
checks = []
sha = lambda data: hashlib.sha256(data).hexdigest()


def check(name, condition):
    assert condition, name
    checks.append(dict(name=name, passed=True))


def rejects(name, callback):
    try:
        callback()
    except ValueError:
        check(name, True)
    else:
        check(name, False)


def originals(when):
    count = 0
    for row in frozen['original_records']:
        if Path(row['path']).suffix in ('.npy','.npz','.pt','.pth'):
            continue
        relative = Path(row['path']).relative_to(frozen['canonical_research_root'])
        data = (packet.parent/relative).read_bytes()
        check(when+':metadata_unchanged:'+str(relative), sha(data)==row['sha256'] and len(data)==row['bytes'])
        count += 1
    return count


before = originals('before')
spec = importlib.util.spec_from_file_location('prepared_Squirrel_stdlib_subject', source_path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
check('syntax_stdlib_import_no_Torch', isinstance(tree, ast.Module) and 'torch' not in sys.modules)
subject.study_guard(frozen)
check('frozen_Squirrel_three_blocks_twelve_fits', len(frozen['cells'])==3 and frozen['fit_slots']==12)
check('seventeen_numeric_descriptors_only', sum(Path(row['path']).suffix in ('.npy','.npz','.pt','.pth')
      for row in frozen['original_records'])==17)
check('original_label_records_TRAIN_only', all(row['kind']=='TRAIN_labels' for row in frozen['original_records']
      if '/labels/' in row['path']))
for key,value in [('mode','fixed_two_graph'),('selector','last epoch'),('continuation',dict(cap=1,patience=250,midpoint=950))]:
    rejects('changed_frozen_policy_refused:'+key, lambda key=key,value=value:
            subject.study_guard(dict(frozen, **{key:value})))
with patch.object(subject,'verify',return_value=Path('/mock/not/opened')) as mocked_verify:
    rejects('final_label_refused_before_open', lambda: subject.verify_inputs([
        dict(path='/mock/labels/final.npz',kind='validation_labels',sha256='0'*64)],allow_validation=True))
    rejects('validation_refused_before_admission', lambda: subject.verify_inputs([
        dict(path='/mock/labels/seed17_split0_validation.npz',kind='validation_labels',sha256='0'*64)]))
    check('forbidden_label_checks_opened_nothing', mocked_verify.call_count==0)
    subject.verify_inputs([dict(path='/mock/labels/seed17_split0_validation.npz',kind='validation_labels',sha256='0'*64)],
                          allow_validation=True)
    check('admitted_validation_descriptor_reaches_verifier', mocked_verify.call_count==1)

choice_relative = Path(frozen['mode_donor_freeze']['path']).relative_to(frozen['canonical_research_root'])
choice_path = packet.parent/choice_relative
release = dict(execution_authorized=True,run_name='mock01',prepared_manifest_sha256='mock_manifest',
               mode_donor_freeze_sha256=frozen['mode_donor_freeze']['sha256'])
with patch.object(subject,'verify',return_value=choice_path):
    admitted = subject.admission_guard(frozen,release,'mock_manifest','mock01')
    check('root_admission_binds_mode_six_donors', admitted['mode']=='staged_validation_gate'
          and len(admitted['six_donors'])==6)
    for key,value in [('execution_authorized',False),('prepared_manifest_sha256','changed'),
                      ('mode_donor_freeze_sha256','changed'),('run_name','changed')]:
        rejects('execution_release_mismatch_refused:'+key,lambda key=key,value=value:
                subject.admission_guard(frozen,dict(release,**{key:value}),'mock_manifest','mock01'))


def blocks(values):
    return [dict(seed=seed,source_split_index=split,status='completed',fits={arm:dict(status='selected',
            selection=dict(primary_validation_nll=values[arm])) for arm in subject.ARMS})
            for seed,split in subject.PAIRS]


complete = blocks(dict(common_only=2.0,train_remasked=1.8,full_node=1.0,full_node_permuted=1.6))
gate = subject.development_gate(complete)
check('complete_twelve_fit_gate_positive', gate['Photo_trigger'] is True and gate['all12_selected_fits'])
check('gate_never_launches_Photo_or_opens_final_labels', gate['Photo_not_launched'] and gate['final_labels_closed'])
tie = blocks(dict(common_only=2.0,train_remasked=1.0,full_node=1.0,full_node_permuted=1.6))
check('strict_tie_gate_negative', subject.development_gate(tie)['Photo_trigger'] is False)
partial = copy.deepcopy(complete); partial[1]['fits']['train_remasked']['status']='resource_deferred'
partial_gate = subject.development_gate(partial)
check('failed_block_no_successful_subset_scoring', partial_gate['Photo_trigger'] is None
      and partial_gate['successful_subset_scored'] is False and 'validation_macro_NLL' not in partial_gate)
pending = copy.deepcopy(complete); pending[2]['fits']['full_node']['status']='pending'
check('pending_terminal_gate_closed', subject.development_gate(pending)['all12_arm_terminals'] is False)
rejects('missing_block_gate_refused',lambda:subject.development_gate(complete[:2]))
nonfinite = blocks(dict(common_only=2.0,train_remasked=1.8,full_node=float('inf'),full_node_permuted=1.6))
rejects('nonfinite_selected_NLL_refused',lambda:subject.development_gate(nonfinite))
text = source.decode()
check('all_four_installations_before_validation', text.index("require(set(result['initializations']) == set(ARMS)")
      < text.index('verify_inputs([cell[\'validation_labels\']], allow_validation=True)'))
check('actual_R_S_trainable_scope_checked', "'Actual continuation must learn R/S'" in text)
check('no_restart_marker_guard', "'*/PRE_EXECUTION_ADMISSION.json'" in text)
check('resource_sentinel_after_paired_return', 'status = shared.paired_return_status(slices, resources)' in text)
calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)]
check('old_phase_registry_entrypoints_absent', not any(isinstance(node.func,ast.Attribute) and node.func.attr in
      {'context_guard','modern_certificate_guard','register','phase_run','compare','report','warm_native'} for node in calls))
check('no_direct_remote_or_environment_mutation', all(value not in text for value in ('ssh ','scp ','rsync ','os.environ')))

# Both early main paths use fake source bytes and a mocked probe. The real frozen
# contexts are metadata only; all original numeric descriptors are replaced.
scratch = packet/'fixtures/scratch'; scratch.mkdir(exist_ok=True)
for status in ('resource_deferred','resource_preflight_passed'):
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory); fake = root/'source.txt'; fake.write_text('mock source only\n')
        fake_frozen = dict(frozen,original_records=[dict(path=str(fake),kind='source_metadata',sha256=sha(fake.read_bytes()))],
                           shared_source=dict(path=str(fake),sha256=sha(fake.read_bytes())))
        (root/'FROZEN_STUDY.json').write_text(json.dumps(fake_frozen))
        data = json.dumps(dict(payload=[])).encode()
        (root/'MANIFEST.json').write_bytes(data)
        (root/'SEAL.json').write_text(json.dumps(dict(manifest_sha256=sha(data))))
        with patch.object(subject,'PACKET',root),patch.object(subject,'load',return_value=SimpleNamespace(
                resource_preflight=lambda frozen,evidence:dict(status=status))),contextlib.redirect_stdout(io.StringIO()):
            code = subject.main(['--run-name','mock','--preflight-only'])
        result = json.loads((root/'runs/mock/STAGE1.json').read_text())
        check('mock_main_no_native:'+status, code==0 and result['blocks']==[] and result['validation_blocks_loaded']==0)
        check('mock_main_closed_final:'+status, result['final_labels_closed'] and result['Photo_launched'] is False)
        check('mock_main_preservation:'+status,result['originals_before']==result['originals_after'])
        check('mock_main_no_execution_marker:'+status,not (root/'runs/mock/PRE_EXECUTION_ADMISSION.json').exists())
after = originals('after')
check('all_mock_paths_no_Torch', 'torch' not in sys.modules)
result = dict(schema='paired-Squirrel-stdlib-source-mock-checks-v1',checks=checks,passed=True,
              source_sha256=sha(source),nonnumeric_originals_unchanged=after,
              author_native_remote_GPU_or_numeric_original_execution=False,
              actual_GPU_probe=False,validation_or_final_label_bytes_opened=False)
(packet/'STDLIB_RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),source_sha256=sha(source),nonnumeric_originals_unchanged=after,
                     native_or_GPU_or_numeric_original_execution=False)))

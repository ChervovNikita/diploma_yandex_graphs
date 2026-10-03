"""Fabricated receipt-only regression; no model, tensor, data or runtime import."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import importlib.util,json,sys,tempfile

HERE=Path(__file__).resolve().parent


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')
    return {'path':str(path),'sha256':sha256(path.read_bytes()).hexdigest()}


def relative_receipt(path):
    return {'path':path.name,'bytes':path.stat().st_size,'sha256':sha256(path.read_bytes()).hexdigest()}


def cost():
    return {'closed_attempt_wall_seconds':1.,'observed_interrupted_wall_seconds':0.,'total_cost_exact':True}


class Attempts:
    def __init__(self):self.phases=[]
    def phase(self,name):self.phases.append(name)


def fixtures(root,failure_count):
    identity={'fixture_only':'synthetic_metadata_never_driver_admitted'}
    selections={};fits={}
    streams=[{'epoch':i,'start_rng_sha256':'synthetic-start','end_rng_sha256':'synthetic-end',
              'stream':{'full_batches':17},'support_digests':[{'positive':'synthetic-positive','negative':'synthetic-negative'}]*17} for i in range(1,101)]
    for arm in ('J','F'):
        directory=root/('fit_'+arm);directory.mkdir()
        checkpoint=directory/('SELECTED_'+arm+'.pt');checkpoint.write_bytes(b'fabricated receipt placeholder, not a Torch checkpoint')
        selection={'candidate_id':'fixture_epoch_1','order':1,'hits50':.25 if arm=='J' else .5}
        selections[arm]=selection
        selected=relative_receipt(checkpoint)
        private=directory/('PRIVATE_SELECTION_'+arm+'.json')
        write(private,{'identity':identity,'arm':arm,'seed':0,'checkpoint':selected,'selection':selection})
        journal=directory/'JOURNAL.json';write(journal,{'fixture_only':True})
        receipt={'schema':'ncnc-pattern-complete-fit-v1','identity':identity,'arm':arm,'seed':0,'epochs':100,'optimizer_steps':1700,
            'selection_candidates':100,'full_VALID_evaluations':102,'selected_roundtrip_and_full_served_replay':True,'resource_state_donor':False,
            'test_file_opened':False,'selected_checkpoint':selected,'journal':relative_receipt(journal),'private_selection':relative_receipt(private),
            'epoch_streams':streams,'initial_state_sha256':'synthetic-initial','initial_rng_sha256':'synthetic-rng','teacher':{'fixture_only':True},'inclusive_accounting':cost()}
        fits[arm]=write(directory/'COMPLETE.json',receipt)
    directory=root/'diagnostics';directory.mkdir()
    private=directory/'PRIVATE_DIAGNOSTICS.json'
    write(private,{'identity':identity,'arms':{arm:{'VALID':{'served_hits50':selection['hits50']},'selected_epoch':1} for arm,selection in selections.items()}})
    diagnostics=write(directory/'COMPLETE.json',{'schema':'ncnc-pattern-diagnostics-complete-v1','identity':identity,'arms':['J','F'],'status':'COMPLETE',
        'full_selected_VALID_evaluations':2,'full_TRAIN_mask_epochs':2,'matched_mask_supports':True,'test_file_opened':False,
        'private_diagnostics':relative_receipt(private),'inclusive_accounting':cost()})
    qualifications={stage:write(root/stage/'FAKE_QUALIFICATION.json',{'schema':'fixture_only_never_driver_admitted','inclusive_accounting':cost()}) for stage in ('numerical','full_graph')}
    failures=[write(root/('prior_failed_'+str(i))/'FAILED.json',{'schema':'ncnc-pattern-failed-stage-v1','identity':identity,'status':'FAILED','inclusive_accounting':cost()}) for i in range(failure_count)]
    return {'identity':identity,'release':{'fit_receipts':fits,'diagnostics_receipt':diagnostics,'qualification':qualifications,'prior_failure_receipts':failures}}


def main():
    forbidden={'torch','numpy','pandas','prototype','graph_ops','pattern_model','pilot_data'}
    assert not forbidden.intersection(sys.modules)
    sys.path.insert(0,str(HERE))
    spec=importlib.util.spec_from_file_location('receipt_only_pattern_close',HERE/'pattern_close.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert Path(sys.modules['pilot_common'].__file__).resolve()==HERE/'pilot_common.py'
    results=[]
    with tempfile.TemporaryDirectory(prefix='.closure_receipt_fixture_',dir=HERE) as name:
        for count in (0,1,2):
            directory=Path(name)/('case_'+str(count));directory.mkdir()
            context=fixtures(directory,count);output=directory/'close';output.mkdir()
            attempts=Attempts();receipt=module.run(context,output,attempts)
            assert receipt['diagnostics_receipt']==context['release']['diagnostics_receipt']
            assert receipt['diagnostics_receipt'] not in context['release']['prior_failure_receipts']
            assert receipt['prior_failure_receipts']==context['release']['prior_failure_receipts']
            assert len(receipt['preclosure_bound_stage_accounting'])==5+count
            assert receipt['preclosure_bound_closed_attempt_wall_seconds']==5.+count
            results.append({'prior_failure_receipts':count,'diagnostics_binding_preserved':True,'failures_retained_and_charged':True})
    assert not forbidden.intersection(sys.modules)
    result={'schema':'ncnc-pattern-V2-closure-custody-fixture-v1','UTC':datetime.now(timezone.utc).isoformat(),'status':'PASS_FABRICATED_STDLIB_RECEIPTS_ONLY',
        'cases':results,'executed_source_hashes':{p.name:sha256(p.read_bytes()).hexdigest() for p in (HERE/'pattern_close.py',HERE/'pilot_common.py',Path(__file__))},
        'real_project_metrics_data_checkpoints_or_models_opened':False,'numerical_or_runtime_code_imported':False,
        'scope':'closure diagnostics pin remains exact with zero/one/two prior failure receipts; no scientific or numerical qualification',
        'temporary_fabricated_receipts_removed':True}
    write(HERE/'CLOSURE_CUSTODY_FIXTURE.json',result)
    print(json.dumps({'status':result['status'],'cases':len(results)}))


if __name__=='__main__':main()

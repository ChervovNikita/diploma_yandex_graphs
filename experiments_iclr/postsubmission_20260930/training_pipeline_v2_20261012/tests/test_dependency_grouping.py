"""Independent method DAGs may share a study while staying host-local."""
import copy
import gzip
from pathlib import Path
import shutil
import tempfile
import unittest

from compile_master_plan import (CONTROLLER_FILES, assemble, controller_bindings,
                                  fit_semantics, readiness, receipt)
from partition import DEFAULT_WORKERS
from spec import canonical, fingerprint
from tests.test_master_plan import study, PACKAGE


class DependencyGrouping(unittest.TestCase):
    def test_separate_arms_balance_across_gpus_without_splitting_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            runtime=root/'controller';runtime.mkdir()
            for name in CONTROLLER_FILES:shutil.copyfile(PACKAGE/name,runtime/name)
            value=study('multiple_arms')
            for cohort in ('b','c'):
                extra=copy.deepcopy(study('extra')['runs'])
                for row in extra:
                    row['run_id']=cohort+'-'+row['run_id']
                    row['parents']=[cohort+'-'+x for x in row['parents']]
                    if 'fit_parents' in row['config']:
                        row['config']['fit_parents']=[cohort+'-'+x for x in row['config']['fit_parents']]
                value['runs']+=extra
            (root/'study.json.gz').write_bytes(gzip.compress(canonical(value).encode(),mtime=0))
            plan={'schema_version':1,'slots_per_gpu':2,'scientific_launch_enabled':False,'human_GO_required':True,
                  'studies':[{'path':'study.json.gz','spec_hash':fingerprint(value)}],
                  'groups':[{'id':'old_whole_study','gpu_assignment':'allocation/gpu0',
                             'run_keys':[value['study_id']+'/'+r['run_id'] for r in value['runs']]}]}
            (root/'PLAN.json').write_text(canonical(plan));(root/'SUMMARY.json').write_text(canonical({'plan_hash':fingerprint(plan)}))
            entry={'id':'packet','plan':receipt(root,'PLAN.json'),'summary':receipt(root,'SUMMARY.json')}
            packets,ready=readiness(root,{'schema':'disabled-master-input-index-v1','packets':[entry]})
            output,specs,summary=assemble(root,packets,ready,'controller',controller_bindings(root,'controller'),
                {'group_granularity':'dependency','worker_capacities_cuda_bytes':{w:100*1024**2 for w in DEFAULT_WORKERS}},'master',[])
            self.assertEqual(len(output['groups']),3)
            self.assertEqual({g['gpu_assignment'] for g in output['groups']},set(DEFAULT_WORKERS))
            placed={key:g['gpu_assignment'] for g in output['groups'] for key in g['run_keys']}
            for row in value['runs']:
                key=value['study_id']+'/'+row['run_id']
                for parent in row['parents']:
                    self.assertEqual(placed[key],placed[value['study_id']+'/'+parent])
            self.assertEqual(summary['per_study'][0]['workers'],sorted(DEFAULT_WORKERS))
            self.assertIsNone(summary['per_study'][0]['worker'])
            self.assertEqual(summary['counts']['totals']['paid_optimizer_starts'],9)
            self.assertEqual(sum(v['counts']['totals']['paid_optimizer_starts'] for v in summary['per_worker'].values()),9)
            self.assertFalse(output['scientific_launch_enabled'])

    def test_declared_molecular_minibatch_calls_do_not_hide_epochs(self):
        row=study('molecule',molecular=True)['runs'][0]
        self.assertIsNone(fit_semantics(row)['known_optimizer_step_calls'])
        row['config']['train_minibatches_per_epoch']=258
        row['config']['architecture']='original_gnnm'
        counts=fit_semantics(row)
        self.assertEqual(counts['known_optimizer_step_calls'],25800)
        self.assertEqual(counts['optimizer_objects'],1)
        row['config']['train_minibatches_per_epoch']=0
        with self.assertRaises(ValueError):fit_semantics(row)

if __name__=='__main__':unittest.main()

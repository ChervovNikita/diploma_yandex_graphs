"""Minimal final-command successor of the actual PortableBackend.

All admission/probing/identity/poll/reap methods are inherited. start retains
the original reservation→Popen→identity-commit transaction, with only the
fixed final request argv substituted. It is the sole process-start owner.
"""
import os
from pathlib import Path
import subprocess
from .common import HERE, digest, exclusive, pin, require, verify
from .projection import build_request
from .preparation import make_capsule


def backend_class(portable):
    class FinalBackend(portable.PortableBackend):
        def __init__(self, config, ownership, budget, recheck, ledger, spec, spec_pin, release, seal, go, profiles, training_index, preparation_pin):
            self.config, self.ownership, self.budget = config, ownership, budget
            self.recheck = recheck
            self.root = Path(config['project_root'])
            self.verify = lambda value: verify(value, self.root)
            self.here, self.ledger = HERE, ledger
            self.assignments = {row['id']: row for row in ownership['assignments']}
            self.tokens, self.identities, self.requests = {}, {}, {}
            self.baseline_stops = set()
            self.supervisor_identity = portable.identity(os.getpid())
            self.spec, self.spec_pin, self.release, self.seal, self.go = spec, spec_pin, release, seal, go
            self.profiles = profiles
            self.final_index = {row['id']: row for row in release['tasks']}
            self.training_index, self.preparation_pin = training_index, preparation_pin
            self.output_root = Path(release['output_root'])

        def request(self, task, states):
            value = build_request(task, self.release, self.profiles[task['preset_id']], self.spec['training_seal'], self.seal, self.go['final_scoring'])
            if task['kind'] == 'I4_TEST_aggregate':
                value['component_results'] = [states[key]['live_native_result_receipt'] for key in task['dependencies']]
                for result in value['component_results']:
                    verify(result, self.root, json_metadata=True)
            return value

        def start(self, task, output, checkpoint, log, states):
            ticket = self.ledger.ticket
            require(ticket is not None, 'Actual Popen requires locked startup reservation')
            token = ticket['token']
            try:
                require(checkpoint is None, 'Final scoring never resumes or repeats TEST')
                execution = task['binding']['execution']
                entry, environment = execution['runtime_entrypoint'], execution['python_environment']
                self.verify(entry); self.verify(environment); self.recheck(task)
                request = self.request(task, states)
                request_path = Path(log.name).parent / 'REQUEST.json'
                exclusive(request_path, request)
                request_pin = pin(request_path, self.root)
                self.requests[task['id']] = request_pin
                capsule = make_capsule(task, self.release['output_root'], self.final_index, self.training_index, self.seal, self.profiles[task['preset_id']], self.preparation_pin)
                capsule_path = Path(log.name).parent / 'ADMISSION_CAPSULE.json'
                exclusive(capsule_path, capsule)
                capsule_pin = pin(capsule_path, self.root)
                # This exact active row is committed by the original identity
                # transaction below. No independent observer owns reservation.
                ticket['state']['active'][token]['final_admission'] = {
                    'preparation': self.preparation_pin, 'capsule': capsule_pin, 'request': request_pin}
                argv = [execution['python_executable'], '-B', '-u', str(HERE / 'qualified_python_entry.py'),
                        '--environment', environment['path'], '--environment-sha256', environment['sha256'],
                        '--entrypoint', entry['path'], '--entrypoint-sha256', entry['sha256'], '--',
                        '--request', request_pin['path'], '--request-sha256', request_pin['sha256'],
                        '--capsule', capsule_pin['path'], '--capsule-sha256', capsule_pin['sha256']]
                threads = str(task['binding']['resources']['thread_count'])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.config['physical_gpu_uuid'],
                           OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                           PYTHONDONTWRITEBYTECODE='1')
                child = subprocess.Popen(argv, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException as exc:
                self.ledger.cancel_unspawned(str(exc))
                raise
            # Exact parent transaction: after Popen all errors return the child
            # to Engine for reap/drain; no second component may own this child.
            self.tokens[task['id']] = token
            ticket['child_created'] = True
            try:
                observed = portable.identity(child.pid)
            except Exception as exc:
                observed = {'pid': child.pid, 'identity_unavailable': str(exc)}
            self.identities[child.pid] = observed
            try:
                self.ledger.commit(observed)
            except BaseException as exc:
                try:
                    self.ledger.stop({'reason': 'child_identity_commit_failed', 'error': str(exc), 'pid': child.pid})
                except OSError:
                    pass
                self.control_error = str(exc)
            return child

        def validate_result(self, task, raw):
            require(raw['campaign_GO'] == self.spec['training_go'] and raw['training_seal'] == self.spec['training_seal'], 'Foreign final GO/selection seal')
            profile = verify(self.profiles[task['preset_id']], self.root, json_metadata=True)
            origin = task.get('training_fit_id', task.get('training_I4_id'))
            section = 'fit_specs' if task['kind'] == 'selected_fit_TEST' else 'I4_specs'
            require(raw['identity'] == profile[section][origin]['identity'] and raw['role_contract'] == task['role_contract'], 'Foreign final predictor/role identity')
            if task['kind'] == 'selected_fit_TEST':
                requests = list((self.output_root / 'control/tasks' / digest(task['id']) / 'attempts').glob('*/REQUEST.json'))
                require(len(requests) == 1, 'Final fit must have exactly one retained request, never a hidden retry')
                request = verify(pin(requests[0]), self.root, json_metadata=True)
                # Frozen scorer uses its own exact canonical request hash.
                import hashlib, json
                expected = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                require(raw['request_sha256'] == expected, 'Final terminal request hash differs')
    return FinalBackend

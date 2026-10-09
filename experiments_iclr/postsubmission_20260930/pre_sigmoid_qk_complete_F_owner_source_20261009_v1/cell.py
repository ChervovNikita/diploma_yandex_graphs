"""Private-instance integration of the exact original full native F selector loop."""
from contextlib import contextmanager
from pathlib import Path
import sys
from common import HERE, PHASE, bound, module, read, require, sha, source_identity


@contextmanager
def aliases(values):
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def run_cell(*, spec, output, cfg, pins, manifest, release_sha256, authorized=False):
    require(authorized is True and spec['seed'] in cfg['seeds']
            and spec['operator'] in cfg['operators'] and spec['kind'] in cfg['kinds']
            and not output.exists(), 'Fresh exact released cell only; no retry/resume')
    adapter = module(PHASE / pins['adapter_directory'] / 'operator_adapter.py', '_complete36_immutable_adapter')
    public_root = PHASE / pins['public_directory']
    public = module(public_root / 'portable.py', '_complete36_original_public')
    with aliases({'portable': public}):
        data = module(public_root / 'data_interface.py', '_complete36_original_roles')
        with aliases({'data_interface': data}):
            driver = module(bound(PHASE, pins['original_driver']), '_complete36_original_complete_driver')
    box = {}
    original_write = driver.json_write
    context = dict(cell=spec, release_sha256=release_sha256, execution_source_commit=cfg['execution_source_commit'],
                   qualification_adoption=cfg['qualification_adoption'],
                   source=source_identity(pins, manifest), roles=pins['roles'], runtime=pins['runtime'])

    def constructor(task, kind, seed, device, polynormer=None, ncn_model=None, ncn_utils=None):
        require(task == 'wikics' and kind == spec['kind'] and seed == spec['seed'] and device == 'cuda:0'
                and Path(polynormer).resolve() == (PHASE / pins['native']['path']).resolve(), 'Original constructor arguments')
        session = adapter.make_session(operator=spec['operator'], kind=kind, seed=seed,
                                       device=device, later_execution_authorized=True)
        box['session'] = session
        original_cpu = session._cpu_tree

        def snapshots(value):
            result = original_cpu(value)
            if isinstance(value, dict) and 'model' in value and ('epoch' in value or value.get('evaluation_only') is True):
                result.update(operator_binding=dict(session.operator_binding), operator_work=dict(session.operator_work),
                              owner_context=context)
            return result
        session._cpu_tree = snapshots
        return session

    def receipts(path, value):
        path = Path(path)
        if path.name in ('VALID_TRACE.json', 'OWN_BEST_BANK.json'):
            path = path.with_name('PRIVATE_' + path.name)
        session = box.get('session')
        if isinstance(value, dict) and path.name in ('RUN.json', 'PROGRESS.json', 'COMPLETE.json', 'FAILURE.json'):
            value.update(owner_context=context, operator_binding=dict(session.operator_binding) if session else None,
                         operator_work=dict(session.operator_work) if session else None)
            if path.name == 'COMPLETE.json':
                members = 1 if spec['kind'] == 'single' else 4
                adam = 4 if spec['kind'] == 'independent4' else 1
                require(session.operator_work == dict(update_attempts=1100, completed_updates=1100,
                    member_view_forwards=2*members*1100, member_view_backwards=2*members*1100, Adam_steps=adam*1100),
                    'Complete old-state F work and native Adam inventory')
                paths = [output / 'RUN.json', output / 'PRIVATE_VALID_TRACE.json', output / 'selected_local.pt',
                         output / 'selected.pt']
                if session.model.independent:
                    paths += [output / ('own_' + which + '_' + str(m) + '.pt')
                              for which in ('local', 'best') for m in range(members)]
                if spec['kind'] == 'independent4':
                    paths += [output / 'PRIVATE_OWN_BEST_BANK.json']
                value['bound_files'] = {str(p.relative_to(output)): sha(p) for p in paths}
                value['all1100_and100_schedule'] = True
                value['comparative_opening_performed'] = False
        original_write(path, value)

    driver.Session, driver.json_write = constructor, receipts  # New private module instance only.
    previous = sys.argv
    try:
        sys.argv = [str(bound(PHASE, pins['original_driver'])), '--task', 'wikics', '--arm', spec['kind'],
                    '--seed', str(spec['seed']), '--device', 'cuda:0', '--train', str(bound(PHASE, pins['roles']['train'])),
                    '--valid', str(bound(PHASE, pins['roles']['valid'])), '--output', str(output),
                    '--polynormer', str(bound(PHASE, pins['native']))]
        driver.main()  # Exact original epoch/selection/own-restoration implementation.
    finally:
        sys.argv = previous

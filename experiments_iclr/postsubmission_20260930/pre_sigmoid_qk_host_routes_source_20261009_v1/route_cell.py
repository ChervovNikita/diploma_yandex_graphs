"""Reuse the reviewed cell/selector wrapper with the route-aware constructor only."""
from contextlib import contextmanager
from pathlib import Path
import sys
import types
import route_adapter as routing


@contextmanager
def aliases(values):
    absent = object()
    previous = {name:sys.modules.get(name,absent) for name in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for name,value in previous.items():
            if value is absent: sys.modules.pop(name,None)
            else: sys.modules[name]=value


def run_cell(*, route_id, spec, output, cfg, release_sha256, authorized=False):
    routing.require(authorized is True, 'Inactive route cell; bounded explicit root queue only')
    route,pins,manifest=routing.sources(route_id)
    routing.verify_route(route)
    routing.verify_roles(route,pins)
    routing.require(cfg['enabled'] is True and cfg['route_id']==route_id and spec in cfg['assigned_cells']
                    and cfg['qualification_adoption']['qualified'] is True,
                    'Fixed disjoint route assignment and successful route qualification required')
    root=routing.PHASE/pins['cell_directory']
    common=routing.module(root/'common.py','_route_original_cell_common')
    with aliases({'common':common}):
        cell=routing.module(root/'cell.py','_route_original_cell_wrapper')
    original_module=cell.module
    math_path=(routing.PHASE/pins['math_directory']/'operator_adapter.py').resolve()
    def constructor_module(path,name):
        if Path(path).resolve()==math_path:
            return types.SimpleNamespace(make_session=lambda **kwargs:routing.make_session(route_id=route_id,**kwargs))
        return original_module(path,name)
    cell.module=constructor_module  # The reviewed run_cell/selector source is unchanged.
    identity=cell.source_identity
    def routed_identity(pins,manifest):
        return dict(identity(pins,manifest), route_id=route_id,route_catalog_sha256=routing.sha(routing.HERE/'ROUTES.json'))
    cell.source_identity=routed_identity
    cell_pins=dict(routing.read(root/'SOURCE_BINDINGS.json'),runtime=route)
    cell.run_cell(spec=spec,output=Path(output),cfg=cfg,pins=cell_pins,manifest=manifest,
                  release_sha256=release_sha256,authorized=True)

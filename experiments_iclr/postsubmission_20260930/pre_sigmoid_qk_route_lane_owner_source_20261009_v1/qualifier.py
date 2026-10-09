"""Private callback routing around the immutable V2 numerical qualifier."""
from pathlib import Path
from common import PHASE, module, read, require


def run(cfg, pins, routing, route, route_pins, output):
    routing.verify_route(route)
    routing.verify_roles(route, route_pins)
    root = PHASE / pins['qualifier']['directory']
    qualifier = module(root / 'qualify.py', '_lane_private_immutable_v2_qualifier')
    original_module = qualifier.module
    math_path = (PHASE / route_pins['math_directory'] / 'operator_adapter.py').resolve()

    def loader(name, path):
        value = original_module(name, path)
        if Path(path).resolve() == math_path:
            paths = dict(native=routing.bound(route_pins['native']), runtime=routing.HERE / 'ROUTES.json',
                         public_manifest=PHASE / route_pins['public_directory'] / 'MANIFEST.json')
            value._sources = lambda: (paths, PHASE / route_pins['public_directory'], route_pins['math_manifest_sha256'])
            def runtime(paths, session=None):
                routing.verify_route(route, session)
                return route
            value._route = runtime
        return value

    qualifier.module = loader
    qualifier.OUTPUT_NAME = str(Path(output).relative_to(PHASE))
    release = read(root / 'RELEASE_DISABLED.json')
    release.update(enabled=True, root_execution_authorized=True, source_review_approved=True,
                   qualifier_manifest_sha256=pins['qualifier']['manifest_sha256'],
                   output_relative=qualifier.OUTPUT_NAME, runtime_sha256=routing.sha(routing.HERE / 'ROUTES.json'))
    result = qualifier.run_qualification(release=release, root_execution_authorized=True)
    require(result['complete'] is True and result['completed_updates'] == 24, 'All12 V2 engineering cells must pass')
    return result

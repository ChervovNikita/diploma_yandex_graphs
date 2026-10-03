"""Stdlib-only source/seal/body-custody check; never imports Torch or data."""
import ast
import inspect
import json

import block_policy
import common as c


def main():
    provenance, manifest_sha = c.packet_guard()
    compiled = []
    for name in ('block_policy.py', 'common.py', 'train_policies.py', 'qualify_policy.py', 'verify_source.py'):
        source = (c.PACKET/name).read_text()
        compile(source, str(c.PACKET/name), 'exec')
        compiled.append(name)
    driver = c.load('mixed_static_original_driver', c.PHASE/provenance['modules']['driver'])
    source = inspect.getsource(driver.fit)
    tree, custody = block_policy.fit_tree(source)
    compile(tree, 'mixed_fit_projection_static_only', 'exec')
    expected = json.loads((c.PACKET/'FIT_BODY_CUSTODY.json').read_text())
    c.require(custody == expected, 'Bound immutable fit custody differs')
    changed = source.replace('train_loss.backward()', '_mixed_backward(model, logits, ids, labels, train_loss)')
    c.require(ast.dump(ast.parse(changed), include_attributes=False) == ast.dump(tree, include_attributes=False),
              'Inspectable one-call text projection differs from runtime AST')
    own = block_policy.fit_function(None, None, driver, None, 'own/own', None)
    c.require(own is driver.fit, 'Own/own must import the exact original fit')
    result = dict(schema='mixed_policy_static_source_check_v1', status='passed',
        prepared_manifest_sha256=manifest_sha, compiled_source=compiled, fit_body_custody=custody,
        own_own_original_fit_identity=True, execution_release_remains_false=True,
        native_runtime_imported=False, data_or_labels_opened=False, fitted_state_or_outcomes_opened=False,
        numerical_execution_performed=False, runtime_or_resource_qualified=False)
    for name in ('TRAINING_RELEASE_TEMPLATE.json', 'QUALIFICATION_RELEASE_TEMPLATE.json'):
        template = json.loads((c.PACKET/name).read_text())
        c.require(template['execution_authorized'] is False and template['qualification_authorized'] is False,
                  'Prepared templates must remain unadmitted')
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

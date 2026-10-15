"""Exact native TaskCallScope/prehead-capture tensor contract, no native model."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

HERE = Path(__file__).resolve().parent
def load(path,package=False):
    name = '_NCNC_capture_contract_'+hashlib.sha256(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name,path,**({'submodule_search_locations':[str(path.parent)]} if package else {}))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


class CaptureContract(unittest.TestCase):
    def test_actual_frozen_scope_ignores_recursive_preheads(self):
        library = load(HERE.parent/'link/__init__.py',True)
        native_scope = load(HERE.parents[1]/'reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/boundaries.py')
        scope = native_scope.TaskCallScope()
        leaf = torch.nn.Linear(64,1,bias=False)
        capture = library.TaskPreheadCapture(leaf,scope.task_output_active)
        recursive = scope.wrap(lambda:leaf(torch.full((2,64),-100.)))
        def original_task(value):
            recursive()
            return leaf(value)
        top = scope.wrap(original_task)
        endpoints = [(0,1),(1,2),(2,3)]
        try:
            for member in range(4):
                value = torch.full((3,64),float(member))
                with capture.task(member,endpoints,[1]*3):
                    top(value)
            aligned = capture.aligned(torch)
            self.assertEqual(tuple(aligned['preheads'].shape),(4,3,64))
            for member in range(4):
                self.assertTrue(torch.equal(aligned['preheads'][member],torch.full((3,64),float(member))))
            self.assertEqual(scope.dispatch_counts,{'task':4,'recursive':4})
            self.assertEqual(scope.depth,0)
            self.assertEqual(len(set(aligned['route_query_signatures'])),1)
            self.assertEqual(sum(len(row) for row in capture.rows.values()),4)
        finally:
            capture.close()


if __name__ == '__main__':
    torch.set_num_threads(2); unittest.main()

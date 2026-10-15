"""Small operation/route/state interlock, independent of Torch or a task."""
from contextlib import contextmanager


class Ownership:
    def __init__(self):
        self.operation_role = self.inference_role = self.state_role = self.member = None

    def idle(self):
        if any(value is not None for value in
               (self.operation_role, self.inference_role, self.state_role, self.member)):
            raise RuntimeError('Normalization owner is busy')

    @contextmanager
    def operation(self, role, *, inference=False):
        allowed = ('VALID', 'TEST') if inference else ('TRAIN', 'VALID', 'TRAIN_DIAGNOSTIC')
        if role not in allowed:
            raise ValueError('Explicit supported task role required')
        self.idle()
        key = 'inference_role' if inference else 'operation_role'
        setattr(self, key, role)
        try:
            yield
        finally:
            setattr(self, key, None)

    @contextmanager
    def route(self, member, training):
        expected_training = self.operation_role == 'TRAIN'
        if type(member) is not int or not 0 <= member < 4 or type(training) is not bool \
                or self.member is not None or self.state_role is not None \
                or (self.operation_role is None and self.inference_role is None) \
                or training != expected_training:
            raise RuntimeError('Owned nonnested native route and matching mode required')
        self.member = member
        try:
            yield
        finally:
            self.member = None

    @contextmanager
    def state(self, role):
        if role not in ('export', 'load'):
            raise ValueError('Explicit checkpoint operation required')
        self.idle()
        self.state_role = role
        try:
            yield
        finally:
            self.state_role = None

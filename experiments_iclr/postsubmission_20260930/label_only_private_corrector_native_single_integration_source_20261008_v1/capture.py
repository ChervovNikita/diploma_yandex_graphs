"""Read-only full active-head captures during the existing native forward."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NativeSingleCapture:
    def __init__(self, native, nodes, feature_width, classes):
        require(native.task == 'wikics' and native.arm == 'single'
                and native.model.members == 1 and len(native.model.models) == 1
                and len(native.optimizers) == 1 and not native.model.contrastive,
                'Fresh original native own-CE single required')
        self.native = native
        self.body = native.model.models[0].body
        self.nodes, self.features, self.classes = nodes, feature_width, classes
        self.mode = None; self.expected = None; self.calls = 0
        self.pending = None; self.latest = None
        self.counts = dict(TRAIN_head_captures=0, VALID_head_captures=0, SERVE_head_captures=0,
                           capture_intervals_started=0, capture_intervals_completed=0)
        self.handles = []
        for name in ('pred_local', 'pred_global'):
            head = getattr(self.body, name)
            self.handles.append(head.register_forward_pre_hook(self._before(name)))
            self.handles.append(head.register_forward_hook(self._after(name)))

    def _active_name(self):
        return 'pred_global' if self.body._global else 'pred_local'

    def _before(self, name):
        def read_input(module, arguments):
            require(self.mode is not None and name == self._active_name() and self.pending is None,
                    'Only the active native head in an explicit capture interval')
            require(len(arguments) == 1 and arguments[0].shape == (self.nodes, self.features),
                    'Full native active-head H required before ID indexing')
            self.pending = (name, arguments[0].detach())
            return None  # Never replace inputs, modes, gradients or RNG state.
        return read_input

    def _after(self, name):
        def read_output(module, arguments, output):
            require(self.pending is not None and self.pending[0] == name
                    and output.shape == (self.nodes, self.classes), 'Matching full native head logits required')
            self.latest = (self.pending[1], output.detach())
            self.pending = None; self.calls += 1
            self.counts[self.mode + '_head_captures'] += 1
            return None  # The original backbone receives its original output.
        return read_output

    def begin(self, mode, expected_forwards):
        require(mode in ('TRAIN', 'VALID', 'SERVE') and self.mode is None and self.pending is None,
                'One serial native capture interval')
        self.mode, self.expected, self.calls = mode, expected_forwards, 0
        self.latest = None
        self.counts['capture_intervals_started'] += 1

    def finish(self):
        require(self.calls == self.expected and self.pending is None and self.latest is not None,
                'Exactly the existing native forwards, no missing or extra capture')
        result = self.latest
        self.mode = None; self.expected = None; self.latest = None
        self.counts['capture_intervals_completed'] += 1
        return result  # For TRAIN this is explicitly the second existing view B.

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []; self.pending = None; self.latest = None; self.mode = None

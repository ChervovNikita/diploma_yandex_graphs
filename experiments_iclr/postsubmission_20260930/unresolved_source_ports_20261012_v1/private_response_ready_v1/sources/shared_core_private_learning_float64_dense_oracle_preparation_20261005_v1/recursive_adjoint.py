"""Exact constant-adjacency recursive adjoint; no imports or execution of torch.

The factory receives the reviewed torch module and saved native spmm_add at
qualification runtime. Its forward always calls that saved native operator.
No derivative for sparse structure or values is admitted.
"""


def make_constant_adjacency_spmm(torch, saved_native_spmm_add):
    class ConstantAdjacencySpMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, adjacency, dense):
            value = adjacency.storage.value()
            if value is not None and value.requires_grad:
                raise ValueError("Recursive adjoint accepts constant sparse values only")
            if dense.ndim != 2 or adjacency.sparse_sizes()[1] != dense.shape[0]:
                raise ValueError("Require the native two-dimensional dense SpMM geometry")
            ctx.transpose = adjacency.t()
            return saved_native_spmm_add(adjacency, dense)

        @staticmethod
        def backward(ctx, grad_output):
            # A recorded application makes the dense-input adjoint recursively
            # differentiable. K is constant; no K/value/structure gradients.
            return None, ConstantAdjacencySpMM.apply(ctx.transpose, grad_output)

    def spmm_add(adjacency, dense):
        return ConstantAdjacencySpMM.apply(adjacency, dense)

    return spmm_add

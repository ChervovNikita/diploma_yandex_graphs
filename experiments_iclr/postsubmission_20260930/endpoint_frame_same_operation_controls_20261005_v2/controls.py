"""Source-only same-operation controls for the pinned endpoint reflection.

No numerical imports at module load, runner, data loader or new objective.
Pass caller-loaded exact parent/graph/adapter modules to make_control_classes.
"""
from hashlib import sha256
from pathlib import Path


PREDECESSOR_ADAPTER = (
    Path(__file__).resolve().parent.parent
    / "native_endpoint_private_factor_mechanism_assessment_20261005_v1"
    / "adapter.py"
)
PREDECESSOR_ADAPTER_SHA256 = "e275f83f8cd65c89d53e0d5580369c7dd4e0e59673e88bf567337d276fc8fc22"


def make_control_classes(prototype_module, graph_ops_module, endpoint_adapter_module):
    """Return three Twin classes; verify the sealed adapter and its parent pins.

    Load the pinned graph_ops.py as `graph_ops` before its prototype.py. Load
    predecessor adapter.py under a caller-chosen module name. All paths/bytes
    are verified. Construction is deferred until the returned classes are used.
    """
    if Path(endpoint_adapter_module.__file__).resolve() != PREDECESSOR_ADAPTER.resolve():
        raise RuntimeError("Unexpected predecessor adapter module path")
    if sha256(PREDECESSOR_ADAPTER.read_bytes()).hexdigest() != PREDECESSOR_ADAPTER_SHA256:
        raise RuntimeError("Sealed predecessor adapter bytes differ")
    FrameDecoder, _ = endpoint_adapter_module.make_adapter_classes(prototype_module, graph_ops_module)
    torch, nn, F = prototype_module.torch, prototype_module.nn, prototype_module.F

    class NativeLinear(nn.Module):
        """Ordinary affine operator using existing native-shaped W/bias draws."""
        def __init__(self, factor_linear):
            super().__init__()
            self.weight = factor_linear.weight
            self.bias = factor_linear.bias
            self.in_features = self.weight.shape[1]
            self.out_features = self.weight.shape[0]

        def forward_member(self, x, member):
            if member != 0:
                raise ValueError("Native single has only member0")
            return F.linear(x, self.weight, self.bias)

    class NativeNorm(nn.Module):
        """Ordinary LayerNorm affine parameters, with no member parameter axis."""
        def __init__(self, private_norm):
            super().__init__()
            self.width = private_norm.width
            self.weight = nn.Parameter(private_norm.weight[0].detach().clone())
            self.bias = nn.Parameter(private_norm.bias[0].detach().clone())

        def forward_member(self, x, member):
            if member != 0:
                raise ValueError("Native single has only member0")
            return F.layer_norm(x, (self.width,), self.weight, self.bias, 1e-5)

    def use_native_single_operators(decoder):
        # Replace wrappers after the one parent construction; consume no RNG.
        # Retain W/bias values, native module order and unused ptlin honestly.
        for name in ("xlin", "xcnlin", "xijlin", "lin", "ptlin"):
            branch = getattr(decoder, name)
            for index, operation in tuple(branch.ops.items()):
                if isinstance(operation, prototype_module.FactorLinear):
                    branch.ops[index] = NativeLinear(operation)
                elif isinstance(operation, prototype_module.PrivateNorm):
                    branch.ops[index] = NativeNorm(operation)

    def require_single_recipe(recipe):
        if recipe.member_count != 1:
            raise ValueError("Native single control requires member_count=1")

    def initialize_twin(twin, recipe, decoder_class, **decoder_arguments):
        # Parent constructor order with one encoder and one decoder; no replacement
        # of an already constructed random decoder, and no second graph trajectory.
        nn.Module.__init__(twin)
        twin.recipe = recipe
        twin.recipe.validate()
        twin.encoder = prototype_module.SharedEncoder(recipe)
        twin.decoder = decoder_class(recipe, **decoder_arguments)

    class SharedFrameF4Decoder(FrameDecoder):
        def __init__(self, recipe):
            if recipe.member_count != 4:
                raise ValueError("Shared-frame F4 requires four members")
            super().__init__(recipe)
            reference = self.xijlin.ops["0"].weight
            self.v = nn.Parameter(torch.eye(1, recipe.hidden, dtype=reference.dtype, device=reference.device))

        def endpoint_product(self, h, queries, member):
            # One learned common frame; same strict H and exact outer site.
            return super().endpoint_product(h, queries, 0)

    class SharedFrameF4Twin(prototype_module.CompletionTwin):
        def __init__(self, recipe=None):
            initialize_twin(self, recipe or prototype_module.Recipe(), SharedFrameF4Decoder)

    class IndependentNativeSingleFrameDecoder(FrameDecoder):
        def __init__(self, recipe, *, axis_index):
            require_single_recipe(recipe)
            if isinstance(axis_index, bool) or not isinstance(axis_index, int) or not 0 <= axis_index < recipe.hidden:
                raise ValueError("axis_index must be an explicit integer in [0,hidden)")
            super().__init__(recipe)
            use_native_single_operators(self)
            reference = self.xijlin.ops["0"].weight
            axis = reference.new_zeros((1, recipe.hidden))
            axis[0, axis_index] = 1
            self.v = nn.Parameter(axis)
            self.axis_index = axis_index

    class IndependentNativeSingleFrameTwin(prototype_module.CompletionTwin):
        def __init__(self, recipe=None, *, axis_index):
            initialize_twin(self, recipe or prototype_module.Recipe(member_count=1),
                            IndependentNativeSingleFrameDecoder, axis_index=axis_index)

    class FourFrameOuterLinear(nn.Module):
        """One wider target affine map, native first block for depth-zero input.

        Inherited recursion supplies d features and uses W0. The one outer target
        supplies4d features and uses all blocks. There is one shared W0/bias,
        no separately learned recursive map and no extra nonlinear/dropout call.
        """
        def __init__(self, native_linear):
            super().__init__()
            self.native_inputs = native_linear.in_features
            self.in_features = 4 * self.native_inputs
            self.out_features = native_linear.out_features
            original = native_linear.weight.detach()
            # One adopted initializer: extra blocks sum to zero in exact algebra.
            common = original / 4
            self.weight = nn.Parameter(torch.cat((original, common, common, -2 * common), dim=1))
            self.bias = native_linear.bias

        def forward_member(self, x, member):
            if member != 0:
                raise ValueError("Native single has only member0")
            if x.shape[-1] == self.native_inputs:
                return F.linear(x, self.weight[:, :self.native_inputs], self.bias)
            if x.shape[-1] == self.in_features:
                return F.linear(x, self.weight, self.bias)
            raise ValueError("Expected native depth-zero d or outer four-frame4d input")

    class SameFourFramesNativeSingleDecoder(FrameDecoder):
        def __init__(self, recipe):
            require_single_recipe(recipe)
            if recipe.hidden < 4:
                raise ValueError("Four explicit axes require hidden>=4")
            super().__init__(recipe)
            use_native_single_operators(self)
            reference = self.xijlin.ops["0"].weight
            self.v = nn.Parameter(torch.eye(4, recipe.hidden, dtype=reference.dtype, device=reference.device))
            self.xijlin.ops["0"] = FourFrameOuterLinear(self.xijlin.ops["0"])

        def endpoint_product(self, h, queries, member):
            if member != 0:
                raise ValueError("Four feature frames feed one native target trajectory")
            # Exact same reflection primitive, axes/order and symmetric products.
            products = [FrameDecoder.endpoint_product(self, h, queries, frame) for frame in range(4)]
            return torch.cat(products, dim=-1)

    class SameFourFramesNativeSingleTwin(prototype_module.CompletionTwin):
        def __init__(self, recipe=None):
            initialize_twin(self, recipe or prototype_module.Recipe(member_count=1),
                            SameFourFramesNativeSingleDecoder)

    return {
        "shared_frame_f4": SharedFrameF4Twin,
        "same_four_frames_native_single": SameFourFramesNativeSingleTwin,
        "independent_native_single_one_frame": IndependentNativeSingleFrameTwin,
    }

"""Logical tensor/array byte identity; data-free signed-zero qualification."""
import argparse
import struct
import sys
sys.dont_write_bytecode = True
import common as c


def tensor_equal(a, b):
    import torch
    return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and
            a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def array_equal(a, b):
    import numpy as np
    return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
            a.dtype == b.dtype and a.shape == b.shape and
            np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def float_equal(a, b):
    return type(a) is float and type(b) is float and struct.pack('!d', a) == struct.pack('!d', b)


def qualify(*, torch_module=None, numpy_module=None):
    checks = {'Python_float_signed_zero_distinguished': not float_equal(0.0, -0.0),
              'Torch_checks_executed': torch_module is not None,
              'NumPy_checks_executed': numpy_module is not None}
    if torch_module is not None:
        t = torch_module
        positive = t.tensor([0.0], dtype=t.float32)
        negative = t.tensor([-0.0], dtype=t.float32)
        scalar_positive = t.tensor(0.0, dtype=t.float32)
        scalar_negative = t.tensor(-0.0, dtype=t.float32)
        noncontiguous = t.arange(6, dtype=t.float32).reshape(2, 3).transpose(0, 1)
        checks.update(Torch_numeric_equal_accepts_signed_zero=t.equal(positive, negative),
                      Torch_bytes_distinguish_signed_zero=not tensor_equal(positive, negative),
                      Torch_scalar_bytes_distinguish_signed_zero=not tensor_equal(scalar_positive, scalar_negative),
                      Torch_dtype_mismatch_rejected=not tensor_equal(positive, positive.to(t.float64)),
                      Torch_shape_mismatch_rejected=not tensor_equal(positive, positive.reshape(1, 1)),
                      Torch_noncontiguous_logical_bytes_match=tensor_equal(noncontiguous, noncontiguous.contiguous()),
                      Torch_unchanged_bytes_match=tensor_equal(positive, positive.clone()),
                      Torch_version=t.__version__)
    if numpy_module is not None:
        np = numpy_module
        positive = np.asarray([0.0], dtype=np.float32)
        negative = np.asarray([-0.0], dtype=np.float32)
        noncontiguous = np.arange(6, dtype=np.float32).reshape(2, 3).T
        checks.update(NumPy_numeric_equal_accepts_signed_zero=np.array_equal(positive, negative),
                      NumPy_bytes_distinguish_signed_zero=not array_equal(positive, negative),
                      NumPy_dtype_mismatch_rejected=not array_equal(positive, positive.astype(np.float64)),
                      NumPy_shape_mismatch_rejected=not array_equal(positive, positive.reshape(1, 1)),
                      NumPy_noncontiguous_logical_bytes_match=array_equal(noncontiguous, np.ascontiguousarray(noncontiguous)),
                      NumPy_unchanged_bytes_match=array_equal(positive, positive.copy()),
                      NumPy_version=np.__version__)
    c.require(all(v for k, v in checks.items() if k not in
                  ('Torch_checks_executed', 'NumPy_checks_executed', 'Torch_version', 'NumPy_version')),
              'Data-free byte identity qualification failed')
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt', required=True)
    p.add_argument('--torch', action='store_true', help='Also run real Torch CPU fixture; never a model/data/GPU call')
    args = p.parse_args()
    source = c.verify_sources()
    output = c.confined(args.receipt)
    c.require(not output.is_relative_to(c.PACKET) and not output.is_relative_to(c.V4),
              'Byte qualification receipt must be external to sealed sources')
    import numpy as np
    if args.torch:
        import torch
    else:
        torch = None
    c.write(output, {'schema': 'amazon_native_byte_identity_data_free_qualification_v1',
                     'UTC': c.utc(), 'packet_manifest': source,
                     'checks': qualify(torch_module=torch, numpy_module=np),
                     'models_datasets_labels_checkpoints_GPU_or_remote_accessed': False})


if __name__ == '__main__':
    main()

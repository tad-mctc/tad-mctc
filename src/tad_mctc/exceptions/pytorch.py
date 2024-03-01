"""
Exceptions: PyTorch
===================

Exceptions related to PyTorch.
"""

__all__ = ["DeviceError", "DtypeError"]


class DeviceError(RuntimeError):
    """
    Error for wrong device of tensor.
    """


class DtypeError(ValueError):
    """
    Error for wrong data type of tensor.
    """

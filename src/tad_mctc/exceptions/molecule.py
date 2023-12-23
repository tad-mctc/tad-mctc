"""
Exceptions: Structure
=====================

Exceptions related to the molecular structure.
"""
__all__ = ["MoleculeError"]


class MoleculeError(RuntimeError):
    """
    Faulty structure detected.
    """

    pass

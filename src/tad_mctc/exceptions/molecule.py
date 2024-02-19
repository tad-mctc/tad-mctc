"""
Exceptions: Structure
=====================

Exceptions and warnings related to the molecular structure.
"""

__all__ = ["MoleculeError", "MoleculeWarning"]


class MoleculeError(RuntimeError):
    """
    Faulty structure detected.
    """


class MoleculeWarning(RuntimeWarning):
    """
    Suspicious/problematic structure detected.
    """

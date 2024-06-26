# SPDX-Identifier: CC0-1.0
from tad_mctc.data.molecules import mols
from tad_mctc.molecule.container import Mol

mol = mols["vancoh2"]
numbers = mol["numbers"]
positions = mol["positions"]

mol = Mol(numbers, positions)
print(mol.sum_formula())

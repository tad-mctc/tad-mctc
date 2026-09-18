# SPDX-Identifier: CC0-1.0
"""
Sum formula of a bespoke test-fixture `Structure` looked up by name from
`tad_mctc.data.structures`.
"""

from tad_mctc.data.structures import structures
from tad_mctc.properties.general import sum_formula

# `structures["vancoh2"]` is a `Structure` instance: species and positions are
# read off it by attribute, not by dict subscript.
structure = structures["vancoh2"]

print(sum_formula(structure.numbers))

Torch Autodiff Utility
======================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/tad-mctc/tad-mctc
    :target: https://github.com/tad-mctc/tad-mctc/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-mctc
    :target: https://pypi.org/project/tad-mctc/
    :alt: PyPI

.. image:: https://img.shields.io/conda/vn/conda-forge/tad-mctc.svg
    :target: https://anaconda.org/conda-forge/tad-mctc
    :alt: Conda Version

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: http://www.apache.org/licenses/LICENSE-2.0
    :alt: Apache-2.0

.. image:: https://github.com/tad-mctc/tad-mctc/actions/workflows/ubuntu.yaml/badge.svg
    :target: https://github.com/tad-mctc/tad-mctc/actions/workflows/ubuntu.yaml
    :alt: Test Status Ubuntu

.. image:: https://github.com/tad-mctc/tad-mctc/actions/workflows/windows.yaml/badge.svg
    :target: https://github.com/tad-mctc/tad-mctc/actions/workflows/windows.yaml
    :alt: Test Status Windows

.. image:: https://github.com/tad-mctc/tad-mctc/actions/workflows/macos-arm.yaml/badge.svg
    :target: https://github.com/tad-mctc/tad-mctc/actions/workflows/macos-arm.yaml
    :alt: Test Status macOS (ARM)

.. image:: https://readthedocs.org/projects/tad-mctc/badge/?version=latest
    :target: https://tad-mctc.readthedocs.io
    :alt: Documentation Status

.. image:: https://codecov.io/gh/tad-mctc/tad-mctc/branch/main/graph/badge.svg?token=OGJJnZ6t4G
    :target: https://codecov.io/gh/tad-mctc/tad-mctc
    :alt: Coverage

.. image:: https://results.pre-commit.ci/badge/github/tad-mctc/tad-mctc/main.svg
    :target: https://results.pre-commit.ci/latest/github/tad-mctc/tad-mctc/main
    :alt: pre-commit.ci status


This library is a collection of utility functions that are used in PyTorch (re-)implementations of projects from the `Grimme group <https://github.com/grimme-lab>`__.
In particular, the *tad-mctc* library provides:

- autograd functions (Jacobian, Hessian)

- atomic data (radii, EN, example molecules, ...)

- batch utility (packing, masks, ...)

- conversion functions (numpy, atomic symbols/numbers, ...)

- coordination numbers (DFT-D3, DFT-D4, EEQ)

- io (reading/writing coordinate files)

- molecular properties (bond lengths/orders/angles, moment of inertia, ...)

- safeops (autograd-safe implementations of common functions)

- typing (base class for tensor-like behavior of arbitrary classes)

- units

The name is inspired by the Fortran pendant "modular computation tool chain library" (`mctc-lib <https://github.com/grimme-lab/mctc-lib/>`__).


If you use this software, please cite the following publication

- \M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *J. Chem. Phys.*, **2024**, *161*, 062501. DOI: `10.1063/5.0216715 <https://doi.org/10.1063/5.0216715>`__


Examples
--------

The following example shows how to calculate the coordination number used in the EEQ model for a single structure.

.. code:: python

    import torch
    import tad_mctc as mctc

    numbers = mctc.convert.symbol_to_number(symbols="C C C C N C S H H H H H".split())

    # coordinates in Bohr
    positions = torch.tensor(
        [
            [-2.56745685564671, -0.02509985979910, 0.00000000000000],
            [-1.39177582455797, +2.27696188880014, 0.00000000000000],
            [+1.27784995624894, +2.45107479759386, 0.00000000000000],
            [+2.62801937615793, +0.25927727028120, 0.00000000000000],
            [+1.41097033661123, -1.99890996077412, 0.00000000000000],
            [-1.17186102298849, -2.34220576284180, 0.00000000000000],
            [-2.39505990368378, -5.22635838332362, 0.00000000000000],
            [+2.41961980455457, -3.62158019253045, 0.00000000000000],
            [-2.51744374846065, +3.98181713686746, 0.00000000000000],
            [+2.24269048384775, +4.24389473203647, 0.00000000000000],
            [+4.66488984573956, +0.17907568006409, 0.00000000000000],
            [-4.60044244782237, -0.17794734637413, 0.00000000000000],
        ]
    )

    # calculate EEQ coordination number
    cn = mctc.ncoord.cn_eeq(numbers, positions)
    torch.set_printoptions(precision=10)
    print(cn)
    # tensor([3.0519218445, 3.0177774429, 3.0132560730, 3.0197706223,
    #         3.0779352188, 3.0095663071, 1.0991339684, 0.9968624115,
    #         0.9943327904, 0.9947233200, 0.9945874214, 0.9945726395])

The next example shows the calculation of the coordination number used in DFT-D4 for a batch of structures.

.. code:: python

    import torch
    import tad_mctc as mctc

    # S22 system 4: formamide dimer
    numbers = mctc.batch.pack((
        mctc.convert.symbol_to_number("C C N N H H H H H H O O".split()),
        mctc.convert.symbol_to_number("C O N H H H".split()),
    ))

    # coordinates in Bohr
    positions = mctc.batch.pack((
        torch.tensor([
            [-3.81469488143921, +0.09993441402912, 0.00000000000000],
            [+3.81469488143921, -0.09993441402912, 0.00000000000000],
            [-2.66030049324036, -2.15898251533508, 0.00000000000000],
            [+2.66030049324036, +2.15898251533508, 0.00000000000000],
            [-0.73178529739380, -2.28237795829773, 0.00000000000000],
            [-5.89039325714111, -0.02589114569128, 0.00000000000000],
            [-3.71254944801331, -3.73605775833130, 0.00000000000000],
            [+3.71254944801331, +3.73605775833130, 0.00000000000000],
            [+0.73178529739380, +2.28237795829773, 0.00000000000000],
            [+5.89039325714111, +0.02589114569128, 0.00000000000000],
            [-2.74426102638245, +2.16115570068359, 0.00000000000000],
            [+2.74426102638245, -2.16115570068359, 0.00000000000000],
        ]),
        torch.tensor([
            [-0.55569743203406, +1.09030425468557, 0.00000000000000],
            [+0.51473634678469, +3.15152550263611, 0.00000000000000],
            [+0.59869690244446, -1.16861263789477, 0.00000000000000],
            [-0.45355203669134, -2.74568780438064, 0.00000000000000],
            [+2.52721209544999, -1.29200800956867, 0.00000000000000],
            [-2.63139587595376, +0.96447869452240, 0.00000000000000],
        ]),
    ))

    # calculate coordination number
    cn = mctc.ncoord.cn_d4(numbers, positions)
    torch.set_printoptions(precision=10)
    print(cn)
    # tensor([[2.6886456013, 2.6886456013, 2.6314170361, 2.6314167976,
    #          0.8594539165, 0.9231414795, 0.8605306745, 0.8605306745,
    #          0.8594539165, 0.9231414795, 0.8568341732, 0.8568341732],
    #         [2.6886456013, 0.8568335176, 2.6314167976, 0.8605306745,
    #          0.8594532013, 0.9231414795, 0.0000000000, 0.0000000000,
    #          0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000]])


.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   modules/index

Torch autodiff Utility
======================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/tad-mctc/tad-mctc
    :target: https://github.com/tad-mctc/tad-mctc/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-mctc
    :target: https://pypi.org/project/tad-mctc/
    :alt: PyPI

.. image:: https://img.shields.io/badge/License-LGPL_v3-blue.svg
    :target: https://www.gnu.org/licenses/lgpl-3.0
    :alt: LGPL-3.0

.. image:: https://github.com/tad-mctc/tad-mctc/workflows/CI/badge.svg
    :target: https://github.com/tad-mctc/tad-mctc/actions
    :alt: CI

.. image:: https://readthedocs.org/projects/tad-mctc/badge/?version=latest
    :target: https://tad-mctc.readthedocs.io
    :alt: Documentation Status

.. image:: https://codecov.io/gh/tad-mctc/tad-mctc/branch/main/graph/badge.svg?token=OGJJnZ6t4G
    :target: https://codecov.io/gh/tad-mctc/tad-mctc
    :alt: Coverage

.. image:: https://results.pre-commit.ci/badge/github/tad-mctc/tad-mctc/main.svg
    :target: https://results.pre-commit.ci/latest/github/tad-mctc/tad-mctc/main
    :alt: pre-commit.ci status


Collection of utility functions for PyTorch implementations of projects from the `Grimme group <https://github.com/grimme-lab>`__.

- autograd functions (Jacobian, Hessian)

- batch utility (packing, masks, ...)

- atomic data (radii, EN, example molecules, ...)

- io (reading coordinate files)

- coordination numbers

- safeops (autograd-safe implementations of common functions)

- units

The name is inspired by the "modular computation tool chain library" `mctc-lib <https://github.com/grimme-lab/mctc-lib/>`__.


Installation
------------

pip
~~~

*tad-mctc* can easily be installed with ``pip``.

.. code::

    pip install tad-mctc


From source
~~~~~~~~~~~

This project is hosted on GitHub at `tad-mctc/tad-mctc <https://github.com/tad-mctc/tad-mctc>`__.
Obtain the source by cloning the repository with

.. code::

    git clone https://github.com/tad-mctc/tad-mctc
    cd tad-mctc

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code::

    pip install .

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

Development
-----------

For development, additionally install the following tools in your environment.

.. code::

    mamba install black covdefaults coverage mypy pre-commit pylint tox

With pip, add the option ``-e`` for installing in development mode, and add ``[dev]`` for the development dependencies

.. code::

    pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

    pre-commit install

For testing all Python environments, simply run `tox`.

.. code::

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code::

    tox -- test

Examples
--------

The following example shows how to calculate the DFT-D4 dispersion energy for a single structure.

.. code:: python

    import torch
    import tad_mctc as mctc

    numbers = mctc.data.pse.symbol_to_number(symbols="C C C C N C S H H H H H".split())

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

The next example shows the calculation of dispersion energies for a batch of structures.

.. code:: python

    import torch
    import tad_mctc as mctc

    # S22 system 4: formamide dimer
    numbers = mctc.batch.pack((
        mctc.data.pse.symbol_to_number("C C N N H H H H H H O O".split()),
        mctc.data.pse.symbol_to_number("C O N H H H".split()),
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

Contributing
------------

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the `contributing guidelines <CONTRIBUTING.md>`__.

License
-------

This project is free software: you can redistribute it and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This project is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the Lesser GNU General Public License for more details.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Lesser GNU General Public license, shall be licensed as above, without any additional terms or conditions.

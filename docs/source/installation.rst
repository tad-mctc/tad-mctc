Installation
------------

pip
~~~

.. image:: https://img.shields.io/pypi/v/tad-mctc
    :target: https://pypi.org/project/tad-mctc/
    :alt: PyPI Version

.. image:: https://img.shields.io/pypi/dm/tad-mctc?color=orange
    :target: https://pypi.org/project/tad-mctc/
    :alt: PyPI Downloads

*tad-mctc* can easily be installed with ``pip``.

.. code::

    pip install tad-mctc

conda
~~~~~

.. image:: https://img.shields.io/conda/vn/conda-forge/tad-mctc.svg
    :target: https://anaconda.org/conda-forge/tad-mctc
    :alt: Conda Version

.. image:: https://img.shields.io/conda/dn/conda-forge/tad-mctc?style=flat&color=orange
    :target: https://anaconda.org/conda-forge/tad-mctc
    :alt: Conda Downloads

*tad-mctc* is also available from ``conda``.

.. code::

    conda install tad-mctc

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
- `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`__
- `psutil <https://psutil.readthedocs.io/en/latest/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)
- `qcelemental <https://molssi.github.io/QCElemental/>`__
- `scipy <https://scipy.org/>`__ (tests only)
- `torch <https://pytorch.org/>`__

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

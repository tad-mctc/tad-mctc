# Torch Autodiff Utility

<table>
  <tr>
    <td>Compatibility:</td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg" alt="Python Versions"/>
      <img src="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg" alt="PyTorch Versions"/>
    </td>
  </tr>
  <tr>
    <td>Availability:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-mctc/releases/latest">
        <img src="https://img.shields.io/github/v/release/tad-mctc/tad-mctc?color=orange" alt="Release"/>
      </a>
      <a href="https://pypi.org/project/tad-mctc/">
        <img src="https://img.shields.io/pypi/v/tad-mctc?color=orange" alt="PyPI"/>
      </a>
      <a href="https://anaconda.org/conda-forge/tad-mctc">
        <img src="https://img.shields.io/conda/vn/conda-forge/tad-mctc.svg" alt="Conda Version"/>
      </a>
      <a href="http://www.apache.org/licenses/LICENSE-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="Apache-2.0"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Status:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-mctc/actions/workflows/ubuntu.yaml">
        <img src="https://github.com/tad-mctc/tad-mctc/actions/workflows/ubuntu.yaml/badge.svg" alt="Test Status Ubuntu"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-mctc/actions/workflows/macos-arm.yaml">
        <img src="https://github.com/tad-mctc/tad-mctc/actions/workflows/macos-arm.yaml/badge.svg" alt="Test Status macOS (ARM)"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-mctc/actions/workflows/windows.yaml">
        <img src="https://github.com/tad-mctc/tad-mctc/actions/workflows/windows.yaml/badge.svg" alt="Test Status Windows"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-mctc/actions/workflows/release.yaml">
        <img src="https://github.com/tad-mctc/tad-mctc/actions/workflows/release.yaml/badge.svg" alt="Build Status"/>
      </a>
      <a href="https://tad-mctc.readthedocs.io">
        <img src="https://readthedocs.org/projects/tad-mctc/badge/?version=latest" alt="Documentation Status"/>
      </a>
      <a href="https://results.pre-commit.ci/latest/github/tad-mctc/tad-mctc/main">
        <img src="https://results.pre-commit.ci/badge/github/tad-mctc/tad-mctc/main.svg" alt="pre-commit.ci Status"/>
      </a>
      <a href="https://codecov.io/gh/tad-mctc/tad-mctc">
        <img src="https://codecov.io/gh/tad-mctc/tad-mctc/branch/main/graph/badge.svg?token=OGJJnZ6t4G" alt="Coverage"/>
      </a>
    </td>
  </tr>
</table>

<br>

This library is a collection of utility functions that are used in PyTorch (re-)implementations of projects from the [Grimme group](https://github.com/grimme-lab).
In particular, the _tad-mctc_ library provides:

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

The name is inspired by the Fortran pendant "modular computation tool chain library" ([mctc-lib](https://github.com/grimme-lab/mctc-lib/)).


## Citation

If you use this software, please cite the following publication

- M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *J. Chem. Phys.*, **2024**, *161*, 062501. DOI: [10.1063/5.0216715](https://doi.org/10.1063/5.0216715)


## Installation

### pip <a href="https://pypi.org/project/tad-mctc/"><img src="https://img.shields.io/pypi/v/tad-mctc" alt="PyPI Version"></a> <a href="https://pypi.org/project/tad-mctc/"><img src="https://img.shields.io/pypi/dm/tad-mctc?color=orange" alt="PyPI Downloads"></a>

_tad-mctc_ can easily be installed with `pip`.

```sh
pip install tad-mctc
```

### conda <a href="https://anaconda.org/conda-forge/tad-mctc"><img src="https://img.shields.io/conda/vn/conda-forge/tad-mctc.svg" alt="Conda Version"></a> <a href="https://anaconda.org/conda-forge/tad-mctc"><img src="https://img.shields.io/conda/dn/conda-forge/tad-mctc?style=flat&color=orange" alt="Conda Downloads"></a>

_tad-mctc_ is also available from `conda`.

```sh
conda install tad-mctc
```

### From source

This project is hosted on GitHub at [tad-mctc/tad-mctc](https://github.com/tad-mctc/tad-mctc/).
Obtain the source by cloning the repository with

```sh
git clone https://github.com/tad-mctc/tad-mctc
cd tad-mctc
```

We recommend using a [conda](https://conda.io/) environment to install the package.
You can setup the environment manager using a [mambaforge](https://github.com/conda-forge/miniforge) installer.
Install the required dependencies from the conda-forge channel.

```sh
mamba env create -n torch -f environment.yaml
mamba activate torch
```

Install this project with `pip` in the environment

```sh
pip install .
```

The following dependencies are required

- [numpy](https://numpy.org/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)
- [psutil](https://psutil.readthedocs.io/en/latest/)
- [pytest](https://docs.pytest.org/) (tests only)
- [qcelemental](https://molssi.github.io/QCElemental/)
- [scipy](https://scipy.org/) (tests only)
- [torch](https://pytorch.org/)


## Compatibility

| PyTorch \ Python | 3.8                | 3.9                | 3.10               | 3.11               | 3.12               |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1.11.0           | :white_check_mark: | :white_check_mark: | :x:                | :x:                | :x:                |
| 1.12.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:                |
| 1.13.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.0.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.1.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.2.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.3.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.4.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.5.1            | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.6.0            | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.7.1            | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

Note that only the latest bug fix version is listed, but all preceding bug fix minor versions are supported.
For example, although only version 2.2.2 is listed, version 2.2.0 and 2.2.1 are also supported.

On macOS and Windows, PyTorch<2.0.0 does only support Python<3.11.


## Development

For development, additionally install the following tools in your environment.

```sh
mamba install black covdefaults mypy pre-commit pylint pytest pytest-cov pytest-xdist tox
pip install pytest-random-order
```

With pip, add the option `-e` for installing in development mode, and add `[dev]` for the development dependencies

```sh
pip install -e .[dev]
```

The pre-commit hooks are initialized by running the following command in the root of the repository.

```sh
pre-commit install
```

For testing all Python environments, simply run `tox`.

```sh
tox
```

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional _posargs_.

```sh
tox -- test
```

## Examples

The following example shows how to calculate the coordination number used in the EEQ model for a single structure.

```python
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
```

The next example shows the calculation of the coordination number used in DFT-D4 for a batch of structures.

```python
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
```

## Contributing

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

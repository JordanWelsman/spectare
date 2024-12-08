# spectare

_A PyTorch visualisation and interpretability framework._

## Overview

`Spectare` is an open-source Python library which enables machine learning developers to create directed graph visualisations of their fully connected models. Spectare supports both `PyTorch` and `TensorFlow`, allowing almost all machine learning developers to optimise and interpret their models.

## Installation

> NOTE: I am currently working on staging the library for PIP. The command will not work yet as a result.

You can install `Spectare` via pip:

```bash
pip install spectare
```

Or locally by downloading the repository and installing via pip:

```bash
pip install -r requirements.txt
python setup.py bdist_wheel
pip install -e .
```

## Usage

Once installed, you can import and use `Spectare` in a Python environment:

```python
import spectare as sp

# PyTorch or TensorFlow model creation code

# Create a model graph
sp.draw_network(
    from_model=model
)
```

## License

`Spectare` is licensed under the MIT License. See the LICENSE.md file for more details.

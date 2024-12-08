"""
# Spectare

The spectare module contains a collection of functions and classes
for visualising neural networks created using the PyTorch and
TensorFlow libraries. The module is designed to be used in
conjuntion with the Jupyter notebook environment, and is
intended to provide a simple and intuitive way to visualise the
structure of neural networks.
"""

# Dunder attributes
__version__ = "v2.0.0"
__author__ = "Jordan Welsman"

# Module Imports
from .visual import *

# External Visibility
__all__ = visual.__all__

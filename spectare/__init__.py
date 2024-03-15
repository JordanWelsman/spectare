"""
# Spectare

The spectare module contains a collection of functions and classes
for visualising neural networks created using the _PyTorch_ and
_TensorFlow_ libraries. The module is designed to be used in
conjuntion with the _Jupyter_ notebook environment, and is
intended to provide a simple and intuitive way to visualise the
structure of neural networks.
"""

# Dunder attributes
__version__ = "v0.1.0" # update setup.py
__author__ = "Jordan Welsman"

# Module Imports
from .visual import *

# External Visibility
__all__ = visual.__all__

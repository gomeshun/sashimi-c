"""Compatibility import for :mod:`sashimi_c._itamae_variance`."""

import sys as _sys
from importlib import import_module as _import_module

_module = _import_module('sashimi_c._itamae_variance')
_sys.modules[__name__] = _module

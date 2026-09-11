"""Compatibility import for :mod:`sashimi_c.picard_tidal_stripping`."""

import sys as _sys
from importlib import import_module as _import_module

_module = _import_module('sashimi_c.picard_tidal_stripping')
_sys.modules[__name__] = _module

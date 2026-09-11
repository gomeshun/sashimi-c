"""Compatibility import for :mod:`sashimi_c.prompt_cusps`."""

import sys as _sys
from importlib import import_module as _import_module

_module = _import_module('sashimi_c.prompt_cusps')
_sys.modules[__name__] = _module

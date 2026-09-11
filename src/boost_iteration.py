"""Compatibility import for :mod:`sashimi_c.boost_iteration`."""

import sys as _sys
from importlib import import_module as _import_module

_module = _import_module('sashimi_c.boost_iteration')

if __name__ == "__main__":
    _module.main()
else:
    _sys.modules[__name__] = _module

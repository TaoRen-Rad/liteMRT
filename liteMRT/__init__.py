# In yourpackage/__init__.py

from numba import jit
import inspect
import sys
import types


def _auto_jit_decorate(module):
    """Apply JIT decorator to all functions in a module"""
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            setattr(module, name, jit(obj))


# Apply to current package
_auto_jit_decorate(sys.modules[__name__])

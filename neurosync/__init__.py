"""NeuroSync package root created during refactor.
Exports high-level objects for external users while we migrate code out of utils/*.
"""

from importlib import import_module

# Temporary shims to keep old import paths working

def __getattr__(name):
    if name == "core":
        return import_module("neurosync.core")
    raise AttributeError(name) 
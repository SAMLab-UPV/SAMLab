"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

import importlib
import pkgutil
import inspect
from analysis_plugins import base

def discover_plugins(package):
    """Discover all subclasses of AnalysisBase in the given package."""
    plugins = []
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{modname}")
        # Recursively discover nested modules
        if ispkg:
            plugins.extend(discover_plugins(module))
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, base.AnalysisBase) and obj is not base.AnalysisBase:
                plugins.append(obj)
    return plugins

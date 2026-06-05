"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU Affero General Public License v3.0 or later.
See the LICENSE file for details.
"""

class AnalysisBase:
    """Base class for all analysis plugins."""
    name = "Unnamed Analysis"
    description = "No description provided."
    default_params = {}  # GUI uses this to build input fields

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def analyze(self, data):
        raise NotImplementedError("Subclasses must implement this method")

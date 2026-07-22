"""
SAMLab - Submarine Acoustic Monitoring Laboratory
Copyright (C) 2026 Universitat Politècnica de València

This file is part of SAMLab.

Licensed under the GNU General Public License v3.0 (GPLv3) or later.
See the LICENSE file for details.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resources"

def resource(filename):
    path = RESOURCE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Resource not found: {path}")

    return str(path)
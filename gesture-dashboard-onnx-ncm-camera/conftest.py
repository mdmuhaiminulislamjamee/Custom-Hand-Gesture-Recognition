"""Pytest configuration and environment bridge.

Automatically discovers the local project .venv and ensures site-packages and
native DLLs are on sys.path and PATH, allowing pytest to run cleanly whether
invoked from inside .venv or from global/IDE Python.
"""
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project root is in sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# If running outside .venv, bridge to .venv site-packages
venv_dir = PROJECT_ROOT / ".venv"
if venv_dir.is_dir():
    site_packages = (venv_dir / "Lib" / "site-packages").resolve()
    if site_packages.is_dir() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    if os.name == "nt":
        scripts_dir = venv_dir / "Scripts"
        if scripts_dir.is_dir():
            os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
